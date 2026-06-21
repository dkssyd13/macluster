"""Unified training driver: model x algorithm x link, on the SimCluster backend.

Wires a Task (data + loss + eval), a synchronization Algorithm (axis 2/3), and
a LinkSchedule (axis 1) into one round-based loop, logging per-round metrics
(train loss, emulated compute/sync time, communication bytes, val accuracy) and
a summary including time-to-accuracy.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from .algorithms.adaptive import AdaptiveSync
from .algorithms.base import Algorithm
from .algorithms.dense import Dense
from .algorithms.diloco import DiLoCo
from .algorithms.sparseloco import SparseLoCo
from .backends.sim import SimCluster
from .data.cifar import make_cifar_task
from .emulation.link import LinkSchedule, allreduce_bytes, get_profile
from .metrics.logger import RunLogger
from .models.gpt import GPT_CONFIGS, gpt_param_count, memory_aware_cut, stage_param_counts
from .task import Task


@dataclass
class TrainConfig:
    task: str = "cifar10"
    model: str = "resnet20"
    algorithm: str = "diloco"            # dense | diloco | sparseloco | adaptive
    world_size: int = 2
    backend: str = "sim"                 # sim (single-machine emulation) | grove (real 2-Mac)
    parallelism: str = "data"            # data (DiLoCo/etc.) | pipeline (model-parallel, GPT-only)
    cut: str | None = None               # pipeline: explicit block-cut indices, e.g. "22" or "8,16"
    n_micro: int = 4                     # pipeline: micro-batches per optimizer step (1F1B)
    stage_mem_gb: str | None = None      # pipeline: per-stage RAM budget for the auto cut, e.g. "48,24"
    rounds: int = 50
    max_steps: int | None = None         # if set, run to an equal TOTAL-local-step
    #                                      budget across algorithms (fair dense-vs-
    #                                      lowcomm compute): rounds derived from H.
    batch_size: int = 128
    seq_len: int = 128                   # transformer tasks only
    H: int = 20
    k_frac: float = 0.02
    outer_lr: float = 0.7
    inner_opt: str = "adam"              # adam | sgd
    inner_lr: float = 1e-3
    link: str = "wifi"                   # wifi | awdl | wifi_degraded | datacenter
    link_switch_to: str | None = None    # if set, switch link mid-run
    link_switch_at: int | None = None
    eval_every: int = 5
    seed: int = 0
    synthetic: bool = False
    data_dir: str = "data/cache"
    target_metric: float | None = None   # for time-to-accuracy (defaults per task)

    def slug(self) -> str:
        link = self.link if not self.link_switch_to else f"{self.link}2{self.link_switch_to}"
        if self.parallelism == "pipeline":
            base = f"{self.task}-{self.model}-pipeline-w{self.world_size}-m{self.n_micro}-{link}-s{self.seed}"
        else:
            base = f"{self.task}-{self.model}-{self.algorithm}-w{self.world_size}-H{self.H}-{link}-s{self.seed}"
        return base if self.backend == "sim" else f"{base}-{self.backend}"


# --------------------------------------------------------------------------- #
# factories
# --------------------------------------------------------------------------- #
def build_task(cfg: TrainConfig) -> Task:
    if cfg.task == "cifar10":
        return make_cifar_task(
            cfg.world_size,
            batch_size=cfg.batch_size,
            synthetic=cfg.synthetic,
            data_dir=cfg.data_dir,
            seed=cfg.seed,
        )
    if cfg.task in ("shakespeare", "wikitext"):
        from .data.text import make_text_task  # lazy: keeps cifar-only runs light

        return make_text_task(
            cfg.world_size,
            variant=cfg.task,
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            synthetic=cfg.synthetic,
            data_dir=cfg.data_dir,
            seed=cfg.seed,
        )
    raise KeyError(f"unknown task {cfg.task!r}")


def build_algorithm(cfg: TrainConfig) -> Algorithm:
    a = cfg.algorithm
    if a == "dense":
        return Dense(H=1)
    if a == "diloco":
        return DiLoCo(H=cfg.H, outer_lr=cfg.outer_lr)
    if a == "sparseloco":
        return SparseLoCo(H=cfg.H, outer_lr=cfg.outer_lr, k_frac=cfg.k_frac)
    if a == "adaptive":
        return AdaptiveSync(H=cfg.H, outer_lr=cfg.outer_lr, k_frac=cfg.k_frac)
    raise KeyError(f"unknown algorithm {cfg.algorithm!r}")


def build_link(cfg: TrainConfig) -> LinkSchedule:
    if cfg.link_switch_to and cfg.link_switch_at is not None:
        return LinkSchedule.switch(
            get_profile(cfg.link), get_profile(cfg.link_switch_to), cfg.link_switch_at
        )
    return LinkSchedule.constant(get_profile(cfg.link))


def build_inner_opt(cfg: TrainConfig):
    if cfg.inner_opt == "sgd":
        return lambda: optim.SGD(learning_rate=cfg.inner_lr, momentum=0.9)
    if cfg.inner_opt == "adam":
        return lambda: optim.Adam(learning_rate=cfg.inner_lr)
    raise KeyError(f"unknown inner_opt {cfg.inner_opt!r}")


def build_cluster(cfg: TrainConfig, task: Task):
    """SimCluster (single-machine emulation) or GroveCluster (real 2-Mac)."""
    if cfg.backend == "grove":
        from .backends.grove_backend import GroveCluster  # lazy: keep grove off the sim path

        return GroveCluster(task, cfg.model, build_inner_opt(cfg))
    if cfg.backend != "sim":
        raise KeyError(f"unknown backend {cfg.backend!r}")
    return SimCluster(task, cfg.model, build_inner_opt(cfg))


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_training(cfg: TrainConfig, run_dir: str) -> dict:
    if cfg.parallelism == "pipeline":
        return run_pipeline_training(cfg, run_dir)
    if cfg.parallelism != "data":
        raise KeyError(f"unknown parallelism {cfg.parallelism!r} (data | pipeline)")
    task = build_task(cfg)
    cluster = build_cluster(cfg, task)
    if cfg.backend == "grove":  # abort loudly if the Macs built different corpora
        from .backends.grove_backend import assert_data_consensus
        assert_data_consensus(task.meta)
    algo = build_algorithm(cfg)
    link = build_link(cfg)
    rng = np.random.default_rng(cfg.seed)

    algo.init_global(cluster.initial_params())
    mx.reset_peak_memory()  # measure peak unified memory during training (per replica/rank)

    target = cfg.target_metric
    if target is None:
        target = 0.45 if task.metric == "accuracy" else None
    goal_max = task.metric_goal == "max"

    logger = RunLogger(run_dir, asdict(cfg))
    sim_time = 0.0
    total_bytes = 0.0
    samples = 0
    tta = None  # time-to-accuracy (emulated seconds)

    # Stopping + eval cadence. With max_steps we run to an equal total-local-step
    # budget (fair across algorithms with different H) and evaluate on a step
    # grid so every run gets ~the same number of eval points; otherwise we run a
    # fixed number of rounds and evaluate every `eval_every` rounds.
    budget = cfg.max_steps
    eval_step_interval = None
    if cfg.eval_every > 0:
        eval_step_interval = cfg.eval_every if budget is not None else None
    next_eval_at = eval_step_interval

    rnd = 0
    steps_done = 0  # per-replica inner steps
    while (steps_done < budget) if budget is not None else (rnd < cfg.rounds):
        cluster.load_global(algo.global_params())
        H = algo.local_steps()
        if budget is not None:
            H = max(1, min(H, budget - steps_done))

        losses: list[float] = []
        if cfg.backend == "grove":
            # Real cluster: this rank runs ONE model; barriers fence the inner
            # loop so the slowest rank physically gates the round.
            cluster.barrier()
            t0 = time.perf_counter()
            for _ in range(H):
                losses.append(cluster.inner_step())
            cluster.barrier()
            compute_s = time.perf_counter() - t0
        else:
            # Single-machine emulation: time each replica; the round's compute is
            # the max (workers run in parallel on real hardware).
            per_rep_times = []
            for rep in cluster.replicas:
                t0 = time.perf_counter()
                for _ in range(H):
                    losses.append(rep.inner_step())
                per_rep_times.append(time.perf_counter() - t0)
            compute_s = max(per_rep_times)

        steps_done += H
        samples += H * cluster.world_size * cfg.batch_size

        if cfg.backend == "grove":
            # MEASURE the real collective wall-clock as sync_s; bytes come from
            # the algorithm's measured payload (no analytic link charge).
            t1 = time.perf_counter()
            stats = algo.sync_collective(cluster.params(), cluster)
            sync_s = time.perf_counter() - t1
            payload = stats.bytes_per_worker
            link_name = cfg.link  # operator-declared link condition (measured, not emulated)
            real_timing = {"compute_s_real": round(compute_s, 5), "sync_s_real": round(sync_s, 5)}
        else:
            stats = algo.sync(cluster.collect_params())
            # In budget mode the link schedule is keyed by step so a mid-run switch
            # lands at the same compute point regardless of each algorithm's H.
            profile = link.at(steps_done if budget is not None else rnd)
            payload = allreduce_bytes(stats.bytes_per_worker, cluster.world_size, stats.topology)
            sync_s = profile.transfer_time(payload, rng)
            link_name = profile.name
            real_timing = {}

        algo.observe(compute_s, sync_s, link_name)

        sim_time += compute_s + sync_s
        total_bytes += payload

        if budget is not None:
            is_last = steps_done >= budget
            do_eval = eval_step_interval is not None and (is_last or steps_done >= next_eval_at)
            if eval_step_interval is not None and steps_done >= next_eval_at:
                next_eval_at += eval_step_interval
        else:
            is_last = rnd == cfg.rounds - 1
            do_eval = cfg.eval_every > 0 and (is_last or rnd % cfg.eval_every == 0)

        rec = {
            "round": rnd,
            "step": steps_done,
            "train_loss": float(np.mean(losses)),
            "compute_s": round(compute_s, 5),
            "sync_s_sim": round(sync_s, 5),
            "round_s_sim": round(compute_s + sync_s, 5),
            "sim_time_s": round(sim_time, 5),
            "comm_bytes": int(payload),
            "comm_bytes_cum": int(total_bytes),
            "link": link_name,
            "samples": samples,
            "throughput_sps": round(H * cluster.world_size * cfg.batch_size / max(compute_s, 1e-6), 1),
            "peak_mem_mb": round(mx.get_peak_memory() / 1e6, 1),  # measured replica/rank footprint
            **real_timing,
            **algo.knobs(),
        }

        # On grove only rank 0 holds the post-sync global params for eval and
        # writes the eval keys; other ranks log compute/comm only.
        if do_eval and (cfg.backend != "grove" or cluster.rank == 0):
            model = cluster.eval_model(algo.global_params())
            metrics = task.eval_fn(model, task.eval_batches)
            rec.update(metrics)
            if target is not None and tta is None:
                val = metrics.get(task.metric)
                if val is not None and ((goal_max and val >= target) or (not goal_max and val <= target)):
                    tta = sim_time
                    rec["reached_target"] = True

        logger.log(rec)
        rnd += 1

    final = logger.records[-1]
    summary = logger.summary(
        {
            "config": asdict(cfg),
            "final_train_loss": final.get("train_loss"),
            f"final_{task.metric}": final.get(task.metric),
            "peak_mem_mb": round(mx.get_peak_memory() / 1e6, 1),
            "total_comm_bytes": int(total_bytes),
            "total_comm_MB": round(total_bytes / 1e6, 3),
            "sim_time_s": round(sim_time, 3),
            "time_to_target_s": None if tta is None else round(tta, 3),
            "target_metric": target,
        }
    )
    logger.close()
    return summary


# --------------------------------------------------------------------------- #
# pipeline (model) parallelism — Phase 8 (see docs/PHASE8_MODELPARALLEL.md)
# --------------------------------------------------------------------------- #
def build_pipeline_task(cfg: TrainConfig) -> Task:
    """A GPT (text) task with a SINGLE data shard: pipeline parallelism splits one
    model across stages, so the data is NOT sharded (that is the orthogonal
    data-parallel axis). Every rank builds this identically (same seed) so the
    first stage's tokens and the last stage's targets stay aligned per micro-batch."""
    if cfg.task not in ("shakespeare", "wikitext"):
        raise KeyError(
            f"pipeline parallelism is GPT-only; --task must be shakespeare|wikitext, got {cfg.task!r}"
        )
    from .data.text import make_text_task  # lazy: keeps cifar-only runs light

    return make_text_task(
        1,
        variant=cfg.task,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        synthetic=cfg.synthetic,
        data_dir=cfg.data_dir,
        seed=cfg.seed,
    )


def _parse_stage_mem(spec: str | None, world_size: int) -> list[float]:
    if spec:
        mem = [float(x) for x in str(spec).split(",") if x.strip() != ""]
        if len(mem) != world_size:
            raise ValueError(
                f"--stage-mem-gb has {len(mem)} entr(ies) but world_size is {world_size}"
            )
        return mem
    return [1.0] * world_size  # equal budgets -> balanced split


def resolve_cuts(cfg: TrainConfig, gpt_cfg) -> list[int]:
    """Block-cut indices for the split: explicit ``--cut`` if given, else a
    memory-aware partition sized by ``--stage-mem-gb`` (Asteroid-style)."""
    if cfg.cut:
        cuts = [int(c) for c in str(cfg.cut).split(",") if c.strip() != ""]
    else:
        cuts = memory_aware_cut(gpt_cfg, _parse_stage_mem(cfg.stage_mem_gb, cfg.world_size))
    if len(cuts) + 1 != cfg.world_size:
        raise ValueError(
            f"cut {cuts} yields {len(cuts) + 1} stage(s) but world_size is {cfg.world_size}; "
            f"need exactly one stage per rank"
        )
    return cuts


def build_pipeline_cluster(cfg: TrainConfig, gpt_cfg, cuts: list[int], task: Task):
    """SimPipelineCluster (all stages, one process) or PipelineCluster (grove,
    one stage per Mac). The seam loss is logits-level (``loss(logits, y)``),
    distinct from the task's ``loss(model, X, y)``."""
    from .backends.pipeline import SimPipelineCluster, seq_cross_entropy

    data_iter = task.train_shards[0]
    if cfg.backend == "grove":
        from .backends.pipeline_grove import PipelineCluster  # lazy: keep grove off the sim path

        return PipelineCluster(
            gpt_cfg, cuts, build_inner_opt(cfg), data_iter, seq_cross_entropy,
            batch_size=cfg.batch_size, seq_len=cfg.seq_len,
        )
    if cfg.backend != "sim":
        raise KeyError(f"unknown backend {cfg.backend!r}")
    return SimPipelineCluster(gpt_cfg, cuts, build_inner_opt(cfg), data_iter, seq_cross_entropy)


def run_pipeline_training(cfg: TrainConfig, run_dir: str) -> dict:
    """Model-parallel (pipeline) training driver — the sibling of
    :func:`run_training`'s data-parallel loop. A round = ONE optimizer step over
    ``n_micro`` micro-batches (1F1B). Logs the same field names as the DP loop so
    ``plot.py`` and the sweep harness work unchanged.

    sim   : compute time is measured per step; the seam transfer is charged
            analytically on the active LinkProfile (mirrors the DP sim path).
    grove : the whole step is timed around real ``send``/``recv`` (communication
            overlaps compute), and the seam bytes are measured, not charged.
    """
    task = build_pipeline_task(cfg)
    if cfg.model not in GPT_CONFIGS:
        raise KeyError(f"pipeline --model must be one of {sorted(GPT_CONFIGS)}, got {cfg.model!r}")
    vocab = int(task.meta["vocab_size"])
    gpt_cfg = GPT_CONFIGS[cfg.model](vocab)
    if cfg.seq_len > gpt_cfg.block_size:
        raise ValueError(f"--seq-len {cfg.seq_len} exceeds {cfg.model} block_size {gpt_cfg.block_size}")
    cuts = resolve_cuts(cfg, gpt_cfg)
    cluster = build_pipeline_cluster(cfg, gpt_cfg, cuts, task)
    if cfg.backend == "grove":  # abort loudly if the Macs built different corpora
        from .backends.grove_backend import assert_data_consensus
        assert_data_consensus(task.meta)
    link = build_link(cfg)
    rng = np.random.default_rng(cfg.seed)
    mx.reset_peak_memory()  # measure this rank's peak unified memory during training

    is_grove = cfg.backend == "grove"
    # On grove the metric-bearing rank is the LAST one (it owns the loss + the
    # logits for eval); on sim a single process holds everything.
    is_metric_rank = (not is_grove) or (cluster.rank == cluster.world_size - 1)

    n_stages = len(cuts) + 1
    part = stage_param_counts(gpt_cfg, cuts)
    knobs = {
        "parallelism": "pipeline",
        "n_micro": cfg.n_micro,
        "n_stages": n_stages,
        "cut": ",".join(map(str, cuts)),
    }
    logger = RunLogger(run_dir, asdict(cfg))

    sim_time = 0.0
    total_bytes = 0.0
    samples = 0

    budget = cfg.max_steps  # interpreted as a number of optimizer steps for pipeline
    eval_step_interval = None
    if cfg.eval_every > 0:
        eval_step_interval = cfg.eval_every if budget is not None else None
    next_eval_at = eval_step_interval

    rnd = 0
    steps_done = 0
    while (steps_done < budget) if budget is not None else (rnd < cfg.rounds):
        if is_grove:
            cluster.barrier()
            t0 = time.perf_counter()
            loss, comm_bytes = cluster.step(cfg.n_micro)
            cluster.barrier()
            compute_s = time.perf_counter() - t0
            sync_s = 0.0  # communication overlaps compute; folded into the measured step
            link_name = cfg.link
            real_timing = {
                "round_s_real": round(compute_s, 5),
                "comm_s_real": round(getattr(cluster, "_comm_s", 0.0), 5),  # seam transfer + bubble
            }
        else:
            t0 = time.perf_counter()
            loss, comm_bytes = cluster.step(cfg.n_micro)
            compute_s = time.perf_counter() - t0
            profile = link.at(steps_done if budget is not None else rnd)
            sync_s = profile.transfer_time(comm_bytes, rng)  # point-to-point seam (no all-reduce factor)
            link_name = profile.name
            real_timing = {}

        steps_done += 1
        samples += cfg.n_micro * cfg.batch_size
        sim_time += compute_s + sync_s
        total_bytes += comm_bytes

        if budget is not None:
            is_last = steps_done >= budget
            do_eval = eval_step_interval is not None and (is_last or steps_done >= next_eval_at)
            if eval_step_interval is not None and steps_done >= next_eval_at:
                next_eval_at += eval_step_interval
        else:
            is_last = rnd == cfg.rounds - 1
            do_eval = cfg.eval_every > 0 and (is_last or rnd % cfg.eval_every == 0)

        rec = {
            "round": rnd,
            "step": steps_done,
            "train_loss": None if loss is None else float(loss),
            "compute_s": round(compute_s, 5),
            "sync_s_sim": round(sync_s, 5),
            "round_s_sim": round(compute_s + sync_s, 5),
            "sim_time_s": round(sim_time, 5),
            "comm_bytes": int(comm_bytes),
            "comm_bytes_cum": int(total_bytes),
            "link": link_name,
            "samples": samples,
            "throughput_sps": round(cfg.n_micro * cfg.batch_size / max(compute_s, 1e-6), 1),
            "peak_mem_mb": round(mx.get_peak_memory() / 1e6, 1),  # measured per-stage footprint
            **real_timing,
            **knobs,
        }

        if do_eval:
            if is_grove:
                # Cap the eval seam transfer well under grove's 120s socket
                # timeout: 16 batches of a big model's hidden states over slow
                # Wi-Fi can exceed it (xl ~105MB, gpt3b ~168MB per eval). ~50MB
                # budget -- mp_mid keeps all 16 batches; only the big xl/3b models
                # trim (their convergence curve is not a headline result). Both
                # ranks compute the same cap, so the eval set stays in lockstep.
                bpb = cfg.batch_size * cfg.seq_len * gpt_cfg.n_embd * 4
                eval_set = task.eval_batches[: max(1, int(50_000_000 / max(bpb, 1)))]
                cluster.barrier()
                metrics = cluster.eval_loss(eval_set)  # None except the last rank
                cluster.barrier()
                if metrics:
                    rec.update(metrics)
            else:
                metrics = task.eval_fn(cluster.eval_model(), task.eval_batches)
                rec.update(metrics)

        logger.log(rec)
        rnd += 1

    final = logger.records[-1]
    summary = logger.summary(
        {
            "config": asdict(cfg),
            "parallelism": "pipeline",
            "n_stages": n_stages,
            "cut": cuts,
            "stage_param_counts": part,
            "model_param_count": gpt_param_count(gpt_cfg),
            "peak_mem_mb": round(mx.get_peak_memory() / 1e6, 1),
            "final_train_loss": final.get("train_loss"),
            f"final_{task.metric}": final.get(task.metric),
            "total_comm_bytes": int(total_bytes),
            "total_comm_MB": round(total_bytes / 1e6, 3),
            "sim_time_s": round(sim_time, 3),
            "target_metric": None,
        }
    )
    logger.close()
    return summary
