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
from .task import Task


@dataclass
class TrainConfig:
    task: str = "cifar10"
    model: str = "resnet20"
    algorithm: str = "diloco"            # dense | diloco | sparseloco | adaptive
    world_size: int = 2
    backend: str = "sim"                 # sim (single-machine emulation) | grove (real 2-Mac)
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
    task = build_task(cfg)
    cluster = build_cluster(cfg, task)
    algo = build_algorithm(cfg)
    link = build_link(cfg)
    rng = np.random.default_rng(cfg.seed)

    algo.init_global(cluster.initial_params())

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
    eval_step_interval = max(1, budget // 20) if budget is not None else None
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
            do_eval = is_last or steps_done >= next_eval_at
            if steps_done >= next_eval_at:
                next_eval_at += eval_step_interval
        else:
            is_last = rnd == cfg.rounds - 1
            do_eval = is_last or rnd % cfg.eval_every == 0

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
            "total_comm_bytes": int(total_bytes),
            "total_comm_MB": round(total_bytes / 1e6, 3),
            "sim_time_s": round(sim_time, 3),
            "time_to_target_s": None if tta is None else round(tta, 3),
            "target_metric": target,
        }
    )
    logger.close()
    return summary
