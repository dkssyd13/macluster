"""grove worker entry point for the 2-MacBook (`--backend grove`) run.

`grove start` / `grove join` can only launch a **Python file that defines
`main()`** (they `importlib`-load the script; the joiner receives only the
script *source*, no argv). So this is the single launchable form of a grove
run, and the `TrainConfig` travels via `MACLUSTER_*` environment variables that
must be exported **identically on every Mac** (a config-consensus check below
aborts loudly on mismatch, since a divergent config silently breaks the
deterministic data sharding).

Launch (see docs/PHASE7_TODO.md §3):

    # mac-A (launcher)
    export MACLUSTER_TASK=cifar10 MACLUSTER_ALGORITHM=diloco MACLUSTER_ROUNDS=25 MACLUSTER_H=20
    uv run grove start scripts/grove_entry.py -n 2 --name macluster --logs

    # mac-B (joiner) -- export the SAME MACLUSTER_* env, then:
    uv run grove join macluster --logs

Single-machine smoke (collectives are W=1 no-ops):

    MACLUSTER_SYNTHETIC=1 MACLUSTER_ALGORITHM=dense MACLUSTER_ROUNDS=2 \
        uv run python scripts/grove_entry.py
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict

import grove
import mlx.core as mx

from macluster.train import TrainConfig, run_training


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "", "false", "no")


def _cfg_from_env() -> TrainConfig:
    """Build a grove TrainConfig from MACLUSTER_* env vars (TrainConfig defaults
    fill the gaps). ``world_size`` is bound to the real cluster size so the task
    builds exactly one data shard per rank."""
    d = TrainConfig()
    g = os.environ.get

    def _opt_int(name, fallback):
        return int(os.environ[name]) if name in os.environ else fallback

    def _opt_float(name, fallback):
        return float(os.environ[name]) if name in os.environ else fallback

    def _opt_str(name, fallback):
        return os.environ[name] if name in os.environ else fallback

    return TrainConfig(
        task=g("MACLUSTER_TASK", d.task),
        model=g("MACLUSTER_MODEL", d.model),
        algorithm=g("MACLUSTER_ALGORITHM", d.algorithm),
        world_size=int(grove.world_size),  # one shard/stage per rank; ignore env
        backend="grove",
        parallelism=g("MACLUSTER_PARALLELISM", d.parallelism),
        cut=_opt_str("MACLUSTER_CUT", d.cut),
        n_micro=int(g("MACLUSTER_N_MICRO", str(d.n_micro))),
        stage_mem_gb=_opt_str("MACLUSTER_STAGE_MEM_GB", d.stage_mem_gb),
        rounds=int(g("MACLUSTER_ROUNDS", str(d.rounds))),
        max_steps=_opt_int("MACLUSTER_MAX_STEPS", d.max_steps),
        batch_size=int(g("MACLUSTER_BATCH_SIZE", str(d.batch_size))),
        seq_len=int(g("MACLUSTER_SEQ_LEN", str(d.seq_len))),
        H=int(g("MACLUSTER_H", str(d.H))),
        k_frac=float(g("MACLUSTER_K_FRAC", str(d.k_frac))),
        outer_lr=float(g("MACLUSTER_OUTER_LR", str(d.outer_lr))),
        inner_opt=g("MACLUSTER_INNER_OPT", d.inner_opt),
        inner_lr=float(g("MACLUSTER_INNER_LR", str(d.inner_lr))),
        link=g("MACLUSTER_LINK", d.link),
        eval_every=int(g("MACLUSTER_EVAL_EVERY", str(d.eval_every))),
        seed=int(g("MACLUSTER_SEED", str(d.seed))),
        synthetic=_env_flag("MACLUSTER_SYNTHETIC", d.synthetic),
        data_dir=g("MACLUSTER_DATA_DIR", d.data_dir),
        target_metric=_opt_float("MACLUSTER_TARGET_METRIC", d.target_metric),
    )


def _assert_config_consensus(cfg: TrainConfig) -> None:
    """Abort if ranks resolved different configs (env mismatch across Macs)."""
    if grove.world_size <= 1:
        return
    blob = json.dumps(asdict(cfg), sort_keys=True, default=str)
    local = int(hashlib.sha256(blob.encode()).hexdigest(), 16) % (2 ** 23)  # exact in fp32
    total = float(grove.all_sum(mx.array([float(local)]))[0])
    if abs(total - local * grove.world_size) > 0.5:
        raise SystemExit(
            f"[grove_entry] rank {grove.rank}: config disagrees across ranks. "
            f"Every Mac must export identical MACLUSTER_* env. (local hash={local})"
        )


def main() -> None:
    # Three launch paths:
    #  1) via `grove start`/`join`: the world is already up (grove._comm set) and
    #     the script was distributed -- nothing to init here.
    #  2) via scripts/p2.sh (GROVE_CLUSTER + GROVE_N>1 set): SELF-rendezvous with
    #     grove.init(transport=...). This bypasses the grove cli, whose tcp path
    #     never delivers the script to the joiner ("script not received"); each Mac
    #     runs its own local copy instead.
    #  3) standalone `python scripts/grove_entry.py`: world_size=1 smoke.
    if grove._comm is None:
        cluster = os.environ.get("GROVE_CLUSTER")
        ws = int(os.environ.get("GROVE_N", "1"))
        if cluster and ws > 1:
            grove.init(
                cluster=cluster,
                world_size=ws,
                transport=os.environ.get("GROVE_TRANSPORT", "tcp"),
            )
        elif grove.world_size <= 1:
            grove.init()

    cfg = _cfg_from_env()
    mx.random.seed(cfg.seed)  # rank-identical model init (no broadcast needed)
    _assert_config_consensus(cfg)

    runs_root = os.environ.get("MACLUSTER_RUNS_ROOT", "runs")
    run_dir = os.path.join(runs_root, f"{cfg.slug()}-rank{grove.rank}")
    print(f"[grove_entry] rank {grove.rank}/{grove.world_size} -> {run_dir}")
    summary = run_training(cfg, run_dir)
    print(f"[grove_entry] rank {grove.rank} done: {json.dumps(summary, default=str)}")

    # Clean teardown for the self-init (scripts/p2.sh / run2.sh) path: the
    # coordinator (rank 0) hosts the TCPStore + CoordinatorServer *inside its own
    # process*, so if it exits first the workers' in-flight store ops reset
    # ("Control connection lost" -> barrier ConnectionResetError). The grove
    # start/join cli manages this lifecycle; here we do it ourselves with an
    # asymmetric done-handshake so the store owner always exits LAST.
    if grove.world_size > 1 and grove._comm is not None:
        try:
            store = grove._comm._group._store
            if grove.rank == 0:
                store.wait(
                    [f"macluster_done/{r}" for r in range(1, int(grove.world_size))],
                    timeout=300.0,
                )
            else:
                store.set(f"macluster_done/{grove.rank}", b"1")
        except Exception as e:  # never let teardown bookkeeping fail a finished run
            print(f"[grove_entry] rank {grove.rank} teardown handshake skipped: {e}")


if __name__ == "__main__":
    main()
