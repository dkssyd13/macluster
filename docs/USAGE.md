# macluster — Usage

MLX-based emulation of low-communication distributed training
(DiLoCo / SparseLoCo / an adaptive policy) across MacBook replicas. Run
everything through `uv run` (uses the project venv).

## Setup

```bash
uv sync                 # install deps (incl. pytest dev group)
uv run pytest           # all tests (60)
```

## Single machine — `--backend sim` (default)

`SimCluster` runs `W` replicas in one process; communication cost is modeled
analytically by `emulation/link.py`.

```bash
# one config
uv run macluster train --task cifar10 --algorithm diloco --rounds 25
# fast smoke (synthetic data, no download)
uv run macluster train --algorithm sparseloco --synthetic --rounds 4
# grid sweep (resumable; --overwrite to rerun)
uv run macluster sweep --config configs/axis2_algorithms.yaml
# plot a sweep
uv run macluster plot --runs runs/axis2_algorithms/*/ --out figures/axis2 \
    --sweep runs/axis2_algorithms/index.json
```

Key knobs: `--algorithm {dense,diloco,sparseloco,adaptive}`, `--world-size`,
`--H`, `--k-frac`, `--link {wifi,awdl,wifi_degraded,datacenter}`,
`--max-steps N` (equal total-local-step budget so dense `H=1` and low-comm
`H=20` compare at the same compute).

## Two MacBooks — `--backend grove` (Phase 7, real wall-clock)

`GroveCluster` runs one model per machine and aggregates over real
`grove.all_sum`; `sync_s` / `comm_bytes` are **measured**, not charged.

**Each Mac**: clone this repo + `uv sync`. Both Macs pass the **same** committed
config file (`scripts/grove_run.sh` sources it into `MACLUSTER_*` env, so there
is no env to retype and no mismatch).

```bash
# 0) network: open Finder -> AirDrop on BOTH Macs (AWDL), or put both on one Wi-Fi
./scripts/grove_run.sh check                            # both: is the peer visible?

# 1) launch (same config on both)
./scripts/grove_run.sh start configs/grove/diloco.env   # mac-A (launcher)
./scripts/grove_run.sh join  configs/grove/diloco.env   # mac-B (joiner)
```

Configs: `configs/grove/{dense,diloco,sparseloco,adaptive}.env` (real CIFAR,
`MAX_STEPS=500` equal-compute budget) and `smoke.env` (synthetic). Switch the
link condition with `MACLUSTER_LINK=awdl` inside the `.env`.

```bash
./scripts/grove_run.sh smoke    # single-machine W=1 smoke (no second Mac)
```

Each rank writes `runs/<slug>-rank<N>/`; eval (accuracy) runs on **rank 0 only**
(see the printed `[grove_entry] rank X/2` line). Full runbook, network details,
and design notes: [`docs/PHASE7_TODO.md`](PHASE7_TODO.md).

## Outputs

Each run → `runs/<slug>/{config.json, metrics.jsonl, summary.json}`; sweeps →
`runs/<name>/index.json`. `scripts/plot.py` reads these (field names are stable
across both backends).
