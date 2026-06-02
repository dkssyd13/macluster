# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`macluster` emulates low-communication distributed training (DiLoCo / SparseLoCo / an adaptive policy) across MacBook replicas **on a single machine**, for an ACN course project. MLX-based, managed with `uv`.

## Commands

Run everything through `uv run` (uses the project venv).

```bash
uv sync                                   # install deps (incl. pytest dev group)
uv run pytest                             # all tests
uv run pytest tests/test_adaptive.py -k observe   # a single test / pattern

uv run macluster train --task cifar10 --algorithm diloco --rounds 25   # one config
uv run macluster train --algorithm sparseloco --synthetic --rounds 4   # fast smoke (no download)
uv run macluster sweep --config configs/axis2_algorithms.yaml          # grid sweep (resumable; --overwrite to rerun)
uv run macluster plot --runs runs/axis2_algorithms/*/ --out figures/axis2 --sweep runs/axis2_algorithms/index.json

uv run macluster train --backend grove --world-size 1 --synthetic --rounds 2   # GroveBackend W=1 smoke
./scripts/grove_run.sh start|join configs/grove/<algo>.env   # real 2-Mac run, same config on both (see docs/PHASE7_TODO.md)
```

## Architecture

The default is **single-machine emulation** (`--backend sim`): `SimCluster` (`backends/sim.py`) runs `W` model replicas in one process on disjoint data shards. There is no real network — communication cost is modeled analytically. Real 2-MacBook wall-clock numbers come from `GroveCluster` (`--backend grove`, `backends/grove_backend.py`): one model per machine, aggregating over real `grove.all_sum` collectives, with `sync_s`/`comm_bytes` **measured** instead of charged. It is implemented and W=1-smoke-tested but awaits the actual two-Mac run (`scripts/grove_entry.py` + the runbook in `docs/PHASE7_TODO.md`). `grove-ml` is imported **only** by `grove_backend.py` (lazily via `build_cluster`), so the sim core works without it. `build_cluster(cfg, task)` dispatches on `cfg.backend`.

The round loop lives in `train.py:run_training`:
1. load the algorithm's global params into every replica;
2. each replica runs `H` real inner optimizer steps on its shard — `compute_s = max` over replicas (workers run in parallel on real hardware);
3. `algorithm.sync(replica_params)` averages the replicas into one new global state and returns bytes/worker;
4. `emulation/link.py` charges `sync_s` for those bytes on the active `LinkProfile`;
5. `algorithm.observe(compute_s, sync_s, link)` lets adaptive policies react.

On `--backend grove` the loop branches to measure real timings; the algorithm aggregates this rank's tree via `sync_collective` over `grove.all_sum` instead of `sync` over the replica list.

**Key decision:** algorithms (`algorithms/*.py`) implement DiLoCo/SparseLoCo as a synchronous-averaging view over the replica list — we own the outer loop rather than calling grove's high-level `grove.diloco()`. This is what lets the **adaptive policy (`adaptive.py`, the novel contribution)** retune `H` and `k_frac` every round. All algorithms satisfy the `Algorithm` contract in `algorithms/base.py` (`init_global` / `global_params` / `local_steps` / `sync` for the sim list-of-replicas path / `sync_collective` for the grove single-rank-over-collectives path / `observe`).

**Three axes** map to code: axis 1 (link) → `emulation/link.py`; axis 2 (algorithm) → `algorithms/{dense,diloco,sparseloco}.py`; axis 3 (adaptive) → `algorithms/adaptive.py`.

`Task` (`task.py`) decouples data/model/loss/eval from the algorithm: built in `data/{cifar,text}.py`, holds `model_fns`, `loss_fn`, `eval_fn`, and one infinite `(X,y)` iterator per worker. Tasks are dispatched in `train.py:build_task`.

`TrainConfig.max_steps`, when set, runs to an **equal total-local-step budget** (rounds derived from `H`) so dense (`H=1`) and low-comm methods (`H=20`) are compared at the same compute — the real-data sweep configs rely on this.

## Conventions

- Adding an **algorithm**: subclass `Algorithm`, register in `train.py:build_algorithm`. Adding a **task**: write `make_*_task(...) -> Task`, route it in `train.py:build_task`. Adding a **model**: add to the relevant task's `model_fns`.
- MLX models are `nn.Module`; gradients via `nn.value_and_grad(model, loss_fn)` called as `fn(model, X, y)`. Conv inputs are NHWC.
- Each run writes `runs/<slug>/{config.json, metrics.jsonl, summary.json}`; sweeps write `runs/<name>/index.json`. `plot.py` reads these.
