# macluster

**Emulating low-communication distributed training (DiLoCo / SparseLoCo / an
adaptive policy) across MacBook replicas — on one machine.**

`macluster` is the experimental engine for an ACN survey/course project on
*distributed learning in heterogeneous environments*. It runs `W` model
replicas in a single process (the **SimCluster** backend), trains them on
disjoint data shards with synchronous averaging, and attaches an analytic
**link cost model** to every synchronization round. This lets us measure how
communication-reduction algorithms and an adaptive synchronization policy behave
across Apple-Silicon-style links (AWDL-only vs. Wi-Fi-upgraded) *before*
committing to real 2-MacBook runs.

The core is proven on real data: a 2-replica DiLoCo run on real CIFAR-10 reaches
**~61% validation accuracy in 25 rounds** (see [Current results](#current-results)).

---

## The three-axis framing (mapped to code)

The project is organized around three axes of heterogeneity, each backed by a
concrete part of the codebase.

| Axis | Question | Where it lives |
| --- | --- | --- |
| **Axis 1 — Network** | AWDL-only vs. Wi-Fi-upgraded links | `emulation/link.py` — `LinkProfile` goodput/latency/jitter, `LinkSchedule` (constant or mid-run switch) |
| **Axis 2 — Algorithm** | How cheaply can we synchronize? | `algorithms/{dense,diloco,sparseloco}.py` |
| **Axis 3 — Adaptive policy** *(novel contribution)* | React to the link and the comm/compute ratio at runtime | `algorithms/adaptive.py` |

### Axis 1 — link profiles (`emulation/link.py`)

A `LinkProfile` turns "bytes a worker must move this round" into "seconds that
transfer would take" via `latency + bytes·8 / bandwidth` (plus optional jitter).
The canonical profiles are representative figures to be **recalibrated by the
2-MacBook runs**, not measurements:

| Profile name | Goodput | Latency | Jitter | Models |
| --- | --- | --- | --- | --- |
| `wifi` (Wi-Fi upgraded) | 300 Mbps | 3 ms | 1 ms | the link after a Wi-Fi upgrade |
| `awdl` (AWDL-only) | 60 Mbps | 10 ms | 6 ms | Apple peer-to-peer discovery link |
| `wifi_degraded` | 40 Mbps | 20 ms | 10 ms | a poor / congested Wi-Fi link |
| `datacenter` | 10 Gbps | 0.1 ms | — | a reference "fast" link |

`LinkSchedule.switch(...)` scripts a mid-run change (e.g. `wifi → awdl` at round
12) so the adaptive policy can be observed reacting to a link that degrades.

### Axis 2 — synchronization algorithms (`algorithms/`)

All three implement the same `Algorithm` contract (`algorithms/base.py`): own
the global synchronized state, run `H` local inner steps per round, then
`sync(...)` the replicas back to one agreed state and report the bytes a worker
transmitted (so the link emulator can charge time).

- **`dense`** (`dense.py`) — averages full parameters every step (`H=1`). This
  is synchronous SGD; it transmits the whole model every step and is the
  **communication-heavy reference**.
- **`diloco`** (`diloco.py`) — DiLoCo (Douillard et al., 2023). `H` local steps,
  then outer Nesterov-SGD on the averaged pseudo-gradient `global − replica`.
  Communicates once per `H` steps, one full model-sized payload — `~H×` cheaper
  than dense.
- **`sparseloco`** (`sparseloco.py`) — SparseLoCo (Sarfi et al., 2025). DiLoCo
  plus **top-`k_frac` sparsification with error feedback** (`compress.py`). Only
  `(index, value)` pairs are sent (`8·k` bytes/tensor vs. `4·n` dense), cutting
  communication by roughly `1/(2·k_frac)`.

### Axis 3 — the adaptive policy (`algorithms/adaptive.py`) — *the contribution*

`AdaptiveSync` is a thin reactive controller layered on SparseLoCo. Each round
it adjusts two knobs from cheaply-measured signals:

- **`H` (how often to sync)** from the EMA of the `sync_s / compute_s` ratio:
  if communication dominates compute, raise `H` (sync less); if compute
  dominates, lower `H` (sync more, better convergence). Bounded to `[8, 256]`.
- **`k_frac` (how hard to compress)** from link health: on a degraded/AWDL-only
  link, halve `k_frac` (compress harder); on healthy Wi-Fi, relax it toward less
  compression. Bounded to `[0.005, 0.1]`.

It is intentionally a small control layer — not a full pipeline planner — which
matches the proposal's "small adaptive layer" framing.

---

## Architecture

```
                Task (data + loss + eval + model_fns)          task.py / data/
                          |
   +----------------------+-----------------------+
   |                                              |
 SimCluster (backends/sim.py)            Algorithm (algorithms/*)   <- axis 2/3
   W replicas, one process                owns global state, H, k_frac
   disjoint shards, real compute
   |                                              |
   +----------------+-----------------------------+
                    |
            run_training (train.py)  --- per round --->  LinkSchedule (axis 1)
            round loop + logging                          charges sync time
                    |
              RunLogger (metrics/logger.py)
              config.json / metrics.jsonl / summary.json
```

**SimCluster** runs `W` in-process replicas on disjoint shards with synchronous
averaging. It implements the distributed-training *semantics* without a second
machine. Each round:

1. every replica is loaded with the algorithm's current global params;
2. each replica runs `H` real inner optimizer steps on its own shard (real
   compute, timed with `time.perf_counter`);
3. the algorithm's `sync(...)` aggregates the replicas into a new global state
   and reports bytes/worker;
4. the link emulator charges communication time for those bytes.

**Round compute time is modeled as the max over replicas**, because on real
hardware the workers compute in parallel — the round is gated by the slowest
worker, not the sum. **`emulation/link.py` charges communication time** on top.

### Honest scope

Single-machine emulation gives us, faithfully:

- **convergence** — real CIFAR-10 / text training, real gradients on disjoint
  shards (averaging is a genuine operation, not a no-op);
- **communication volume** — exact bytes/round per algorithm and topology;
- **policy behavior** — how the adaptive controller moves `H` and `k_frac` as
  the link and comm/compute ratio change.

It does **not** give real wall-clock *speedup*: the emulator models parallel
compute and link time analytically rather than measuring two machines actually
running concurrently over a real radio. **Real wall-clock numbers require the
2-MacBook `GroveBackend`, which is deferred to Phase 7** (see
[`docs/PHASE7_TODO.md`](docs/PHASE7_TODO.md)). The link goodput/latency figures
above are representative and are what the 2-MacBook runs will recalibrate.

---

## Install

This is a [uv](https://docs.astral.sh/uv/) project. From the project root:

```bash
uv sync
```

Run **everything** through `uv run` so the project virtualenv is used. The only
runtime dependency for the emulation core is MLX + NumPy; `grove-ml` is listed
in `pyproject.toml` but is imported only by the (deferred) `GroveBackend`, so the
emulation works even if grove is unavailable.

---

## Usage

### Train one configuration

`macluster train` runs a single model × algorithm × link configuration and
writes a run directory under `runs/`.

```bash
# Real CIFAR-10, 2-replica DiLoCo, small ResNet, Wi-Fi link (the headline run)
uv run macluster train \
    --task cifar10 --model resnet20 --algorithm diloco \
    --world-size 2 --rounds 25 --H 20 --link wifi \
    --run-dir runs/real_diloco

# SparseLoCo with aggressive top-k compression (k_frac = 2%)
uv run macluster train --task cifar10 --algorithm sparseloco --k-frac 0.02 --rounds 25

# The adaptive policy across a link that degrades mid-run (Wi-Fi -> AWDL at round 12)
uv run macluster train --task cifar10 --algorithm adaptive \
    --link wifi --link-switch-to awdl --link-switch-at 12 --rounds 25

# Larger model (the compute-load axis: resnet56 is ~3x resnet20)
uv run macluster train --task cifar10 --model resnet56 --algorithm diloco --rounds 25

# Synthetic data (no download) for a fast smoke test
uv run macluster train --algorithm sparseloco --synthetic --rounds 4 --batch-size 64
```

**Text tasks** route through `data/text.py` (`shakespeare` char-level, or
`wikitext`) and select a transformer with `--model`:

```bash
# Character-level Shakespeare with a small GPT (charGPT)
uv run macluster train --task shakespeare --model chargpt \
    --algorithm diloco --seq-len 128 --rounds 25

# WikiText with a GPT-2-style model
uv run macluster train --task wikitext --model gpt2 \
    --algorithm sparseloco --seq-len 256 --rounds 25
```

> Note: the text task builder (`data/text.py`) and its models (`chargpt`,
> `gpt2`) are wired into `train.build_task` and exposed by these flags. The CIFAR
> path is the validated, headline result; the text path uses the identical
> `Task` contract.

**Available models / algorithms / links**

- `--model`: `resnet20` (272K params), `resnet56` (856K params) for `cifar10`;
  `chargpt`, `gpt2` for text tasks.
- `--algorithm`: `dense`, `diloco`, `sparseloco`, `adaptive`.
- `--link`: `wifi`, `awdl`, `wifi_degraded`, `datacenter`
  (plus `--link-switch-to` / `--link-switch-at` for a scripted switch).
- key knobs: `--H` (local steps/round), `--k-frac` (top-k fraction),
  `--outer-lr`, `--inner-opt {adam,sgd}`, `--inner-lr`, `--world-size`,
  `--rounds`, `--max-steps` (equal total-local-step budget across algorithms;
  overrides `--rounds`), `--batch-size`, `--seq-len`, `--eval-every`, `--seed`.

Run `uv run macluster train --help` for the full list.

### Experiment sweeps

A sweep harness runs a grid of configurations from a YAML file under `configs/`
and collects their run directories for comparison:

```bash
# via the module ...
uv run python -m macluster.experiment --config configs/axis2_algorithms.yaml
# ... or the CLI subcommand
uv run macluster sweep --config configs/axis1_link.yaml
uv run macluster sweep --config configs/axis3_adaptive.yaml
```

Shipped configs: `smoke` (synthetic sanity), `axis2_algorithms` (dense vs
DiLoCo vs SparseLoCo), `axis1_link` (link sweep), `axis3_adaptive` (adaptive
vs fixed across a mid-run link switch). The real-data configs use an **equal
total-local-step budget** (`max_steps`) so dense (`H=1`) and the low-comm
methods (`H=20`) are compared at the same compute. Sweeps are **resumable**
(a run whose `summary.json` exists is skipped unless `--overwrite`) and write
`runs/<name>/index.json` aggregating every run's summary.

### Plotting

`scripts/plot.py` reads the `metrics.jsonl` / `summary.json` files in `runs/` and
renders the figures (saved under `figures/`):

```bash
# overlay specific runs + a sweep index (fig1 metric-vs-time, fig2 comm-vs-round,
# fig3 sweep bars, fig4 adaptive H/k_frac schedule)
uv run macluster plot --runs runs/axis2_algorithms/*/ \
    --out figures/axis2 --sweep runs/axis2_algorithms/index.json
# equivalently: uv run python scripts/plot.py --runs ... --out ... --sweep ...
```

Typical figures: primary metric vs. emulated `sim_time` (the time-to-accuracy
story), **cumulative communication bytes vs. round** (the low-communication
story), per-axis sweep bars (final metric + total comm), and the adaptive
policy's `H` / `k_frac` trajectories.

---

## Metrics

Each run directory (`runs/<slug>/`) contains `config.json` (resolved
`TrainConfig`), `metrics.jsonl` (one JSON record per sync round), and
`summary.json` (final aggregates). The important fields:

**Per-round (`metrics.jsonl`)**

| Field | Meaning |
| --- | --- |
| `train_loss` | mean inner-step loss over the round |
| `compute_s` | **real** measured compute time = max over replicas (parallel model) |
| `sync_s_sim` | **emulated** communication time for this round's payload on the active link |
| `round_s_sim` | `compute_s + sync_s_sim` |
| `sim_time_s` | cumulative emulated wall-clock |
| `comm_bytes` / `comm_bytes_cum` | bytes a worker moved this round / cumulatively |
| `link` | active link profile this round |
| `throughput_sps` | samples/sec from real compute |
| `accuracy` / `val_loss` | on eval rounds (`--eval-every`) |
| `H`, `k_frac`, `ratio_ema` | current knobs (adaptive policy logs all three) |

**Summary (`summary.json`)**

| Field | Meaning |
| --- | --- |
| `final_accuracy` (or `final_<metric>`) | metric at the last round |
| `total_comm_MB` | total communication volume — the cost axis |
| `sim_time_s` | total emulated wall-clock |
| `time_to_target_s` | emulated seconds to first reach `target_metric` (default 0.45 acc) — the speed axis |

The headline comparison is **accuracy and `time_to_target_s` vs.
`total_comm_MB`**: low-communication methods should reach the target at a small
fraction of dense's communication budget.

---

## Current results

All numbers below are real outputs from `runs/` (CIFAR-10, `world_size=2`).

**Headline — real CIFAR-10, DiLoCo, ResNet-20, Wi-Fi link** (`runs/real_diloco`):

- **~61.0% validation accuracy** in 25 rounds (`H=20`, Adam inner, `outer_lr=0.7`).
- **Time-to-45%-accuracy ≈ 14.55 s** of emulated wall-clock.
- **~27.2 MB** total communication (one ResNet-20-sized pseudo-gradient,
  ~1.09 MB/round over 25 rounds).
- Per round: ~0.88 s real compute, ~0.033 s emulated Wi-Fi sync — i.e.
  communication is a small fraction of compute on the healthy Wi-Fi link, which
  is exactly the regime the adaptive policy probes by switching to AWDL.

**Communication contrast — SparseLoCo vs. DiLoCo** (synthetic smoke runs, same 4
rounds, `H=5`): SparseLoCo transmitted **0.349 MB** vs. DiLoCo's **4.36 MB** —
about **12× less communication** at `k_frac = 0.02`, consistent with the
`~1/(2·k_frac)` expectation. On the full real run this is the difference between
~27 MB (DiLoCo) and a few MB (SparseLoCo) for comparable accuracy.

**Fair algorithm comparison — equal 500-step budget, real CIFAR-10, Wi-Fi**
(`configs/axis2_algorithms.yaml`):

| Algorithm | Val acc | Total comm | Sim time | Time-to-45% |
| --- | --- | --- | --- | --- |
| dense (`H=1`) | 0.639 | 545 MB | 38.5 s | 11.7 s |
| diloco (`H=20`) | 0.607 | 27.3 MB | 22.4 s | 9.0 s |
| sparseloco (`k=2%`) | 0.562 | 2.2 MB | 22.1 s | 11.6 s |

At equal compute, dense buys a few points of accuracy with **20–250× more
communication**. On a fast Wi-Fi link the time penalty is modest — the
low-communication payoff shows up on slow links:

**Link robustness — DiLoCo vs SparseLoCo across links** (`configs/axis1_link.yaml`):

| Link | DiLoCo sim time | SparseLoCo sim time |
| --- | --- | --- |
| `wifi` | 23.6 s | 22.6 s |
| `awdl` | 25.7 s | 23.6 s |
| `wifi_degraded` | 28.6 s | 23.8 s |

DiLoCo's full-payload sync slows as the link degrades; SparseLoCo's 2.2 MB
payload makes it nearly link-insensitive (~22–24 s throughout).

**Adaptive policy — link degrades Wi-Fi → AWDL mid-run** (`configs/axis3_adaptive.yaml`):

| Policy | Val acc | Time-to-45% | Total comm |
| --- | --- | --- | --- |
| sparseloco (fixed) | 0.616 | 12.4 s | 2.6 MB |
| **adaptive** | **0.662** | **6.8 s** | 15.1 MB |

The adaptive policy relaxes compression while Wi-Fi is healthy (spending
communication when it is cheap) and compresses harder once the link drops to
AWDL — reaching the target accuracy **~1.8× faster** and ending higher.

> These establish **convergence + communication-volume + policy-behavior**
> results on a single machine. **Real wall-clock speedup** (two MacBooks actually
> training concurrently over AWDL/Wi-Fi) is the job of the `GroveBackend` in
> [Phase 7](docs/PHASE7_TODO.md).

---

## Repository layout

```
src/macluster/
  task.py              Task contract (data + loss + eval + model_fns)
  train.py             TrainConfig + run_training driver
  cli.py               `macluster train` entry point
  backends/sim.py      SimCluster: W in-process replicas (single-machine emulation)
  algorithms/
    base.py            Algorithm contract + parameter-tree helpers
    dense.py           full-parameter averaging (comm-heavy reference)
    diloco.py          DiLoCo (axis 2)
    sparseloco.py      SparseLoCo: top-k + error feedback (axis 2)
    compress.py        top-k sparsification kernel
    adaptive.py        AdaptiveSync policy (axis 3, the contribution)
  emulation/link.py    LinkProfile / LinkSchedule (axis 1)
  data/cifar.py        CIFAR-10 task builder (+ synthetic fallback)
  data/text.py         text (shakespeare / wikitext) task builder
  models/resnet.py     CIFAR ResNet-20 / ResNet-56
  metrics/logger.py    per-run JSONL + summary logging
configs/               sweep configs (YAML)
scripts/plot.py        figure generation
runs/                  per-run outputs
docs/PHASE7_TODO.md    deferred 2-MacBook GroveBackend plan
```

---

## License / attribution

Course project for CAU 26-1 Advanced Computer Networks. Algorithms follow
DiLoCo (Douillard et al., 2023) and SparseLoCo (Sarfi et al., 2025); the
adaptive synchronization policy is this project's contribution. `grove-ml` is
the centerpiece tool for the real-hardware backend planned in Phase 7.
