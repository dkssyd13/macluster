# Phase 7: `GroveBackend` + 2-MacBook runbook

**Status: IMPLEMENTED, pending the real 2-Mac run.** `GroveCluster`
(`backends/grove_backend.py`), the `--backend {sim,grove}` flag, per-algorithm
`Algorithm.sync_collective` over `grove.all_sum`, and the
`scripts/grove_entry.py` launcher are written and pass single-machine
(`world_size=1`) smoke + numerical-equivalence tests
(`tests/test_grove_backend.py`; grove's collectives are W=1 no-ops, so
`sync_collective` reduces exactly to `sync([local])`). What remains is the
actual two-MacBook run and the §4 emulator recalibration. The launch commands in
§3.2 below are the **real, working** invocation (the original
`grove start -- macluster train ...` form does **not** work — grove only launches
a Python file with `main()`, and the joiner receives no argv; config travels via
`MACLUSTER_*` env on both Macs).

> **Decisions taken at implementation (differ from / refine the plan below):**
> (1) **SparseLoCo/Adaptive collective = `all_sum` of the sparsified dense
> tensor `/W`** (numerically identical to gather-(idx,val); `comm_bytes` logs the
> compact `8*k` payload, but the *measured* `sync_s` reflects a dense transfer —
> so the sparse-bandwidth win shows in comm-volume, not in real wall-clock).
> (2) **BatchNorm running stats are NOT synced** (matches `SimCluster`; rank-0
> eval uses rank-0's local BN stats — documented discrepancy).
> (3) **§4 recalibration fitting script deferred**; the grove path already logs
> everything it needs (`comm_bytes`, `sync_s_real`, `compute_s_real`, `link`).

Phases 1–6 deliver convergence, communication-volume, and adaptive-policy
results entirely on **one machine** via `SimCluster`
(`src/macluster/backends/sim.py`) plus the analytic link emulator
(`src/macluster/emulation/link.py`). What single-machine emulation *cannot*
produce is **real wall-clock speedup**: two MacBooks actually computing
concurrently and exchanging updates over a real AWDL/Wi-Fi radio. Phase 7 adds a
real backend, `GroveBackend`, built on [`grove-ml`](https://pypi.org/project/grove-ml/)
(`grove` import name, the centerpiece tool), and a runbook for pairing two
MacBooks. Crucially, **the `Task` / `Algorithm` / `TrainConfig` contracts do not
change** — only the backend that executes a round.

---

## 1. The grove API to wrap

`grove` is already a dependency (`pyproject.toml`) and is installed in the venv.
The public surface (`import grove; dir(grove)`) we will wrap:

| grove symbol | Signature (as installed, v0.1.0) | Use in `GroveBackend` |
| --- | --- | --- |
| `grove.init(...)` | `init(cluster=None, world_size=None, sync_every=1, timeout=120.0, transport='p2p') -> World` | join the cluster, learn `rank` / `world_size` |
| `grove.World` | returned by `init` | process-group handle |
| `grove.rank()` / `grove.world_size()` | `-> int` | this process's id / cluster size |
| `grove.all_sum(x)` | element-wise all-reduce (sum) of an MLX array | dense averaging, pseudo-gradient reduce |
| `grove.all_gather(x)` | gather arrays from all ranks | sparse `(idx, val)` gather for SparseLoCo |
| `grove.average_gradients(grads: dict) -> dict` | all-reduce + divide a gradient tree | one-call dense gradient averaging |
| `grove.barrier() -> None` | synchronization barrier | round boundaries / timing fences |
| `grove.diloco(model, outer_lr=0.7, outer_momentum=0.9, H=500, overlap=False, quantize=False)` | built-in DiLoCo optimizer | maps to our `DiLoCo(H, outer_lr)` |
| `grove.sparseloco(model, outer_lr=1.0, H=30, error_decay=0.95, topk=64, chunk=4096, overlap=True)` | built-in SparseLoCo | maps to our `SparseLoCo(H, outer_lr, k_frac, error_decay)` |
| `grove.send` / `grove.recv` / `grove.recv_like` | point-to-point | low-level fallback if needed |
| `grove.report(...)` | progress / metrics reporting to the coordinator | feed our `RunLogger` |
| `grove.is_available()` | guard | skip the backend gracefully when grove can't init |

> **Knob mapping (macluster → grove).** `TrainConfig.H` → `H` (and grove's
> `init(sync_every=...)`); `outer_lr`/`outer_momentum` map directly;
> SparseLoCo's `k_frac` (a *fraction*) must be converted to grove's `topk` (an
> *integer count*): `topk = max(1, round(k_frac * n_params_per_chunk))`, with
> grove's `chunk=4096`. Decide whether to **reuse our `Algorithm`
> implementations** over `grove.all_sum`/`all_gather` (keeps axis-2/axis-3 code
> identical, including the adaptive controller) or **delegate to
> `grove.diloco`/`grove.sparseloco`** (less code, but the adaptive policy in
> `algorithms/adaptive.py` still needs our own loop to vary `H`/`k_frac` per
> round). Recommended: reuse our `Algorithm`s on top of grove's collectives, so
> axis 3 stays ours.

The grove CLI (`uv run grove --help`) exposes `run | start | join | status`,
used in the runbook below.

---

## 2. `GroveBackend` — mirror `SimCluster`

Create `src/macluster/backends/grove_backend.py` mirroring the role of
`SimCluster` so that `train.run_training` can target it behind a
`--backend {sim,grove}` flag. **Each MacBook runs ONE replica** (its own
`rank`), instead of `W` replicas in one process.

`SimCluster` responsibilities and their grove equivalents:

| `SimCluster` | `GroveCluster` (Phase 7) |
| --- | --- |
| holds `W` `Replica`s in one process | holds **this rank's single** model + optimizer; `world_size` comes from `grove.world_size()` |
| `load_global(params)` | local model already holds global params after each `sync`; on round start just keep them (no broadcast needed if every rank applied the same averaged update) |
| `collect_params()` → list of `W` trees | **not** a local list — replaced by a collective inside `sync`: each rank contributes its own tree, grove reduces |
| `inner_step()` per replica, timed | identical local inner step; time it per-rank |
| round compute time = **max over replicas** (emulated parallel) | round compute time = **real**: `grove.barrier()` before and after the inner loop; the slowest rank gates the round for real |
| `sync(...)` mutates replicas back to one state | `Algorithm.sync` reimplemented over collectives: dense → `all_sum` then `/world_size`; DiLoCo → `all_sum` of the pseudo-gradient; SparseLoCo → `all_gather` of `(idx, val)` buffers |
| link cost charged **analytically** by `emulation/link.py` | link cost is **measured** as the real wall-clock of the collective (`perf_counter` around `all_sum`/`all_gather`) |

Key adaptation in the round loop (`train.run_training`): when `backend ==
"grove"`, **do not** call `LinkSchedule`/`allreduce_bytes` to *charge* time;
instead **measure** `sync_s` directly around the collective and **measure**
`comm_bytes` from the actual serialized payload. The logged record keeps the
same field names (`compute_s`, `sync_s_sim` → rename to `sync_s_real` or keep
the key and note the unit), so `scripts/plot.py` works unchanged. The adaptive
controller's `observe(compute_s, sync_s, link_name)` now receives **real**
timings — the whole point of Phase 7.

A minimal sketch (illustrative, not final):

```python
# src/macluster/backends/grove_backend.py  (Phase 7)
import grove, mlx.core as mx, time

class GroveCluster:
    def __init__(self, task, model_name, inner_opt_fn):
        self.world = grove.init()               # cluster/world_size from `grove start`
        self.rank = grove.rank()
        self.model = task.model_fns[model_name]()
        self.model.train()
        self.opt = inner_opt_fn()
        self.data = task.train_shards[self.rank]   # this rank's shard
        self._lvg = mx.nn.value_and_grad(self.model, task.loss_fn) \
            if False else None                     # use nn.value_and_grad like sim.py

    def inner_step(self) -> float:
        X, y = next(self.data)
        loss, grads = self._lvg(self.model, X, y)
        self.opt.update(self.model, grads)
        mx.eval(self.model.parameters(), self.opt.state, loss)
        return float(loss)

    # Algorithm.sync is reimplemented over grove.all_sum / all_gather:
    #   dense:      avg = all_sum(param) / world_size()
    #   diloco:     avg_pseudo = all_sum(global - local) / world_size()
    #   sparseloco: gathered = all_gather(pack(idx, val)); decode; mean
```

`grove.is_available()` should gate construction so `--backend sim` keeps working
on a machine without a cluster.

---

## 3. 2-MacBook pairing runbook

Two MacBooks (`mac-A` = launcher, `mac-B` = joiner) on Apple Silicon, both with
this repo synced and `uv sync` run.

### 3.1 Network: AWDL discovery, then Wi-Fi upgrade

1. **AWDL bring-up (discovery).** AWDL is Apple's peer-to-peer link; the easiest
   way to force the interface up between two Macs is to **open an AirDrop window
   on both** (Finder → AirDrop, "Allow me to be discovered by: Everyone").
   This activates `awdl0`. `grove status` should then list nearby peers.
2. **Wi-Fi upgrade (the real link).** For higher goodput, put both Macs on the
   **same Wi-Fi network** (or have the launcher create a hotspot the joiner
   joins). grove's `transport='p2p'` will use the best available path; the
   distinction between "AWDL-only" and "Wi-Fi-upgraded" is exactly axis 1, now
   measured rather than emulated.
3. Confirm reachability before training (`grove status` on both; optionally a
   plain `ping` between the two Wi-Fi addresses to read real RTT — this is the
   first number to feed back into the emulator calibration, §4).

### 3.2 Launch the cluster

Both Macs run the committed `scripts/grove_entry.py` (the only launchable form:
grove imports a Python file with `main()`, and the **joiner gets no argv**). The
`TrainConfig` is supplied through `MACLUSTER_*` env vars that must be exported
**identically on both Macs** — `grove_entry` runs a config-consensus check
(`grove.all_sum` of a config hash) and aborts loudly on mismatch, since a
divergent config silently breaks the deterministic data sharding. `world_size`
is bound to the real cluster size automatically (one data shard per rank).

**Recommended: the `scripts/grove_run.sh` wrapper** — it sources a committed
`configs/grove/*.env` (same bytes on both repos → parity for free) and issues the
right `grove` command, so each Mac is one line:

```bash
./scripts/grove_run.sh check                            # both: peer discoverable?
./scripts/grove_run.sh start configs/grove/diloco.env   # mac-A (launcher)
./scripts/grove_run.sh join  configs/grove/diloco.env   # mac-B (joiner, same config)
```

Configs provided: `dense.env`, `diloco.env`, `sparseloco.env`, `adaptive.env`
(real CIFAR, `MAX_STEPS=500` equal-compute budget) and `smoke.env` (synthetic).
Edit `MACLUSTER_LINK=awdl` to exercise the degraded-link adaptive response. The
manual form below is equivalent.

```bash
# on mac-A (launcher): start a 2-node cluster named `macluster` (-n 2 = world size 2)
export MACLUSTER_TASK=cifar10 MACLUSTER_MODEL=resnet20 \
       MACLUSTER_ALGORITHM=diloco MACLUSTER_ROUNDS=25 MACLUSTER_H=20 \
       MACLUSTER_LINK=wifi MACLUSTER_SEED=0 MACLUSTER_RUNS_ROOT=runs
uv run grove start scripts/grove_entry.py -n 2 --name macluster --logs

# on mac-B (joiner): export the SAME MACLUSTER_* env, then join the named cluster
export MACLUSTER_TASK=cifar10 MACLUSTER_MODEL=resnet20 \
       MACLUSTER_ALGORITHM=diloco MACLUSTER_ROUNDS=25 MACLUSTER_H=20 \
       MACLUSTER_LINK=wifi MACLUSTER_SEED=0 MACLUSTER_RUNS_ROOT=runs
uv run grove join macluster --logs
```

Single-machine smoke (no second Mac; collectives are W=1 no-ops), to validate
the entry/backend before pairing:

```bash
MACLUSTER_SYNTHETIC=1 MACLUSTER_ALGORITHM=dense MACLUSTER_ROUNDS=2 \
    uv run python scripts/grove_entry.py            # or: uv run grove run scripts/grove_entry.py --logs
# and the in-process CLI path:
uv run macluster train --backend grove --world-size 1 --synthetic --rounds 2
```

Recognised env vars (all optional; default to `TrainConfig`'s defaults):
`MACLUSTER_{TASK,MODEL,ALGORITHM,ROUNDS,MAX_STEPS,BATCH_SIZE,SEQ_LEN,H,K_FRAC,`
`OUTER_LR,INNER_OPT,INNER_LR,LINK,EVAL_EVERY,SEED,SYNTHETIC,DATA_DIR,`
`TARGET_METRIC,RUNS_ROOT}`.

Each rank writes its **own** `runs/<slug>/` on its **own** machine; collect both
and compare rank-0's metrics (eval runs only on rank 0, mirroring
`SimCluster.eval_model`).

### 3.3 What to capture

- **Real round wall-clock**: `compute_s` (local inner loop), `sync_s_real`
  (collective), and the gap between the fastest and slowest rank (straggler).
- **Real comm bytes**: serialized payload of each `all_sum` / `all_gather`.
- **Speedup**: real wall-clock to target accuracy on 2 Macs vs. a 1-Mac
  single-replica baseline — the number Phase 1–6 explicitly could not produce.

---

## 4. Recalibrate the emulator against real measurements

The single-machine results are only as honest as the link model in
`emulation/link.py`. Phase 7's measurements are what calibrate it. For each link
condition exercised on real hardware, measure and overwrite the corresponding
`LinkProfile` fields:

| Emulated field (`emulation/link.py`) | Real measurement to calibrate it |
| --- | --- |
| `WIFI_UPGRADED.bandwidth_mbps` (300) | effective goodput of a large `all_sum` over Wi-Fi: `payload_bits / measured_sync_s` |
| `WIFI_UPGRADED.latency_ms` (3) / `jitter_ms` (1) | RTT/2 and its std-dev from a tiny-payload sync (or `ping`) on Wi-Fi |
| `AWDL_ONLY.bandwidth_mbps` (60) / `latency_ms` (10) / `jitter_ms` (6) | same measurements with both Macs on AWDL-only (AirDrop window open, no shared Wi-Fi) |
| `WIFI_DEGRADED.*` (40 / 20 / 10) | measure on a congested / distant Wi-Fi link, or drop it if not reproduced |
| `allreduce_bytes(...)` topology factor | check the real per-worker bytes for `W=2` against the `ring` (`2(W-1)/W`) and `gather` (`×W`) models |

**Workflow.** Run the 2-Mac job under each link condition, read the real
`sync_s` vs. `comm_bytes` pairs from rank-0's `metrics.jsonl`, fit
`bandwidth_mbps` and `latency_ms` (slope/intercept of `sync_s` vs. bytes), then
update the profile constants. Re-run the **emulated** sweeps (Phases 1–6) with
the recalibrated profiles so the single-machine figures in the report agree with
the two-machine ground truth — and explicitly note in the report which numbers
are emulated-but-calibrated vs. directly measured.

---

## 5. Open questions / risks to resolve in Phase 7

- **Eval placement.** Only rank 0 evaluates (matches `SimCluster.eval_model`);
  confirm rank 0 always holds the post-sync global params on a real cluster.
- **BatchNorm running stats.** `ResNetCIFAR` uses `nn.BatchNorm`; these buffers
  are not in `trainable_parameters()` and are therefore **not** synchronized
  (true today in `SimCluster` too). Decide whether to all-reduce them for the
  real run or document the discrepancy.
- **Straggler / timeout.** `grove.init(timeout=120.0)` and `barrier()` behavior
  if one Mac sleeps or drops AWDL mid-run; needs a reconnect story.
- **`overlap` / `quantize`.** grove's `diloco`/`sparseloco` support
  compute/comm `overlap` and `quantize`; out of scope for the first correctness
  run, but a natural follow-up for additional speedup once correctness holds.
- **Reuse vs. delegate.** Final call on §1's choice — keeping our `Algorithm`s
  over grove collectives is required to keep the adaptive policy (axis 3) intact.
