# Phase 8: Model Parallelism (pipeline) — DP vs MP on 2 heterogeneous Macs

**Status: IMPLEMENTED (sim path fully validated single-machine; grove path
W=1-smoke-tested, awaits the real 2-Mac run).** This adds a **model-parallel**
axis to macluster so we can train a model **too big for a single node** by
splitting it across the two Macs, and compare it against the existing
**data-parallel** methods (DiLoCo / SparseLoCo / adaptive). The maturity bar
matches Phase 7's GroveBackend: the single-machine emulation is end-to-end
validated, and the real-cluster backend is correct-by-construction + W=1
smoke-tested, pending the borrowed 24GB Mac for the two-node run.

Motivation and the why-it's-needed argument: the existing algorithms are
data-parallel — every node holds a FULL model replica, so the cluster cannot
train a model larger than the smallest node (the borrowed **24GB** Mac). To train
a model that does not fit in a single node we need to **split one model across
devices**. The reference method is **Asteroid** (`docs/asteroid.pdf`, MobiCom'24):
Hybrid Pipeline Parallelism (inter-stage pipeline + intra-stage data parallelism)
with a **memory-aware partition** for heterogeneous edge devices — a close match
to our asymmetric {48GB, 24GB} pair over Wi-Fi/AWDL. The other three `docs/`
papers are not model-parallel (StellaTrain = data-parallel; FedConv = federated
model-shrink; Crux = datacenter network scheduler).

## Hardware target
- mac-A: 48GB (in hand). mac-B: 24GB (**borrowable later**, not now).
- Both Apple Silicon, MLX, unified memory. Real link = AWDL/Wi-Fi (axis 1).
- Build + validate everything **single-machine** now; run on 2 Macs later.

## Validated technical core (the seam)
A 2-stage split reproduces the monolithic backward exactly in MLX (verified:
grads match to ~1e-8, loss diff 0). The seam:
1. **stage0** forward `h = f0(x)` → send `h` to stage1 (grove `send`).
2. **stage1** forward+loss; `mx.value_and_grad(stage1_loss, argnums=(params1, h))`
   yields stage1 param grads **and** `dL/dh`. Send `dL/dh` back (grove `send`).
3. **stage0** backward via the surrogate scalar `⟨stop_grad(dL/dh), h(params0)⟩`;
   `mx.grad` of it gives stage0's param grads, identical to monolithic.
4. each stage runs its own optimizer step on its own params.

This avoids `mx.vjp`'s list-of-arrays API and uses only `value_and_grad`/`grad`.

## Design decisions
- **Untied LM head.** The base `GPT` ties the LM head to `wte` (`wte.as_linear`),
  which lives on stage0 — incompatible with a clean stage cut. The pipeline model
  uses an **untied** head (a separate `nn.Linear`) on the last stage (+`vocab*n_embd`
  params, ~38.6M at vocab 50257). For an apples-to-apples DP-vs-MP comparison the
  **DP runs use the same untied-head model**, so both paradigms train identical
  architectures. (Documented discrepancy vs the tied gpt124m used in Phase 1-6.)
- **Fixed cut first, profiling later.** v1 = a single hand-chosen cut point sized
  by a per-Mac parameter-memory budget (more layers on the 48GB Mac, fewer on the
  24GB Mac). v2 = an Asteroid-style profile-then-search auto-partition.
- **1F1B micro-batch schedule** to bound peak activation memory at O(warmup)
  instead of O(#micro-batches). For 2 stages the warmup is tiny (1). On the real
  grove path the schedule is ordered to keep each seam **complementary** (when a
  stage sends, its neighbour receives), so the blocking `send`/`recv` cannot
  deadlock; for 2 stages this is a strictly alternating send/recv on the single
  seam. The single-process `SimPipelineCluster` runs the micro-batches
  sequentially (one process — no schedule/deadlock concerns) and just accumulates
  grads, so the schedule only matters for `PipelineCluster` (grove).
- **2 nodes ⇒ no intra-stage DP groups, no straggler offloading, no fault
  tolerance** — those Asteroid pieces assume ≥4 devices; skip for the 2-Mac case.
  grove `send`/`recv`/`barrier` suffice; `all_sum`/`all_gather` not needed for MP.

## Code plan
- `models/gpt.py`: `GPTStage` (owns optional embed / a block range / optional
  `ln_f`+untied head) and `split_gpt(cfg, cut, untie_head=True) -> [stage0, stage1]`.
  Add larger configs for the comparison (e.g. `gpt_xl` ~1.5B, `gpt3b` ~3B).
- `backends/pipeline.py`: `SimPipelineCluster` (both stages in one process —
  single-machine validation + analytic link/timing emulation, mirroring
  `SimCluster`) and `PipelineCluster` (grove: rank0=stage0, rank1=stage1, real
  `send`/`recv`, measured timings, mirroring `GroveCluster`). 1F1B schedule lives
  here; per-stage optimizer step here.
- `train.py` + `cli.py`: a `--parallelism {data,pipeline}` flag. `data` keeps the
  existing Algorithm/sync round loop unchanged; `pipeline` routes to the pipeline
  executor. Logged record keeps stable field names so `plot.py` works.
- `tests/test_pipeline.py`: numerical equivalence (pipeline step == monolithic
  step), single-machine pipeline smoke, memory-aware cut sizing.

## Comparison plan (the headline)
- **Mid model** that fits one node: run BOTH data-parallel (DiLoCo/SparseLoCo) and
  model-parallel (pipeline) → compare wall-clock, comm volume, convergence
  apples-to-apples.
- **Large model** (~3B, fp32 state >48GB) that does NOT fit one node: only the
  pipeline can train it → demonstrates breaking the single-node memory wall, which
  data parallelism provably cannot. Memory math: split ~2/3 of params onto the
  48GB Mac, ~1/3 onto the 24GB Mac (each within its budget + 1F1B activation room).

## Logged metrics (for the report)
Each run writes `runs/<slug>-rank{r}/{config.json, metrics.jsonl, summary.json}`.
Per-round records (DP and MP share field names so `plot.py`/sweeps work on both):
- **convergence**: `train_loss` every round; `val_loss` + `perplexity` (text) /
  `accuracy` (cifar) on eval rounds. On MP+grove these live on the LAST rank.
- **communication**: `comm_bytes`, `comm_bytes_cum`, summary `total_comm_MB`
  (DP = parameter all-reduce; MP = activation-forward + cotangent-back seam).
- **time**: `compute_s`, `sync_s_sim`, `round_s_sim`, cumulative `sim_time_s`;
  on grove the MEASURED `round_s_real` (+ `comm_s_real` for MP = seam transfer +
  pipeline-bubble wait; `compute_s_real`/`sync_s_real` for DP).
- **memory**: `peak_mem_mb` — MEASURED peak unified memory (`mx.get_peak_memory`)
  per replica (DP) / per stage (MP). This is the hard evidence for the headline:
  on the gpt3b MP run rank0~30GB / rank1~15GB, each within its node, while the
  full model's ~44GB fp32 Adam state never fits one node for DP.
- **throughput / partition**: `throughput_sps`, `samples`; MP summary adds
  `n_stages`, `cut`, `stage_param_counts`, `model_param_count`.

Both ranks log locally, so rank1's loss/perplexity start on the 24GB Mac;
`scripts/run_mac_24gb.sh` copies `runs/` to the 48GB Mac (set `RESULTS_DEST`).

## Open questions
- Cross-Wi-Fi/AWDL latency in the 1F1B seam: real jitter may bubble the pipeline
  more than Asteroid's wired-Jetson numbers; measure on the real 2-Mac run.
- Eval under MP: the full model is split; eval runs the whole pipeline forward
  (or assemble stages on one node if it fits). Decide per model size.
  **Resolved (v1):** `PipelineCluster.eval_loss` runs a forward-only pipeline
  pass and reports `val_loss`/`perplexity` on the last rank; `SimPipelineCluster`
  reassembles a whole-model view (`eval_model()`) for the task's `eval_fn`.
- Where the embedding/positional params and final head sit relative to the cut
  (parameter-dense; keep them on the larger-memory Mac). **v1:** embeddings on
  stage 0, untied head on the last stage; the memory-aware cut already accounts
  for both when balancing utilisation, so the dense ends bias block placement.

## Implementation status (what shipped)
- `models/gpt.py`: `GPTStage`/`split_gpt`/`build_stage` (single-rank), the
  `gpt_xl` (~1.6B) / `gpt3b` (~2.7B) configs + `GPT_CONFIGS` registry, and the
  analytic `gpt_param_count`/`stage_param_counts`/`memory_aware_cut` (no
  instantiation — a 3B model is sized without ever being built).
- `backends/pipeline.py`: `pipeline_grads` (the validated seam oracle),
  `seq_cross_entropy`, and `SimPipelineCluster` (all stages in one process;
  single-machine validation + correctness oracle).
- `backends/pipeline_grove.py`: `PipelineCluster` (grove; rank == stage; 1F1B
  over real `send`/`recv`; forward-only eval). Imports grove at module top like
  `grove_backend.py`, so the sim core still runs without grove installed.
- `train.py`/`cli.py`: `--parallelism {data,pipeline}` plus `--cut`,
  `--n-micro`, `--stage-mem-gb`. `run_training` delegates to
  `run_pipeline_training`, which logs the same field names as the DP loop (so
  `plot.py` / sweeps work unchanged).
- `scripts/grove_entry.py` + `configs/grove/pipeline.env`: the 2-Mac launch path.

**Still open (NOT in this slice):** the apples-to-apples comparison wants the DP
runs to use the *untied-head* model too (Design decisions §1); the DP path still
trains the tied `gpt124m`. Add an untied full-GPT `model_fn` before running the
mid-model DP-vs-MP head-to-head. Also: v2 auto-partition by profiling, and >2
stages (the grove schedule is written generally but only validated for 2).

## How to run
```bash
# Single-machine validation NOW (exercises the real seam across 2 stages in one
# process — the 24GB Mac isn't needed): 
uv run macluster train --task shakespeare --synthetic --parallelism pipeline \
    --model chargpt --world-size 2 --cut 2 --n-micro 4 --rounds 10

# Memory-aware auto cut for the asymmetric pair (more blocks on the 48GB Mac):
uv run macluster train --task wikitext --parallelism pipeline --model gpt2 \
    --world-size 2 --stage-mem-gb 48,24 --rounds 50

# Real 2-Mac run -- turnkey. The 48GB Mac MUST be the launcher (rank0 = the bigger
# stage); start it first, then the 24GB Mac within ~2 min:
./scripts/run_mac_48gb.sh    # on the 48GB Mac (rank0 = stage0)
./scripts/run_mac_24gb.sh    # on the 24GB Mac (rank1 = stage1)
```
The two scripts run three phases in order — `smoke` (synthetic connectivity),
`xl` (gpt_xl ~1.6B, fits both), `3b` (gpt3b ~2.78B, the headline) — each a full
2-Mac run, so even if `3b` OOMs you still have the `smoke`+`xl` results. Run a
subset with the same arg on both Macs, e.g. `./scripts/run_mac_48gb.sh xl`. The
memory-aware cut `[48,24]` puts ~2/3 of the blocks (stage0) on the 48GB Mac;
`gpt3b`'s fp32 Adam state (~44GB) exceeds a single 48GB node, so only the split
can train it — the point data parallelism cannot make. Configs live in
`configs/grove/pipeline_{smoke,xl,3b}.env` (same bytes sourced on both Macs).
