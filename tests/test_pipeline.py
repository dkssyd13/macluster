"""Phase 8 pipeline (model) parallelism: the split must reproduce the monolithic
backward exactly. See docs/PHASE8_MODELPARALLEL.md."""
import json
import os

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from macluster.backends.pipeline import SimPipelineCluster, pipeline_grads, seq_cross_entropy
from macluster.models.gpt import (
    GPTConfig,
    build_stage,
    gpt3b_cfg,
    gpt_param_count,
    gpt_xl_cfg,
    memory_aware_cut,
    num_params,
    split_gpt,
    stage_param_counts,
)


def _ce(logits, y):
    B, T, V = logits.shape
    return nn.losses.cross_entropy(logits.reshape(B * T, V), y.reshape(B * T), reduction="mean")


def _maxdiff(ga, gb):
    fa, fb = dict(tree_flatten(ga)), dict(tree_flatten(gb))
    assert fa.keys() == fb.keys(), set(fa) ^ set(fb)
    return max(float(mx.max(mx.abs(fa[k] - fb[k]))) for k in fa)


def _monolithic(stages, x, y):
    """Reference: forward through the composed stages and grad over all params."""

    def loss(params_list):
        h = x
        for s, p in zip(stages, params_list):
            s.update(p)
            h = s(h)
        return _ce(h, y)

    return mx.value_and_grad(loss)([s.trainable_parameters() for s in stages])


def _check_equivalence(cfg, cuts, B=8, T=32, seed=0):
    mx.random.seed(seed)
    stages = split_gpt(cfg, cuts)
    mx.eval([s.parameters() for s in stages])
    x = mx.random.randint(0, cfg.vocab_size, (B, T))
    y = mx.random.randint(0, cfg.vocab_size, (B, T))

    loss_pp, grads_pp = pipeline_grads(stages, x, y, _ce)
    loss_ref, grads_ref = _monolithic(stages, x, y)

    assert abs(float(loss_pp) - float(loss_ref)) < 1e-5
    assert len(grads_pp) == len(grads_ref) == len(cuts) + 1
    for gp, gr in zip(grads_pp, grads_ref):
        assert _maxdiff(gp, gr) < 1e-4


def test_pipeline_seam_matches_monolithic_2stage():
    cfg = GPTConfig(vocab_size=256, block_size=64, n_layer=4, n_head=4, n_embd=128)
    _check_equivalence(cfg, cuts=[2])


def test_pipeline_seam_matches_monolithic_3stage():
    cfg = GPTConfig(vocab_size=256, block_size=64, n_layer=6, n_head=4, n_embd=128)
    _check_equivalence(cfg, cuts=[2, 4])


def test_split_partitions_all_params():
    """Stages must collectively cover every parameter (no layer dropped); first
    stage owns embeddings, last owns the untied head."""
    cfg = GPTConfig(vocab_size=256, block_size=64, n_layer=4, n_head=4, n_embd=128)
    stages = split_gpt(cfg, [1])
    mx.eval([s.parameters() for s in stages])
    assert stages[0].is_first and not stages[0].is_last
    assert stages[-1].is_last and not stages[-1].is_first
    # every stage holds a positive parameter count
    assert all(num_params(s) > 0 for s in stages)
    # untied head lives on the last stage (vocab*n_embd weight)
    assert hasattr(stages[-1], "head") and not hasattr(stages[0], "head")
    assert hasattr(stages[0], "wte") and not hasattr(stages[-1], "wte")


def test_split_rejects_bad_cuts():
    cfg = GPTConfig(vocab_size=256, block_size=64, n_layer=4, n_head=4, n_embd=128)
    for bad in ([0], [4], [2, 2], [3, 1]):
        try:
            split_gpt(cfg, bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for cuts={bad}")


def test_build_stage_matches_split_structure():
    """A single-rank stage (real-cluster path) has the same shape/param-count as
    the matching stage from the all-in-one-process split."""
    cfg = GPTConfig(vocab_size=256, block_size=64, n_layer=4, n_head=4, n_embd=128)
    mx.random.seed(0)
    full = split_gpt(cfg, [2])
    s0, s1 = build_stage(cfg, [2], 0), build_stage(cfg, [2], 1)
    mx.eval([s.parameters() for s in (s0, s1, *full)])
    assert num_params(s0) == num_params(full[0])
    assert num_params(s1) == num_params(full[1])
    assert s0.is_first and not s0.is_last
    assert s1.is_last and not s1.is_first


# --------------------------------------------------------------------------- #
# analytic param accounting + memory-aware partition
# --------------------------------------------------------------------------- #
def test_large_configs_param_counts():
    """gpt_xl ~1.5B, gpt3b ~3B, computed analytically (never instantiated)."""
    assert 1.4e9 < gpt_param_count(gpt_xl_cfg(50257)) < 1.8e9
    assert 2.4e9 < gpt_param_count(gpt3b_cfg(50257)) < 3.2e9
    for cfg in (gpt_xl_cfg(50257), gpt3b_cfg(50257)):
        assert cfg.n_embd % cfg.n_head == 0  # head_dim divides evenly


def test_stage_param_counts_sum_to_total():
    cfg = gpt_xl_cfg(50257)
    counts = stage_param_counts(cfg, [16, 32])
    assert sum(counts) == gpt_param_count(cfg)
    assert len(counts) == 3


def test_memory_aware_cut_is_heterogeneity_aware():
    cfg = gpt3b_cfg(50257)  # 32 layers
    even = memory_aware_cut(cfg, [1, 1])
    skew = memory_aware_cut(cfg, [48, 24])           # 2:1 budget -> more blocks on stage 0
    assert len(even) == len(skew) == 1
    assert skew[0] >= even[0]                          # bigger budget -> later cut
    c = stage_param_counts(cfg, skew)
    assert c[0] > c[1]                                 # stage 0 holds the larger share
    # stage 0 fits 48GB and stage 1 fits 24GB under fp32 Adam (16 B/param)
    assert c[0] * 16 < 48e9 and c[1] * 16 < 24e9
    # a 3-way split returns 2 cuts, one stage per budget
    assert len(memory_aware_cut(cfg, [40, 24, 16])) == 2


# --------------------------------------------------------------------------- #
# SimPipelineCluster: one optimizer step == monolithic grad-accumulation step
# --------------------------------------------------------------------------- #
def test_sim_pipeline_cluster_step_matches_monolithic_accum():
    import mlx.optimizers as optim

    cfg = GPTConfig(vocab_size=256, block_size=64, n_layer=4, n_head=4, n_embd=128)
    mx.random.seed(3)
    n_micro = 3
    batches = [
        (mx.random.randint(0, cfg.vocab_size, (4, 16)), mx.random.randint(0, cfg.vocab_size, (4, 16)))
        for _ in range(n_micro)
    ]

    def data_gen():
        k = 0
        while True:
            yield batches[k % n_micro]
            k += 1

    lr = 0.05
    cl = SimPipelineCluster(cfg, [2], lambda: optim.SGD(learning_rate=lr), data_gen(), seq_cross_entropy)

    # monolithic reference: accumulate (then mean) the grads at the INIT params,
    # apply one plain-SGD step by hand, and check the cluster lands in the same place.
    init = [tree_map(lambda x: mx.array(x), s.trainable_parameters()) for s in cl.stages]
    acc = [tree_map(lambda x: mx.zeros_like(x), s.trainable_parameters()) for s in cl.stages]
    for X, y in batches:
        _, grads = _monolithic(cl.stages, X, y)
        acc = [tree_map(lambda a, b: a + b, ai, gi) for ai, gi in zip(acc, grads)]
    expected = [
        tree_map(lambda p, g: p - lr * (g / n_micro), p_init, g_acc)
        for p_init, g_acc in zip(init, acc)
    ]

    cl.step(n_micro)
    for e, stage in zip(expected, cl.stages):
        assert _maxdiff(e, stage.trainable_parameters()) < 1e-4


def test_sim_pipeline_cluster_seam_bytes():
    cfg = GPTConfig(vocab_size=256, block_size=64, n_layer=6, n_head=4, n_embd=128)
    import mlx.optimizers as optim

    def data_gen():
        while True:
            yield mx.random.randint(0, cfg.vocab_size, (4, 16)), mx.random.randint(0, cfg.vocab_size, (4, 16))

    cl = SimPipelineCluster(cfg, [2, 4], lambda: optim.Adam(learning_rate=1e-3), data_gen(), seq_cross_entropy)
    _, seam = cl.step(n_micro=2)
    # 3 stages -> 2 seams; per seam per micro: h fwd + cot bwd = 2*B*T*C fp32 bytes
    B, T, C = 4, 16, 128
    per_seam_per_micro = 2 * B * T * C * 4
    assert seam == 2 * 2 * per_seam_per_micro  # n_micro * n_seams * per_seam_per_micro


# --------------------------------------------------------------------------- #
# end-to-end executor smoke (sim backend, 2 stages in one process)
# --------------------------------------------------------------------------- #
def test_run_pipeline_training_sim_smoke(tmp_path):
    from macluster.train import TrainConfig, run_training

    cfg = TrainConfig(
        task="shakespeare", model="chargpt", parallelism="pipeline",
        world_size=2, cut="2", n_micro=2, rounds=4, batch_size=8, seq_len=32,
        synthetic=True, inner_opt="adam", inner_lr=1e-3,
    )
    run_dir = str(tmp_path / "pp")
    summary = run_training(cfg, run_dir)
    assert summary["parallelism"] == "pipeline"
    assert summary["n_stages"] == 2
    assert summary["cut"] == [2]
    assert summary["final_train_loss"] is not None
    assert summary["model_param_count"] == sum(summary["stage_param_counts"])

    recs = [json.loads(line) for line in open(os.path.join(run_dir, "metrics.jsonl"))]
    assert len(recs) == 4
    # stable field names so plot.py / sweeps keep working
    for r in recs:
        for k in ("round", "train_loss", "sim_time_s", "comm_bytes_cum", "samples"):
            assert k in r


def test_run_pipeline_training_auto_cut(tmp_path):
    """No --cut: the memory-aware partition sizes the split from --stage-mem-gb."""
    from macluster.train import TrainConfig, run_training

    cfg = TrainConfig(
        task="shakespeare", model="gpt2", parallelism="pipeline",
        world_size=2, stage_mem_gb="48,24", n_micro=2, rounds=2,
        batch_size=4, seq_len=32, synthetic=True,
    )
    summary = run_training(cfg, str(tmp_path / "auto"))
    assert summary["n_stages"] == 2
    # heterogeneous budget -> stage 0 gets the larger share
    assert summary["stage_param_counts"][0] > summary["stage_param_counts"][1]


def test_pipeline_rejects_non_text_task():
    from macluster.train import TrainConfig, run_training

    cfg = TrainConfig(task="cifar10", parallelism="pipeline", world_size=2, cut="2")
    try:
        run_training(cfg, "/tmp/should_not_exist")
    except KeyError:
        return
    raise AssertionError("expected KeyError for a non-GPT pipeline task")
