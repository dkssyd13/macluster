"""DP-vs-MP head-to-head: the data-parallel path must be able to train the SAME
untied-head GPT that the pipeline split trains, so the comparison is
apples-to-apples (see docs/PHASE8_MODELPARALLEL.md). Exercises the new
``gpt2_untied`` model_fn and the monolithic-untied == pipeline-split parity."""
import json
import os

import mlx.core as mx

from macluster.models.gpt import (
    GPT,
    GPTConfig,
    gpt124m_cfg,
    gpt124m_untied,
    gpt_param_count,
    num_params,
    split_gpt,
    stage_param_counts,
)

_SMALL = GPTConfig(vocab_size=256, block_size=64, n_layer=4, n_head=4, n_embd=128)


def test_untied_head_param_accounting():
    """An untied head adds exactly vocab*n_embd params over the tied model, and
    the untied count matches the analytic gpt_param_count (which assumes untied)."""
    mx.random.seed(0)
    tied = GPT(_SMALL)                       # default: weight-tied head
    untied = GPT(_SMALL, untie_head=True)
    mx.eval(tied.parameters(), untied.parameters())
    assert untied.untie_head and not tied.untie_head
    assert hasattr(untied, "head") and not hasattr(tied, "head")
    assert num_params(untied) - num_params(tied) == _SMALL.vocab_size * _SMALL.n_embd
    assert num_params(untied) == gpt_param_count(_SMALL)


def test_dp_monolith_matches_mp_split_param_for_param():
    """The DP model (monolithic untied GPT) and the MP model (pipeline split)
    have IDENTICAL parameter counts -> they train the same architecture."""
    mx.random.seed(0)
    monolith = GPT(_SMALL, untie_head=True)
    stages = split_gpt(_SMALL, [2])
    mx.eval(monolith.parameters(), [s.parameters() for s in stages])
    split_total = sum(num_params(s) for s in stages)
    assert num_params(monolith) == split_total
    assert split_total == sum(stage_param_counts(_SMALL, [2]))


def test_gpt124m_untied_is_registered_model():
    """gpt124m_untied builds and its size matches the analytic untied count
    (~163M at vocab 50257: the gpt2-class apples-to-apples comparison model)."""
    m = gpt124m_untied(256)  # tiny vocab to keep the test light
    mx.eval(m.parameters())
    assert num_params(m) == gpt_param_count(gpt124m_cfg(256))
    # at the real wikitext vocab the analytic count is ~163M
    assert 1.5e8 < gpt_param_count(gpt124m_cfg(50257)) < 1.8e8


def test_dp_gpt_sim_smoke(tmp_path):
    """The data-parallel path trains gpt2_untied end-to-end (the half that was
    previously impossible: gpt_xl/gpt3b have no DP model_fn, gpt2_untied does)."""
    from macluster.train import TrainConfig, run_training

    cfg = TrainConfig(
        task="wikitext", model="gpt2_untied", algorithm="diloco", parallelism="data",
        world_size=2, max_steps=4, H=2, batch_size=4, seq_len=32,
        synthetic=True, inner_opt="adam", inner_lr=3e-4,
    )
    summary = run_training(cfg, str(tmp_path / "dp"))
    assert summary["final_train_loss"] is not None
    assert summary["total_comm_bytes"] > 0           # DP all-reduces parameters
    assert summary["peak_mem_mb"] > 0


def test_dp_and_mp_same_model_both_run(tmp_path):
    """Both paradigms run the SAME --model gpt2_untied to completion: this is the
    turnkey comparison (mp_mid.env / dp_mid.env) exercised single-machine."""
    from macluster.train import TrainConfig, run_training

    common = dict(task="wikitext", model="gpt2_untied", world_size=2,
                  batch_size=4, seq_len=32, synthetic=True,
                  inner_opt="adam", inner_lr=3e-4)
    dp = run_training(TrainConfig(parallelism="data", algorithm="diloco",
                                  max_steps=4, H=2, **common), str(tmp_path / "dp"))
    mp = run_training(TrainConfig(parallelism="pipeline", stage_mem_gb="48,24",
                                  n_micro=2, max_steps=2, **common), str(tmp_path / "mp"))
    assert dp["final_train_loss"] is not None and mp["final_train_loss"] is not None
    # MP reports the split; its stage sum is the shared architecture's size
    assert mp["model_param_count"] == sum(mp["stage_param_counts"])
    # both logged the comm + memory the comparison plots need
    for s in (dp, mp):
        assert s["total_comm_bytes"] >= 0 and s["peak_mem_mb"] > 0
    # metrics files exist for the post-run comparison
    for d in ("dp", "mp"):
        assert os.path.exists(tmp_path / d / "summary.json")
        recs = [json.loads(line) for line in open(tmp_path / d / "metrics.jsonl")]
        assert recs and all("peak_mem_mb" in r for r in recs)
