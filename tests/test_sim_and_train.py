"""Integration tests for backends/sim.py (SimCluster) and train.py (run_training).

These use the synthetic CIFAR task (no download) and a tiny resnet20 so the
2-round run is fast. We check that replicas start identical and that a full
run produces the expected summary keys and a metrics.jsonl with one record
per round.
"""

from __future__ import annotations

import json
import os

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from macluster.backends.sim import SimCluster
from macluster.data.cifar import make_cifar_task
from macluster.train import TrainConfig, build_inner_opt, run_training


def _trees_equal(a: dict, b: dict) -> bool:
    fa = dict(tree_flatten(a))
    fb = dict(tree_flatten(b))
    if set(fa) != set(fb):
        return False
    for name in fa:
        if not bool(mx.allclose(fa[name], fb[name]).item()):
            return False
    return True


def _make_cluster(world_size: int = 2):
    task = make_cifar_task(world_size, batch_size=16, synthetic=True, seed=0)
    cfg = TrainConfig(synthetic=True, world_size=world_size, model="resnet20", batch_size=16)
    return SimCluster(task, "resnet20", build_inner_opt(cfg)), task


def test_simcluster_builds_world_size_replicas():
    cluster, _ = _make_cluster(world_size=2)
    assert cluster.world_size == 2
    assert len(cluster.replicas) == 2


def test_simcluster_replicas_start_identical():
    cluster, _ = _make_cluster(world_size=2)
    p0 = cluster.replicas[0].params()
    p1 = cluster.replicas[1].params()
    assert _trees_equal(p0, p1), "replicas must start from identical params"


def test_simcluster_load_global_overwrites_all():
    cluster, _ = _make_cluster(world_size=2)
    # take replica-0 params, zero them, broadcast, and confirm both replicas match
    target = cluster.replicas[0].params()
    zeroed = {n: mx.zeros(v.shape) for n, v in tree_flatten(target)}
    from mlx.utils import tree_unflatten

    zeroed_tree = tree_unflatten(list(zeroed.items()))
    cluster.load_global(zeroed_tree)
    for rep in cluster.replicas:
        for _, leaf in tree_flatten(rep.params()):
            assert bool(mx.all(leaf == 0).item())


def test_simcluster_unknown_model_raises():
    task = make_cifar_task(2, batch_size=16, synthetic=True, seed=0)
    cfg = TrainConfig(synthetic=True, world_size=2, model="resnet20", batch_size=16)
    with pytest.raises(KeyError):
        SimCluster(task, "not_a_model", build_inner_opt(cfg))


def test_inner_step_returns_finite_loss():
    cluster, _ = _make_cluster(world_size=2)
    loss = cluster.replicas[0].inner_step()
    assert isinstance(loss, float)
    assert loss == loss  # not NaN
    assert loss < float("inf")


def test_run_training_summary_and_metrics(tmp_path):
    cfg = TrainConfig(
        synthetic=True,
        rounds=2,
        world_size=2,
        model="resnet20",
        batch_size=16,
        eval_every=1,
    )
    run_dir = os.path.join(str(tmp_path), "run0")
    summary = run_training(cfg, run_dir)

    # ---- summary shape ----
    for key in (
        "run_dir",
        "n_records",
        "config",
        "final_train_loss",
        "final_accuracy",
        "total_comm_bytes",
        "total_comm_MB",
        "sim_time_s",
        "time_to_target_s",
        "target_metric",
    ):
        assert key in summary, f"missing summary key {key!r}"

    assert summary["n_records"] == 2
    assert isinstance(summary["final_train_loss"], float)
    assert summary["total_comm_bytes"] > 0

    # ---- run dir artifacts ----
    assert os.path.exists(os.path.join(run_dir, "config.json"))
    assert os.path.exists(os.path.join(run_dir, "summary.json"))
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    assert os.path.exists(metrics_path)

    with open(metrics_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 2

    # eval_every=1 -> every round carries the eval metrics
    for i, rec in enumerate(records):
        assert rec["round"] == i
        assert "train_loss" in rec
        assert "comm_bytes" in rec and rec["comm_bytes"] > 0
        assert "accuracy" in rec
        assert "val_loss" in rec


def test_run_training_rounds_count(tmp_path):
    cfg = TrainConfig(
        synthetic=True,
        rounds=3,
        world_size=2,
        model="resnet20",
        batch_size=16,
        eval_every=5,  # only round 0 and final (round 2) get eval
    )
    run_dir = os.path.join(str(tmp_path), "run1")
    summary = run_training(cfg, run_dir)
    assert summary["n_records"] == 3
    with open(os.path.join(run_dir, "metrics.jsonl")) as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert [r["round"] for r in records] == [0, 1, 2]
    # final round always evaluated
    assert "accuracy" in records[-1]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
