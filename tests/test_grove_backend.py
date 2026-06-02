"""Tests for the Phase 7 grove backend (backends/grove_backend.py) and the
``sync_collective`` algorithm paths.

The correctness gate is **numerical equivalence at world_size==1**: grove's
collectives are identity no-ops there, so each algorithm's ``sync_collective``
must reduce to the single-machine ``sync([local])`` exactly. We also check the
averaging math with a fake 2-rank cluster, and smoke-run ``run_training`` on the
grove backend at W=1 (no second machine needed).
"""

from __future__ import annotations

import json
import os

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

pytest.importorskip("grove")  # skip the whole module if grove is unavailable

from macluster.algorithms.dense import Dense
from macluster.algorithms.diloco import DiLoCo
from macluster.algorithms.sparseloco import SparseLoCo
from macluster.backends.grove_backend import GroveCluster
from macluster.data.cifar import make_cifar_task
from macluster.train import TrainConfig, build_inner_opt, run_training


# --------------------------------------------------------------------------- #
# helpers + fixtures
# --------------------------------------------------------------------------- #
def _trees_equal(a: dict, b: dict) -> bool:
    fa, fb = dict(tree_flatten(a)), dict(tree_flatten(b))
    if set(fa) != set(fb):
        return False
    return all(bool(mx.allclose(fa[n], fb[n]).item()) for n in fa)


INIT = {
    "l1": {"w": mx.array([1.0, 2.0, 3.0, 4.0]), "b": mx.array([0.5, -0.5])},
    "l2": {"w": mx.array([0.0, 1.0, -1.0, 2.0])},
}
# A short sequence of post-inner-step "local" trees, fed to successive syncs so
# error-feedback / outer-optimizer state carry-over is exercised.
LOCALS = [
    {"l1": {"w": mx.array([1.1, 1.9, 3.2, 3.8]), "b": mx.array([0.4, -0.6])},
     "l2": {"w": mx.array([0.1, 0.9, -1.2, 2.1])}},
    {"l1": {"w": mx.array([0.9, 2.1, 2.8, 4.2]), "b": mx.array([0.6, -0.4])},
     "l2": {"w": mx.array([-0.1, 1.1, -0.9, 1.9])}},
]


class FakeCluster:
    """Stand-in for GroveCluster: identity all_sum (W=1) or x*world (identical ranks)."""

    def __init__(self, world_size: int = 1, identical_ranks: bool = False):
        self.world_size = world_size
        self.rank = 0
        self._identical = identical_ranks

    def all_sum(self, x: mx.array) -> mx.array:
        return x * self.world_size if self._identical else x

    def barrier(self) -> None:  # pragma: no cover - trivial
        pass


def _drive_sim(make_algo):
    a = make_algo()
    a.init_global(INIT)
    for local in LOCALS:
        a.sync([local])
    return a.global_params()


def _drive_collective(make_algo, cluster):
    a = make_algo()
    a.init_global(INIT)
    for local in LOCALS:
        a.sync_collective(local, cluster)
    return a.global_params()


# --------------------------------------------------------------------------- #
# W=1 numerical equivalence: sync_collective == sync([local])
# --------------------------------------------------------------------------- #
def test_dense_sync_collective_matches_sim_w1():
    fake = FakeCluster(world_size=1)
    assert _trees_equal(_drive_sim(lambda: Dense(H=1)),
                        _drive_collective(lambda: Dense(H=1), fake))


def test_diloco_sync_collective_matches_sim_w1():
    fake = FakeCluster(world_size=1)
    mk = lambda: DiLoCo(H=20, outer_lr=0.7)
    assert _trees_equal(_drive_sim(mk), _drive_collective(mk, fake))


def test_sparseloco_sync_collective_matches_sim_w1():
    fake = FakeCluster(world_size=1)
    mk = lambda: SparseLoCo(H=30, outer_lr=1.0, k_frac=0.5)
    assert _trees_equal(_drive_sim(mk), _drive_collective(mk, fake))


def test_sparseloco_collective_bytes_match_sim_w1():
    # Same compact (idx, val) byte accounting on both paths at W=1.
    a1 = SparseLoCo(H=30, k_frac=0.5); a1.init_global(INIT)
    a2 = SparseLoCo(H=30, k_frac=0.5); a2.init_global(INIT)
    s1 = a1.sync([LOCALS[0]])
    s2 = a2.sync_collective(LOCALS[0], FakeCluster(world_size=1))
    assert s1.bytes_per_worker == s2.bytes_per_worker
    assert s1.topology == s2.topology == "gather"


# --------------------------------------------------------------------------- #
# averaging math with a fake 2-rank cluster
# --------------------------------------------------------------------------- #
def test_dense_collective_averages_over_world():
    # Two identical ranks each holding `local`: all_sum = 2*local, /W -> local.
    a = Dense(H=1)
    a.init_global(INIT)
    a.sync_collective(LOCALS[0], FakeCluster(world_size=2, identical_ranks=True))
    assert _trees_equal(a.global_params(), LOCALS[0])


def test_diloco_collective_averages_over_world():
    # Identical ranks -> averaged pseudo-grad equals a single rank's pseudo-grad,
    # so the W=2-identical global matches the W=1 global.
    fake2 = FakeCluster(world_size=2, identical_ranks=True)
    fake1 = FakeCluster(world_size=1)
    mk = lambda: DiLoCo(H=20, outer_lr=0.7)
    assert _trees_equal(_drive_collective(mk, fake2), _drive_collective(mk, fake1))


# --------------------------------------------------------------------------- #
# GroveCluster (real grove, world_size==1 in-process)
# --------------------------------------------------------------------------- #
def _grove_cluster(world_size: int = 1):
    task = make_cifar_task(world_size, batch_size=16, synthetic=True, seed=0)
    cfg = TrainConfig(backend="grove", world_size=world_size, model="resnet20", batch_size=16)
    return GroveCluster(task, "resnet20", build_inner_opt(cfg))


def test_grove_all_sum_identity_w1():
    cluster = _grove_cluster()
    x = mx.array([1.0, 2.0, 3.0])
    assert bool(mx.allclose(cluster.all_sum(x), x).item())
    assert cluster.world_size == 1 and cluster.rank == 0


def test_grove_cluster_builds_single_model_and_steps():
    cluster = _grove_cluster()
    assert isinstance(cluster.params(), dict)
    loss = cluster.inner_step()
    assert isinstance(loss, float)
    assert loss == loss and loss < float("inf")  # finite, not NaN


def test_grove_cluster_unknown_model_raises():
    task = make_cifar_task(1, batch_size=16, synthetic=True, seed=0)
    cfg = TrainConfig(backend="grove", world_size=1, model="resnet20", batch_size=16)
    with pytest.raises(KeyError):
        GroveCluster(task, "not_a_model", build_inner_opt(cfg))


# --------------------------------------------------------------------------- #
# run_training on the grove backend (W=1 smoke; no second machine)
# --------------------------------------------------------------------------- #
def test_run_training_grove_w1_smoke(tmp_path):
    cfg = TrainConfig(
        backend="grove", world_size=1, synthetic=True, rounds=2,
        model="resnet20", batch_size=16, eval_every=1, algorithm="dense",
    )
    run_dir = os.path.join(str(tmp_path), "grove0")
    summary = run_training(cfg, run_dir)

    assert summary["n_records"] == 2
    assert summary["total_comm_bytes"] > 0
    assert summary["sim_time_s"] > 0  # real accumulated wall-clock

    with open(os.path.join(run_dir, "metrics.jsonl")) as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 2
    for rec in records:
        assert rec["comm_bytes"] > 0
        assert "sim_time_s" in rec
        assert "sync_s_real" in rec and "compute_s_real" in rec  # grove-only keys
        assert "accuracy" in rec  # rank 0 evaluates every round
    assert records[-1]["sim_time_s"] > 0


def test_run_training_grove_w1_adaptive_smoke(tmp_path):
    # Exercises the adaptive controller's observe() with REAL measured timings.
    cfg = TrainConfig(
        backend="grove", world_size=1, synthetic=True, rounds=3,
        model="resnet20", batch_size=16, eval_every=1, algorithm="adaptive",
    )
    run_dir = os.path.join(str(tmp_path), "grove_adapt")
    summary = run_training(cfg, run_dir)
    assert summary["n_records"] == 3
    with open(os.path.join(run_dir, "metrics.jsonl")) as f:
        records = [json.loads(line) for line in f if line.strip()]
    for rec in records:
        assert "H" in rec and "k_frac" in rec  # adaptive knobs logged


def test_grove_slug_has_backend_suffix():
    assert TrainConfig(backend="grove").slug().endswith("-grove")
    assert not TrainConfig(backend="sim").slug().endswith("-sim")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
