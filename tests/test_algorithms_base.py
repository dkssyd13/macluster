"""Unit tests for algorithms/base.py parameter-tree helpers.

Covers numeric correctness of tree_sub / tree_mean on small mlx trees and the
byte accounting of tree_nbytes (used by the dense link cost model).
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from macluster.algorithms.base import tree_mean, tree_nbytes, tree_sub


def _close(a: mx.array, b) -> bool:
    return bool(mx.allclose(a, mx.array(b)).item())


def test_tree_sub_elementwise():
    a = {"w": mx.array([3.0, 5.0]), "b": mx.array([[1.0, 2.0], [3.0, 4.0]])}
    b = {"w": mx.array([1.0, 2.0]), "b": mx.array([[0.5, 1.0], [1.5, 2.0]])}
    out = tree_sub(a, b)
    assert _close(out["w"], [2.0, 3.0])
    assert _close(out["b"], [[0.5, 1.0], [1.5, 2.0]])


def test_tree_sub_does_not_mutate_inputs():
    a = {"w": mx.array([3.0, 5.0])}
    b = {"w": mx.array([1.0, 2.0])}
    _ = tree_sub(a, b)
    # originals untouched
    assert _close(a["w"], [3.0, 5.0])
    assert _close(b["w"], [1.0, 2.0])


def test_tree_mean_two_trees():
    t1 = {"w": mx.array([2.0, 4.0]), "b": mx.array([10.0])}
    t2 = {"w": mx.array([4.0, 8.0]), "b": mx.array([20.0])}
    out = tree_mean([t1, t2])
    assert _close(out["w"], [3.0, 6.0])
    assert _close(out["b"], [15.0])


def test_tree_mean_three_trees():
    trees = [
        {"x": mx.array([1.0, 1.0])},
        {"x": mx.array([2.0, 5.0])},
        {"x": mx.array([3.0, 6.0])},
    ]
    out = tree_mean(trees)
    # means: (1+2+3)/3 = 2 ; (1+5+6)/3 = 4
    assert _close(out["x"], [2.0, 4.0])


def test_tree_mean_singleton_is_clone():
    t = {"x": mx.array([7.0, 9.0])}
    out = tree_mean([t])
    assert _close(out["x"], [7.0, 9.0])
    # must be a clone, not the same array aliased back
    assert out["x"] is not t["x"]


def test_tree_nbytes_single_leaf_float32():
    # 12 float32 entries * 4 bytes = 48
    t = {"w": mx.zeros((3, 4), dtype=mx.float32)}
    assert tree_nbytes(t) == 48


def test_tree_nbytes_multi_leaf():
    # (6 + 4) float32 entries * 4 bytes = 40
    t = {"w": mx.ones((2, 3)), "b": mx.ones((4,))}
    assert tree_nbytes(t) == 40


def test_tree_nbytes_respects_dtype_size():
    f32 = {"w": mx.zeros((10,), dtype=mx.float32)}
    f16 = {"w": mx.zeros((10,), dtype=mx.float16)}
    assert tree_nbytes(f32) == 10 * 4
    assert tree_nbytes(f16) == 10 * 2


def test_tree_nbytes_nested_tree():
    t = {"layer": {"w": mx.zeros((5,), dtype=mx.float32), "b": mx.zeros((5,), dtype=mx.float32)}}
    assert tree_nbytes(t) == (5 + 5) * 4


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
