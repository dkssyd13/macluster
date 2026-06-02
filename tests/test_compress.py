"""Unit tests for algorithms/compress.py (top-k sparsification, byte accounting)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from macluster.algorithms.compress import sparse_bytes, topk_sparsify


def _dense(sparse: mx.array) -> np.ndarray:
    return np.asarray(sparse, dtype=np.float32).reshape(-1)


def test_topk_keeps_round_k_when_k_lt_n():
    # n = 10, k_frac = 0.3 -> k = round(3.0) = 3
    leaf = mx.array(
        [0.1, -9.0, 0.2, 8.0, 0.05, -7.0, 0.3, 0.01, 0.02, 0.04]
    )
    sparse, k, n = topk_sparsify(leaf, 0.3)
    assert n == 10
    assert k == 3
    arr = _dense(sparse)
    assert int(np.count_nonzero(arr)) == 3


def test_topk_round_to_nearest():
    # n = 10, k_frac = 0.24 -> round(2.4) = 2
    leaf = mx.array(np.arange(1, 11, dtype=np.float32))
    _, k, n = topk_sparsify(leaf, 0.24)
    assert (k, n) == (2, 10)
    # k_frac = 0.26 -> round(2.6) = 3
    _, k2, _ = topk_sparsify(leaf, 0.26)
    assert k2 == 3


def test_topk_preserves_largest_magnitude():
    # Largest |.| are 9, 8, 7 at their original (signed) values.
    leaf = mx.array(
        [0.1, -9.0, 0.2, 8.0, 0.05, -7.0, 0.3, 0.01, 0.02, 0.04]
    )
    sparse, _, _ = topk_sparsify(leaf, 0.3)
    arr = _dense(sparse)
    kept = arr[arr != 0.0]
    assert sorted(kept.tolist()) == sorted([-9.0, 8.0, -7.0])


def test_topk_zeros_everything_else():
    leaf = mx.array([5.0, 0.1, -6.0, 0.2, 0.3])
    sparse, k, n = topk_sparsify(leaf, 0.4)  # round(2.0) = 2
    arr = _dense(sparse)
    assert k == 2 and n == 5
    # the two largest-|.| (5, -6) survive; rest are exactly zero
    nonzero_positions = np.nonzero(arr)[0].tolist()
    assert nonzero_positions == [0, 2]
    assert arr[1] == 0.0 and arr[3] == 0.0 and arr[4] == 0.0


def test_topk_preserves_shape():
    leaf = mx.array(np.arange(12, dtype=np.float32).reshape(3, 4))
    sparse, _, n = topk_sparsify(leaf, 0.25)
    assert tuple(sparse.shape) == (3, 4)
    assert n == 12


def test_topk_full_when_k_ge_n_returns_dense():
    leaf = mx.array([1.0, -2.0, 3.0])
    sparse, k, n = topk_sparsify(leaf, 1.0)  # k = n
    assert (k, n) == (3, 3)
    arr = _dense(sparse)
    np.testing.assert_allclose(arr, [1.0, -2.0, 3.0])


def test_topk_min_one_kept():
    # tiny k_frac still keeps at least one entry (max(1, ...))
    leaf = mx.array(np.arange(100, dtype=np.float32))
    _, k, n = topk_sparsify(leaf, 1e-6)
    assert k == 1 and n == 100


def test_sparse_bytes_is_8k():
    assert sparse_bytes(0) == 0
    assert sparse_bytes(1) == 8
    assert sparse_bytes(7) == 56
    assert sparse_bytes(1000) == 8000


def test_sparse_bytes_matches_topk_k():
    leaf = mx.array(np.random.default_rng(0).standard_normal(50).astype(np.float32))
    _, k, n = topk_sparsify(leaf, 0.1)  # round(5.0) = 5
    assert k == 5 and n == 50
    assert sparse_bytes(k) == 8 * k


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
