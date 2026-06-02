"""Top-k sparsification with error feedback, the core of SparseLoCo (axis 2).

We keep the largest-magnitude ``k_frac`` fraction of each tensor's entries and
zero the rest. The transmitted payload is ``(index, value)`` pairs, so the
communication cost is ``8 * k`` bytes per tensor (int32 index + float32 value),
versus ``4 * n`` for a dense send. Compression is done host-side (numpy) for a
robust scatter; at course scale (<=1M params, infrequent syncs) this is cheap.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def topk_sparsify(leaf: mx.array, k_frac: float) -> tuple[mx.array, int, int]:
    """Return (sparse_tensor, k_kept, n_total) keeping the top ``k_frac`` by |x|.

    ``sparse_tensor`` has the same shape as ``leaf`` with all but the kept
    entries zeroed (this doubles as the decompressed update). ``k_kept`` counts
    transmitted entries for byte accounting.
    """
    e = np.asarray(leaf, dtype=np.float32).reshape(-1)
    n = int(e.size)
    k = max(1, int(round(k_frac * n)))
    if k >= n:
        return mx.array(e.reshape(leaf.shape)), n, n
    idx = np.argpartition(np.abs(e), n - k)[n - k :]
    sparse = np.zeros(n, dtype=np.float32)
    sparse[idx] = e[idx]
    return mx.array(sparse.reshape(leaf.shape)), k, n


def sparse_bytes(k_kept: int) -> int:
    """Bytes to transmit ``k_kept`` (index, value) pairs: int32 + float32."""
    return k_kept * 8
