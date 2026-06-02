"""SparseLoCo (Sarfi et al., 2025): DiLoCo + top-k error-feedback compression
(axis 2).

Like DiLoCo it accumulates a pseudo-gradient over H local steps, but before
communicating it sparsifies each tensor to its top ``k_frac`` entries while
carrying the dropped mass in a per-replica error-feedback buffer. Only
``(index, value)`` pairs are transmitted, cutting communication by ~``1/(2*
k_frac)`` versus DiLoCo's full pseudo-gradient.
"""

from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from .base import Algorithm, SyncStats, tree_clone, tree_eval, tree_mean
from .compress import sparse_bytes, topk_sparsify


class SparseLoCo(Algorithm):
    name = "sparseloco"

    def __init__(
        self,
        H: int = 30,
        outer_lr: float = 1.0,
        k_frac: float = 0.02,
        error_decay: float = 0.95,
    ):
        self._H = H
        self._outer_lr = outer_lr
        self._k_frac = k_frac
        self._decay = error_decay
        self._global: dict | None = None
        self._names: list[str] = []
        self._ef: list[dict[str, mx.array]] | None = None  # per-replica error buffers (sim)
        self._ef_single: dict[str, mx.array] | None = None  # this rank's buffer (grove)

    def init_global(self, params: dict) -> None:
        self._global = tree_clone(params)
        self._names = [n for n, _ in tree_flatten(params)]

    def global_params(self) -> dict:
        return self._global

    def local_steps(self) -> int:
        return self._H

    def _ensure_ef(self, world_size: int) -> None:
        if self._ef is None:
            flat = dict(tree_flatten(self._global))
            self._ef = [
                {n: mx.zeros(flat[n].shape) for n in self._names} for _ in range(world_size)
            ]

    def sync(self, replica_params: list[dict]) -> SyncStats:
        self._ensure_ef(len(replica_params))
        gflat = dict(tree_flatten(self._global))
        bytes_per_worker = 0  # symmetric workers -> count one
        decompressed: list[dict] = []

        for r, rp in enumerate(replica_params):
            rflat = dict(tree_flatten(rp))
            sparse_leaves = {}
            for n in self._names:
                pseudo = gflat[n] - rflat[n]
                ef = self._ef[r][n] * self._decay + pseudo
                sparse, k_kept, _ = topk_sparsify(ef, self._k_frac)
                self._ef[r][n] = ef - sparse  # error feedback: keep the residual
                sparse_leaves[n] = sparse
                if r == 0:
                    bytes_per_worker += sparse_bytes(k_kept)
            decompressed.append(tree_unflatten(list(sparse_leaves.items())))

        avg_update = tree_mean(decompressed)
        aflat = dict(tree_flatten(avg_update))
        new_global = {n: gflat[n] - self._outer_lr * aflat[n] for n in self._names}
        self._global = tree_unflatten(list(new_global.items()))
        tree_eval(self._global, *self._ef)
        return SyncStats(bytes_per_worker=float(bytes_per_worker), topology="gather")

    def _ensure_ef_single(self) -> None:
        if self._ef_single is None:
            flat = dict(tree_flatten(self._global))
            self._ef_single = {n: mx.zeros(flat[n].shape) for n in self._names}

    def sync_collective(self, local_params: dict, cluster) -> SyncStats:
        # Each rank carries ONE local error-feedback buffer (the faithful
        # SparseLoCo worker view), sparsifies its own pseudo-gradient, then the
        # cluster averages the (decompressed) sparse updates via all_sum / W.
        # We measure bytes from the COMPACT (idx, val) payload (8*k) even though
        # all_sum moves the dense tensor on the wire -- so comm_bytes reflects the
        # sparse volume while the measured sync_s reflects a dense transfer.
        self._ensure_ef_single()
        W = cluster.world_size
        gflat = dict(tree_flatten(self._global))
        lflat = dict(tree_flatten(local_params))
        bytes_per_worker = 0
        new_global = {}
        for n in self._names:
            pseudo = gflat[n] - lflat[n]
            ef = self._ef_single[n] * self._decay + pseudo
            sparse, k_kept, _ = topk_sparsify(ef, self._k_frac)
            self._ef_single[n] = ef - sparse  # error feedback: keep the residual
            avg = cluster.all_sum(sparse) / W
            new_global[n] = gflat[n] - self._outer_lr * avg
            bytes_per_worker += sparse_bytes(k_kept)
        self._global = tree_unflatten(list(new_global.items()))
        tree_eval(self._global, self._ef_single)
        return SyncStats(bytes_per_worker=float(bytes_per_worker), topology="gather")

    def knobs(self) -> dict:
        return {"H": self._H, "k_frac": self._k_frac}
