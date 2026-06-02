"""Dense baseline: synchronize every step by averaging parameters (axis 2).

With H=1 and plain SGD inner steps this is mathematically synchronous SGD
(all-reduced gradients). It transmits the full model every step, so it is the
communication-heavy reference the low-communication methods are measured
against.
"""

from __future__ import annotations

from mlx.utils import tree_flatten, tree_unflatten

from .base import Algorithm, SyncStats, tree_clone, tree_eval, tree_mean, tree_nbytes


class Dense(Algorithm):
    name = "dense"

    def __init__(self, H: int = 1):
        self._H = H
        self._global: dict | None = None

    def init_global(self, params: dict) -> None:
        self._global = tree_clone(params)

    def global_params(self) -> dict:
        return self._global

    def local_steps(self) -> int:
        return self._H

    def sync(self, replica_params: list[dict]) -> SyncStats:
        self._global = tree_mean(replica_params)
        tree_eval(self._global)
        return SyncStats(bytes_per_worker=tree_nbytes(self._global), topology="ring")

    def sync_collective(self, local_params: dict, cluster) -> SyncStats:
        # Averaged params = all_sum(local) / W (grove all_sum is a SUM, not mean).
        # At W=1 all_sum is the identity, so this matches sync([local]) exactly.
        W = cluster.world_size
        new = {n: cluster.all_sum(v) / W for n, v in tree_flatten(local_params)}
        self._global = tree_unflatten(list(new.items()))
        tree_eval(self._global)
        return SyncStats(bytes_per_worker=tree_nbytes(self._global), topology="ring")

    def knobs(self) -> dict:
        return {"H": self._H}
