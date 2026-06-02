"""DiLoCo (Douillard et al., 2023): H local steps, then outer Nesterov SGD on
the averaged pseudo-gradient (axis 2).

Each round every replica trains H inner steps from the shared global params.
The pseudo-gradient ``global - replica`` is averaged across replicas and fed to
an outer SGD-with-Nesterov-momentum optimizer that updates the global params.
Communication happens once per H steps and transmits one full model-sized
tensor, so it is ~H x cheaper than dense.
"""

from __future__ import annotations

import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from .base import (
    Algorithm,
    SyncStats,
    tree_clone,
    tree_eval,
    tree_mean,
    tree_nbytes,
    tree_sub,
)


class DiLoCo(Algorithm):
    name = "diloco"

    def __init__(self, H: int = 50, outer_lr: float = 0.7, outer_momentum: float = 0.9):
        self._H = H
        self._outer = optim.SGD(learning_rate=outer_lr, momentum=outer_momentum, nesterov=True)
        self._global: dict | None = None

    def init_global(self, params: dict) -> None:
        self._global = tree_clone(params)
        self._outer.init(self._global)

    def global_params(self) -> dict:
        return self._global

    def local_steps(self) -> int:
        return self._H

    def sync(self, replica_params: list[dict]) -> SyncStats:
        pseudo = [tree_sub(self._global, rp) for rp in replica_params]
        avg = tree_mean(pseudo)
        # Treat the averaged pseudo-gradient as a gradient for the outer optimizer.
        self._global = self._outer.apply_gradients(avg, self._global)
        tree_eval(self._global)
        return SyncStats(bytes_per_worker=tree_nbytes(avg), topology="ring")

    def sync_collective(self, local_params: dict, cluster) -> SyncStats:
        # Averaged pseudo-gradient = all_sum(global - local) / W, fed to the SAME
        # outer optimizer. Every rank applies the identical averaged pseudo-grad,
        # so the outer-optimizer state stays identical across ranks (rank-0 holds
        # the true global). At W=1 this reduces to sync([local]) exactly.
        W = cluster.world_size
        gflat = dict(tree_flatten(self._global))
        avg = {n: cluster.all_sum(gflat[n] - lv) / W for n, lv in tree_flatten(local_params)}
        avg_tree = tree_unflatten(list(avg.items()))
        self._global = self._outer.apply_gradients(avg_tree, self._global)
        tree_eval(self._global)
        return SyncStats(bytes_per_worker=tree_nbytes(avg_tree), topology="ring")

    def knobs(self) -> dict:
        return {"H": self._H}
