"""Synchronization-algorithm contract + shared parameter-tree helpers.

In SimCluster (single-machine emulation), ``W`` model replicas train on
disjoint data shards. An :class:`Algorithm` owns the *global* synchronized
state and any per-replica auxiliary state (DiLoCo's outer optimizer,
SparseLoCo's error-feedback buffers). Each sync round:

  1. the driver sets every replica's params to the algorithm's global params,
  2. each replica runs ``local_steps()`` inner optimizer steps on its shard,
  3. the driver calls ``sync(replicas)`` -> :class:`SyncStats`, which mutates
     the replicas back to a single agreed global state and reports the bytes a
     worker transmitted (so the link emulator can charge communication time),
  4. the driver calls ``observe(...)`` so adaptive policies can react.

This synchronous-averaging view is mathematically identical to real DiLoCo /
SparseLoCo while giving us full control over H and top-k per round, which is
what axis-3 (the adaptive policy) needs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map


# --------------------------------------------------------------------------- #
# parameter-tree helpers
# --------------------------------------------------------------------------- #
def tree_clone(params: dict) -> dict:
    return tree_map(lambda x: mx.array(x), params)


def tree_sub(a: dict, b: dict) -> dict:
    return tree_map(lambda x, y: x - y, a, b)


def tree_add_scaled(a: dict, b: dict, scale: float) -> dict:
    return tree_map(lambda x, y: x + scale * y, a, b)


def tree_mean(trees: list[dict]) -> dict:
    if len(trees) == 1:
        return tree_clone(trees[0])
    acc = trees[0]
    for t in trees[1:]:
        acc = tree_map(lambda x, y: x + y, acc, t)
    inv = 1.0 / len(trees)
    return tree_map(lambda x: x * inv, acc)


def tree_nbytes(params: dict) -> int:
    """Total bytes of all leaf arrays in a parameter tree."""
    total = 0
    for _, leaf in tree_flatten(params):
        total += leaf.size * leaf.dtype.size
    return total


def tree_eval(*trees: dict) -> None:
    """Force evaluation of every leaf in the given parameter trees."""
    leaves = []
    for t in trees:
        leaves.extend(leaf for _, leaf in tree_flatten(t))
    if leaves:
        mx.eval(*leaves)


# --------------------------------------------------------------------------- #
# contract
# --------------------------------------------------------------------------- #
@dataclass
class SyncStats:
    """What one sync round cost, for metrics + link emulation."""

    bytes_per_worker: float           # payload a worker transmits (pre-topology)
    topology: str = "ring"            # "ring" (dense) | "gather" (sparse)
    extra: dict = field(default_factory=dict)


class Algorithm(ABC):
    name: str = "base"

    @abstractmethod
    def init_global(self, params: dict) -> None:
        """Seed the global synchronized state from an initial parameter tree."""

    @abstractmethod
    def global_params(self) -> dict:
        """Current global params; the driver loads these into every replica."""

    @abstractmethod
    def local_steps(self) -> int:
        """Inner steps to run this round (dynamic for adaptive policies)."""

    @abstractmethod
    def sync(self, replica_params: list[dict]) -> SyncStats:
        """Aggregate post-inner-step replica params into a new global state.

        This is the **single-machine** path: the driver hands over a list of all
        ``W`` replica trees (every replica lives in one process, SimCluster).
        """

    def sync_collective(self, local_params: dict, cluster) -> SyncStats:
        """Aggregate via real collectives on a multi-machine cluster (Phase 7).

        Unlike :meth:`sync`, on a real cluster each rank holds **only its own**
        post-inner-step tree (``local_params``); aggregation happens over
        ``cluster.all_sum`` (and ``cluster.world_size``). Mutates the global
        state to the new agreed value and reports the bytes a worker actually
        transmitted. Default: unsupported (override per algorithm).
        """
        raise NotImplementedError(f"{type(self).__name__} has no collective sync path")

    def observe(self, compute_s: float, sync_s: float, link_name: str) -> None:
        """Feedback hook for adaptive policies; no-op for fixed schedules."""

    def knobs(self) -> dict:
        """Current tunable knobs (H, topk, ...) for logging."""
        return {"H": self.local_steps()}
