"""SimCluster: W model replicas trained in one process (single-machine emulation).

This backend implements the distributed-training *semantics* (disjoint shards,
synchronous averaging) without a second machine. It measures real per-replica
compute time and models parallel execution by charging the round the *max*
replica time (workers compute simultaneously on real hardware), while the link
emulator (emulation/link.py) charges communication time. Real wall-clock
speed-up numbers come later from the 2-MacBook GroveBackend (Phase 7).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ..algorithms.base import tree_clone
from ..task import Task


class Replica:
    def __init__(self, model: nn.Module, optimizer, data_iter, loss_fn):
        self.model = model
        self.opt = optimizer
        self.data = data_iter
        self._lvg = nn.value_and_grad(model, loss_fn)

    def set_params(self, params: dict) -> None:
        self.model.update(tree_clone(params))
        mx.eval(self.model.parameters())

    def params(self) -> dict:
        return self.model.trainable_parameters()

    def inner_step(self) -> float:
        X, y = next(self.data)
        loss, grads = self._lvg(self.model, X, y)
        self.opt.update(self.model, grads)
        mx.eval(self.model.parameters(), self.opt.state, loss)
        return float(loss)


class SimCluster:
    def __init__(self, task: Task, model_name: str, inner_opt_fn):
        if model_name not in task.model_fns:
            raise KeyError(f"model {model_name!r} not in task {task.name!r}: {list(task.model_fns)}")
        self.task = task
        self.replicas: list[Replica] = []
        for r in range(task.world_size):
            model = task.model_fns[model_name]()
            model.train()
            self.replicas.append(Replica(model, inner_opt_fn(), task.train_shards[r], task.loss_fn))
        # Start every replica from identical parameters.
        init = self.replicas[0].params()
        for rep in self.replicas[1:]:
            rep.set_params(init)

    @property
    def world_size(self) -> int:
        return len(self.replicas)

    def load_global(self, params: dict) -> None:
        for rep in self.replicas:
            rep.set_params(params)

    def initial_params(self) -> dict:
        """Params to seed the algorithm's global state (replica 0's)."""
        return self.replicas[0].params()

    def collect_params(self) -> list[dict]:
        return [rep.params() for rep in self.replicas]

    def eval_model(self, params: dict) -> nn.Module:
        """Load params into replica 0 and return its model for evaluation."""
        self.replicas[0].set_params(params)
        return self.replicas[0].model
