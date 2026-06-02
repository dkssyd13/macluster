"""A Task bundles everything the training driver needs that is *not* the
synchronization algorithm or the link emulation: the data shards (one infinite
(X, y) iterator per worker), a loss, an eval function, and the candidate model
constructors. Tasks are built in ``data/`` (e.g. ``data/cifar.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import mlx.core as mx
import mlx.nn as nn


@dataclass
class Task:
    name: str
    kind: str                                  # "image" | "text"
    loss_fn: Callable[[nn.Module, mx.array, mx.array], mx.array]
    eval_fn: Callable[[nn.Module, list], dict]
    metric: str                                # primary metric key, e.g. "accuracy"
    metric_goal: str                           # "max" | "min"
    train_shards: list[Iterator]               # one infinite (X, y) iterator per worker
    eval_batches: list[tuple]                  # held-out (X, y) batches
    model_fns: dict[str, Callable[[], nn.Module]]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def world_size(self) -> int:
        return len(self.train_shards)


def classification_loss(model: nn.Module, X: mx.array, y: mx.array) -> mx.array:
    logits = model(X)
    return nn.losses.cross_entropy(logits, y, reduction="mean")


def classification_eval(model: nn.Module, batches: list) -> dict:
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    for X, y in batches:
        logits = model(X)
        loss_sum += float(nn.losses.cross_entropy(logits, y, reduction="sum"))
        correct += int(mx.sum(mx.argmax(logits, axis=1) == y))
        total += int(X.shape[0])
    model.train()
    return {"val_loss": loss_sum / max(total, 1), "accuracy": correct / max(total, 1)}
