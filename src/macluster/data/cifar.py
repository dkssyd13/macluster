"""CIFAR-10 task builder (real download + a synthetic fallback for smoke tests).

Each worker gets an infinite, independently-shuffled iterator over a disjoint
shard of the training set, so replicas compute genuinely different
pseudo-gradients (otherwise distributed averaging would be a no-op).
"""

from __future__ import annotations

import os
import pickle
import tarfile
import urllib.request

import mlx.core as mx
import numpy as np

from ..models.resnet import resnet20, resnet56
from ..task import Task, classification_eval, classification_loss

_CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


def _infinite_batches(X: np.ndarray, y: np.ndarray, batch_size: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(X)
    while True:
        perm = rng.permutation(n)
        for i in range(0, n - batch_size + 1, batch_size):
            idx = perm[i : i + batch_size]
            yield mx.array(X[idx]), mx.array(y[idx])


def _eval_batches(X: np.ndarray, y: np.ndarray, batch_size: int, max_batches: int | None):
    out = []
    for i in range(0, len(X) - batch_size + 1, batch_size):
        out.append((mx.array(X[i : i + batch_size]), mx.array(y[i : i + batch_size])))
        if max_batches and len(out) >= max_batches:
            break
    return out


def _shard(X: np.ndarray, y: np.ndarray, world_size: int):
    n = len(X)
    per = n // world_size
    return [(X[r * per : (r + 1) * per], y[r * per : (r + 1) * per]) for r in range(world_size)]


def _synthetic(n_train: int, n_eval: int, seed: int):
    rng = np.random.default_rng(seed)
    Xtr = rng.standard_normal((n_train, 32, 32, 3)).astype(np.float32)
    ytr = rng.integers(0, 10, size=n_train).astype(np.int32)
    Xte = rng.standard_normal((n_eval, 32, 32, 3)).astype(np.float32)
    yte = rng.integers(0, 10, size=n_eval).astype(np.int32)
    return Xtr, ytr, Xte, yte


def _load_real(data_dir: str):
    cache = os.path.join(data_dir, "cifar")
    os.makedirs(cache, exist_ok=True)
    base = os.path.join(cache, "cifar-10-batches-py")
    if not os.path.isdir(base):
        tgz = os.path.join(cache, "cifar-10-python.tar.gz")
        if not os.path.exists(tgz):
            urllib.request.urlretrieve(_CIFAR_URL, tgz)
        with tarfile.open(tgz) as t:
            t.extractall(cache)

    def _read(files):
        Xs, ys = [], []
        for f in files:
            with open(os.path.join(base, f), "rb") as fh:
                d = pickle.load(fh, encoding="bytes")
            Xs.append(d[b"data"])
            ys.append(np.array(d[b"labels"]))
        X = np.concatenate(Xs).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # -> NHWC
        X = (X.astype(np.float32) / 255.0 - _MEAN) / _STD
        return X.astype(np.float32), np.concatenate(ys).astype(np.int32)

    Xtr, ytr = _read([f"data_batch_{i}" for i in range(1, 6)])
    Xte, yte = _read(["test_batch"])
    return Xtr, ytr, Xte, yte


def make_cifar_task(
    world_size: int,
    batch_size: int = 128,
    synthetic: bool = False,
    n_synth_train: int = 4096,
    n_synth_eval: int = 1024,
    eval_max_batches: int | None = 16,
    data_dir: str = "data/cache",
    seed: int = 0,
) -> Task:
    if synthetic:
        Xtr, ytr, Xte, yte = _synthetic(n_synth_train, n_synth_eval, seed)
    else:
        Xtr, ytr, Xte, yte = _load_real(data_dir)

    shards = _shard(Xtr, ytr, world_size)
    train_shards = [
        _infinite_batches(sx, sy, batch_size, seed + 1 + r) for r, (sx, sy) in enumerate(shards)
    ]
    eval_batches = _eval_batches(Xte, yte, batch_size, eval_max_batches)

    return Task(
        name="cifar10",
        kind="image",
        loss_fn=classification_loss,
        eval_fn=classification_eval,
        metric="accuracy",
        metric_goal="max",
        train_shards=train_shards,
        eval_batches=eval_batches,
        model_fns={"resnet20": resnet20, "resnet56": resnet56},
        meta={"synthetic": synthetic, "batch_size": batch_size, "n_classes": 10},
    )
