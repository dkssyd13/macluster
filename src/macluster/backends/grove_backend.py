"""GroveCluster: ONE model replica per machine, over a real grove cluster.

This is the Phase 7 two-MacBook backend. Unlike :class:`SimCluster` (W replicas
in one process, *emulated* parallelism + *analytic* link cost), each MacBook
runs exactly ONE rank here: a single local model + inner optimizer bound to this
rank's data shard. Aggregation happens over real grove collectives
(``grove.all_sum``), and the round loop *measures* communication time around the
collective instead of charging it analytically -- the whole point of Phase 7.

At ``world_size == 1`` grove's collectives are identity no-ops, so this backend
also runs on a single machine: useful for smoke tests and for the
numerical-equivalence check against ``SimCluster`` at W=1.

The world (``grove.rank`` / ``grove.world_size``, module-level ints) is brought
up by the launcher (``scripts/grove_entry.py`` via ``grove start`` / ``grove
join``) BEFORE this class is constructed; we never call ``grove.init()`` here.
"""

from __future__ import annotations

import hashlib

import grove
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from ..algorithms.base import tree_clone
from ..task import Task


def assert_data_consensus(meta: dict) -> None:
    """Abort if the ranks built DIFFERENT data across the Macs.

    ``grove_entry._assert_config_consensus`` hashes the *config*
    (``task='wikitext'``, ``seed``, ...), which is identical on both Macs no
    matter which corpus actually loaded — so it CANNOT catch the silent case
    where one Mac's wikitext download fails and falls back to
    ``tinyshakespeare-bpe`` (same vocab=50257, so nothing crashes). Here we hash
    the *resolved* data fingerprint (corpus/source/tokenizer + token count +
    vocab) and ``all_sum`` it across ranks; a mismatch means the stages would
    train/score against different token streams and the logged
    loss/perplexity would be silently meaningless. Cheap collective, run once
    before the round loop. No-op at ``world_size == 1``."""
    if int(grove.world_size) <= 1:
        return
    fp = "|".join(
        str(meta.get(k)) for k in ("corpus", "bpe_source", "tokenizer", "n_tokens", "vocab_size")
    )
    local = int(hashlib.sha256(fp.encode()).hexdigest(), 16) % (2 ** 23)  # exact in fp32
    total = float(grove.all_sum(mx.array([float(local)]))[0])
    if abs(total - local * int(grove.world_size)) > 0.5:
        raise SystemExit(
            f"[grove] rank {grove.rank}: DATA disagrees across Macs "
            f"(fingerprint={fp!r}). Most likely one Mac fell back to "
            f"tinyshakespeare because the wikitext download failed. Pre-seed "
            f"data/cache/text/wikitext2_1.txt on EVERY Mac (same bytes) and retry."
        )


class GroveCluster:
    def __init__(self, task: Task, model_name: str, inner_opt_fn):
        if model_name not in task.model_fns:
            raise KeyError(f"model {model_name!r} not in task {task.name!r}: {list(task.model_fns)}")
        self.task = task
        self.rank = int(grove.rank)
        self._world_size = int(grove.world_size)
        if self.rank >= len(task.train_shards):
            raise IndexError(
                f"rank {self.rank} has no data shard: task built for "
                f"{len(task.train_shards)} shard(s) but world_size is {self._world_size} "
                f"(ensure --world-size matches the cluster size)"
            )
        self.model = task.model_fns[model_name]()
        self.model.train()
        self.opt = inner_opt_fn()
        self.data = task.train_shards[self.rank]
        self._lvg = nn.value_and_grad(self.model, task.loss_fn)
        # Every rank must start from byte-identical trainable params: DiLoCo
        # applies the shared averaged pseudo-gradient to each rank's base, so
        # divergent starts would never reconcile. Same-seed construction should
        # already match across Apple-Silicon Macs; this averaging guarantees it.
        self._broadcast_initial_params()

    @property
    def world_size(self) -> int:
        return self._world_size

    def _broadcast_initial_params(self) -> None:
        if self._world_size <= 1:
            return  # W=1: identical by construction; keep the path a no-op
        flat = tree_flatten(self.model.trainable_parameters())
        avg = {n: grove.all_sum(v) / self._world_size for n, v in flat}
        self.model.update(tree_unflatten(list(avg.items())))
        mx.eval(self.model.parameters())

    # --- collective helpers: algorithms aggregate via the cluster handle ------
    def all_sum(self, x: mx.array) -> mx.array:
        """Element-wise SUM across ranks (identity at world_size==1)."""
        return grove.all_sum(x)

    def barrier(self) -> None:
        grove.barrier()

    # --- mirror SimCluster's per-round surface for a single local model -------
    def load_global(self, params: dict) -> None:
        self.model.update(tree_clone(params))
        mx.eval(self.model.parameters())

    def params(self) -> dict:
        return self.model.trainable_parameters()

    def initial_params(self) -> dict:
        """Params to seed the algorithm's global state (this rank's model)."""
        return self.params()

    def inner_step(self) -> float:
        X, y = next(self.data)
        loss, grads = self._lvg(self.model, X, y)
        self.opt.update(self.model, grads)
        mx.eval(self.model.parameters(), self.opt.state, loss)
        return float(loss)

    def eval_model(self, params: dict) -> nn.Module:
        """Load params into the local model and return it for evaluation."""
        self.model.update(tree_clone(params))
        mx.eval(self.model.parameters())
        return self.model
