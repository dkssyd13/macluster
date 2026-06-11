"""PipelineCluster: ONE pipeline STAGE per machine, over a real grove cluster.

This is the Phase 8 model-parallel two-MacBook backend, the model-parallel sibling
of :class:`backends.grove_backend.GroveCluster`. Where GroveCluster runs a full
model replica per rank and aggregates with ``all_sum`` (data parallelism), here
each rank owns exactly ONE contiguous slice of a single model (``rank == stage``)
and the seam crosses the wire: hidden activations flow forward over
``grove.send``/``recv``, cotangents flow backward, and a model too big for one Mac
is trained by splitting it across both. Only the rank's own stage is ever
materialised (:func:`models.gpt.build_stage`), so a 3B model never has to fit on
one node.

The seam math is identical to the validated single-process oracle
(:func:`backends.pipeline.pipeline_grads`); only the ``h`` / ``dL/dh`` hand-off
moves to real ``send``/``recv``.

Schedule: 1F1B (one-forward-one-backward) to bound peak activation memory at
O(warmup) rather than O(#micro-batches). ``grove.send``/``recv`` are *blocking*
point-to-point ops, so a naive interleaving could deadlock (both neighbours
blocked on send). We avoid that with the standard 1F1B ordering that keeps each
seam **complementary** -- whenever one stage sends, its neighbour receives:

    warmup  = (world_size - 1) - rank   forward-only micro-batches
    steady  : last stage  -> forward THEN backward;
              earlier      -> backward THEN forward   (drain before pushing)
    cooldown= warmup                    backward-only micro-batches

For the 2-Mac target (2 stages) this reduces to a strictly alternating
send/recv on the single seam -- deadlock-free by construction.

At ``world_size == 1`` there are no seams: the single stage is the whole model
(both first and last), runs entirely locally, and exercises no collectives, so
this backend also runs as a single-machine smoke (mirroring GroveCluster's W=1).

The world is brought up by the launcher (``scripts/grove_entry.py``) BEFORE this
class is constructed; we never call ``grove.init()`` here. Every rank must build
the data iterator identically (same seed, single shard) so that, drawn in
forward order, micro-batch ``i``'s tokens on the first stage line up with its
targets on the last stage.
"""

from __future__ import annotations

import grove
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from ..models.gpt import GPTConfig, build_stage


class PipelineCluster:
    def __init__(
        self,
        gpt_cfg: GPTConfig,
        cuts: list[int],
        inner_opt_fn,
        data_iter,
        loss_fn,
        batch_size: int,
        seq_len: int,
    ):
        self.rank = int(grove.rank)
        self.world = int(grove.world_size)
        n_stages = len(cuts) + 1
        if n_stages != self.world:
            raise ValueError(
                f"pipeline needs n_stages ({n_stages}) == world_size ({self.world}); "
                f"set --world-size {n_stages} (or adjust --cut) so one stage lands per rank"
            )
        self.cfg = gpt_cfg
        self.cuts = list(cuts)
        self.stage = build_stage(gpt_cfg, cuts, self.rank)
        self.stage.train()
        mx.eval(self.stage.parameters())
        self.opt = inner_opt_fn()
        self.data = data_iter
        self.loss_fn = loss_fn
        self.is_first = self.rank == 0
        self.is_last = self.rank == self.world - 1
        # seam tensor shape for blocking recv (hidden states are n_embd-wide)
        self.B, self.T, self.C = batch_size, seq_len, gpt_cfg.n_embd

    @property
    def world_size(self) -> int:
        return self.world

    def barrier(self) -> None:
        grove.barrier()

    # --- gradient accumulation across micro-batches --------------------------
    def _accum(self, g: dict) -> None:
        if self._grad_accum is None:
            self._grad_accum = g
        else:
            self._grad_accum = tree_map(lambda a, b: a + b, self._grad_accum, g)

    # --- 1F1B primitives: one micro-batch forward / backward across the seam --
    def _forward_micro(self, i: int) -> None:
        # Every rank draws in lockstep so micro i's tokens (first stage) and
        # targets (last stage) come from the same RNG draw; first/middle ranks
        # ignore y, middle/last ranks ignore X.
        X, y = next(self.data)
        if self.is_first:
            inp = X
        else:
            inp = grove.recv((self.B, self.T, self.C), mx.float32, src=self.rank - 1)
        self._inp[i] = inp
        if self.is_last:
            self._y[i] = y
            return  # last stage defers its forward into the backward's value_and_grad
        out = self.stage(inp)
        mx.eval(out)
        grove.send(out, dst=self.rank + 1)
        self._seam_bytes += int(out.size) * 4  # fp32 hidden state, forward

    def _backward_micro(self, i: int) -> None:
        inp = self._inp.pop(i)
        if self.is_last:
            y = self._y.pop(i)

            def last_loss(params, x):
                self.stage.update(params)
                return self.loss_fn(self.stage(x), y)

            loss, (g, cot) = mx.value_and_grad(last_loss, argnums=(0, 1))(
                self.stage.trainable_parameters(), inp
            )
            mx.eval(loss, g, cot)
            self._loss_sum += float(loss)
            self._accum(g)
            if not self.is_first:  # send dL/dh upstream (guarded for the W=1 single stage)
                grove.send(cot, dst=self.rank - 1)
                self._seam_bytes += int(cot.size) * 4
            return

        cot = grove.recv((self.B, self.T, self.C), mx.float32, src=self.rank + 1)
        if self.is_first:
            def surrogate0(params):
                self.stage.update(params)
                return (self.stage(inp) * cot).sum()

            g = mx.grad(surrogate0)(self.stage.trainable_parameters())
            mx.eval(g)
            self._accum(g)
        else:
            def surrogate(params, x):
                self.stage.update(params)
                return (self.stage(x) * cot).sum()

            _, (g, cot_prev) = mx.value_and_grad(surrogate, argnums=(0, 1))(
                self.stage.trainable_parameters(), inp
            )
            mx.eval(g, cot_prev)
            self._accum(g)
            grove.send(cot_prev, dst=self.rank - 1)
            self._seam_bytes += int(cot_prev.size) * 4

    # --- one optimizer step over n_micro micro-batches (1F1B) ----------------
    def step(self, n_micro: int) -> tuple[float | None, int]:
        """Run the 1F1B schedule over ``n_micro`` micro-batches, then apply ONE
        optimizer update to this rank's stage. Returns ``(loss, seam_bytes)``
        where ``loss`` is the mean train loss on the LAST rank (the only rank
        that sees it) and ``None`` elsewhere."""
        self._inp = {}
        self._y = {}
        self._grad_accum = None
        self._loss_sum = 0.0
        self._seam_bytes = 0

        warmup = max(0, min(self.world - 1 - self.rank, n_micro))
        steady = n_micro - warmup
        fwd = bwd = 0
        for _ in range(warmup):
            self._forward_micro(fwd); fwd += 1
        for _ in range(steady):
            if self.is_last:
                self._forward_micro(fwd); fwd += 1
                self._backward_micro(bwd); bwd += 1
            else:
                self._backward_micro(bwd); bwd += 1
                self._forward_micro(fwd); fwd += 1
        for _ in range(warmup):
            self._backward_micro(bwd); bwd += 1

        scale = 1.0 / n_micro
        self.opt.update(self.stage, tree_map(lambda v: v * scale, self._grad_accum))
        mx.eval(self.stage.parameters(), self.opt.state)
        loss = (self._loss_sum / n_micro) if self.is_last else None
        return loss, self._seam_bytes

    # --- forward-only pipeline eval (full model, split across ranks) ---------
    def eval_loss(self, eval_batches: list) -> dict | None:
        """Run every eval batch forward through the whole pipeline (no backward).
        Returns ``{'val_loss', 'perplexity'}`` on the LAST rank, ``None`` else.
        Every rank must iterate the SAME ``eval_batches`` in the same order."""
        import numpy as np

        self.stage.eval()
        loss_sum = 0.0
        n_tokens = 0
        for X, y in eval_batches:
            B, T = int(X.shape[0]), int(X.shape[1])
            if self.is_first:
                h = self.stage(X)
                mx.eval(h)
                if not self.is_last:
                    grove.send(h, dst=self.rank + 1)
            else:
                h = grove.recv((B, T, self.C), mx.float32, src=self.rank - 1)
                h = self.stage(h)
                mx.eval(h)
                if not self.is_last:
                    grove.send(h, dst=self.rank + 1)
            if self.is_last:
                logits = h  # last stage's output is the logits
                V = int(logits.shape[-1])
                loss_sum += float(
                    nn.losses.cross_entropy(
                        logits.reshape(B * T, V), y.reshape(B * T), reduction="sum"
                    )
                )
                n_tokens += B * T
        self.stage.train()
        if not self.is_last:
            return None
        val_loss = loss_sum / max(n_tokens, 1)
        return {"val_loss": val_loss, "perplexity": float(np.exp(min(val_loss, 20.0)))}
