"""Pipeline (model) parallelism core — Phase 8. See docs/PHASE8_MODELPARALLEL.md.

The split that lets a model bigger than one node train across the two Macs:
activations cross each stage seam forward, gradients cross backward. This module
holds the **seam math**, validated to reproduce the monolithic backward to ~1e-8.

``pipeline_grads`` runs all stages in ONE process (single-machine emulation +
correctness oracle). The grove backend will reuse exactly this seam logic but
move each ``h`` / ``dL/dh`` across a real ``grove.send``/``recv`` between ranks.

Seam (per stage boundary):
  1. stage i forward  ``h_i = f_i(h_{i-1})``                (h crosses forward)
  2. last stage: ``value_and_grad(loss, argnums=(params, input))`` yields its
     param grads AND ``dL/dh`` for its input.
  3. earlier stage i: backprop via the surrogate scalar ``<stop_grad(cot), f_i(h_{i-1})>``;
     its grad w.r.t. params is stage i's grad, and (for non-first stages) its grad
     w.r.t. the input is the cotangent ``dL/dh_{i-1}`` for the previous seam.
"""
from __future__ import annotations

from typing import Callable

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from ..models.gpt import GPTConfig, split_gpt


def seq_cross_entropy(logits: mx.array, y: mx.array) -> mx.array:
    """Mean next-token cross-entropy over ``(B, T, V)`` logits and ``(B, T)``
    int targets. This is the **logits-level** loss the seam differentiates
    (``loss_fn(logits, y)``), distinct from a task's ``loss_fn(model, X, y)``."""
    B, T, V = logits.shape
    return nn.losses.cross_entropy(logits.reshape(B * T, V), y.reshape(B * T), reduction="mean")


def pipeline_grads(stages: list, x: mx.array, y: mx.array, loss_fn: Callable):
    """Forward ``x`` through ``stages`` (activations stashed at each seam), then
    backprop stage-by-stage. Returns ``(loss, grads)`` where ``grads[i]`` is a
    param pytree matching ``stages[i].trainable_parameters()``.

    Numerically identical to a monolithic forward/backward of the composed
    stages — the split is purely a device-placement device. ``stages[0]`` must be
    the embedding (first) stage taking int token ids; later stages take float
    hidden states.
    """
    n = len(stages)
    if n == 0:
        raise ValueError("no stages")

    # ---- forward: stash the input activation of each stage (the seam values) ----
    acts = [x]
    h = x
    for i in range(n - 1):
        h = stages[i](h)
        mx.eval(h)
        h = mx.array(h)  # detach -> next stage's input leaf (would be send/recv)
        acts.append(h)

    grads: list = [None] * n

    # ---- last stage: loss + cotangent dL/d(its input) ----
    last = stages[-1]

    def last_loss(params, inp):
        last.update(params)
        return loss_fn(last(inp), y)

    loss, (grads[n - 1], cot) = mx.value_and_grad(last_loss, argnums=(0, 1))(
        last.trainable_parameters(), acts[-1]
    )
    mx.eval(loss, grads[n - 1], cot)

    # ---- earlier stages: surrogate <cot, f_i(input)> ----
    for i in range(n - 2, -1, -1):
        stage = stages[i]
        inp = acts[i]
        if i == 0:
            # first stage: int token input, differentiate w.r.t. params only
            def surrogate0(params, _stage=stage, _inp=inp, _cot=cot):
                _stage.update(params)
                return (_stage(_inp) * _cot).sum()

            grads[i] = mx.grad(surrogate0)(stage.trainable_parameters())
        else:
            # middle stage: need param grads AND the cotangent for the prev seam
            def surrogate(params, sinp, _stage=stage, _cot=cot):
                _stage.update(params)
                return (_stage(sinp) * _cot).sum()

            _, (grads[i], cot) = mx.value_and_grad(surrogate, argnums=(0, 1))(
                stage.trainable_parameters(), inp
            )
        mx.eval(grads[i], cot)

    return loss, grads


def _tree_add(a: dict, b: dict) -> dict:
    return tree_map(lambda x, y: x + y, a, b)


class _StagedModel(nn.Module):
    """A read-only full-model view over a stage list, for evaluation: chains every
    stage's forward so a task's ``eval_fn`` (which wants one ``nn.Module`` with
    ``__call__`` / ``train`` / ``eval``) sees the whole pipelined model. Shares the
    SAME stage objects, so it reflects the latest trained params."""

    def __init__(self, stages: list):
        super().__init__()
        self.stages = stages

    def __call__(self, idx: mx.array) -> mx.array:
        x = idx
        for s in self.stages:
            x = s(x)
        return x


class SimPipelineCluster:
    """All pipeline stages in ONE process: single-machine emulation + the
    correctness oracle for the real :class:`PipelineCluster`, mirroring
    :class:`SimCluster` for the data-parallel path. A pipeline *step* applies ONE
    optimizer update per stage after accumulating grads over ``n_micro``
    micro-batches.

    The micro-batches run sequentially here (one process), so the 1F1B schedule
    and its deadlock concerns are irrelevant -- those only matter for the real
    cross-machine seam (:mod:`backends.pipeline_grove`). Communication cost is
    modelled analytically: each seam moves the hidden state forward and the
    cotangent backward, ``2 * B*T*C`` fp32 values per seam per micro-batch."""

    def __init__(self, gpt_cfg: GPTConfig, cuts: list[int], inner_opt_fn, data_iter, loss_fn):
        self.cfg = gpt_cfg
        self.cuts = list(cuts)
        self.stages = split_gpt(gpt_cfg, cuts)
        for s in self.stages:
            s.train()
        mx.eval([s.parameters() for s in self.stages])
        self.opts = [inner_opt_fn() for _ in self.stages]
        self.data = data_iter
        self.loss_fn = loss_fn

    @property
    def n_stages(self) -> int:
        return len(self.stages)

    def _seam_bytes(self, B: int, T: int) -> int:
        # per seam: hidden h forward (B,T,C) + cotangent dL/dh backward (B,T,C), fp32
        return (self.n_stages - 1) * 2 * B * T * self.cfg.n_embd * 4

    def step(self, n_micro: int) -> tuple[float, int]:
        """One optimizer step over ``n_micro`` micro-batches. Returns
        ``(mean_loss, seam_bytes_moved)``."""
        accum: list = [None] * self.n_stages
        loss_sum = 0.0
        seam_bytes = 0
        for _ in range(n_micro):
            X, y = next(self.data)
            loss, grads = pipeline_grads(self.stages, X, y, self.loss_fn)
            loss_sum += float(loss)
            seam_bytes += self._seam_bytes(int(X.shape[0]), int(X.shape[1]))
            accum = [g if a is None else _tree_add(a, g) for a, g in zip(accum, grads)]
        scale = 1.0 / n_micro
        for opt, stage, g in zip(self.opts, self.stages, accum):
            opt.update(stage, tree_map(lambda v: v * scale, g))
        mx.eval([s.parameters() for s in self.stages], [o.state for o in self.opts])
        return loss_sum / n_micro, seam_bytes

    def eval_model(self) -> nn.Module:
        """A whole-model view (all stages chained) for the task's ``eval_fn``."""
        return _StagedModel(self.stages)
