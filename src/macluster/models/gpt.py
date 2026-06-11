"""Compact decoder-only GPT in MLX (the proposal's large-compute axis).

A standard pre-LayerNorm Transformer decoder: token + learned positional
embeddings, ``n_layer`` blocks of (causal multi-head self-attention -> MLP with
GELU), residual connections, a final LayerNorm, and a linear LM head. This is
the same architecture family as GPT-2; ``gpt124m`` at vocab 50257 reproduces the
~124M-parameter GPT-2 small size so we can study compute-vs-communication
scaling against the tiny ``gpt_small`` (chargpt).

RANDOM INIT ONLY: we never load pretrained GPT-2 weights. The point of this
project is the *systems* behaviour (how a big model changes the compute/comm
ratio of low-communication distributed training), not language-model quality, so
training from scratch on a small corpus is exactly what we want.

Causal self-attention uses ``mx.fast.scaled_dot_product_attention`` with the
built-in ``"causal"`` mask, so attention runs as a fused kernel on the MLX GPU.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable

import mlx.core as mx
import mlx.nn as nn


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256       # maximum context length (positions)
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    mlp_ratio: int = 4          # MLP hidden = mlp_ratio * n_embd


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention via fused scaled-dot-product attention."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)
        # One projection for q, k, v (split after), plus an output projection.
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=True)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, C = x.shape
        qkv = self.c_attn(x)                       # (B, T, 3C)
        q, k, v = mx.split(qkv, 3, axis=-1)        # each (B, T, C)

        # (B, T, C) -> (B, n_head, T, head_dim)
        def heads(t: mx.array) -> mx.array:
            return t.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        q, k, v = heads(q), heads(k), heads(v)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)  # merge heads
        return self.c_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = cfg.mlp_ratio * cfg.n_embd
        self.c_fc = nn.Linear(cfg.n_embd, hidden, bias=True)
        self.c_proj = nn.Linear(hidden, cfg.n_embd, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.c_proj(nn.gelu(self.c_fc(x)))


class Block(nn.Module):
    """Pre-LayerNorm Transformer block: x + attn(ln(x)); x + mlp(ln(x))."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # token embeddings
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)   # learned positions
        self.blocks = [Block(cfg) for _ in range(cfg.n_layer)]
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        # Weight-tied LM head (Press & Wolf, 2017; as in GPT-2): the output
        # projection reuses the token-embedding matrix, so there is no separate
        # head parameter. This is what gives the canonical ~124M count at vocab
        # 50257 (an untied head would add vocab*n_embd ~= 38.6M extra params).

    def __call__(self, idx: mx.array) -> mx.array:
        """idx: int (B, T) token ids -> logits (B, T, vocab_size)."""
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        pos = mx.arange(T)
        x = self.wte(idx) + self.wpe(pos)        # (B, T, n_embd), pos broadcasts over B
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.wte.as_linear(x)         # tied LM head: logits over vocab


# --------------------------------------------------------------------------- #
# configs + constructors (compute-load axis). The ``*_cfg`` functions return the
# geometry only (cheap, no allocation) so the pipeline splitter can size a model
# too big to instantiate (gpt3b); the matching ``gpt_*`` build the full model.
# --------------------------------------------------------------------------- #
def gpt_small_cfg(vocab_size: int) -> GPTConfig:
    return GPTConfig(vocab_size=vocab_size, block_size=256, n_layer=4, n_head=4, n_embd=128)


def gpt124m_cfg(vocab_size: int) -> GPTConfig:
    return GPTConfig(vocab_size=vocab_size, block_size=1024, n_layer=12, n_head=12, n_embd=768)


def gpt_xl_cfg(vocab_size: int) -> GPTConfig:
    """GPT-2-XL geometry (~1.5B params): 48 layers, 1600-wide, 25 heads (head_dim 64)."""
    return GPTConfig(vocab_size=vocab_size, block_size=1024, n_layer=48, n_head=25, n_embd=1600)


def gpt3b_cfg(vocab_size: int) -> GPTConfig:
    """~2.7B-param geometry (GPT-Neo-2.7B-like): 32 layers, 2560-wide, 20 heads
    (head_dim 128). fp32 Adam state (params + grad + 2 moments = 16 B/param) puts
    this well over a single 48GB node, which is exactly why it needs the split."""
    return GPTConfig(vocab_size=vocab_size, block_size=1024, n_layer=32, n_head=20, n_embd=2560)


def gpt_small(vocab_size: int) -> GPT:
    """Tiny char-level GPT for chargpt smoke + real-learning runs."""
    return GPT(gpt_small_cfg(vocab_size))


def gpt124m(vocab_size: int) -> GPT:
    """GPT-2-small-sized model (~124M params at vocab 50257)."""
    return GPT(gpt124m_cfg(vocab_size))


def gpt_xl(vocab_size: int) -> GPT:
    """~1.5B-param GPT (pipeline target; usually too big for a single node)."""
    return GPT(gpt_xl_cfg(vocab_size))


def gpt3b(vocab_size: int) -> GPT:
    """~2.7B-param GPT (pipeline-only headline model; does not fit one node)."""
    return GPT(gpt3b_cfg(vocab_size))


# Model name -> (vocab_size -> GPTConfig). The first two names match the
# ``model_fns`` keys in data/text.py (so DP and MP share a name); ``gpt_xl`` /
# ``gpt3b`` are pipeline-only (no monolithic model_fn — they intentionally don't
# fit a single node). The pipeline executor looks a ``--model`` up here.
GPT_CONFIGS: dict[str, Callable[[int], GPTConfig]] = {
    "chargpt": gpt_small_cfg,
    "gpt2": gpt124m_cfg,
    "gpt_xl": gpt_xl_cfg,
    "gpt3b": gpt3b_cfg,
}


def num_params(model: nn.Module) -> int:
    from mlx.utils import tree_flatten

    return int(sum(p.size for _, p in tree_flatten(model.trainable_parameters())))


# --------------------------------------------------------------------------- #
# pipeline (model) parallelism — Phase 8 (see docs/PHASE8_MODELPARALLEL.md)
# --------------------------------------------------------------------------- #
class GPTStage(nn.Module):
    """One contiguous slice of a GPT, for pipeline parallelism. Owns the input
    embeddings iff ``is_first``, a contiguous range of ``n_blocks`` transformer
    blocks, and the final LayerNorm + an **untied** LM head iff ``is_last``.

    forward input/output by position in the pipeline:
      - first stage:  int token ids ``(B, T)``      -> hidden ``(B, T, C)``
      - middle stage: hidden ``(B, T, C)``           -> hidden ``(B, T, C)``
      - last stage:   hidden ``(B, T, C)``           -> logits ``(B, T, V)``

    The head is **untied** (a separate ``nn.Linear``) so the output projection
    does not depend on ``wte`` living on the first stage; this is what makes a
    clean stage cut possible (docs/PHASE8 design decision)."""

    def __init__(self, cfg: GPTConfig, n_blocks: int, is_first: bool, is_last: bool):
        super().__init__()
        self.cfg = cfg
        self.is_first = is_first
        self.is_last = is_last
        if is_first:
            self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
            self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = [Block(cfg) for _ in range(n_blocks)]
        if is_last:
            self.ln_f = nn.LayerNorm(cfg.n_embd)
            self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)  # untied

    def __call__(self, inp: mx.array) -> mx.array:
        if self.is_first:
            B, T = inp.shape
            if T > self.cfg.block_size:
                raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
            x = self.wte(inp) + self.wpe(mx.arange(T))
        else:
            x = inp
        for blk in self.blocks:
            x = blk(x)
        if self.is_last:
            return self.head(self.ln_f(x))
        return x


def _stage_bounds(cfg: GPTConfig, cuts: list[int]) -> list[int]:
    """Validate block-cut indices and return the stage boundaries
    ``[0, *cuts, n_layer]``. Cuts must be strictly inside ``(0, n_layer)`` and
    strictly increasing."""
    bounds = [0, *cuts, cfg.n_layer]
    if not all(0 < c < cfg.n_layer for c in cuts) or bounds != sorted(set(bounds)):
        raise ValueError(f"cuts {cuts} must be increasing within (0, {cfg.n_layer})")
    return bounds


def split_gpt(cfg: GPTConfig, cuts: list[int]) -> list[GPTStage]:
    """Split ``GPT(cfg)`` into ``len(cuts)+1`` contiguous pipeline stages at the
    given block-index boundaries. Stage 0 owns the embeddings; the last stage
    owns ``ln_f`` + the untied head. e.g. ``split_gpt(cfg, [k])`` -> two stages
    (blocks[:k], blocks[k:]). Builds ALL stages in one process (single-machine
    emulation / the correctness oracle); use :func:`build_stage` to materialise
    just one stage per rank on a real cluster."""
    bounds = _stage_bounds(cfg, cuts)
    n_stages = len(bounds) - 1
    return [build_stage(cfg, cuts, i) for i in range(n_stages)]


def build_stage(cfg: GPTConfig, cuts: list[int], rank: int) -> GPTStage:
    """Materialise ONLY stage ``rank`` of the ``len(cuts)+1``-way split. On a real
    cluster each machine builds just its own stage (so a model too big for one
    node never has to be fully instantiated anywhere)."""
    bounds = _stage_bounds(cfg, cuts)
    n_stages = len(bounds) - 1
    if not 0 <= rank < n_stages:
        raise ValueError(f"rank {rank} out of range for {n_stages} stage(s)")
    return GPTStage(
        cfg,
        n_blocks=bounds[rank + 1] - bounds[rank],
        is_first=(rank == 0),
        is_last=(rank == n_stages - 1),
    )


# --------------------------------------------------------------------------- #
# analytic parameter accounting + memory-aware partition (Asteroid-style).
# Counts are derived from geometry alone (no allocation) so a 3B model can be
# sized and partitioned without ever being built.
# --------------------------------------------------------------------------- #
def _block_param_count(cfg: GPTConfig) -> int:
    e = cfg.n_embd
    h = cfg.mlp_ratio * e
    attn = (e * 3 * e + 3 * e) + (e * e + e)   # c_attn + c_proj (+ biases)
    norms = (2 * e) + (2 * e)                   # ln_1 + ln_2 (weight + bias)
    mlp = (e * h + h) + (h * e + e)             # c_fc + c_proj (+ biases)
    return attn + norms + mlp


def _embed_param_count(cfg: GPTConfig) -> int:
    """Token + positional embeddings (owned by the first stage)."""
    return cfg.vocab_size * cfg.n_embd + cfg.block_size * cfg.n_embd


def _head_param_count(cfg: GPTConfig) -> int:
    """Final LayerNorm + the untied LM head (owned by the last stage)."""
    return 2 * cfg.n_embd + cfg.vocab_size * cfg.n_embd


def gpt_param_count(cfg: GPTConfig) -> int:
    """Trainable params of the untied-head GPT(cfg), computed analytically."""
    return _embed_param_count(cfg) + cfg.n_layer * _block_param_count(cfg) + _head_param_count(cfg)


def stage_param_counts(cfg: GPTConfig, cuts: list[int]) -> list[int]:
    """Per-stage trainable-param counts for the given split (analytic)."""
    bounds = _stage_bounds(cfg, cuts)
    n_stages = len(bounds) - 1
    counts = []
    for i in range(n_stages):
        c = (bounds[i + 1] - bounds[i]) * _block_param_count(cfg)
        if i == 0:
            c += _embed_param_count(cfg)
        if i == n_stages - 1:
            c += _head_param_count(cfg)
        counts.append(c)
    return counts


def memory_aware_cut(cfg: GPTConfig, mem_gb: list[float]) -> list[int]:
    """Choose stage boundaries so each stage's PARAMETER memory fits its node's
    budget as evenly as possible, i.e. minimise the worst-case utilisation
    ``max_i(stage_param_bytes_i / mem_gb_i)``. Heterogeneity-aware (Asteroid,
    MobiCom'24): a larger-RAM node is handed more transformer blocks. Returns
    ``len(mem_gb) - 1`` cut indices (``[]`` for a single stage).

    Sizing is by parameter memory only (what decides whether a stage *fits*);
    optimizer-state and activation room are left as headroom in ``mem_gb``."""
    n_stages = len(mem_gb)
    if n_stages < 2:
        return []
    if n_stages > cfg.n_layer:
        raise ValueError(f"{n_stages} stages > {cfg.n_layer} blocks: cannot give each stage a block")
    if any(g <= 0 for g in mem_gb):
        raise ValueError(f"memory budgets must be positive, got {mem_gb}")
    best: tuple[float, list[int]] | None = None
    for cuts in itertools.combinations(range(1, cfg.n_layer), n_stages - 1):
        counts = stage_param_counts(cfg, list(cuts))
        util = max(c / g for c, g in zip(counts, mem_gb))   # bytes are 4*count; ratio cancels
        if best is None or util < best[0]:
            best = (util, list(cuts))
    assert best is not None
    return best[1]
