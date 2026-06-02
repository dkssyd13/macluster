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

import math
from dataclasses import dataclass

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
# constructors (compute-load axis: ~0.8M vs ~124M params)
# --------------------------------------------------------------------------- #
def gpt_small(vocab_size: int) -> GPT:
    """Tiny char-level GPT for chargpt smoke + real-learning runs."""
    return GPT(GPTConfig(vocab_size=vocab_size, block_size=256, n_layer=4, n_head=4, n_embd=128))


def gpt124m(vocab_size: int) -> GPT:
    """GPT-2-small-sized model (~124M params at vocab 50257)."""
    return GPT(GPTConfig(vocab_size=vocab_size, block_size=1024, n_layer=12, n_head=12, n_embd=768))


def num_params(model: nn.Module) -> int:
    from mlx.utils import tree_flatten

    return int(sum(p.size for _, p in tree_flatten(model.trainable_parameters())))
