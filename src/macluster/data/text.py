"""Text (language-modeling) task builder for the Transformer / large-compute axis.

Three corpora, all reduced to a single 1-D stream of int token ids that is then
sharded into ``W`` contiguous segments (one per worker) plus a held-out tail for
evaluation. Each worker draws random fixed-length windows from its own segment,
so replicas see genuinely different data (otherwise distributed averaging is a
no-op):

  - ``shakespeare`` (default): char-level tinyshakespeare. Vocab is the set of
    distinct characters (~65). Tiny vocab keeps the LM-head / embedding small so
    the char model (``gpt_small``) is genuinely small.
  - ``wikitext``: GPT-2 BPE via ``tiktoken`` (vocab 50257). We first try to
    download the wikitext-2-raw corpus; the upstream mirrors for that dataset
    are flaky, so if the download fails we FALL BACK to BPE-tokenizing
    tinyshakespeare. The fallback is recorded in ``meta['bpe_source']`` and noted
    in this module so results stay reproducible/honest. Either way the *vocab*
    is the full 50257-token GPT-2 vocab, which is what makes ``gpt124m`` a real
    ~124M-parameter model.
  - ``synthetic=True``: random token ids over a small vocab (256), no download —
    used by the smoke tests.

Contract (see ``task.py``):
  loss_fn(model, X, y) -> mean next-token cross-entropy (scalar mlx array)
  eval_fn(model, eval_batches) -> {'val_loss', 'perplexity'}
  metric='val_loss', metric_goal='min'
  model_fns = {'chargpt': gpt_small, 'gpt2': gpt124m (tied), 'gpt2_untied': gpt124m_untied}
  X, y are int32 mlx arrays of shape (batch_size, seq_len); y is X shifted by one.
"""

from __future__ import annotations

import os
import urllib.request

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..models.gpt import gpt124m, gpt124m_untied, gpt_small
from ..task import Task

_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
# wikitext-2-raw mirror (HuggingFace datasets resolve). Kept best-effort; on any
# failure we fall back to BPE-tokenizing tinyshakespeare (see module docstring).
_WIKITEXT_URLS = [
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
    "wikitext-2-raw-v1/train-00000-of-00001.parquet",
    "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
]
_SYNTH_VOCAB = 256


# --------------------------------------------------------------------------- #
# corpus loading -> raw text
# --------------------------------------------------------------------------- #
def _download_text(url: str, dest: str) -> str | None:
    try:
        if not os.path.exists(dest):
            req = urllib.request.Request(url, headers={"User-Agent": "macluster/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            with open(dest, "wb") as f:
                f.write(data)
        with open(dest, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        # corrupt/partial cache should not poison later runs
        if os.path.exists(dest):
            try:
                os.remove(dest)
            except OSError:
                pass
        return None


def _load_shakespeare_text(data_dir: str) -> str:
    cache = os.path.join(data_dir, "text")
    os.makedirs(cache, exist_ok=True)
    text = _download_text(_SHAKESPEARE_URL, os.path.join(cache, "tinyshakespeare.txt"))
    if text is None:
        raise RuntimeError(
            "could not download tinyshakespeare; run a --synthetic smoke test, or "
            "place input.txt at data_dir/text/tinyshakespeare.txt"
        )
    return text


def _load_wikitext_text(data_dir: str) -> tuple[str, str]:
    """Return (text, source_tag). Falls back to tinyshakespeare on download failure."""
    cache = os.path.join(data_dir, "text")
    os.makedirs(cache, exist_ok=True)
    for i, url in enumerate(_WIKITEXT_URLS):
        # only the plain-text mirror is usable without a parquet reader
        if url.endswith(".parquet"):
            continue
        text = _download_text(url, os.path.join(cache, f"wikitext2_{i}.txt"))
        if text is not None and len(text) > 10000:
            return text, "wikitext-2-raw"
    # Fallback: BPE-tokenize tinyshakespeare (documented in docstring + meta).
    return _load_shakespeare_text(data_dir), "tinyshakespeare-bpe-fallback"


# --------------------------------------------------------------------------- #
# text -> 1-D int token-id stream + vocab size
# --------------------------------------------------------------------------- #
def _encode_char(text: str) -> tuple[np.ndarray, int, dict]:
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.fromiter((stoi[c] for c in text), dtype=np.int32, count=len(text))
    return ids, len(chars), {"tokenizer": "char", "vocab_size": len(chars)}


def _encode_bpe(text: str) -> tuple[np.ndarray, int, dict]:
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    ids = np.asarray(enc.encode_ordinary(text), dtype=np.int32)
    return ids, enc.n_vocab, {"tokenizer": "gpt2-bpe", "vocab_size": enc.n_vocab}


def _build_ids(variant: str, synthetic: bool, data_dir: str, seed: int):
    if synthetic:
        rng = np.random.default_rng(seed)
        ids = rng.integers(0, _SYNTH_VOCAB, size=200_000).astype(np.int32)
        return ids, _SYNTH_VOCAB, {"tokenizer": "synthetic", "vocab_size": _SYNTH_VOCAB}

    if variant == "shakespeare":
        text = _load_shakespeare_text(data_dir)
        ids, vocab, meta = _encode_char(text)
        meta["corpus"] = "tinyshakespeare"
        return ids, vocab, meta

    if variant == "wikitext":
        text, source = _load_wikitext_text(data_dir)
        ids, vocab, meta = _encode_bpe(text)
        meta["corpus"] = "wikitext"
        meta["bpe_source"] = source  # 'wikitext-2-raw' or the documented fallback
        return ids, vocab, meta

    raise KeyError(f"unknown text variant {variant!r}")


# --------------------------------------------------------------------------- #
# stream -> sharded infinite (X, y) window iterators + eval batches
# --------------------------------------------------------------------------- #
def _shard_stream(ids: np.ndarray, world_size: int, eval_frac: float = 0.1):
    """Split into W contiguous train segments + a held-out tail eval segment."""
    n = len(ids)
    n_eval = max(int(n * eval_frac), 1)
    train = ids[: n - n_eval]
    eval_ids = ids[n - n_eval :]
    per = len(train) // world_size
    shards = [train[r * per : (r + 1) * per] for r in range(world_size)]
    return shards, eval_ids


def _infinite_windows(segment: np.ndarray, batch_size: int, seq_len: int, seed: int):
    """Yield (X, y) int32 mlx arrays (batch_size, seq_len); y = X shifted by one."""
    rng = np.random.default_rng(seed)
    # need seq_len + 1 tokens per window (inputs + the shifted target)
    hi = len(segment) - seq_len - 1
    if hi <= 0:
        raise ValueError(
            f"shard too small ({len(segment)} tokens) for seq_len={seq_len}; "
            "use a smaller --seq-len or fewer workers"
        )
    while True:
        starts = rng.integers(0, hi, size=batch_size)
        xb = np.stack([segment[s : s + seq_len] for s in starts]).astype(np.int32)
        yb = np.stack([segment[s + 1 : s + 1 + seq_len] for s in starts]).astype(np.int32)
        yield mx.array(xb), mx.array(yb)


def _eval_batches(segment: np.ndarray, batch_size: int, seq_len: int, max_batches: int):
    """Deterministic non-overlapping windows over the held-out tail."""
    out = []
    step = seq_len
    i = 0
    while i + seq_len + 1 <= len(segment) and len(out) < max_batches * batch_size:
        out.append((segment[i : i + seq_len], segment[i + 1 : i + 1 + seq_len]))
        i += step
    batches = []
    for b in range(0, len(out) - batch_size + 1, batch_size):
        xs = np.stack([out[b + j][0] for j in range(batch_size)]).astype(np.int32)
        ys = np.stack([out[b + j][1] for j in range(batch_size)]).astype(np.int32)
        batches.append((mx.array(xs), mx.array(ys)))
        if len(batches) >= max_batches:
            break
    return batches


# --------------------------------------------------------------------------- #
# loss + eval (next-token cross-entropy / perplexity)
# --------------------------------------------------------------------------- #
def lm_loss(model: nn.Module, X: mx.array, y: mx.array) -> mx.array:
    logits = model(X)                                  # (B, T, V)
    B, T, V = logits.shape
    return nn.losses.cross_entropy(
        logits.reshape(B * T, V), y.reshape(B * T), reduction="mean"
    )


def lm_eval(model: nn.Module, batches: list) -> dict:
    model.eval()
    loss_sum = 0.0
    n_tokens = 0
    for X, y in batches:
        logits = model(X)
        B, T, V = logits.shape
        loss_sum += float(
            nn.losses.cross_entropy(logits.reshape(B * T, V), y.reshape(B * T), reduction="sum")
        )
        n_tokens += B * T
    model.train()
    val_loss = loss_sum / max(n_tokens, 1)
    # clamp exponent so a random-init model's huge loss does not overflow to inf
    perplexity = float(np.exp(min(val_loss, 20.0)))
    return {"val_loss": val_loss, "perplexity": perplexity}


# --------------------------------------------------------------------------- #
# public builder
# --------------------------------------------------------------------------- #
def make_text_task(
    world_size: int,
    variant: str = "shakespeare",
    batch_size: int = 32,
    seq_len: int = 128,
    synthetic: bool = False,
    data_dir: str = "data/cache",
    seed: int = 0,
    eval_max_batches: int = 16,
) -> Task:
    ids, vocab_size, meta = _build_ids(variant, synthetic, data_dir, seed)
    shards, eval_ids = _shard_stream(ids, world_size)

    train_shards = [
        _infinite_windows(shards[r], batch_size, seq_len, seed=seed + 1 + r)
        for r in range(world_size)
    ]
    eval_batches = _eval_batches(eval_ids, batch_size, seq_len, eval_max_batches)

    # Close over the built vocab so the candidate models match the tokenizer.
    # ``gpt2_untied`` is the apples-to-apples DP-vs-MP model: a monolithic GPT
    # with the SAME untied head the pipeline split uses, so a data-parallel run
    # and a model-parallel run of ``--model gpt2_untied`` train identical nets.
    model_fns = {
        "chargpt": lambda: gpt_small(vocab_size),
        "gpt2": lambda: gpt124m(vocab_size),
        "gpt2_untied": lambda: gpt124m_untied(vocab_size),
    }

    meta.update(
        {
            "variant": variant,
            "synthetic": synthetic,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "world_size": world_size,
            "n_tokens": int(len(ids)),
            "vocab_size": vocab_size,
        }
    )

    return Task(
        name=variant,
        kind="text",
        loss_fn=lm_loss,
        eval_fn=lm_eval,
        metric="val_loss",
        metric_goal="min",
        train_shards=train_shards,
        eval_batches=eval_batches,
        model_fns=model_fns,
        meta=meta,
    )
