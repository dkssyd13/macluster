#!/usr/bin/env bash
# =============================================================================
# Provision data/cache/ on a fresh Mac (e.g. the borrowed 24GB one) WITHOUT
# AirDrop. Downloads the wikitext-2-raw corpus into data/cache/text/ and then
# VERIFIES it is the real corpus -- NOT the silent tinyshakespeare fallback that
# make_text_task drops to when the single wikitext URL is unreachable. Exits 1
# (loud) if it only got the fallback, so you never start a timed 2-Mac phase on
# mismatched data.
#
#   ./scripts/fetch_data.sh           # wikitext (what the 2-Mac run needs)
#   ./scripts/fetch_data.sh cifar     # also warm the CIFAR-10 cache (sim runs)
#
# If this fails (network blocks the GitHub-raw mirror), fall back to AirDropping
# data/cache/text/ from the 48GB Mac -- both paths are byte-checked the same way.
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

WANT_CIFAR=0
for a in "$@"; do [[ "$a" == "cifar" ]] && WANT_CIFAR=1; done

echo "[fetch_data] repo: $ROOT"
echo "[fetch_data] downloading + verifying wikitext-2-raw into data/cache/text/ ..."

# The expected fingerprint of the REAL corpus (what the 48GB Mac has).
EXPECTED_SOURCE="wikitext-2-raw"
EXPECTED_TOKENS="2448382"

uv run python - "$EXPECTED_SOURCE" "$EXPECTED_TOKENS" <<'PY'
import sys
from macluster.data.text import make_text_task

want_source, want_tokens = sys.argv[1], int(sys.argv[2])
t = make_text_task(1, variant="wikitext", batch_size=8, seq_len=128, data_dir="data/cache", seed=0)
src = t.meta.get("bpe_source")
tok = t.meta["n_tokens"]
print(f"  [data] source={src} vocab={t.meta['vocab_size']} tokens={tok}")
if src != want_source:
    print("############################################################")
    print(f"  [data] !!! GOT THE FALLBACK ({src}), NOT real wikitext !!!")
    print("  The single wikitext mirror was unreachable from this Mac, so the")
    print("  code silently fell back to tinyshakespeare. Do NOT run a timed phase.")
    print("  Fix: AirDrop data/cache/text/ from the 48GB Mac, or retry on a")
    print("       network that can reach raw.githubusercontent.com .")
    print("############################################################")
    sys.exit(1)
if tok != want_tokens:
    print(f"  [data] WARN: tokens={tok} != expected {want_tokens}. The corpus differs")
    print("         from the 48GB Mac -- AirDrop its data/cache/text/ to match exactly.")
    sys.exit(1)
print(f"  [data] OK: real wikitext, matches the 48GB Mac ({want_source}, {want_tokens} tokens).")
PY
RC=$?
if [[ $RC -ne 0 ]]; then
  echo "[fetch_data] wikitext provisioning FAILED (rc=$RC). See message above."
  exit 1
fi

if [[ $WANT_CIFAR -eq 1 ]]; then
  echo "[fetch_data] warming CIFAR-10 cache (data/cache/cifar/) ..."
  uv run python - <<'PY' || { echo "[fetch_data] CIFAR warm failed."; exit 1; }
from macluster.data.cifar import make_cifar_task
t = make_cifar_task(1, batch_size=128, data_dir="data/cache", seed=0)
print(f"  [data] cifar ready: classes={t.meta.get('num_classes', '?')}")
PY
fi

echo "[fetch_data] done. data/cache/text/ is verified-real wikitext."
echo "[fetch_data] sanity: the 24GB Mac and 48GB Mac MUST both print"
echo "             source=wikitext-2-raw tokens=2448382 before any timed phase."
