#!/usr/bin/env bash
# =============================================================================
# RUN THIS ON THE 48GB MAC.  (the LAUNCHER / coordinator = rank 0 = the BIGGER
# pipeline stage.)  The memory-aware cut [48,24] puts the larger stage on rank 0,
# and `grove start` is always rank 0 -- so the 48GB Mac MUST run this script.
#
#   1) On the 48GB Mac:  ./scripts/run_mac_48gb.sh
#   2) On the 24GB Mac:  ./scripts/run_mac_24gb.sh      (start within ~2 min)
#
# Runs three phases in order, each a full 2-Mac run; later phases reuse nothing
# from earlier ones, so even if gpt3b OOMs you still have the smoke + gpt_xl runs:
#   smoke  synthetic connectivity + seam check (instant)
#   xl     gpt_xl  (~1.6B) on wikitext  -- fits both Macs comfortably
#   3b     gpt3b   (~2.78B) on wikitext -- the headline (only MP can train it)
#
# Run a subset by naming phases:  ./scripts/run_mac_48gb.sh xl 3b
# (run the SAME phase args on BOTH Macs.)
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PHASES=("$@"); [[ ${#PHASES[@]} -eq 0 ]] && PHASES=(smoke xl 3b)

config_for () {  # phase name -> config path
  case "$1" in
    smoke) echo configs/grove/pipeline_smoke.env ;;
    xl)    echo configs/grove/pipeline_xl.env ;;
    3b)    echo configs/grove/pipeline_3b.env ;;
    *)     echo "" ;;
  esac
}

echo "[48GB] repo: $ROOT"
echo "[48GB] syncing deps (uv sync)..."
uv sync -q || { echo "[48GB] uv sync failed -- is uv installed?"; exit 1; }

# Pre-warm the wikitext cache so the timed runs aren't blocked by a download.
# (Each Mac downloads to its own cache; confirm both print the same source+tokens.)
if printf '%s\n' "${PHASES[@]}" | grep -qvx smoke; then
  echo "[48GB] pre-warming wikitext cache..."
  uv run python - <<'PY' || echo "[48GB] WARN: wikitext pre-warm failed; the run will try again at launch."
from macluster.data.text import make_text_task
t = make_text_task(1, variant="wikitext", batch_size=8, seq_len=128, data_dir="data/cache", seed=0)
print(f"  [data] source={t.meta.get('bpe_source')} vocab={t.meta['vocab_size']} tokens={t.meta['n_tokens']}")
print("  [data] ^ the 24GB Mac MUST show the SAME source + tokens (else the two stages see different data).")
PY
fi

run_phase () {  # phase name
  local phase="$1" cfg; cfg="$(config_for "$phase")"
  if [[ -z "$cfg" || ! -f "$cfg" ]]; then echo "[48GB] unknown/missing phase '$phase' ($cfg) -- skipping"; return; fi
  set -a; source "$cfg"; set +a
  local name="${CLUSTER:-macluster}" n="${N:-2}"
  echo "============================================================"
  echo "[48GB] PHASE '$phase'  model=${MACLUSTER_MODEL:-?} task=${MACLUSTER_TASK:-?} cluster=$name"
  echo "[48GB] grove start -> WAITING for the 24GB Mac to join '$name'..."
  echo "============================================================"
  if uv run grove start scripts/grove_entry.py -n "$n" --name "$name" --logs; then
    echo "[48GB] PHASE '$phase' DONE -> runs/ (look for *-rank0)"
  else
    echo "[48GB] PHASE '$phase' FAILED (rc=$?) -- continuing to the next phase."
  fi
  sleep 3  # let the cluster tear down before the next phase
}

for p in "${PHASES[@]}"; do run_phase "$p"; done
echo "[48GB] all requested phases finished. Metrics under runs/  (this Mac = rank0 = stage0)."
