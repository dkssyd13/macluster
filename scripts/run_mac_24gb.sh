#!/usr/bin/env bash
# =============================================================================
# RUN THIS ON THE 24GB MAC.  (the JOINER = rank 1 = the SMALLER pipeline stage.)
# Start scripts/run_mac_48gb.sh on the 48GB Mac FIRST, then run this within ~2 min.
#
#   1) On the 48GB Mac:  ./scripts/run_mac_48gb.sh
#   2) On the 24GB Mac:  ./scripts/run_mac_24gb.sh   <-- this script
#
# Same five phases (smoke mp_mid dp_mid xl 3b), in the same order, as the 48GB
# script. Each phase retries the join for a while so it doesn't matter who reaches
# the phase first. mp_mid + dp_mid are the apples-to-apples DP-vs-MP comparison
# (same model, same data budget); 3b is the MP-only headline.
#
# Run a subset by naming phases:  ./scripts/run_mac_24gb.sh mp_mid dp_mid
# (run the SAME phase args as the 48GB Mac.)
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PHASES=("$@"); [[ ${#PHASES[@]} -eq 0 ]] && PHASES=(smoke mp_mid dp_mid xl 3b)

config_for () {
  case "$1" in
    smoke)  echo configs/grove/pipeline_smoke.env ;;
    mp_mid) echo configs/grove/mp_mid.env ;;
    dp_mid) echo configs/grove/dp_mid.env ;;
    xl)     echo configs/grove/pipeline_xl.env ;;
    3b)     echo configs/grove/pipeline_3b.env ;;
    *)      echo "" ;;
  esac
}

echo "[24GB] repo: $ROOT"
echo "[24GB] syncing deps (uv sync)..."
uv sync -q || { echo "[24GB] uv sync failed -- is uv installed?"; exit 1; }

if printf '%s\n' "${PHASES[@]}" | grep -qvx smoke; then
  echo "[24GB] pre-warming wikitext cache..."
  uv run python - <<'PY' || echo "[24GB] WARN: wikitext pre-warm failed; the run will try again at launch."
from macluster.data.text import make_text_task
t = make_text_task(1, variant="wikitext", batch_size=8, seq_len=128, data_dir="data/cache", seed=0)
print(f"  [data] source={t.meta.get('bpe_source')} vocab={t.meta['vocab_size']} tokens={t.meta['n_tokens']}")
print("  [data] ^ MUST match the 48GB Mac's source + tokens.")
PY
fi

join_phase () {  # phase name
  local phase="$1" cfg; cfg="$(config_for "$phase")"
  if [[ -z "$cfg" || ! -f "$cfg" ]]; then echo "[24GB] unknown/missing phase '$phase' ($cfg) -- skipping"; return; fi
  set -a; source "$cfg"; set +a
  local name="${CLUSTER:-macluster}"
  echo "============================================================"
  echo "[24GB] PHASE '$phase'  model=${MACLUSTER_MODEL:-?} -> joining '$name' (retrying until the 48GB Mac is on this phase)..."
  echo "============================================================"
  local ok=0
  for attempt in $(seq 1 30); do
    if uv run grove join "$name" --logs; then ok=1; break; fi
    echo "[24GB] join attempt $attempt/30 failed (48GB Mac not on '$phase' yet?). Retrying in 5s..."
    sleep 5
  done
  if [[ $ok -eq 1 ]]; then echo "[24GB] PHASE '$phase' DONE -> runs/ (look for *-rank1: has train_loss + eval)";
  else echo "[24GB] PHASE '$phase' could not join after 30 retries -- continuing."; fi
  sleep 3
}

for p in "${PHASES[@]}"; do join_phase "$p"; done

echo "============================================================"
echo "[24GB] all requested phases finished. This Mac = rank1 = stage1."
echo "[24GB] rank1 metrics (train_loss + val_loss/perplexity + peak_mem): runs/*-rank1/"
# Collect results onto the 48GB Mac so everything for the report lives there.
# This is a HARD GATE: the MP convergence curve (train_loss/val_loss/perplexity)
# exists ONLY on this rank1 Mac, so a forgotten/failed copy is UNRECOVERABLE once
# the Mac is returned. We exit non-zero (loud) unless the copy verifiably lands.
DEST="${RESULTS_DEST:-}"
if [[ -z "$DEST" ]]; then
  echo "############################################################"
  echo "[24GB] !!! RESULTS NOT COLLECTED !!!  rank1 holds the ONLY copy of the MP"
  echo "       loss/perplexity curves. Do NOT return this Mac yet. Either:"
  echo "         (a) enable Remote Login on the 48GB Mac, take the RESULTS_DEST=..."
  echo "             value its script printed, and rerun THIS script with it, or"
  echo "         (b) AirDrop the whole runs/ folder to the 48GB Mac by hand."
  echo "############################################################"
  exit 1
fi
echo "[24GB] copying runs/ -> $DEST (rsync)..."
if rsync -az runs/ "$DEST"; then
  echo "[24GB] results copied to the 48GB Mac. Verify there: ls runs/*-rank1/metrics.jsonl"
else
  echo "############################################################"
  echo "[24GB] !!! rsync FAILED -- RESULTS NOT COLLECTED !!!  Do NOT return this Mac."
  echo "       Fix RESULTS_DEST (Remote Login on the 48GB Mac) and rerun, or AirDrop"
  echo "       the runs/ folder by hand. rank1's loss/perplexity live ONLY here."
  echo "############################################################"
  exit 1
fi
echo "============================================================"
