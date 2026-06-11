#!/usr/bin/env bash
# =============================================================================
# RUN THIS ON THE 24GB MAC.  (the JOINER = rank 1 = the SMALLER pipeline stage.)
# Start scripts/run_mac_48gb.sh on the 48GB Mac FIRST, then run this within ~2 min.
#
#   1) On the 48GB Mac:  ./scripts/run_mac_48gb.sh
#   2) On the 24GB Mac:  ./scripts/run_mac_24gb.sh   <-- this script
#
# Same three phases, in the same order, as the 48GB script. Each phase retries the
# join for a while so it doesn't matter who reaches the phase first.
#
# Run a subset by naming phases:  ./scripts/run_mac_24gb.sh xl 3b
# (run the SAME phase args as the 48GB Mac.)
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PHASES=("$@"); [[ ${#PHASES[@]} -eq 0 ]] && PHASES=(smoke xl 3b)

config_for () {
  case "$1" in
    smoke) echo configs/grove/pipeline_smoke.env ;;
    xl)    echo configs/grove/pipeline_xl.env ;;
    3b)    echo configs/grove/pipeline_3b.env ;;
    *)     echo "" ;;
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
echo "[24GB] all requested phases finished. Metrics under runs/  (this Mac = rank1 = stage1)."
