#!/usr/bin/env bash
# =============================================================================
# 2-Mac experiment launcher over grove's TCP transport via SELF-rendezvous,
# bypassing the grove `start`/`join` cli (whose tcp path fails to deliver the
# script to the joiner). Each Mac runs its LOCAL scripts/grove_entry.py; only the
# data plane crosses the wire (validated by scripts/net_check.sh + p2.sh).
#
#   On the 48GB Mac (rank 0 / coordinator), FIRST:
#       ./scripts/run2.sh coord
#   On the 24GB Mac (rank 1 / joiner), within ~1 min:
#       RESULTS_DEST=<user>@192.168.1.130:<repo>/runs/  ./scripts/run2.sh join
#
# Phases (same as run_mac_*.sh): smoke mp_mid dp_mid xl 3b. Run a subset:
#       ./scripts/run2.sh coord mp_mid dp_mid     (same args on both Macs)
# Prereqs: run ./scripts/p2.sh coord|join once first to confirm connectivity.
# =============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROLE="${1:-}"; shift || true
case "$ROLE" in coord|join) ;; *) echo "usage: ./scripts/run2.sh {coord|join} [phases...]"; exit 2 ;; esac
PHASES=("$@"); [[ ${#PHASES[@]} -eq 0 ]] && PHASES=(smoke mp_mid dp_mid xl 3b)
LOG_ROOT="${RUN2_LOG_ROOT:-runs/logs}"
mkdir -p "$LOG_ROOT"

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

echo "[$ROLE] repo: $(pwd)"
echo "[$ROLE] phases: ${PHASES[*]}"
echo "[$ROLE] syncing deps (uv sync)..."
uv sync -q || { echo "[$ROLE] uv sync failed -- is uv installed?"; exit 1; }

# Pre-warm + VERIFY wikitext so both Macs train on the SAME corpus (the run also
# aborts via assert_data_consensus if they differ, but verify early/cheaply).
if printf '%s\n' "${PHASES[@]}" | grep -qvx smoke; then
  echo "[$ROLE] pre-warming + verifying wikitext..."
  uv run python - <<'PY' || { echo "[$ROLE] wikitext NOT real (fallback). Pre-seed data/cache/text/ and retry."; exit 1; }
from macluster.data.text import make_text_task
t = make_text_task(1, variant="wikitext", batch_size=8, seq_len=128, data_dir="data/cache", seed=0)
src = t.meta.get("bpe_source")
print(f"  [data] source={src} tokens={t.meta['n_tokens']}")
import sys
sys.exit(0 if src == "wikitext-2-raw" else 1)
PY
fi

copy_join_results () {
  [[ "$ROLE" == join ]] || return 0
  local dest="${RESULTS_DEST:-}"
  [[ -n "$dest" ]] || return 0
  echo "[join] copying runs/ -> $dest (rsync)..."
  if rsync -az runs/ "$dest"; then
    echo "[join] results copied. Verify on the 48GB Mac: ls runs/*-rank1/metrics.jsonl"
  else
    echo "[join] WARN: rsync failed; will retry at the end."
    return 1
  fi
}

run_phase () {
  local phase="$1" cfg; cfg="$(config_for "$phase")"
  if [[ -z "$cfg" || ! -f "$cfg" ]]; then echo "[$ROLE] unknown/missing phase '$phase' -- skipping"; return; fi
  local log="$LOG_ROOT/run2-${ROLE}-${phase}-$(date +%Y%m%d-%H%M%S).log"
  # Run each phase in a SUBSHELL so its MACLUSTER_*/GROVE_* env CANNOT leak into
  # the next phase. Without this, smoke's MACLUSTER_SYNTHETIC=1 / MACLUSTER_CUT=2
  # persist into mp_mid/dp_mid/xl/3b and silently corrupt them (synthetic random
  # data, vocab=256 instead of 50257, hand cut instead of the memory-aware cut).
  (
    export PYTHONUNBUFFERED=1
    set -a; source "$cfg"; set +a
    export GROVE_TRANSPORT="${GROVE_TRANSPORT:-tcp}"
    export GROVE_N="${N:-2}"
    export GROVE_CLUSTER="${CLUSTER:-macluster}"   # per-phase name (mac_smoke/mac_mp/...)
    export GROVE_TIMEOUT="${GROVE_TIMEOUT:-180.0}"
    export MACLUSTER_RUNS_ROOT="${MACLUSTER_RUNS_ROOT:-runs}"
    [[ "$ROLE" == coord ]] && export GROVE_IS_COORDINATOR=1
    echo "============================================================"
    echo "[$ROLE] PHASE '$phase'  model=${MACLUSTER_MODEL:-?}  synthetic=${MACLUSTER_SYNTHETIC:-0}  cut=${MACLUSTER_CUT:-auto}  cluster=$GROVE_CLUSTER"
    echo "[$ROLE] (coord starts first; joiner discovers within grove's ${GROVE_TIMEOUT}s window)"
    echo "[$ROLE] phase log: $log"
    echo "============================================================"
    uv run python -u scripts/grove_entry.py
  ) 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  if [[ $rc -eq 0 ]]; then echo "[$ROLE] PHASE '$phase' DONE"
  else echo "[$ROLE] PHASE '$phase' FAILED (rc=$rc) -- continuing to next phase."; fi
  if [[ "$ROLE" == join ]]; then copy_join_results || true; fi
  sleep 3
}

for p in "${PHASES[@]}"; do run_phase "$p"; done

echo "============================================================"
if [[ "$ROLE" == coord ]]; then
  IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo '<this-mac-ip>')"
  echo "[coord] all phases finished. rank0 metrics: runs/*-rank0/{metrics.jsonl,summary.json}"
  echo "[coord] collect rank1 (MP loss/ppl) from the 24GB Mac. Enable Remote Login here, then on the 24GB Mac:"
  echo "          RESULTS_DEST=$(whoami)@$IP:$(pwd)/runs/  ./scripts/run2.sh join"
else
  DEST="${RESULTS_DEST:-}"
  if [[ -z "$DEST" ]]; then
    echo "############################################################"
    echo "[join] !!! RESULTS NOT COLLECTED !!! rank1 holds the ONLY copy of the MP"
    echo "       loss/perplexity. Do NOT return this Mac. Set RESULTS_DEST (the coord"
    echo "       script prints it) and rerun, or AirDrop the runs/ folder."
    echo "############################################################"
    exit 1
  fi
  if ! copy_join_results; then
    echo "############################################################"
    echo "[join] !!! rsync FAILED -- RESULTS NOT COLLECTED !!! Do NOT return this Mac."
    echo "       Fix RESULTS_DEST (Remote Login on the 48GB Mac) and rerun, or AirDrop runs/."
    echo "############################################################"
    exit 1
  fi
fi
echo "============================================================"
