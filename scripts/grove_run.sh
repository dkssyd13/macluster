#!/usr/bin/env bash
# grove_run.sh -- launch a 2-MacBook macluster run (Phase 7) with one command.
#
#   mac-A (launcher):  ./scripts/grove_run.sh start configs/grove/diloco.env
#   mac-B (joiner):    ./scripts/grove_run.sh join  configs/grove/diloco.env
#
# Both Macs MUST pass the SAME committed config file: it is sourced into the
# MACLUSTER_* env that scripts/grove_entry.py reads, and since the file is the
# same bytes on both repos there is no env-mismatch (grove_entry's
# config-consensus check would otherwise abort the run).
#
# Pre-flight:  ./scripts/grove_run.sh check          # is the peer discoverable?
# No 2nd Mac:  ./scripts/grove_run.sh smoke          # single-machine W=1 smoke
set -euo pipefail

usage() {
  cat <<'EOF'
usage: scripts/grove_run.sh <start|join|smoke|check> [config.env]

  start [CONFIG]   start the cluster on this Mac (coordinator) and run a rank
  join  [CONFIG]   join the named cluster from this Mac
  smoke [CONFIG]   single-machine W=1 smoke (no second Mac needed)
  check            `grove status` -- verify the peer Mac is visible

CONFIG defaults: start/join -> configs/grove/diloco.env, smoke -> configs/grove/smoke.env
Use the SAME config file on both Macs.
EOF
}

# Run from the repo root regardless of where the script is invoked.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

ROLE="${1:-}"
CONFIG_ARG="${2:-}"

load_config() {
  local cfg="${CONFIG_ARG:-$1}"
  if [[ ! -f "$cfg" ]]; then
    echo "[grove_run] config not found: $cfg" >&2
    exit 1
  fi
  set -a
  # shellcheck disable=SC1090
  source "$cfg"
  set +a
  CLUSTER="${CLUSTER:-macluster}"
  N="${N:-2}"
  echo "[grove_run] config=$cfg cluster=$CLUSTER N=$N" \
       "task=${MACLUSTER_TASK:-?} algo=${MACLUSTER_ALGORITHM:-?}" \
       "rounds=${MACLUSTER_ROUNDS:-?} max_steps=${MACLUSTER_MAX_STEPS:-?}" \
       "H=${MACLUSTER_H:-?} link=${MACLUSTER_LINK:-?} synthetic=${MACLUSTER_SYNTHETIC:-0}"
}

case "$ROLE" in
  start)
    load_config configs/grove/diloco.env
    exec uv run grove start scripts/grove_entry.py -n "$N" --name "$CLUSTER" --logs
    ;;
  join)
    load_config configs/grove/diloco.env
    exec uv run grove join "$CLUSTER" --logs
    ;;
  smoke)
    load_config configs/grove/smoke.env
    exec uv run python scripts/grove_entry.py
    ;;
  check)
    exec uv run grove status
    ;;
  ""|-h|--help|help)
    usage
    [[ "$ROLE" == "" ]] && exit 1 || exit 0
    ;;
  *)
    echo "[grove_run] unknown role: $ROLE" >&2
    usage
    exit 1
    ;;
esac
