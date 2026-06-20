#!/usr/bin/env bash
# =============================================================================
# Self-rendezvous 2-Mac launcher over grove's TCP transport, BYPASSING the grove
# `start`/`join` cli (whose tcp path fails to deliver the script to the joiner).
# Each Mac runs its LOCAL copy of the script -- nothing crosses the wire except
# the data plane (proven working by scripts/net_check.sh).
#
#   On the 48GB Mac (rank 0 / coordinator):
#       ./scripts/p2.sh coord [script]
#   On the 24GB Mac (rank 1 / joiner), within ~1 min:
#       ./scripts/p2.sh join  [script]
#
# Defaults: GROVE_TRANSPORT=tcp, GROVE_CLUSTER=mac2, GROVE_N=2,
#           script=scripts/grove_selftest.py (the connectivity test).
# Override any with env, e.g.:  GROVE_CLUSTER=mac_mp ./scripts/p2.sh coord scripts/grove_entry.py
# =============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROLE="${1:-}"; SCRIPT="${2:-scripts/grove_selftest.py}"
export GROVE_TRANSPORT="${GROVE_TRANSPORT:-tcp}"
export GROVE_CLUSTER="${GROVE_CLUSTER:-mac2}"
export GROVE_N="${GROVE_N:-2}"

case "$ROLE" in
  coord) export GROVE_IS_COORDINATOR=1 ;;
  join)  unset GROVE_IS_COORDINATOR || true ;;
  *) echo "usage: ./scripts/p2.sh {coord|join} [script]"; exit 2 ;;
esac

echo "[p2] role=$ROLE transport=$GROVE_TRANSPORT cluster=$GROVE_CLUSTER N=$GROVE_N script=$SCRIPT"
exec uv run python "$SCRIPT"
