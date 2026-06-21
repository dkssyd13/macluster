#!/usr/bin/env bash
# Convenience launcher for the 24GB Mac (rank 1).
#
# Defaults are for the current two-Mac setup:
#   rank0 / 48GB: 192.168.1.130
#   rank1 / 24GB: 192.168.1.97
#   rank0 repo:   /Users/vladkim/Personal/CAU/26-1/ACN/project/macluster
#
# Override only if the Wi-Fi IPs or rank0 repo path change:
#   MACLUSTER_COORD_IP=... MACLUSTER_JOIN_IP=... MACLUSTER_COORD_REPO=... ./scripts/run2_join.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

COORD_IP="${MACLUSTER_COORD_IP:-192.168.1.130}"
JOIN_IP="${MACLUSTER_JOIN_IP:-192.168.1.97}"
COORD_USER="${MACLUSTER_COORD_USER:-vladkim}"
COORD_REPO="${MACLUSTER_COORD_REPO:-/Users/vladkim/Personal/CAU/26-1/ACN/project/macluster}"

export GROVE_PEERS="${GROVE_PEERS:-$COORD_IP,$JOIN_IP}"
export GROVE_TIMEOUT="${GROVE_TIMEOUT:-600}"
export RESULTS_DEST="${RESULTS_DEST:-$COORD_USER@$COORD_IP:$COORD_REPO/runs/}"

PHASES=("$@")
if [[ ${#PHASES[@]} -eq 0 ]]; then
  PHASES=(mp_mid dp_mid xl 3b)
fi

echo "[run2_join] GROVE_PEERS=$GROVE_PEERS"
echo "[run2_join] RESULTS_DEST=$RESULTS_DEST"
echo "[run2_join] phases: ${PHASES[*]}"
exec ./scripts/run2.sh join "${PHASES[@]}"
