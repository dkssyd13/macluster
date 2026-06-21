#!/usr/bin/env bash
# Convenience launcher for the 48GB Mac (rank 0).
#
# Defaults are for the current two-Mac setup:
#   rank0 / 48GB: 192.168.1.130
#   rank1 / 24GB: 192.168.1.97
#
# Override only if the Wi-Fi IPs change:
#   MACLUSTER_COORD_IP=... MACLUSTER_JOIN_IP=... ./scripts/run2_coord.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

COORD_IP="${MACLUSTER_COORD_IP:-192.168.1.130}"
JOIN_IP="${MACLUSTER_JOIN_IP:-192.168.1.97}"

export GROVE_PEERS="${GROVE_PEERS:-$COORD_IP,$JOIN_IP}"
export GROVE_TIMEOUT="${GROVE_TIMEOUT:-600}"
export GROVE_SOCKET_TIMEOUT="${GROVE_SOCKET_TIMEOUT:-600}"

PHASES=("$@")
if [[ ${#PHASES[@]} -eq 0 ]]; then
  PHASES=(mp_mid dp_mid xl 3b)
fi

echo "[run2_coord] GROVE_PEERS=$GROVE_PEERS"
echo "[run2_coord] phases: ${PHASES[*]}"
exec ./scripts/run2.sh coord "${PHASES[@]}"
