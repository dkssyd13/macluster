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

detect_ip () {
  ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true
}

COORD_IP="${MACLUSTER_COORD_IP:-$(detect_ip)}"
JOIN_IP="${MACLUSTER_JOIN_IP:-192.168.1.97}"

if [[ -z "$COORD_IP" ]]; then
  echo "[run2_coord] could not detect this Mac's Wi-Fi IP. Reconnect Wi-Fi and retry,"
  echo "             or pass MACLUSTER_COORD_IP=<this-mac-ip>."
  exit 1
fi

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
