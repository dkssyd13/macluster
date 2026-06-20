#!/usr/bin/env bash
# =============================================================================
# Two-Mac LAN connectivity check WITHOUT grove. Isolates whether grove's
# "connection reset by peer" is the macOS "Local Network" privacy permission
# (or a firewall/filter) blocking a terminal-launched python's LAN TCP, while
# system ping/mDNS still work. Tests the SAME direction grove uses: rank0 (48GB)
# connects to rank1 (24GB) on port 29501.
#
#   1) On the BORROWED 24GB Mac:   ./scripts/net_check.sh server
#        -> prints diagnostics + this Mac's IP, then listens on 29501
#   2) On the 48GB Mac:            ./scripts/net_check.sh client <24GB-IP>
#        -> connects and prints PASS/FAIL with the fix
#
# First run may pop "<Terminal> wants to find devices on your local network" ->
# click ALLOW. If it FAILS with no popup: System Settings > Privacy & Security >
# Local Network > enable your terminal app, fully quit & reopen Terminal, retry.
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"; cd "$ROOT"
PORT=29501
MODE="${1:-}"

diag () {
  echo "[net_check] ---- local diagnostics ----"
  echo -n "[net_check] firewall: "; /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate 2>/dev/null || echo "?"
  uv run python -c "from grove._utils import get_local_ip; print('[net_check] grove advertises IP:', get_local_ip())" 2>/dev/null || true
  echo "[net_check] IPv4 interfaces:"; ifconfig 2>/dev/null | grep -w inet | grep -v 127.0.0.1
  echo "[net_check] ---------------------------"
}

case "$MODE" in
  server)
    diag
    MYIP="$(uv run python -c "from grove._utils import get_local_ip; print(get_local_ip())" 2>/dev/null)"
    echo "[net_check] ============================================"
    echo "[net_check]  THIS Mac's IP = $MYIP"
    echo "[net_check]  On the OTHER (48GB) Mac run:"
    echo "[net_check]      ./scripts/net_check.sh client $MYIP"
    echo "[net_check]  >>> if a popup asks to allow local network access: click ALLOW <<<"
    echo "[net_check] ============================================"
    uv run python - "$PORT" <<'PY'
import socket, sys
p = int(sys.argv[1])
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("0.0.0.0", p)); s.listen(1)
print(f"[net_check] LISTENING on {p} -- now run the client on the 48GB Mac (Ctrl-C to stop)...", flush=True)
c, a = s.accept(); data = c.recv(16); c.send(b"pong")
print(f"[net_check] PASS: accepted from {a}, recv={data!r}. This Mac CAN receive LAN TCP.")
PY
    ;;
  client)
    PEER="${2:-}"
    if [[ -z "$PEER" ]]; then echo "[net_check] usage: ./scripts/net_check.sh client <24GB-Mac-IP>"; exit 2; fi
    diag
    echo "[net_check] connecting to $PEER:$PORT ..."
    echo "[net_check] >>> if a popup asks to allow local network access: click ALLOW <<<"
    uv run python - "$PEER" "$PORT" <<'PY'
import socket, sys
peer, p = sys.argv[1], int(sys.argv[2])
try:
    s = socket.socket(); s.settimeout(6)
    s.connect((peer, p)); s.send(b"ping")
    print(f"[net_check] PASS: connected to {peer}:{p}, got {s.recv(16)!r}.")
    print("[net_check] LAN python TCP works -> grove WILL work. Next: GROVE_TRANSPORT=tcp ping.")
except Exception as e:
    print(f"[net_check] FAIL: {type(e).__name__}: {e}")
    print("[net_check] -> a terminal-launched python CANNOT reach the LAN peer.")
    print("[net_check]    Almost always macOS 'Local Network' permission. Fix on BOTH Macs:")
    print("[net_check]    System Settings > Privacy & Security > Local Network > enable Terminal/iTerm,")
    print("[net_check]    then FULLY quit & reopen the terminal and rerun this.")
    sys.exit(1)
PY
    ;;
  *)
    echo "usage:"
    echo "  on the borrowed 24GB Mac:  ./scripts/net_check.sh server"
    echo "  on the 48GB Mac:           ./scripts/net_check.sh client <24GB-IP>"
    exit 2
    ;;
esac
