#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUNTIME_DIR="${WEB_CONSOLE_RUNTIME_DIR:-$REPO_ROOT/.web_runtime}"
PID_FILE="${WEB_CONSOLE_PID_FILE:-$RUNTIME_DIR/web_console.pid}"
PORT="${WEB_CONSOLE_PORT:-8765}"
CONSOLE_REPO_ROOT="${WEB_CONSOLE_REPO_ROOT:-$REPO_ROOT}"
SYSTEMD_UNIT="${WEB_CONSOLE_SYSTEMD_UNIT:-script-new-web-console.service}"
SYSTEMD_UNIT_PATH="${HOME}/.config/systemd/user/${SYSTEMD_UNIT}"

find_running_pid() {
  ps -eo pid=,args= | awk -v repo="$CONSOLE_REPO_ROOT" -v port="$PORT" '
    index($0, "-m mobility_agent.web_console") && index($0, " --repo-root " repo) && index($0, " --port " port) {
      print $1
      exit
    }
  '
}

if command -v systemctl >/dev/null 2>&1 && [[ -f "$SYSTEMD_UNIT_PATH" ]]; then
  if systemctl --user is-active --quiet "$SYSTEMD_UNIT"; then
    systemctl --user stop "$SYSTEMD_UNIT"
    rm -f "$PID_FILE"
    echo "web_console stopped via systemd"
  else
    rm -f "$PID_FILE"
    echo "web_console not running"
  fi
  exit 0
fi

if [[ ! -f "$PID_FILE" ]]; then
  PID="$(find_running_pid || true)"
  if [[ -z "$PID" ]]; then
    echo "web_console not running"
    exit 0
  fi
else
  PID="$(tr -d '[:space:]' < "$PID_FILE")"
fi

if [[ -z "$PID" ]]; then
  rm -f "$PID_FILE"
  echo "removed empty pid file"
  exit 0
fi

if ! kill -0 "$PID" 2>/dev/null; then
  DISCOVERED_PID="$(find_running_pid || true)"
  if [[ -n "$DISCOVERED_PID" ]] && kill -0 "$DISCOVERED_PID" 2>/dev/null; then
    PID="$DISCOVERED_PID"
  else
    rm -f "$PID_FILE"
    echo "removed stale pid file for pid=$PID"
    exit 0
  fi
fi

kill "$PID" 2>/dev/null || true

for _ in $(seq 1 20); do
if ! kill -0 "$PID" 2>/dev/null; then
  rm -f "$PID_FILE"
  echo "web_console stopped"
  exit 0
fi
  sleep 0.5
done

kill -9 "$PID" 2>/dev/null || true
rm -f "$PID_FILE"
echo "web_console killed"
