#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUNTIME_DIR="${WEB_CONSOLE_RUNTIME_DIR:-$REPO_ROOT/.web_runtime}"
PID_FILE="${WEB_CONSOLE_PID_FILE:-$RUNTIME_DIR/web_console.pid}"
LOG_FILE="${WEB_CONSOLE_LOG_FILE:-$RUNTIME_DIR/web_console.log}"
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
    PID="$(systemctl --user show --property MainPID --value "$SYSTEMD_UNIT" | tr -d '[:space:]')"
    echo "status=running"
    echo "unit=$SYSTEMD_UNIT"
    echo "pid=${PID:-unknown}"
  else
    echo "status=stopped"
    echo "unit=$SYSTEMD_UNIT"
  fi
  echo "log=$LOG_FILE"
  echo "listen_matches:"
  ss -tulpn 2>/dev/null | grep ":$PORT" || true
  if command -v curl >/dev/null 2>&1; then
    echo "health:"
    curl -fsS "http://127.0.0.1:$PORT/api/health" || true
    printf '\n'
  fi
  exit 0
fi

if [[ -f "$PID_FILE" ]]; then
  PID="$(tr -d '[:space:]' < "$PID_FILE")"
else
  PID=""
fi

if [[ -z "$PID" ]] || ! kill -0 "$PID" 2>/dev/null; then
  PID="$(find_running_pid || true)"
  if [[ -n "$PID" ]]; then
    mkdir -p "$RUNTIME_DIR"
    echo "$PID" >"$PID_FILE"
  fi
fi

if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
  echo "status=running"
  echo "pid=$PID"
else
  echo "status=stopped"
  if [[ -n "$PID" ]]; then
    echo "stale_pid=$PID"
  fi
fi

echo "log=$LOG_FILE"
echo "listen_matches:"
ss -tulpn 2>/dev/null | grep ":$PORT" || true

if command -v curl >/dev/null 2>&1; then
  echo "health:"
  curl -fsS "http://127.0.0.1:$PORT/api/health" || true
  printf '\n'
fi
