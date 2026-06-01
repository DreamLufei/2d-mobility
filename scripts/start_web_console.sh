#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUNTIME_DIR="${WEB_CONSOLE_RUNTIME_DIR:-$REPO_ROOT/.web_runtime}"
PID_FILE="${WEB_CONSOLE_PID_FILE:-$RUNTIME_DIR/web_console.pid}"
LOG_FILE="${WEB_CONSOLE_LOG_FILE:-$RUNTIME_DIR/web_console.log}"
PYTHON_BIN="${WEB_CONSOLE_PYTHON:-$REPO_ROOT/.venv/bin/python}"
HOST="${WEB_CONSOLE_HOST:-127.0.0.1}"
PORT="${WEB_CONSOLE_PORT:-8765}"
CONSOLE_REPO_ROOT="${WEB_CONSOLE_REPO_ROOT:-$REPO_ROOT}"
SYSTEMD_UNIT="${WEB_CONSOLE_SYSTEMD_UNIT:-script-new-web-console.service}"
SYSTEMD_UNIT_PATH="${HOME}/.config/systemd/user/${SYSTEMD_UNIT}"

mkdir -p "$RUNTIME_DIR"

find_running_pid() {
  ps -eo pid=,args= | awk -v repo="$CONSOLE_REPO_ROOT" -v port="$PORT" '
    index($0, "-m mobility_agent.web_console") && index($0, " --repo-root " repo) && index($0, " --port " port) {
      print $1
      exit
    }
  '
}

wait_for_health() {
  if ! command -v curl >/dev/null 2>&1; then
    sleep 2
    return 0
  fi
  local _attempt
  for _attempt in $(seq 1 20); do
    if curl -fsS "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

if command -v systemctl >/dev/null 2>&1 && [[ -f "$SYSTEMD_UNIT_PATH" ]]; then
  systemctl --user daemon-reload
  systemctl --user start "$SYSTEMD_UNIT" >/dev/null
  PID="$(systemctl --user show --property MainPID --value "$SYSTEMD_UNIT" | tr -d '[:space:]')"
  if ! wait_for_health; then
    echo "web_console systemd service started but health check did not become ready in time" >&2
    systemctl --user status --no-pager "$SYSTEMD_UNIT" >&2 || true
    exit 1
  fi
  LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
  echo "web_console started via systemd"
  echo "unit=$SYSTEMD_UNIT"
  echo "pid=${PID:-unknown}"
  echo "log=$LOG_FILE"
  echo "local_url=http://127.0.0.1:$PORT/"
  if [[ -n "$LAN_IP" ]]; then
    echo "lan_url=http://$LAN_IP:$PORT/"
  fi
  exit 0
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "missing_python: $PYTHON_BIN" >&2
  echo "hint: create the virtualenv and install requirements first." >&2
  exit 1
fi

if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ -n "$EXISTING_PID" ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "web_console already running"
    echo "pid=$EXISTING_PID"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

RUNNING_PID="$(find_running_pid || true)"
if [[ -n "$RUNNING_PID" ]] && kill -0 "$RUNNING_PID" 2>/dev/null; then
  echo "$RUNNING_PID" >"$PID_FILE"
  echo "web_console already running"
  echo "pid=$RUNNING_PID"
  exit 0
fi

declare -a JOB_ROOTS=()

add_job_root() {
  local candidate="$1"
  if [[ -z "$candidate" ]]; then
    return
  fi
  candidate="$(cd -- "$candidate" 2>/dev/null && pwd || printf '%s' "$candidate")"
  for existing in "${JOB_ROOTS[@]:-}"; do
    if [[ "$existing" == "$candidate" ]]; then
      return
    fi
  done
  JOB_ROOTS+=("$candidate")
}

add_job_root "$CONSOLE_REPO_ROOT"

if [[ -n "${WEB_CONSOLE_JOB_ROOTS:-}" ]]; then
  IFS=':' read -r -a EXTRA_JOB_ROOTS <<< "${WEB_CONSOLE_JOB_ROOTS}"
  for candidate in "${EXTRA_JOB_ROOTS[@]}"; do
    add_job_root "$candidate"
  done
fi

declare -a CMD=(
  "$PYTHON_BIN"
  -m
  mobility_agent.web_console
  --host
  "$HOST"
  --port
  "$PORT"
  --repo-root
  "$CONSOLE_REPO_ROOT"
)

for root in "${JOB_ROOTS[@]}"; do
  CMD+=(--job-root "$root")
done

(
  cd -- "$REPO_ROOT"
  nohup "${CMD[@]}" >>"$LOG_FILE" 2>&1 </dev/null &
  echo "$!" >"$PID_FILE"
)

sleep 1
PID="$(tr -d '[:space:]' < "$PID_FILE")"

if [[ -z "$PID" ]] || ! kill -0 "$PID" 2>/dev/null; then
  RUNNING_PID="$(find_running_pid || true)"
  if [[ -n "$RUNNING_PID" ]] && kill -0 "$RUNNING_PID" 2>/dev/null; then
    echo "$RUNNING_PID" >"$PID_FILE"
    PID="$RUNNING_PID"
  fi
fi

if [[ -z "$PID" ]] || ! kill -0 "$PID" 2>/dev/null; then
  echo "web_console failed to start" >&2
  if [[ -f "$LOG_FILE" ]]; then
    tail -n 60 "$LOG_FILE" >&2 || true
  fi
  exit 1
fi

if ! wait_for_health; then
  echo "web_console process started but health check did not become ready in time" >&2
  if [[ -f "$LOG_FILE" ]]; then
    tail -n 60 "$LOG_FILE" >&2 || true
  fi
  exit 1
fi

LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"

echo "web_console started"
echo "pid=$PID"
echo "log=$LOG_FILE"
echo "local_url=http://127.0.0.1:$PORT/"
if [[ -n "$LAN_IP" ]]; then
  echo "lan_url=http://$LAN_IP:$PORT/"
fi
if [[ "$HOST" == "0.0.0.0" ]]; then
  echo "warning=host is 0.0.0.0 and the web console has no built-in authentication"
fi
