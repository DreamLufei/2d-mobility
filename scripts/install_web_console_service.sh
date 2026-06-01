#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${WEB_CONSOLE_PYTHON:-$REPO_ROOT/.venv/bin/python}"
HOST="${WEB_CONSOLE_HOST:-127.0.0.1}"
PORT="${WEB_CONSOLE_PORT:-8765}"
CONSOLE_REPO_ROOT="${WEB_CONSOLE_REPO_ROOT:-$REPO_ROOT}"
SYSTEMD_UNIT="${WEB_CONSOLE_SYSTEMD_UNIT:-script-new-web-console.service}"
SYSTEMD_UNIT_PATH="${HOME}/.config/systemd/user/${SYSTEMD_UNIT}"
ENABLE_ON_INSTALL="${WEB_CONSOLE_ENABLE_ON_INSTALL:-0}"
START_ON_INSTALL="${WEB_CONSOLE_START_ON_INSTALL:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "missing_python: $PYTHON_BIN" >&2
  echo "hint=run ./scripts/bootstrap_cluster.sh first" >&2
  exit 1
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

mkdir -p "${HOME}/.config/systemd/user"

CMD="$PYTHON_BIN -m mobility_agent.web_console --host $HOST --port $PORT --repo-root $CONSOLE_REPO_ROOT"
for root in "${JOB_ROOTS[@]}"; do
  CMD+=" --job-root $root"
done

cat >"$SYSTEMD_UNIT_PATH" <<EOF
[Unit]
Description=script_new web console
After=network.target

[Service]
Type=simple
WorkingDirectory=$REPO_ROOT
ExecStart=$CMD
Restart=on-failure
RestartSec=2
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload

if [[ "$ENABLE_ON_INSTALL" == "1" ]]; then
  systemctl --user enable "$SYSTEMD_UNIT" >/dev/null
fi

if [[ "$START_ON_INSTALL" == "1" ]]; then
  systemctl --user start "$SYSTEMD_UNIT" >/dev/null
fi

echo "service_file=$SYSTEMD_UNIT_PATH"
echo "host=$HOST"
echo "port=$PORT"
printf 'job_roots='
printf '%s:' "${JOB_ROOTS[@]}"
printf '\n'
if [[ "$HOST" == "0.0.0.0" ]]; then
  echo "warning=host is 0.0.0.0 and the web console has no built-in authentication"
fi
echo "next_steps:"
echo "  systemctl --user start $SYSTEMD_UNIT"
echo "  systemctl --user status $SYSTEMD_UNIT"
echo "  systemctl --user disable $SYSTEMD_UNIT"
