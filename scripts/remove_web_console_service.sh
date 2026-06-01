#!/usr/bin/env bash
set -euo pipefail

SYSTEMD_UNIT="${WEB_CONSOLE_SYSTEMD_UNIT:-script-new-web-console.service}"
SYSTEMD_UNIT_PATH="${HOME}/.config/systemd/user/${SYSTEMD_UNIT}"

if command -v systemctl >/dev/null 2>&1; then
  systemctl --user stop "$SYSTEMD_UNIT" >/dev/null 2>&1 || true
  systemctl --user disable "$SYSTEMD_UNIT" >/dev/null 2>&1 || true
  systemctl --user daemon-reload >/dev/null 2>&1 || true
fi

if [[ -f "$SYSTEMD_UNIT_PATH" ]]; then
  rm -f "$SYSTEMD_UNIT_PATH"
fi

if command -v systemctl >/dev/null 2>&1; then
  systemctl --user daemon-reload >/dev/null 2>&1 || true
fi

echo "removed_unit=$SYSTEMD_UNIT_PATH"
