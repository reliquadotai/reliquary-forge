#!/usr/bin/env bash
set -euo pipefail

STATE_ROOT="${STATE_ROOT:-/srv/reliquary}"
timestamp="$(date -u +%Y%m%d%H%M%S)"
archive_path="${STATE_ROOT}/backups/state-${timestamp}.tar.gz"

mkdir -p "${STATE_ROOT}/backups"
tar -C "${STATE_ROOT}" -czf "${archive_path}" state
find "${STATE_ROOT}/backups" -type f -name 'state-*.tar.gz' -mtime +7 -delete

echo "Wrote ${archive_path}"
