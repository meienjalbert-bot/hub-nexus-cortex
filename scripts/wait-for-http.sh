#!/usr/bin/env bash
set -euo pipefail
URL="${1:-http://localhost:8100/health}"
TIMEOUT="${2:-60}"
until curl -fsS "$URL" >/dev/null 2>&1; do
  ((TIMEOUT--)) || { echo "Timeout waiting for $URL"; exit 1; }
  sleep 1
done
echo "OK: $URL"
