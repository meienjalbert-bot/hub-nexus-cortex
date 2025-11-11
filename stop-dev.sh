#!/usr/bin/env bash
set -euo pipefail
echo "⛔ Stopping stack…"
docker compose stop
echo
docker compose ps
