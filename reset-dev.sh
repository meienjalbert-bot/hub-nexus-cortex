#!/usr/bin/env bash
set -euo pipefail

echo "⚠️  This will REMOVE volumes (qdrant_data, meili_data, ollama_models)"
read -r -p "Type RESET to continue: " CONFIRM
if [[ "${CONFIRM:-}" != "RESET" ]]; then
  echo "Aborted."
  exit 1
fi

echo "🔻 docker compose down -v --remove-orphans"
docker compose down -v --remove-orphans

echo "📦 Remaining volumes (should be none for this project):"
docker volume ls | grep -E 'hub-nexus-cortex|qdrant|meili|ollama' || true

echo "✅ Reset done."
