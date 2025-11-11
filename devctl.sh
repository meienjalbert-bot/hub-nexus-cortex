#!/usr/bin/env bash
set -euo pipefail

PROFILE_DEFAULT="${PROFILE_DEFAULT:-deps}"
APP_PROFILE="${APP_PROFILE:-cortex}" # changer en "hub" dans marland-nexus-hub
API_HEALTH="${API_HEALTH:-http://localhost:8100/health}" # 8000 pour le hub
MEILI_HEALTH="${MEILI_HEALTH:-http://localhost:7702/health}" # 7701 pour le hub
QDRANT_HEALTH="${QDRANT_HEALTH:-http://localhost:6337/readyz}" # 6336 pour le hub

case "${1:-}" in
  up)
    docker compose --profile deps up -d
    docker compose --profile mesh up -d || true
    docker compose --profile ollama up -d || true
    docker compose --profile "$APP_PROFILE" up -d --build
    ;;
  down)
    docker compose down
    ;;
  reset)
    echo "Tape 'RESET' pour confirmer la suppression des volumes:"
    read -r confirm
    [[ "$confirm" == "RESET" ]] || { echo "Abandon."; exit 1; }
    docker compose down -v
    rm -rf ./data ./volumes 2>/dev/null || true
    ;;
  logs)
    docker compose logs -f --tail=200
    ;;
  health)
    set +e
    curl -fsS "$MEILI_HEALTH" && echo " ✓ Meili"
    curl -fsS "$QDRANT_HEALTH" && echo " ✓ Qdrant"
    curl -fsS "$API_HEALTH" && echo " ✓ API"
    set -e
    ;;
  ingest-folder)
    FOLDER="${2:-./corpus}"
    curl -s -X POST 'http://localhost:8000/api/v1/rag/ingest' \
      -H 'Content-Type: application/json' \
      -d "{\"path\":\"$(realpath "$FOLDER")\"}" | jq .
    ;;
  query)
    Q="${2:-nexus}"
    curl -s "http://localhost:8000/api/v1/rag/query?q=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))" "$Q")&k=5" | jq .
    ;;
  *)
    echo "usage: $0 {up|down|reset|logs|health|ingest-folder [PATH]|query [TEXT]}"
    exit 1
    ;;
esac
