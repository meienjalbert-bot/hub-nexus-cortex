# Debug rapide — hub-nexus-cortex

## 1) Process & ports
docker compose ps
ss -lntp | grep -E '6337|7702|8100|11434|4223|6380' || true

## 2) Health endpoints
curl -fsS http://localhost:7702/health    # Meili (CORTEX)
curl -fsS http://localhost:6337/readyz    # Qdrant (CORTEX)
curl -fsS http://localhost:8100/health    # Cortex API

## 3) Logs ciblés
docker compose logs meili   --tail=200 | tail -n +1
docker compose logs qdrant  --tail=200 | tail -n +1
docker compose logs cortex  --tail=200 | tail -n +1

## 4) ENV critiques
# Meili
echo "MEILI_MASTER_KEY=${MEILI_MASTER_KEY:-<empty>}"
# Qdrant: pas de clé par défaut; vérifier mapping port 6337:6333

## 5) Ingestion (pièges fréquents)
- Le chemin passé à /rag/ingest doit être **visible depuis le conteneur** qui fait l’ingestion.
  - Si l’ingestion est faite par le service `hub`/`cortex`, monte le dossier hôte en volume.
- Si 401/403 Meili: vérifier MEILI_MASTER_KEY côté service et côté client.
- Si Qdrant unhealthy: conflit de port ou volume corrompu → `reset-dev.sh`.
