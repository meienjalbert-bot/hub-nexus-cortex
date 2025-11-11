# Dev B — Ingestion & Query (RRF + MMR)

Scripts prêts pour **ingestion** (Qdrant + Meili) et **requête hybride** (RRF adaptative + MMR optionnel).

## Install

```bash
python -m pip install httpx PyMuPDF numpy pyyaml
```

## Env

```env
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=nexus_docs
MEILI_URL=http://localhost:7700
MEILI_MASTER_KEY=meili_key
MEILI_INDEX=docs
OLLAMA_HOST=http://localhost:11434
EMBED_MODEL=nomic-embed-text
```

## Ingest

```bash
python scripts/ingest.py --path ./corpus --collection nexus_docs --index docs   --chunk_chars 2000 --overlap 200 --batch 64
```

## Query

```bash
python scripts/query.py --q "langgraph vs workflow" --k 5
python scripts/query.py --q "langgraph" --k 5 --mmr 0.3
```

Sortie JSON: `{ results: [...], explain: {...} }`
