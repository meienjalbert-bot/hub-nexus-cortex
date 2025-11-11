# NEW: pour exécuter la fonction sync d’ingest sans bloquer l’event loop

# Tes imports existants

# NEW: on tente d’importer le module d’ingestion (scripts/ingest.py)
try:
    from scripts import ingest as ingest_mod
except Exception:
    ingest_mod = None
