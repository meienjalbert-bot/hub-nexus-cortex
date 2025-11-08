import os, httpx
from typing import Dict, Any
from .memory_fusion import rrf_merge, dedup

QDRANT_HOST = os.getenv("QDRANT_HOST", "http://qdrant:6333").rstrip("/")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "nexus_docs")
MEILI_HOST = os.getenv("MEILI_HOST", "http://meili:7700").rstrip("/")
MEILI_INDEX = os.getenv("MEILI_INDEX", "docs")

async def search_semantic(query: str, k: int = 5):
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.post(f"{QDRANT_HOST}/collections/{QDRANT_COLLECTION}/points/search",
                                  json={"vector": [0.0,0.0,0.0,0.0], "limit": k})
            pts = r.json().get("result", [])
            return [{"text": p.get("payload", {}).get("text", ""), "score": p.get("score"), "expert":"semantic"} for p in pts]
    except Exception:
        return []

async def search_lexical(query: str, k: int = 5):
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.post(f"{MEILI_HOST}/indexes/{MEILI_INDEX}/search", json={"q": query, "limit": k})
            hits = r.json().get("hits", [])
            return [{"text": h.get("text") or h.get("content") or "", "score": 1.0, "expert":"lexical"} for h in hits]
    except Exception:
        return []

async def route(query: str, k: int = 5) -> Dict[str, Any]:
    buckets = {
        "lexical": await search_lexical(query, k),
        "semantic": await search_semantic(query, k),
    }
    fused = rrf_merge(buckets, k=60)
    fused = dedup(fused)[:k]
    explain = {k: len(v) for k,v in buckets.items()}
    return {"results": fused, "explain": explain}
