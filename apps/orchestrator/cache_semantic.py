# apps/orchestrator/cache_semantic.py
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import redis
from sentence_transformers import SentenceTransformer

# ------------------ Config ------------------ #
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "3600"))  # 1h
THRESH = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.93"))
MAX_SCAN = int(os.getenv("MAX_CACHE_SCAN", "200"))

# ------------------ Singletons ------------------ #
_redis_client: Optional[redis.Redis] = None
_embedder: Optional[SentenceTransformer] = None


def _get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        # Multilingue, léger et déjà packagé dans sentence-transformers
        _embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _embedder


def _keyspace() -> str:
    return "cache"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)


# --------------- API fonctionnelle --------------- #
def get_from_cache(query: str) -> Dict[str, Any]:
    """
    Cherche une entrée sémantiquement proche de `query` dans Redis.
    Retourne {"hit": bool, "answer": str, "sources": list, "cosine": float} si hit.
    """
    try:
        r = _get_redis()
        emb = _get_embedder().encode([query])[0].astype(np.float32)

        best_score: float = -1.0
        best_val: Optional[Dict[str, Any]] = None

        # Balayage simple (MVP) ; on pourra passer à Redis-Vector/RediSearch plus tard
        count = 0
        for key in r.scan_iter(f"{_keyspace()}:*"):
            if count >= MAX_SCAN:
                break
            count += 1

            data = r.get(key)
            if not data:
                continue

            rec = json.loads(data)
            if "embedding" not in rec:
                continue

            cached_vec = np.array(rec["embedding"], dtype=np.float32)
            score = _cosine(emb, cached_vec)

            if score > best_score:
                best_score = score
                best_val = rec

        if best_val is not None and best_score >= THRESH:
            return {
                "hit": True,
                "answer": best_val.get("answer"),
                "sources": best_val.get("sources", []),
                "cosine": float(best_score),
            }

        return {"hit": False}
    except Exception as e:
        # On ne bloque jamais la route sur un échec cache
        return {"hit": False, "error": str(e)}


def set_in_cache(query: str, answer: str, sources: List[Dict[str, Any]]) -> None:
    """
    Stocke la réponse et son embedding.
    """
    try:
        r = _get_redis()
        vec = _get_embedder().encode([query])[0].astype(np.float32).tolist()

        key = f"{_keyspace()}:{hashlib.sha256(query.encode()).hexdigest()[:16]}"
        payload = {
            "query": query,
            "answer": answer,
            "sources": sources,
            "embedding": vec,
        }
        r.setex(key, CACHE_TTL, json.dumps(payload))
    except Exception:
        # best-effort
        pass


# --------------- API orientée objet (tests/usage futur) --------------- #
class SemanticCache:
    def __init__(
        self,
        redis_url: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        threshold: Optional[float] = None,
        max_scan: Optional[int] = None,
    ) -> None:
        self.redis_url = redis_url or REDIS_URL
        self.ttl = ttl_seconds or CACHE_TTL
        self.threshold = threshold or THRESH
        self.max_scan = max_scan or MAX_SCAN

    def get(self, query: str) -> Dict[str, Any]:
        # Délègue aux fonctions globales, MVP
        return get_from_cache(query)

    def set(self, query: str, answer: str, sources: List[Dict[str, Any]]) -> None:
        set_in_cache(query, answer, sources)
