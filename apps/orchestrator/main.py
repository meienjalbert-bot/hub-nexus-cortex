# apps/orchestrator/main.py
import os
import time
from typing import Any, Dict, Optional

import httpx
# ✅ Import plat (advanced_health.py doit être dans le même dossier que main.py une fois copié dans l'image)
from advanced_health import get_advanced_health
from advanced_health import router as health_router
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
# Prometheus
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Histogram,
                               generate_latest)
from pydantic import BaseModel

# ===================== FastAPI APP ======================
app = FastAPI(title="Nexus Cortex Orchestrator", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health_router, prefix="/health", tags=["health"])

# =================== Configuration env ==================
MEILI_HOST = os.getenv("MEILI_HOST", "http://meili:7700")
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://qdrant:6333")
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
ENABLE_CACHE = os.getenv("ENABLE_SEMANTIC_CACHE", "1") == "1"

# ===================== Metrics ==========================
REQ_LAT_ROUTE = Histogram(
    "cortex_route_latency_seconds", "Latency of /route", ["cache_state"]
)
CACHE_HIT = Counter("cortex_cache_hits_total", "Semantic cache hits")
CACHE_MISS = Counter("cortex_cache_miss_total", "Semantic cache misses")
ROUTE_ERRORS = Counter("cortex_route_errors_total", "Errors in /route")
VOTE_ERRORS = Counter("cortex_vote_errors_total", "Errors in /vote")

# ===================== Imports optionnels =========================
# Cache sémantique (optionnel)
try:
    from cache_semantic import get_from_cache, set_in_cache  # type: ignore
except Exception:
    get_from_cache = None  # type: ignore
    set_in_cache = None  # type: ignore

# MoME router (optionnel)
try:
    from core.mome_router import run_mome as _mome_run  # type: ignore

    _MOME_AVAILABLE = True
except Exception:
    _MOME_AVAILABLE = False
    _mome_run = None  # type: ignore

# Consensus (optionnel, async)
try:
    from core.multi_llm_voting import vote as consensus_vote  # type: ignore

    _CONS_AVAILABLE = True
except Exception:
    _CONS_AVAILABLE = False
    consensus_vote = None  # type: ignore


# ===================== Utils ==========================
async def _check(url: str, timeout: float = 1.5) -> bool:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url)
            return r.status_code < 400
    except Exception:
        return False


def _ensure_mome_available():
    if not _MOME_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="MoME pipeline indisponible (core/mome_router.run_mome).",
        )


def _ensure_consensus_available():
    if not _CONS_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Consensus indisponible (core/multi_llm_voting.vote).",
        )


# ====================== Schemas =======================
class VoteRequest(BaseModel):
    prompt: str
    context: Optional[str] = ""
    mode: Optional[str] = "precision"  # "precision" (32B) ou "interactive"


# ====================== Endpoints =====================
@app.get("/health")
async def health() -> Dict[str, Any]:
    meili_ok = await _check(f"{MEILI_HOST}/health")
    qdrant_ok = await _check(f"{QDRANT_HOST}/readyz")
    ollama_ok = await _check(f"{OLLAMA_BASE}/api/tags")
    cache_ok = get_from_cache is not None and set_in_cache is not None

    advanced = await get_advanced_health()
    return {
        "status": "ok" if (meili_ok and qdrant_ok) else "degraded",
        "deps": {
            "meili": {"url": MEILI_HOST, "ok": meili_ok},
            "qdrant": {"url": QDRANT_HOST, "ok": qdrant_ok},
            "ollama": {"url": OLLAMA_BASE, "ok": ollama_ok},
            "semantic_cache": {"enabled": ENABLE_CACHE, "ok": cache_ok},
        },
        "mome_available": _MOME_AVAILABLE,
        "consensus_available": _CONS_AVAILABLE,
        "suggested_mode": advanced.get("suggested_mode", "interactive"),
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/route")
async def route(q: str = Query(..., min_length=1), k: int = 5) -> Dict[str, Any]:
    """
    Pipeline : (1) cache sémantique → (2) MoME + génération → (3) cache set
    """
    t0 = time.time()

    # 1) Semantic cache
    if ENABLE_CACHE and get_from_cache is not None:
        try:
            cached = get_from_cache(q)
            if cached and cached.get("hit"):
                CACHE_HIT.inc()
                REQ_LAT_ROUTE.labels(cache_state="hit").observe(time.time() - t0)
                return {
                    "query": q,
                    "k": k,
                    "cache": {"hit": True, "cosine": cached.get("cosine")},
                    "answer": cached.get("answer"),
                    "sources": cached.get("sources", []),
                }
        except Exception:
            pass

    CACHE_MISS.inc()

    # 2) MoME + génération
    _ensure_mome_available()
    try:
        result = _mome_run(q, k)  # type: ignore
        if not isinstance(result, dict) or "answer" not in result:
            raise RuntimeError("Format de retour MoME invalide.")
    except Exception as e:
        ROUTE_ERRORS.inc()
        raise HTTPException(status_code=500, detail=f"Erreur MoME: {e}")

    # 3) Set cache (best-effort)
    if ENABLE_CACHE and set_in_cache is not None:
        try:
            set_in_cache(q, result.get("answer", ""), result.get("sources", []))
        except Exception:
            pass

    REQ_LAT_ROUTE.labels(cache_state="miss").observe(time.time() - t0)
    return {"query": q, "k": k, "cache": {"hit": False}, **result}


@app.post("/vote")
async def vote(req: VoteRequest) -> Dict[str, Any]:
    """
    Consensus Multi-LLM.
    - mode="precision" => 32B requis (qwen32b_local), deadlines longues
    - mode="interactive" => 7B rapide
    """
    _ensure_consensus_available()
    try:
        res = await consensus_vote(  # type: ignore
            prompt=req.prompt,
            context=req.context or "",
            mode=req.mode or "precision",
        )
        if not isinstance(res, dict) or "final_answer" not in res:
            raise RuntimeError("Format de retour Consensus invalide.")
        return res
    except Exception as e:
        VOTE_ERRORS.inc()
        raise HTTPException(status_code=500, detail=f"Erreur consensus: {e}")


# =============== Uvicorn (debug local) ==================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8100")))
