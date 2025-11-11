# apps/orchestrator/advanced_health.py
import asyncio
import time

from fastapi import APIRouter

from core.cache import metrics as cache_metrics
from core.heavy_gate import heavy_gate
from core.model_manager import generate, health_check

router = APIRouter()


async def _test_heavy(timeout_s: int = 15) -> dict:
    try:
        t0 = time.time()
        r = await asyncio.wait_for(
            generate("qwen32b_local", "OK ?", max_tokens=2, temperature=0),
            timeout=timeout_s,
        )
        dt = round((time.time() - t0) * 1000, 1)
        ok = "ok" in r.lower() or "o" == r.lower().strip()
        return {
            "ready": ok,
            "rt_ms": dt,
            "sample": r[:60],
            "error": None if ok else "invalid",
        }
    except asyncio.TimeoutError:
        return {
            "ready": False,
            "rt_ms": None,
            "sample": None,
            "error": f"timeout_{timeout_s}s",
        }
    except Exception as e:
        return {"ready": False, "rt_ms": None, "sample": None, "error": str(e)}


@router.get("/advanced_health")
async def advanced_health():
    base_ok = await health_check()
    heavy = await _test_heavy()
    cache = await cache_metrics()
    hg = heavy_gate.metrics()

    status = "ok" if base_ok and heavy["ready"] else "degraded"
    suggested_mode = "precision" if heavy["ready"] else "interactive"

    return {
        "status": status,
        "deps": {
            "ollama": base_ok,
            "heavy_model": heavy["ready"],
            "semantic_cache": cache.get("connected", False),
        },
        "heavy_model": heavy,
        "cache": cache,
        "heavy_concurrency": hg,
        "suggested_mode": suggested_mode,
        "timestamp": time.time(),
    }
