from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional

import redis.asyncio as redis

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
# decode_responses=True → strings, pas de bytes
r = redis.from_url(REDIS_URL, decode_responses=True)


def _key(ns: str, **params: Any) -> str:
    # Hash stable (insensible à l'ordre) et UTF-8 safe
    blob = json.dumps(params, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return f"{ns}:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


async def cache_get(ns: str, **params: Any) -> Optional[Dict[str, Any]]:
    """Retourne l'objet JSON si présent, sinon None."""
    k = _key(ns, **params)
    v = await r.get(k)
    return json.loads(v) if v else None


async def cache_set(
    ns: str, value: Dict[str, Any] | Any, ttl: int = 3600, **params: Any
) -> None:
    """Sérialise en JSON et set avec TTL (par défaut 1h)."""
    k = _key(ns, **params)
    await r.setex(k, ttl, json.dumps(value, ensure_ascii=False, separators=(",", ":")))
