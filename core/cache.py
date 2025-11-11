# core/cache.py
import hashlib
import json
import os
from typing import Optional

import redis.asyncio as redis

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "3600"))  # 1h

_client: Optional[redis.Redis] = None


async def _cli() -> Optional[redis.Redis]:
    global _client
    if _client is None:
        try:
            _client = redis.from_url(REDIS_URL, decode_responses=True)
            await _client.ping()
        except Exception:
            _client = None
    return _client


def make_key(**parts) -> str:
    dump = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return "vote_cache:" + hashlib.sha256(dump.encode()).hexdigest()


async def get(key: str) -> Optional[dict]:
    c = await _cli()
    if not c:
        return None
    val = await c.get(key)
    return json.loads(val) if val else None


async def set(key: str, value: dict, ttl: int | None = None) -> bool:
    c = await _cli()
    if not c:
        return False
    await c.setex(key, ttl or CACHE_TTL, json.dumps(value, ensure_ascii=False))
    return True


async def metrics() -> dict:
    c = await _cli()
    if not c:
        return {"connected": False}
    keys = await c.keys("vote_cache:*")
    return {"connected": True, "items": len(keys), "ttl": CACHE_TTL}
