import os, httpx, asyncio

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434").rstrip("/")

async def ensure_present(model: str):
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            rl = await client.get(f"{OLLAMA_HOST}/api/tags")
            if rl.status_code == 200 and any(m.get("name")==model for m in rl.json().get("models", [])):
                return True
            await client.post(f"{OLLAMA_HOST}/api/pull", json={"name": model})
            return True
    except Exception:
        return False

async def prewarm(models: list[str]):
    async def warm(m: str):
        ok = await ensure_present(m)
        if not ok: 
            return False
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                payload = {"model": m, "prompt": "ok", "options": {"num_predict": 1}}
                await client.post(f"{OLLAMA_HOST}/api/generate", json=payload)
            return True
        except Exception:
            return False
    return await asyncio.gather(*[warm(m) for m in models])

async def generate(model: str, prompt: str, max_tokens: int = 256, temperature: float = 0.2, timeout_s: int = 12):
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        payload = {"model": model, "prompt": prompt, "options": {"num_predict": max_tokens, "temperature": temperature}}
        r = await client.post(f"{OLLAMA_HOST}/api/generate", json=payload)
        r.raise_for_status()
        return r.json().get("response", "").strip()
