from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, Optional

import aiohttp

from core.heavy_gate import heavy_gate

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")


async def _post_json(
    url: str, payload: Dict[str, Any], timeout_s: int
) -> Dict[str, Any]:
    """POST JSON robuste avec timeout total."""
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as resp:
            data: Dict[str, Any]
            if resp.status == 200:
                data = await resp.json()
            else:
                # On capture aussi le texte pour debug
                data = {"error": f"HTTP_{resp.status}", "text": await resp.text()}
            return data


async def _get(url: str, timeout_s: int = 5) -> Optional[Dict[str, Any]]:
    """GET JSON best-effort."""
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
                return None
    except Exception:
        return None


def _build_options(
    max_tokens: int,
    temperature: float,
    top_p: Optional[float],
    repetition_penalty: Optional[float],
) -> Dict[str, Any]:
    options: Dict[str, Any] = {
        "num_predict": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,  # défaut raisonnable
        "repeat_penalty": 1.1,  # limite la verbosité
    }
    if top_p is not None:
        options["top_p"] = top_p
    if repetition_penalty is not None:
        options["repeat_penalty"] = repetition_penalty
    return options


async def generate(
    model: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.2,
    timeout_s: int = 30,
    top_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    max_retries: int = 1,
) -> str:
    """
    Génération via Ollama (non-stream), avec:
      - backoff/retry léger,
      - sémaphore pour les modèles lourds (32B, etc.).
    Retourne le texte (ou un tag d'erreur encadré par []).
    """
    options = _build_options(max_tokens, temperature, top_p, repetition_penalty)
    payload = {"model": model, "prompt": prompt, "stream": False, "options": options}

    async def _one_call() -> str:
        data = await _post_json(f"{OLLAMA_BASE}/api/generate", payload, timeout_s)
        if "response" in data:
            return str(data["response"]).strip()
        # Normalise l'erreur pour la couche appelante
        return f"[ERROR] {data.get('error') or data.get('text') or 'unknown'}"

    async def _guarded() -> str:
        # Protège les modèles lourds ; modèle léger => passe-through
        async with heavy_gate.section(model):
            return await _one_call()

    last_err = ""
    for attempt in range(max_retries + 1):
        try:
            return await _guarded()
        except asyncio.TimeoutError:
            last_err = f"TIMEOUT_{timeout_s}s"
        except Exception as e:  # noqa: BLE001 - on normalise vers une string
            last_err = f"ERROR {e!s}"
        if attempt < max_retries:
            # petit backoff progressif
            await asyncio.sleep(1.5 * (attempt + 1))
    return f"[{last_err}]"


async def prewarm(models: list[str]) -> None:
    """
    Pré-chauffe Ollama:
      - ping /api/tags
      - "touche" les modèles en envoyant une requête très courte
    """
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # ping global
            await session.get(f"{OLLAMA_BASE}/api/tags")
            # touche chaque modèle unique
            for m in set(models or []):
                await session.post(
                    f"{OLLAMA_BASE}/api/generate",
                    json={"model": m, "prompt": "ping", "stream": False},
                )
    except Exception:
        # best-effort
        pass


async def health_check() -> bool:
    """
    Vérifie qu'Ollama répond.
    """
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{OLLAMA_BASE}/api/tags") as resp:
                return resp.status == 200
    except Exception:
        return False
