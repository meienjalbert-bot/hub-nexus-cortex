# core/multi_llm_voting.py
import os
from typing import Any, Dict, List

import httpx

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
PRIMARY_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
FALLBACK_MODEL = os.getenv("OLLAMA_FALLBACK_MODEL", "llama3.2:1b")
TIMEOUT_S = int(os.getenv("OLLAMA_TIMEOUT", "60"))


def _ollama_generate(prompt: str, model: str) -> str:
    """Essaie /api/generate puis /api/chat; degrade vers FALLBACK_MODEL si besoin."""
    headers = {"Content-Type": "application/json"}

    def post(path: str, payload: Dict[str, Any]):
        with httpx.Client(timeout=TIMEOUT_S) as client:
            return client.post(f"{OLLAMA_BASE}{path}", json=payload, headers=headers)

    # 1) /api/generate (modèle demandé)
    r = post("/api/generate", {"model": model, "prompt": prompt, "stream": False})
    if r.status_code == 200:
        return r.json().get("response", "").strip()

    # 2) downgrade modèle + retry
    if model != FALLBACK_MODEL:
        r2 = post(
            "/api/generate",
            {"model": FALLBACK_MODEL, "prompt": prompt, "stream": False},
        )
        if r2.status_code == 200:
            return r2.json().get("response", "").strip()

        # 3) essai /api/chat
        r3 = post(
            "/api/chat",
            {
                "model": FALLBACK_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
        )
        if r3.status_code == 200:
            return r3.json().get("message", {}).get("content", "").strip()

        return f"[OLLAMA_ERR {r2.status_code}] {r2.text[:200]}"
    else:
        # 3) essai /api/chat direct si generate a échoué même sur FALLBACK
        r3 = post(
            "/api/chat",
            {
                "model": FALLBACK_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
        )
        if r3.status_code == 200:
            return r3.json().get("message", {}).get("content", "").strip()
        return f"[OLLAMA_ERR {r.status_code}] {r.text[:200]}"


def vote(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    payload attendu:
    {
      "prompt": "…",
      "context": "…",          # optionnel
      "experts": ["analyst","creative"]  # optionnel
    }
    """
    prompt = str(payload.get("prompt", "")).strip()
    context = str(payload.get("context", "")).strip()
    experts: List[str] = payload.get("experts", ["analyst", "creative"])
    if not prompt:
        return {
            "final_answer": "",
            "votes": [],
            "confidence": 0.0,
            "consensus_method": "invalid_payload",
        }

    # mapping simple (même modèle pour tous les experts en CPU)
    model_map = {e: PRIMARY_MODEL for e in experts}

    votes: List[Dict[str, Any]] = []
    for expert in experts:
        model = model_map.get(expert, PRIMARY_MODEL)
        try:
            full_prompt = (
                context + "\n\n" if context else ""
            ) + f"Question: {prompt}\nRéponse brève et précise:"
            ans = _ollama_generate(full_prompt, model)
            votes.append(
                {
                    "model": expert,
                    "backend": model,
                    "answer": ans,
                    "confidence": 0.7 if ans else 0.0,
                }
            )
        except Exception as e:
            votes.append(
                {
                    "model": expert,
                    "backend": model,
                    "answer": f"[ERROR] {e}",
                    "confidence": 0.0,
                }
            )

    # agrégation très simple (MVP) : réponse la plus longue non-vide
    non_empty = [v for v in votes if v.get("answer")]
    if not non_empty:
        return {
            "final_answer": "Aucune réponse exploitable.",
            "votes": votes,
            "confidence": 0.0,
            "consensus_method": "longest_nonempty",
        }

    winner = max(non_empty, key=lambda x: len(x["answer"]))
    confidence = sum(v["confidence"] for v in non_empty) / max(len(non_empty), 1)
    return {
        "final_answer": winner["answer"],
        "votes": votes,
        "confidence": round(confidence, 3),
        "consensus_method": "longest_nonempty",
    }
