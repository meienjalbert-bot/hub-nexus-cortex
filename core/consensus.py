# core/consensus.py
import asyncio
import json
import time
from typing import Any, Dict, List

import yaml

from .cache import get as cache_get
from .cache import make_key
from .cache import set as cache_set
from .grounding import make_context
from .model_manager import generate, prewarm


def _load_mode_cfg(path: str, mode: str) -> Dict[str, Any]:
    cfg = yaml.safe_load(open(path, "r"))
    modes = cfg.get("modes", {})
    if mode not in modes:
        raise ValueError(f"Mode inconnu: {mode}")
    return {
        "committee": modes[mode]["committee"],
        "soft": modes[mode]["soft_deadline_s"],
        "hard": modes[mode]["hard_deadline_s"],
        "grace": modes[mode]["grace_s"],
        "require_heavy": modes[mode]["require_heavy"],
        "conductor": cfg.get("conductor", {}),
    }


def _is_heavy(model: str) -> bool:
    m = model.lower()
    return "32b" in m or "qwen32b" in m


async def vote(
    prompt: str,
    context: str = "",
    config_path: str = "configs/consensus_models.yaml",
    mode: str = "precision",
) -> Dict[str, Any]:
    # cache exact (évite recalcul 32B)
    ck = make_key(prompt=prompt, context=context, mode=mode, cfg=config_path)
    if cached := await cache_get(ck):
        cached["cache_hit"] = True
        return cached

    cfg = _load_mode_cfg(config_path, mode)
    committee = cfg["committee"]
    conductor = cfg["conductor"]

    await prewarm([m["model"] for m in committee])

    ctx = make_context(context, extra_terms=["MoME", "RAG"])

    async def ask(m: Dict[str, Any]) -> Dict[str, Any]:
        sys = m.get("system", "")
        text = await generate(
            m["model"],
            f"""{sys}
Contexte projet (OBLIGATOIRE) :
{ctx}

Question:
{prompt}

Contraintes: Réponds en français, concis. N'invente pas d'acronymes.""",
            m.get("max_tokens", 256),
            m.get("temperature", 0.2),
            m.get("timeout_s", 12),
            top_p=m.get("top_p"),
            repetition_penalty=m.get("repetition_penalty"),
        )
        return {
            "role": m["role"],
            "model": m["model"],
            "answer": text,
            "success": not text.startswith("[ERROR"),
        }

    tasks = [asyncio.create_task(ask(m)) for m in committee]
    start = time.time()

    done, pending = await asyncio.wait(
        tasks, timeout=cfg["soft"], return_when=asyncio.FIRST_COMPLETED
    )
    results = [t.result() for t in done if not t.cancelled() and not t.exception()]

    def have_heavy(rs: List[Dict[str, Any]]):
        models_ok = {r["model"] for r in rs if r.get("success")}
        heavy_in_committee = [m["model"] for m in committee if _is_heavy(m["model"])]
        return any(h in models_ok for h in heavy_in_committee)

    if cfg["require_heavy"] and not have_heavy(results):
        more, pending = await asyncio.wait(pending, timeout=cfg["grace"])
        results += [t.result() for t in more if not t.cancelled() and not t.exception()]

    if cfg["require_heavy"] and not have_heavy(results):
        remain = max(0, cfg["hard"] - (time.time() - start))
        more, pending = await asyncio.wait(pending, timeout=remain)
        results += [t.result() for t in more if not t.cancelled() and not t.exception()]

    for t in pending:
        t.cancel()

    elapsed = round(time.time() - start, 3)

    if cfg["require_heavy"] and not have_heavy(results):
        out = {
            "status": "timeout",
            "final_answer": "Mode précision: 32B indisponible.",
            "votes": results,
            "confidence": 0.0,
            "elapsed_s": elapsed,
            "mode": mode,
        }
        await cache_set(ck, out)  # on cache aussi l’échec court pour épargner 32B
        out["cache_hit"] = False
        return out

    # un seul membre (ex: 32B seul) ⇒ direct
    valid = [r for r in results if r.get("success")]
    if len(valid) == 1 and len(committee) == 1:
        out = {
            "status": "ok",
            "final_answer": valid[0]["answer"],
            "votes": results,
            "confidence": 0.9,
            "elapsed_s": elapsed,
            "mode": mode,
        }
        await cache_set(ck, out)
        out["cache_hit"] = False
        return out

    # synthèse conductor
    synth = await generate(
        conductor["model"],
        f"""{conductor.get("system","")}
Contexte projet (OBLIGATOIRE):
{ctx}

Réponses du comité:
{json.dumps(valid, ensure_ascii=False)}

Donne une synthèse unique, courte, fidèle au contexte projet.""",
        conductor.get("max_tokens", 256),
        conductor.get("temperature", 0.2),
        10,
    )

    base = 0.7 if any(_is_heavy(r["model"]) for r in valid) else 0.55
    conf = min(0.95, base + 0.15)
    out = {
        "status": "ok",
        "final_answer": synth,
        "votes": results,
        "confidence": round(conf, 2),
        "elapsed_s": elapsed,
        "mode": mode,
    }
    await cache_set(ck, out)
    out["cache_hit"] = False
    return out
