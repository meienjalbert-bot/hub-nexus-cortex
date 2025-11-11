import asyncio
from typing import Any, Dict, List

import yaml

from .model_manager import generate, prewarm


def _cosine_sim(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    denom = (len(a) * len(b)) ** 0.5
    return inter / denom if denom else 0.0


async def vote(
    prompt: str, context: str, config_path: str = "configs/consensus_models.yaml"
) -> Dict[str, Any]:
    cfg = yaml.safe_load(open(config_path, "r"))
    committee = cfg.get("committee", [])
    conductor = cfg.get("conductor", {})

    async def ask(m):
        text = await generate(
            m["model"],
            f"{prompt}\n\nContext:\n{context}",
            m.get("max_tokens", 256),
            m.get("temperature", 0.2),
            m.get("timeout_s", 12),
        )
        return {"role": m["role"], "model": m["model"], "answer": text}

    await prewarm([m["model"] for m in committee])
    answers: List[Dict[str, Any]] = await asyncio.gather(
        *[ask(m) for m in committee], return_exceptions=False
    )

    clusters = []
    for ans in answers:
        tok = set(ans["answer"].lower().split())
        placed = False
        for c in clusters:
            inter = len(c["tok"] & tok)
            denom = (len(c["tok"]) * len(tok)) ** 0.5 if c["tok"] and tok else 1
            sim = inter / denom if denom else 0.0
            if sim >= 0.8:
                c["members"].append(ans)
                c["tok"] |= tok
                placed = True
                break
        if not placed:
            clusters.append({"tok": tok, "members": [ans]})

    clusters.sort(key=lambda c: len(c["members"]), reverse=True)
    top = clusters[0] if clusters else {"members": answers}
    final = top["members"][0]["answer"] if top["members"] else ""

    if conductor:
        try:
            joined = "\n\n---\n\n".join([m["answer"] for m in top["members"]])
            final = await generate(
                conductor["model"],
                f"Synthétise et consolide ces réponses en une seule, fiable et concise:\n\n{joined}",
                conductor.get("max_tokens", 384),
                conductor.get("temperature", 0.2),
                conductor.get("timeout_s", 12),
            )
        except Exception:
            pass

    confidence = min(
        1.0, 0.5 + 0.5 * (len(top.get("members", [])) / max(1, len(answers)))
    )
    return {"final_answer": final, "votes": answers, "confidence": round(confidence, 2)}
