from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Result:
    doc_id: str
    text: str
    source: str
    score: float
    expert: str


def _minmax(xs: List[float]) -> List[float]:
    if not xs:
        return []
    mn, mx = min(xs), max(xs)
    if mx - mn < 1e-9:
        return [0.5 for _ in xs]
    return [(x - mn) / (mx - mn) for x in xs]


def normalize_scores(buckets: Dict[str, List[Result]]) -> Dict[str, List[Result]]:
    out = {}
    for expert, items in buckets.items():
        scores = [r.score for r in items]
        ns = _minmax(scores)
        new_items = []
        for r, s in zip(items, ns):
            new_items.append(
                Result(
                    doc_id=r.doc_id,
                    text=r.text,
                    source=r.source,
                    score=s,
                    expert=r.expert,
                )
            )
        out[expert] = new_items
    return out


def rrf(
    buckets: Dict[str, List[Result]], rrf_k: int = 60
) -> Dict[str, Dict[str, float]]:
    per_expert_scores: Dict[str, Dict[str, float]] = {}
    for expert, items in buckets.items():
        ranked = sorted(items, key=lambda r: r.score, reverse=True)
        scores: Dict[str, float] = {}
        for rank, r in enumerate(ranked, start=1):
            scores[r.doc_id] = scores.get(r.doc_id, 0.0) + 1.0 / (rrf_k + rank)
        per_expert_scores[expert] = scores
    return per_expert_scores


def fuse_rrf_adaptive(
    buckets: Dict[str, List[Result]],
    query: str,
    weights: Dict[str, float],
    rrf_k: int = 60,
    heuristics: Dict[str, float] = None,
) -> Tuple[List[Dict], Dict]:
    heuristics = heuristics or {}
    short_chars = int(heuristics.get("short_query_chars", 20))
    short_tokens = int(heuristics.get("short_query_tokens", 3))
    boost_lex_short = float(heuristics.get("boost_lexical_on_short", 0.7))
    boost_sem_long = float(heuristics.get("boost_semantic_on_long", 0.7))

    token_count = len(query.strip().split())
    is_short = (len(query) <= short_chars) or (token_count <= short_tokens)

    w_sem = float(weights.get("semantic", 0.6))
    w_lex = float(weights.get("lexical", 0.4))

    if is_short:
        w_lex = boost_lex_short
        w_sem = 1 - w_lex
    else:
        w_sem = boost_sem_long
        w_lex = 1 - w_sem

    norm = normalize_scores(buckets)
    per_expert_rrf = rrf(norm, rrf_k=rrf_k)

    doc_ids = set()
    for s in per_expert_rrf.values():
        doc_ids.update(s.keys())

    combined = {}
    for doc in doc_ids:
        sem = per_expert_rrf.get("semantic", {}).get(doc, 0.0)
        lex = per_expert_rrf.get("lexical", {}).get(doc, 0.0)
        combined[doc] = w_sem * sem + w_lex * lex

    by_doc: Dict[str, Dict] = {}
    for expert, items in norm.items():
        for r in items:
            if (
                r.doc_id not in by_doc
                or combined.get(r.doc_id, 0.0) > by_doc[r.doc_id]["score"]
            ):
                by_doc[r.doc_id] = {
                    "doc_id": r.doc_id,
                    "text": r.text,
                    "source": r.source,
                    "expert": expert,
                    "score": combined.get(r.doc_id, 0.0),
                }

    fused = sorted(by_doc.values(), key=lambda x: x["score"], reverse=True)
    explain = {
        "is_short_query": is_short,
        "weights_applied": {"semantic": w_sem, "lexical": w_lex},
        "rrf_k": rrf_k,
        "counts": {k: len(v) for k, v in buckets.items()},
    }
    return fused, explain


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))


def mmr(
    diversity: float,
    candidates: Dict[str, Dict],
    embeddings: Dict[str, np.ndarray],
    top_k: int,
) -> list:
    selected = []
    remaining = set(candidates.keys())
    scores = [candidates[k]["score"] for k in remaining] if remaining else []
    if scores:
        mn, mx = min(scores), max(scores)
        for k in list(remaining):
            if mx - mn > 1e-9:
                candidates[k]["rel"] = (candidates[k]["score"] - mn) / (mx - mn)
            else:
                candidates[k]["rel"] = 0.5
    while remaining and len(selected) < top_k:
        best_key, best_val = None, -1e9
        for key in remaining:
            rel = candidates[key]["rel"]
            if not selected:
                div = 0.0
            else:
                div = max(cosine(embeddings[key], embeddings[s]) for s in selected)
            val = (1 - diversity) * rel - diversity * div
            if val > best_val:
                best_key, best_val = key, val
        selected.append(best_key)
        remaining.remove(best_key)
    return selected
