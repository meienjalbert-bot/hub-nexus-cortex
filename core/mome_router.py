# core/mome_router.py

import os
from collections import defaultdict
from typing import Any, Dict, List

import httpx

MEILI_HOST = os.getenv("MEILI_HOST", "http://meili:7700")
MEILI_KEY = os.getenv("MEILI_MASTER_KEY", "meili_key")
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://qdrant:6333")

FUSION_WEIGHTS = {
    "factual": {"lexical": 0.4, "semantic": 0.3, "temporal": 0.2, "graph": 0.1},
    "conceptual": {"semantic": 0.5, "lexical": 0.2, "temporal": 0.15, "graph": 0.15},
    "recent": {"temporal": 0.5, "lexical": 0.25, "semantic": 0.2, "graph": 0.05},
    "default": {"semantic": 0.35, "lexical": 0.35, "temporal": 0.2, "graph": 0.1},
}


def _detect_query_type(query: str) -> str:
    query_lower = query.lower()
    temporal_keywords = ["récent", "dernier", "nouveau", "aujourd'hui", "2024", "2025"]
    if any(kw in query_lower for kw in temporal_keywords):
        return "recent"
    factual_keywords = ["qui est", "qu'est-ce", "définition", "combien", "quand"]
    if any(kw in query_lower for kw in factual_keywords):
        return "factual"
    conceptual_keywords = ["pourquoi", "comment", "expliquer", "concept", "principe"]
    if any(kw in query_lower for kw in conceptual_keywords):
        return "conceptual"
    return "default"


def _search_lexical(query: str, k: int = 5) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.post(
                f"{MEILI_HOST}/indexes/nexus_docs/search",
                json={"q": query, "limit": k},
                headers={"Authorization": f"Bearer {MEILI_KEY}"},
            )
            if resp.status_code == 200:
                hits = resp.json().get("hits", [])
                for i, hit in enumerate(hits):
                    results.append(
                        {
                            "text": hit.get("content", hit.get("text", "")),
                            "score": 1.0 / (i + 1),
                            "source": hit.get("source", "unknown"),
                            "expert": "lexical",
                            "id": hit.get("id", f"meili_{i}"),
                        }
                    )
    except Exception as e:
        print(f"[MoME] Lexical search error: {e}")
    return results


def _search_semantic(query: str, k: int = 5) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    try:
        # Stub implementation for now
        results.append(
            {
                "text": f"Résultat sémantique pour '{query[:30]}...'",
                "score": 0.92,
                "source": "qdrant_stub",
                "expert": "semantic",
                "id": "qdrant_stub_1",
            }
        )
    except Exception as e:
        print(f"[MoME] Semantic search error: {e}")
    return results


def _search_temporal(query: str, k: int = 5) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.post(
                f"{MEILI_HOST}/indexes/nexus_docs/search",
                json={"q": query, "limit": k, "sort": ["timestamp:desc"]},
                headers={"Authorization": f"Bearer {MEILI_KEY}"},
            )
            if resp.status_code == 200:
                hits = resp.json().get("hits", [])
                for i, hit in enumerate(hits):
                    results.append(
                        {
                            "text": hit.get("content", ""),
                            "score": 0.85,
                            "source": hit.get("source", "unknown"),
                            "expert": "temporal",
                            "id": hit.get("id", f"temporal_{i}"),
                            "timestamp": hit.get("timestamp", ""),
                        }
                    )
    except Exception as e:
        print(f"[MoME] Temporal search error: {e}")
    return results


def _search_graph(query: str, k: int = 5) -> List[Dict[str, Any]]:
    # Stub for graph search (Phase 2)
    return []


def _reciprocal_rank_fusion(
    results_by_expert: Dict[str, List[Dict[str, Any]]],
    weights: Dict[str, float],
    k_param: int = 60,
) -> List[Dict[str, Any]]:
    scores: Dict[str, float] = defaultdict(float)
    doc_map: Dict[str, Dict[str, Any]] = {}
    for expert, results in results_by_expert.items():
        weight = weights.get(expert, 0.1)
        for rank, doc in enumerate(results, start=1):
            doc_id = doc.get("id", f"{expert}_{rank}")
            scores[doc_id] += weight * (1.0 / (k_param + rank))
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{**doc_map[doc_id], "final_score": score} for doc_id, score in sorted_docs]


def run_mome(query: str, k: int = 5) -> Dict[str, Any]:
    query_type = _detect_query_type(query)
    weights = FUSION_WEIGHTS.get(query_type, FUSION_WEIGHTS["default"])
    results_by_expert: Dict[str, List[Dict[str, Any]]] = {}
    if weights.get("lexical", 0) > 0:
        results_by_expert["lexical"] = _search_lexical(query, k)
    if weights.get("semantic", 0) > 0:
        results_by_expert["semantic"] = _search_semantic(query, k)
    if weights.get("temporal", 0) > 0:
        results_by_expert["temporal"] = _search_temporal(query, k)
    if weights.get("graph", 0) > 0:
        results_by_expert["graph"] = _search_graph(query, k)
    fused = _reciprocal_rank_fusion(results_by_expert, weights)
    top_k = fused[:k]
    answer = _generate_answer(query, top_k, query_type)
    return {
        "answer": answer,
        "sources": top_k,
        "experts_used": list(results_by_expert.keys()),
        "query_type": query_type,
        "fusion_method": "rrf_adaptive",
        "fusion_weights": weights,
    }


def _generate_answer(
    query: str, context_docs: List[Dict[str, Any]], query_type: str
) -> str:
    context_str = "\n\n".join(
        f"[{i+1}] {doc.get('text', '')[:200]}..."
        for i, doc in enumerate(context_docs[:3])
    )
    answer = (
        f"Basé sur les sources disponibles, voici une réponse pour '{query}':\n\n"
        f"{context_str}\n\n"
        "(Note: génération LLM à implémenter en Phase 1.5)"
    )
    return answer
