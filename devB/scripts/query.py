#!/usr/bin/env python3
import argparse
import json
from typing import List

import httpx

from core.memory.memory_fusion import Result, fuse_rrf_adaptive
from core.utils.common import env

QDRANT_URL = env("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = env("QDRANT_COLLECTION", "nexus_docs")
MEILI_URL = env("MEILI_URL", "http://localhost:7700")
MEILI_MASTER_KEY = env("MEILI_MASTER_KEY", "meili_key")
MEILI_INDEX = env("MEILI_INDEX", "docs")
OLLAMA_HOST = env("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = env("EMBED_MODEL", "nomic-embed-text")


def embed(text: str) -> List[float]:
    url = f"{OLLAMA_HOST.rstrip('/')}/api/embeddings"
    payload = {"model": EMBED_MODEL, "prompt": text}
    with httpx.Client(timeout=60.0) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        vec = data.get("embedding") or data.get("data", [{}])[0].get("embedding")
        if not vec:
            raise RuntimeError("No embedding from Ollama")
        return vec


def search_semantic(q: str, k: int) -> List[Result]:
    vec = embed(q)
    url = f"{QDRANT_URL.rstrip('/')}/collections/{QDRANT_COLLECTION}/points/search"
    body = {
        "vector": vec,
        "limit": max(k * 3, 10),
        "with_payload": True,
        "with_vector": False,
    }
    with httpx.Client(timeout=30.0) as client:
        r = client.post(url, json=body)
        r.raise_for_status()
        data = r.json()
    points = data.get("result", []) or data.get("points", [])
    out = []
    for p in points:
        payload = p.get("payload", {})
        out.append(
            Result(
                doc_id=str(payload.get("doc_id") or p.get("id")),
                text=payload.get("text", ""),
                source=payload.get("source", ""),
                score=float(p.get("score", 0.0)),
                expert="semantic",
            )
        )
    return out


def search_lexical(q: str, k: int) -> List[Result]:
    url = f"{MEILI_URL.rstrip('/')}/indexes/{MEILI_INDEX}/search"
    headers = {"X-Meili-API-Key": MEILI_MASTER_KEY}
    body = {"q": q, "limit": max(k * 3, 10)}
    with httpx.Client(timeout=30.0) as client:
        r = client.post(url, headers=headers, json=body)
        r.raise_for_status()
        data = r.json()
    hits = data.get("hits", [])
    out = []
    for i, h in enumerate(hits):
        raw_score = 1.0 / (i + 1)
        out.append(
            Result(
                doc_id=str(h.get("doc_id") or h.get("id") or f"meili_{i}"),
                text=h.get("text", "") or h.get("content", ""),
                source=h.get("source", "") or h.get("url", ""),
                score=raw_score,
                expert="lexical",
            )
        )
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Hybrid RAG query (Qdrant + Meili) with RRF + optional MMR"
    )
    ap.add_argument("--q", required=True, help="query text")
    ap.add_argument("--k", type=int, default=5, help="top-k")
    ap.add_argument(
        "--mmr", type=float, default=0.0, help="diversity factor 0..1 (0=off)"
    )
    args = ap.parse_args()

    sem = search_semantic(args.q, args.k)
    lex = search_lexical(args.q, args.k)

    buckets = {"semantic": sem, "lexical": lex}
    weights = {"semantic": 0.6, "lexical": 0.4}
    fused, explain = fuse_rrf_adaptive(
        buckets,
        args.q,
        weights,
        rrf_k=60,
        heuristics={
            "short_query_chars": 20,
            "short_query_tokens": 3,
            "boost_lexical_on_short": 0.7,
            "boost_semantic_on_long": 0.7,
        },
    )

    if args.mmr > 0 and fused:
        topN = min(10, len(fused))
        texts = [f["text"] for f in fused[:topN]]
        embs = []
        for t in texts:
            vec = embed(t[:2000])
            import numpy as np

            embs.append(np.array(vec, dtype=float))
        candidates = {fused[i]["doc_id"]: fused[i] for i in range(topN)}
        emb_map = {fused[i]["doc_id"]: embs[i] for i in range(topN)}
        from core.memory.memory_fusion import mmr as mmr_func

        selected_ids = mmr_func(
            diversity=args.mmr, candidates=candidates, embeddings=emb_map, top_k=args.k
        )
        fused = [candidates[i] for i in selected_ids]

    out = {"results": fused[: args.k], "explain": explain}
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
