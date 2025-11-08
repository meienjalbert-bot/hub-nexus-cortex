from typing import List, Dict, Any

def normalize_scores(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not items:
        return []
    max_s = max((i.get("score", 0.0) or 0.0) for i in items) or 1.0
    out = []
    for i in items:
        j = dict(i)
        j["norm"] = (i.get("score", 0.0) or 0.0) / max_s
        out.append(j)
    return out

def rrf_merge(buckets: Dict[str, List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
    agg = {}
    for expert, results in buckets.items():
        for rank, item in enumerate(results, start=1):
            key = item.get("id") or item.get("doc_id") or item.get("text")
            if not key:
                continue
            agg.setdefault(key, {"item": item, "score": 0.0, "experts": set()})
            agg[key]["score"] += 1.0 / (k + rank)
            agg[key]["experts"].add(expert)
    merged = []
    for v in agg.values():
        it = dict(v["item"])
        it["fusion_score"] = v["score"]
        it["experts"] = sorted(list(v["experts"]))
        merged.append(it)
    merged.sort(key=lambda x: x["fusion_score"], reverse=True)
    return merged

def dedup(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for it in items:
        key = it.get("id") or it.get("doc_id") or it.get("text")
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out
