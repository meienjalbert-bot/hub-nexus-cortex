#!/usr/bin/env python3
import argparse
import json
import os
import pathlib
from typing import Dict, List

import fitz  # PyMuPDF
import httpx

# tiny helpers you already have in repo
try:
    from core.utils.chunkers import chunk_text
    from core.utils.common import env, file_id
except Exception:
    # fallbacks if needed
    def env(k: str, default: str = "") -> str:
        return os.getenv(k, default)

    def file_id(path: str) -> str:
        import hashlib

        return hashlib.sha1(path.encode("utf-8")).hexdigest()[:16]

    def chunk_text(text: str, chunk_chars: int = 2000, overlap: int = 200):
        chunks = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + chunk_chars, n)
            chunks.append({"text": text[start:end]})
            if end == n:
                break
            start = end - overlap
        return chunks


QDRANT_URL = env("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = env("QDRANT_COLLECTION", "nexus_docs")
MEILI_URL = env("MEILI_URL", "http://localhost:7700")
MEILI_MASTER_KEY = env("MEILI_MASTER_KEY", "meili_key")
MEILI_INDEX = env("MEILI_INDEX", "docs")
OLLAMA_HOST = env("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = env("EMBED_MODEL", "nomic-embed-text")


def read_text(path: str) -> str:
    p = pathlib.Path(path)
    if p.suffix.lower() == ".pdf":
        parts: List[str] = []
        with fitz.open(path) as doc:
            for page in doc:
                parts.append(page.get_text())
        return "\n".join(parts)
    return p.read_text(encoding="utf-8", errors="ignore")


def ensure_qdrant_collection(dim: int, collection: str) -> None:
    url = f"{QDRANT_URL.rstrip('/')}/collections/{collection}"
    with httpx.Client(timeout=30) as client:
        r = client.get(url)
        if r.status_code == 200:
            return
        body = {"vectors": {"size": dim, "distance": "Cosine"}}
        cr = client.put(url, json=body)
        cr.raise_for_status()


def embed(text: str) -> List[float]:
    url = f"{OLLAMA_HOST.rstrip('/')}/api/embeddings"
    payload = {"model": EMBED_MODEL, "prompt": text}
    with httpx.Client(timeout=60) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        if "embedding" in data:
            return data["embedding"]
        arr = data.get("data", [])
        if arr and "embedding" in arr[0]:
            return arr[0]["embedding"]
        return []


def upsert_qdrant(points: List[Dict], collection: str) -> None:
    url = f"{QDRANT_URL.rstrip('/')}/collections/{collection}/points"
    with httpx.Client(timeout=60) as client:
        r = client.put(url, json={"points": points})
        r.raise_for_status()


def ensure_meili_index(index_uid: str) -> None:
    url = f"{MEILI_URL.rstrip('/')}/indexes/{index_uid}"
    headers = {"X-Meili-API-Key": MEILI_MASTER_KEY}
    with httpx.Client(timeout=30) as client:
        r = client.get(url, headers=headers)
        if r.status_code == 200:
            return
        cr = client.post(
            f"{MEILI_URL.rstrip('/')}/indexes", headers=headers, json={"uid": index_uid}
        )
        cr.raise_for_status()


def add_meili_docs(index_uid: str, docs: List[Dict]) -> None:
    url = f"{MEILI_URL.rstrip('/')}/indexes/{index_uid}/documents"
    headers = {"X-Meili-API-Key": MEILI_MASTER_KEY, "Content-Type": "application/json"}
    with httpx.Client(timeout=60) as client:
        r = client.post(url, headers=headers, json=docs)
        r.raise_for_status()


def walk_files(root: str, exts: List[str]) -> List[str]:
    out: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            p = os.path.join(dirpath, name)
            if pathlib.Path(p).suffix.lower() in exts:
                out.append(p)
    return out


def ingest_path(
    path: str,
    collection: str,
    index: str,
    chunk_chars: int = 2000,
    overlap: int = 200,
    batch: int = 64,
) -> Dict:
    files = walk_files(path, [".md", ".markdown", ".txt", ".pdf"])
    if not files:
        return {"ingested": 0, "chunks": 0, "message": "no files"}

    dim_probe = len(embed("probe")) or 768
    ensure_qdrant_collection(dim_probe, collection)
    ensure_meili_index(index)

    qdrant_points: List[Dict] = []
    meili_docs: List[Dict] = []

    files_ing = 0
    chunks_ing = 0

    for f in files:
        raw = read_text(f)
        files_ing += 1
        for i, ch in enumerate(
            chunk_text(raw, chunk_chars=chunk_chars, overlap=overlap)
        ):
            docid = f"{file_id(f)}_{i:04d}"
            vec = embed(ch["text"][:3000])
            qdrant_points.append(
                {
                    "id": docid,
                    "vector": vec,
                    "payload": {
                        "doc_id": docid,
                        "source": f"file://{os.path.abspath(f)}",
                        "text": ch["text"],
                    },
                }
            )
            meili_docs.append(
                {
                    "id": docid,
                    "doc_id": docid,
                    "source": f"file://{os.path.abspath(f)}",
                    "text": ch["text"],
                }
            )
            chunks_ing += 1

            if len(qdrant_points) >= batch:
                upsert_qdrant(qdrant_points, collection)
                qdrant_points = []

            if len(meili_docs) >= batch * 2:
                add_meili_docs(index, meili_docs)
                meili_docs = []

    if qdrant_points:
        upsert_qdrant(qdrant_points, collection)
    if meili_docs:
        add_meili_docs(index, meili_docs)

    return {
        "ingested": files_ing,
        "chunks": chunks_ing,
        "collection": collection,
        "index": index,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest folder into Qdrant + Meili")
    ap.add_argument("--path", required=True)
    ap.add_argument("--collection", default=QDRANT_COLLECTION)
    ap.add_argument("--index", default=MEILI_INDEX)
    ap.add_argument("--chunk_chars", type=int, default=2000)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    res = ingest_path(
        args.path,
        args.collection,
        args.index,
        args.chunk_chars,
        args.overlap,
        args.batch,
    )
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
