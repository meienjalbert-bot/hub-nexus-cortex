#!/usr/bin/env python3
import argparse
import os
import pathlib
from typing import Dict, List

import fitz
import httpx

from core.utils.chunkers import chunk_text
from core.utils.common import env, file_id

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
        text = []
        with fitz.open(path) as doc:
            for page in doc:
                text.append(page.get_text())
        return "\n".join(text)
    else:
        return p.read_text(encoding="utf-8", errors="ignore")


def ensure_qdrant_collection(dim: int, collection: str):
    url = f"{QDRANT_URL.rstrip('/')}/collections/{collection}"
    with httpx.Client(timeout=30.0) as client:
        r = client.get(url)
        if r.status_code == 200:
            return
        body = {"vectors": {"size": dim, "distance": "Cosine"}}
        cr = client.put(url, json=body)
        cr.raise_for_status()


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


def upsert_qdrant(points: List[Dict], collection: str):
    url = f"{QDRANT_URL.rstrip('/')}/collections/{collection}/points"
    body = {"points": points}
    with httpx.Client(timeout=60.0) as client:
        r = client.put(url, json=body)
        r.raise_for_status()


def ensure_meili_index(index_uid: str):
    url = f"{MEILI_URL.rstrip('/')}/indexes/{index_uid}"
    headers = {"X-Meili-API-Key": MEILI_MASTER_KEY}
    with httpx.Client(timeout=30.0) as client:
        r = client.get(url, headers=headers)
        if r.status_code == 200:
            return
        cr = client.post(
            f"{MEILI_URL.rstrip('/')}/indexes", headers=headers, json={"uid": index_uid}
        )
        cr.raise_for_status()


def add_meili_docs(index_uid: str, docs: List[Dict]):
    url = f"{MEILI_URL.rstrip('/')}/indexes/{index_uid}/documents"
    headers = {"X-Meili-API-Key": MEILI_MASTER_KEY, "Content-Type": "application/json"}
    with httpx.Client(timeout=60.0) as client:
        r = client.post(url, headers=headers, json=docs)
        r.raise_for_status()


def walk_files(root: str, exts: List[str]) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            p = os.path.join(dirpath, name)
            if pathlib.Path(p).suffix.lower() in exts:
                out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Ingest folder into Qdrant (vectors) + Meili (lexical)"
    )
    ap.add_argument("--path", required=True, help="Folder with .md/.txt/.pdf")
    ap.add_argument("--collection", default=QDRANT_COLLECTION)
    ap.add_argument("--index", default=MEILI_INDEX)
    ap.add_argument("--chunk_chars", type=int, default=2000)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    files = walk_files(args.path, [".md", ".markdown", ".txt", ".pdf"])
    if not files:
        print("No files found")
        return

    dim_probe = len(embed("probe"))
    ensure_qdrant_collection(dim_probe, args.collection)
    ensure_meili_index(args.index)

    qdrant_points = []
    meili_docs = []

    for f in files:
        raw = read_text(f)
        for i, ch in enumerate(
            chunk_text(raw, chunk_chars=args.chunk_chars, overlap=args.overlap)
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
            if len(qdrant_points) >= args.batch:
                upsert_qdrant(qdrant_points, args.collection)
                qdrant_points = []
            if len(meili_docs) >= args.batch * 2:
                add_meili_docs(args.index, meili_docs)
                meili_docs = []

    if qdrant_points:
        upsert_qdrant(qdrant_points, args.collection)
    if meili_docs:
        add_meili_docs(args.index, meili_docs)

    print("Ingestion complete")


if __name__ == "__main__":
    main()
