from typing import Dict, List


def chunk_text(text: str, chunk_chars: int = 2000, overlap: int = 200) -> List[Dict]:
    text = text.replace("\x00", " ")
    n = len(text)
    chunks = []
    start = 0
    while start < n:
        end = min(n, start + chunk_chars)
        chunk = text[start:end]
        chunks.append({"text": chunk, "start": start, "end": end})
        if end == n:
            break
        start = max(end - overlap, start + 1)
    return chunks
