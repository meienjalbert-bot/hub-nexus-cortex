import hashlib
import os
from typing import Iterable


def env(k: str, default: str = "") -> str:
    return os.getenv(k, default)


def file_id(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:16]


def batched(iterable: Iterable, n: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch
