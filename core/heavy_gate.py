from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

# Heuristiques simples pour repérer les modèles "lourds"
HEAVY_HINTS = ("32b", "70b", "72b", "qwen32b", "mixtral-8x7b")


def is_heavy_model(name: str) -> bool:
    n = (name or "").lower()
    return any(h in n for h in HEAVY_HINTS)


class HeavyGate:
    """Sémaphore globale pour limiter la concurrence des modèles lourds."""

    def __init__(self, max_heavy: int = 1) -> None:
        # Une seule requête lourde à la fois par défaut
        self._sem = asyncio.Semaphore(max_heavy)

    @asynccontextmanager
    async def section(self, model: str):
        """Protège une section si le modèle est 'lourd'."""
        if is_heavy_model(model):
            async with self._sem:
                yield
        else:
            # Pas de limite pour les modèles légers
            yield


heavy_gate = HeavyGate()
