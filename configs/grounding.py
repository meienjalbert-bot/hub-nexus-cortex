# core/grounding.py
from pathlib import Path

import yaml

_GLOSS = None


def _load_glossary(path: str = "configs/glossary.yaml") -> dict:
    global _GLOSS
    if _GLOSS is None:
        _GLOSS = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return _GLOSS


def make_context(user_context: str = "", extra_terms=None) -> str:
    g = _load_glossary()
    parts = []
    terms = extra_terms or []
    for t in terms:
        data = g.get("terms", {}).get(t)
        if data:
            parts.append(f"- {data['name']} ({data['full']}): {data['definition']}")
    constraints = g.get("notes", {}).get("constraints", "")
    ux = f"\n[Contexte utilisateur]\n{user_context.strip()}" if user_context else ""
    return (
        "[Glossaire projet]\n"
        + "\n".join(parts)
        + "\n"
        + ux
        + (f"\n[Contraintes]\n{constraints}" if constraints else "")
    )
