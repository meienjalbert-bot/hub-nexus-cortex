# apps/orchestrator/tests/test_consensus.py

import pytest

# On importe la version patchée
from core import consensus as C


@pytest.mark.asyncio
async def test_vote_majority_cluster(monkeypatch, tmp_path):
    # 1) fake model_manager
    async def fake_prewarm(models):
        return None

    async def fake_generate(model, prompt, max_tokens, temperature, timeout_s):
        if "analyst" in prompt:
            return "Le MoME combine plusieurs experts pour converger."
        if "creative" in prompt:
            return "MoME, c'est un jury de spécialistes qui accorde les violons."
        if "Synthétise et consolide" in prompt:
            return "MoME agrège des avis d'experts en une réponse fiable et concise."
        return "Réponse neutre."

    monkeypatch.setattr(C, "prewarm", fake_prewarm)
    monkeypatch.setattr(C, "generate", fake_generate)

    # 2) config YAML éphémère
    cfg = tmp_path / "consensus_models.yaml"
    cfg.write_text(
        """
committee:
  - role: analyst
    model: a
  - role: creative
    model: b
conductor:
  model: a
""",
        encoding="utf-8",
    )

    out = await C.vote("Qu'est-ce que MoME ?", "Contexte", str(cfg))
    assert out["final_answer"]
    assert 0.5 <= out["confidence"] <= 1.0
    assert len(out["votes"]) == 2


@pytest.mark.asyncio
async def test_vote_handles_failures(monkeypatch, tmp_path):
    async def fake_prewarm(models):
        return None

    async def fake_generate(model, prompt, max_tokens, temperature, timeout_s):
        if model == "bad":
            raise RuntimeError("boom")
        return "Réponse ok"

    monkeypatch.setattr(C, "prewarm", fake_prewarm)
    monkeypatch.setattr(C, "generate", fake_generate)

    cfg = tmp_path / "consensus_models.yaml"
    cfg.write_text(
        """
committee:
  - role: a
    model: good
  - role: b
    model: bad
""",
        encoding="utf-8",
    )

    out = await C.vote("Q?", "ctx", str(cfg))
    # 1 succès, 1 échec → on doit quand même renvoyer une réponse
    assert out["votes"]  # liste non vide
    assert "errors" in out and out["errors"]


@pytest.mark.asyncio
async def test_vote_no_config(monkeypatch):
    async def fake_prewarm(models):
        return None

    async def fake_generate(model, prompt, max_tokens, temperature, timeout_s):
        return "Réponse fallback"

    monkeypatch.setattr(C, "prewarm", fake_prewarm)
    monkeypatch.setattr(C, "generate", fake_generate)

    out = await C.vote("Q?", "ctx", config_path="/does/not/exist.yaml")
    assert out["final_answer"]
    assert out["confidence"] >= 0.5
