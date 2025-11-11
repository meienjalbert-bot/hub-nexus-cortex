# tests/test_mome_router.py
from core.mome_router import run_mome


def test_run_mome_basic():
    result = run_mome("Qu'est-ce que la terre ?", k=3)
    assert "answer" in result
    assert "sources" in result
    assert isinstance(result["sources"], list)
    assert result["query_type"] in ["factual", "conceptual", "recent", "default"]
    assert result["fusion_method"] == "rrf_adaptive"

    # sources non vides
    assert len(result["sources"]) > 0
    for src in result["sources"]:
        assert "text" in src
        assert "expert" in src
        assert "final_score" in src

    # answer contient query
    assert "terre" in result["answer"].lower()
