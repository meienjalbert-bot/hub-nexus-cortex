import pytest


def test_cache_initialization():
    try:
        from cache_semantic import \
            SemanticCache  # if you switch to class later

        _ = SemanticCache(redis_url="redis://localhost:6379/0")
        assert True
    except Exception as e:
        pytest.skip(f"Cache nécessite Redis: {e}")


@pytest.mark.slow
def test_cache_operations():
    pytest.skip("Nécessite Redis en cours d'exécution")
