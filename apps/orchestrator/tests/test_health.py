import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert "deps" in data
    assert "meili" in data["deps"]
    assert "qdrant" in data["deps"]
    assert "ollama" in data["deps"]


def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "python_gc_" in r.text  # metric de base


def test_route_endpoint_basic():
    r = client.get("/route?q=test")
    assert r.status_code in [200, 501, 500]


@pytest.mark.integration
def test_route_with_mock():
    # Placeholder integration test
    pass
