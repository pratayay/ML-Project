"""Baseline tests for API health endpoint."""

from fastapi.testclient import TestClient

from adaptiveguard.api.main import app


def test_app_import_and_startup_contract() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
