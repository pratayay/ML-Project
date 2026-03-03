"""Endpoint tests for moderation APIs."""

from fastapi.testclient import TestClient

from adaptiveguard.api.main import app


client = TestClient(app)


def test_moderate_valid_request_path() -> None:
    response = client.post(
        "/moderate",
        json={
            "risk_score": 0.6,
            "category": "violence",
            "strictness": 0.5,
            "policy_weights": {"violence": 1.0},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["final_score"] == 0.6
    assert body["decision"] == "review"
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["policy_version"] == "v1"


def test_moderate_strictness_boundary_behavior() -> None:
    response_at_boundary = client.post(
        "/moderate",
        json={
            "risk_score": 0.5,
            "category": "spam",
            "strictness": 0.5,
            "policy_weights": {"spam": 1.0},
        },
    )
    response_below_boundary = client.post(
        "/moderate",
        json={
            "risk_score": 0.4999,
            "category": "spam",
            "strictness": 0.5,
            "policy_weights": {"spam": 1.0},
        },
    )

    assert response_at_boundary.status_code == 200
    assert response_below_boundary.status_code == 200
    assert response_at_boundary.json()["decision"] == "review"
    assert response_below_boundary.json()["decision"] == "allow"


def test_moderate_default_weight_behavior() -> None:
    response = client.post(
        "/moderate",
        json={
            "risk_score": 0.4,
            "category": "unknown_category",
            "strictness": 0.4,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["final_score"] == 0.4


def test_batch_moderate_response_shape() -> None:
    response = client.post(
        "/batch_moderate",
        json=[
            {
                "risk_score": 0.2,
                "category": "spam",
                "strictness": 0.5,
                "policy_weights": {"spam": 1.0},
            },
            {
                "risk_score": 0.9,
                "category": "violence",
                "strictness": 0.5,
                "policy_weights": {"violence": 1.0},
            },
        ],
    )

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) == 2
    for item in body:
        assert set(item.keys()) == {"final_score", "decision", "confidence", "policy_version"}
