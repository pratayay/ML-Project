"""Endpoint tests for moderation APIs."""

import pytest
from pydantic import ValidationError

from adaptiveguard.api.main import batch_moderate, metrics, moderate
from adaptiveguard.api.schemas import BatchModerateRequest, ModerateRequest


def test_single_item_moderation_returns_canonical_schema() -> None:
    payload = ModerateRequest(
        category="violence",
        strictness=0.6,
        features={"text": "example"},
        policy_weights={"violence": 1.1},
        policy_version="v2",
    )

    body = moderate(payload).model_dump()

    assert body == {
        "risk_score": 0.55,
        "decision": "review",
        "strictness": 0.6,
        "category": "violence",
        "confidence": 0.1,
        "policy_version": "v2",
        "model_version": "0.0.1",
    }


def test_batch_moderate_preserves_shape_and_ordering() -> None:
    payload = BatchModerateRequest(
        items=[
            ModerateRequest(category="spam", strictness=0.1, policy_weights={"spam": 0.5}),
            ModerateRequest(
                category="self_harm",
                strictness=0.9,
                policy_weights={"self_harm": 1.4},
            ),
        ]
    )

    body = batch_moderate(payload).model_dump()

    assert list(body.keys()) == ["results"]
    assert len(body["results"]) == 2
    assert body["results"][0]["category"] == "spam"
    assert body["results"][0]["risk_score"] == 0.25
    assert body["results"][1]["category"] == "self_harm"
    assert body["results"][1]["risk_score"] == 0.7


def test_invalid_strictness_and_category_return_validation_errors() -> None:
    with pytest.raises(ValidationError):
        ModerateRequest(category="spam", strictness=1.4)

    with pytest.raises(ValidationError):
        ModerateRequest(category="not-a-category", strictness=0.5)


def test_metrics_endpoint_exposes_prometheus_format() -> None:
    output = metrics()

    assert "# TYPE adaptiveguard_requests_total counter" in output
    assert "adaptiveguard_requests_total" in output
