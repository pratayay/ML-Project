"""Unit tests for deterministic policy engine behavior."""

import pytest

from adaptiveguard.api.policy_engine import apply_category_weight, decision


def test_decision_boundary_review_threshold() -> None:
    strictness = 0.5
    review_threshold = 0.35 + 0.30 * strictness

    assert decision(review_threshold - 0.0001, strictness) == "allow"
    assert decision(review_threshold, strictness) == "review"


def test_decision_boundary_block_threshold() -> None:
    strictness = 0.5
    block_threshold = 0.65 + 0.30 * strictness

    assert decision(block_threshold - 0.0001, strictness) == "review"
    assert decision(block_threshold, strictness) == "block"


def test_apply_category_weight_uses_mapping_when_present() -> None:
    weighted = apply_category_weight(
        score=0.4,
        category="violence",
        weights={"violence": 1.5, "spam": 0.5},
    )

    assert weighted == pytest.approx(0.6)


def test_apply_category_weight_defaults_to_identity_when_missing() -> None:
    weighted = apply_category_weight(score=0.4, category="unknown", weights={"spam": 0.5})

    assert weighted == 0.4


def test_apply_category_weight_is_clamped() -> None:
    weighted = apply_category_weight(score=0.9, category="violence", weights={"violence": 2.0})

    assert weighted == 1.0
