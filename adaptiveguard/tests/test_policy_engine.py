"""Unit tests for deterministic policy engine behavior."""

import pytest

from adaptiveguard.api.policy_engine import apply_category_weight, calibrate_score, decision


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


def test_calibrate_score_identity_and_none_methods() -> None:
    assert calibrate_score(0.25, method="identity") == pytest.approx(0.25)
    assert calibrate_score(0.25, method="none") == pytest.approx(0.25)
    assert calibrate_score(0.25, method=None) == pytest.approx(0.25)


def test_calibrate_score_temperature_monotonic() -> None:
    lower = calibrate_score(0.2, method="temperature", temperature=2.0)
    upper = calibrate_score(0.8, method="temperature", temperature=2.0)

    assert lower < upper


def test_calibrate_score_temperature_behavior() -> None:
    high_sharpened = calibrate_score(0.8, method="temperature", temperature=0.5)
    high_softened = calibrate_score(0.8, method="temperature", temperature=2.0)
    low_sharpened = calibrate_score(0.2, method="temperature", temperature=0.5)
    low_softened = calibrate_score(0.2, method="temperature", temperature=2.0)

    assert high_sharpened > 0.8 > high_softened
    assert low_sharpened < 0.2 < low_softened


def test_calibrate_score_bayesian_monotonic() -> None:
    lower = calibrate_score(0.2, method="bayesian", prior_alpha=2.0, prior_beta=3.0)
    upper = calibrate_score(0.8, method="bayesian", prior_alpha=2.0, prior_beta=3.0)

    assert lower < upper


def test_calibrate_score_bayesian_expected_value() -> None:
    calibrated = calibrate_score(0.8, method="bayesian", prior_alpha=2.0, prior_beta=3.0)

    assert calibrated == pytest.approx((0.8 + 2.0) / (1.0 + 2.0 + 3.0))


def test_calibrate_score_boundary_values_are_bounded() -> None:
    methods_and_kwargs = [
        ("none", {}),
        ("temperature", {"temperature": 0.5}),
        ("bayesian", {"prior_alpha": 1.0, "prior_beta": 1.0}),
    ]

    for method, kwargs in methods_and_kwargs:
        for score in (-100.0, 0.0, 1.0, 100.0):
            calibrated = calibrate_score(score, method=method, **kwargs)
            assert 0.0 <= calibrated <= 1.0


def test_calibrate_score_invalid_method_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported calibration method"):
        calibrate_score(0.5, method="unknown")


def test_calibrate_score_invalid_temperature_raises() -> None:
    with pytest.raises(ValueError, match="temperature must be > 0"):
        calibrate_score(0.5, method="temperature", temperature=0.0)


def test_calibrate_score_invalid_bayesian_priors_raises() -> None:
    with pytest.raises(ValueError, match="prior_alpha and prior_beta must be > 0"):
        calibrate_score(0.5, method="bayesian", prior_alpha=0.0, prior_beta=1.0)
