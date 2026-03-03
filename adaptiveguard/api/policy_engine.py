"""Deterministic policy engine utilities."""


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def apply_category_weight(score: float, category: str, weights: dict[str, float]) -> float:
    """Apply category-specific weight multiplier to a score."""
    weight = weights.get(category, 1.0)
    return _clamp(score * weight)


def calibrate_score(
    score: float,
    method: str | None = None,
    temperature: float = 1.0,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> float:
    """Calibration extension point.

    Placeholder implementation to keep a stable API while future
    temperature or Bayesian calibration methods are added.
    """
    _ = (method, temperature, prior_alpha, prior_beta)
    return _clamp(score)


def decision(score: float, strictness: float) -> str:
    """Map score to deterministic moderation decision."""
    bounded_strictness = _clamp(strictness)
    review_threshold = 0.35 + 0.30 * bounded_strictness
    block_threshold = 0.65 + 0.30 * bounded_strictness

    if score < review_threshold:
        return "allow"
    if score < block_threshold:
        return "review"
    return "block"
