"""Deterministic policy engine utilities."""

import math
from typing import Final


_SUPPORTED_CALIBRATION_METHODS: Final[set[str]] = {
    "none",
    "identity",
    "temperature",
    "bayesian",
}


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
    """Calibrate a score using a selected method.

    Supported methods:
    - ``none`` / ``identity``: no-op except clamping into ``[0, 1]``.
    - ``temperature``: scales log-odds by ``1 / temperature`` and maps back
      with a sigmoid. ``temperature > 1`` flattens confidence and
      ``temperature < 1`` sharpens it.
    - ``bayesian``: Beta posterior mean using ``score`` as a soft Bernoulli
      observation with prior ``Beta(prior_alpha, prior_beta)``:
      ``(score + prior_alpha) / (1 + prior_alpha + prior_beta)``.

    Numerical behavior:
    - Inputs are clamped to ``[0, 1]`` before calibration.
    - Temperature calibration clips probability away from exact ``0`` and
      ``1`` in logit-space for numerical stability, then returns a clamped
      value in ``[0, 1]``.
    - Unsupported methods and invalid parameters raise ``ValueError``.
    """
    bounded_score = _clamp(score)
    normalized_method = (method or "none").lower()

    if normalized_method not in _SUPPORTED_CALIBRATION_METHODS:
        supported = ", ".join(sorted(_SUPPORTED_CALIBRATION_METHODS))
        raise ValueError(f"Unsupported calibration method '{method}'. Supported methods: {supported}.")

    if normalized_method in {"none", "identity"}:
        return bounded_score

    if normalized_method == "temperature":
        if temperature <= 0:
            raise ValueError("temperature must be > 0 for temperature calibration.")

        epsilon = 1e-12
        stable_score = min(max(bounded_score, epsilon), 1.0 - epsilon)
        logit = math.log(stable_score / (1.0 - stable_score))
        calibrated = 1.0 / (1.0 + math.exp(-(logit / temperature)))
        return _clamp(calibrated)

    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("prior_alpha and prior_beta must be > 0 for bayesian calibration.")

    posterior_mean = (bounded_score + prior_alpha) / (1.0 + prior_alpha + prior_beta)
    return _clamp(posterior_mean)


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
