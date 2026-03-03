# Policy Engine Specification

## Purpose
The policy engine converts a model-provided risk score into a deterministic moderation decision that can be versioned, audited, and calibrated over time.

## Inputs
The engine consumes the following inputs:

- `risk_score` (`float`, expected range `0.0..1.0`): baseline risk estimate from an upstream model.
- `category` (`str`): semantic label for the request content (for example: `violence`, `self_harm`, `spam`).
- `strictness` (`float`, expected range `0.0..1.0`): controls decision thresholds; larger values produce stricter outcomes.
- `policy_weights` (`dict[str, float]`): category-specific multipliers applied before thresholding.

## Outputs
The engine returns a moderation payload with:

- `final_score` (`float`): weighted (and optionally calibrated) score used for decisioning.
- `decision` (`str`): one of `allow`, `review`, or `block`.
- `confidence` (`float`): deterministic confidence estimate for downstream UX/logging.
- `policy_version` (`str`): version identifier for reproducibility and audit trails.

## Deterministic Decision Rule
1. Apply category weighting:
   - `weighted_score = clamp(risk_score * policy_weights.get(category, 1.0), 0.0, 1.0)`
2. Optionally calibrate (`calibrate_score`) to adjust score reliability while preserving deterministic behavior for a fixed configuration.
3. Compute strictness-adjusted thresholds:
   - `review_threshold = 0.35 + 0.30 * strictness`
   - `block_threshold = 0.65 + 0.30 * strictness`
4. Emit decision:
   - `weighted_or_calibrated_score < review_threshold` -> `allow`
   - `review_threshold <= score < block_threshold` -> `review`
   - `score >= block_threshold` -> `block`

This rule is deterministic because all thresholds and transformations are pure functions of explicit inputs.

## Calibration Hook Points
Calibration is intentionally extensible and should execute between weighting and thresholding.

- **Temperature scaling hook**
  - Signature example: `calibrate_score(score, method="temperature", temperature=1.0)`
  - Usage: post-hoc score sharpening/flattening with versioned temperature values.

- **Bayesian calibration hook**
  - Signature example: `calibrate_score(score, method="bayesian", prior_alpha=1.0, prior_beta=1.0)`
  - Usage: incorporate prior uncertainty and low-sample corrections.

Both hooks must be tied to `policy_version` so historical decisions remain reproducible.
