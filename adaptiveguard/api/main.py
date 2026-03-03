"""API entrypoint for AdaptiveGuard."""

from fastapi import FastAPI

from adaptiveguard.api.policy_engine import apply_category_weight, calibrate_score, decision
from adaptiveguard.api.schemas import ModerateRequest, ModerateResponse, PolicyConfig

app = FastAPI(title="AdaptiveGuard API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Simple liveness endpoint for service checks."""
    return {"status": "ok"}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _get_policy_config() -> PolicyConfig:
    configured = getattr(app.state, "policy_config", None)
    if isinstance(configured, PolicyConfig):
        return configured
    return PolicyConfig()


def _resolve_strictness(request_payload: ModerateRequest, config: PolicyConfig) -> float:
    if "strictness" in request_payload.model_fields_set:
        return request_payload.strictness
    return config.strictness


def _resolve_weights(request_payload: ModerateRequest, config: PolicyConfig) -> dict[str, float]:
    if "policy_weights" in request_payload.model_fields_set:
        return request_payload.policy_weights
    return config.policy_weights


def _compute_confidence(score: float, strictness: float) -> float:
    bounded_strictness = _clamp(strictness)
    review_threshold = 0.35 + 0.30 * bounded_strictness
    block_threshold = 0.65 + 0.30 * bounded_strictness
    distance_to_boundary = min(abs(score - review_threshold), abs(score - block_threshold))
    return _clamp(0.5 + distance_to_boundary)


def _moderate(payload: ModerateRequest, config: PolicyConfig) -> ModerateResponse:
    strictness = _resolve_strictness(payload, config)
    weights = _resolve_weights(payload, config)

    weighted_score = apply_category_weight(payload.risk_score, payload.category, weights)
    final_score = calibrate_score(weighted_score)
    moderation_decision = decision(final_score, strictness)
    confidence = _compute_confidence(final_score, strictness)

    return ModerateResponse(
        final_score=final_score,
        decision=moderation_decision,
        confidence=confidence,
        policy_version=config.policy_version,
    )


@app.post("/moderate", response_model=ModerateResponse)
def moderate(request_payload: ModerateRequest) -> ModerateResponse:
    """Moderate a single payload using policy engine utilities."""
    config = _get_policy_config()
    return _moderate(request_payload, config)


@app.post("/batch_moderate", response_model=list[ModerateResponse])
def batch_moderate(request_payloads: list[ModerateRequest]) -> list[ModerateResponse]:
    """Moderate a batch of payloads using policy engine utilities."""
    config = _get_policy_config()
    return [_moderate(payload, config) for payload in request_payloads]
