"""API entrypoint for AdaptiveGuard."""

from fastapi import FastAPI

from adaptiveguard.api.policy_engine import apply_category_weight, decision
from adaptiveguard.api.preprocessing import preprocess_text
from adaptiveguard.api.risk_logging import build_risk_log
from adaptiveguard.api.schemas import ModerateRequest, ModerateResponse

app = FastAPI(title="AdaptiveGuard API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Simple liveness endpoint for service checks."""
    return {"status": "ok"}


@app.post("/moderate", response_model=ModerateResponse)
def moderate(payload: ModerateRequest) -> ModerateResponse:
    """Apply deterministic moderation with obfuscation-aware preprocessing metadata."""
    preprocessed = preprocess_text(payload.text)
    weighted_score = apply_category_weight(payload.risk_score, payload.category, payload.policy_weights)

    score_with_obfuscation = min(1.0, weighted_score + preprocessed.obfuscation_score * 0.15)
    result_decision = decision(score_with_obfuscation, payload.strictness)

    confidence = min(1.0, max(0.0, 1.0 - abs(score_with_obfuscation - 0.5)))

    return ModerateResponse(
        final_score=score_with_obfuscation,
        decision=result_decision,
        confidence=confidence,
        policy_version="v1",
        risk_logs=[build_risk_log(preprocessed)],
    )
