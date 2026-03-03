"""API entrypoint for AdaptiveGuard."""

from __future__ import annotations

from collections import Counter
from time import perf_counter

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from adaptiveguard.api.policy_engine import apply_category_weight, calibrate_score, decision
from adaptiveguard.api.schemas import (
    BatchModerateRequest,
    BatchModerateResponse,
    ModerateRequest,
    ModerationResult,
)
from adaptiveguard.model.risk_model import RiskModel

app = FastAPI(title="AdaptiveGuard API", version="0.1.0")
model = RiskModel()

REQUEST_COUNTER = Counter()
DECISION_COUNTER = Counter()
LATENCY_SUM = 0.0
LATENCY_COUNT = 0


def _build_confidence(score: float) -> float:
    return round(abs(score - 0.5) * 2.0, 6)


def _moderate_item(request: ModerateRequest) -> ModerationResult:
    raw_risk = model.predict(request.features)
    weighted_risk = apply_category_weight(raw_risk, request.category, request.policy_weights)
    calibrated_risk = calibrate_score(weighted_risk)
    moderation_decision = decision(calibrated_risk, request.strictness)

    return ModerationResult(
        risk_score=calibrated_risk,
        decision=moderation_decision,
        strictness=request.strictness,
        category=request.category,
        confidence=_build_confidence(calibrated_risk),
        policy_version=request.policy_version,
        model_version=model.model_version,
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Simple liveness endpoint for service checks."""
    return {"status": "ok"}


@app.post("/moderate", response_model=ModerationResult)
def moderate(payload: ModerateRequest) -> ModerationResult:
    global LATENCY_SUM, LATENCY_COUNT

    start = perf_counter()
    result = _moderate_item(payload)
    elapsed = perf_counter() - start

    REQUEST_COUNTER["moderate"] += 1
    DECISION_COUNTER[result.decision] += 1
    LATENCY_SUM += elapsed
    LATENCY_COUNT += 1

    return result


@app.post("/batch_moderate", response_model=BatchModerateResponse)
def batch_moderate(payload: BatchModerateRequest) -> BatchModerateResponse:
    global LATENCY_SUM, LATENCY_COUNT

    start = perf_counter()
    results = [_moderate_item(item) for item in payload.items]
    elapsed = perf_counter() - start

    REQUEST_COUNTER["batch_moderate"] += 1
    for result in results:
        DECISION_COUNTER[result.decision] += 1
    LATENCY_SUM += elapsed
    LATENCY_COUNT += 1

    return BatchModerateResponse(results=results)


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    """Prometheus text exposition for basic service metrics."""
    lines = [
        "# HELP adaptiveguard_requests_total Total API requests by endpoint.",
        "# TYPE adaptiveguard_requests_total counter",
    ]
    for endpoint, count in REQUEST_COUNTER.items():
        lines.append(f'adaptiveguard_requests_total{{endpoint="{endpoint}"}} {count}')

    lines.extend(
        [
            "# HELP adaptiveguard_decisions_total Total moderation decisions by label.",
            "# TYPE adaptiveguard_decisions_total counter",
        ]
    )
    for label, count in DECISION_COUNTER.items():
        lines.append(f'adaptiveguard_decisions_total{{decision="{label}"}} {count}')

    lines.extend(
        [
            "# HELP adaptiveguard_moderation_latency_seconds_sum Total moderation latency in seconds.",
            "# TYPE adaptiveguard_moderation_latency_seconds_sum counter",
            f"adaptiveguard_moderation_latency_seconds_sum {LATENCY_SUM}",
            "# HELP adaptiveguard_moderation_latency_seconds_count Total moderation latency observations.",
            "# TYPE adaptiveguard_moderation_latency_seconds_count counter",
            f"adaptiveguard_moderation_latency_seconds_count {LATENCY_COUNT}",
        ]
    )

    return "\n".join(lines) + "\n"
