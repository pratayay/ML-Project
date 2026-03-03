"""Tests for moderation function contract."""

from adaptiveguard.api.main import moderate
from adaptiveguard.api.schemas import ModerateRequest


def test_moderate_returns_attack_signals_in_risk_logs() -> None:
    response = moderate(
        ModerateRequest(
            text="y0u ar3 stuuuuupid",
            risk_score=0.6,
            category="abuse",
            strictness=0.5,
            policy_weights={},
        )
    )

    assert response.risk_logs
    attack_signals = response.risk_logs[0].attack_signals
    assert attack_signals.obfuscation_score >= 0.0
    assert attack_signals.typo_density >= 0.0
    assert isinstance(attack_signals.anomaly_flags, list)
