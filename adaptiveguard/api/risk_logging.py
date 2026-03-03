"""Risk-log construction utilities."""

from __future__ import annotations

from adaptiveguard.api.preprocessing import PreprocessingResult


def build_risk_log(preprocessed: PreprocessingResult) -> dict[str, object]:
    """Return structured risk log payload without raw text."""
    return {
        "transformation_diff": preprocessed.diff_metadata,
        "attack_signals": {
            "obfuscation_score": preprocessed.obfuscation_score,
            "typo_density": preprocessed.typo_density,
            "anomaly_flags": preprocessed.anomaly_flags,
        },
    }
