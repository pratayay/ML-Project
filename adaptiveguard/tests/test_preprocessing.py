"""Tests for obfuscation-aware preprocessing and risk logging."""

from adaptiveguard.api.preprocessing import preprocess_text
from adaptiveguard.api.risk_logging import build_risk_log


def test_preprocess_text_detects_multiple_obfuscation_strategies() -> None:
    result = preprocess_text("h3llllo yоu")

    assert result.leetspeak_decoded >= 1
    assert result.repeated_collapses >= 1
    assert result.homoglyph_replacements >= 1
    assert "homoglyph_substitution" in result.anomaly_flags
    assert result.obfuscation_score > 0


def test_risk_log_uses_safe_diff_metadata_without_raw_text() -> None:
    result = preprocess_text("s3cr3t")
    risk_log = build_risk_log(result)

    transformation_diff = risk_log["transformation_diff"]
    assert "before_sha256_8" in transformation_diff
    assert "after_sha256_8" in transformation_diff
    assert "s3cr3t" not in str(transformation_diff)

    attack_signals = risk_log["attack_signals"]
    assert set(attack_signals) == {"obfuscation_score", "typo_density", "anomaly_flags"}
