"""Tests for robustness benchmark risk-delta reporting."""

from pathlib import Path

from adaptiveguard.evaluation.robustness_benchmark import risk_delta_distribution


def test_risk_delta_distribution_reports_per_strictness() -> None:
    dataset = Path(__file__).resolve().parents[1] / "evaluation" / "data" / "robustness_benchmark.jsonl"
    report = risk_delta_distribution(dataset)

    assert set(report.keys()) == {0.2, 0.5, 0.8}
    for strictness_report in report.values():
        assert strictness_report["count"] > 0
        assert "mean" in strictness_report
        assert "p90" in strictness_report
