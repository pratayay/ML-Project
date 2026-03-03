"""Tests for drift monitoring scheduled jobs and auto-actions."""

from __future__ import annotations

import importlib.util
import sqlite3


def _load_module():
    spec = importlib.util.spec_from_file_location("drift_jobs", "monitoring/drift_jobs.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sqlite_storage(drift_jobs):
    return drift_jobs.Storage(sqlite3.connect(":memory:"), backend="sqlite")


def test_yaml_driven_thresholds_are_honored(monkeypatch) -> None:
    drift_jobs = _load_module()

    monkeypatch.setattr(
        drift_jobs,
        "_parse_registry_yaml",
        lambda: {
            "signals": {
                "risk_mean_shift": {"warning": 0.5, "critical": 0.7},
                "category_shift": {"warning": 0.6, "critical": 0.8},
                "psi": {"warning": 0.9, "critical": 1.1},
                "kl_divergence": {"warning": 0.4, "critical": 0.6},
            },
            "severity_levels": {
                "warning": {"alert_destinations": ["slack:test-warning"]},
                "critical": {"alert_destinations": ["pagerduty:test-critical"]},
            },
        },
    )

    with _sqlite_storage(drift_jobs) as storage:
        events = drift_jobs.run_drift_job(
            storage,
            model_name="risk_model",
            model_version="v1",
            signal_values={
                "risk_mean_shift": 0.08,
                "category_shift": 0.2,
                "psi": 0.35,
                "kl_divergence": 0.15,
            },
        )

        assert events == []
        drift_count = storage.execute("SELECT COUNT(*) FROM drift_events").fetchone()[0]
        assert drift_count == 0


def test_warning_and_critical_transitions_trigger_expected_actions() -> None:
    drift_jobs = _load_module()

    with _sqlite_storage(drift_jobs) as storage:
        warning_events = drift_jobs.run_drift_job(
            storage,
            model_name="risk_model",
            model_version="v1",
            signal_values={
                "risk_mean_shift": 0.09,
                "category_shift": 0.05,
                "psi": 0.1,
                "kl_divergence": 0.01,
            },
        )

        assert len(warning_events) == 1
        assert warning_events[0]["severity"] == "warning"

        version_state = storage.execute(
            "SELECT drift_state, freeze_auto_updates FROM model_versions"
        ).fetchone()
        assert version_state == ("warning", 0)

        critical_events = drift_jobs.run_drift_job(
            storage,
            model_name="risk_model",
            model_version="v1",
            signal_values={
                "risk_mean_shift": 0.16,
                "category_shift": 0.21,
                "psi": 0.36,
                "kl_divergence": 0.22,
            },
        )

        assert len(critical_events) == 4
        assert all(event["severity"] == "critical" for event in critical_events)

        version_state = storage.execute(
            "SELECT drift_state, freeze_auto_updates FROM model_versions"
        ).fetchone()
        assert version_state == ("critical", 1)

        shadow_count = storage.execute("SELECT COUNT(*) FROM shadow_evaluations").fetchone()[0]
        ticket_count = storage.execute("SELECT COUNT(*) FROM retraining_tickets").fetchone()[0]
        drift_count = storage.execute("SELECT COUNT(*) FROM drift_events").fetchone()[0]
        assert shadow_count == 5
        assert ticket_count == 4
        assert drift_count == 5


def test_missing_required_registry_signal_fails_without_fallback(monkeypatch) -> None:
    drift_jobs = _load_module()

    monkeypatch.setattr(
        drift_jobs,
        "_parse_registry_yaml",
        lambda: {
            "signals": {
                "risk_mean_shift": {"warning": 0.08, "critical": 0.15},
                "category_shift": {"warning": 0.12, "critical": 0.2},
                "psi": {"warning": 0.2, "critical": 0.35},
            },
            "severity_levels": {
                "warning": {"alert_destinations": ["slack:test-warning"]},
                "critical": {"alert_destinations": ["pagerduty:test-critical"]},
            },
        },
    )

    with _sqlite_storage(drift_jobs) as storage:
        try:
            drift_jobs.run_drift_job(
                storage,
                model_name="risk_model",
                model_version="v1",
                signal_values={
                    "risk_mean_shift": 0.09,
                    "category_shift": 0.13,
                    "psi": 0.22,
                    "kl_divergence": 0.11,
                },
            )
        except ValueError as exc:
            assert "missing required signals" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("Expected run_drift_job to fail when registry signal is missing")
