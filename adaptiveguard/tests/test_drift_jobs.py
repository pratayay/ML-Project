"""Tests for drift monitoring scheduled jobs and auto-actions."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path


def _load_module():
    module_path = Path("monitoring/drift_jobs.py")
    spec = importlib.util.spec_from_file_location("drift_jobs", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



def test_registry_backed_configuration_is_loaded() -> None:
    drift_jobs = _load_module()

    assert drift_jobs.THRESHOLD_REGISTRY["risk_mean_shift"] == {"warning": 0.08, "critical": 0.15}
    assert drift_jobs.ALERT_DESTINATIONS["warning"] == [
        "slack:#ml-ops-alerts",
        "pagerduty:ml-risk-warning",
    ]



def test_warning_drift_writes_events_and_shadow_eval() -> None:
    drift_jobs = _load_module()

    with sqlite3.connect(":memory:") as conn:
        events = drift_jobs.run_drift_job(
            conn,
            model_name="risk_model",
            model_version="v1",
            signal_values={
                "risk_mean_shift": 0.09,
                "category_shift": 0.05,
                "psi": 0.1,
                "kl_divergence": 0.01,
            },
        )

        assert len(events) == 1
        assert events[0]["severity"] == "warning"

        version_state = conn.execute(
            "SELECT drift_state, freeze_auto_updates FROM model_versions"
        ).fetchone()
        assert version_state == ("warning", 0)

        shadow_count = conn.execute("SELECT COUNT(*) FROM shadow_evaluations").fetchone()[0]
        ticket_count = conn.execute("SELECT COUNT(*) FROM retraining_tickets").fetchone()[0]
        alert_destinations = conn.execute(
            "SELECT alert_destinations FROM drift_events LIMIT 1"
        ).fetchone()[0]
        assert shadow_count == 1
        assert ticket_count == 0
        assert json.loads(alert_destinations) == drift_jobs.ALERT_DESTINATIONS["warning"]


def test_critical_drift_triggers_all_auto_actions() -> None:
    drift_jobs = _load_module()

    with sqlite3.connect(":memory:") as conn:
        events = drift_jobs.run_drift_job(
            conn,
            model_name="risk_model",
            model_version="v1",
            signal_values={
                "risk_mean_shift": 0.16,
                "category_shift": 0.21,
                "psi": 0.36,
                "kl_divergence": 0.22,
            },
        )

        assert len(events) == 4
        assert all(event["severity"] == "critical" for event in events)

        version_state = conn.execute(
            "SELECT drift_state, freeze_auto_updates FROM model_versions"
        ).fetchone()
        assert version_state == ("critical", 1)

        shadow_count = conn.execute("SELECT COUNT(*) FROM shadow_evaluations").fetchone()[0]
        ticket_count = conn.execute("SELECT COUNT(*) FROM retraining_tickets").fetchone()[0]
        drift_count = conn.execute("SELECT COUNT(*) FROM drift_events").fetchone()[0]
        assert shadow_count == 4
        assert ticket_count == 4
        assert drift_count == 4
