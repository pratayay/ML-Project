"""Scheduled drift monitoring jobs and auto-actions.

This module evaluates drift signals against threshold registry values,
persists drift outcomes, and triggers operational actions.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REGISTRY_PATH = Path(__file__).with_name("drift_threshold_registry.yaml")

# Mirror the registry to avoid adding a yaml dependency.
THRESHOLD_REGISTRY: dict[str, dict[str, float]] = {
    "risk_mean_shift": {"warning": 0.08, "critical": 0.15},
    "category_shift": {"warning": 0.12, "critical": 0.20},
    "psi": {"warning": 0.20, "critical": 0.35},
    "kl_divergence": {"warning": 0.10, "critical": 0.20},
}

ALERT_DESTINATIONS: dict[str, list[str]] = {
    "warning": ["slack:#ml-ops-alerts", "pagerduty:ml-risk-warning"],
    "critical": [
        "slack:#incident-ml-platform",
        "pagerduty:ml-risk-critical",
        "email:ml-oncall@example.com",
    ],
}



def utc_now() -> str:
    """Return a UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def evaluate_severity(signal_name: str, value: float) -> str | None:
    """Return warning/critical when threshold is met, otherwise None."""
    thresholds = THRESHOLD_REGISTRY[signal_name]
    if value >= thresholds["critical"]:
        return "critical"
    if value >= thresholds["warning"]:
        return "warning"
    return None


def ensure_tables(conn: sqlite3.Connection) -> None:
    """Create storage tables used by drift jobs if they do not exist."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS model_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            version TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            drift_state TEXT NOT NULL DEFAULT 'normal',
            freeze_auto_updates INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS drift_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_ts TEXT NOT NULL,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            signal_name TEXT NOT NULL,
            signal_value REAL NOT NULL,
            severity TEXT NOT NULL,
            alert_destinations TEXT NOT NULL,
            auto_actions TEXT NOT NULL,
            diagnostics_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS shadow_evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            requested_at TEXT NOT NULL,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            dataset_window TEXT NOT NULL,
            status TEXT NOT NULL,
            reason TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS retraining_tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            opened_at TEXT NOT NULL,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            status TEXT NOT NULL,
            diagnostics_json TEXT NOT NULL
        );
        """
    )


def _ensure_model_version(conn: sqlite3.Connection, model_name: str, model_version: str) -> None:
    now = utc_now()
    conn.execute(
        """
        INSERT INTO model_versions (model_name, version, status, drift_state, freeze_auto_updates, updated_at)
        SELECT ?, ?, 'active', 'normal', 0, ?
        WHERE NOT EXISTS (
            SELECT 1 FROM model_versions WHERE model_name = ? AND version = ?
        )
        """,
        (model_name, model_version, now, model_name, model_version),
    )


def trigger_shadow_evaluation(
    conn: sqlite3.Connection,
    model_name: str,
    model_version: str,
    reason: str,
) -> None:
    """Queue a shadow evaluation on recent data."""
    conn.execute(
        """
        INSERT INTO shadow_evaluations (
            requested_at, model_name, model_version, dataset_window, status, reason
        ) VALUES (?, ?, ?, 'recent_7d', 'queued', ?)
        """,
        (utc_now(), model_name, model_version, reason),
    )


def freeze_policy_auto_updates(conn: sqlite3.Connection, model_name: str, model_version: str) -> None:
    """Freeze policy auto-updates when critical drift is detected."""
    conn.execute(
        """
        UPDATE model_versions
        SET freeze_auto_updates = 1,
            drift_state = 'critical',
            updated_at = ?
        WHERE model_name = ? AND version = ?
        """,
        (utc_now(), model_name, model_version),
    )


def open_retraining_ticket(
    conn: sqlite3.Connection,
    model_name: str,
    model_version: str,
    diagnostics: dict[str, Any],
) -> None:
    """Open a retraining ticket with attached diagnostics."""
    conn.execute(
        """
        INSERT INTO retraining_tickets (opened_at, model_name, model_version, status, diagnostics_json)
        VALUES (?, ?, ?, 'open', ?)
        """,
        (utc_now(), model_name, model_version, json.dumps(diagnostics, sort_keys=True)),
    )


def run_drift_job(
    conn: sqlite3.Connection,
    model_name: str,
    model_version: str,
    signal_values: dict[str, float],
) -> list[dict[str, Any]]:
    """Evaluate drift, persist events, and perform auto-actions.

    Returns inserted event payloads to support tests and local runs.
    """
    ensure_tables(conn)
    _ensure_model_version(conn, model_name, model_version)

    events: list[dict[str, Any]] = []
    highest_severity = None

    for signal_name, value in signal_values.items():
        severity = evaluate_severity(signal_name, value)
        if severity is None:
            continue

        if severity == "critical":
            highest_severity = "critical"
        elif highest_severity is None:
            highest_severity = "warning"

        actions = ["trigger_shadow_evaluation"]
        if severity == "critical":
            actions.extend(["freeze_policy_auto_updates", "open_retraining_ticket"])

        diagnostics = {
            "registry": str(REGISTRY_PATH),
            "signal": signal_name,
            "value": value,
            "thresholds": THRESHOLD_REGISTRY[signal_name],
            "detected_at": utc_now(),
        }

        conn.execute(
            """
            INSERT INTO drift_events (
                event_ts, model_name, model_version, signal_name, signal_value,
                severity, alert_destinations, auto_actions, diagnostics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now(),
                model_name,
                model_version,
                signal_name,
                value,
                severity,
                json.dumps(ALERT_DESTINATIONS[severity]),
                json.dumps(actions),
                json.dumps(diagnostics, sort_keys=True),
            ),
        )

        trigger_shadow_evaluation(
            conn,
            model_name=model_name,
            model_version=model_version,
            reason=f"{severity} drift detected for {signal_name}",
        )
        if severity == "critical":
            freeze_policy_auto_updates(conn, model_name, model_version)
            open_retraining_ticket(conn, model_name, model_version, diagnostics)

        events.append(
            {
                "signal_name": signal_name,
                "signal_value": value,
                "severity": severity,
                "auto_actions": actions,
            }
        )

    if highest_severity is None:
        conn.execute(
            """
            UPDATE model_versions
            SET drift_state = 'normal', updated_at = ?
            WHERE model_name = ? AND version = ?
            """,
            (utc_now(), model_name, model_version),
        )
    elif highest_severity == "warning":
        conn.execute(
            """
            UPDATE model_versions
            SET drift_state = 'warning', updated_at = ?
            WHERE model_name = ? AND version = ?
            """,
            (utc_now(), model_name, model_version),
        )

    conn.commit()
    return events


def run_scheduled_jobs(
    db_path: str,
    model_name: str,
    model_version: str,
    signal_values: dict[str, float],
    interval_seconds: int,
    iterations: int,
) -> None:
    """Run drift job on a fixed schedule."""
    for idx in range(iterations):
        with sqlite3.connect(db_path) as conn:
            run_drift_job(conn, model_name, model_version, signal_values)
        if idx < iterations - 1:
            time.sleep(interval_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scheduled drift monitoring job.")
    parser.add_argument("--db-path", default="monitoring/drift_monitoring.db")
    parser.add_argument("--model-name", default="risk_model")
    parser.add_argument("--model-version", default="v1")
    parser.add_argument("--risk-mean-shift", type=float, default=0.0)
    parser.add_argument("--category-shift", type=float, default=0.0)
    parser.add_argument("--psi", type=float, default=0.0)
    parser.add_argument("--kl-divergence", type=float, default=0.0)
    parser.add_argument("--interval-seconds", type=int, default=3600)
    parser.add_argument("--iterations", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signal_values = {
        "risk_mean_shift": args.risk_mean_shift,
        "category_shift": args.category_shift,
        "psi": args.psi,
        "kl_divergence": args.kl_divergence,
    }
    run_scheduled_jobs(
        db_path=args.db_path,
        model_name=args.model_name,
        model_version=args.model_version,
        signal_values=signal_values,
        interval_seconds=args.interval_seconds,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
