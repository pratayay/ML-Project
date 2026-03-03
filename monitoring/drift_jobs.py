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
from typing import Any, Literal

REGISTRY_PATH = Path(__file__).with_name("drift_threshold_registry.yaml")

REQUIRED_SIGNALS = (
    "risk_mean_shift",
    "category_shift",
    "psi",
    "kl_divergence",
)
REQUIRED_THRESHOLD_KEYS = ("warning", "critical")


class Storage:
    """Small SQL storage abstraction supporting SQLite and PostgreSQL."""

    def __init__(self, connection: Any, backend: Literal["sqlite", "postgres"]):
        self.connection = connection
        self.backend = backend

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> Any:
        return self.connection.execute(self._sql(sql), self._params(params))

    def executescript(self, sql_script: str) -> None:
        if self.backend == "sqlite":
            self.connection.executescript(sql_script)
            return
        statements = [stmt.strip() for stmt in sql_script.split(";") if stmt.strip()]
        for statement in statements:
            self.connection.execute(statement)

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    def _sql(self, sql: str) -> str:
        if self.backend == "postgres":
            return sql.replace("?", "%s")
        return sql

    @staticmethod
    def _params(params: tuple[Any, ...]) -> tuple[Any, ...]:
        return params


def connect_storage(db_target: str) -> Storage:
    """Build a storage adapter for SQLite paths or PostgreSQL DSNs."""
    if db_target.startswith(("postgres://", "postgresql://")):
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - only when postgres is requested
            raise RuntimeError(
                "PostgreSQL target requested but psycopg is not installed. "
                "Install psycopg to run drift jobs against PostgreSQL."
            ) from exc
        conn = psycopg.connect(db_target)
        return Storage(conn, backend="postgres")

    conn = sqlite3.connect(db_target)
    return Storage(conn, backend="sqlite")


def utc_now() -> str:
    """Return a UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _parse_registry_yaml() -> dict[str, Any]:
    """Parse the registry YAML structure used by drift jobs."""
    text = REGISTRY_PATH.read_text(encoding="utf-8")
    root: dict[str, Any] = {}
    section: str | None = None
    signal_name: str | None = None
    severity_name: str | None = None

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        if indent == 0 and line.endswith(":"):
            section = line[:-1]
            root.setdefault(section, {})
            signal_name = None
            severity_name = None
            continue

        if section == "signals":
            if indent == 2 and line.endswith(":"):
                signal_name = line[:-1]
                root[section].setdefault(signal_name, {})
                continue
            if indent == 4 and signal_name and ":" in line:
                key, value = [part.strip() for part in line.split(":", 1)]
                if value:
                    if key in REQUIRED_THRESHOLD_KEYS:
                        root[section][signal_name][key] = float(value)
                    else:
                        root[section][signal_name][key] = value
                continue

        if section == "severity_levels":
            if indent == 2 and line.endswith(":"):
                severity_name = line[:-1]
                root[section].setdefault(severity_name, {})
                continue
            if indent == 4 and line == "alert_destinations:":
                root[section][severity_name]["alert_destinations"] = []
                continue
            if indent == 6 and line.startswith("- ") and severity_name:
                destination = line[2:].strip()
                root[section][severity_name].setdefault("alert_destinations", []).append(destination)
                continue

        if indent == 0 and ":" in line:
            key, value = [part.strip() for part in line.split(":", 1)]
            root[key] = float(value) if value.replace(".", "", 1).isdigit() else value

    return root


def load_registry() -> dict[str, Any]:
    """Load thresholds and alert routes from the YAML registry file."""
    parsed = _parse_registry_yaml()
    validate_registry(parsed)

    signals = {
        name: {
            "warning": float(defn["warning"]),
            "critical": float(defn["critical"]),
        }
        for name, defn in parsed["signals"].items()
    }
    alert_destinations = {
        level: list(defn["alert_destinations"])
        for level, defn in parsed["severity_levels"].items()
    }
    return {"signals": signals, "alert_destinations": alert_destinations}


def validate_registry(registry: dict[str, Any]) -> None:
    """Fail fast when required registry keys are missing or invalid."""
    if "signals" not in registry or not isinstance(registry["signals"], dict):
        raise ValueError("drift threshold registry is missing top-level 'signals' mapping")

    missing_signals = [name for name in REQUIRED_SIGNALS if name not in registry["signals"]]
    if missing_signals:
        raise ValueError(f"drift threshold registry missing required signals: {', '.join(missing_signals)}")

    for signal_name in REQUIRED_SIGNALS:
        signal_def = registry["signals"][signal_name]
        missing_keys = [key for key in REQUIRED_THRESHOLD_KEYS if key not in signal_def]
        if missing_keys:
            raise ValueError(
                f"drift threshold registry signal '{signal_name}' missing required keys: {', '.join(missing_keys)}"
            )
        warning = float(signal_def["warning"])
        critical = float(signal_def["critical"])
        if critical < warning:
            raise ValueError(
                f"drift threshold registry signal '{signal_name}' has critical < warning ({critical} < {warning})"
            )

    if "severity_levels" not in registry or not isinstance(registry["severity_levels"], dict):
        raise ValueError("drift threshold registry is missing top-level 'severity_levels' mapping")

    for severity in ("warning", "critical"):
        level_def = registry["severity_levels"].get(severity)
        if level_def is None:
            raise ValueError(f"drift threshold registry missing severity level '{severity}'")
        if "alert_destinations" not in level_def:
            raise ValueError(
                f"drift threshold registry severity '{severity}' missing 'alert_destinations'"
            )


def evaluate_severity(signal_name: str, value: float, threshold_registry: dict[str, dict[str, float]]) -> str | None:
    """Return warning/critical when threshold is met, otherwise None."""
    thresholds = threshold_registry[signal_name]
    if value >= thresholds["critical"]:
        return "critical"
    if value >= thresholds["warning"]:
        return "warning"
    return None


def ensure_tables(storage: Storage) -> None:
    """Create storage tables used by drift jobs if they do not exist."""
    auto_id = "SERIAL PRIMARY KEY" if storage.backend == "postgres" else "INTEGER PRIMARY KEY AUTOINCREMENT"
    storage.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS model_versions (
            id {auto_id},
            model_name TEXT NOT NULL,
            version TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            drift_state TEXT NOT NULL DEFAULT 'normal',
            freeze_auto_updates INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS drift_events (
            id {auto_id},
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
            id {auto_id},
            requested_at TEXT NOT NULL,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            dataset_window TEXT NOT NULL,
            status TEXT NOT NULL,
            reason TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS retraining_tickets (
            id {auto_id},
            opened_at TEXT NOT NULL,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            status TEXT NOT NULL,
            diagnostics_json TEXT NOT NULL
        );
        """
    )


def _ensure_model_version(storage: Storage, model_name: str, model_version: str) -> None:
    now = utc_now()
    storage.execute(
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
    storage: Storage,
    model_name: str,
    model_version: str,
    reason: str,
) -> None:
    """Queue a shadow evaluation on recent data."""
    storage.execute(
        """
        INSERT INTO shadow_evaluations (
            requested_at, model_name, model_version, dataset_window, status, reason
        ) VALUES (?, ?, ?, 'recent_7d', 'queued', ?)
        """,
        (utc_now(), model_name, model_version, reason),
    )


def freeze_policy_auto_updates(storage: Storage, model_name: str, model_version: str) -> None:
    """Freeze policy auto-updates when critical drift is detected."""
    storage.execute(
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
    storage: Storage,
    model_name: str,
    model_version: str,
    diagnostics: dict[str, Any],
) -> None:
    """Open a retraining ticket with attached diagnostics."""
    storage.execute(
        """
        INSERT INTO retraining_tickets (opened_at, model_name, model_version, status, diagnostics_json)
        VALUES (?, ?, ?, 'open', ?)
        """,
        (utc_now(), model_name, model_version, json.dumps(diagnostics, sort_keys=True)),
    )


def run_drift_job(
    storage: Storage,
    model_name: str,
    model_version: str,
    signal_values: dict[str, float],
) -> list[dict[str, Any]]:
    """Evaluate drift, persist events, and perform auto-actions.

    Returns inserted event payloads to support tests and local runs.
    """
    registry = load_registry()

    ensure_tables(storage)
    _ensure_model_version(storage, model_name, model_version)

    events: list[dict[str, Any]] = []
    highest_severity = None

    for signal_name, value in signal_values.items():
        severity = evaluate_severity(signal_name, value, registry["signals"])
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
            "thresholds": registry["signals"][signal_name],
            "detected_at": utc_now(),
        }

        storage.execute(
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
                json.dumps(registry["alert_destinations"][severity]),
                json.dumps(actions),
                json.dumps(diagnostics, sort_keys=True),
            ),
        )

        trigger_shadow_evaluation(
            storage,
            model_name=model_name,
            model_version=model_version,
            reason=f"{severity} drift detected for {signal_name}",
        )
        if severity == "critical":
            freeze_policy_auto_updates(storage, model_name, model_version)
            open_retraining_ticket(storage, model_name, model_version, diagnostics)

        events.append(
            {
                "signal_name": signal_name,
                "signal_value": value,
                "severity": severity,
                "auto_actions": actions,
            }
        )

    if highest_severity is None:
        storage.execute(
            """
            UPDATE model_versions
            SET drift_state = 'normal', updated_at = ?
            WHERE model_name = ? AND version = ?
            """,
            (utc_now(), model_name, model_version),
        )
    elif highest_severity == "warning":
        storage.execute(
            """
            UPDATE model_versions
            SET drift_state = 'warning', updated_at = ?
            WHERE model_name = ? AND version = ?
            """,
            (utc_now(), model_name, model_version),
        )

    storage.commit()
    return events


def run_scheduled_jobs(
    db_target: str,
    model_name: str,
    model_version: str,
    signal_values: dict[str, float],
    interval_seconds: int,
    iterations: int,
) -> None:
    """Run drift job on a fixed schedule."""
    for idx in range(iterations):
        with connect_storage(db_target) as storage:
            run_drift_job(storage, model_name, model_version, signal_values)
        if idx < iterations - 1:
            time.sleep(interval_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scheduled drift monitoring job.")
    parser.add_argument("--db-target", default="monitoring/drift_monitoring.db")
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
        db_target=args.db_target,
        model_name=args.model_name,
        model_version=args.model_version,
        signal_values=signal_values,
        interval_seconds=args.interval_seconds,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
