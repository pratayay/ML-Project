-- SQL reference for scheduler-triggered drift persistence.
-- Use this transaction from cron/Airflow operators after drift signals are computed.

BEGIN TRANSACTION;

INSERT INTO drift_events (
    event_ts,
    model_name,
    model_version,
    signal_name,
    signal_value,
    severity,
    alert_destinations,
    auto_actions,
    diagnostics_json
)
VALUES
    (:event_ts, :model_name, :model_version, :signal_name, :signal_value,
     :severity, :alert_destinations, :auto_actions, :diagnostics_json);

UPDATE model_versions
SET drift_state = :drift_state,
    freeze_auto_updates = :freeze_auto_updates,
    updated_at = :event_ts
WHERE model_name = :model_name
  AND version = :model_version;

COMMIT;
