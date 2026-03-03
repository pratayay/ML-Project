# Drift Incident Runbook

## Purpose
This runbook describes how to detect, triage, and recover from model drift incidents for the production risk model.

## Drift signal threshold registry
The operational threshold registry lives in `monitoring/drift_threshold_registry.yaml` and defines warning and critical thresholds for these signals:

| Signal | Warning | Critical | Meaning |
|---|---:|---:|---|
| `risk_mean_shift` | 0.08 | 0.15 | Absolute shift in mean model risk score |
| `category_shift` | 0.12 | 0.20 | Total variation distance in category distribution |
| `psi` | 0.20 | 0.35 | Population Stability Index |
| `kl_divergence` | 0.10 | 0.20 | KL divergence score |

## Severity levels and alert destinations
- **Warning**
  - Destinations: `slack:#ml-ops-alerts`, `pagerduty:ml-risk-warning`
  - Expected response window: same business day
- **Critical**
  - Destinations: `slack:#incident-ml-platform`, `pagerduty:ml-risk-critical`, `email:ml-oncall@example.com`
  - Expected response window: immediate (P1)

## Scheduled job execution
The drift scheduler is implemented in `monitoring/drift_jobs.py`.

Example hourly execution:

```bash
python monitoring/drift_jobs.py \
  --db-path monitoring/drift_monitoring.db \
  --model-name risk_model \
  --model-version v1 \
  --risk-mean-shift 0.09 \
  --category-shift 0.10 \
  --psi 0.22 \
  --kl-divergence 0.11
```

The job writes:
- `model_versions`: current drift state and auto-update freeze status.
- `drift_events`: one row per triggered signal with diagnostics and alert route metadata.

## Auto-actions
When drift thresholds are met, the system executes these actions:

1. **Trigger shadow evaluation on recent data**
   - Always for warning or critical events.
   - Stored in `shadow_evaluations` as `status=queued`, `dataset_window=recent_7d`.
2. **Freeze policy auto-updates on critical drift**
   - Sets `model_versions.freeze_auto_updates=1`.
   - Sets `model_versions.drift_state=critical`.
3. **Open retraining ticket with diagnostics**
   - For critical events only.
   - Inserts into `retraining_tickets` with serialized diagnostics payload.

## Incident response playbook

### 1) Acknowledge incident
1. Acknowledge PagerDuty or Slack alert.
2. Confirm model and version from latest `drift_events` row.
3. Check if severity is warning or critical.

### 2) Validate signal quality
1. Verify input data freshness for baseline and recent windows.
2. Confirm no feature pipeline outage or schema drift.
3. Recompute the signal manually for a spot-check.

### 3) Assess operational impact
1. Review decision-rate changes (`allow/review/block`).
2. Review false-positive and false-negative proxy metrics.
3. Estimate impacted traffic segments.

### 4) Execute auto-actions and verify
1. Confirm a `shadow_evaluations` row was created.
2. For critical incidents, verify `freeze_auto_updates=1` in `model_versions`.
3. For critical incidents, verify a row exists in `retraining_tickets`.

### 5) Decide mitigation path
- **If warning and impact is low:** monitor with increased frequency and prepare rollback plan.
- **If warning and impact is moderate/high:** promote to critical handling and begin retraining.
- **If critical:** keep policy updates frozen, execute retraining ticket immediately, evaluate rollback candidate.

### 6) Recovery criteria
A drift incident can be closed when all are true:
1. Drift signals are below warning thresholds for 3 consecutive scheduled runs.
2. Shadow evaluation performance is within accepted bands.
3. Any retrained model passes offline/online guardrails.
4. On-call and model owner sign off in ticket.

### 7) Post-incident follow-up
1. Add a timeline and root-cause summary to incident document.
2. Update threshold registry if needed (with rationale).
3. Add new tests/checks that would have reduced time-to-detection or time-to-mitigation.
