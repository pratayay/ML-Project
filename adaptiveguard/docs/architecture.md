# AdaptiveGuard Architecture

## End-to-End Pipeline

AdaptiveGuard is designed as an iterative ML risk platform with clear boundaries between modeling, training, serving, and monitoring.

1. **Data ingestion and feature generation**
   - Transaction/event data enters through upstream systems.
   - Features are normalized and persisted for online and offline use.

2. **Model training and validation**
   - `adaptiveguard/training` orchestrates dataset assembly, training jobs, and artifact packaging.
   - `adaptiveguard/evaluation` computes offline metrics (AUC, precision/recall, calibration, drift).

3. **Model serving and decisioning**
   - `adaptiveguard/api` exposes health and scoring endpoints.
   - The service loads approved model artifacts from a registry/storage layer.

4. **Observability and feedback loop**
   - `adaptiveguard/dashboard` visualizes risk decisions and model performance.
   - Outcomes and analyst labels are routed back into retraining workflows.

## Core Components

- **Model layer (`adaptiveguard/model`)**: Risk model classes, feature contracts, and inference logic.
- **Training layer (`adaptiveguard/training`)**: CLI + job orchestration for periodic/triggered retraining.
- **Evaluation layer (`adaptiveguard/evaluation`)**: Benchmarking, drift detection, and experiment reports.
- **API layer (`adaptiveguard/api`)**: FastAPI service for liveness/readiness and scoring endpoints.
- **Infra layer (`adaptiveguard/infra`, `docker/`)**: Local environment definitions and deployment primitives.

## Phased 12-Week Plan

### Weeks 1–2: Foundation
- Finalize repository scaffold, coding standards, and test conventions.
- Stand up API service skeleton, CI checks, and local Docker stack.

### Weeks 3–4: Data + Baseline Model
- Define feature schema and build baseline data pipeline.
- Implement first `RiskModel` baseline and static evaluation notebook/report.

### Weeks 5–6: Training Orchestration
- Expand training CLI into configurable pipelines.
- Add artifact versioning and model metadata tracking.

### Weeks 7–8: Evaluation + Governance
- Add robust offline evaluation metrics and threshold analysis.
- Introduce model acceptance criteria and promotion workflow.

### Weeks 9–10: Serving + Monitoring
- Implement scoring endpoint(s) and model loading strategy.
- Add request/decision logging and performance monitoring hooks.

### Weeks 11–12: Dashboard + Hardening
- Deliver initial dashboard views for KPIs and drift.
- Run load/security tests and finalize production-readiness checklist.
