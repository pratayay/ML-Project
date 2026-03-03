from adaptiveguard.training.validate_targets import validate


def test_validate_passes_balanced_rows() -> None:
    rows = [
        {"dataset_name": "a", "category": "safe", "risk_target": 0.1},
        {"dataset_name": "a", "category": "abuse_harassment", "risk_target": 0.6},
        {"dataset_name": "b", "category": "safe", "risk_target": 0.1},
        {"dataset_name": "b", "category": "abuse_harassment", "risk_target": 0.6},
    ]
    issues, _ = validate(rows, max_mean_gap=0.2, max_tvd=0.35)
    assert issues == []


def test_validate_flags_extreme_mismatch() -> None:
    rows = [
        {"dataset_name": "a", "category": "safe", "risk_target": 0.05},
        {"dataset_name": "a", "category": "safe", "risk_target": 0.08},
        {"dataset_name": "b", "category": "violence_threat", "risk_target": 0.98},
        {"dataset_name": "b", "category": "violence_threat", "risk_target": 0.99},
    ]
    issues, _ = validate(rows, max_mean_gap=0.2, max_tvd=0.35)
    assert issues
