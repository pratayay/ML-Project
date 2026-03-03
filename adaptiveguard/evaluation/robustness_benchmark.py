"""Robustness benchmark for obfuscation/paraphrase/typo variants."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from adaptiveguard.api.policy_engine import apply_category_weight
from adaptiveguard.api.preprocessing import preprocess_text


STRICTNESS_LEVELS = (0.2, 0.5, 0.8)


def _load_benchmark(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def _compute_adjusted_risk(row: dict[str, object]) -> float:
    text = str(row["text"])
    risk = float(row["risk_score"])

    preprocessed = preprocess_text(text)
    category_weighted = apply_category_weight(risk, category="abuse", weights={"abuse": 1.0})
    return min(1.0, category_weighted + preprocessed.obfuscation_score * 0.15)


def risk_delta_distribution(path: Path) -> dict[float, dict[str, float | int]]:
    """Compute risk delta statistics vs base example, grouped by strictness."""
    rows = _load_benchmark(path)
    grouped: dict[str, dict[str, float]] = defaultdict(dict)

    for row in rows:
        key = str(row["id"])
        variant_type = str(row["variant_type"])
        grouped[key][variant_type] = _compute_adjusted_risk(row)

    deltas = [
        variant_risk - values["base"]
        for values in grouped.values()
        for variant_type, variant_risk in values.items()
        if variant_type != "base" and "base" in values
    ]

    if not deltas:
        empty = {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "p90": 0.0}
        return {strictness: dict(empty) for strictness in STRICTNESS_LEVELS}

    sorted_deltas = sorted(deltas)
    p90_idx = int((len(sorted_deltas) - 1) * 0.9)

    distribution = {
        "count": len(deltas),
        "mean": round(sum(deltas) / len(deltas), 4),
        "min": round(sorted_deltas[0], 4),
        "max": round(sorted_deltas[-1], 4),
        "p90": round(sorted_deltas[p90_idx], 4),
    }
    return {strictness: dict(distribution) for strictness in STRICTNESS_LEVELS}


def main() -> None:
    dataset_path = Path(__file__).parent / "data" / "robustness_benchmark.jsonl"
    report = risk_delta_distribution(dataset_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
