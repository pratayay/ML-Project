"""Validate inter-dataset risk target distributions before training."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to unified JSONL or CSV file")
    parser.add_argument(
        "--max-mean-gap",
        type=float,
        default=0.20,
        help="Maximum allowed risk_target mean gap vs global mean",
    )
    parser.add_argument(
        "--max-tvd",
        type=float,
        default=0.35,
        help="Maximum allowed category total-variation distance vs global",
    )
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    raise ValueError("Input must be .jsonl or .csv")


def _as_float(record: dict[str, Any], key: str) -> float:
    value = float(record[key])
    if value < 0 or value > 1:
        raise ValueError(f"{key} outside [0,1]: {value}")
    return value


def _compute_tvd(local: dict[str, float], global_dist: dict[str, float]) -> float:
    categories = set(local) | set(global_dist)
    return 0.5 * sum(abs(local.get(cat, 0.0) - global_dist.get(cat, 0.0)) for cat in categories)


def validate(rows: list[dict[str, Any]], *, max_mean_gap: float, max_tvd: float) -> tuple[list[str], str]:
    if not rows:
        return ["No rows found in input."], ""

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    global_counts: Counter[str] = Counter()
    all_targets: list[float] = []

    for row in rows:
        dataset_name = row.get("dataset_name")
        category = row.get("category")
        if not dataset_name or not category:
            raise KeyError("Rows must include dataset_name and category")
        target = _as_float(row, "risk_target")

        by_dataset[str(dataset_name)].append(row)
        global_counts[str(category)] += 1
        all_targets.append(target)

    global_mean = sum(all_targets) / len(all_targets)
    total_rows = sum(global_counts.values())
    global_dist = {cat: count / total_rows for cat, count in global_counts.items()}

    issues: list[str] = []
    report_lines = [
        f"Global mean risk_target: {global_mean:.4f}",
        f"Global category distribution: {global_dist}",
        "",
        "Per-dataset diagnostics:",
    ]

    for dataset_name in sorted(by_dataset):
        rows_for_dataset = by_dataset[dataset_name]
        targets = [_as_float(row, "risk_target") for row in rows_for_dataset]
        mean = sum(targets) / len(targets)
        mean_gap = abs(mean - global_mean)

        local_counts = Counter(str(row["category"]) for row in rows_for_dataset)
        local_dist = {cat: count / len(rows_for_dataset) for cat, count in local_counts.items()}
        tvd = _compute_tvd(local_dist, global_dist)

        report_lines.append(
            f"- {dataset_name}: n={len(rows_for_dataset)}, mean={mean:.4f}, "
            f"mean_gap={mean_gap:.4f}, tvd={tvd:.4f}"
        )

        if mean_gap > max_mean_gap:
            issues.append(
                f"{dataset_name} mean_gap {mean_gap:.4f} exceeded threshold {max_mean_gap:.4f}"
            )
        if tvd > max_tvd:
            issues.append(f"{dataset_name} tvd {tvd:.4f} exceeded threshold {max_tvd:.4f}")

    return issues, "\n".join(report_lines)


def main() -> None:
    args = parse_args()
    rows = _load_rows(Path(args.input))
    issues, report = validate(rows, max_mean_gap=args.max_mean_gap, max_tvd=args.max_tvd)
    print(report)
    if issues:
        print("\nExtreme mismatch detected:")
        for issue in issues:
            print(f" - {issue}")
        raise SystemExit(1)

    print("\nNo extreme inter-dataset mismatch detected.")


if __name__ == "__main__":
    main()
