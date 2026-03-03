"""Dataset normalization utilities for cross-source risk training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MAPPING_VERSION = "2026.03.v1"


@dataclass(frozen=True)
class DatasetSpec:
    """Defines how a source dataset maps into the shared ontology."""

    dataset_name: str
    label_field: str
    text_field: str
    label_mapping: dict[str, str]


DATASET_SPECS: dict[str, DatasetSpec] = {
    "civil_comments": DatasetSpec(
        dataset_name="civil_comments",
        label_field="toxicity_label",
        text_field="comment_text",
        label_mapping={
            "non_toxic": "safe",
            "toxic": "abuse_harassment",
            "threat": "violence_threat",
            "identity_attack": "hate_identity",
        },
    ),
    "moderation_v2": DatasetSpec(
        dataset_name="moderation_v2",
        label_field="moderation_tag",
        text_field="text",
        label_mapping={
            "ok": "safe",
            "harassment": "abuse_harassment",
            "self-harm": "self_harm",
            "violence": "violence_threat",
            "hate": "hate_identity",
        },
    ),
    "incident_reports": DatasetSpec(
        dataset_name="incident_reports",
        label_field="incident_type",
        text_field="narrative",
        label_mapping={
            "benign": "safe",
            "abusive_language": "abuse_harassment",
            "suicide_instruction": "self_harm",
            "violent_planning": "violence_threat",
            "protected_group_slur": "hate_identity",
        },
    ),
}

SEVERITY_PRIORS: dict[str, float] = {
    "safe": 0.02,
    "abuse_harassment": 0.58,
    "self_harm": 0.86,
    "violence_threat": 0.91,
    "hate_identity": 0.82,
}


class UnknownLabelError(ValueError):
    """Raised when a source label is not registered for mapping."""


def _clamp_probability(value: Any, field_name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - narrow utility
        raise ValueError(f"{field_name} must be numeric, got {value!r}") from exc
    if numeric < 0 or numeric > 1:
        raise ValueError(f"{field_name} must be in [0, 1], got {numeric}")
    return numeric


def map_label_to_category(dataset_name: str, original_label: str) -> str:
    """Translate source labels into the shared ontology category."""

    if dataset_name not in DATASET_SPECS:
        raise KeyError(f"Unknown dataset: {dataset_name}")
    spec = DATASET_SPECS[dataset_name]
    category = spec.label_mapping.get(original_label)
    if category is None:
        raise UnknownLabelError(
            f"Label {original_label!r} is not mapped for dataset {dataset_name!r}."
        )
    return category


def compute_risk_target(
    category: str,
    *,
    annotator_agreement: Any | None = None,
    empirical_harm_rate: Any | None = None,
) -> float:
    """Compute risk target from observed quality signals and priors.

    The function avoids fixed constants by using whichever evidence is available:
    annotator agreement and/or empirical harm rates, then falls back to severity priors.
    """

    if category not in SEVERITY_PRIORS:
        raise KeyError(f"Unknown category: {category}")

    evidence: list[float] = []
    if annotator_agreement is not None:
        evidence.append(_clamp_probability(annotator_agreement, "annotator_agreement"))
    if empirical_harm_rate is not None:
        evidence.append(_clamp_probability(empirical_harm_rate, "empirical_harm_rate"))
    if not evidence:
        evidence.append(SEVERITY_PRIORS[category])
    else:
        evidence.append(SEVERITY_PRIORS[category])
    return sum(evidence) / len(evidence)


def unify_record(dataset_name: str, row: dict[str, Any]) -> dict[str, Any]:
    """Convert one source row to the training-ready unified schema."""

    if dataset_name not in DATASET_SPECS:
        raise KeyError(f"Unknown dataset: {dataset_name}")

    spec = DATASET_SPECS[dataset_name]
    original_label = row.get(spec.label_field)
    if original_label is None:
        raise KeyError(
            f"Missing label field {spec.label_field!r} for dataset {dataset_name!r}"
        )

    category = map_label_to_category(dataset_name, str(original_label))
    risk_target = compute_risk_target(
        category,
        annotator_agreement=row.get("annotator_agreement"),
        empirical_harm_rate=row.get("empirical_harm_rate"),
    )

    return {
        "text": row.get(spec.text_field, ""),
        "category": category,
        "risk_target": round(risk_target, 6),
        "dataset_name": dataset_name,
        "original_label": str(original_label),
        "mapping_version": MAPPING_VERSION,
    }


def unify_dataset_rows(dataset_name: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply label and target normalization to a full dataset."""

    return [unify_record(dataset_name, row) for row in rows]
