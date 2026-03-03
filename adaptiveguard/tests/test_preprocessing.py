from adaptiveguard.training.preprocessing import (
    MAPPING_VERSION,
    UnknownLabelError,
    compute_risk_target,
    map_label_to_category,
    unify_record,
)


def test_map_label_to_category() -> None:
    assert map_label_to_category("moderation_v2", "self-harm") == "self_harm"


def test_unknown_label_raises() -> None:
    try:
        map_label_to_category("civil_comments", "not_in_mapping")
    except UnknownLabelError:
        return
    raise AssertionError("Expected UnknownLabelError")


def test_compute_risk_target_uses_observed_signals() -> None:
    value = compute_risk_target(
        "abuse_harassment", annotator_agreement=0.8, empirical_harm_rate=0.7
    )
    assert value == (0.8 + 0.7 + 0.58) / 3


def test_unify_record_captures_provenance() -> None:
    row = {
        "moderation_tag": "hate",
        "text": "example",
        "annotator_agreement": 0.9,
    }
    unified = unify_record("moderation_v2", row)
    assert unified["dataset_name"] == "moderation_v2"
    assert unified["original_label"] == "hate"
    assert unified["mapping_version"] == MAPPING_VERSION
    assert unified["category"] == "hate_identity"
