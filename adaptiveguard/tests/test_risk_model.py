"""Unit tests for the RiskModel module."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from adaptiveguard.model.risk_model import RiskModel, RiskModelConfig


def _tiny_config(num_categories: int = 0) -> RiskModelConfig:
    return RiskModelConfig(
        distilbert_hidden_size=32,
        distilbert_num_layers=2,
        distilbert_num_heads=4,
        distilbert_ffn_dim=64,
        distilbert_vocab_size=100,
        dropout=0.0,
        num_categories=num_categories,
    )


def test_forward_shapes_and_optional_category_logits() -> None:
    model = RiskModel(_tiny_config(num_categories=3))
    model.eval()

    input_ids = torch.randint(0, 100, (2, 8))
    attention_mask = torch.ones_like(input_ids)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    assert outputs["risk_score"].shape == (2,)
    assert outputs["risk_logits"].shape == (2,)
    assert outputs["category_logits"] is not None
    assert outputs["category_logits"].shape == (2, 3)
    assert outputs["loss"] is None


def test_risk_score_range_is_bounded() -> None:
    model = RiskModel(_tiny_config())
    model.eval()

    input_ids = torch.randint(0, 100, (4, 10))
    outputs = model(input_ids=input_ids)

    assert torch.all(outputs["risk_score"] >= 0.0)
    assert torch.all(outputs["risk_score"] <= 1.0)


def test_deterministic_output_with_fixed_seed() -> None:
    torch.manual_seed(1234)
    model_a = RiskModel(_tiny_config())
    model_a.eval()

    torch.manual_seed(1234)
    model_b = RiskModel(_tiny_config())
    model_b.eval()

    input_ids = torch.randint(0, 100, (2, 6))

    score_a = model_a(input_ids=input_ids)["risk_score"]
    score_b = model_b(input_ids=input_ids)["risk_score"]

    assert torch.allclose(score_a, score_b)
