"""PyTorch risk model implementation backed by a HuggingFace encoder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from transformers import DistilBertConfig, DistilBertModel


@dataclass(frozen=True)
class RiskModelConfig:
    """Configuration and metadata for the risk model."""

    model_name: str = "risk_model"
    model_version: str = "1.0.0"
    metadata_version: str = "2026-01-01"
    backbone_name: str = "distilbert"
    distilbert_hidden_size: int = 768
    distilbert_vocab_size: int = 30522
    distilbert_num_layers: int = 6
    distilbert_num_heads: int = 12
    distilbert_ffn_dim: int = 3072
    dropout: float = 0.1
    num_categories: int = 0
    category_loss_weight: float = 1.0

    def to_metadata(self) -> dict[str, Any]:
        """Return model metadata consumed by APIs and drift jobs."""
        config = asdict(self)
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "metadata_version": self.metadata_version,
            "backbone_name": self.backbone_name,
            "task": "multi_task" if self.num_categories > 0 else "risk_regression",
            "config": config,
        }


class RiskModel(nn.Module):
    """Risk scoring model with optional multi-task category classification head."""

    def __init__(self, config: RiskModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or RiskModelConfig()

        encoder_config = DistilBertConfig(
            vocab_size=self.config.distilbert_vocab_size,
            n_layers=self.config.distilbert_num_layers,
            n_heads=self.config.distilbert_num_heads,
            dim=self.config.distilbert_hidden_size,
            hidden_dim=self.config.distilbert_ffn_dim,
            dropout=self.config.dropout,
            attention_dropout=self.config.dropout,
        )
        self.encoder = DistilBertModel(encoder_config)
        self.dropout = nn.Dropout(self.config.dropout)
        self.risk_head = nn.Linear(self.config.distilbert_hidden_size, 1)
        self.category_head = (
            nn.Linear(self.config.distilbert_hidden_size, self.config.num_categories)
            if self.config.num_categories > 0
            else None
        )

    @property
    def model_version(self) -> str:
        """Convenience alias for API compatibility."""
        return self.config.model_version

    def get_metadata(self) -> dict[str, Any]:
        """Get model metadata for API responses and drift monitoring payloads."""
        return self.config.to_metadata()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        category_labels: Tensor | None = None,
    ) -> dict[str, Tensor | None]:
        """Run inference/training step.

        Returns a dictionary with:
        - ``risk_score``: sigmoid-normalized risk score of shape ``(batch_size,)``.
        - ``risk_logits``: pre-sigmoid risk logits of shape ``(batch_size,)``.
        - ``category_logits``: optional class logits of shape ``(batch_size, num_categories)``.
        - ``loss``: optional aggregated training loss when labels are provided.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.last_hidden_state[:, 0])

        risk_logits = self.risk_head(pooled).squeeze(-1)
        risk_score = torch.sigmoid(risk_logits)

        category_logits = self.category_head(pooled) if self.category_head is not None else None

        loss: Tensor | None = None
        if labels is not None:
            target = labels.float().view_as(risk_logits)
            loss = nn.functional.mse_loss(risk_score, target)

        if category_labels is not None and category_logits is not None:
            category_loss = nn.functional.cross_entropy(category_logits, category_labels.long())
            loss = (
                category_loss * self.config.category_loss_weight
                if loss is None
                else loss + (category_loss * self.config.category_loss_weight)
            )

        return {
            "risk_score": risk_score,
            "risk_logits": risk_logits,
            "category_logits": category_logits,
            "loss": loss,
        }

    @torch.inference_mode()
    def predict(self, features: dict[str, Tensor]) -> float:
        """Predict a single scalar risk score from pre-tokenized tensor features."""
        self.eval()
        outputs = self.forward(
            input_ids=features["input_ids"],
            attention_mask=features.get("attention_mask"),
        )
        risk_scores = outputs["risk_score"]
        assert isinstance(risk_scores, Tensor)
        return float(risk_scores.mean().item())
