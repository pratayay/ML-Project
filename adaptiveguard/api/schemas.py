"""Pydantic schemas for policy moderation requests and responses."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Category = Literal["violence", "hate", "self_harm", "sexual", "spam", "other"]
Decision = Literal["allow", "review", "block"]


class PolicyConfig(BaseModel):
    """Policy-level configuration supplied by callers or service config."""

    strictness: float = Field(default=0.5, ge=0.0, le=1.0)
    policy_weights: dict[str, float] = Field(default_factory=dict)
    policy_version: str = Field(default="v1")


class ModerateRequest(BaseModel):
    """Input payload for policy moderation."""

    category: Category
    strictness: float = Field(default=0.5, ge=0.0, le=1.0)
    features: dict[str, Any] = Field(default_factory=dict)
    policy_weights: dict[str, float] = Field(default_factory=dict)
    policy_version: str = Field(default="v1")


class ModerationResult(BaseModel):
    """Canonical output payload from moderation endpoints."""

    risk_score: float = Field(ge=0.0, le=1.0)
    decision: Decision
    strictness: float = Field(ge=0.0, le=1.0)
    category: Category
    confidence: float = Field(ge=0.0, le=1.0)
    policy_version: str
    model_version: str


class BatchModerateRequest(BaseModel):
    """Batch moderation request payload."""

    items: list[ModerateRequest] = Field(min_length=1)


class BatchModerateResponse(BaseModel):
    """Batch moderation response payload."""

    results: list[ModerationResult]
