"""Pydantic schemas for policy moderation requests and responses."""

from pydantic import BaseModel, Field


class PolicyConfig(BaseModel):
    """Policy-level configuration supplied by callers or service config."""

    strictness: float = Field(default=0.5, ge=0.0, le=1.0)
    policy_weights: dict[str, float] = Field(default_factory=dict)
    policy_version: str = Field(default="v1")


class ModerateRequest(BaseModel):
    """Input payload for policy moderation."""

    risk_score: float = Field(ge=0.0, le=1.0)
    category: str
    strictness: float = Field(default=0.5, ge=0.0, le=1.0)
    policy_weights: dict[str, float] = Field(default_factory=dict)


class ModerateResponse(BaseModel):
    """Output payload from policy moderation."""

    final_score: float = Field(ge=0.0, le=1.0)
    decision: str
    confidence: float = Field(ge=0.0, le=1.0)
    policy_version: str
