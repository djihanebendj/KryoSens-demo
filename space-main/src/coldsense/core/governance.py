"""Records that keep models and hypotheses honest.

These are data structures only. Nothing here trains, promotes or deploys
anything; that workflow is future work (see docs/ARCHITECTURE.md).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field

from .evidence import EvidenceClass


class HypothesisStatus(str, Enum):
    PROPOSED = "PROPOSED"
    UNDER_TEST = "UNDER_TEST"
    SUPPORTED = "SUPPORTED"
    REJECTED = "REJECTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class Hypothesis(BaseModel):
    """A claim the system or team may test. Never a fact by itself."""

    id: str
    statement: str
    motivating_observation: str
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    data_needed: list[str] = Field(default_factory=list)
    test: str
    success_criteria: str
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    generated_by: str = Field(description="'team', a model id, or 'llm:<model>'")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ModelCard(BaseModel):
    """Required for every model that produces a number shown to a user.

    A model without a baseline, a metric and a known failure mode does not
    get a card, and without a card it does not ship.
    """

    name: str
    version: str
    purpose: str
    baseline: str = Field(description="name of the model it must beat, or 'none: this IS the baseline'")
    metric: str
    failure_modes: list[str] = Field(min_length=1)
    product_benefit: str
    validation_evidence: EvidenceClass = Field(
        description="best evidence that it works: REAL only after real outcome tests")
    validation_summary: str = "not yet validated"
