"""ColdSense (SmartClim) core package.

Epistemic AI-assisted decision-support system for cold rooms and refrigeration.
Follows the strict 4-stage loop: Predict -> Explain -> Investigate -> Verify.
"""

from coldsense.models import (
    CaseRecord,
    CopilotAnswer,
    CopilotRefusal,
    DataQualityReport,
    EvidenceBoard,
    EvidenceClass,
    EvidenceItem,
    EventType,
    FindingRecord,
    Hypothesis,
    NextBestCheck,
    PredictionRefusal,
    PredictionSuccess,
    RefusalReasonCode,
    TemperatureEvent,
    TemperaturePoint,
    TrustedConfirmationToken,
    VerificationStatus,
    WorkflowStep,
)

__version__ = "1.0.0"

__all__ = [
    "CaseRecord",
    "CopilotAnswer",
    "CopilotRefusal",
    "DataQualityReport",
    "EvidenceBoard",
    "EvidenceClass",
    "EvidenceItem",
    "EventType",
    "FindingRecord",
    "Hypothesis",
    "NextBestCheck",
    "PredictionRefusal",
    "PredictionSuccess",
    "RefusalReasonCode",
    "TemperatureEvent",
    "TemperaturePoint",
    "TrustedConfirmationToken",
    "VerificationStatus",
    "WorkflowStep",
]
