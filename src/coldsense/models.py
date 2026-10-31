"""Domain models, enums, and epistemic contracts for ColdSense (SmartClim).

Enforces the 'Rules That Never Bend':
1. Every result carries an explicit EvidenceClass (REAL, SIMULATED, EXPECTED, UNVALIDATED).
2. Uploaded data always runs as UNVALIDATED; user claims of real data are stored separately.
3. PredictionSuccess and PredictionRefusal are strictly separate response types.
4. Hypotheses come only from deterministic template rendering (Rule 7).
5. Findings start as UNKNOWN and can only be confirmed via a human TrustedConfirmationToken.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class EvidenceClass(str, Enum):
    """Mandatory provenance label attached to every ColdSense datum and result."""

    REAL = "REAL"
    SIMULATED = "SIMULATED"
    EXPECTED = "EXPECTED"
    UNVALIDATED = "UNVALIDATED"


class WorkflowStep(int, Enum):
    """Sequential 10-step technician workflow."""

    UPLOAD_DATA = 1
    DATA_QUALITY = 2
    EVENT_ANALYSIS = 3
    PREDICTION = 4
    EVIDENCE = 5
    NEXT_BEST_CHECK = 6
    COPILOT = 7
    TECHNICIAN_FINDING = 8
    VERIFICATION = 9
    REPORT = 10


class EventType(str, Enum):
    """Detected thermal regime or anomaly event types."""

    POWER_CUT = "POWER_CUT"
    DOOR_OPEN = "DOOR_OPEN"
    DEFROST_CYCLE = "DEFROST_CYCLE"
    COOLING_RECOVERY = "COOLING_RECOVERY"
    SETPOINT_EXCURSION = "SETPOINT_EXCURSION"


class RefusalReasonCode(str, Enum):
    """Structured reason codes when the Reliability Gate or Copilot refuses."""

    INSUFFICIENT_SAMPLES = "INSUFFICIENT_SAMPLES"
    EXCESSIVE_DATA_GAPS = "EXCESSIVE_DATA_GAPS"
    NON_PHYSICAL_TRAJECTORY = "NON_PHYSICAL_TRAJECTORY"
    POOR_MODEL_FIT = "POOR_MODEL_FIT"
    NO_DRIFT_EVENT_FOUND = "NO_DRIFT_EVENT_FOUND"
    HAZARD_GATE_BLOCKED = "HAZARD_GATE_BLOCKED"
    CONFIG_ERROR = "CONFIG_ERROR"
    UNSUPPORTED_SCRIPT_OR_LANGUAGE = "UNSUPPORTED_SCRIPT_OR_LANGUAGE"
    INSUFFICIENT_RETRIEVAL_EVIDENCE = "INSUFFICIENT_RETRIEVAL_EVIDENCE"
    NUMERIC_INTEGRITY_FAILURE = "NUMERIC_INTEGRITY_FAILURE"
    FEATURE_DISABLED = "FEATURE_DISABLED"


class VerificationStatus(str, Enum):
    """Verification outcome of a diagnostic finding. Always UNKNOWN until technician confirms."""

    UNKNOWN = "UNKNOWN"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"


class EvidenceCategory(str, Enum):
    """Three-column separation on the Evidence page."""

    FACT = "FACT"
    MODEL_INFERENCE = "MODEL_INFERENCE"
    RECOMMENDATION = "RECOMMENDATION"


@dataclass(frozen=True)
class TemperaturePoint:
    """Single timestamped reading from a cold-room temperature log."""

    timestamp_sec: float
    iso_timestamp: str
    temp_c: float
    ambient_temp_c: Optional[float] = None
    door_open: Optional[bool] = None
    power_on: Optional[bool] = None
    defrost_active: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DataQualityIssue:
    """Specific quality anomaly found in the uploaded log."""

    code: str
    severity: str  # "INFO", "WARNING", "CRITICAL"
    message: str
    start_index: Optional[int] = None
    end_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DataQualityReport:
    """Step 2 output: completeness, cadence, and anomaly check of the log."""

    case_id: str
    evidence_class: EvidenceClass
    total_rows: int
    valid_rows: int
    duplicate_timestamps: int
    missing_value_rows: int
    gap_count: int
    max_gap_sec: float
    median_cadence_sec: float
    cadence_jitter_sec: float
    out_of_range_count: int
    flatline_segments: int
    completeness_ratio: float
    is_usable_for_analysis: bool
    issues: List[DataQualityIssue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence_class"] = self.evidence_class.value
        return d


@dataclass(frozen=True)
class TemperatureEvent:
    """Step 3 output: detected thermal regime or anomaly window."""

    event_id: str
    event_type: EventType
    evidence_class: EvidenceClass
    start_index: int
    end_index: int
    start_timestamp_sec: float
    end_timestamp_sec: float
    start_iso: str
    end_iso: str
    duration_sec: float
    start_temp_c: float
    end_temp_c: float
    peak_temp_c: float
    temp_delta_c: float
    drift_rate_c_per_hour: float
    description: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        d["evidence_class"] = self.evidence_class.value
        return d


@dataclass(frozen=True)
class ForecastPoint:
    """Single point on the thermal model forecast curve vs naive baseline."""

    timestamp_sec: float
    elapsed_sec: float
    observed_temp_c: Optional[float]
    thermal_model_temp_c: float
    lower_95_c: float
    upper_95_c: float
    lower_50_c: float
    upper_50_c: float
    naive_baseline_temp_c: float
    evidence_class: EvidenceClass = EvidenceClass.EXPECTED

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence_class"] = self.evidence_class.value
        return d


@dataclass(frozen=True)
class PredictionSuccess:
    """Step 4 valid forecast output when the Reliability Gate passes.

    Strictly separate from PredictionRefusal so a refusal can never be rendered as a curve.
    """

    response_type: str  # Always "SUCCESS"
    case_id: str
    event_id: str
    evidence_class: EvidenceClass
    baseline_evidence_class: EvidenceClass
    initial_temp_c: float
    ambient_temp_c: float
    time_constant_tau_sec: float
    time_constant_tau_hours: float
    r_squared: float
    thermal_model_rmse_c: float
    naive_baseline_rmse_c: float
    thermal_model_mae_c: float
    naive_baseline_mae_c: float
    critical_threshold_c: float
    estimated_time_to_threshold_sec: Optional[float]
    uncertainty_95_halfwidth_c: float
    reliability_gate_passed: bool
    reliability_notes: List[str]
    forecast_curve: List[ForecastPoint]

    def __post_init__(self) -> None:
        if self.response_type != "SUCCESS":
            raise ValueError("PredictionSuccess.response_type must be 'SUCCESS'")
        if not self.reliability_gate_passed:
            raise ValueError("PredictionSuccess requires reliability_gate_passed=True")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence_class"] = self.evidence_class.value
        d["baseline_evidence_class"] = self.baseline_evidence_class.value
        d["forecast_curve"] = [pt.to_dict() for pt in self.forecast_curve]
        return d


@dataclass(frozen=True)
class PredictionRefusal:
    """Step 4 refusal output when the Reliability Gate determines the model cannot be trusted.

    Contains NO forecast curve or guessed numbers.
    """

    response_type: str  # Always "REFUSAL"
    case_id: str
    event_id: Optional[str]
    evidence_class: EvidenceClass
    reason_code: RefusalReasonCode
    refusal_explanation: str
    failed_checks: List[str]
    diagnostics_metrics: Dict[str, float]
    reliability_gate_passed: bool = False

    def __post_init__(self) -> None:
        if self.response_type != "REFUSAL":
            raise ValueError("PredictionRefusal.response_type must be 'REFUSAL'")
        if self.reliability_gate_passed:
            raise ValueError("PredictionRefusal cannot have reliability_gate_passed=True")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence_class"] = self.evidence_class.value
        d["reason_code"] = self.reason_code.value
        return d


PredictionOutcome = Union[PredictionSuccess, PredictionRefusal]


@dataclass(frozen=True)
class EvidenceItem:
    """Single item on the Step 5 Evidence page with explicit provenance."""

    evidence_id: str
    category: EvidenceCategory
    evidence_class: EvidenceClass
    title: str
    statement: str
    provenance_source: str
    supports: List[str]
    does_not_support: List[str]
    numeric_values: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        d["evidence_class"] = self.evidence_class.value
        return d


@dataclass(frozen=True)
class EvidenceBoard:
    """Step 5 output: separated Facts, Model Inferences, and Recommendations."""

    case_id: str
    evidence_class: EvidenceClass
    facts: List[EvidenceItem]
    model_inferences: List[EvidenceItem]
    recommendations: List[EvidenceItem]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "evidence_class": self.evidence_class.value,
            "facts": [f.to_dict() for f in self.facts],
            "model_inferences": [m.to_dict() for m in self.model_inferences],
            "recommendations": [r.to_dict() for r in self.recommendations],
        }


@dataclass(frozen=True)
class Hypothesis:
    """Structured diagnostic hypothesis rendered strictly from fixed templates (Rule 7).

    Never confirmed automatically; always starts at VerificationStatus.UNKNOWN.
    """

    hypothesis_id: str
    template_id: str
    title: str
    rendered_statement: str
    evidence_class: EvidenceClass
    supporting_evidence_ids: List[str]
    contradicting_evidence_ids: List[str]
    prior_plausibility_rank: int
    verification_status: VerificationStatus = VerificationStatus.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence_class"] = self.evidence_class.value
        d["verification_status"] = self.verification_status.value
        return d


@dataclass(frozen=True)
class NextBestCheck:
    """Step 6 output: ranked physical inspection that settles open hypotheses."""

    check_id: str
    rank: int
    title: str
    inspection_procedure: str
    target_hypothesis_ids: List[str]
    confirm_indicator: str
    reject_indicator: str
    estimated_minutes: int
    evidence_class: EvidenceClass

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence_class"] = self.evidence_class.value
        return d


@dataclass(frozen=True)
class FindingDraftRecord:
    """Exact record reviewed by a technician in Step 8 before confirmation."""

    case_id: str
    hypothesis_id: str
    check_id: str
    proposed_status: VerificationStatus
    technician_id: str
    technician_observation: str
    evidence_class: EvidenceClass

    def canonical_json(self) -> str:
        payload = {
            "case_id": self.case_id,
            "check_id": self.check_id,
            "evidence_class": self.evidence_class.value,
            "hypothesis_id": self.hypothesis_id,
            "proposed_status": self.proposed_status.value,
            "technician_id": self.technician_id,
            "technician_observation": self.technician_observation.strip(),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def record_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["proposed_status"] = self.proposed_status.value
        d["evidence_class"] = self.evidence_class.value
        d["record_hash"] = self.record_hash()
        return d


@dataclass(frozen=True)
class TrustedConfirmationToken:
    """Single-use, expiring token bound to the SHA-256 hash of a FindingDraftRecord."""

    token_id: str
    case_id: str
    record_hash: str
    issued_at_epoch: float
    expires_at_epoch: float
    signature: str
    used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FindingRecord:
    """Step 9 immutable, append-only technician-confirmed finding."""

    finding_id: str
    case_id: str
    hypothesis_id: str
    check_id: str
    verification_status: VerificationStatus
    technician_id: str
    technician_observation: str
    confirmed_at_iso: str
    confirmed_at_epoch: float
    record_hash: str
    token_id: str
    source_evidence_class: EvidenceClass

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["verification_status"] = self.verification_status.value
        d["source_evidence_class"] = self.source_evidence_class.value
        return d


@dataclass(frozen=True)
class CopilotCitation:
    """Verbatim quote and source reference from the local knowledge base."""

    doc_id: str
    doc_title: str
    section: str
    verbatim_quote: str
    relevance_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CopilotAnswer:
    """Step 7 grounded answer after passing Hazard Gate, Quote Check, and Numeric Audit."""

    response_type: str  # "ANSWER"
    query: str
    answer_text: str
    verbatim_citations: List[CopilotCitation]
    allowed_numbers_used: List[float]
    evidence_class: EvidenceClass
    retrieval_mode: str  # "SEMANTIC_HYBRID" or "LEXICAL_FALLBACK"
    conditions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence_class"] = self.evidence_class.value
        d["verbatim_citations"] = [c.to_dict() for c in self.verbatim_citations]
        return d


@dataclass(frozen=True)
class CopilotRefusal:
    """Step 7 refusal or fixed Hazard Gate template response."""

    response_type: str  # "REFUSAL" or "HAZARD_TEMPLATE"
    query: str
    reason_code: RefusalReasonCode
    message: str
    hazard_matched_terms: List[str] = field(default_factory=list)
    evidence_class: EvidenceClass = EvidenceClass.EXPECTED

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["reason_code"] = self.reason_code.value
        d["evidence_class"] = self.evidence_class.value
        return d


@dataclass
class CaseRecord:
    """Private visitor session case tracking progress through the 10-step workflow."""

    case_id: str
    owner_session_id: str
    title: str
    evidence_class: EvidenceClass
    user_claimed_real: bool
    created_at_epoch: float
    expires_at_epoch: float
    critical_threshold_c: float = 8.0
    points: List[TemperaturePoint] = field(default_factory=list)
    data_quality: Optional[DataQualityReport] = None
    events: List[TemperatureEvent] = field(default_factory=list)
    prediction: Optional[PredictionOutcome] = None
    evidence_board: Optional[EvidenceBoard] = None
    hypotheses: List[Hypothesis] = field(default_factory=list)
    next_best_checks: List[NextBestCheck] = field(default_factory=list)
    draft_findings: Dict[str, FindingDraftRecord] = field(default_factory=dict)
    confirmed_findings: List[FindingRecord] = field(default_factory=list)
    unlocked_step: int = 1

    def __post_init__(self) -> None:
        # Rule 1: Uploaded data always runs as UNVALIDATED; user claims never upgrade it.
        if self.evidence_class not in (EvidenceClass.SIMULATED, EvidenceClass.UNVALIDATED):
            raise ValueError(
                f"CaseRecord must be SIMULATED or UNVALIDATED at ingestion, got {self.evidence_class}"
            )

    def is_step_unlocked(self, step: Union[WorkflowStep, int]) -> bool:
        step_num = int(step)
        return step_num <= self.unlocked_step

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "title": self.title,
            "evidence_class": self.evidence_class.value,
            "user_claimed_real": self.user_claimed_real,
            "created_at_epoch": self.created_at_epoch,
            "expires_at_epoch": self.expires_at_epoch,
            "critical_threshold_c": self.critical_threshold_c,
            "point_count": len(self.points),
            "event_count": len(self.events),
            "unlocked_step": self.unlocked_step,
            "confirmed_findings_count": len(self.confirmed_findings),
        }
