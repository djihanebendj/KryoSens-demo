"""Response and request types of the HTTP service: the API contract (docs/PHASE9_ARCHITECTURE.md §3).

These are views over the accepted Phase 2-5 results, built in ``service.py``; they never compute anything. The
OpenAPI document generated from them is the source of the frontend's TypeScript types.

The prediction response is a discriminated union on ``outcome``. ``PredictionRefusal`` forbids extra fields and has
no field that could carry a forecast: no trajectory, crossing time, selected model, parameters, baseline
prediction, interval or reliability status. A refusal therefore cannot leak a number the engine withheld.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from coldsense.copilot.findings import Outcome
from coldsense.core.evidence import EvidenceClass
from coldsense.core.governance import Hypothesis
from coldsense.core.results import Issue, Verdict
from coldsense.domains.cold_room.diagnostics import DiagnosisStatus, Finding, NextCheck, ObservedFact
from coldsense.domains.cold_room.schemas import DataQualityReport, Event
from coldsense.domains.cold_room.thermal.limits import MAX_HORIZON_MIN
from coldsense.domains.cold_room.thermal.schemas import (
    CrossingStatus,
    Direction,
    ForecastQuestion,
    Necessity,
    Priority,
    Provenance,
    RefusalCode,
    ReliabilityStatus,
    ValidationScope,
)
from coldsense.knowledge.schemas import Answer

# Response fields with a default are still always present: mark them required in the (serialization) schema.
_STRICT = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)

DeclaredEvidence = Literal["REAL", "SIMULATED"]
EvidenceBasis = Literal["ANONYMOUS_UPLOAD", "BUNDLED_SAMPLE"]
DeclarationStatus = Literal["NOT_DECLARED", "DECLARED_BY_UPLOADER_NOT_VERIFIED"]
DECLARATION_NOTE = "declared by uploader, not verified"


class ResultProvenance(BaseModel):
    """Where every number in a response came from (amendment G: the uploader's declaration is shown separately and
    never changes the evidence class or the Phase 5 scope)."""
    model_config = _STRICT
    data_evidence_class: EvidenceClass = Field(
        description="the class the engine analysed the data as: UNVALIDATED for every anonymous upload, SIMULATED only "
                    "for a bundled ColdSense sample")
    evidence_basis: EvidenceBasis = Field(description="why the data has that class")
    declared_evidence: DeclaredEvidence | None = Field(
        None, description="what the uploader said the data is: declared by uploader, not verified; no effect")
    declaration_status: DeclarationStatus
    result_evidence_class: EvidenceClass | None = Field(
        None, description="class of model outputs in this response: SIMULATED for simulated input, otherwise "
                          "EXPECTED; never REAL. None when the response holds no model output")
    source_filename: str = Field(description="display name of the uploaded file (sanitised; never used as a path)")
    source_sha256: str
    source_size_bytes: int
    method: str = Field(description="which accepted ColdSense stage produced the result")
    method_version: str | None = None
    validation_scope: ValidationScope = Field(description="Phase 5 scope: what any check on this data can speak for")
    validation_scope_source: Literal["PHASE5_ASSESSMENT", "PHASE5_SCOPE_TABLE"] = Field(
        description="from this result's Phase 5 assessment, or from the Phase 5 evidence-class table "
                    "(SCOPE_FOR_EVIDENCE) when no assessment ran")
    coldsense_version: str
    computed_at: datetime


# ---------------------------------------------------------------- health and cases
class FeatureState(str, Enum):
    ENABLED = "ENABLED"
    FEATURE_DISABLED = "FEATURE_DISABLED"


class Health(BaseModel):
    model_config = _STRICT
    status: Literal["ok"] = "ok"
    coldsense_version: str
    copilot: FeatureState
    copilot_reason: str


class KnowledgeView(BaseModel):
    """A read-only Phase 7 retrieval result. ``status`` distinguishes a working retrieval from a missing index;
    an answer with ``NO_RELEVANT_DOCUMENTATION`` is still a working result (the refusal is the answer)."""
    model_config = _STRICT
    query: str
    status: Literal["AVAILABLE", "NOT_AVAILABLE"] = "AVAILABLE"
    answer: Answer | None = None
    reason: str | None = None


class CsrfToken(BaseModel):
    model_config = _STRICT
    csrf_token: str = Field(description="send as X-CSRF-Token on every POST/DELETE; bound to this browser's cs_csrf "
                                        "cookie")
    header: Literal["X-CSRF-Token"] = "X-CSRF-Token"


class SampleInfo(BaseModel):
    model_config = _STRICT
    sample_id: str
    filename: str
    description: str
    evidence_class: Literal["SIMULATED"] = "SIMULATED"


class StepState(str, Enum):
    READY = "READY"                    # result available
    REFUSED = "REFUSED"                # the stage ran and refused, with reasons
    NOT_RUN = "NOT_RUN"                # can be requested
    BLOCKED = "BLOCKED"                # cannot run: an earlier stage refused the data


class QualityHeadline(BaseModel):
    model_config = _STRICT
    verdict: Verdict
    suitable_for_analysis: bool
    rows_read: int
    rows_usable: int
    start: datetime | None
    end: datetime | None
    duration_hours: float
    coverage_pct: float
    blocking_issues: int
    warnings: int


class CaseSteps(BaseModel):
    model_config = _STRICT
    quality: StepState
    analysis: StepState
    prediction: StepState


class CaseStatus(BaseModel):
    model_config = _STRICT
    case_id: str
    created_at: datetime
    expires_at: datetime
    filename: str
    size_bytes: int
    sha256: str
    evidence_class: EvidenceClass = Field(description="engine evidence class (see ResultProvenance)")
    evidence_basis: EvidenceBasis
    declared_evidence: DeclaredEvidence | None
    declaration_status: DeclarationStatus
    validation_scope: ValidationScope
    quality: QualityHeadline
    steps: CaseSteps
    last_prediction_outcome: Literal["FORECAST", "REFUSED"] | None = None


# ---------------------------------------------------------------- quality (Phase 2)
class QualityView(BaseModel):
    model_config = _STRICT
    case_id: str
    headline: QualityHeadline
    issues: list[Issue]
    limitations: list[str]
    report: DataQualityReport = Field(description="the full Phase 2 data-quality report, unchanged")
    provenance: ResultProvenance


# ---------------------------------------------------------------- analysis (Phase 3 events)
class AnalysisRequest(BaseModel):
    model_config = _STRICT
    upper_threshold_c: float | None = Field(None, allow_inf_nan=False, description="°C; set by you, never inferred")
    lower_threshold_c: float | None = Field(None, allow_inf_nan=False, description="°C; set by you, never inferred")


class DiagnosisSummary(BaseModel):
    """Phase 6 outcome of this analysis run (details: evidence, next-check and investigation views)."""
    model_config = _STRICT
    status: DiagnosisStatus = Field(description="HYPOTHESES, NO_FINDING or INSUFFICIENT_EVIDENCE (a refusal)")
    refusal_reason: str | None
    findings: int
    hypotheses: int
    next_checks: int
    next_best_check: str | None = Field(description="the check ranked first by Phase 6: a work priority, not a "
                                                     "confidence in any explanation")


class AnalysisView(BaseModel):
    model_config = _STRICT
    case_id: str
    outcome: Literal["COMPLETED", "REFUSED"]
    reasons: list[str] = Field(default_factory=list, description="why segmentation refused (empty when COMPLETED)")
    diagnosis: DiagnosisSummary
    upper_threshold_c: float | None
    lower_threshold_c: float | None
    time_basis: str
    summary: dict[str, int | float | str | bool]
    events: list[Event]
    limitations: list[str]
    provenance: ResultProvenance


# ---------------------------------------------------------------- prediction (Phases 4-5)
class PredictionRequest(BaseModel):
    model_config = _STRICT
    horizon_min: float = Field(gt=0, le=MAX_HORIZON_MIN, allow_inf_nan=False, description="forecast horizon, minutes")
    threshold_c: float | None = Field(None, allow_inf_nan=False, description="°C; a crossing is reported only if set")
    direction: Direction = Direction.UPWARD
    question: ForecastQuestion = ForecastQuestion.NO_COOLING
    passive_since: datetime | None = Field(None, description="declared start of a period without active cooling")
    assess_reliability: bool = Field(True, description="add the Phase 5 reliability assessment (causal view of "
                                                        "the log)")


class RecommendationView(BaseModel):
    model_config = _STRICT
    missing_input: str
    priority: Priority
    necessity: Necessity
    rationale: str
    acquisition_guidance: str
    retry_conditions: str | None = None


class _PredictionBase(BaseModel):
    model_config = _STRICT
    case_id: str
    request: PredictionRequest
    data_verdict: Verdict
    evidence_class: EvidenceClass = Field(description="class of the result: SIMULATED or EXPECTED, never REAL")
    question: ForecastQuestion
    operating_regime: str
    regime_evidence: list[str]
    warnings: list[str]
    limitations: list[str]
    recommendations: list[RecommendationView]
    provenance: ResultProvenance


class SelectedModel(BaseModel):
    model_config = _STRICT
    name: str
    version: str | None
    solution_method: str | None
    forecast_kind: str | None = Field(None, description="PHYSICAL or BASELINE")
    selection_reason: str | None = None
    fallback_reason: str | None = None


class ObservedTemperature(BaseModel):
    model_config = _STRICT
    value_c: float | None
    provenance: Provenance
    measured_at: datetime | None
    note: str


class ThresholdView(BaseModel):
    model_config = _STRICT
    value_c: float
    direction: Direction
    provenance: Provenance
    note: str


class CrossingView(BaseModel):
    model_config = _STRICT
    status: CrossingStatus
    time: datetime | None
    minutes: float | None


class TrajectoryPointView(BaseModel):
    model_config = _STRICT
    ts: datetime
    minutes: float
    temp_c: float
    provenance: Provenance


class IntervalView(BaseModel):
    """Phase 5 interval at the horizon. Bounds are present only when the interval was emitted (SUPPORTED or
    CONSERVATIVE); otherwise ``status`` says why there is none."""
    model_config = _STRICT
    status: str
    horizon_min: float | None
    nominal_coverage: float
    lower_c: float | None
    upper_c: float | None
    conditional_on: list[str]


class CrossingIntervalView(BaseModel):
    model_config = _STRICT
    status: str
    lower_min: float | None
    upper_min: float | None


class ReliabilityView(BaseModel):
    model_config = _STRICT
    status: ReliabilityStatus = Field(description="rule-based classification; NOT a probability")
    validation_scope: ValidationScope
    method: str
    reason: str | None = Field(None, description="why the status is below HIGH")
    gates_failed: list[str]


class BaselineView(BaseModel):
    """A naive reference forecast (persistence, recent slope) as the Phase 4 engine reported it."""
    model_config = _STRICT
    method: str
    applicable: bool
    reason: str | None
    temp_at_horizon_c: float | None
    crossing_status: CrossingStatus | None
    crossing_minutes: float | None
    selection_mae_c: float | None
    test_mae_c: float | None


class PredictionSuccess(_PredictionBase):
    outcome: Literal["FORECAST"] = "FORECAST"
    model: SelectedModel
    prediction_start: datetime
    horizon_min: float
    validated_horizon_min: float | None
    initial_temperature: ObservedTemperature
    threshold: ThresholdView | None
    crossing: CrossingView
    trajectory: list[TrajectoryPointView] = Field(min_length=1)
    scenario: list[str]
    assumptions: list[str]
    baselines: list[BaselineView] = Field(description="baseline comparison from the Phase 4 engine")
    interval: IntervalView | None = None
    crossing_interval: CrossingIntervalView | None = None
    reliability: ReliabilityView | None = None


class RefusalDetail(BaseModel):
    model_config = _STRICT
    code: RefusalCode
    reason: str
    failed_condition: str | None
    needed_to_retry: str | None


class PredictionRefusal(_PredictionBase):
    """A refused prediction. By construction it has no field for a forecast value (see module docstring)."""
    outcome: Literal["REFUSED"] = "REFUSED"
    refusal: RefusalDetail


PredictionResponse = Annotated[PredictionSuccess | PredictionRefusal, Field(discriminator="outcome")]

# Field names a refusal must never carry, at any depth (tests check the schema and every refusal response).
REFUSAL_FORBIDDEN_FIELDS = frozenset({
    "trajectory", "crossing", "predicted_crossing_time", "predicted_crossing_minutes", "model", "selected_model",
    "parameter_estimates", "parameters", "baseline_results", "interval", "crossing_interval", "reliability",
    "initial_temperature", "prediction_start", "temp_c", "scenario", "baselines", "temp_at_horizon_c",
    "crossing_minutes",
})


# ---------------------------------------------------------------- copilot (disabled in Stage 1)
class CopilotRequest(BaseModel):
    model_config = _STRICT
    question: str = Field(min_length=1, max_length=2000)


# ---------------------------------------------------------------- Stage 2: analysis workspace
MODEL_INFERENCE = "MODEL_INFERENCE"
SYSTEM_RECOMMENDATION = "SYSTEM_RECOMMENDATION"
OBSERVED_FACT = "OBSERVED_FACT"
HYPOTHESIS_NOTE = "hypothesis, not a confirmed fault"


class TimelinePoint(BaseModel):
    """One usable Phase 2 observation (the analysis frame), copied; °C."""
    model_config = _STRICT
    ts: datetime
    temp_c: float
    ambient_c: float | None
    compressor_on: bool | None
    source_line: int


class TimelineRun(BaseModel):
    """Readings between two Phase 3 DATA_GAP events. Runs are never joined: a gap is drawn as a gap."""
    model_config = _STRICT
    run_index: int
    start: datetime
    end: datetime
    points_total: int
    points: list[TimelinePoint]


class TimelineGap(BaseModel):
    model_config = _STRICT
    event_id: str
    start: datetime = Field(description="last usable reading before the gap")
    end: datetime = Field(description="first usable reading after the gap")
    duration_min: float | None
    rationale: str


class DisplaySampling(BaseModel):
    """Display-only thinning of long series; every analysis used every reading."""
    model_config = _STRICT
    downsampled: bool
    method: str
    points_total: int
    points_shown: int
    max_points: int


class TimelineView(BaseModel):
    model_config = _STRICT
    case_id: str
    unit: Literal["°C"] = "°C"
    time_basis: str
    start: datetime | None
    end: datetime | None
    upper_threshold_c: float | None
    lower_threshold_c: float | None
    ambient_available: bool
    compressor_recorded: bool
    runs: list[TimelineRun]
    gaps: list[TimelineGap]
    events: list[Event] = Field(description="Phase 3 events, unchanged (trends, excursions, compressor, gaps)")
    display: DisplaySampling
    limitations: list[str]
    provenance: ResultProvenance


class NotAvailableBlock(BaseModel):
    model_config = _STRICT
    status: Literal["NOT_AVAILABLE"] = "NOT_AVAILABLE"
    reason: str


class PredictionStatusView(BaseModel):
    """Support/refusal status of the case's latest prediction. A refusal carries its code and reason only."""
    model_config = _STRICT
    outcome: Literal["FORECAST", "REFUSED", "NOT_RUN"]
    refusal_code: RefusalCode | None = None
    refusal_reason: str | None = None
    reliability_status: ReliabilityStatus | None = None
    validation_scope: ValidationScope | None = None


class EvidenceView(BaseModel):
    """Facts, inferences and recommendations kept apart (spec §8): OBSERVED_FACT from the log (Phase 2/3 via Phase 6),
    MODEL_INFERENCE (Phase 6 rule assessments of candidate explanations), SYSTEM_RECOMMENDATION (Phase 6 checks).
    Document facts need Phase 7 retrieval, which is not exposed: NOT_AVAILABLE."""
    model_config = _STRICT
    case_id: str
    diagnosis_status: DiagnosisStatus
    diagnosis_refusal_reason: str | None
    rules_validation: str
    data_verdict: Verdict
    prediction: PredictionStatusView
    observed_facts: list[ObservedFact]
    observed_fact_label: Literal["OBSERVED_FACT"] = OBSERVED_FACT
    document_facts: NotAvailableBlock
    findings: list[Finding] = Field(description="Phase 6 findings with every explanation's supporting, contradicting, "
                                                "neutral and missing evidence")
    inference_label: Literal["MODEL_INFERENCE"] = MODEL_INFERENCE
    recommendations: list[NextCheck]
    recommendation_label: Literal["SYSTEM_RECOMMENDATION"] = SYSTEM_RECOMMENDATION
    data_quality_limitations: list[str]
    limitations: list[str]
    provenance: ResultProvenance


class NextCheckView(BaseModel):
    model_config = _STRICT
    case_id: str
    status: DiagnosisStatus
    refusal_reason: str | None
    next_best_check: str | None
    checks: list[NextCheck] = Field(description="Phase 6 checks in Phase 6's rank order; ranks are its own, no score")
    label: Literal["SYSTEM_RECOMMENDATION"] = SYSTEM_RECOMMENDATION
    limitations: list[str]
    provenance: ResultProvenance


class InvestigationView(BaseModel):
    model_config = _STRICT
    case_id: str
    status: DiagnosisStatus
    refusal_reason: str | None
    hypotheses: list[Hypothesis] = Field(description="Phase 6 hypothesis objects, unchanged (status PROPOSED)")
    hypothesis_label: Literal["MODEL_INFERENCE"] = MODEL_INFERENCE
    hypothesis_note: Literal["hypothesis, not a confirmed fault"] = HYPOTHESIS_NOTE
    findings: list[Finding]
    next_checks: list[NextCheck]
    recommendation_label: Literal["SYSTEM_RECOMMENDATION"] = SYSTEM_RECOMMENDATION
    rules_validation: str
    limitations: list[str]
    provenance: ResultProvenance


class IncidentView(BaseModel):
    """Technician findings for this case: drafts (not history), confirmed records, and the outcome state of each.
    The records are the Phase 8 trusted-confirmation store's records, copied; nothing is inferred from timing."""
    model_config = _STRICT
    case_id: str
    status: Literal["AVAILABLE"] = "AVAILABLE"
    drafts: list[FindingDraftView]
    confirmed: list[FindingRecordView]
    counts: FindingCounts
    provenance: ResultProvenance


# ---------------------------------------------------------------- Stage 3: technician findings
FindingOutcome = Literal["UNKNOWN", "RESOLVED", "NOT_RESOLVED"]


class FindingDraftRequest(BaseModel):
    """The technician's own statement. It is never generated by an LLM and is written only on explicit confirm."""
    model_config = _STRICT
    statement: str = Field(min_length=1, max_length=4000,
                           description="what the technician observed, in their own words; shown verbatim")
    event_window: str | None = Field(None, max_length=200,
                                     description="free-text window the technician associates with the observation")

    @field_validator("statement")
    @classmethod
    def _statement_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("statement must not be blank")
        if "\x00" in v:
            raise ValueError("statement must not contain NUL")
        return v

    @field_validator("event_window")
    @classmethod
    def _window_clean(cls, v: str | None) -> str | None:
        if v is not None and "\x00" in v:
            raise ValueError("event_window must not contain NUL")
        return v


class FindingConfirmRequest(BaseModel):
    """The token issued with the exact draft, and the outcome the technician explicitly entered (default UNKNOWN)."""
    model_config = _STRICT
    token: str = Field(min_length=1, max_length=512,
                       description="the single-use token returned with this draft; never shown by the UI")
    outcome: FindingOutcome = Field("UNKNOWN", description="set only from a value the user entered; PENDING is refused")


class FindingDraftView(BaseModel):
    model_config = _STRICT
    draft_id: str
    asset_id: str
    statement: str
    event_window: str | None
    content_hash: str
    created_at: datetime
    status: Literal["PENDING", "CANCELLED"]


class FindingDraftCreated(FindingDraftView):
    confirmation_token: str = Field(description="single-use, hash-bound, expiring; returned once, never logged")
    token_expires_at: datetime


class FindingRecordView(BaseModel):
    model_config = _STRICT
    record_id: str
    asset_id: str
    statement: str
    event_window: str | None
    content_hash: str
    created_at: datetime
    confirmed_at: datetime
    confirmed_via: str = Field(description="the Phase 8 store's audit label, preserved verbatim")
    outcome: Outcome
    source: str = Field(description="the Phase 8 store's source label, preserved verbatim")
    confirmation_channel: Literal["WEB_APPLICATION"] = Field(
        "WEB_APPLICATION", description="this application's actual confirming channel (the Phase 8 labels predate it)")


class FindingCounts(BaseModel):
    model_config = _STRICT
    confirmed: int
    drafts_pending: int
    drafts_cancelled: int
    outcomes_unknown: int
    outcomes_verified: int = Field(description="confirmed records with RESOLVED or NOT_RESOLVED")


class FindingsView(BaseModel):
    model_config = _STRICT
    case_id: str
    drafts: list[FindingDraftView]
    records: list[FindingRecordView]
    counts: FindingCounts
    provenance: ResultProvenance


class ReportView(BaseModel):
    """Built by the Phase 8 ``generate_service_report`` tool from confirmed records only. A refusal has no text."""
    model_config = _STRICT
    case_id: str
    status: Literal["OK", "REFUSED"]
    report: str | None = Field(None, description="present only when the numeric check passed; plain text")
    n_confirmed: int
    n_drafts_excluded: int
    violations: list[str] = Field(default_factory=list)
    confirmed: list[FindingRecordView]
    drafts_excluded: list[FindingDraftView]
    numeric_check: Literal["PASSED", "REFUSED"]
    generated_at: datetime
    provenance: ResultProvenance


# IncidentView refers to the finding views defined below it; resolve those forward references once, here.
IncidentView.model_rebuild()
