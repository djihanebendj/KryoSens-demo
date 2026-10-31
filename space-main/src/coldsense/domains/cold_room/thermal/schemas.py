"""Result and request types of the thermal engine (Phase 4).

Conventions carried over from Phases 1-3: a refusal is a result with reasons,
never an exception; every number says where it came from; a missing value is
``None`` with an explanation, never 0.
"""

from __future__ import annotations

import math
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from coldsense.core.evidence import EvidenceClass

from . import limits as L


class Provenance(str, Enum):
    OBSERVED = "OBSERVED"        # read from a usable row of the log
    FITTED = "FITTED"            # estimated by fitting a model to observations
    DERIVED = "DERIVED"          # computed from observed values (mean, interpolation, difference)
    PREDICTED = "PREDICTED"      # model output for a time not observed
    SIMULATED = "SIMULATED"      # produced by a synthetic generator
    ASSUMED = "ASSUMED"          # stated by the user or by a documented scenario, not measured
    UNAVAILABLE = "UNAVAILABLE"  # not computable from the available data


class InputStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    MISSING = "MISSING"
    INVALID = "INVALID"
    NOT_VERIFIED = "NOT_VERIFIED"   # present, but what it physically represents is not established


class ForecastStatus(str, Enum):
    FORECAST = "FORECAST"
    REFUSED = "REFUSED"


class ForecastQuestion(str, Enum):
    NO_COOLING = "NO_COOLING"        # room air if there is no active cooling from the origin (default)
    CONTINUATION = "CONTINUATION"    # room air if the behaviour observed at the origin continues


class RefusalCode(str, Enum):
    INVALID_REQUEST = "INVALID_REQUEST"
    INPUT_CONFLICT = "INPUT_CONFLICT"                      # two inputs contradict each other
    DATA_REFUSED = "DATA_REFUSED"                          # Phase 2/3 refused the log
    MISSING_REQUIRED_INPUT = "MISSING_REQUIRED_INPUT"
    MODEL_NOT_APPLICABLE = "MODEL_NOT_APPLICABLE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"                # no usable segment
    INSUFFICIENT_IDENTIFIABILITY = "INSUFFICIENT_IDENTIFIABILITY"
    IMPLAUSIBLE_PARAMETERS = "IMPLAUSIBLE_PARAMETERS"
    POOR_FIT = "POOR_FIT"
    FIT_FAILED = "FIT_FAILED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"        # cannot be checked on held-out data
    NO_ADEQUATE_MODEL = "NO_ADEQUATE_MODEL"                # every method's held-out error too large
    NUMERICAL_ERROR = "NUMERICAL_ERROR"


class CrossingStatus(str, Enum):
    NO_THRESHOLD = "NO_THRESHOLD"                # none requested
    CROSSING_PREDICTED = "CROSSING_PREDICTED"
    ALREADY_CROSSED = "ALREADY_CROSSED"          # the initial observation is at or beyond the threshold
    NOT_REACHABLE = "NOT_REACHABLE"              # never, under the model and its stated assumptions
    NOT_WITHIN_HORIZON = "NOT_WITHIN_HORIZON"    # reachable under the model, but after the horizon
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    INSUFFICIENT_IDENTIFIABILITY = "INSUFFICIENT_IDENTIFIABILITY"
    MODEL_NOT_APPLICABLE = "MODEL_NOT_APPLICABLE"
    MISSING_REQUIRED_INPUT = "MISSING_REQUIRED_INPUT"
    INVALID_REQUEST = "INVALID_REQUEST"


class Direction(str, Enum):
    UPWARD = "UPWARD"        # crossing = temperature rises to or above the threshold
    DOWNWARD = "DOWNWARD"    # crossing = temperature falls to or below the threshold


class Regime(str, Enum):
    PASSIVE = "PASSIVE"      # no active cooling: recorded compressor OFF, or declared by the user
    ACTIVE = "ACTIVE"        # recorded compressor ON
    UNKNOWN = "UNKNOWN"      # no evidence either way (no compressor state, no declaration)


class RegimeBasis(str, Enum):
    RECORDED_COMPRESSOR_STATE = "RECORDED_COMPRESSOR_STATE"
    DECLARED_BY_USER = "DECLARED_BY_USER"
    RECORDED_EVENT = "RECORDED_EVENT"        # a power interruption in the supplied event log
    NONE = "NONE"


class Identifiability(str, Enum):
    IDENTIFIABLE = "IDENTIFIABLE"
    INSUFFICIENT_IDENTIFIABILITY = "INSUFFICIENT_IDENTIFIABILITY"
    NOT_ASSESSED = "NOT_ASSESSED"


class CandidateOutcome(str, Enum):
    SELECTED = "SELECTED"
    VALIDATED_NOT_SELECTED = "VALIDATED_NOT_SELECTED"
    HINDCAST_ONLY = "HINDCAST_ONLY"      # fitted and checked with recorded inputs; cannot forecast
    REJECTED = "REJECTED"


# ---------------------------------------------------------------- request
class ForecastRequest(BaseModel):
    """What the caller asks. Every float must be finite; nothing is clamped."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    horizon_min: float = Field(gt=0, le=L.MAX_HORIZON_MIN, description="explicit forecast horizon, minutes")
    threshold_c: float | None = Field(None, description="°C; crossing is reported only if set")
    direction: Direction = Field(Direction.UPWARD, description="which way counts as crossing the threshold")
    origin: datetime | None = Field(None, description="forecast from the last usable observation at or "
                                                      "before this time; default: the last observation")
    passive_since: datetime | None = Field(
        None, description="declare that the room had NO active cooling from this time to the origin "
                          "(e.g. power cut). An ASSUMPTION stated by the user; contradicted by recorded ON state")
    assumed_equilibrium_c: float | None = Field(
        None, description="°C the room would settle at without cooling; an ASSUMPTION, enables that variant")
    future_ambient_c: float | None = Field(
        None, description="°C ambient over the horizon (scenario, ASSUMED); needed when ambient varied")
    step_min: float | None = Field(None, gt=0, description="trajectory spacing; default: median sampling interval")
    question: ForecastQuestion = Field(
        ForecastQuestion.NO_COOLING, description="NO_COOLING (default): no active cooling from the origin; "
                                                 "CONTINUATION: the behaviour observed at the origin continues")
    allow_extrapolation: bool = Field(
        False, description="allow a horizon beyond the one checked on held-out data; only a validated physical "
                           "model may then extrapolate, and the result says so. Default: refuse")

    @model_validator(mode="after")
    def _consistent(self) -> ForecastRequest:
        for name in ("origin", "passive_since"):
            v = getattr(self, name)
            if v is not None and v != v:            # pandas NaT ("not a time") passes as a datetime
                raise ValueError(f"{name} is not a valid time")
        if self.passive_since is not None and self.origin is not None and self.passive_since >= self.origin:
            raise ValueError("passive_since must be before origin")
        if self.step_min is not None and self.horizon_min / self.step_min > L.MAX_TRAJECTORY_POINTS:
            raise ValueError(f"horizon_min / step_min exceeds {L.MAX_TRAJECTORY_POINTS} trajectory points")
        return self


# ---------------------------------------------------------------- building blocks
class Quantity(BaseModel):
    value: float | None
    unit: str
    provenance: Provenance
    note: str = ""


class InputItem(BaseModel):
    name: str
    family: str
    unit: str | None
    status: InputStatus
    detail: str


class ParameterEstimate(BaseModel):
    name: str
    unit: str
    value: float | None
    provenance: Provenance
    method: str
    objective: str = ""
    constraints: str = ""
    identifiability: Identifiability = Identifiability.NOT_ASSESSED
    profile_range: tuple[float, float] | None = Field(
        None, description="range compatible with the data by a profile-likelihood test (identifiability check); "
                          "NOT a calibrated forecast uncertainty")
    note: str = ""


class SegmentRecord(BaseModel):
    segment_id: str
    regime: str = Field(description="PASSIVE_RECORDED_OFF, PASSIVE_DECLARED or ACTIVE_RECORDED_ON")
    start: datetime
    end: datetime
    n_observations: int
    duration_min: float
    change_c: float | None
    first_line: int
    last_line: int
    phase3_events: list[str] = Field(default_factory=list)
    role: str | None = Field(None, description="FIT, SELECT or TEST (chronological split) when accepted")
    accepted: bool
    reason: str | None = None


class FitDiagnostics(BaseModel):
    model: str
    data_role: str = Field(description="which data were fitted, e.g. 'FIT split' or 'all eligible before origin'")
    metrics_kind: str = "TRAINING (in-sample); not forecast performance"
    n_observations: int
    n_segments: int
    fit_duration_min: float
    median_interval_s: float | None
    n_parameters: int = Field(description="rate + equilibrium parameters + one initial state per segment")
    n_effective: float | None = Field(None, description="observations corrected for residual autocorrelation")
    parameters: list[ParameterEstimate]
    residual_mean_c: float | None
    residual_std_c: float | None
    mae_c: float | None
    rmse_c: float | None
    r2_within_segment: float | None = Field(
        None, description="1 - SSE / sum of squares about each segment's own mean; None if undefined")
    residual_lag1_autocorr: float | None
    residual_trend_c: float | None = Field(
        None, description="slope of residuals vs position within segment (°C per segment length)")
    fit_status: str = Field(description="CONVERGED, AT_SEARCH_BOUND, SINGULAR or FAILED")
    identifiability: Identifiability
    identifiability_reason: str
    warnings: list[str] = Field(default_factory=list)


class HorizonMetrics(BaseModel):
    horizon_min: float
    n_cases: int = Field(description="origin-horizon cases with observed truth")
    n_forecasts: int
    coverage: float | None = Field(description="n_forecasts / n_cases; None when there are no cases")
    refusal_rate: float | None
    mae_c: float | None
    rmse_c: float | None
    n_common: int = Field(0, description="cases where every compared method forecast")
    mae_common_c: float | None = None
    rmse_common_c: float | None = None
    n_observed_crossings: int = 0
    crossing_mae_min: float | None = Field(None, description="mean |predicted - observed| crossing time, cases "
                                                             "where both crossed within the horizon")
    n_missed_crossings: int = 0
    n_false_crossings: int = 0


class MethodEvaluation(BaseModel):
    method: str
    role: str = Field(description="SELECT (method choice), TEST (final evaluation) or HINDCAST")
    regime: str
    fitted_on: str
    horizons: list[HorizonMetrics]


class Comparison(BaseModel):
    role: str
    horizon_min: float
    physical_method: str
    best_baseline: str | None
    n_common: int
    physical_better: int
    baseline_better: int
    tie: int
    neither_adequate: int


class ValidationDiagnostics(BaseModel):
    status: str = Field(description="VALIDATED or NOT_VALIDATED")
    reason: str
    split: str
    origin_stride_min: float | None = None
    horizons_min: list[float] = Field(default_factory=list)
    adequacy_threshold_c: float = L.ADEQUATE_ABS_ERROR_C
    selection_horizon_min: float | None = None
    evaluations: list[MethodEvaluation] = Field(default_factory=list)
    metrics_kind: str = "OUT-OF-SAMPLE: parameters fitted on earlier data only; SIMULATED or EXPECTED per input"


class CandidateResult(BaseModel):
    name: str
    version: str
    kind: str = Field(description="BASELINE or PHYSICAL")
    outcome: CandidateOutcome
    required_inputs: list[str]
    missing_inputs: list[str] = Field(default_factory=list)
    rejection_code: RefusalCode | None = None
    rejection_reason: str | None = None
    fit: FitDiagnostics | None = None
    selection_mae_c: float | None = None
    test_mae_c: float | None = None
    assumptions: list[str] = Field(default_factory=list)


class RejectedModel(BaseModel):
    name: str
    code: RefusalCode | str
    reason: str


class TrajectoryPoint(BaseModel):
    ts: datetime
    minutes: float
    temp_c: float
    provenance: Provenance = Provenance.PREDICTED


def supported_fit(fd: FitDiagnostics) -> bool:
    """The checks a schema can verify: identifiable, converged, within-segment R² >= FIT_MIN_R2. (Plausibility
    of the equilibrium is checked by the engine, which withholds a rejected final fit.)"""
    return (fd.identifiability == Identifiability.IDENTIFIABLE and fd.fit_status == "CONVERGED"
            and fd.r2_within_segment is not None and fd.r2_within_segment >= L.FIT_MIN_R2)


BASELINE_PREDICTION_FIELDS = ("temp_at_horizon_c", "slope_c_per_min", "crossing_status", "crossing_minutes")
FORECAST_CROSSINGS = frozenset({CrossingStatus.CROSSING_PREDICTED, CrossingStatus.ALREADY_CROSSED,
                                CrossingStatus.NOT_REACHABLE, CrossingStatus.NOT_WITHIN_HORIZON})


class BaselineForecast(BaseModel):
    method: str
    applicable: bool
    reason: str | None = None
    temp_at_horizon_c: float | None = None
    slope_c_per_min: float | None = None
    crossing_status: CrossingStatus | None = None
    crossing_minutes: float | None = None
    selection_mae_c: float | None = None
    test_mae_c: float | None = None


class Threshold(BaseModel):
    value_c: float
    direction: Direction
    provenance: Provenance = Provenance.ASSUMED
    note: str = "set by the user"


# ---------------------------------------------------------------- follow-up: inputs, selection, advice
class Priority(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Necessity(str, Enum):
    NECESSARY_FOR_THIS_QUESTION = "NECESSARY_FOR_THIS_QUESTION"   # without it this question cannot be answered here
    POTENTIAL_IMPROVEMENT = "POTENTIAL_IMPROVEMENT"   # useful improvement: an implemented capability could use it
    #                                                   for this forecast; any benefit must be shown on held-out data
    OPTIONAL_CONTEXT = "OPTIONAL_CONTEXT"             # changes no computed value; documents what the result means
    FUTURE_CAPABILITY = "FUTURE_CAPABILITY"           # only deferred (unimplemented) models would use it


NECESSITY_ORDER = (Necessity.NECESSARY_FOR_THIS_QUESTION, Necessity.POTENTIAL_IMPROVEMENT,
                   Necessity.OPTIONAL_CONTEXT, Necessity.FUTURE_CAPABILITY)
NO_EFFECT_BENEFIT = {
    Necessity.OPTIONAL_CONTEXT: "none on any computed value: it documents what the result represents",
    Necessity.FUTURE_CAPABILITY: "none on current results: no implemented model uses it; any benefit would have to "
                                 "be shown by held-out comparison once a model that uses it exists",
}


class InputState(BaseModel):
    name: str
    family: str
    unit: str | None
    status: InputStatus
    source: str
    provenance: str
    used_by: list[str] = Field(default_factory=list)
    detail: str


class InputCapabilities(BaseModel):
    available_inputs: list[InputState] = Field(default_factory=list)
    missing_inputs: list[InputState] = Field(default_factory=list)
    invalid_inputs: list[InputState] = Field(default_factory=list)
    assumptions_used: list[str] = Field(default_factory=list)
    quality_warnings: list[str] = Field(default_factory=list)
    analysed_sensor: str | None = Field(None, description="the single temperature column this result describes")


class Alternative(BaseModel):
    model: str
    status: str = Field(description="UNAVAILABLE_MISSING_INPUT, NOT_APPLICABLE, REJECTED, NOT_APPLICABLE_AT_ORIGIN, "
                                    "WITHIN_MARGIN_SIMPLER_KEPT, HIGHER_HELD_OUT_ERROR, HINDCAST_ONLY or SELECTED")
    detail: str


class ModelSelection(BaseModel):
    question: ForecastQuestion
    candidate_models: list[str]
    eligible_models: list[str] = Field(description="inputs and regime allow the model; it was fitted/evaluated")
    validated_models: list[str] = Field(description="compared on held-out data at the validated horizon")
    rejected_models: list[RejectedModel] = Field(default_factory=list)
    selected_model: str | None = None
    forecast_kind: str = Field(description="PHYSICAL, BASELINE or NONE")
    selection_reason: str
    fallback_reason: str | None = Field(None, description="why more physically informative alternatives were not used")
    alternatives: list[Alternative] = Field(default_factory=list)


class LimitationDetails(BaseModel):
    missing_input_impact: list[str] = Field(default_factory=list)
    model_limitations: list[str] = Field(default_factory=list)
    horizon_limitations: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    missing_input: str
    kind: str = Field(description="INPUT (a signal or record), SCENARIO (a stated future value) or DATA (more history)")
    rationale: str
    potential_capability: str
    enabled_models: list[str] = Field(default_factory=list)
    acquisition_guidance: str
    units_and_resolution: str | None = None
    priority: Priority
    priority_rationale: str
    necessity: Necessity
    expected_benefit: str = Field("not guaranteed: compare held-out errors on this room with and without the input "
                                  "to measure any improvement")


# ---------------------------------------------------------------- Phase 5: uncertainty and reliability
_FINITE = ConfigDict(allow_inf_nan=False)     # a reported number is finite or absent (None), never inf/NaN


class ValidationScope(str, Enum):
    """What the held-out checks can speak for, set by the input's evidence class (SCOPE_FOR_EVIDENCE)."""
    SIMULATED_ONLY = "SIMULATED_ONLY"                    # SIMULATED input: statements hold for the simulation only
    REAL_WORLD_UNVALIDATED = "REAL_WORLD_UNVALIDATED"    # REAL (measured) input; method not validated prospectively
    EXPECTED_ONLY = "EXPECTED_ONLY"                      # EXPECTED input (a model output): checks compare with that
    #                                                      output, not with a room
    UNVALIDATED_PROVENANCE = "UNVALIDATED_PROVENANCE"    # provenance unknown: treated neither as measured nor as
    #                                                      simulated; no real-world statement
    REAL_WORLD_VALIDATED = "REAL_WORLD_VALIDATED"        # REAL input with recorded real-world validation evidence


# The only mapping from input evidence to scope. REAL_WORLD_VALIDATED is never derived from a label: it needs REAL
# input AND referenced validation evidence (ReliabilityAssessment / ForecastResult validators).
SCOPE_FOR_EVIDENCE = {
    EvidenceClass.SIMULATED: ValidationScope.SIMULATED_ONLY,
    EvidenceClass.REAL: ValidationScope.REAL_WORLD_UNVALIDATED,
    EvidenceClass.EXPECTED: ValidationScope.EXPECTED_ONLY,
    EvidenceClass.UNVALIDATED: ValidationScope.UNVALIDATED_PROVENANCE,
}
HIGH_ALLOWED_SCOPES = frozenset({ValidationScope.SIMULATED_ONLY, ValidationScope.REAL_WORLD_VALIDATED})


def min_conformal_sample(level: float) -> int:
    """Smallest number of independent units for which both two-sided conformal order statistics exist at this
    level: floor((n+1)a/2) >= 1 and ceil((n+1)(1-a/2)) <= n, a = 1 - level  <=>  n >= 2/a - 1."""
    return math.ceil(2.0 / (1.0 - level) - 1.0 - 1e-9)


class ReliabilityStatus(str, Enum):
    """An engineering classification from explicit gates; NOT a probability and not a score."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNSUPPORTED = "UNSUPPORTED"


RELIABILITY_ORDER = (ReliabilityStatus.HIGH, ReliabilityStatus.MEDIUM, ReliabilityStatus.LOW,
                     ReliabilityStatus.UNSUPPORTED)


class GateOutcome(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ReliabilityGate(BaseModel):
    name: str
    outcome: GateOutcome
    cap_if_failed: ReliabilityStatus = Field(description="the highest status allowed when this gate fails")
    evidence: str


class ReliabilityAssessment(BaseModel):
    status: ReliabilityStatus
    validation_scope: ValidationScope
    method: str
    decision_reasons: list[str] = Field(default_factory=list)
    gates: list[ReliabilityGate]
    gates_passed: list[str]
    gates_failed: list[str]
    real_world_evidence: list[str] = Field(default_factory=list, description="references to recorded real-world "
                                           "validation; required for REAL_WORLD_VALIDATED")

    @model_validator(mode="after")
    def _consistent(self) -> ReliabilityAssessment:
        failed = [g for g in self.gates if g.outcome == GateOutcome.FAIL]
        if self.gates_failed != [g.name for g in failed] or self.gates_passed != [
                g.name for g in self.gates if g.outcome == GateOutcome.PASS]:
            raise ValueError("gates_passed / gates_failed must list the gate outcomes")
        expected = max((g.cap_if_failed for g in failed), key=RELIABILITY_ORDER.index, default=ReliabilityStatus.HIGH)
        if self.status != expected:
            raise ValueError(f"status {self.status.value} does not follow from the failed gates ({expected.value})")
        if self.validation_scope == ValidationScope.REAL_WORLD_VALIDATED and not self.real_world_evidence:
            raise ValueError("REAL_WORLD_VALIDATED needs recorded real-world validation evidence")
        if self.status == ReliabilityStatus.HIGH and self.validation_scope not in HIGH_ALLOWED_SCOPES:
            raise ValueError(f"HIGH is not allowed for scope {self.validation_scope.value}: only SIMULATED_ONLY "
                             "(within the simulation) or REAL_WORLD_VALIDATED (with recorded evidence)")
        return self


class IntervalStatus(str, Enum):
    SUPPORTED = "SUPPORTED"                  # evaluation coverage consistent with the nominal level: emitted
    CONSERVATIVE = "CONSERVATIVE"            # evaluation coverage significantly above nominal (too wide): emitted
    UNDER_COVERING = "UNDER_COVERING"        # evaluation coverage significantly below nominal: not emitted
    INSUFFICIENT_CALIBRATION_DATA = "INSUFFICIENT_CALIBRATION_DATA"   # too few calibration periods: not emitted
    UNVALIDATED = "UNVALIDATED"              # calibrated, but too few evaluation periods to check: not emitted
    DEPENDENT_PERIODS = "DEPENDENT_PERIODS"  # period-level errors serially dependent (screen): the coverage check's
    #                                          independence assumption is contradicted: not emitted
    UNSUPPORTED_HORIZON = "UNSUPPORTED_HORIZON"   # beyond the validated horizon: not emitted
    NOT_APPLICABLE = "NOT_APPLICABLE"        # no forecast


EMITTED_INTERVALS = frozenset({IntervalStatus.SUPPORTED, IntervalStatus.CONSERVATIVE})


class UncertaintyAssessment(BaseModel):
    model_config = _FINITE
    status: IntervalStatus
    method: str
    horizon_min: float | None = None
    nominal_coverage: float = Field(ge=0.5, le=0.99)
    inference_unit: str | None = Field(None, description="what one independent observation is for the quantiles, "
                                       "the coverage estimate, its confidence interval and the gates")
    empirical_coverage: float | None = Field(None, description="EVALUATION periods only: mean over periods of each "
                                             "period's share of covered cases (each period weighs 1)")
    coverage_ci: tuple[float, float] | None = Field(None, description="Wilson 95 % interval of empirical_coverage "
                                                     "with n = evaluation_periods; approximate: the binomial "
                                                     "variance bounds the variance of the period shares, but "
                                                     "finite-sample coverage is not guaranteed (docs/RELIABILITY.md "
                                                     "§4); assumes independent periods")
    coverage_error: float | None = Field(None, description="empirical - nominal")
    error_offsets_c: tuple[float, float] | None = Field(None, description="calibrated quantiles of "
                                                         "(observed - forecast) from the CALIBRATION periods, each "
                                                         "period weighing 1")
    forecast_c: float | None = Field(None, description="the Phase 4 forecast at the horizon (reference)")
    interval_lower_c: float | None = None
    interval_upper_c: float | None = None
    interval_width_c: float | None = None
    reference_method: str | None = Field(None, description="naive reference for interval width (persistence)")
    reference_interval_width_c: float | None = Field(
        None, description="width of the persistence interval calibrated the same way on the same calibration "
                          "cases; descriptive (its coverage is not evaluated)")
    calibration_sample_count: int = Field(0, description="calibration cases (descriptive; cases in one period are "
                                          "dependent)")
    calibration_non_overlapping_windows: int = Field(0, description="descriptive only; not used for inference")
    calibration_periods: int = Field(0, description="calibration periods: the inferential sample size")
    evaluation_sample_count: int = Field(0, description="evaluation cases (descriptive)")
    evaluation_non_overlapping_windows: int = Field(0, description="descriptive only; not used for inference")
    evaluation_periods: int = Field(0, description="evaluation periods (disjoint from calibration periods): the "
                                    "inferential sample size")
    supported_horizon_min: float | None = Field(None, description="Phase 4 validated horizon")
    conditional_on: list[str] = Field(default_factory=list)
    includes: list[str] = Field(default_factory=list)
    excludes: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _interval_only_when_supported(self) -> UncertaintyAssessment:
        bounds = (self.interval_lower_c, self.interval_upper_c, self.interval_width_c)
        if self.status in EMITTED_INTERVALS:
            if None in bounds or self.forecast_c is None or self.error_offsets_c is None:
                raise ValueError("an emitted interval needs its bounds, width, offsets and forecast")
            if self.interval_lower_c > self.interval_upper_c:
                raise ValueError("interval bounds out of order")
            # an interval need not contain the forecast itself: with biased errors both offsets can share a sign
            if self.empirical_coverage is None or self.coverage_ci is None or self.evaluation_periods == 0:
                raise ValueError("an emitted interval needs evaluation coverage evidence")
            need = min_conformal_sample(self.nominal_coverage)
            if self.calibration_periods < need or self.evaluation_periods < need:
                raise ValueError(f"an emitted interval at {self.nominal_coverage:.0%} needs {need} calibration and "
                                 f"{need} evaluation periods (has {self.calibration_periods} / "
                                 f"{self.evaluation_periods})")
        elif any(b is not None for b in bounds):
            raise ValueError(f"no interval may be reported with status {self.status.value}")
        return self


class HorizonDiagnostics(BaseModel):
    model_config = _FINITE
    horizon_min: float
    data_role: str = "Phase 4 TEST data (model fitted on FIT+SELECT): out-of-sample"
    n_cases: int = Field(description="held-out cases (descriptive: overlapping cases share their truth)")
    prediction_coverage: float | None = Field(None, description="share of held-out cases the method forecast")
    non_overlapping_windows: int = Field(description="descriptive only: windows in one period are not independent")
    periods: int = Field(description="passive periods with a case: the inferential sample size")
    mae_c: float | None = Field(None, description="over all cases (descriptive)")
    rmse_c: float | None = Field(None, description="over all cases (descriptive)")
    bias_c: float | None = Field(None, description="mean of (observed - forecast) over all cases (descriptive)")
    abs_error_quantiles_c: dict[str, float] = Field(default_factory=dict, description="over all cases (descriptive)")
    signed_error_quantiles_c: dict[str, float] = Field(default_factory=dict,
                                                       description="over all cases (descriptive)")
    period_mean_bias_c: float | None = Field(None, description="mean of the period-mean errors (each period weighs "
                                             "1): the quantity the bias test uses")
    period_lag1_autocorrelation: float | None = Field(None, description="lag-1 autocorrelation of the "
                                                      "chronological period-mean errors (dependence screen)")
    calibration_status: IntervalStatus
    error_offsets_c: tuple[float, float] | None = None
    empirical_coverage: float | None = Field(None, description="period-weighted, EVALUATION periods only")
    interval_width_c: float | None = None


class CalibrationRecord(BaseModel):
    model_config = _FINITE
    design: str
    test_period_start: datetime
    boundary: datetime
    calibration_origins: tuple[datetime, datetime] | None = None
    calibration_latest_truth: datetime | None = None
    evaluation_origins: tuple[datetime, datetime] | None = None
    purge_rule: str
    independence: str
    cross_room: str


class CrossingUncertaintyStatus(str, Enum):
    NO_THRESHOLD = "NO_THRESHOLD"
    ALREADY_CROSSED = "ALREADY_CROSSED"
    NOT_REACHABLE = "NOT_REACHABLE"
    NOT_WITHIN_HORIZON = "NOT_WITHIN_HORIZON"
    CALIBRATED_INTERVAL = "CALIBRATED_INTERVAL"
    INSUFFICIENT_CROSSING_DATA = "INSUFFICIENT_CROSSING_DATA"
    UNBOUNDED = "UNBOUNDED"                  # missed or false crossings in calibration exceed the tail level
    UNVALIDATED = "UNVALIDATED"
    DEPENDENT_PERIODS = "DEPENDENT_PERIODS"  # period-level errors at this horizon serially dependent (screen)
    UNDER_COVERING = "UNDER_COVERING"
    UNSUPPORTED_HORIZON = "UNSUPPORTED_HORIZON"
    NOT_APPLICABLE = "NOT_APPLICABLE"        # no forecast


class CrossingUncertainty(BaseModel):
    model_config = _FINITE
    status: CrossingUncertaintyStatus
    deterministic_status: CrossingStatus
    deterministic_minutes: float | None = None
    method: str
    nominal_coverage: float | None = None
    lower_min: float | None = None
    upper_min: float | None = None
    extends_beyond_horizon: bool | None = None
    calibration_events: int = Field(0, description="calibration periods with a scored crossing case: the unit")
    evaluation_events: int = Field(0, description="evaluation periods (disjoint from calibration) with a scored "
                                   "crossing case")
    empirical_coverage: float | None = Field(None, description="mean over evaluation periods of each period's "
                                             "covered share")
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _interval_only_when_calibrated(self) -> CrossingUncertainty:
        has = self.lower_min is not None or self.upper_min is not None
        if self.status == CrossingUncertaintyStatus.CALIBRATED_INTERVAL:
            if self.lower_min is None or self.upper_min is None or self.lower_min > self.upper_min:
                raise ValueError("a calibrated crossing interval needs ordered bounds")
            if self.deterministic_status != CrossingStatus.CROSSING_PREDICTED or self.deterministic_minutes is None:
                raise ValueError("a crossing-time interval exists only around a predicted crossing")
        elif has:
            raise ValueError(f"no crossing-time interval may be reported with status {self.status.value}")
        return self


class ParameterUncertainty(BaseModel):
    model_config = _FINITE
    status: str = Field(description="ASSESSED, NOT_APPLICABLE or UNAVAILABLE")
    method: str
    meaning: str
    horizon_min: float | None = None
    envelope_lower_c: float | None = None
    envelope_upper_c: float | None = None
    envelope_width_c: float | None = None
    parameters: dict[str, tuple[float, float]] = Field(default_factory=dict)
    note: str = "a parameter envelope is NOT a prediction interval: it excludes measurement noise, model " \
                "inadequacy and future-input uncertainty"


class UncertaintySource(BaseModel):
    model_config = _FINITE
    source: str
    category: str = Field(description="MEASUREMENT, PARAMETER, MODEL_INADEQUACY, REGIME, EXTRAPOLATION, "
                                      "FUTURE_INPUT or GENERALIZATION")
    treatment: str = Field(description="QUANTIFIED, QUALITATIVE or NOT_ASSESSED")
    value: float | None = None
    unit: str | None = None
    in_empirical_interval: bool | None = Field(None, description="whether the empirical interval reflects it")
    note: str


class FutureInput(BaseModel):
    name: str
    status: str = Field(description="OBSERVED, DERIVED, ASSUMED, SCENARIO or UNKNOWN")
    note: str


class EvidenceSummary(BaseModel):
    data_quality: str
    model_quality: str
    identifiability: str
    regime_applicability: str
    horizon_applicability: str
    extrapolation_status: str
    future_input_assumptions: list[FutureInput] = Field(default_factory=list)


class CausalView(BaseModel):
    origin: datetime | None
    truncated: bool
    rows_in_source: int
    rows_excluded_after_origin: int
    causal_input_sha256: str | None
    note: str


class EvidenceRecommendation(Recommendation):
    retry_conditions: str


class ReliabilityConfig(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")
    interval_level: float = Field(0.80, ge=0.5, le=0.99, description="nominal level of the empirical interval; an "
                                  "evaluation target, never a guaranteed coverage")


class ReliabilityReport(BaseModel):
    model_config = _FINITE
    phase: str = "5"
    method_version: str
    forecast_reference: str
    forecast_available: bool
    config: ReliabilityConfig
    causal_view: CausalView
    reliability: ReliabilityAssessment
    uncertainty: UncertaintyAssessment
    horizon_diagnostics: list[HorizonDiagnostics] = Field(default_factory=list)
    calibration: CalibrationRecord | None = None
    crossing: CrossingUncertainty
    parameter_uncertainty: ParameterUncertainty
    uncertainty_sources: list[UncertaintySource] = Field(default_factory=list)
    evidence: EvidenceSummary
    limitations: list[str] = Field(default_factory=list)
    recommendations: list[EvidenceRecommendation] = Field(default_factory=list)
    refusal_or_downgrade_reason: str | None = None

    @model_validator(mode="after")
    def _availability_consistent(self) -> ReliabilityReport:
        if not self.forecast_available:
            if self.reliability.status != ReliabilityStatus.UNSUPPORTED:
                raise ValueError("without a forecast the reliability is UNSUPPORTED")
            if self.uncertainty.status != IntervalStatus.NOT_APPLICABLE:
                raise ValueError("without a forecast there is no uncertainty assessment")
            if self.crossing.status != CrossingUncertaintyStatus.NOT_APPLICABLE:
                raise ValueError("without a forecast there is no crossing uncertainty")
            if self.parameter_uncertainty.status != "NOT_APPLICABLE" or self.horizon_diagnostics \
                    or self.calibration is not None:
                raise ValueError("without a forecast there are no parameter envelopes or interval diagnostics")
        elif self.uncertainty.status == IntervalStatus.NOT_APPLICABLE:
            raise ValueError("a forecast always has an uncertainty status")
        if self.reliability.status == ReliabilityStatus.HIGH and self.uncertainty.status not in EMITTED_INTERVALS:
            raise ValueError("HIGH needs an emitted, evaluated interval")
        if self.reliability.status != ReliabilityStatus.HIGH and not self.refusal_or_downgrade_reason:
            raise ValueError("a status below HIGH states why")
        return self


# ---------------------------------------------------------------- result
class ForecastResult(BaseModel):
    status: ForecastStatus
    evidence: EvidenceClass = Field(description="SIMULATED for simulated input, otherwise EXPECTED: never REAL")
    provenance: dict[str, str | float | int | None]
    prediction_start: datetime | None = None
    prediction_horizon_min: float | None = None
    validated_horizon_min: float | None = Field(
        None, description="longest horizon at which the methods were compared on held-out data")
    initial_temperature: Quantity | None = None
    initial_measurement_time: datetime | None = None
    threshold: Threshold | None = None
    crossing_status: CrossingStatus = CrossingStatus.NO_THRESHOLD
    predicted_crossing_time: datetime | None = None
    predicted_crossing_minutes: float | None = None
    trajectory: list[TrajectoryPoint] = Field(default_factory=list)
    selected_model: str | None = None
    selected_model_version: str | None = None
    solution_method: str | None = None
    operating_regime: Regime = Regime.UNKNOWN
    regime_evidence: list[str] = Field(default_factory=list)
    scenario: list[str] = Field(default_factory=list, description="what is ASSUMED about the horizon")
    input_inventory: list[InputItem] = Field(default_factory=list)
    candidate_models: list[CandidateResult] = Field(default_factory=list)
    rejected_models: list[RejectedModel] = Field(default_factory=list)
    parameter_estimates: list[ParameterEstimate] = Field(default_factory=list)
    parameter_units: dict[str, str] = Field(default_factory=dict)
    segments_used: list[SegmentRecord] = Field(default_factory=list)
    segments_rejected: list[SegmentRecord] = Field(default_factory=list)
    observations_used: int = 0
    observations_excluded: list[str] = Field(default_factory=list)
    fit_diagnostics: FitDiagnostics | None = None
    validation_diagnostics: ValidationDiagnostics | None = None
    baseline_results: list[BaselineForecast] = Field(default_factory=list)
    model_comparison: list[Comparison] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    refusal_code: RefusalCode | None = None
    refusal_reason: str | None = None
    refusal_failed_condition: str | None = None
    refusal_needed_to_retry: str | None = None
    # ---- follow-up fields (additive; every field above keeps its meaning)
    question: ForecastQuestion = ForecastQuestion.NO_COOLING
    input_capabilities: InputCapabilities | None = None
    model_selection: ModelSelection | None = None
    limitation_details: LimitationDetails | None = None
    recommendations: list[Recommendation] = Field(default_factory=list)
    # ---- Phase 5 (absent from the output unless a reliability assessment was requested)
    reliability_report: ReliabilityReport | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_report(self, handler):
        data = handler(self)
        if self.reliability_report is None and isinstance(data, dict):
            data.pop("reliability_report", None)          # Phase 4 output stays byte-identical
        return data

    @model_validator(mode="after")
    def _forecast_xor_refusal(self) -> ForecastResult:
        if self.status == ForecastStatus.REFUSED:
            if self.refusal_code is None or not self.refusal_reason:
                raise ValueError("a refusal needs a code and a reason")
            if (self.trajectory or self.predicted_crossing_time is not None
                    or self.predicted_crossing_minutes is not None):
                raise ValueError("a refusal carries no trajectory and no crossing time")
            if self.crossing_status in FORECAST_CROSSINGS:
                raise ValueError(f"a refusal carries no forecast crossing status ({self.crossing_status.value})")
            if self.model_selection is not None and (self.model_selection.selected_model is not None
                                                     or self.model_selection.forecast_kind != "NONE"):
                raise ValueError("a refusal selects no model")
            if (self.parameter_estimates or self.parameter_units or self.selected_model is not None
                    or self.selected_model_version is not None or self.solution_method is not None):
                raise ValueError("a refusal carries no parameter estimates and no selected model")
            if self.fit_diagnostics is not None and not supported_fit(self.fit_diagnostics):
                raise ValueError("a refusal carries no unsupported final fit")
            if self.scenario:
                raise ValueError("a refusal states no forecast scenario")
            if any(c.outcome == CandidateOutcome.SELECTED for c in self.candidate_models):
                raise ValueError("a refusal has no SELECTED candidate")
            if any(getattr(b, f) is not None for b in self.baseline_results for f in BASELINE_PREDICTION_FIELDS):
                raise ValueError("a refusal carries no baseline predictions (horizon temperature, slope, crossing)")
        else:
            if self.refusal_code is not None or not self.trajectory or self.selected_model is None:
                raise ValueError("a forecast needs a trajectory and a selected model, and no refusal code")
            if self.model_selection is not None and self.model_selection.selected_model != self.selected_model:
                raise ValueError("model_selection must name the selected model")
        if any(c.fit is not None and c.outcome != CandidateOutcome.REJECTED and not supported_fit(c.fit)
               for c in self.candidate_models):
            raise ValueError("only a REJECTED candidate may carry an unsupported fit")
        if (self.predicted_crossing_time is not None) != (self.crossing_status == CrossingStatus.CROSSING_PREDICTED):
            raise ValueError("a crossing time is given exactly when a crossing is predicted")
        rep = self.reliability_report
        if rep is not None:
            if rep.forecast_available != (self.status == ForecastStatus.FORECAST):
                raise ValueError("reliability_report.forecast_available must match the forecast status")
            raw = self.provenance.get("data_evidence_class")      # absent: provenance not established
            data_class = EvidenceClass(raw) if raw in {e.value for e in EvidenceClass} else EvidenceClass.UNVALIDATED
            scope = rep.reliability.validation_scope
            if scope != SCOPE_FOR_EVIDENCE[data_class] and not (
                    scope == ValidationScope.REAL_WORLD_VALIDATED and data_class == EvidenceClass.REAL):
                raise ValueError(f"validation scope {scope.value} does not follow from the input evidence class "
                                 f"{data_class.value} (expected {SCOPE_FOR_EVIDENCE[data_class].value})")
            if rep.crossing.deterministic_status != self.crossing_status:
                raise ValueError("the crossing uncertainty must refer to this result's crossing status")
            if rep.crossing.deterministic_minutes != self.predicted_crossing_minutes:
                raise ValueError("the crossing uncertainty must refer to this result's crossing time")
            u = rep.uncertainty
            if u.status in EMITTED_INTERVALS and (
                    u.horizon_min != self.prediction_horizon_min or not self.trajectory
                    or abs(u.forecast_c - self.trajectory[-1].temp_c) > 1e-9):
                raise ValueError("an interval must surround this result's forecast at its horizon")
        return self
