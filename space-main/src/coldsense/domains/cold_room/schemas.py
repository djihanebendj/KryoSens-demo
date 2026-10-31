"""Cold-room result types.

Every number ColdSense shows must carry an evidence class, and every
prediction must be able to say "no" with reasons. These types enforce that.
Domain-agnostic types live in ``coldsense.core`` and are re-exported here so
existing imports keep working.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from coldsense.core.evidence import EvidenceClass
from coldsense.core.results import Issue, Severity, Verdict

from . import thresholds as C

__all__ = [
    "EvidenceClass", "Issue", "Severity", "Verdict",
    "ROW_FLAGS", "ValidationLimits", "Transformation", "MalformedRecord", "InputIdentity",
    "RecordCounts", "IntervalStats",
    "DataQualityReport", "WarmingSegment", "Prediction",
    "EventType", "Event", "SegmentationParams", "SegmentationResult",
]

TemperatureUnit = Literal["C", "F", "K"]

# Every per-row flag, whether it excludes the row from analysis input, and why.
# Excluded rows are KEPT in the observations table; they are only left out of
# the frame handed to later analysis.
ROW_FLAGS: dict[str, tuple[bool, str]] = {
    "TS_MISSING": (True, "timestamp cell is empty"),
    "TS_INVALID": (True, "timestamp does not match the file's timestamp format"),
    "TS_LOCAL_AMBIGUOUS": (True, "local time does not exist or occurs twice in the given time zone"),
    "TEMP_MISSING": (True, "temperature cell is empty or a missing-value marker"),
    "TEMP_NON_NUMERIC": (True, "temperature is not a number"),
    "TEMP_INFINITE": (True, "temperature is infinite"),
    "DUPLICATE_EXACT": (True, "repeat of an earlier row: same timestamp and identical telemetry"),
    "DUPLICATE_CONFLICT": (True, "same timestamp as an earlier row but different telemetry (temperature, "
                                   "ambient or compressor state); earlier row used, see duplicate_differs"),
    "TEMP_OUT_OF_RANGE": (True, "outside the physically plausible range for room air"),
    "TEMP_IMPLAUSIBLE_SPIKE": (True, "isolated jump faster than room air can change, up and back"),
    "DUPLICATE_CONFLICT_KEPT": (False, "first of several rows sharing a timestamp with different values"),
    "OUT_OF_ORDER": (False, "earlier in time than a previous row in the file"),
    "TEMP_SUSPICIOUS": (False, "unusual compared with neighbouring readings, but physically possible"),
    "AMBIENT_NON_NUMERIC": (False, "ambient temperature is not a number (set to missing)"),
    "AMBIENT_INFINITE": (False, "ambient temperature is infinite (set to missing)"),
    "AMBIENT_OUT_OF_RANGE": (False, "ambient temperature outside the plausible range"),
    "COMPRESSOR_INVALID": (False, "compressor state is not a recognised on/off value (set to missing)"),
}


class ValidationLimits(BaseModel):
    """Measurement assumptions a caller may override for one file (e.g. a
    freezer). Sufficiency limits are not here: they stay fixed in code."""

    model_config = ConfigDict(frozen=True)

    plausible_min_c: float = C.PLAUSIBLE_MIN_C
    plausible_max_c: float = C.PLAUSIBLE_MAX_C
    max_step_c_per_min: float = C.MAX_STEP_C_PER_MIN
    spike_min_jump_c: float = C.SPIKE_MIN_JUMP_C
    ambient_min_c: float = C.AMBIENT_PLAUSIBLE_MIN_C
    ambient_max_c: float = C.AMBIENT_PLAUSIBLE_MAX_C


class Transformation(BaseModel):
    """Anything that changed or reinterpreted a value or the row order."""

    code: str
    description: str
    rows_affected: int


class MalformedRecord(BaseModel):
    """A row with more fields than the header. It cannot be assigned to columns,
    so it is NOT an observation; its exact source text is kept here instead."""

    line: int
    n_fields: int
    expected_fields: int
    raw_text: str


class InputIdentity(BaseModel):
    source: str
    sha256: str = Field(description="SHA-256 of the bytes read (of the UTF-8 form when given text)")
    size_bytes: int
    encoding: str
    delimiter: str
    decimal_separator: Literal[".", ". or ,"]
    header: list[str]
    column_mapping: dict[str, str] = Field(description="role -> header in the file")
    ignored_columns: list[str]
    temperature_unit: TemperatureUnit
    temperature_unit_source: Literal["option", "header", "cells", "assumed"]
    timestamp_format: str
    timestamp_format_source: Literal["option", "iso8601", "detected"]
    timezone_handling: str


class RecordCounts(BaseModel):
    data_rows_read: int = Field(description="rows that became observations (excludes malformed rows)")
    rows_malformed: int = Field(description="rows with more fields than the header; NOT observations, "
                                            "kept as raw text in DataQualityReport.malformed_records")
    malformed_lines: list[int] = Field(description="file line numbers of all malformed rows")
    rows_short: int = Field(description="rows with fewer fields than the header; missing fields left empty")
    blank_lines: int
    rows_usable: int
    rows_excluded: int
    rows_suspicious: int


class IntervalStats(BaseModel):
    n_intervals: int
    min_s: float | None
    p05_s: float | None
    median_s: float | None
    p95_s: float | None
    max_s: float | None
    n_irregular: int
    irregular_pct: float
    n_gaps: int
    longest_gap_min: float
    n_out_of_order: int


class DataQualityReport(BaseModel):
    source: str
    evidence: EvidenceClass
    n_rows_raw: int
    n_rows_clean: int
    start: datetime | None
    end: datetime | None
    duration_hours: float
    median_interval_s: float | None
    coverage_pct: float = Field(description="share of expected samples present")
    n_gaps: int
    longest_gap_min: float
    n_implausible: int
    n_duplicates: int
    temp_min_c: float | None
    temp_max_c: float | None
    ambient_available: bool
    issues: list[Issue]
    verdict: Verdict
    # --- Phase 2 additions (defaults keep older constructors valid) ---
    suitable_for_downstream: bool = Field(
        default=False, description="False whenever the verdict is REFUSED. True does NOT mean the data is "
                                   "correct; see limitations")
    input: InputIdentity | None = None
    counts: RecordCounts | None = None
    missing_by_field: dict[str, int] = Field(default_factory=dict)
    invalid_by_field: dict[str, int] = Field(default_factory=dict)
    flag_counts: dict[str, int] = Field(default_factory=dict)
    n_duplicates_exact: int = 0
    n_duplicates_conflicting: int = 0
    duplicate_conflict_fields: dict[str, int] = Field(
        default_factory=dict, description="per telemetry field, how many excluded duplicates differ in it")
    intervals: IntervalStats | None = None
    malformed_records: list[MalformedRecord] = Field(default_factory=list)
    transformations: list[Transformation] = Field(default_factory=list)
    limits_used: ValidationLimits | None = None
    limitations: list[str] = Field(default_factory=list)

    @property
    def blocking(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == Severity.BLOCKING]


class WarmingSegment(BaseModel):
    """A run where the air temperature rises steadily: the candidate
    compressor-off period. Inferred from temperature only, so it is a
    hypothesis about compressor state, not a measurement of it."""

    start: datetime
    end: datetime
    duration_min: float
    t_start_c: float
    t_end_c: float
    rate_c_per_min: float
    r2_linear: float
    clean: bool
    reject_reason: str | None = None


class Prediction(BaseModel):
    """A time-to-threshold answer, or a refusal. Never both."""

    method: str
    evidence: EvidenceClass
    refused: bool
    reasons: list[str] = Field(default_factory=list)
    current_temp_c: float | None = None
    threshold_c: float | None = None
    minutes_to_threshold: float | None = None
    low_minutes: float | None = None
    high_minutes: float | None = None
    n_segments_used: int = 0
    observed_range_c: tuple[float, float] | None = None
    extrapolation_c: float | None = Field(
        default=None,
        description="how far the threshold lies beyond the warmest temperature seen in any clean segment",
    )
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ============================================================== Phase 3: events
class EventType(str, Enum):
    """What an event says about the DATA. None of these is a diagnosis."""

    DATA_GAP = "DATA_GAP"                        # no usable observation for longer than the gap rule allows
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"      # too few observations to classify: refused
    TEMP_STABLE = "TEMP_STABLE"                  # net change within the stable band
    TEMP_RISING = "TEMP_RISING"                  # sustained, reasonably steady increase
    TEMP_FALLING = "TEMP_FALLING"                # sustained, reasonably steady decrease
    TEMP_IRREGULAR = "TEMP_IRREGULAR"            # changes too briefly or too unevenly to call a trend: refused
    THRESHOLD_EXCURSION = "THRESHOLD_EXCURSION"  # beyond a user-set limit for at least the minimum duration
    BRIEF_EXCURSION = "BRIEF_EXCURSION"          # beyond the limit, but shorter than the minimum duration
    COMPRESSOR_ON = "COMPRESSOR_ON"              # recorded state ON (from the data, never inferred)
    COMPRESSOR_OFF = "COMPRESSOR_OFF"
    COMPRESSOR_UNKNOWN = "COMPRESSOR_UNKNOWN"    # state missing or invalid in these rows
    COMPRESSOR_TRANSITION = "COMPRESSOR_TRANSITION"  # state changed between two consecutive observations
    COMPRESSOR_CYCLE = "COMPRESSOR_CYCLE"        # a complete ON period followed by a complete OFF period


class SegmentationParams(BaseModel):
    """Every number segmentation uses. Defaults are PROVISIONAL (see thresholds.py).

    Checked here, without data (docs/EVENTS.md, "Parameter validation"): every
    float is finite (NaN and ±inf are rejected, including where a bound would
    let +inf through), the domain bounds below, and three consistency rules.
    Checks that need the log (gap rule, durations vs the data) are made by
    ``segment_events``, which refuses instead of producing misleading events.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    gap_factor: float = Field(C.EVT_GAP_FACTOR, gt=1)
    smooth_window_min: float = Field(C.EVT_SMOOTH_WINDOW_MIN, gt=0)
    stable_band_c: float = Field(C.EVT_STABLE_BAND_C, gt=0)
    piece_tolerance_c: float = Field(C.EVT_PIECE_TOLERANCE_C, gt=0)
    min_trend_min: float = Field(C.EVT_MIN_TREND_MIN, ge=0)
    max_trend_resid_c: float = Field(C.EVT_MAX_TREND_RESID_C, gt=0)
    min_points: int = Field(C.EVT_MIN_POINTS, ge=2)
    upper_threshold_c: float | None = Field(None, allow_inf_nan=False,
                                            description="no default: depends on the product; finite if set")
    lower_threshold_c: float | None = Field(None, allow_inf_nan=False,
                                            description="no default: depends on the product; finite if set")
    min_excursion_min: float = Field(C.EVT_MIN_EXCURSION_MIN, ge=0)
    compressor_min_valid: float = Field(C.EVT_COMPRESSOR_MIN_VALID, ge=0, le=1)

    @model_validator(mode="after")
    def _consistent(self) -> SegmentationParams:
        problems = []
        lo, hi = self.lower_threshold_c, self.upper_threshold_c
        if lo is not None and hi is not None and not lo < hi:
            problems.append(f"lower_threshold_c ({lo}) must be below upper_threshold_c ({hi}); "
                            "with these limits a reading could be both above and below")
        if self.piece_tolerance_c > self.stable_band_c:
            problems.append(f"piece_tolerance_c ({self.piece_tolerance_c}) must not exceed stable_band_c "
                            f"({self.stable_band_c}): the splitter would hide changes the classifier reports")
        if self.max_trend_resid_c > self.stable_band_c:
            problems.append(f"max_trend_resid_c ({self.max_trend_resid_c}) must not exceed stable_band_c "
                            f"({self.stable_band_c}): a rise or fall could stray from its line by more than "
                            "a reportable change and still be called steady")
        if self.smooth_window_min * 60e9 < 1:          # (a product, so huge windows cannot overflow round())
            problems.append(f"smooth_window_min ({self.smooth_window_min}) is shorter than one nanosecond, "
                            "the resolution of timestamps")
        if problems:
            raise ValueError("; ".join(problems))
        return self


class Event(BaseModel):
    """One observed period or change, traceable to the rows that support it.

    ``observed`` holds values read directly from supporting rows; ``derived``
    holds computed features (fits, durations, ratios). Neither is a cause.
    """

    event_id: str
    type: EventType
    start: datetime = Field(description="first supporting observation (time basis: see SegmentationResult)")
    end: datetime = Field(description="last supporting observation")
    duration_min: float | None = Field(description="end - start in minutes; None for a single observation")
    duration_is_lower_bound: bool = Field(description="True when start or end is censored")
    start_censored: bool = Field(description="the event may have begun before the data shows (log start or gap)")
    end_censored: bool = Field(description="the event may continue after the data shows (log end or gap)")
    before: datetime | None = Field(None, description="last observation just before the event, if adjacent")
    after: datetime | None = Field(None, description="first observation just after the event, if adjacent")
    n_observations: int
    source_lines: list[int] = Field(description="Phase 2 source_line of every supporting observation")
    observed: dict[str, float | str | bool | None] = Field(default_factory=dict)
    derived: dict[str, float | str | None] = Field(default_factory=dict)
    rationale: str
    limitations: list[str] = Field(default_factory=list)
    evidence: EvidenceClass = Field(description="evidence class of the underlying data; the classification itself "
                                                "is an interpretation of it")


class SegmentationResult(BaseModel):
    refused: bool
    reasons: list[str] = Field(default_factory=list)
    params: SegmentationParams
    evidence: EvidenceClass
    time_basis: str = Field(description="Phase 2 time handling: UTC-normalised or local time of unknown zone")
    events: list[Event] = Field(default_factory=list)
    summary: dict[str, int | float | str | bool] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)

    def of_type(self, *types: EventType) -> list[Event]:
        return [e for e in self.events if e.type in types]
