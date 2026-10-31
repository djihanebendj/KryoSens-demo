"""Phase 6: observed facts, cautious diagnostic hypotheses and ranked next-best checks.

Deterministic and rule-based. It reads what earlier phases already established and adds no number:

* Phase 2: the data-quality report (verdict, issues) and the per-row flags;
* Phase 3: the events, each with its supporting source lines (excursions; data gaps; temperature behaviour over
  each excursion window, shown as context);
* the optional operational event log and room metadata (equipment and sensor information; Phase 4 adapters),
  with their provenance;
* optionally a Phase 4 forecast (model output, held-out comparison with the baselines) with its Phase 5
  reliability report: CONTEXT only. Forecast numbers are never used as evidence for an explanation, and Phase 4's
  missing-input advice is referenced, never restated.

Rules (docs/DIAGNOSTICS.md):

* an event is an observation, never a diagnosis. Only SUSTAINED excursions beyond a limit the user set are
  diagnosed; without a limit nothing is called abnormal (INSUFFICIENT_EVIDENCE);
* each candidate explanation of an excursion is assessed only against RECORDED evidence, with exact rules: the
  recorded compressor state in every reading of the excursion (all OFF, all ON, otherwise neither), supplied
  records overlapping the excursion, readings flagged by validation. No probability, score, confidence,
  measurement or predicted benefit is produced, and absence of a record is never evidence of absence;
* hypotheses use ``core.governance.Hypothesis`` and stay PROPOSED: no rule here confirms or rejects one;
* when no recorded evidence bears on any explanation, the finding (and, if all are like that, the result) is
  INSUFFICIENT_EVIDENCE: missing evidence lowers specificity, it never triggers a guess;
* every explanation that becomes a hypothesis names a required next check from the ranked list; one decided by a
  single recorded source (or neutral by a record) needs that source corroborated;
* next-best checks look back at the excursion windows (records, run history, a reference reading, an inspection),
  each with its evidence, the information it would give, a qualitative evidence value and the access it needs.
  They are ranked by an explicit rule: the number of unresolved items a check addresses, then effort, then
  evidence value, then id: a work priority, never a confidence in an explanation.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from coldsense.core.evidence import EvidenceClass
from coldsense.core.governance import Hypothesis, HypothesisStatus

from .events import segment_events
from .pipeline import validate_source
from .schemas import DataQualityReport, Event, EventType, SegmentationParams, SegmentationResult, Verdict

METHOD_VERSION = "phase6-diagnostics-2"
GENERATED_BY = "rules:" + METHOD_VERSION
RULES_VALIDATION = ("UNVALIDATED: the rules were checked on SIMULATED logs only; no real fault outcome has been "
                    "compared with them, and every explanation is a hypothesis, not a diagnosis")


# ====================================================================== result types
class DiagnosisStatus(str, Enum):
    HYPOTHESES = "HYPOTHESES"                        # recorded evidence bears on at least one explanation
    NO_FINDING = "NO_FINDING"                        # a limit was given; no excursion beyond it in the usable data
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"  # refused, no limit, or no recorded evidence for any explanation


class FactSource(str, Enum):
    PHASE2_VALIDATION = "PHASE2_VALIDATION"
    PHASE3_EVENT = "PHASE3_EVENT"
    EVENT_LOG = "EVENT_LOG"                          # a record the user supplied (with its own provenance)
    PHASE4_FORECAST = "PHASE4_FORECAST"
    PHASE5_RELIABILITY = "PHASE5_RELIABILITY"
    ROOM_METADATA = "ROOM_METADATA"                  # equipment and sensor information the user supplied


class ObservedFact(BaseModel):
    """What a record says, with where it comes from. Never an interpretation of cause."""
    fact_id: str
    source: FactSource
    kind: str = Field(description="event type, issue code, record kind or status")
    start: datetime | None = None
    end: datetime | None = None
    source_lines: list[int] = Field(default_factory=list, description="Phase 2 source lines of the readings")
    source_ids: list[str] = Field(default_factory=list, description="Phase 3 event ids or record references")
    statement: str
    limitations: list[str] = Field(default_factory=list)
    evidence: EvidenceClass = Field(description="evidence class of the underlying data or output")
    record_provenance: str | None = Field(None, description="event-log records: RECORDED_BY_SYSTEM or MANUAL_LOG; "
                                          "room metadata: its stated provenance (e.g. MANUFACTURER_SPECIFICATION)")


class Explanation(str, Enum):
    NO_ACTIVE_COOLING = "NO_ACTIVE_COOLING"
    CAPACITY_SHORTFALL_WHILE_RUNNING = "CAPACITY_SHORTFALL_WHILE_RUNNING"
    COOLING_RAN_THROUGHOUT = "COOLING_RAN_THROUGHOUT"
    HEAT_INGRESS = "HEAT_INGRESS"
    PRODUCT_LOADING = "PRODUCT_LOADING"
    DEFROST = "DEFROST"
    CONTROL_CHANGE = "CONTROL_CHANGE"
    RECORDED_INTERVENTION = "RECORDED_INTERVENTION"
    SENSOR_NOT_REPRESENTATIVE = "SENSOR_NOT_REPRESENTATIVE"


class ExplanationAssessment(BaseModel):
    code: Explanation
    statement: str
    supporting: list[str] = Field(default_factory=list, description="recorded evidence consistent with it")
    contradicting: list[str] = Field(default_factory=list, description="recorded evidence against it")
    neutral: list[str] = Field(default_factory=list, description="assessed, but neither for nor against")
    missing: list[str] = Field(default_factory=list, description="evidence that would bear on it, not available")
    missing_records: list[str] = Field(default_factory=list, description="event-log kinds (input catalogue names) "
                                       "with no usable record for the window")
    next_check_ids: list[str] = Field(default_factory=list, description="checks that would bear on it")
    required_next_check: str | None = Field(None, description="the check this explanation needs next (the first of "
                                            "next_check_ids); always set, and ranked, when it is a hypothesis")


class FindingStatus(str, Enum):
    EVIDENCE_AVAILABLE = "EVIDENCE_AVAILABLE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"    # some explanation has both supporting and contradicting records
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"  # no recorded evidence bears on any explanation


class Finding(BaseModel):
    finding_id: str
    event_id: str
    fact_id: str
    side: Literal["above", "below"]
    limit_c: float
    window_start: datetime = Field(description="reading before the excursion (or its first reading if censored)")
    window_end: datetime
    statement: str
    status: FindingStatus
    explanations: list[ExplanationAssessment]
    hypothesis_ids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    context_fact_ids: list[str] = Field(default_factory=list, description="temperature behaviour and data gaps over "
                                        "the window: observations shown with the finding, never used as evidence")


class Effort(str, Enum):
    EXISTING_RECORDS = "EXISTING_RECORDS"            # look up what already exists
    SPOT_CHECK = "SPOT_CHECK"                        # one visit, a reference reading
    SITE_INSPECTION = "SITE_INSPECTION"              # a technician inspects the equipment


EFFORT_ORDER = (Effort.EXISTING_RECORDS, Effort.SPOT_CHECK, Effort.SITE_INSPECTION)


class EvidenceValue(str, Enum):
    """What the check's result can do to the assessment, under the rules of this module. Qualitative and defined
    by the rules; it is not an information gain (no probabilities exist to compute one)."""
    RESOLVES_CONFLICT = "RESOLVES_CONFLICT"          # decides between records that disagree
    DISCRIMINATES = "DISCRIMINATES"                  # one result supports one explanation and contradicts another
    SUPPORTS_OR_CONTRADICTS = "SUPPORTS_OR_CONTRADICTS"  # can support or contradict the explanations it addresses
    CORROBORATES = "CORROBORATES"                    # confirms or questions evidence already assessed
    SUPPORTS_ONLY = "SUPPORTS_ONLY"                  # can support; finding nothing is not evidence against
    BOUNDS_OBSERVATION = "BOUNDS_OBSERVATION"        # completes an observation (duration, peak), no explanation


EVIDENCE_VALUE_ORDER = tuple(EvidenceValue)


class Safety(str, Enum):
    """Access the check needs: a statement about the check, not about the room."""
    OFFICE_ONLY = "OFFICE_ONLY"                      # records and files; no site access
    ROOM_ENTRY = "ROOM_ENTRY"                        # entering the cold room
    EQUIPMENT_ACCESS = "EQUIPMENT_ACCESS"            # opening or working on the refrigeration equipment


class NextCheck(BaseModel):
    rank: int
    check_id: str
    action: str
    windows: list[str] = Field(description="the excursion windows to look at")
    required_input: str
    units: str | None = None
    resolution_or_duration: str | None = None
    why_useful: str
    resolves: list[str] = Field(description="finding / explanation pairs (or observation uncertainties) addressed")
    open_items: int = Field(description="len(resolves): the first ranking key; a count, not a score")
    effort: Effort
    phase4_recommendations: list[str] = Field(default_factory=list, description="Phase 4 advice on acquiring the "
                                              "same input; referenced, not restated")
    ranking_basis: str
    evidence: list[str] = Field(default_factory=list, description="EVIDENCE: the facts and assessment items that "
                                "make the check necessary, by fact id")
    expected_information: list[str] = Field(default_factory=list, description="EXPECTED INFORMATION: what the "
                                            "result would be and how it would bear on each explanation, by the rules")
    evidence_value: EvidenceValue
    safety: Safety
    safety_note: str


class DiagnosticReport(BaseModel):
    phase: str = "6"
    method_version: str = METHOD_VERSION
    status: DiagnosisStatus
    refusal_reason: str | None = None
    data_evidence: EvidenceClass = Field(description="evidence class of the analysed log (Phase 2)")
    rules_validation: str = RULES_VALIDATION
    time_basis: str
    as_of: datetime | None = Field(None, description="end of the analysed data; also the hypotheses' created_at, "
                                   "so identical inputs give identical output")
    observed_facts: list[ObservedFact] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    next_checks: list[NextCheck] = Field(default_factory=list)
    next_best_check: str | None = Field(None, description="NEXT BEST CHECK: the check ranked first (a work priority, "
                                        "not a confidence in any explanation)")
    data_quality_limitations: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> DiagnosticReport:
        if self.status == DiagnosisStatus.INSUFFICIENT_EVIDENCE:
            if not self.refusal_reason or self.hypotheses:
                raise ValueError("INSUFFICIENT_EVIDENCE states a reason and proposes no hypothesis")
        elif self.status == DiagnosisStatus.HYPOTHESES and not self.hypotheses:
            raise ValueError("HYPOTHESES needs at least one hypothesis")
        elif self.status == DiagnosisStatus.NO_FINDING and (self.findings or self.hypotheses):
            raise ValueError("NO_FINDING has no finding and no hypothesis")
        facts = {f.fact_id for f in self.observed_facts}
        for h in self.hypotheses:
            if h.status != HypothesisStatus.PROPOSED:
                raise ValueError("Phase 6 only proposes hypotheses; it never confirms or rejects one")
            if not h.test or not h.success_criteria or h.motivating_observation.split(" ", 1)[0] not in facts:
                raise ValueError(f"hypothesis {h.id} needs a test, a success criterion and a motivating fact")
        for f in self.findings:
            if f.status == FindingStatus.INSUFFICIENT_EVIDENCE and f.hypothesis_ids:
                raise ValueError(f"finding {f.finding_id} has no recorded evidence: no hypothesis may be proposed")
        if [c.rank for c in self.next_checks] != list(range(1, len(self.next_checks) + 1)):
            raise ValueError("next checks are ranked 1..n")
        if self.next_best_check != (self.next_checks[0].check_id if self.next_checks else None):
            raise ValueError("the next best check is the check ranked first")
        for c in self.next_checks:
            if not c.evidence or not c.expected_information or c.open_items != len(c.resolves):
                raise ValueError(f"check {c.check_id} states its evidence, its expected information and what it "
                                 "resolves")
        ranked = {c.check_id for c in self.next_checks}
        tests = {h.id: h.test for h in self.hypotheses}
        for f in self.findings:
            for hid in f.hypothesis_ids:
                code = hid.removeprefix(f"H-{f.finding_id}-")
                a = next((x for x in f.explanations if x.code.value == code), None)
                if a is None or a.required_next_check not in ranked or not tests.get(hid, "").startswith(
                        f"next check {a.required_next_check}: "):
                    raise ValueError(f"hypothesis {hid} needs a required next check from the ranked list")
        return self


# ====================================================================== explanation catalogue
_STATEMENTS = {
    Explanation.NO_ACTIVE_COOLING: "cooling was not running while the temperature was above the limit",
    Explanation.CAPACITY_SHORTFALL_WHILE_RUNNING: "cooling was running but did not hold the temperature below "
                                                  "the limit",
    Explanation.COOLING_RAN_THROUGHOUT: "cooling kept running while the temperature was below the limit",
    Explanation.HEAT_INGRESS: "a door opening or product loading added heat",
    Explanation.PRODUCT_LOADING: "product loading coincided with it",
    Explanation.DEFROST: "a defrost cycle raised the temperature",
    Explanation.CONTROL_CHANGE: "a setpoint or control change",
    Explanation.RECORDED_INTERVENTION: "maintenance work or a recorded equipment fault coincided with it",
    Explanation.SENSOR_NOT_REPRESENTATIVE: "the reading does not represent the room air (sensor artefact or "
                                           "position)",
}
_UPPER = (Explanation.NO_ACTIVE_COOLING, Explanation.CAPACITY_SHORTFALL_WHILE_RUNNING, Explanation.HEAT_INGRESS,
          Explanation.DEFROST, Explanation.CONTROL_CHANGE, Explanation.RECORDED_INTERVENTION,
          Explanation.SENSOR_NOT_REPRESENTATIVE)
_LOWER = (Explanation.COOLING_RAN_THROUGHOUT, Explanation.PRODUCT_LOADING, Explanation.CONTROL_CHANGE,
          Explanation.RECORDED_INTERVENTION, Explanation.SENSOR_NOT_REPRESENTATIVE)
_RECORD_KINDS = {                                  # explanation -> event-log kinds that bear on it
    Explanation.HEAT_INGRESS: ("DOOR_OPEN", "PRODUCT_LOAD"),
    Explanation.PRODUCT_LOADING: ("PRODUCT_LOAD",),
    Explanation.DEFROST: ("DEFROST",),
    Explanation.CONTROL_CHANGE: ("SETPOINT_CHANGE",),
    Explanation.RECORDED_INTERVENTION: ("MAINTENANCE", "EQUIPMENT_FAULT"),
    Explanation.NO_ACTIVE_COOLING: ("POWER_INTERRUPTION",),
}
_COMPRESSOR_BASED = {Explanation.NO_ACTIVE_COOLING, Explanation.CAPACITY_SHORTFALL_WHILE_RUNNING,
                     Explanation.COOLING_RAN_THROUGHOUT}
_TESTS = {
    Explanation.NO_ACTIVE_COOLING: (
        "check the compressor run history (controller, BMS) and power records for the window",
        "supported if the compressor was not running, or the unit had no power, from the reading before the "
        "excursion to its end; refuted if it was running throughout that window"),
    Explanation.CAPACITY_SHORTFALL_WHILE_RUNNING: (
        "check the compressor run history for the window; if it ran throughout, inspect the refrigeration unit",
        "supported if the compressor ran throughout the window and an inspection or measurement shows reduced "
        "cooling; refuted if it was off throughout the window; a mixed record bears on neither"),
    Explanation.COOLING_RAN_THROUGHOUT: (
        "check the compressor run history and controller settings for the window",
        "supported if the compressor ran throughout the window; refuted if it was off throughout it; a mixed "
        "record bears on neither"),
    Explanation.HEAT_INGRESS: (
        "look up door and loading records for the window",
        "supported if a door opening or loading is recorded between the reading before the excursion and its "
        "end; refuted only by a complete, system-recorded door and loading record showing none"),
    Explanation.PRODUCT_LOADING: (
        "look up loading records for the window",
        "supported if a loading is recorded in the window; refuted only by a complete, system-recorded loading "
        "record showing none"),
    Explanation.DEFROST: (
        "look up the controller's defrost history for the window",
        "supported if a defrost is recorded in the window; refuted if a complete controller history shows none"),
    Explanation.CONTROL_CHANGE: (
        "look up setpoint and controller-setting changes for the window",
        "supported if a setpoint or control change is recorded in the window; refuted if the controller's change "
        "history shows none"),
    Explanation.RECORDED_INTERVENTION: (
        "look up maintenance and fault records for the window",
        "supported if maintenance or a fault is recorded in the window; refuted if complete records show none"),
    Explanation.SENSOR_NOT_REPRESENTATIVE: (
        "compare the logger's sensor with a calibrated reference thermometer at the same position",
        "supported if the two disagree by more than their combined stated accuracy; refuted if they agree"),
}


# ====================================================================== helpers
def _lines(lines: list[int]) -> str:
    return f"{len(lines)} reading(s), lines {min(lines)}–{max(lines)}" if lines else "no reading"


def _t(x: datetime | None) -> str:
    return "?" if x is None else f"{x:%Y-%m-%d %H:%M}"


def _compressor_map(seg: SegmentationResult) -> dict[int, str] | None:
    """Recorded state per source line, from Phase 3 compressor events; None when Phase 3 did not segment it."""
    if not str(seg.summary.get("compressor", "")).startswith("segmented"):
        return None
    out: dict[int, str] = {}
    for e in seg.of_type(EventType.COMPRESSOR_ON, EventType.COMPRESSOR_OFF, EventType.COMPRESSOR_UNKNOWN):
        state = {EventType.COMPRESSOR_ON: "ON", EventType.COMPRESSOR_OFF: "OFF"}.get(e.type, "UNKNOWN")
        for ln in e.source_lines:
            out[ln] = state
    return out


def _record_interval(r, log_end: datetime) -> tuple[datetime, datetime]:
    """Phase 4 semantics: a power interruption without an end is ongoing; any other record without an end is a
    point in time."""
    if r.end is not None:
        return r.start, r.end
    return r.start, (log_end if r.kind.value == "POWER_INTERRUPTION" else r.start)


def _suspicious_lines(observations: pd.DataFrame | None) -> set[int]:
    if observations is None or "flags" not in observations:
        return set()
    hit = observations["flags"].str.contains("TEMP_SUSPICIOUS", regex=False)
    return {int(x) for x in observations.loc[hit, "source_line"]}


# a cause suggested as a question inside a validation message: "(door opening, defrost, sensor touch?)" or a
# trailing ": two sensors merged or a clock change?". Facts state what was observed, so these are left out here
_CAUSE_QUESTION = re.compile(r"\s*\([^()]*\?\)|:\s[^:]*\?$")


def _observation_only(message: str) -> str:
    return _CAUSE_QUESTION.sub("", message)


def _metadata_facts(metadata) -> list[dict]:
    """Supplied equipment and sensor information, as stated. Only the sensor location is used by a rule."""
    if metadata is None:
        return []
    out = []
    for name in type(metadata).model_fields:
        v = getattr(metadata, name)
        if name == "sensors":
            for s in v:
                out.append(dict(kind="SENSOR", source_ids=[s.id], record_provenance=s.provenance.value,
                                statement=f"sensor {s.id}: column {s.column or 'not in the log'}, location "
                                          f"{s.location or 'not stated'}, height "
                                          f"{'not stated' if s.height_m is None else f'{s.height_m:g} m'}, calibrated "
                                          f"{s.calibration_date or 'date not stated'}"
                                          + ("" if s.calibration_offset_c is None else
                                             f", documented offset {s.calibration_offset_c:g} °C (readings are "
                                             "never corrected)")))
        elif name == "notes" and v:
            out.append(dict(kind="NOTES", statement=f"notes: {v}"))
        elif v is not None and hasattr(v, "value"):
            out.append(dict(kind=name.upper(), source_ids=[v.source] if v.source else [],
                            record_provenance=v.provenance.value,
                            statement=f"{name} = {v.value:g} {v.unit} ({v.provenance.value})"))
    return out


def _forecast_output_facts(result) -> list[dict]:
    """Phase 4 model output and its held-out comparison with the baselines, quoted (no number is added here)."""
    out = []
    if result.status.value == "FORECAST" and result.trajectory:
        last = result.trajectory[-1]
        validated = ("" if result.validated_horizon_min is None
                     else f" (validated to {result.validated_horizon_min:g} min)")
        crossing = result.crossing_status.value + (f" at {_t(result.predicted_crossing_time)}"
                                                   if result.predicted_crossing_time else "")
        out.append(dict(kind="FORECAST_OUTPUT", start=result.prediction_start, end=last.ts,
                        statement=f"Phase 4 {result.selected_model}: {last.temp_c:.2f} °C predicted at "
                                  f"{_t(last.ts)}, {result.prediction_horizon_min:g} min after "
                                  f"{_t(result.prediction_start)}{validated}; threshold crossing {crossing}"))
    scored = [c for c in result.candidate_models if c.test_mae_c is not None]
    if scored:
        sel = next((c for c in scored if c.outcome.value == "SELECTED"), None)
        parts = [f"{c.name} ({c.kind.lower()}{', selected' if c is sel else ''}) {c.test_mae_c:.3f} °C"
                 for c in scored]
        text = "Phase 4 held-out test error (mean absolute): " + "; ".join(parts)
        h_sel = result.validation_diagnostics.selection_horizon_min if result.validation_diagnostics else None
        rows = [c for c in result.model_comparison if c.role == "TEST" and sel is not None
                and c.physical_method == sel.name and h_sel is not None and abs(c.horizon_min - h_sel) < 1e-9]
        for c in rows:
            text += (f"; at {c.horizon_min:g} min the selected model was closer in {c.physical_better} of "
                     f"{c.n_common} common held-out cases, {c.best_baseline} in {c.baseline_better}, tied in {c.tie}")
        out.append(dict(kind="BASELINE_COMPARISON", statement=text))
    return out


# ====================================================================== assessment of one excursion
def _assess(e: Event, side: str, comp: dict[int, str] | None, comp_status: str, records: list, rec_ids: dict,
            suspicious: set[int], log_end: datetime, sensor_location: str | None) -> list[ExplanationAssessment]:
    w0, w1 = e.before or e.start, e.end
    states = None if comp is None else [comp.get(ln, "UNKNOWN") for ln in e.source_lines]
    n = len(e.source_lines)
    supplied = {r.kind.value for r in records}
    system = {r.kind.value for r in records if r.provenance.value == "RECORDED_BY_SYSTEM"}
    overlap = [r for r in records if _overlaps(_record_interval(r, log_end), w0, w1)]
    decisive = states is not None and "UNKNOWN" not in states and len(set(states)) == 1
    out = []
    for code in (_UPPER if side == "above" else _LOWER):
        a = ExplanationAssessment(code=code, statement=_STATEMENTS[code].replace(
            "the limit", f"the limit ({e.observed.get('limit_c')} °C)"))
        if code in _COMPRESSOR_BASED:
            if states is None:
                a.missing.append(f"compressor_state ({comp_status})")
                a.next_check_ids.append("RETRIEVE_RUN_HISTORY")
            elif "UNKNOWN" in states:
                k, on, off = states.count("UNKNOWN"), states.count("ON"), states.count("OFF")
                a.neutral.append(f"compressor state missing or invalid in {k} of {_lines(e.source_lines)} (ON in "
                                 f"{on}, OFF in {off}); the missing states are never inferred")
                a.missing.append("compressor_state in every reading of the window")
                a.next_check_ids.append("RETRIEVE_RUN_HISTORY")
            else:
                on, off = states.count("ON"), states.count("OFF")
                text = f"compressor recorded {'OFF' if off == n else 'ON' if on == n else 'mixed'}"
                text += (f" in all {_lines(e.source_lines)}" if on in (0, n) else
                         f": ON in {on}, OFF in {off} of {_lines(e.source_lines)}")
                running = {Explanation.CAPACITY_SHORTFALL_WHILE_RUNNING, Explanation.COOLING_RAN_THROUGHOUT}
                if off == n:
                    (a.contradicting if code in running else a.supporting).append(text)
                elif on == n:
                    (a.supporting if code in running else a.contradicting).append(text)
                else:
                    a.neutral.append(text + ": neither supported nor contradicted by the state record")
        for kind in _RECORD_KINDS.get(code, ()):
            hits = [r for r in overlap if r.kind.value == kind]
            for r in hits:
                a.supporting.append(f"{rec_ids[id(r)]}: {kind} record ({r.provenance.value}) {_t(r.start)} – "
                                    f"{_t(r.end) if r.end else 'no end'} overlaps the window")
            if hits:
                continue
            if kind in system:
                a.neutral.append(f"no system-recorded {kind} overlaps the window (a missing record is not proof "
                                 "of absence)")
                continue
            if kind == "POWER_INTERRUPTION" and decisive:
                continue                      # the recorded compressor state already bears on it
            a.missing.append(f"{kind.lower()} records for the window" + (
                ": only manual records were supplied, and none overlaps" if kind in supplied else ""))
            if kind == "POWER_INTERRUPTION":
                if "RETRIEVE_RUN_HISTORY" not in a.next_check_ids:
                    a.next_check_ids.append("RETRIEVE_RUN_HISTORY")
            else:
                a.missing_records.append(kind.lower())
                if "REVIEW_OPERATIONAL_RECORDS" not in a.next_check_ids:
                    a.next_check_ids.append("REVIEW_OPERATIONAL_RECORDS")
        if code == Explanation.SENSOR_NOT_REPRESENTATIVE:
            flagged = sorted(set(e.source_lines) & suspicious)
            if flagged:
                a.supporting.append(f"{len(flagged)} of {n} reading(s) flagged TEMP_SUSPICIOUS by validation "
                                    f"(lines {flagged[:10]})" + ("" if len(flagged) == n else
                                                                 ": bears on those readings only"))
            a.missing.append("a reference measurement at the sensor position")
            if not sensor_location:
                a.missing.append("the analysed sensor's location (metadata)")
            a.next_check_ids.append("REFERENCE_READING")
        if code == Explanation.CAPACITY_SHORTFALL_WHILE_RUNNING and a.supporting and not a.contradicting:
            a.missing.append("an inspection or measurement of the cooling effect")
            a.next_check_ids.append("INSPECT_REFRIGERATION_UNIT")
        if a.supporting and a.contradicting:
            a.next_check_ids.insert(0, "RECONCILE_RECORDS")
        out.append(a)
    return out


def _overlaps(interval: tuple[datetime, datetime], w0: datetime, w1: datetime) -> bool:
    return interval[0] <= w1 and interval[1] >= w0


def _status(ex: list[ExplanationAssessment]) -> FindingStatus:
    if any(a.supporting and a.contradicting for a in ex):
        return FindingStatus.CONFLICTING_EVIDENCE
    if any(a.supporting or a.contradicting for a in ex):
        return FindingStatus.EVIDENCE_AVAILABLE
    return FindingStatus.INSUFFICIENT_EVIDENCE


# ====================================================================== next-best checks
def _p4_names(result, prefix: tuple[str, ...]) -> list[str]:
    recs = getattr(result, "recommendations", None) or []
    return sorted({r.missing_input for r in recs if r.missing_input.startswith(prefix)})


def _why(a: ExplanationAssessment) -> str:
    parts = [(lbl, xs) for lbl, xs in (("for", a.supporting), ("against", a.contradicting), ("neutral", a.neutral),
                                       ("missing", a.missing)) if xs]
    return "; ".join(f"{lbl}: " + " | ".join(xs) for lbl, xs in parts) or "no recorded evidence"


_KIND_EXPLANATIONS = {k: [c for c, kinds in _RECORD_KINDS.items() if k in kinds]
                      for k in {k for kinds in _RECORD_KINDS.values() for k in kinds}}


def _expected(cid: str, codes: set[Explanation], kinds: set[str]) -> list[str]:
    """What the check's result would be and what it would mean, by the rules of this module."""
    E = Explanation
    if cid == "REVIEW_OPERATIONAL_RECORDS":
        out = [f"{k}: a record overlapping a window would support "
               + ", ".join(c.value for c in _KIND_EXPLANATIONS[k.upper()] if c in codes) for k in sorted(kinds)]
        return out + ["no record found: neutral; an absence is never evidence against an explanation"]
    if cid == "RETRIEVE_RUN_HISTORY":
        out = []
        if codes & {E.NO_ACTIVE_COOLING, E.CAPACITY_SHORTFALL_WHILE_RUNNING}:
            out += ["compressor OFF in every reading of a window: supports NO_ACTIVE_COOLING and contradicts "
                    "CAPACITY_SHORTFALL_WHILE_RUNNING",
                    "ON in every reading: supports CAPACITY_SHORTFALL_WHILE_RUNNING and contradicts NO_ACTIVE_COOLING"]
        if E.COOLING_RAN_THROUGHOUT in codes:
            out += ["ON in every reading of a window: supports COOLING_RAN_THROUGHOUT; OFF in every reading "
                    "contradicts it"]
        if E.NO_ACTIVE_COOLING in codes:
            out += ["a power-interruption record overlapping a window: supports NO_ACTIVE_COOLING"]
        return out + ["a mix of ON and OFF readings: bears on neither"]
    if cid == "RECONCILE_RECORDS":
        return ["which of the disagreeing records is right for each window; the explanation is then assessed again "
                "on the agreed record"]
    if cid == "RECOVER_MISSING_READINGS":
        return ["the readings missing at the cut-off: the full duration and peak of the excursion, and whether a "
                "cut-off excursion is sustained (then diagnosed) or brief (then a fact only)"]
    if cid == "REFERENCE_READING":
        return ["the difference between the logger and the reference at the same position: larger than their "
                "combined stated accuracy supports SENSOR_NOT_REPRESENTATIVE, within it contradicts it",
                "the sensor's position relative to doors, the evaporator air outlet and heat sources"]
    if cid == "INSPECT_REFRIGERATION_UNIT":
        return ["dated findings for the evaporator coil, fans and condenser airflow: reduced cooling found supports "
                "CAPACITY_SHORTFALL_WHILE_RUNNING",
                "nothing found does not contradict it: a visual inspection does not measure cooling capacity"]
    out = ["an independent source that agrees: the assessment rests on two sources, and stays a hypothesis",
           "an independent source that disagrees: a conflict, to be reconciled before anything is concluded"]
    if codes & _COMPRESSOR_BASED:
        out.append("for the compressor state: a mix of ON and OFF readings stays neutral, whatever its source")
    if codes & set(_RECORD_KINDS) - {E.NO_ACTIVE_COOLING}:
        out.append("a record source shown to be complete, with no event of a kind in a window: contradicts the "
                   "record-based explanation, as its success criterion states")
    return out


_SAFETY = {
    Effort.EXISTING_RECORDS: (Safety.OFFICE_ONLY, "records and files only; no site access"),
    Effort.SPOT_CHECK: (Safety.ROOM_ENTRY, "entering the cold room: follow the site's cold-room entry rules; nothing "
                                           "on the refrigeration equipment is opened"),
    Effort.SITE_INSPECTION: (Safety.EQUIPMENT_ACCESS, "a qualified refrigeration technician only, following the "
                                                      "site's electrical isolation and refrigerant-handling rules"),
}
RANKING_BASIS = ("rule: relevance (more unresolved finding/explanation items first), then feasibility (less effort: "
                 "existing records, spot check, site inspection), then evidence value ("
                 + " > ".join(v.value for v in EVIDENCE_VALUE_ORDER) + "), then check id; a work priority, not a "
                 "confidence in any explanation")


def _checks(findings: list[Finding], cut: list[Event], time_basis: str, median_s: float | None,
            result, ev_fact: dict[str, str]) -> list[NextCheck]:
    plans: dict[str, dict] = {}

    def add(cid, window, item, evidence, code=None, finding=None):
        p = plans.setdefault(cid, {"windows": [], "resolves": [], "evidence": [], "kinds": set(), "codes": set(),
                                   "per_finding": {}})
        if window not in p["windows"]:
            p["windows"].append(window)
        p["resolves"].append(item)
        p["evidence"].append(evidence)
        if code is not None:
            p["codes"].add(code)
            p["per_finding"].setdefault(finding, set()).add(code)

    for f in findings:
        win = f"{f.event_id} ({f.finding_id}): {_t(f.window_start)} to {_t(f.window_end)}"
        for a in f.explanations:
            for cid in a.next_check_ids:
                add(cid, win, f"{f.finding_id} {a.code.value}", f"{f.fact_id} {f.finding_id} {a.code.value}: "
                    + _why(a), a.code, f.finding_id)
            if a.missing_records:
                plans["REVIEW_OPERATIONAL_RECORDS"]["kinds"] |= set(a.missing_records)
        censored = [x for x in f.limitations if x.startswith("censored")]
        if censored:
            add("RECOVER_MISSING_READINGS", win, f"{f.finding_id} duration and peak (cut off by a gap or the log edge)",
                f"{f.fact_id} {f.finding_id}: {censored[0]}")
    for e in cut:
        add("RECOVER_MISSING_READINGS", f"{e.event_id}: {_t(e.start)} to {_t(e.end)}",
            f"{e.event_id} sustained or brief (cut off by a gap or the log edge)",
            f"{ev_fact[e.event_id]} {e.event_id}: beyond the limit, cut off by a gap or the log edge")

    sampling = f"at the temperature sampling interval ({median_s:.0f} s) or finer" if median_s else \
        "at the temperature sampling interval or finer"
    spec = {
        "REVIEW_OPERATIONAL_RECORDS": dict(
            action="Look up the site's records for these windows ({kinds}): controller event history, door-contact "
                   "log, delivery and maintenance logs. Supply what exists as an event log (--events: kind, start, "
                   "end, provenance)",
            required_input="{kinds}", units=f"start and end timestamps in the log's time basis ({time_basis})",
            resolution_or_duration="to the minute or finer; each window runs from the reading before the "
                                   "excursion to its end",
            why_useful="a record in the window supports the matching explanation; without records these "
                       "explanations cannot be told apart", effort=Effort.EXISTING_RECORDS,
            value=EvidenceValue.SUPPORTS_ONLY, p4=("door_open", "defrost", "product_temperature and product_load")),
        "RETRIEVE_RUN_HISTORY": dict(
            action="Retrieve the compressor on/off history for these windows (controller, BMS or a current-clamp "
                   "record), or power-interruption records",
            required_input="compressor_state (or power_interruption records)", units="on/off",
            resolution_or_duration=sampling,
            why_useful="shows whether cooling ran during the excursion: it separates 'cooling not running' from "
                       "'cooling running but not holding the temperature'", effort=Effort.EXISTING_RECORDS,
            value=EvidenceValue.SUPPORTS_OR_CONTRADICTS, p4=("compressor_state",)),
        "RECONCILE_RECORDS": dict(
            action="These windows have records that disagree (e.g. a power-interruption record while the "
                   "compressor is recorded ON): check each record's times and source against the logger clock",
            required_input="the disagreeing records and the clock settings of each source", units=None,
            resolution_or_duration=f"time basis of the log: {time_basis}",
            why_useful="neither record is taken as fact over the other until they are reconciled",
            effort=Effort.EXISTING_RECORDS, value=EvidenceValue.RESOLVES_CONFLICT, p4=()),
        "CORROBORATE_RECORDS": dict(
            action="Confirm the records that decide these explanations with an independent source for the same "
                   "windows (controller alarm or event history, BMS export, power records, site log or technician "
                   "report), and establish whether each record source is complete for the windows",
            required_input="an independent record of the same states or events; the completeness of each record "
                           "source", units=f"timestamps in the log's time basis ({time_basis}); on/off for states",
            resolution_or_duration="the excursion windows, to the minute or finer",
            why_useful="an explanation supported or contradicted by one source stays a hypothesis until a second "
                       "source agrees; only a record known to be complete can show that an event did not happen",
            effort=Effort.EXISTING_RECORDS, value=EvidenceValue.CORROBORATES, p4=()),
        "RECOVER_MISSING_READINGS": dict(
            action="Check whether the logger or BMS still holds the readings missing next to these excursions",
            required_input="room_temperature", units="°C", resolution_or_duration="the log's own sampling interval",
            why_useful="the excursion is cut off by a gap or the log edge, so its duration and peak are lower "
                       "bounds", effort=Effort.EXISTING_RECORDS, value=EvidenceValue.BOUNDS_OBSERVATION, p4=()),
        "REFERENCE_READING": dict(
            action="Compare the logger's sensor with a calibrated reference thermometer at the same position, and "
                   "record the position relative to doors, the evaporator air outlet and heat sources",
            required_input="calibrated reference thermometer; sensor location (metadata)", units="°C",
            resolution_or_duration="simultaneous readings at the logger's sampling interval; accuracy as stated "
                                   "on each instrument's calibration certificate",
            why_useful="the only check that bears on whether the readings represent the room air",
            effort=Effort.SPOT_CHECK, value=EvidenceValue.SUPPORTS_OR_CONTRADICTS,
            p4=("sensor location", "spatial temperature")),
        "INSPECT_REFRIGERATION_UNIT": dict(
            action="Inspect the refrigeration unit: evaporator coil (frost or ice), evaporator fans, condenser "
                   "airflow; record the findings with date and time",
            required_input="technician inspection (evaporator_condenser_measurements are not a log input in this "
                           "version)", units="qualitative findings; °C or bar if measured",
            resolution_or_duration=None,
            why_useful="the compressor was recorded running throughout the excursion; ColdSense cannot observe the "
                       "cooling effect, an inspection can", effort=Effort.SITE_INSPECTION,
            value=EvidenceValue.SUPPORTS_ONLY, p4=()),
    }
    two = {Explanation.NO_ACTIVE_COOLING, Explanation.CAPACITY_SHORTFALL_WHILE_RUNNING}
    rows = []
    for cid, p in plans.items():
        s = spec[cid]
        kinds = ", ".join(sorted(p["kinds"]))
        value = s["value"]
        if cid == "RETRIEVE_RUN_HISTORY" and any(two <= codes for codes in p["per_finding"].values()):
            value = EvidenceValue.DISCRIMINATES          # one result supports one and contradicts the other
        safety, note = _SAFETY[s["effort"]]
        rows.append(dict(check_id=cid, action=s["action"].format(kinds=kinds), windows=p["windows"],
                         required_input=s["required_input"].format(kinds=kinds), units=s["units"],
                         resolution_or_duration=s["resolution_or_duration"], why_useful=s["why_useful"],
                         resolves=p["resolves"], open_items=len(p["resolves"]), effort=s["effort"],
                         phase4_recommendations=_p4_names(result, s["p4"]) if s["p4"] else [],
                         evidence=p["evidence"], expected_information=_expected(cid, p["codes"], p["kinds"]),
                         evidence_value=value, safety=safety, safety_note=note))
    rows.sort(key=lambda r: (-r["open_items"], EFFORT_ORDER.index(r["effort"]),
                             EVIDENCE_VALUE_ORDER.index(r["evidence_value"]), r["check_id"]))
    return [NextCheck(rank=i, ranking_basis=RANKING_BASIS, **r) for i, r in enumerate(rows, 1)]


# ====================================================================== entry points
def diagnose(report: DataQualityReport, seg: SegmentationResult, *, observations: pd.DataFrame | None = None,
             operational_events: list | None = None, metadata=None, result=None) -> DiagnosticReport:
    """Facts, hypotheses and next checks from existing Phase 2-5 evidence; nothing is recomputed."""
    records = sorted(operational_events or [], key=lambda r: (r.start, r.kind.value, r.end or r.start,
                                                              r.provenance.value, r.source or "", r.note or ""))
    out = DiagnosticReport(status=DiagnosisStatus.INSUFFICIENT_EVIDENCE, refusal_reason="not assessed",
                           data_evidence=report.evidence, time_basis=seg.time_basis, as_of=report.end)
    facts: list[ObservedFact] = []

    def fact(**kw) -> ObservedFact:
        f = ObservedFact(fact_id=f"O{len(facts) + 1:03d}", **kw)
        facts.append(f)
        return f

    # ---- Phase 2: data quality (always visible)
    fact(source=FactSource.PHASE2_VALIDATION, kind=f"VERDICT_{report.verdict.value}", start=report.start,
         end=report.end, statement=f"validation verdict {report.verdict.value}: {report.n_rows_clean} usable of "
         f"{report.n_rows_raw} rows, coverage {report.coverage_pct:.1f} %, {report.n_gaps} gap(s)",
         evidence=report.evidence)
    trimmed = False
    for i in report.issues:
        msg = _observation_only(i.message)
        trimmed |= msg != i.message
        fact(source=FactSource.PHASE2_VALIDATION, kind=i.code, statement=msg, evidence=report.evidence)
        out.data_quality_limitations.append(f"{i.code}: {msg}")
    out.data_quality_limitations += [x for x in seg.limitations if x not in out.data_quality_limitations]

    # ---- Phase 3 compressor status and the supplied records
    comp_status = str(seg.summary.get("compressor", "absent: no compressor_state column"))
    if not seg.refused:
        fact(source=FactSource.PHASE3_EVENT, kind="COMPRESSOR_RECORD", statement=f"compressor state: {comp_status}",
             evidence=seg.evidence)
    gap_facts = []
    for e in seg.of_type(EventType.DATA_GAP):
        gap_facts.append((e, fact(source=FactSource.PHASE3_EVENT, kind=e.type.value, start=e.start, end=e.end,
                                  source_lines=list(e.source_lines), source_ids=[e.event_id], statement=e.rationale,
                                  limitations=list(e.limitations), evidence=e.evidence)))
    rec_ids = {}
    for r in records:
        f = fact(source=FactSource.EVENT_LOG, kind=r.kind.value, start=r.start, end=r.end,
                 source_ids=[r.source] if r.source else [], record_provenance=r.provenance.value,
                 statement=f"{r.kind.value} recorded ({r.provenance.value})" + (f": {r.note}" if r.note else ""),
                 limitations=["a supplied record: taken as stated, not verified by ColdSense"],
                 evidence=EvidenceClass.SIMULATED if report.evidence == EvidenceClass.SIMULATED
                 else EvidenceClass.UNVALIDATED)
        rec_ids[id(r)] = f.fact_id
    stated = EvidenceClass.SIMULATED if report.evidence == EvidenceClass.SIMULATED else EvidenceClass.UNVALIDATED
    for m in _metadata_facts(metadata):
        fact(source=FactSource.ROOM_METADATA, evidence=stated,
             limitations=["supplied metadata: taken as stated, not verified by ColdSense; no rule uses it except the "
                          "analysed sensor's location"], **m)

    # ---- Phase 4 / 5 context (never evidence for an explanation)
    if result is not None:
        model = f" by {result.selected_model}" if result.selected_model else ""
        why = f": {result.refusal_code.value}: {result.refusal_reason}" if result.refusal_code else ""
        fact(source=FactSource.PHASE4_FORECAST, kind=f"FORECAST_{result.status.value}",
             statement=f"Phase 4 forecast {result.status.value}{model}{why}", evidence=result.evidence)
        for m in _forecast_output_facts(result):
            fact(source=FactSource.PHASE4_FORECAST, evidence=result.evidence, **m)
        rr = result.reliability_report
        if rr is not None:
            fact(source=FactSource.PHASE5_RELIABILITY, kind=f"RELIABILITY_{rr.reliability.status.value}",
                 statement=f"Phase 5 reliability {rr.reliability.status.value} "
                           f"[{rr.reliability.validation_scope.value}]; interval {rr.uncertainty.status.value}"
                           + (f"; failed gates {rr.reliability.gates_failed}" if rr.reliability.gates_failed else ""),
                 evidence=result.evidence)
        out.limitations.append("the Phase 4 forecast and Phase 5 reliability are context only: no forecast number "
                               "is used as evidence for an explanation")

    out.limitations += [
        "events and facts are observations; every explanation is a hypothesis, never a diagnosis of cause",
        "no probability, score or confidence is given; hypotheses are listed in a fixed order, not ranked",
        "absence of a record is never evidence of absence; supplied records are taken as stated, not verified "
        "(SIMULATED in a simulated scenario, otherwise UNVALIDATED)",
        "only sustained excursions beyond the limit you set are diagnosed; brief excursions are reported as facts",
        f"data evidence {report.evidence.value}: statements hold only for these data; {RULES_VALIDATION}",
        "next checks are ranked as a work priority (relevance, effort, evidence value), not by any confidence in "
        "an explanation",
    ]
    if trimmed:
        out.limitations.append("causes that validation messages suggest as questions are left out of the facts; "
                               "the full messages are in the data-quality report (coldsense validate)")

    def finish(status: DiagnosisStatus, reason: str | None) -> DiagnosticReport:
        out.observed_facts, out.status, out.refusal_reason = facts, status, reason
        return DiagnosticReport.model_validate(out.model_dump())

    # ---- refusals before any finding
    if report.verdict == Verdict.REFUSED or seg.refused:
        why = "; ".join(seg.reasons) if seg.refused else "validation refused the log"
        return finish(DiagnosisStatus.INSUFFICIENT_EVIDENCE, f"no usable events: {why}")
    limits = [x for x in (seg.params.upper_threshold_c, seg.params.lower_threshold_c) if x is not None]
    if not limits:
        return finish(DiagnosisStatus.INSUFFICIENT_EVIDENCE,
                      "no temperature limit was given (--upper-threshold / --lower-threshold): ColdSense does not "
                      "decide what temperature is abnormal")

    # ---- Phase 3 excursions as facts
    sustained = seg.of_type(EventType.THRESHOLD_EXCURSION)
    brief = seg.of_type(EventType.BRIEF_EXCURSION)
    cut = [e for e in seg.of_type(EventType.INSUFFICIENT_DATA) if "limit_c" in e.observed]
    ev_fact = {}
    for e in sorted(sustained + brief + cut, key=lambda x: x.event_id):
        f = fact(source=FactSource.PHASE3_EVENT, kind=e.type.value, start=e.start, end=e.end,
                 source_lines=list(e.source_lines), source_ids=[e.event_id], statement=e.rationale,
                 limitations=list(e.limitations), evidence=e.evidence)
        ev_fact[e.event_id] = f.fact_id

    # ---- findings: sustained excursions, one per event
    comp = _compressor_map(seg)
    suspicious = _suspicious_lines(observations)
    col = report.input.column_mapping.get("temperature") if report.input else None
    sensor = next((s for s in (metadata.sensors if metadata else []) if s.column == col), None) if col else None
    log_end = report.end
    for k, e in enumerate(sustained, 1):
        side = e.observed.get("side", "above")
        ex = _assess(e, side, comp, comp_status, records, rec_ids, suspicious, log_end,
                     sensor.location if sensor else None)
        lim = (["censored: cut off by a gap or the log edge; duration and peak are lower bounds"]
               if e.start_censored or e.end_censored else [])
        if e.before is None:
            lim.append("no reading before the excursion: the window starts at its first reading")
        out.findings.append(Finding(
            finding_id=f"F{k}", event_id=e.event_id, fact_id=ev_fact[e.event_id], side=side,
            limit_c=float(e.observed.get("limit_c")), window_start=e.before or e.start, window_end=e.end,
            statement=f"{e.event_id}: {_lines(e.source_lines)} {side} {e.observed.get('limit_c')} °C from "
                      f"{_t(e.start)} to {_t(e.end)} (peak {e.observed.get('peak_c')} °C)",
            status=_status(ex), explanations=ex, limitations=lim))

    # ---- temperature behaviour and gaps over each finding's window: observations, never evidence
    behaviour = seg.of_type(EventType.TEMP_STABLE, EventType.TEMP_RISING, EventType.TEMP_FALLING,
                            EventType.TEMP_IRREGULAR)
    near = {e.event_id for f in out.findings for e in behaviour if _overlaps((e.start, e.end), f.window_start,
                                                                                f.window_end)}
    for e in sorted((e for e in behaviour if e.event_id in near), key=lambda x: x.event_id):
        ev_fact[e.event_id] = fact(source=FactSource.PHASE3_EVENT, kind=e.type.value, start=e.start, end=e.end,
                                   source_lines=list(e.source_lines), source_ids=[e.event_id], statement=e.rationale,
                                   limitations=list(e.limitations), evidence=e.evidence).fact_id
    for f in out.findings:
        f.context_fact_ids = [ev_fact[e.event_id] for e in sorted(behaviour, key=lambda x: x.event_id)
                              if _overlaps((e.start, e.end), f.window_start, f.window_end)]
        f.context_fact_ids += [g.fact_id for e, g in gap_facts if _overlaps((e.start, e.end), f.window_start,
                                                                             f.window_end)]

    # ---- the required next check of every explanation that becomes a hypothesis
    for f in out.findings:
        for a in f.explanations:
            if f.status != FindingStatus.INSUFFICIENT_EVIDENCE and not a.next_check_ids:
                a.next_check_ids.append("CORROBORATE_RECORDS")    # decided by one source, or neutral by a record
            a.required_next_check = a.next_check_ids[0] if a.next_check_ids else None

    # ---- hypotheses: only where recorded evidence bears on an explanation
    hyps = []
    for f in out.findings:
        if f.status == FindingStatus.INSUFFICIENT_EVIDENCE:
            continue
        for a in f.explanations:
            hid = f"H-{f.finding_id}-{a.code.value}"
            test, success = _TESTS[a.code]
            hyps.append(Hypothesis(
                id=hid, statement=f"{f.event_id}: {a.statement}",
                motivating_observation=f"{f.fact_id} {f.statement}",
                supporting_evidence=list(a.supporting), contradicting_evidence=list(a.contradicting),
                alternatives=[f"{b.code.value}: {b.statement}" for b in f.explanations if b.code != a.code],
                data_needed=list(a.missing),
                test=f"next check {a.required_next_check}: {test}: {_t(f.window_start)} to {_t(f.window_end)}",
                success_criteria=success, generated_by=GENERATED_BY, created_at=out.as_of))
            f.hypothesis_ids.append(hid)
    out.hypotheses = hyps
    out.next_checks = _checks(out.findings, cut, seg.time_basis, report.median_interval_s, result, ev_fact)
    out.next_best_check = out.next_checks[0].check_id if out.next_checks else None

    gaps = report.n_gaps
    if not sustained:
        if cut:
            return finish(DiagnosisStatus.INSUFFICIENT_EVIDENCE,
                          f"{len(cut)} excursion(s) beyond the limit are cut off by gaps or the log edge: whether "
                          "they were sustained cannot be told")
        out.limitations.append(f"no sustained excursion in the usable data; {len(brief)} brief excursion(s)"
                               + (f"; {gaps} gap(s) could hide excursions" if gaps else ""))
        return finish(DiagnosisStatus.NO_FINDING, None)
    if not hyps:
        return finish(DiagnosisStatus.INSUFFICIENT_EVIDENCE,
                      f"no recorded evidence bears on any explanation of the {len(sustained)} sustained "
                      "excursion(s): compressor state, operational records and validation flags say nothing about "
                      "them; the next checks list what would")
    return finish(DiagnosisStatus.HYPOTHESES, None)


def diagnose_source(src, *, source: str | None = None, evidence: EvidenceClass = EvidenceClass.UNVALIDATED,
                    options=None, limits=None, params: SegmentationParams | None = None,
                    operational_events: list | None = None, metadata=None, request=None, reliability_config=None):
    """Phase 2 + Phase 3 (+ Phase 4/5 when a forecast request is given, on the log as it was at the origin), then
    Phase 6. Returns (DataQualityReport, SegmentationResult, ForecastResult or None, DiagnosticReport)."""
    result = None
    if request is not None:
        from .thermal.reliability import assess_source

        clean, obs, report, result = assess_source(src, source=source, evidence=evidence, options=options,
                                                   limits=limits, request=request, events=operational_events,
                                                   metadata=metadata, config=reliability_config)
    else:
        clean, obs, report = validate_source(src, source=source, evidence=evidence, options=options, limits=limits)
    seg = segment_events(clean, report, obs, params)
    diag = diagnose(report, seg, observations=obs, operational_events=operational_events, metadata=metadata,
                    result=result)
    return report, seg, result, diag


__all__ = ["METHOD_VERSION", "DiagnosisStatus", "DiagnosticReport", "Effort", "Explanation", "FactSource",
           "Finding", "FindingStatus", "NextCheck", "ObservedFact", "diagnose", "diagnose_source"]
