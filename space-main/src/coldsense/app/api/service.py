"""Service layer: calls the accepted Phase 2-5 functions and maps their results to the API schemas.

No validation rule, model or refusal decision lives here (ARCHITECTURE.md §4, ``app``: "any validation or modelling
logic" does not belong). Every number in a response is copied from a domain result; nothing is recomputed.
"""

from __future__ import annotations

import io
import logging
import math
import re
import threading
from datetime import UTC, datetime

import pandas as pd
from pydantic import ValidationError

from coldsense import __version__
from coldsense.core.errors import ColdSenseError
from coldsense.core.evidence import EvidenceClass
from coldsense.core.results import Severity
from coldsense.domains.cold_room.diagnostics import METHOD_VERSION as DIAGNOSIS_VERSION
from coldsense.domains.cold_room.diagnostics import diagnose
from coldsense.domains.cold_room.ingest import IngestOptions
from coldsense.domains.cold_room.pipeline import segment_source, validate_source
from coldsense.domains.cold_room.schemas import DataQualityReport, EventType, SegmentationParams
from coldsense.domains.cold_room.thermal import ENGINE_VERSION, ForecastRequest, ForecastStatus, forecast_source
from coldsense.domains.cold_room.thermal.reliability import METHOD_VERSION, assess_source
from coldsense.domains.cold_room.thermal.schemas import SCOPE_FOR_EVIDENCE, ValidationScope

from . import limits as L
from .errors import ApiError, ErrorCode
from .schemas import (
    AnalysisRequest,
    AnalysisView,
    BaselineView,
    CaseStatus,
    CaseSteps,
    CrossingIntervalView,
    CrossingView,
    DiagnosisSummary,
    DisplaySampling,
    EvidenceView,
    IncidentView,
    IntervalView,
    InvestigationView,
    NextCheckView,
    NotAvailableBlock,
    ObservedTemperature,
    PredictionRefusal,
    PredictionRequest,
    PredictionStatusView,
    PredictionSuccess,
    QualityHeadline,
    QualityView,
    RecommendationView,
    RefusalDetail,
    ReliabilityView,
    ResultProvenance,
    SelectedModel,
    StepState,
    ThresholdView,
    TimelineGap,
    TimelinePoint,
    TimelineRun,
    TimelineView,
    TrajectoryPointView,
)
from .store import CaseRecord

log = logging.getLogger(__name__)

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def _stream(content: bytes, filename: str) -> io.BytesIO:
    """In-memory file named after the upload, so reports say which file they describe (never a server path)."""
    buf = io.BytesIO(content)
    buf.name = filename
    return buf


def _now() -> datetime:
    return datetime.now(UTC)


def _ts(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, UTC)


def sanitize_filename(name: str | None) -> str:
    """Display name only: last path component, no control characters, bounded length."""
    base = re.split(r"[\\/]", name or "")[-1]
    base = _CONTROL.sub("", base).strip()
    return (base or "upload.csv")[: L.FILENAME_MAX_CHARS]


def check_upload(filename: str, content: bytes) -> None:
    if not filename.lower().endswith(L.UPLOAD_ALLOWED_SUFFIXES):
        raise ApiError(415, ErrorCode.UPLOAD_TYPE_NOT_ALLOWED,
                       f"only {', '.join(L.UPLOAD_ALLOWED_SUFFIXES)} files are accepted")
    if not content.strip():
        raise ApiError(422, ErrorCode.UPLOAD_EMPTY, "the uploaded file is empty")
    if content.count(b"\n") > L.UPLOAD_MAX_ROWS + 1:          # header + rows; cheap bound before parsing
        raise ApiError(413, ErrorCode.UPLOAD_TOO_MANY_ROWS, f"at most {L.UPLOAD_MAX_ROWS} data rows are accepted")


def ingest_options(*, unit: str | None, timezone: str | None, timestamp_format: str | None,
                   delimiter: str | None, encoding: str | None) -> IngestOptions:
    try:
        return IngestOptions(temperature_unit=unit, timezone=timezone, timestamp_format=timestamp_format,
                             delimiter=delimiter, encoding=encoding)
    except ValidationError as e:
        raise ApiError(422, ErrorCode.INVALID_REQUEST, "invalid ingest option: " + "; ".join(
            f"{'.'.join(str(p) for p in x['loc'])}: {x['msg']}" for x in e.errors())) from None


def _unreadable(e: ColdSenseError) -> ApiError:
    return ApiError(422, ErrorCode.DATA_UNREADABLE, f"the file cannot be interpreted: {e}")


def validate_upload(content: bytes, filename: str, evidence: EvidenceClass,
                    options: IngestOptions) -> DataQualityReport:
    """Phase 2 ingest + validation. A file that cannot be interpreted is an error; poor data is a REFUSED verdict."""
    try:
        _, _, report = validate_source(_stream(content, filename), source=filename, evidence=evidence, options=options)
    except ColdSenseError as e:
        raise _unreadable(e) from None
    return report


def provenance(rec: CaseRecord, *, method: str, method_version: str | None = None,
               result_evidence: EvidenceClass | None = None, scope: ValidationScope | None = None) -> ResultProvenance:
    """``scope``: from this result's Phase 5 assessment; otherwise the Phase 5 table for the engine evidence class.
    The uploader's declaration is copied for display only and enters neither."""
    return ResultProvenance(
        data_evidence_class=EvidenceClass(rec.evidence), evidence_basis=rec.source_kind,
        declared_evidence=rec.declared_evidence,
        declaration_status="DECLARED_BY_UPLOADER_NOT_VERIFIED" if rec.declared_evidence else "NOT_DECLARED",
        result_evidence_class=result_evidence, source_filename=rec.filename, source_sha256=rec.sha256,
        source_size_bytes=len(rec.content), method=method, method_version=method_version,
        validation_scope=scope or SCOPE_FOR_EVIDENCE[EvidenceClass(rec.evidence)],
        validation_scope_source="PHASE5_ASSESSMENT" if scope else "PHASE5_SCOPE_TABLE",
        coldsense_version=__version__, computed_at=_now())


def headline(rep: DataQualityReport) -> QualityHeadline:
    c = rep.counts
    return QualityHeadline(
        verdict=rep.verdict, suitable_for_analysis=rep.suitable_for_downstream,
        rows_read=c.data_rows_read if c else rep.n_rows_raw, rows_usable=c.rows_usable if c else rep.n_rows_clean,
        start=rep.start, end=rep.end, duration_hours=rep.duration_hours, coverage_pct=rep.coverage_pct,
        blocking_issues=sum(i.severity == Severity.BLOCKING for i in rep.issues),
        warnings=sum(i.severity == Severity.WARNING for i in rep.issues))


def quality_view(rec: CaseRecord) -> QualityView:
    rep: DataQualityReport = rec.results["quality_report"]
    return QualityView(case_id=rec.case_id, headline=headline(rep), issues=rep.issues, limitations=rep.limitations,
                       report=rep, provenance=provenance(rec, method="Phase 2 ingestion and data validation"))


def case_status(rec: CaseRecord) -> CaseStatus:
    rep: DataQualityReport = rec.results["quality_report"]
    usable = rep.suitable_for_downstream
    last = rec.results.get("prediction")
    analysis_view_ = rec.results.get("analysis")
    if not usable:
        analysis = prediction = StepState.BLOCKED
    else:
        analysis = (StepState.NOT_RUN if analysis_view_ is None else
                    StepState.REFUSED if analysis_view_.outcome == "REFUSED" else StepState.READY)
        prediction = (StepState.NOT_RUN if last is None else
                      StepState.READY if last.outcome == "FORECAST" else StepState.REFUSED)
    return CaseStatus(case_id=rec.case_id, created_at=_ts(rec.created_at), expires_at=_ts(rec.expires_at),
                      filename=rec.filename, size_bytes=len(rec.content), sha256=rec.sha256,
                      evidence_class=EvidenceClass(rec.evidence), evidence_basis=rec.source_kind,
                      declared_evidence=rec.declared_evidence,
                      declaration_status="DECLARED_BY_UPLOADER_NOT_VERIFIED" if rec.declared_evidence
                      else "NOT_DECLARED",
                      validation_scope=SCOPE_FOR_EVIDENCE[EvidenceClass(rec.evidence)], quality=headline(rep),
                      steps=CaseSteps(quality=StepState.READY if usable else StepState.REFUSED, analysis=analysis,
                                      prediction=prediction),
                      last_prediction_outcome=None if last is None else last.outcome)


# ---------------------------------------------------------------- analysis slots
class AnalysisSlots:
    """Bounds concurrent CPU-heavy domain calls; a request that cannot get a slot in time gets BUSY."""

    def __init__(self, n: int = L.MAX_CONCURRENT_ANALYSES, wait_s: float = L.ANALYSIS_QUEUE_WAIT_S) -> None:
        self._sem, self._wait = threading.BoundedSemaphore(n), wait_s

    def __enter__(self):
        if not self._sem.acquire(timeout=self._wait):
            raise ApiError(503, ErrorCode.BUSY, "the service is busy with other analyses; try again shortly",
                           retryable=True)
        return self

    def __exit__(self, *_):
        self._sem.release()


# ---------------------------------------------------------------- analysis (Phases 3 and 6)
ANALYSIS_METHOD = "Phase 3 event segmentation + Phase 6 diagnostics"
FRAME_COLUMNS = ("ts", "temp_c", "ambient_c", "compressor_on", "source_line")


def run_analysis(rec: CaseRecord, req: AnalysisRequest, slots: AnalysisSlots) -> AnalysisView:
    """Phase 2 -> Phase 3 -> Phase 6 with the limits the visitor set, composed exactly as
    ``diagnostics.diagnose_source`` does without a forecast request. Replaces the case's previous analysis."""
    try:
        params = SegmentationParams(upper_threshold_c=req.upper_threshold_c, lower_threshold_c=req.lower_threshold_c)
    except ValidationError as e:
        raise ApiError(422, ErrorCode.INVALID_REQUEST, "invalid analysis setting: " + "; ".join(
            x["msg"].removeprefix("Value error, ") for x in e.errors())) from None
    with slots:
        try:
            clean, obs, report, seg = segment_source(_stream(rec.content, rec.filename), source=rec.filename,
                                                     evidence=EvidenceClass(rec.evidence),
                                                     options=rec.options["ingest"], params=params)
        except ColdSenseError as e:
            raise _unreadable(e) from None
        diag = diagnose(report, seg, observations=obs)
    frame = clean.loc[:, [c for c in FRAME_COLUMNS if c in clean.columns]].copy()
    view = AnalysisView(
        case_id=rec.case_id, outcome="REFUSED" if seg.refused else "COMPLETED", reasons=seg.reasons,
        diagnosis=DiagnosisSummary(status=diag.status, refusal_reason=diag.refusal_reason, findings=len(diag.findings),
                                   hypotheses=len(diag.hypotheses), next_checks=len(diag.next_checks),
                                   next_best_check=diag.next_best_check),
        upper_threshold_c=req.upper_threshold_c, lower_threshold_c=req.lower_threshold_c,
        time_basis=seg.time_basis, summary=seg.summary, events=seg.events, limitations=seg.limitations,
        provenance=provenance(rec, method=ANALYSIS_METHOD, method_version=DIAGNOSIS_VERSION))
    rec.results.update(analysis=view, segmentation=seg, diagnosis=diag, frame=frame)
    return view


def _analysis(rec: CaseRecord):
    if "analysis" not in rec.results:
        raise ApiError(404, ErrorCode.ANALYSIS_NOT_RUN, "run the analysis for this case first")
    return rec.results["analysis"], rec.results["segmentation"], rec.results["diagnosis"]


# ---------------------------------------------------------------- Stage 2 views (read-only; nothing recomputed)
TIMELINE_MAX_POINTS = 2000      # display only; above this, each run is thinned to its min and max per bucket


def _points(df) -> list[TimelinePoint]:
    out = []
    for r in df.itertuples(index=False):
        amb = getattr(r, "ambient_c", None)
        comp = getattr(r, "compressor_on", None)
        out.append(TimelinePoint(ts=r.ts.to_pydatetime(), temp_c=float(r.temp_c),
                                 ambient_c=None if amb is None or pd.isna(amb) else float(amb),
                                 compressor_on=None if comp is None or pd.isna(comp) else bool(comp),
                                 source_line=int(r.source_line)))
    return out


def _thin(df, bucket: int):
    """Keep, per bucket of consecutive readings, the readings with the lowest and highest temperature, in time
    order. Display only: it never creates a value and never bridges a gap (runs are thinned separately)."""
    if bucket <= 1 or len(df) <= 2:
        return df
    keep: set[int] = set()
    temps = df["temp_c"].to_numpy()
    for i in range(0, len(df), bucket):
        chunk = temps[i:i + bucket]
        keep.update((i + int(chunk.argmin()), i + int(chunk.argmax())))
    keep.update((0, len(df) - 1))
    return df.iloc[sorted(keep)]


def timeline_view(rec: CaseRecord) -> TimelineView:
    _, seg, _ = _analysis(rec)
    frame = rec.results["frame"].sort_values("ts")
    gaps = sorted((e for e in seg.events if e.type == EventType.DATA_GAP), key=lambda e: e.start)
    run_of = [0] * len(frame)
    if gaps:                                   # a reading belongs to the run after every gap that ended at or before it
        ends = [pd.Timestamp(g.end) for g in gaps]
        run_of = [sum(t >= e for e in ends) for t in frame["ts"]]
    frame = frame.assign(_run=run_of)
    total = len(frame)
    bucket = math.ceil(total / (TIMELINE_MAX_POINTS / 2)) if total > TIMELINE_MAX_POINTS else 1
    runs = []
    for idx, part in frame.groupby("_run", sort=True):
        shown = _thin(part, bucket)
        runs.append(TimelineRun(run_index=int(idx), start=part["ts"].iloc[0].to_pydatetime(),
                                end=part["ts"].iloc[-1].to_pydatetime(), points_total=len(part),
                                points=_points(shown)))
    shown_n = sum(len(r.points) for r in runs)
    comp = "compressor_on" in frame.columns and frame["compressor_on"].notna().any()
    amb = "ambient_c" in frame.columns and frame["ambient_c"].notna().any()
    analysis, _, _ = _analysis(rec)
    return TimelineView(
        case_id=rec.case_id, time_basis=seg.time_basis,
        start=frame["ts"].iloc[0].to_pydatetime() if total else None,
        end=frame["ts"].iloc[-1].to_pydatetime() if total else None,
        upper_threshold_c=analysis.upper_threshold_c, lower_threshold_c=analysis.lower_threshold_c,
        ambient_available=bool(amb), compressor_recorded=bool(comp), runs=runs,
        gaps=[TimelineGap(event_id=g.event_id, start=g.start, end=g.end, duration_min=g.duration_min,
                          rationale=g.rationale) for g in gaps],
        events=seg.events,
        display=DisplaySampling(downsampled=bucket > 1, method=(
            f"display only: min and max temperature per {bucket} consecutive readings, within each run"
            if bucket > 1 else "every usable reading is shown"), points_total=total, points_shown=shown_n,
            max_points=TIMELINE_MAX_POINTS),
        limitations=seg.limitations,
        provenance=provenance(rec, method="Phase 2 analysis frame + Phase 3 event segmentation"))


def _prediction_status(rec: CaseRecord) -> PredictionStatusView:
    p = rec.results.get("prediction")
    if p is None:
        return PredictionStatusView(outcome="NOT_RUN")
    if p.outcome == "REFUSED":
        return PredictionStatusView(outcome="REFUSED", refusal_code=p.refusal.code, refusal_reason=p.refusal.reason)
    return PredictionStatusView(outcome="FORECAST", reliability_status=p.reliability.status if p.reliability else None,
                                validation_scope=p.provenance.validation_scope)


def _diag_provenance(rec: CaseRecord):
    return provenance(rec, method=ANALYSIS_METHOD, method_version=DIAGNOSIS_VERSION)


def evidence_view(rec: CaseRecord) -> EvidenceView:
    _, _, d = _analysis(rec)
    return EvidenceView(
        case_id=rec.case_id, diagnosis_status=d.status, diagnosis_refusal_reason=d.refusal_reason,
        rules_validation=d.rules_validation, data_verdict=rec.results["quality_report"].verdict,
        prediction=_prediction_status(rec), observed_facts=d.observed_facts,
        document_facts=NotAvailableBlock(reason="document facts come from Phase 7 retrieval over technical "
                                                "documents, which this application does not expose; no document "
                                                "is attached to a case"),
        findings=d.findings, recommendations=d.next_checks, data_quality_limitations=d.data_quality_limitations,
        limitations=d.limitations, provenance=_diag_provenance(rec))


def next_check_view(rec: CaseRecord) -> NextCheckView:
    _, _, d = _analysis(rec)
    return NextCheckView(case_id=rec.case_id, status=d.status, refusal_reason=d.refusal_reason,
                         next_best_check=d.next_best_check, checks=d.next_checks, limitations=d.limitations,
                         provenance=_diag_provenance(rec))


def investigation_view(rec: CaseRecord) -> InvestigationView:
    _, _, d = _analysis(rec)
    return InvestigationView(case_id=rec.case_id, status=d.status, refusal_reason=d.refusal_reason,
                             hypotheses=d.hypotheses, findings=d.findings, next_checks=d.next_checks,
                             rules_validation=d.rules_validation, limitations=d.limitations,
                             provenance=_diag_provenance(rec))


def incident_view(rec: CaseRecord) -> IncidentView:
    """Delegates to the Stage 3 findings state; kept here so the service surface stays the same."""
    from . import findings as F
    return F.incident_view(rec)


def cached(rec: CaseRecord, key: str, code: ErrorCode, message: str):
    out = rec.results.get(key)
    if out is None:
        raise ApiError(404, code, message)
    return out


# ---------------------------------------------------------------- prediction (Phases 4-5)
def _recommendations(res) -> list[RecommendationView]:
    out = [RecommendationView(missing_input=r.missing_input, priority=r.priority, necessity=r.necessity,
                              rationale=r.rationale, acquisition_guidance=r.acquisition_guidance)
           for r in res.recommendations]
    if res.reliability_report is not None:
        out += [RecommendationView(missing_input=r.missing_input, priority=r.priority, necessity=r.necessity,
                                   rationale=r.rationale, acquisition_guidance=r.acquisition_guidance,
                                   retry_conditions=r.retry_conditions)
                for r in res.reliability_report.recommendations]
    return out


def predict(rec: CaseRecord, req: PredictionRequest, slots: AnalysisSlots) -> PredictionSuccess | PredictionRefusal:
    try:
        freq = ForecastRequest(horizon_min=req.horizon_min, threshold_c=req.threshold_c, direction=req.direction,
                               question=req.question, passive_since=req.passive_since)
    except ValidationError as e:
        raise ApiError(422, ErrorCode.INVALID_REQUEST, "invalid prediction request: " + "; ".join(
            x["msg"].removeprefix("Value error, ") for x in e.errors())) from None
    run = assess_source if req.assess_reliability else forecast_source
    with slots:
        try:
            _, _, rep, res = run(_stream(rec.content, rec.filename), source=rec.filename,
                                 evidence=EvidenceClass(rec.evidence), options=rec.options["ingest"], request=freq)
        except ColdSenseError as e:
            raise _unreadable(e) from None
    rr = res.reliability_report
    method = "Phase 4 thermal engine" + (" + Phase 5 reliability assessment" if rr else "")
    version = ENGINE_VERSION + (f" / {METHOD_VERSION}" if rr else "")
    common = dict(case_id=rec.case_id, request=req, data_verdict=rep.verdict, evidence_class=res.evidence,
                  question=res.question, operating_regime=res.operating_regime.value,
                  regime_evidence=res.regime_evidence, warnings=res.warnings, limitations=res.limitations,
                  recommendations=_recommendations(res))
    if res.status != ForecastStatus.FORECAST:
        out = PredictionRefusal(**common, refusal=RefusalDetail(
            code=res.refusal_code, reason=res.refusal_reason, failed_condition=res.refusal_failed_condition,
            needed_to_retry=res.refusal_needed_to_retry),
            provenance=provenance(rec, method=method, method_version=version))
    else:
        ms, t0, u = res.model_selection, res.initial_temperature, rr.uncertainty if rr else None
        out = PredictionSuccess(
            **common,
            model=SelectedModel(name=res.selected_model, version=res.selected_model_version,
                                solution_method=res.solution_method, forecast_kind=ms.forecast_kind if ms else None,
                                selection_reason=ms.selection_reason if ms else None,
                                fallback_reason=ms.fallback_reason if ms else None),
            prediction_start=res.prediction_start, horizon_min=res.prediction_horizon_min,
            validated_horizon_min=res.validated_horizon_min,
            initial_temperature=ObservedTemperature(value_c=t0.value, provenance=t0.provenance,
                                                    measured_at=res.initial_measurement_time, note=t0.note),
            threshold=None if res.threshold is None else ThresholdView(**res.threshold.model_dump()),
            crossing=CrossingView(status=res.crossing_status, time=res.predicted_crossing_time,
                                  minutes=res.predicted_crossing_minutes),
            trajectory=[TrajectoryPointView(**p.model_dump()) for p in res.trajectory],
            scenario=res.scenario, assumptions=res.assumptions,
            baselines=[BaselineView(**b.model_dump(exclude={"slope_c_per_min"})) for b in res.baseline_results],
            interval=None if u is None else IntervalView(
                status=u.status.value, horizon_min=u.horizon_min, nominal_coverage=u.nominal_coverage,
                lower_c=u.interval_lower_c, upper_c=u.interval_upper_c, conditional_on=u.conditional_on),
            crossing_interval=None if rr is None else CrossingIntervalView(
                status=rr.crossing.status.value, lower_min=rr.crossing.lower_min, upper_min=rr.crossing.upper_min),
            reliability=None if rr is None else ReliabilityView(
                status=rr.reliability.status, validation_scope=rr.reliability.validation_scope,
                method=rr.reliability.method, reason=rr.refusal_or_downgrade_reason,
                gates_failed=rr.reliability.gates_failed),
            provenance=provenance(rec, method=method, method_version=version, result_evidence=res.evidence,
                                  scope=rr.reliability.validation_scope if rr else None))
    log.info("api prediction case=%s outcome=%s code=%s evidence=%s", rec.case_id, out.outcome,
             getattr(getattr(out, "refusal", None), "code", None), res.evidence.value)
    return out


__all__ = ["AnalysisSlots", "cached", "case_status", "check_upload", "evidence_view", "headline", "incident_view",
           "ingest_options", "investigation_view", "next_check_view", "predict", "provenance", "quality_view",
           "run_analysis", "sanitize_filename", "timeline_view", "validate_upload"]
