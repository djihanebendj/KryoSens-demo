"""Copilot tools (docs/COPILOT.md §3). Each tool has a Pydantic argument schema and returns a structured
``ToolResult``. A tool wraps an EXISTING Phase 2-7 backend, or the Phase 8 findings/report backends; where no
backend exists it returns NOT_AVAILABLE. No tool reads an arbitrary path, runs code, or touches the network: the
data source, asset and stores are bound in the ``CaseContext``, and the LLM supplies only typed parameters.

Every result carries the quantities (value + unit) and identifiers it introduces, so the orchestrator can admit
exactly those numbers into the case's allowed set for the numeric-claim check. Invented data is never returned:
a missing backend is NOT_AVAILABLE, a backend refusal is REFUSED, an unexpected failure is ERROR.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from coldsense.core.evidence import EvidenceClass


class ToolStatus(str, Enum):
    OK = "OK"
    NOT_AVAILABLE = "NOT_AVAILABLE"      # no backend exists for this capability
    REFUSED = "REFUSED"                  # the backend refused (e.g. data unusable, missing required input)
    ERROR = "ERROR"                      # unexpected failure; structured, never a traceback to the user
    CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"   # a write needs trusted confirmation (findings)


class Quantity(BaseModel):
    value: float
    unit: str
    label: str = ""


class ToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    tool: str
    status: ToolStatus
    summary: str
    data: dict = Field(default_factory=dict)
    quantities: list[Quantity] = Field(default_factory=list, description="numbers this result introduces, with "
                                       "units; the only engineering values the answer may then state")
    identifiers: list[str] = Field(default_factory=list)
    provenance: str | None = Field(None, description="evidence/provenance scope of any numbers (REAL, SIMULATED, "
                                   "EXPECTED, UNVALIDATED, or a document reference)")
    error: str | None = None
    diagnostic: dict | None = Field(None, description="validated Phase 6 DiagnosticReport (JSON); the copilot "
                                  "renders it through application templates, never as LLM prose")


@dataclass
class CaseContext:
    """The bound case: which asset, which log, what evidence class, and the stores the tools may use. The LLM
    never supplies any of these; it chooses the tool and its typed parameters only."""
    asset_id: str
    source: str | Path | pd.DataFrame | None = None
    evidence: EvidenceClass = EvidenceClass.UNVALIDATED
    retriever: object | None = None               # knowledge.Retriever, for search_service_manual
    assets: dict[str, dict] = field(default_factory=dict)       # asset_id -> profile (get_asset_profile)
    findings: object | None = None                # findings.FindingStore
    model: str | None = None                      # equipment model of the asset, if known (gate/model context)
    diagnostic: object | None = None              # the latest validated Phase 6 DiagnosticReport (never LLM-authored)

    def _src(self):
        if self.source is None:
            raise _NoData()
        return self.source


class _NoData(Exception):
    pass


def _q(value, unit, label=""):
    return Quantity(value=float(value), unit=unit, label=label) if value is not None else None


def _qs(*items) -> list[Quantity]:
    return [q for q in items if q is not None]


# ---------------------------------------------------------------- argument schemas
class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NoArgs(_Args):
    pass


class AnalyzeArgs(_Args):
    threshold_c: float = Field(description="°C temperature whose crossing time is wanted")
    current_temp_c: float | None = Field(None, description="°C now; default: the last reading")


class ValidateArgs(_Args):
    pass


class SegmentArgs(_Args):
    pass


class ForecastArgs(_Args):
    horizon_min: float = Field(gt=0, description="forecast horizon, minutes")
    threshold_c: float | None = Field(None, description="°C; a crossing is reported only if set")
    passive_since: datetime | None = Field(None, description="naive time from which the room had no active cooling")


class UncertaintyArgs(ForecastArgs):
    pass


class CompareArgs(ForecastArgs):
    pass


class DiagnoseArgs(_Args):
    upper_threshold_c: float | None = Field(None, description="°C upper limit that defines an abnormal excursion")
    lower_threshold_c: float | None = None


class SearchArgs(_Args):
    query: str = Field(min_length=1, max_length=1000)


class RecordFindingArgs(_Args):
    statement: str = Field(min_length=1, max_length=1000, description="the technician's observation, verbatim")
    event_window: str | None = Field(None, description="which excursion/window it refers to")
    confirmed: bool | None = Field(None, description="IGNORED as proof of consent: trusted confirmation is "
                                   "obtained outside the LLM loop (docs/COPILOT.md §9)")


class ReportArgs(_Args):
    pass


# ---------------------------------------------------------------- backend adapters
def _evidence_label(ev: EvidenceClass) -> str:
    return ev.value


def _validate(ctx: CaseContext, _: ValidateArgs) -> ToolResult:
    from coldsense.domains.cold_room.pipeline import validate_source
    _clean, _obs, rep = validate_source(ctx._src(), source=ctx.asset_id, evidence=ctx.evidence)
    return ToolResult(tool="validate_data", status=ToolStatus.OK,
                      summary=f"verdict {rep.verdict.value}; {rep.n_rows_clean}/{rep.n_rows_raw} usable rows",
                      data={"verdict": rep.verdict.value, "evidence": rep.evidence.value, "issues":
                            [i.code for i in rep.issues], "suitable": rep.suitable_for_downstream},
                      quantities=_qs(_q(rep.n_rows_clean, "rows"), _q(rep.temp_min_c, "°C", "min temperature"),
                                     _q(rep.temp_max_c, "°C", "max temperature"),
                                     _q(rep.duration_hours, "h"), _q(rep.coverage_pct, "%", "coverage")),
                      identifiers=[ctx.asset_id], provenance=_evidence_label(rep.evidence))


def _analyze(ctx: CaseContext, a: AnalyzeArgs) -> ToolResult:
    from coldsense.domains.cold_room.pipeline import analyze
    res = analyze(ctx._src(), source=ctx.asset_id, evidence=ctx.evidence)
    pred = res.predict(a.threshold_c, a.current_temp_c)
    if pred.refused:
        return ToolResult(tool="analyze_temperature_log", status=ToolStatus.REFUSED,
                          summary="prediction refused: " + "; ".join(pred.reasons[:2]),
                          data={"reasons": pred.reasons, "evidence": pred.evidence.value},
                          provenance=_evidence_label(pred.evidence))
    return ToolResult(tool="analyze_temperature_log", status=ToolStatus.OK,
                      summary=f"~{pred.minutes_to_threshold:.1f} min to {a.threshold_c:g} °C (method {pred.method})",
                      data={"method": pred.method, "evidence": pred.evidence.value, "refused": False},
                      quantities=_qs(_q(pred.minutes_to_threshold, "min", "time to threshold"),
                                     _q(pred.low_minutes, "min", "low"), _q(pred.high_minutes, "min", "high"),
                                     _q(pred.current_temp_c, "°C", "current"), _q(a.threshold_c, "°C", "threshold")),
                      provenance=_evidence_label(pred.evidence))


def _segment(ctx: CaseContext, _: SegmentArgs) -> ToolResult:
    from coldsense.domains.cold_room.pipeline import segment_source
    _c, _o, _r, seg = segment_source(ctx._src(), source=ctx.asset_id, evidence=ctx.evidence)
    if seg.refused:
        return ToolResult(tool="segment_events", status=ToolStatus.REFUSED,
                          summary="segmentation refused: " + "; ".join(seg.reasons[:2]), data={"reasons": seg.reasons})
    return ToolResult(tool="segment_events", status=ToolStatus.OK,
                      summary=f"{len(seg.events)} events", data={"n_events": len(seg.events), "summary": seg.summary},
                      quantities=_qs(_q(len(seg.events), "events")), provenance=_evidence_label(seg.evidence))


def _forecast_call(ctx: CaseContext, a: ForecastArgs):
    from coldsense.domains.cold_room.thermal.engine import forecast_source
    from coldsense.domains.cold_room.thermal.schemas import ForecastRequest
    req = ForecastRequest(horizon_min=a.horizon_min, threshold_c=a.threshold_c, passive_since=a.passive_since)
    return forecast_source(ctx._src(), source=ctx.asset_id, evidence=ctx.evidence, request=req)


def _fit(ctx: CaseContext, a: ForecastArgs) -> ToolResult:
    _c, _o, _r, fr = _forecast_call(ctx, a)
    if fr.status.value == "REFUSED":
        return ToolResult(tool="fit_thermal_model", status=ToolStatus.REFUSED,
                          summary=f"forecast refused: {fr.refusal_reason}",
                          data={"refusal_code": fr.refusal_code.value if fr.refusal_code else None},
                          provenance=_evidence_label(fr.evidence))
    qs = [_q(p.value, p.unit, p.name) for p in fr.parameter_estimates]
    if fr.initial_temperature:
        qs.append(_q(fr.initial_temperature.value, "°C", "initial temperature"))
    return ToolResult(tool="fit_thermal_model", status=ToolStatus.OK,
                      summary=f"model {fr.selected_model}", data={"selected_model": fr.selected_model,
                      "regime": fr.operating_regime.value, "status": fr.status.value},
                      quantities=_qs(*qs), identifiers=[fr.selected_model or ""],
                      provenance=_evidence_label(fr.evidence))


def _predict_crossing(ctx: CaseContext, a: ForecastArgs) -> ToolResult:
    _c, _o, _r, fr = _forecast_call(ctx, a)
    if fr.status.value == "REFUSED":
        return ToolResult(tool="predict_threshold_crossing", status=ToolStatus.REFUSED,
                          summary=f"refused: {fr.refusal_reason}",
                          data={"refusal_code": fr.refusal_code.value if fr.refusal_code else None},
                          provenance=_evidence_label(fr.evidence))
    return ToolResult(tool="predict_threshold_crossing", status=ToolStatus.OK,
                      summary=f"crossing {fr.crossing_status.value}", data={"crossing_status": fr.crossing_status.value,
                      "selected_model": fr.selected_model},
                      quantities=_qs(_q(fr.predicted_crossing_minutes, "min", "crossing time")),
                      provenance=_evidence_label(fr.evidence))


def _uncertainty(ctx: CaseContext, a: UncertaintyArgs) -> ToolResult:
    from coldsense.domains.cold_room.thermal.reliability import assess_source
    from coldsense.domains.cold_room.thermal.schemas import ForecastRequest
    req = ForecastRequest(horizon_min=a.horizon_min, threshold_c=a.threshold_c, passive_since=a.passive_since)
    _c, _o, _r, fr = assess_source(ctx._src(), source=ctx.asset_id, evidence=ctx.evidence, request=req)
    rr = fr.reliability_report
    return ToolResult(tool="calculate_uncertainty", status=ToolStatus.OK,
                      summary=f"reliability {rr.reliability.status.value}; interval {rr.crossing.status.value}",
                      data={"reliability": rr.reliability.status.value, "scope": rr.reliability.validation_scope.value,
                            "crossing_interval": rr.crossing.status.value, "uncertainty": rr.uncertainty.status.value},
                      quantities=_qs(_q(rr.crossing.lower_min, "min", "crossing lower"),
                                     _q(rr.crossing.upper_min, "min", "crossing upper"),
                                     _q(rr.crossing.empirical_coverage, "", "coverage")),
                      provenance=rr.reliability.validation_scope.value)


def _compare(ctx: CaseContext, a: CompareArgs) -> ToolResult:
    _c, _o, _r, fr = _forecast_call(ctx, a)
    if fr.status.value == "REFUSED":
        return ToolResult(tool="compare_with_baseline", status=ToolStatus.REFUSED,
                          summary=f"refused: {fr.refusal_reason}", provenance=_evidence_label(fr.evidence))
    comps = [{"role": c.role, "horizon_min": c.horizon_min, "physical_method": c.physical_method,
              "best_baseline": c.best_baseline, "physical_better": c.physical_better,
              "baseline_better": c.baseline_better} for c in fr.model_comparison]
    return ToolResult(tool="compare_with_baseline", status=ToolStatus.OK,
                      summary=f"{len(comps)} comparison(s) vs baselines", data={"comparisons": comps,
                      "selected_model": fr.selected_model}, provenance=_evidence_label(fr.evidence))


def _diagnose(ctx: CaseContext, a: DiagnoseArgs) -> ToolResult:
    from coldsense.domains.cold_room.diagnostics import diagnose_source
    kw = {}
    if a.upper_threshold_c is not None or a.lower_threshold_c is not None:
        from coldsense.domains.cold_room.schemas import SegmentationParams
        kw["params"] = SegmentationParams(upper_threshold_c=a.upper_threshold_c, lower_threshold_c=a.lower_threshold_c)
    rep, seg, _res, diag = diagnose_source(ctx._src(), source=ctx.asset_id, evidence=ctx.evidence, **kw)
    from coldsense.copilot.diagnostics import DiagnosticValidationError, validate_report
    try:
        validated = validate_report(diag)
    except DiagnosticValidationError as e:
        return ToolResult(tool="rank_diagnostic_checks", status=ToolStatus.ERROR,
                          summary="the diagnostic report failed application validation; nothing is shown",
                          error=f"INVALID_DIAGNOSTIC_REPORT: {e}")
    checks = [{"rank": c.rank, "check_id": c.check_id, "action": c.action, "safety": c.safety.value}
              for c in validated.next_checks]
    return ToolResult(tool="rank_diagnostic_checks", status=ToolStatus.OK,
                      summary=f"status {validated.status.value}; {len(checks)} ranked check(s)",
                      data={"status": validated.status.value, "refusal_reason": validated.refusal_reason,
                            "next_checks": checks, "next_best_check": validated.next_best_check},
                      quantities=_qs(_q(len(checks), "checks")), provenance=_evidence_label(validated.data_evidence),
                      identifiers=[c.check_id for c in validated.next_checks],
                      diagnostic=validated.model_dump(mode="json"))


def _search(ctx: CaseContext, a: SearchArgs) -> ToolResult:
    if ctx.retriever is None:
        return ToolResult(tool="search_service_manual", status=ToolStatus.NOT_AVAILABLE,
                          summary="no knowledge base is attached to this case")
    from coldsense.copilot.evidence import answer_to_result
    return answer_to_result(ctx.retriever.answer(a.query), ctx.model)


def _get_asset_profile(ctx: CaseContext, _: NoArgs) -> ToolResult:
    profile = ctx.assets.get(ctx.asset_id)
    if profile is None:
        return ToolResult(tool="get_asset_profile", status=ToolStatus.NOT_AVAILABLE,
                          summary=f"no profile is registered for asset {ctx.asset_id}")
    return ToolResult(tool="get_asset_profile", status=ToolStatus.OK,
                      summary=f"asset {ctx.asset_id}: {profile.get('manufacturer', '?')} {profile.get('model', '?')}",
                      data=dict(profile), identifiers=[ctx.asset_id, str(profile.get("model") or "")])


def _record_finding(ctx: CaseContext, a: RecordFindingArgs) -> ToolResult:
    if ctx.findings is None:
        return ToolResult(tool="record_technician_finding", status=ToolStatus.NOT_AVAILABLE,
                          summary="no findings store is attached to this case")
    draft = ctx.findings.draft(asset_id=ctx.asset_id, statement=a.statement, event_window=a.event_window)
    return ToolResult(tool="record_technician_finding", status=ToolStatus.CONFIRMATION_REQUIRED,
                      summary="a PENDING draft was created; a write needs trusted confirmation outside the LLM loop",
                      data={"draft_id": draft.draft_id, "content_hash": draft.content_hash,
                            "note": "the LLM-supplied 'confirmed' flag is ignored as proof of consent"})


def _generate_report(ctx: CaseContext, _: ReportArgs) -> ToolResult:
    if ctx.findings is None:
        return ToolResult(tool="generate_service_report", status=ToolStatus.NOT_AVAILABLE,
                          summary="no findings store is attached to this case")
    from coldsense.copilot.report import build_report
    return build_report(ctx)


@dataclass(frozen=True)
class Tool:
    name: str
    args_model: type[_Args]
    fn: Callable[[CaseContext, _Args], ToolResult]
    description: str


TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("validate_data", ValidateArgs, _validate, "data-quality gate on the case log"),
    Tool("analyze_temperature_log", AnalyzeArgs, _analyze, "naive time-to-threshold from observed warming"),
    Tool("segment_events", SegmentArgs, _segment, "detect warming/cooling/door/defrost events"),
    Tool("fit_thermal_model", ForecastArgs, _fit, "fit the physics-aware thermal model"),
    Tool("predict_threshold_crossing", ForecastArgs, _predict_crossing, "predict a threshold crossing time"),
    Tool("calculate_uncertainty", UncertaintyArgs, _uncertainty, "reliability gate and calibrated interval"),
    Tool("compare_with_baseline", CompareArgs, _compare, "compare the physical model with baselines"),
    Tool("rank_diagnostic_checks", DiagnoseArgs, _diagnose, "rank next-best diagnostic checks"),
    Tool("search_service_manual", SearchArgs, _search, "cited passages from technical documentation"),
    Tool("get_asset_profile", NoArgs, _get_asset_profile, "the asset's manufacturer, model and location"),
    Tool("record_technician_finding", RecordFindingArgs, _record_finding, "draft a technician finding (PENDING)"),
    Tool("generate_service_report", ReportArgs, _generate_report, "a service report from confirmed records only"),
]}


def tool_schemas() -> list[dict]:
    """JSON-serialisable schemas offered to the LLM (name, description, argument JSON schema)."""
    return [{"name": t.name, "description": t.description, "parameters": t.args_model.model_json_schema()}
            for t in TOOLS.values()]


def run_tool(name: str, raw_args: dict, ctx: CaseContext) -> ToolResult:
    """Validate the arguments against the tool's schema and run it. Unknown tool or invalid arguments become a
    structured ERROR result; a backend exception becomes an ERROR result (never a traceback to the user)."""
    tool = TOOLS.get(name)
    if tool is None:
        return ToolResult(tool=name, status=ToolStatus.ERROR, summary=f"unknown tool {name!r}",
                          error="NOT_IN_ALLOWLIST")
    try:
        args = tool.args_model.model_validate(raw_args)
    except Exception as e:                                        # noqa: BLE001 - reported as a structured error
        return ToolResult(tool=name, status=ToolStatus.ERROR, summary="invalid arguments",
                          error=f"INVALID_ARGUMENTS: {type(e).__name__}")
    try:
        return tool.fn(ctx, args)
    except _NoData:
        return ToolResult(tool=name, status=ToolStatus.NOT_AVAILABLE, summary="no log is attached to this case")
    except Exception as e:                                        # noqa: BLE001 - never surface a traceback
        return ToolResult(tool=name, status=ToolStatus.ERROR, summary="the tool backend failed",
                          error=f"BACKEND_ERROR: {type(e).__name__}")
