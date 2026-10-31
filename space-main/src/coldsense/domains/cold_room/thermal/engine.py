"""Thermal forecasting engine: validated data -> forecast or structured refusal.

Flow (each step can end in a refusal that keeps everything gathered so far):

    VALIDATED DATA (Phase 2) + EVENTS (Phase 3)
    -> OPERATING-REGIME EVIDENCE (recorded compressor state, or a declared passive period)
    -> SEGMENTS (accepted / rejected with reasons; chronological FIT / SELECT / TEST split)
    -> CANDIDATE ELIGIBILITY (required inputs per model)
    -> PARAMETER FITTING on FIT data + IDENTIFIABILITY + plausibility + fit quality
    -> INDEPENDENT VALIDATION on SELECT data, against both baselines, same cases
    -> SELECTION (lowest held-out error at the requested horizon) + adequacy gate
    -> FINAL EVALUATION on TEST data (refit on FIT+SELECT; gate only, never tuning)
    -> final refit on all eligible data before the origin -> trajectory + crossing

The forecast question is fixed and stated: the room temperature over the
horizon IF THERE IS NO ACTIVE COOLING from the origin (power cut, compressor
failure, or the current off-cycle continuing). Baselines extrapolate the
observed regime, so they can answer that question only when the room is
already passive at the origin.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from coldsense.core.evidence import EvidenceClass, derived_evidence

from ..events import segment_events
from ..schemas import DataQualityReport, SegmentationParams, ValidationLimits, Verdict
from . import advice
from . import evaluation as EV
from . import limits as L
from . import registry as R
from .baselines import persistence_law, recent_slope, slope_law
from .models import FitOutcome, FitSegment, Law, fit_first_order
from .schemas import (
    BASELINE_PREDICTION_FIELDS,
    BaselineForecast,
    CandidateOutcome,
    CandidateResult,
    CrossingStatus,
    FitDiagnostics,
    ForecastQuestion,
    ForecastRequest,
    ForecastResult,
    ForecastStatus,
    Identifiability,
    MethodEvaluation,
    ParameterEstimate,
    Provenance,
    Quantity,
    RefusalCode,
    Regime,
    RegimeBasis,
    RejectedModel,
    Threshold,
    TrajectoryPoint,
    ValidationDiagnostics,
)
from .selection import (
    ACTIVE_RECORDED,
    ANY_REGIME,
    PASSIVE_RECORDED,
    Frame,
    Piece,
    all_pieces,
    apply_events,
    assign_roles,
    candidate_pieces,
    check_piece,
    chronological_cuts,
    origin_index,
    power_windows,
    regime_at_origin,
    regime_from_power,
)

log = logging.getLogger(__name__)

ENGINE_VERSION = "thermal-engine-1"
PHYSICAL_PASSIVE = (R.FO_ASSUMED, R.FO_MEASURED, R.FO_FITTED, R.FO_VARYING)
BASELINES = (R.PERSISTENCE, R.RECENT_SLOPE)
# simplest first: no parameter; one slope; one rate (equilibrium given); rate + equilibrium; rate + forcing series
COMPLEXITY = (R.PERSISTENCE, R.RECENT_SLOPE, R.FO_ASSUMED, R.FO_MEASURED, R.FO_FITTED, R.FO_VARYING)

_CROSS_FROM_REFUSAL = {
    RefusalCode.INVALID_REQUEST: CrossingStatus.INVALID_REQUEST,
    RefusalCode.MISSING_REQUIRED_INPUT: CrossingStatus.MISSING_REQUIRED_INPUT,
    RefusalCode.MODEL_NOT_APPLICABLE: CrossingStatus.MODEL_NOT_APPLICABLE,
    RefusalCode.INSUFFICIENT_IDENTIFIABILITY: CrossingStatus.INSUFFICIENT_IDENTIFIABILITY,
}

LIMITATIONS = [
    "Answers one question: room-air temperature at the sensor if there is NO active cooling from the origin. "
    "It is not a forecast of normal thermostat cycling.",
    "Parameters are EFFECTIVE values for this room and period; they are not measured physical properties and "
    "do not transfer to another room.",
    "Checked on SIMULATED data only; no real-room accuracy is claimed. Outputs from real logs are EXPECTED, "
    "never REAL, until an outcome confirms them.",
    "No calibrated prediction interval is given (Phase 5). Profile ranges are identifiability checks only.",
    "Door openings, loading, defrost and internal loads are not modelled (no such inputs); they appear only "
    "as unexplained disturbances or as fit error.",
]
EVENTS_LIMITATION = (
    "Door openings, loading, defrost and internal loads are not modelled. Readings inside recorded event "
    "intervals are only excluded from fitting and scoring; unrecorded ones still appear as unexplained "
    "disturbances or as fit error.")
CONTINUATION_QUESTION = (
    "Answers one question: room-air temperature at the sensor if the behaviour observed at the origin continues "
    "(CONTINUATION). From a passive origin this uses the passive-exchange models; otherwise only the baselines "
    "apply, and they are not physical models.")


# ====================================================================== helpers
class _Out:
    """Accumulates result fields so a refusal at any step still carries the evidence gathered."""

    def __init__(self, report: DataQualityReport, request: ForecastRequest):
        self.kw: dict = {
            "evidence": derived_evidence(report.evidence),
            "provenance": {
                "engine": ENGINE_VERSION,
                "source": report.source,
                "input_sha256": report.input.sha256 if report.input else None,
                "data_evidence_class": report.evidence.value,
                "data_verdict": report.verdict.value,
                "time_basis": report.input.timezone_handling if report.input else None,
                "request": json.dumps(request.model_dump(mode="json"), sort_keys=True),
            },
            "prediction_horizon_min": request.horizon_min,
            "limitations": list(LIMITATIONS) if request.question == ForecastQuestion.NO_COOLING
            else [CONTINUATION_QUESTION, *LIMITATIONS[1:]],
            "warnings": [],
            "assumptions": [],
        }
        if request.threshold_c is not None:
            self.kw["threshold"] = Threshold(value_c=request.threshold_c, direction=request.direction)
        self.request = request
        self.ctx: dict = {"request": request, "report": report}     # what advice.py reads; never numbers it recomputes
        self.forecast_only: list[str] = []      # warnings that describe an issued forecast; dropped on refusal
        self.sink: dict | None = None           # Phase 5 hand-off of held-out evidence; never part of the result

    def keep(self, **kw) -> None:
        """Hand internal evidence (held-out cases, final fit) to a Phase 5 caller; a no-op otherwise."""
        if self.sink is not None:
            self.sink.update(kw)

    def forecast_warning(self, msg: str) -> None:
        """A warning that only makes sense if a forecast is issued (it names the selected model, the forecast's
        reach or its values). Kept in place for a forecast; a later refusal drops it."""
        self.kw["warnings"].append(msg)
        self.forecast_only.append(msg)

    def refuse(self, code: RefusalCode, reason: str, failed: str, needed: str | None = None) -> ForecastResult:
        kw = dict(self.kw)
        for forecast_only in ("trajectory", "selected_model", "selected_model_version", "solution_method",
                              "parameter_estimates", "parameter_units", "scenario", "assumptions",
                              "predicted_crossing_time", "predicted_crossing_minutes"):
            kw.pop(forecast_only, None)          # a refusal presents no forecast and no model as selected
        kw["warnings"] = [w for w in kw.get("warnings", []) if w not in self.forecast_only]
        if kw.get("baseline_results"):           # applicability and held-out errors stay; their predictions do not
            kw["baseline_results"] = [b.model_copy(update=dict.fromkeys(BASELINE_PREDICTION_FIELDS))
                                      for b in kw["baseline_results"]]
        crossing = (CrossingStatus.NO_THRESHOLD if self.request.threshold_c is None
                    else _CROSS_FROM_REFUSAL.get(code, CrossingStatus.INSUFFICIENT_EVIDENCE))
        kw.update(advice.annotate(self.ctx, True, code, reason))
        return ForecastResult(status=ForecastStatus.REFUSED, crossing_status=crossing, refusal_code=code,
                              refusal_reason=reason, refusal_failed_condition=failed,
                              refusal_needed_to_retry=needed, **kw)


def _fit_segment(variant: str, p: Piece, f: Frame, request: ForecastRequest) -> tuple[FitSegment | None, str | None]:
    rows = p.rows
    t = f.t[rows] - f.t[rows[0]]
    y = f.y[rows]
    sid = p.seg_id
    if variant == R.FO_FITTED:
        return FitSegment(sid, t, y, "param", eq_param="E"), None
    if variant == R.FO_ASSUMED:
        return FitSegment(sid, t, y, "known", eq_value=float(request.assumed_equilibrium_c)), None
    a = f.amb[rows]
    ok = np.isfinite(a)
    if variant in (R.FO_MEASURED, R.FO_ACTIVE) and not (variant == R.FO_ACTIVE and p.regime == ACTIVE_RECORDED):
        steady = bool(ok.any()) and ok.mean() >= L.AMBIENT_MIN_COVERAGE and np.ptp(a[ok]) <= L.AMBIENT_STEADY_RANGE_C
        if steady:
            return FitSegment(sid, t, y, "known", eq_value=float(np.mean(a[ok]))), None
        if variant == R.FO_ACTIVE:
            return FitSegment(sid, t, y, "param", eq_param="E_off"), None
        if not ok.any() or ok.mean() < L.AMBIENT_MIN_COVERAGE:
            return None, f"ambient present in {100 * ok.mean():.0f}% of rows; need {100 * L.AMBIENT_MIN_COVERAGE:.0f}%"
        return None, (f"ambient varies by {np.ptp(a[ok]):.2f} °C within the segment (> {L.AMBIENT_STEADY_RANGE_C} °C); "
                      "only the time-varying variant may use it")
    if variant == R.FO_ACTIVE:                         # recorded ON run
        return FitSegment(sid, t, y, "param", eq_param="E_on"), None
    # time-varying ambient: bridge only short holes, never the ends
    if not (ok[0] and ok[-1]):
        return None, "ambient missing at the first or last reading of the segment"
    tv = t[ok]
    if len(tv) > 1 and np.max(np.diff(tv)) > L.AMBIENT_MAX_INTERP_FACTOR * f.med_s / 60:
        return None, (f"ambient missing for {np.max(np.diff(tv)):.1f} min; holes are bridged only up to "
                      f"{L.AMBIENT_MAX_INTERP_FACTOR:g} median intervals")
    return FitSegment(sid, t, y, "series", amb=np.interp(t, tv, a[ok])), None


def _diagnostics(name: str, role: str, segs: list[FitSegment], fo: FitOutcome, f: Frame,
                 variant: str, request: ForecastRequest) -> FitDiagnostics:
    res = fo.residuals
    allr = np.concatenate(res) if res else np.array([])
    sst = sum(float(((s.y - s.y.mean()) ** 2).sum()) for s in segs)
    sse = float(allr @ allr) if len(allr) else None
    trend = None
    if len(allr) >= 3:
        u = np.concatenate([s.t / s.t[-1] for s in segs])
        uc = u - u.mean()
        trend = float(uc @ (allr - allr.mean()) / (uc @ uc)) if (uc @ uc) > 0 else None
    ident = Identifiability.IDENTIFIABLE if fo.identifiable else Identifiability.INSUFFICIENT_IDENTIFIABILITY
    params = [ParameterEstimate(
        name="k", unit="1/min", value=fo.k, provenance=Provenance.FITTED,
        method="1-D search over log k (grid + bounded refinement); linear parameters by least squares",
        objective="sum of squared room-temperature errors over the fitted segments (output error)",
        constraints=f"k searched in [{fo.k_bounds[0]:.3g}, {fo.k_bounds[1]:.3g}] /min: time constant between one "
                    f"sampling interval and {L.K_MIN_SPAN_FACTOR:g}x the fitted time span",
        identifiability=ident, profile_range=fo.k_range,
        note="effective exchange rate of this room in this period; not a measured property")]
    if fo.k:
        rng = None if fo.k_range is None else (1 / fo.k_range[1], 1 / fo.k_range[0])
        params.append(ParameterEstimate(name="tau", unit="min", value=1 / fo.k, provenance=Provenance.DERIVED,
                                        method="1 / k", identifiability=ident, profile_range=rng))
    for nm, v in fo.eq.items():
        label = {"E": "effective equilibrium (fitted; NOT a measured ambient)", "E_on": "equilibrium while the "
                 "compressor is recorded ON (fitted)", "E_off": "equilibrium while OFF (fitted)"}[nm]
        params.append(ParameterEstimate(name=nm, unit="°C", value=v, provenance=Provenance.FITTED,
                                        method="least squares for fixed k", identifiability=ident,
                                        profile_range=fo.eq_range.get(nm), note=label))
    known = [s.eq_value for s in segs if s.eq_kind == "known"]
    if known and variant == R.FO_MEASURED:
        params.append(ParameterEstimate(name="E", unit="°C", value=float(np.mean(known)),
                                        provenance=Provenance.DERIVED, method="per-segment mean of measured ambient",
                                        note=f"ambient means {min(known):.2f}-{max(known):.2f} °C over "
                                             f"{len(known)} segment(s); OBSERVED readings, used as equilibrium "
                                             "under an ASSUMPTION"))
    if known and variant == R.FO_ASSUMED:
        params.append(ParameterEstimate(name="E", unit="°C", value=float(request.assumed_equilibrium_c),
                                        provenance=Provenance.ASSUMED, method="given by the user"))
    if variant == R.FO_ACTIVE and fo.k and "E_on" in fo.eq:
        e_off = fo.eq.get("E_off", float(np.mean(known)) if known else None)
        if e_off is not None:
            params.append(ParameterEstimate(name="b", unit="°C/min", value=fo.k * (e_off - fo.eq["E_on"]),
                                            provenance=Provenance.DERIVED, method="k (E_off - E_on)",
                                            note="effective cooling effect while ON under u in {0,1}; NOT a cooling "
                                                 "capacity or power"))
    warns = []
    if fo.rho is not None and fo.rho > L.RESID_AUTOCORR_WARN:
        warns.append(f"residuals are strongly autocorrelated (lag-1 {fo.rho:.2f}): systematic misfit or "
                     "unmodelled dynamics")
    return FitDiagnostics(
        model=name, data_role=role, n_observations=fo.n, n_segments=len(segs),
        fit_duration_min=round(sum(float(s.t[-1]) for s in segs), 4),
        median_interval_s=f.med_s, n_parameters=fo.p, n_effective=fo.n_eff, parameters=params,
        residual_mean_c=float(allr.mean()) if len(allr) else None,
        residual_std_c=float(allr.std(ddof=1)) if len(allr) > 1 else None,
        mae_c=float(np.abs(allr).mean()) if len(allr) else None,
        rmse_c=float(np.sqrt((allr ** 2).mean())) if len(allr) else None,
        r2_within_segment=(1 - sse / sst) if sst > 0 and sse is not None else None,
        residual_lag1_autocorr=fo.rho, residual_trend_c=trend, fit_status=fo.status,
        identifiability=ident, identifiability_reason=fo.ident_reason or fo.message, warnings=warns)


def _fit_variant(variant: str, pieces: list[Piece], f: Frame, request: ForecastRequest, limits: ValidationLimits,
                 role: str) -> tuple[FitOutcome | None, FitDiagnostics | None, tuple[RefusalCode, str] | None]:
    segs, whys = [], []
    for p in pieces:
        if len(p.rows) < 3:
            continue
        s, why = _fit_segment(variant, p, f, request)
        if s is not None:
            segs.append(s)
        else:
            whys.append(f"{p.seg_id}: {why}")
    if not segs:
        if whys:
            return None, None, (RefusalCode.MODEL_NOT_APPLICABLE,
                                f"no fitting segment meets this model's input requirements ({'; '.join(whys[:3])}"
                                + (f"; and {len(whys) - 3} more" if len(whys) > 3 else "") + ")")
        return None, None, (RefusalCode.INSUFFICIENT_DATA, "no fitting segment with at least 3 readings")
    k_hi = 60.0 / f.med_s
    span = sum(float(s.t[-1]) for s in segs)
    k_lo = 1.0 / (L.K_MIN_SPAN_FACTOR * span)
    try:
        fo = fit_first_order(segs, k_lo, k_hi)
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as e:
        return None, None, (RefusalCode.NUMERICAL_ERROR, f"fit failed numerically: {e}")
    if fo.status == "FAILED" or fo.k is None:
        return fo, None, (RefusalCode.FIT_FAILED, fo.message or "fit failed")
    diag = _diagnostics(variant, role, segs, fo, f, variant, request)
    # a model that does not describe the data is rejected for that, before asking whether it is identified
    if diag.r2_within_segment is None or diag.r2_within_segment < L.FIT_MIN_R2:
        r2 = "undefined" if diag.r2_within_segment is None else f"{diag.r2_within_segment:.3f}"
        return fo, diag, (RefusalCode.POOR_FIT, f"within-segment R² {r2} < {L.FIT_MIN_R2}: the model does not "
                                                "describe the fitted data")
    if not fo.identifiable:
        return fo, diag, (RefusalCode.INSUFFICIENT_IDENTIFIABILITY, fo.ident_reason)
    for nm, v in fo.eq.items():
        # E and E_off stand for the surroundings and must be plausible surroundings; E_on is the virtual
        # target of the cooled system (T_amb - b/k), checked through b > 0 below instead
        if nm != "E_on" and not limits.ambient_min_c <= v <= limits.ambient_max_c:
            return fo, diag, (RefusalCode.IMPLAUSIBLE_PARAMETERS,
                              f"fitted {nm} = {v:.1f} °C lies outside the plausible surroundings range "
                              f"{limits.ambient_min_c}..{limits.ambient_max_c} °C: not physically interpretable")
    if variant == R.FO_ACTIVE:
        b = next((p.value for p in diag.parameters if p.name == "b"), None)
        if b is None or b <= 0:
            return fo, diag, (RefusalCode.IMPLAUSIBLE_PARAMETERS,
                              "fitted cooling effect b <= 0: the ON runs do not cool relative to the OFF runs")
    return fo, diag, None


def _last_ambient(f: Frame, first: int, i0: int) -> tuple[float | None, int | None]:
    a = f.amb[first:i0 + 1]
    ok = np.flatnonzero(np.isfinite(a))
    if not len(ok):
        return None, None
    return float(a[ok[-1]]), first + int(ok[-1])


def _physical_predictor(variant: str, fo: FitOutcome, f: Frame, request: ForecastRequest) -> EV.Predictor:
    def pred(p: Piece, i0: int) -> tuple[Law | None, str | None]:
        if variant == R.FO_FITTED:
            e = fo.eq["E"]
        elif variant == R.FO_ASSUMED:
            e = float(request.assumed_equilibrium_c)
        elif variant == R.FO_ACTIVE and p.regime == ACTIVE_RECORDED:
            e = fo.eq["E_on"]
        elif variant == R.FO_ACTIVE and "E_off" in fo.eq:
            e = fo.eq["E_off"]
        else:   # measured / varying / active-OFF: ambient at the origin, held constant (no future data)
            e, _ = _last_ambient(f, p.parent_first, i0)
            if e is None:
                return None, "no ambient reading at or before the origin"
        return Law("exponential", float(f.y[i0]), eq=e, k=fo.k), None
    return pred


def _baseline_predictors(f: Frame) -> dict[str, EV.Predictor]:
    def pers(p: Piece, i0: int):
        return persistence_law(float(f.y[i0])), None

    def slope(p: Piece, i0: int):
        fit, why = recent_slope(f.t, f.y, i0, p.parent_first)
        return (None, why) if fit is None else (slope_law(float(f.y[i0]), fit), None)
    return {R.PERSISTENCE: pers, R.RECENT_SLOPE: slope}


def _evaluate(f: Frame, pieces: list[Piece], methods: dict[str, EV.Predictor], horizons: list[float],
              request: ForecastRequest) -> list[EV.Case]:
    stride = max(L.EVAL_ORIGIN_STRIDE_MIN, f.med_s / 60)
    cases = EV.build_cases(f, pieces, horizons, stride, request.threshold_c, request.direction)
    EV.run(cases, methods, request.threshold_c, request.direction)
    return cases


def _method_evals(cases: list[EV.Case], methods: list[str], horizons: list[float], role: str, regime: str,
                  fitted_on: str) -> list[MethodEvaluation]:
    return [MethodEvaluation(method=m, role=role, regime=regime, fitted_on=fitted_on,
                             horizons=[EV.horizon_metrics(cases, m, h, methods) for h in horizons])
            for m in methods]


def _common_mae(cases: list[EV.Case], methods: list[str], h: float) -> tuple[int, dict[str, float]]:
    common = [c for c in cases if c.horizon == h and all(c.preds.get(m) is not None for m in methods)]
    return len(common), {m: float(np.mean([abs(c.preds[m] - c.truth) for c in common])) for m in methods} \
        if common else {}


def _physical_summary(cands: dict[str, CandidateResult]) -> str:
    return "; ".join(f"{c.name}: {c.rejection_code.value} ({c.rejection_reason})" for c in cands.values()
                     if c.kind == "PHYSICAL" and c.outcome == CandidateOutcome.REJECTED and c.rejection_code
                     and c.rejection_code not in (RefusalCode.MISSING_REQUIRED_INPUT,
                                                  RefusalCode.MODEL_NOT_APPLICABLE))


# ====================================================================== entry points
def forecast(clean: pd.DataFrame, report: DataQualityReport, observations: pd.DataFrame | None,
             request: ForecastRequest, seg_params: SegmentationParams | None = None, *,
             events: list | None = None, metadata=None, sink: dict | None = None) -> ForecastResult:
    """``events`` (inputs.OperationalEvent) and ``metadata`` (inputs.RoomMetadata) are optional Phase 4
    adapters; without them the result is the same as before they existed, plus the advisory fields.
    ``sink`` (Phase 5): a dict that receives the held-out evaluation cases and the final fit, so the
    reliability layer reuses this run's evidence instead of re-fitting; it never changes the result."""
    out = _Out(report, request)
    out.sink = sink
    out.ctx.update(events=list(events or []), metadata=metadata)
    if events:
        out.kw["provenance"]["event_records"] = len(events)
        out.kw["limitations"][-1] = EVENTS_LIMITATION
    if metadata is not None:
        out.kw["provenance"]["metadata_fields"] = ",".join(metadata.supplied()) or "none"
    try:
        return _forecast(clean, report, observations, request, seg_params, out)
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as e:    # last line of defence
        log.debug("thermal engine numerical failure", exc_info=True)
        return out.refuse(RefusalCode.NUMERICAL_ERROR, f"numerical failure: {e}", "a finite numerical result",
                          "report this log; the failure is not expected for validated data")


def _forecast(clean, report, obs, request, seg_params, out: _Out) -> ForecastResult:
    for name in ("origin", "passive_since"):
        v = getattr(request, name)
        if v is not None and v.tzinfo is not None:
            return out.refuse(RefusalCode.INVALID_REQUEST, f"{name} carries a time zone",
                              f"{name} in the log's own time basis (naive)",
                              f"give {name} without a zone, as in the log ({out.kw['provenance']['time_basis']})")
    if report.verdict == Verdict.REFUSED:
        return out.refuse(RefusalCode.DATA_REFUSED,
                          "validation refused this log: " + "; ".join(i.message for i in report.blocking),
                          "a log that passes Phase 2 validation", "fix the blocking data-quality issues")
    seg = segment_events(clean, report, obs, seg_params)
    if seg.refused:
        return out.refuse(RefusalCode.DATA_REFUSED, "event segmentation refused: " + "; ".join(seg.reasons),
                          "Phase 3 segmentation", None)
    f = Frame.build(clean, obs, report, seg)
    limits = report.limits_used or ValidationLimits()
    out.kw["input_inventory"] = R.audit_inputs(clean, report, seg)
    out.ctx["inventory"] = out.kw["input_inventory"]
    events = out.ctx["events"]
    if events:
        out.ctx["event_notes"] = apply_events(f, events)
    power = power_windows(f, events) if events else []
    out.kw["rejected_models"] = [RejectedModel(name=m.name, code=R.DEFERRED, reason="needs: " + "; ".join(
        m.missing_for_implementation)) for m in R.REGISTRY if m.status == R.DEFERRED]

    i0, why = origin_index(f, request)
    if i0 is None:
        return out.refuse(RefusalCode.INVALID_REQUEST, why, "a usable observation at the origin",
                          "choose an origin inside the log")
    t0 = f.ts.iloc[i0].to_pydatetime()
    out.kw.update(prediction_start=t0, initial_measurement_time=t0,
                  initial_temperature=Quantity(value=float(f.y[i0]), unit="°C", provenance=Provenance.OBSERVED,
                                               note=f"usable reading at line {int(f.lines[i0])}"))
    regime, basis, evidence, contradiction = regime_at_origin(f, seg, i0, request)
    if power and request.passive_since is None and not contradiction:
        p_regime, p_evidence, conflict = regime_from_power(f, i0, power)
        if conflict:
            out.kw.update(operating_regime=Regime.UNKNOWN, regime_evidence=evidence)
            out.ctx.update(regime=Regime.UNKNOWN, basis=basis)
            return out.refuse(RefusalCode.INPUT_CONFLICT, conflict, "event log and compressor state consistent",
                              "check the event log times and the compressor column")
        if p_regime is not None:
            if regime != Regime.PASSIVE:
                basis = RegimeBasis.RECORDED_EVENT
                evidence = [e.replace(": the regime is not known", "; the regime is taken from the event log")
                            for e in evidence]
            regime, evidence = Regime.PASSIVE, p_evidence + evidence
    out.kw.update(operating_regime=regime, regime_evidence=evidence)
    out.ctx.update(regime=regime, basis=basis)
    if contradiction:
        return out.refuse(RefusalCode.INVALID_REQUEST, contradiction, "a declared passive period without "
                          "recorded active cooling", "correct passive_since or check the compressor log")
    if request.question == ForecastQuestion.CONTINUATION and regime != Regime.PASSIVE:
        return _continuation(f, i0, request, out, t0)

    # ---------------------------------------------------------------- segments
    passive, active, dropped = candidate_pieces(f, seg, i0, request, basis, power)
    _report_dropped(out, f, dropped)
    acc_p, acc_a, rejected = [], [], []
    for p in passive + active:
        why = check_piece(f, p)
        (rejected if why else (acc_p if p in passive else acc_a)).append((p, why))
    out.kw["segments_rejected"] = [p.record(f, False, why) for p, why in rejected]
    acc_p = [p for p, _ in acc_p]
    acc_a = [p for p, _ in acc_a]
    if not passive:
        out.ctx["no_passive_evidence"] = True
        need = ("a compressor_state column with recorded OFF periods, or passive_since (--passive-since) "
                "declaring when cooling stopped")
        if basis == RegimeBasis.RECORDED_COMPRESSOR_STATE:
            msg = "the compressor is never recorded OFF before the origin"
        else:
            msg = "there is no evidence of any period without active cooling"
        return out.refuse(RefusalCode.MISSING_REQUIRED_INPUT, msg + "; the physical models need one",
                          "operating-regime evidence for a passive period", need)
    if not acc_p:
        return out.refuse(RefusalCode.INSUFFICIENT_DATA,
                          f"none of the {len(passive)} passive segment(s) meets the fitting requirements",
                          f">= {L.SEG_MIN_POINTS} readings, >= {L.SEG_MIN_DURATION_MIN:g} min and >= "
                          f"{L.SEG_MIN_CHANGE_C} °C change in a passive segment",
                          "a longer passive period (see segments_rejected)")
    cuts = chronological_cuts(f, acc_p)
    roles_p = assign_roles(f, acc_p, cuts)
    roles_a = assign_roles(f, acc_a, cuts) if acc_a else []
    out.kw["segments_used"] = [p.record(f, True, None) for p in roles_p + roles_a]
    t_of = lambda m: (f.ts.iloc[0] + pd.to_timedelta(m, unit="min")).isoformat()   # noqa: E731
    split = (f"chronological over {sum(p.duration(f) for p in acc_p):.1f} min of accepted passive data: FIT up to "
             f"{t_of(cuts[0])}, SELECT up to {t_of(cuts[1])}, TEST after; nothing after the origin is used")
    by_role = {r: [p for p in roles_p if p.role == r] for r in ("FIT", "SELECT", "TEST")}
    out.keep(path="main", frame=f, cuts=cuts, test_pieces=by_role["TEST"], basis=basis, origin_regime=regime,
             i0=i0)

    # ---------------------------------------------------------------- candidates
    amb_varies = f.has_amb and any(
        np.isfinite(f.amb[p.rows]).any() and np.ptp(f.amb[p.rows][np.isfinite(f.amb[p.rows])]) >
        L.AMBIENT_STEADY_RANGE_C for p in acc_p)
    eligibility: dict[str, tuple[bool, list[str], str]] = {
        R.FO_FITTED: (True, [], ""),
        R.FO_MEASURED: (f.has_amb and report.ambient_available,
                        [] if f.has_amb and report.ambient_available else ["ambient temperature"],
                        "no usable ambient temperature"),
        R.FO_ASSUMED: (request.assumed_equilibrium_c is not None,
                       [] if request.assumed_equilibrium_c is not None else ["assumed equilibrium"],
                       "no assumed equilibrium given (assumed_equilibrium_c / --assume-equilibrium)"),
        R.FO_VARYING: (bool(amb_varies), [] if f.has_amb else ["ambient temperature"],
                       "no ambient temperature" if not f.has_amb else
                       f"ambient was steady (<= {L.AMBIENT_STEADY_RANGE_C} °C) within every segment: the "
                       "constant-ambient variant covers this case"),
        R.FO_ACTIVE: (basis in (RegimeBasis.RECORDED_COMPRESSOR_STATE, RegimeBasis.RECORDED_EVENT) and bool(acc_a),
                      [] if f.has_comp else ["compressor state"],
                      "needs recorded compressor ON and OFF runs" if basis != RegimeBasis.DECLARED_BY_USER
                      else "not used with a declared passive period"),
    }
    cands: dict[str, CandidateResult] = {}
    fits: dict[str, FitOutcome] = {}
    for name in BASELINES:
        spec = R.BY_NAME[name]
        cands[name] = CandidateResult(name=name, version=spec.version, kind="BASELINE",
                                      outcome=CandidateOutcome.VALIDATED_NOT_SELECTED,
                                      required_inputs=list(spec.required_inputs), assumptions=list(spec.assumptions))
    for name in (*PHYSICAL_PASSIVE, R.FO_ACTIVE):
        spec = R.BY_NAME[name]
        ok, missing, why = eligibility[name]
        c = CandidateResult(name=name, version=spec.version, kind="PHYSICAL", outcome=CandidateOutcome.REJECTED,
                            required_inputs=list(spec.required_inputs), missing_inputs=missing,
                            assumptions=list(spec.assumptions))
        cands[name] = c
        if not ok:
            c.rejection_code = RefusalCode.MISSING_REQUIRED_INPUT if missing else RefusalCode.MODEL_NOT_APPLICABLE
            c.rejection_reason = why
            continue
        fit_pieces = by_role["FIT"] + ([p for p in roles_a if p.role == "FIT"] if name == R.FO_ACTIVE else [])
        fo, diag, rej = _fit_variant(name, fit_pieces, f, request, limits, "FIT split")
        c.fit = diag
        if rej:
            c.rejection_code, c.rejection_reason = rej
            continue
        fits[name] = fo
        c.outcome = (CandidateOutcome.HINDCAST_ONLY if name == R.FO_ACTIVE
                     else CandidateOutcome.VALIDATED_NOT_SELECTED)
    out.ctx.update(cands=cands, used_off_runs=any(p.regime == PASSIVE_RECORDED for p in acc_p))

    # ---------------------------------------------------------------- validation (SELECT)
    horizons = sorted({h for h in L.EVAL_HORIZONS_MIN if h < request.horizon_min} | {request.horizon_min})
    compared = [*BASELINES, *[n for n in PHYSICAL_PASSIVE if n in fits]]
    methods = {**_baseline_predictors(f), **{n: _physical_predictor(n, fits[n], f, request)
                                             for n in compared if n in fits}}
    sel_cases = _evaluate(f, by_role["SELECT"], methods, horizons, request)
    regime_label = roles_p[0].regime
    evals = _method_evals(sel_cases, compared, horizons, "SELECT", regime_label, "FIT split")
    comps = EV.comparisons(sel_cases, "SELECT", [n for n in compared if n in fits], list(BASELINES), horizons)
    vd = ValidationDiagnostics(status="NOT_VALIDATED", reason="", split=split,
                               origin_stride_min=max(L.EVAL_ORIGIN_STRIDE_MIN, f.med_s / 60), horizons_min=horizons,
                               evaluations=evals)
    out.kw.update(validation_diagnostics=vd, model_comparison=comps)

    # active-cooling hindcast (recorded control input); reported, never selected
    if R.FO_ACTIVE in fits:
        held = [p for p in roles_p + roles_a if p.role in ("SELECT", "TEST")]
        hm = {**_baseline_predictors(f), R.FO_ACTIVE: _physical_predictor(R.FO_ACTIVE, fits[R.FO_ACTIVE], f, request)}
        for reg in sorted({p.regime for p in held}):
            hc = _evaluate(f, [p for p in held if p.regime == reg], hm, horizons, request)
            vd.evaluations += _method_evals(hc, [*BASELINES, R.FO_ACTIVE], horizons, "HINDCAST", reg, "FIT split")
        cands[R.FO_ACTIVE].rejection_code = RefusalCode.MISSING_REQUIRED_INPUT
        cands[R.FO_ACTIVE].rejection_reason = ("fitted and checked as a hindcast with the recorded compressor state; "
                                               "it cannot forecast because the future compressor state is unknown")

    def finish_candidates() -> None:
        out.kw["candidate_models"] = list(cands.values())
        out.kw["rejected_models"] = [RejectedModel(name=c.name, code=c.rejection_code, reason=c.rejection_reason)
                                     for c in cands.values() if c.outcome == CandidateOutcome.REJECTED] + \
            out.kw["rejected_models"]
        out.kw["baseline_results"] = baseline_results

    # ---------------------------------------------------------------- applicability at the origin
    origin_piece = next((p for p in acc_p if p.rows[0] <= i0 <= p.rows[-1]), None)
    baseline_results: list[BaselineForecast] = []
    applicable = {n_: True for n_ in fits if n_ != R.FO_ACTIVE}    # the no-cooling scenario, from any origin
    for b, pred in _baseline_predictors(f).items():
        br = BaselineForecast(method=b, applicable=False)
        if regime != Regime.PASSIVE or origin_piece is None:
            br.reason = ("baselines extrapolate the observed regime; at the origin the room is not in an accepted "
                         "passive segment, so they do not describe the no-cooling scenario")
        else:
            law, why = pred(origin_piece, i0)
            if law is None:
                br.reason = why
            else:
                br.applicable = True
                br.temp_at_horizon_c = float(law.at(request.horizon_min))
                br.slope_c_per_min = law.slope
                if request.threshold_c is not None:
                    br.crossing_status, br.crossing_minutes = law.crossing(request.threshold_c, request.direction,
                                                                           request.horizon_min)
        applicable[b] = br.applicable
        baseline_results.append(br)

    sel_h, maes = None, {}
    for h in reversed(horizons):
        n, m = _common_mae(sel_cases, compared, h)
        if n >= L.EVAL_MIN_CASES:
            sel_h, maes = h, m
            break
    for n_, c in cands.items():
        if n_ in maes:
            c.selection_mae_c = maes[n_]
    for br in baseline_results:
        br.selection_mae_c = maes.get(br.method)
    out.ctx.update(maes=maes, sel_h=sel_h, applicable=applicable)
    if sel_h is None:
        vd.reason = (f"fewer than {L.EVAL_MIN_CASES} held-out cases where every method forecast at any horizon "
                     f"<= {request.horizon_min:g} min")
        finish_candidates()
        return out.refuse(RefusalCode.INSUFFICIENT_EVIDENCE, "the methods cannot be checked against held-out data: "
                          + vd.reason, "independent validation on held-out passive data",
                          "a longer passive record (more or longer off-periods before the origin)")
    vd.selection_horizon_min = sel_h
    vd.status = "VALIDATED"
    usable = {m: v for m, v in maes.items() if applicable.get(m)}
    listing = ", ".join(f"{m} {v:.3f} °C" for m, v in sorted(maes.items(), key=lambda x: x[1]))
    if not usable:
        vd.reason = f"held-out MAE at {sel_h:g} min: {listing}; none of these methods applies at the origin"
        finish_candidates()
        phys = _physical_summary(cands)
        codes = [c.rejection_code for c in cands.values() if c.kind == "PHYSICAL" and c.rejection_code]
        code = next((c for c in (RefusalCode.POOR_FIT, RefusalCode.INSUFFICIENT_IDENTIFIABILITY,
                                 RefusalCode.IMPLAUSIBLE_PARAMETERS) if c in codes), RefusalCode.MODEL_NOT_APPLICABLE)
        return out.refuse(code, "no validated method can answer the no-cooling question from this origin: the "
                          "baselines only extrapolate the observed (active) regime"
                          + (f", and the physical models were rejected: {phys}" if phys else ""),
                          "a validated physical model, or a passive regime at the origin for a baseline",
                          "data that identify the physical model (see candidate_models), or an origin inside a "
                          "passive period")
    out.kw["validated_horizon_min"] = sel_h
    n_common = _common_mae(sel_cases, compared, sel_h)[0]
    vd.reason = f"held-out MAE at {sel_h:g} min over {n_common} common cases: {listing}"
    if request.horizon_min > sel_h:
        # no held-out evidence at the requested horizon: refuse, or (on explicit request) extrapolate only a
        # physical model that was validated adequate. A constant slope or persistence has no structural basis
        # for extrapolation (see baselines.py), so neither may be used beyond the checked horizon.
        if not request.allow_extrapolation:
            finish_candidates()
            return out.refuse(RefusalCode.INSUFFICIENT_EVIDENCE,
                              f"held-out data check the methods only up to {sel_h:g} min; a {request.horizon_min:g}-"
                              "min forecast would be an unvalidated extrapolation",
                              f"requested horizon <= validated horizon ({sel_h:g} min)",
                              f"a horizon <= {sel_h:g} min, a longer passive record, or allow_extrapolation "
                              "(--allow-extrapolation) to extrapolate a validated physical model")
        usable = {m: v for m, v in usable.items() if m not in BASELINES and v <= L.ADEQUATE_ABS_ERROR_C}
        if not usable:
            finish_candidates()
            return out.refuse(RefusalCode.INSUFFICIENT_EVIDENCE,
                              f"extrapolation beyond the validated {sel_h:g} min was allowed, but no physical model "
                              "passed held-out validation", "a validated physical model",
                              "data that identify the physical model, or a horizon within the validated one")
        out.forecast_warning(f"EXTRAPOLATED at the user's request: held-out checks reach {sel_h:g} min, the "
                             f"forecast runs to {request.horizon_min:g} min; nothing beyond {sel_h:g} min has "
                             "been checked against data. Baselines are excluded from extrapolation")
    best = min(usable.values())
    # the simplest applicable method whose held-out error is within the materiality margin of the best one
    winner = next(m for m in COMPLEXITY if m in usable and usable[m] <= best + L.SELECTION_MARGIN_C)
    vd.reason += (f". Selected: the simplest method applicable at the origin within {L.SELECTION_MARGIN_C} °C of the "
                  f"lowest ({winner})")
    out.ctx.update(winner=winner, selection_reason=vd.reason)
    for m, v in maes.items():
        if m not in usable and v < maes[winner] - L.SELECTION_MARGIN_C:
            out.forecast_warning(f"{m} had a lower held-out MAE ({v:.3f} °C) than the selected {winner} "
                                 f"({maes[winner]:.3f} °C) but cannot answer the no-cooling question from this "
                                 "origin")
    if maes[winner] > L.ADEQUATE_ABS_ERROR_C:
        finish_candidates()
        return out.refuse(RefusalCode.NO_ADEQUATE_MODEL,
                          f"the best method ({winner}) has a held-out MAE of {maes[winner]:.2f} °C at {sel_h:g} min, "
                          f"above the {L.ADEQUATE_ABS_ERROR_C} °C adequacy limit; no method is reliable here",
                          f"held-out MAE <= {L.ADEQUATE_ABS_ERROR_C} °C", "cleaner or more representative passive data")

    # ---------------------------------------------------------------- final evaluation (TEST)
    test_fits: dict[str, FitOutcome] = {}
    for n_ in fits:
        if n_ == R.FO_ACTIVE:
            continue
        fo, _, rej = _fit_variant(n_, by_role["FIT"] + by_role["SELECT"], f, request, limits, "FIT+SELECT")
        if rej is None:
            test_fits[n_] = fo
    t_compared = [*BASELINES, *test_fits]
    t_methods = {**_baseline_predictors(f), **{n: _physical_predictor(n, test_fits[n], f, request) for n in test_fits}}
    test_cases = _evaluate(f, by_role["TEST"], t_methods, horizons, request)
    vd.evaluations += _method_evals(test_cases, t_compared, horizons, "TEST", regime_label, "FIT+SELECT")
    out.kw["model_comparison"] = comps + EV.comparisons(test_cases, "TEST", list(test_fits), list(BASELINES), horizons)
    n_test, test_maes = _common_mae(test_cases, t_compared, sel_h)
    out.keep(test_cases=test_cases, n_test=n_test, horizons=horizons, sel_h=sel_h, winner=winner)
    for n_, c in cands.items():
        if n_ in test_maes:
            c.test_mae_c = test_maes[n_]
    for br in baseline_results:
        br.test_mae_c = test_maes.get(br.method)
    if winner not in t_compared:
        finish_candidates()
        return out.refuse(RefusalCode.INSUFFICIENT_IDENTIFIABILITY,
                          f"{winner} could not be refitted on FIT+SELECT data", "a stable refit", None)
    if n_test >= L.EVAL_MIN_CASES:
        if test_maes[winner] > L.ADEQUATE_ABS_ERROR_C:
            finish_candidates()
            return out.refuse(RefusalCode.NO_ADEQUATE_MODEL,
                              f"{winner} was selected, but its final-evaluation MAE at {sel_h:g} min is "
                              f"{test_maes[winner]:.2f} °C (> {L.ADEQUATE_ABS_ERROR_C} °C)",
                              f"final-evaluation MAE <= {L.ADEQUATE_ABS_ERROR_C} °C", "more representative data")
    else:
        out.kw["warnings"].append(f"final evaluation not possible: {n_test} common held-out TEST case(s) at "
                                  f"{sel_h:g} min (need {L.EVAL_MIN_CASES}); selection rests on the SELECT data only")

    # ---------------------------------------------------------------- final forecast
    spec = R.BY_NAME[winner]
    out.kw.update(selected_model=winner, selected_model_version=spec.version)
    scenario = [f"no active cooling from {t0} for {request.horizon_min:g} min (ASSUMED scenario)"
                if request.question == ForecastQuestion.NO_COOLING else
                f"the observed passive regime continues: no active cooling from {t0} for {request.horizon_min:g} min "
                "(ASSUMED scenario)"]
    if regime == Regime.ACTIVE:
        scenario.append("the compressor is recorded ON at the origin: this is a what-if (cooling lost now), not a "
                        "continuation of the observed regime")
    elif regime == Regime.UNKNOWN:
        out.kw["warnings"].append("the operating regime at the origin is unknown")
    y0 = float(f.y[i0])
    if winner in BASELINES:
        phys = _physical_summary(cands)
        if phys:
            out.kw["warnings"].append("physical models not usable: " + phys)
        law = _baseline_predictors(f)[winner](origin_piece, i0)[0]
        out.keep(origin_piece=origin_piece)
        out.kw.update(solution_method=f"closed form ({law.kind})", assumptions=list(spec.assumptions))
        out.kw["parameter_estimates"] = [] if law.slope is None else [ParameterEstimate(
            name="slope", unit="°C/min", value=law.slope, provenance=Provenance.FITTED,
            method=f"OLS over the trailing {L.SLOPE_WINDOW_MIN:g} min")]
    else:
        fo, diag, rej = _fit_variant(winner, acc_p, f, request, limits, "all eligible passive data before the origin")
        out.kw["fit_diagnostics"] = diag
        if rej:
            out.kw.pop("fit_diagnostics")         # a rejected final fit's parameters are unsupported: withheld
            finish_candidates()
            return out.refuse(rej[0], f"final refit of {winner} on all data before the origin: {rej[1]}",
                              "an identifiable, plausible, adequate final fit", None)
        out.kw["observations_used"] = fo.n
        e_note, e_prov = "", Provenance.FITTED
        if winner == R.FO_FITTED:
            e = fo.eq["E"]
        elif winner == R.FO_ASSUMED:
            e, e_prov = float(request.assumed_equilibrium_c), Provenance.ASSUMED
        elif request.future_ambient_c is not None:
            e, e_prov, e_note = float(request.future_ambient_c), Provenance.ASSUMED, "future ambient given by the user"
        else:
            last, j = _last_ambient(f, 0, i0)
            win = f.amb[(f.t >= f.t[i0] - 60) & (np.arange(len(f.t)) <= i0)]
            win = win[np.isfinite(win)]
            if winner == R.FO_VARYING or last is None or not len(win) or np.ptp(win) > L.AMBIENT_STEADY_RANGE_C:
                out.ctx["needs_future_ambient"] = True
                finish_candidates()
                return out.refuse(RefusalCode.MISSING_REQUIRED_INPUT,
                                  f"{winner} needs the ambient temperature over the horizon; it varied recently or "
                                  "is time-varying, and future ambient is never extrapolated",
                                  "future ambient (scenario)", "future_ambient_c / --future-ambient")
            e, e_prov = last, Provenance.ASSUMED
            e_note = (f"ambient held at its last measured value {last:.2f} °C (OBSERVED at line {int(f.lines[j])}; "
                      "steady over the last 60 min)")
        scenario.append(f"equilibrium {e:.2f} °C over the horizon ({e_prov.value}{': ' + e_note if e_note else ''})")
        law = Law("exponential", y0, eq=e, k=fo.k)
        out.keep(final_fit=fo, e_forecast=e, e_provenance=e_prov, origin_piece=origin_piece)
        out.kw.update(solution_method="analytic: T(t) = E + (T0 - E) exp(-k t)", assumptions=list(spec.assumptions))
        out.kw["parameter_estimates"] = diag.parameters + [ParameterEstimate(
            name="E_forecast", unit="°C", value=e, provenance=e_prov, method="equilibrium used over the horizon",
            note=e_note)]
        span_fit = max(p.duration(f) for p in acc_p)
        out.ctx["longest_fit_min"] = span_fit
        if request.horizon_min > span_fit:
            out.forecast_warning(f"horizon {request.horizon_min:g} min is longer than the longest fitted segment "
                                 f"({span_fit:.0f} min): the model is extrapolated in time")
        fitted_rows = np.concatenate([p.rows for p in acc_p])
        lo, hi = float(np.min(f.y[fitted_rows])), float(np.max(f.y[fitted_rows]))
        end_t = float(law.at(request.horizon_min))
        if end_t > hi + 0.5 or end_t < lo - 0.5:
            out.forecast_warning(f"the forecast reaches {end_t:.1f} °C, outside the {lo:.1f}..{hi:.1f} °C seen in "
                                 "the fitted segments: extrapolated in temperature")
    out.kw["parameter_units"] = {p.name: p.unit for p in out.kw["parameter_estimates"]}
    out.kw["scenario"] = scenario
    out.ctx["scenario"] = scenario

    if not _trajectory(out, law, request, f, t0):
        finish_candidates()
        return out.refuse(RefusalCode.NUMERICAL_ERROR, "the trajectory is not finite", "finite trajectory", None)
    cands[winner].outcome = CandidateOutcome.SELECTED        # only once a forecast is actually produced
    finish_candidates()
    return _emit(out, winner, request)


def _trajectory(out: _Out, law: Law, request: ForecastRequest, f: Frame, t0) -> bool:
    """Trajectory from 0 to exactly the horizon, and the crossing; False if not finite."""
    step = request.step_min or (f.med_s / 60)
    if request.step_min is None and request.horizon_min / step > L.MAX_TRAJECTORY_POINTS:
        step = request.horizon_min / L.MAX_TRAJECTORY_POINTS
        out.kw["warnings"].append(f"trajectory spacing set to {step:.3g} min to stay within "
                                  f"{L.MAX_TRAJECTORY_POINTS} points")
    hs = np.append(np.arange(0.0, request.horizon_min, step), request.horizon_min)
    temps = np.asarray(law.at(hs), dtype=float)
    if not np.all(np.isfinite(temps)):
        return False
    base = pd.Timestamp(t0)
    at = lambda m: (base + pd.to_timedelta(m, unit="min")).floor("us").to_pydatetime()   # noqa: E731
    out.kw["trajectory"] = [TrajectoryPoint(ts=at(float(h)), minutes=round(float(h), 6), temp_c=float(v))
                            for h, v in zip(hs, temps, strict=True)]
    if request.threshold_c is not None:
        status, mins = law.crossing(request.threshold_c, request.direction, request.horizon_min)
        out.kw["crossing_status"] = status
        if status == CrossingStatus.CROSSING_PREDICTED:
            out.kw["predicted_crossing_minutes"] = mins
            out.kw["predicted_crossing_time"] = at(mins)
    return True


def _emit(out: _Out, winner: str, request: ForecastRequest) -> ForecastResult:
    out.kw.update(advice.annotate(out.ctx, False, None, None))
    res = ForecastResult(status=ForecastStatus.FORECAST, **out.kw)
    log.info("thermal forecast model=%s horizon=%s crossing=%s", winner, request.horizon_min, res.crossing_status.value)
    return res


def _report_dropped(out: _Out, f: Frame, dropped: list[int]) -> None:
    susp = [i for i in dropped if i in f.suspicious]
    evx = [i for i in dropped if i in f.event_excluded and i not in f.suspicious]
    msgs = []
    if susp:
        msgs.append(f"{len(susp)} reading(s) flagged TEMP_SUSPICIOUS by validation split the fitting segments and are "
                    f"not fitted (lines {[int(f.lines[i]) for i in susp][:20]}); their cause is unknown")
    if evx:
        kinds = sorted({f.event_excluded[i] for i in evx})
        msgs.append(f"{len(evx)} reading(s) inside {', '.join(kinds)} records of the event log are not fitted and "
                    f"not used as held-out truth (lines {[int(f.lines[i]) for i in evx][:20]})")
    if msgs:
        out.kw["observations_excluded"] = msgs
    out.ctx["dropped_suspicious"] = len(susp)


def _inadequate_horizon(cases: list[EV.Case], method: str, up_to: float) -> tuple[float, float] | None:
    """First horizon <= up_to where the method's own held-out MAE (>= EVAL_MIN_CASES cases) exceeds the limit."""
    for h in sorted({c.horizon for c in cases if c.horizon <= up_to}):
        m = EV.horizon_metrics(cases, method, h, [method])
        if m.n_forecasts >= L.EVAL_MIN_CASES and m.mae_c is not None and m.mae_c > L.ADEQUATE_ABS_ERROR_C:
            return h, m.mae_c
    return None


# ====================================================================== continuation, non-passive origin
_CONT_PHYS = ("CONTINUATION from an active or unknown regime: the implemented physical models describe passive "
              "exchange only, and continued thermostat cycling would need the future compressor state, which is "
              "unknown")


def _continuation(f: Frame, i0: int, request: ForecastRequest, out: _Out, t0) -> ForecastResult:
    """Baselines only, evaluated on every usable stretch before the origin (any regime). Same held-out
    rules as the main path: validated horizon, 0.02 °C simplicity margin, 0.5 °C adequacy on SELECT and TEST.
    Baselines are never extrapolated beyond the validated horizon."""
    out.ctx["path"] = "continuation"
    cands: dict[str, CandidateResult] = {}
    for name in BASELINES:
        spec = R.BY_NAME[name]
        cands[name] = CandidateResult(name=name, version=spec.version, kind="BASELINE",
                                      outcome=CandidateOutcome.VALIDATED_NOT_SELECTED,
                                      required_inputs=list(spec.required_inputs), assumptions=list(spec.assumptions))
    for name in (*PHYSICAL_PASSIVE, R.FO_ACTIVE):
        spec = R.BY_NAME[name]
        cands[name] = CandidateResult(name=name, version=spec.version, kind="PHYSICAL",
                                      outcome=CandidateOutcome.REJECTED, required_inputs=list(spec.required_inputs),
                                      assumptions=list(spec.assumptions),
                                      rejection_code=RefusalCode.MODEL_NOT_APPLICABLE, rejection_reason=_CONT_PHYS)
    out.ctx["cands"] = cands
    brs: list[BaselineForecast] = []

    def finish() -> None:
        out.kw["candidate_models"] = list(cands.values())
        out.kw["rejected_models"] = [RejectedModel(name=c.name, code=c.rejection_code, reason=c.rejection_reason)
                                     for c in cands.values() if c.outcome == CandidateOutcome.REJECTED] + \
            out.kw["rejected_models"]
        out.kw["baseline_results"] = brs

    pieces, dropped = all_pieces(f, i0)
    _report_dropped(out, f, dropped)
    pieces = [p for p in pieces if len(p.rows) >= 2 and p.duration(f) > 0]
    if not pieces:
        finish()
        return out.refuse(RefusalCode.INSUFFICIENT_DATA, "no usable stretch of readings before the origin",
                          "at least one gap-free stretch before the origin", "a longer log")
    cuts = chronological_cuts(f, pieces)
    roles = assign_roles(f, pieces, cuts)
    out.kw["segments_used"] = [p.record(f, True, None) for p in roles]
    t_of = lambda m: (f.ts.iloc[0] + pd.to_timedelta(m, unit="min")).isoformat()   # noqa: E731
    split = (f"chronological over {sum(p.duration(f) for p in pieces):.1f} min of usable data (any regime): FIT up to "
             f"{t_of(cuts[0])} (nothing is fitted: baselines only), SELECT up to {t_of(cuts[1])}, TEST after; nothing "
             "after the origin is used")
    by_role = {r: [p for p in roles if p.role == r] for r in ("FIT", "SELECT", "TEST")}
    out.keep(path="continuation", frame=f, cuts=cuts, test_pieces=by_role["TEST"], basis=out.ctx.get("basis"),
             origin_regime=out.ctx.get("regime"), i0=i0)
    horizons = sorted({h for h in L.EVAL_HORIZONS_MIN if h < request.horizon_min} | {request.horizon_min})
    methods = _baseline_predictors(f)
    sel_cases = _evaluate(f, by_role["SELECT"], methods, horizons, request)
    vd = ValidationDiagnostics(status="NOT_VALIDATED", reason="", split=split,
                               origin_stride_min=max(L.EVAL_ORIGIN_STRIDE_MIN, f.med_s / 60), horizons_min=horizons,
                               evaluations=_method_evals(sel_cases, list(BASELINES), horizons, "SELECT", ANY_REGIME,
                                                         "nothing fitted (baselines)"))
    out.kw["validation_diagnostics"] = vd
    origin_piece = next((p for p in pieces if p.rows[0] <= i0 <= p.rows[-1]), None)
    applicable: dict[str, bool] = {}
    laws: dict[str, Law] = {}
    for b, pred in methods.items():
        br = BaselineForecast(method=b, applicable=False)
        if origin_piece is None:
            br.reason = "the origin reading is not in a usable stretch (flagged or inside an excluding event)"
        else:
            law, why = pred(origin_piece, i0)
            if law is None:
                br.reason = why
            else:
                laws[b] = law
                br.applicable = True
                br.temp_at_horizon_c = float(law.at(request.horizon_min))
                br.slope_c_per_min = law.slope
                if request.threshold_c is not None:
                    br.crossing_status, br.crossing_minutes = law.crossing(request.threshold_c, request.direction,
                                                                           request.horizon_min)
        applicable[b] = br.applicable
        brs.append(br)
    sel_h, maes = None, {}
    for h in reversed(horizons):
        n, m = _common_mae(sel_cases, list(BASELINES), h)
        if n >= L.EVAL_MIN_CASES:
            sel_h, maes = h, m
            break
    for n_, v in maes.items():
        cands[n_].selection_mae_c = v
    for br in brs:
        br.selection_mae_c = maes.get(br.method)
    out.ctx.update(maes=maes, sel_h=sel_h, applicable=applicable)
    if sel_h is None:
        vd.reason = (f"fewer than {L.EVAL_MIN_CASES} held-out cases where both baselines forecast at any horizon "
                     f"<= {request.horizon_min:g} min")
        finish()
        return out.refuse(RefusalCode.INSUFFICIENT_EVIDENCE, "the baselines cannot be checked against held-out data: "
                          + vd.reason, "independent validation on held-out data", "a longer log before the origin")
    vd.selection_horizon_min, vd.status = sel_h, "VALIDATED"
    out.kw["validated_horizon_min"] = sel_h
    listing = ", ".join(f"{m} {v:.3f} °C" for m, v in sorted(maes.items(), key=lambda x: x[1]))
    vd.reason = f"held-out MAE at {sel_h:g} min over {_common_mae(sel_cases, list(BASELINES), sel_h)[0]} common " \
                f"cases: {listing}"
    if request.horizon_min > sel_h:
        finish()
        return out.refuse(RefusalCode.INSUFFICIENT_EVIDENCE,
                          f"held-out data check the baselines only up to {sel_h:g} min; baselines are never "
                          f"extrapolated, so a {request.horizon_min:g}-min continuation forecast is not supported",
                          f"requested horizon <= validated horizon ({sel_h:g} min)",
                          f"a horizon <= {sel_h:g} min or a longer log (allow_extrapolation applies to physical "
                          "models only)")
    usable = {m: v for m, v in maes.items() if applicable.get(m)}
    if not usable:
        finish()
        return out.refuse(RefusalCode.MODEL_NOT_APPLICABLE, "no baseline can be computed at the origin: "
                          + "; ".join(f"{b.method}: {b.reason}" for b in brs),
                          "a usable reading and trailing window at the origin", "an origin with usable readings")
    best = min(usable.values())
    winner = next(m for m in COMPLEXITY if m in usable and usable[m] <= best + L.SELECTION_MARGIN_C)
    vd.reason += (f". Selected: the simplest baseline computable at the origin within {L.SELECTION_MARGIN_C} °C of the "
                  f"lowest ({winner})")
    out.ctx.update(winner=winner, selection_reason=vd.reason)
    # adequacy at EVERY checked horizon up to the selection horizon: under cycling, errors are not monotone in
    # the horizon (persistence looks good one full cycle ahead and is wrong half a cycle ahead)
    bad = _inadequate_horizon(sel_cases, winner, sel_h)
    if maes[winner] > L.ADEQUATE_ABS_ERROR_C or bad:
        h_bad, mae_bad = (sel_h, maes[winner]) if maes[winner] > L.ADEQUATE_ABS_ERROR_C else bad
        finish()
        return out.refuse(RefusalCode.NO_ADEQUATE_MODEL,
                          f"the best baseline ({winner}) has a held-out MAE of {mae_bad:.2f} °C at {h_bad:g} min, "
                          f"above the {L.ADEQUATE_ABS_ERROR_C} °C adequacy limit; no method is reliable here",
                          f"held-out MAE <= {L.ADEQUATE_ABS_ERROR_C} °C at every checked horizon up to {sel_h:g} min",
                          "a shorter horizon or steadier operation")
    test_cases = _evaluate(f, by_role["TEST"], methods, horizons, request)
    vd.evaluations += _method_evals(test_cases, list(BASELINES), horizons, "TEST", ANY_REGIME,
                                    "nothing fitted (baselines)")
    n_test, test_maes = _common_mae(test_cases, list(BASELINES), sel_h)
    out.keep(test_cases=test_cases, n_test=n_test, horizons=horizons, sel_h=sel_h, winner=winner)
    for n_, v in test_maes.items():
        cands[n_].test_mae_c = v
    for br in brs:
        br.test_mae_c = test_maes.get(br.method)
    if n_test >= L.EVAL_MIN_CASES:
        bad = _inadequate_horizon(test_cases, winner, sel_h)
        if test_maes[winner] > L.ADEQUATE_ABS_ERROR_C or bad:
            h_bad, mae_bad = (sel_h, test_maes[winner]) if test_maes[winner] > L.ADEQUATE_ABS_ERROR_C else bad
            finish()
            return out.refuse(RefusalCode.NO_ADEQUATE_MODEL,
                              f"{winner} was selected, but its final-evaluation MAE at {h_bad:g} min is "
                              f"{mae_bad:.2f} °C (> {L.ADEQUATE_ABS_ERROR_C} °C)",
                              f"final-evaluation MAE <= {L.ADEQUATE_ABS_ERROR_C} °C", "more representative data")
    else:
        out.kw["warnings"].append(f"final evaluation not possible: {n_test} common held-out TEST case(s) at "
                                  f"{sel_h:g} min (need {L.EVAL_MIN_CASES}); selection rests on the SELECT data only")
    spec = R.BY_NAME[winner]
    law = laws[winner]
    out.keep(origin_piece=origin_piece)
    what = ("the temperature stays at the origin reading" if winner == R.PERSISTENCE else
            f"the slope of the last {L.SLOPE_WINDOW_MIN:g} min ({law.slope:+.4f} °C/min) continues")
    scenario = [f"the behaviour observed at the origin continues for {request.horizon_min:g} min: {what} "
                "(ASSUMED scenario; BASELINE, not a physical model)"]
    out.kw.update(selected_model=winner, selected_model_version=spec.version,
                  solution_method=f"closed form ({law.kind})", assumptions=list(spec.assumptions), scenario=scenario,
                  parameter_estimates=[] if law.slope is None else [ParameterEstimate(
                      name="slope", unit="°C/min", value=law.slope, provenance=Provenance.FITTED,
                      method=f"OLS over the trailing {L.SLOPE_WINDOW_MIN:g} min")])
    out.kw["parameter_units"] = {p.name: p.unit for p in out.kw["parameter_estimates"]}
    out.ctx["scenario"] = scenario
    if not _trajectory(out, law, request, f, t0):
        finish()
        return out.refuse(RefusalCode.NUMERICAL_ERROR, "the trajectory is not finite", "finite trajectory", None)
    cands[winner].outcome = CandidateOutcome.SELECTED
    finish()
    return _emit(out, winner, request)


def forecast_source(src, *, source: str | None = None, evidence: EvidenceClass = EvidenceClass.UNVALIDATED,
                    options=None, limits: ValidationLimits | None = None, request: ForecastRequest,
                    events: list | None = None, metadata=None,
                    ) -> tuple[pd.DataFrame, pd.DataFrame, DataQualityReport, ForecastResult]:
    """Ingest + validate (Phase 2) + forecast. Reuses the accepted validation unchanged."""
    from ..pipeline import validate_source

    clean, obs, report = validate_source(src, source=source, evidence=evidence, options=options, limits=limits)
    return clean, obs, report, forecast(clean, report, obs, request, events=events, metadata=metadata)


__all__ = ["ENGINE_VERSION", "forecast", "forecast_source"]
