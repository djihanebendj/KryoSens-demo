"""Input capabilities, model-selection record, limitation details and missing-input advice.

Everything here is derived from what the engine actually did on this log (the
context it records); nothing is recomputed, and no number is produced. Rules:

* a model that could not run for lack of an input is "not compared": no claim is
  made that it would have been better or worse;
* a baseline is labelled as a baseline, never as a validated physical forecast;
* a recommendation states the potential capability and how to measure any
  benefit; it never promises better accuracy and never changes equipment.
"""

from __future__ import annotations

from typing import Any

from . import registry as R
from .inputs import BY_NAME as INPUT
from .inputs import CATALOGUE, InputFamily, InputSource
from .schemas import (
    NECESSITY_ORDER,
    NO_EFFECT_BENEFIT,
    Alternative,
    CandidateOutcome,
    ForecastQuestion,
    InputCapabilities,
    InputState,
    InputStatus,
    LimitationDetails,
    ModelSelection,
    Necessity,
    Priority,
    Recommendation,
    RefusalCode,
    RegimeBasis,
    RejectedModel,
)

INFORMATIVENESS = (R.FO_VARYING, R.FO_MEASURED, R.FO_FITTED, R.FO_ASSUMED, R.RECENT_SLOPE, R.PERSISTENCE)
_INVENTORY_NAMES = {"room air temperature": "room_temperature", "timestamps": "timestamp",
                    "ambient temperature": "ambient_temperature", "compressor state": "compressor_state"}
_TEMP_HINTS = ("temp", "t_", "°c", "degc", "sonde", "probe", "sensor")


def _state(name: str, status: InputStatus, detail: str, provenance: str | None = None) -> InputState:
    spec = INPUT[name]
    return InputState(name=name, family=spec.family.value, unit=spec.unit, status=status, source=spec.source.value,
                      provenance=provenance or spec.provenance, used_by=list(spec.used_by), detail=detail)


# ====================================================================== capabilities
def capabilities(ctx: dict[str, Any], refused: bool = False) -> InputCapabilities:
    report, request = ctx["report"], ctx["request"]
    events, meta = ctx.get("events") or [], ctx.get("metadata")
    out = InputCapabilities()
    seen = set()
    for item in ctx.get("inventory") or []:
        name = _INVENTORY_NAMES.get(item.name)
        if name is None:
            if item.status == InputStatus.NOT_VERIFIED:
                out.quality_warnings.append(f"{item.name}: {item.detail}")
            continue
        seen.add(name)
        st = _state(name, item.status, item.detail)
        (out.available_inputs if item.status in (InputStatus.AVAILABLE, InputStatus.NOT_VERIFIED) else
         out.invalid_inputs if item.status == InputStatus.INVALID else out.missing_inputs).append(st)
    if not ctx.get("inventory"):                                   # refused before the inventory was taken
        for name in ("room_temperature", "timestamp"):
            seen.add(name)
            out.available_inputs.append(_state(name, InputStatus.AVAILABLE, f"{report.n_rows_clean} usable rows"))
    kinds = {}
    for e in events:
        kinds.setdefault(e.kind.value.lower(), set()).add(e.provenance.value)
    for spec in CATALOGUE:
        if spec.name in seen:
            continue
        if spec.family == InputFamily.EVENT and spec.source == InputSource.EVENT_LOG:
            if spec.name in kinds:
                out.available_inputs.append(_state(spec.name, InputStatus.AVAILABLE,
                                                   f"{sum(e.kind.value.lower() == spec.name for e in events)} "
                                                   "record(s) in the supplied event log",
                                                   "OBSERVED (" + ", ".join(sorted(kinds[spec.name])) + ")"))
            else:
                out.missing_inputs.append(_state(spec.name, InputStatus.MISSING, "no record supplied: "
                                                 + spec.when_missing))
        elif spec.family == InputFamily.METADATA:
            if meta is not None and spec.name in meta.supplied():
                mv = getattr(meta, spec.name)
                detail = (f"{len(meta.sensors)} sensor description(s)" if spec.name == "sensors"
                          else f"{mv.value:g} {mv.unit} ({mv.provenance.value}); disclosed, not used in calculations")
                out.available_inputs.append(_state(spec.name, InputStatus.NOT_VERIFIED, detail))
            else:
                out.missing_inputs.append(_state(spec.name, InputStatus.MISSING, spec.when_missing))
        elif spec.family == InputFamily.SCENARIO:
            given = {"passive_since": request.passive_since, "assumed_equilibrium": request.assumed_equilibrium_c,
                     "future_ambient": request.future_ambient_c}[spec.name]
            if given is not None:
                out.available_inputs.append(_state(spec.name, InputStatus.AVAILABLE, f"{given} (stated by the user)",
                                                   "ASSUMED"))
                out.assumptions_used.append(f"{spec.name} = {given} (ASSUMED, not measured)")
            else:
                out.missing_inputs.append(_state(spec.name, InputStatus.MISSING, spec.when_missing))
        else:
            out.missing_inputs.append(_state(spec.name, InputStatus.MISSING,
                                             ("no input role in this version: " if spec.source ==
                                              InputSource.NOT_SUPPORTED else "") + spec.when_missing))
    # ---- quality warnings and the analysed sensor
    mapping = report.input.column_mapping if report.input else {}
    col = mapping.get("temperature")
    out.analysed_sensor = col
    sensor = next((s for s in (meta.sensors if meta else []) if s.column == col), None) if col else None
    if col:
        where = f" at {sensor.location}" if sensor and sensor.location else ""
        out.quality_warnings.append(f"results describe the air at one sensor, column {col!r}{where}; ColdSense "
                                    "does not average sensors or assume one point represents the whole room")
    if meta and len(meta.sensors) > 1:
        others = [s.id + (f" ({s.column})" if s.column else "") for s in meta.sensors if s is not sensor]
        out.quality_warnings.append(f"{len(meta.sensors)} sensors are described; only {col!r} is analysed here. "
                                    f"Others: {others}. Run once per sensor (--temperature-column) to compare them")
    for s in (meta.sensors if meta else []):
        if s.calibration_offset_c is not None:
            out.quality_warnings.append(f"sensor {s.id}: calibration offset {s.calibration_offset_c:+g} °C is "
                                        "documented; readings are NOT corrected")
    ignored = report.input.ignored_columns if report.input else []
    maybe = [c for c in ignored if any(h in c.lower() for h in _TEMP_HINTS)]
    if maybe:
        out.quality_warnings.append(f"unmapped column(s) {maybe} may be other temperature sensors; they are not "
                                    "analysed (select one with --temperature-column)")
    if ctx.get("dropped_suspicious"):
        out.quality_warnings.append(f"{ctx['dropped_suspicious']} reading(s) flagged TEMP_SUSPICIOUS split the "
                                    "fitting segments; cause unknown (not attributed)")
    out.quality_warnings += ctx.get("event_notes") or []
    if events and request.passive_since is not None and any(e.kind.value == "POWER_INTERRUPTION" for e in events):
        out.quality_warnings.append("power-interruption records were not used as regime evidence because "
                                    "passive_since was given (a declared period takes precedence)")
    if any(i.name == "meaning of compressor OFF" for i in ctx.get("inventory") or []) and ctx.get("used_off_runs"):
        out.assumptions_used.append("compressor recorded OFF is taken to mean no active cooling")
    if not refused:                                   # a refused forecast states no horizon scenario
        out.assumptions_used += ctx.get("scenario") or []
    return out


# ====================================================================== model selection
def selection(ctx: dict[str, Any], refused: bool, code: RefusalCode | None, reason: str | None) -> ModelSelection:
    request = ctx["request"]
    cands = ctx.get("cands") or {}
    maes = ctx.get("maes") or {}
    applicable = ctx.get("applicable") or {}
    winner = None if refused else ctx.get("winner")
    candidates = [m.name for m in R.REGISTRY if m.status == R.IMPLEMENTED]
    rejected, eligible = [], []
    for n, c in cands.items():
        if c.outcome == CandidateOutcome.REJECTED:
            rejected.append(RejectedModel(name=n, code=c.rejection_code, reason=c.rejection_reason or ""))
            if c.rejection_code not in (RefusalCode.MISSING_REQUIRED_INPUT, RefusalCode.MODEL_NOT_APPLICABLE):
                eligible.append(n)
        else:
            eligible.append(n)
    alts = []
    ref = maes.get(winner) if winner else None
    for m in INFORMATIVENESS:
        c = cands.get(m)
        if m == winner:
            alts.append(Alternative(model=m, status="SELECTED", detail=ctx.get("selection_reason") or ""))
        elif c is None:
            alts.append(Alternative(model=m, status="NOT_EVALUATED", detail="the run stopped before model evaluation"))
        elif c.outcome == CandidateOutcome.REJECTED and c.rejection_code == RefusalCode.MISSING_REQUIRED_INPUT:
            alts.append(Alternative(model=m, status="UNAVAILABLE_MISSING_INPUT",
                                    detail=f"{c.rejection_reason}; not compared, so no claim is made that it would be "
                                           "better or worse"))
        elif c.outcome == CandidateOutcome.REJECTED and c.rejection_code == RefusalCode.MODEL_NOT_APPLICABLE:
            alts.append(Alternative(model=m, status="NOT_APPLICABLE", detail=c.rejection_reason or ""))
        elif c.outcome == CandidateOutcome.REJECTED:
            alts.append(Alternative(model=m, status="REJECTED",
                                    detail=f"{c.rejection_code.value}: {c.rejection_reason}"))
        elif m in maes and not applicable.get(m, False):
            alts.append(Alternative(model=m, status="NOT_APPLICABLE_AT_ORIGIN",
                                    detail=f"held-out MAE {maes[m]:.3f} °C, but it cannot answer this question from "
                                           "this origin"))
        elif m in maes and ref is not None and maes[m] <= ref:
            alts.append(Alternative(model=m, status="WITHIN_MARGIN_SIMPLER_KEPT",
                                    detail=f"held-out MAE {maes[m]:.3f} °C vs {ref:.3f} °C for {winner}: within the "
                                           "materiality margin, so the simpler method was kept"))
        elif m in maes and ref is not None:
            alts.append(Alternative(model=m, status="HIGHER_HELD_OUT_ERROR",
                                    detail=f"held-out MAE {maes[m]:.3f} °C vs {ref:.3f} °C for {winner}"))
        elif m in maes:
            alts.append(Alternative(model=m, status="VALIDATED_NOT_USED", detail=f"held-out MAE {maes[m]:.3f} °C; "
                                    "no forecast was produced"))
        else:
            alts.append(Alternative(model=m, status="NOT_COMPARED",
                                    detail="no common held-out cases at the validated horizon"))
    if R.FO_ACTIVE in cands:
        c = cands[R.FO_ACTIVE]
        status = ("HINDCAST_ONLY" if c.outcome == CandidateOutcome.HINDCAST_ONLY else
                  "UNAVAILABLE_MISSING_INPUT" if c.rejection_code == RefusalCode.MISSING_REQUIRED_INPUT else
                  "NOT_APPLICABLE" if c.rejection_code == RefusalCode.MODEL_NOT_APPLICABLE else "REJECTED")
        detail = c.rejection_reason or ("a hindcast check only: the future compressor state is unknown"
                                        if status == "HINDCAST_ONLY" else "")
        if status == "REJECTED" and c.rejection_code is not None:
            detail = f"{c.rejection_code.value}: {detail}"
        alts.append(Alternative(model=R.FO_ACTIVE, status=status, detail=detail))
    kind = "NONE" if winner is None else ("BASELINE" if winner in (R.PERSISTENCE, R.RECENT_SLOPE) else "PHYSICAL")
    if refused:
        sel_reason = f"no forecast: {code.value if code else 'refused'}: {reason}"
    else:
        sel_reason = ctx.get("selection_reason") or ""
    fallback = None
    above = [a for a in alts if a.model in INFORMATIVENESS and winner and
             INFORMATIVENESS.index(a.model) < INFORMATIVENESS.index(winner)]
    if winner and above:
        fallback = (f"{winner} was used; more physically informative alternatives: "
                    + "; ".join(f"{a.model} {a.status}" for a in above))
    elif refused and cands:
        fallback = "no approach met its requirements: " + "; ".join(
            f"{a.model} {a.status}" for a in alts if a.status != "SELECTED")
    elif refused:
        fallback = f"no model was evaluated: {reason}"
        if ctx.get("no_passive_evidence") and request.question == ForecastQuestion.NO_COOLING:
            fallback += (". The baselines can answer only a different question here, whether the observed behaviour "
                         "continues (question CONTINUATION, --question continuation); that is not a fallback for "
                         "the no-cooling question")
    if kind == "BASELINE":
        fallback = (fallback + ". " if fallback else "") + (
            f"{winner} is a BASELINE: it extrapolates the observed regime and is not a validated physical forecast")
    return ModelSelection(question=request.question, candidate_models=candidates, eligible_models=eligible,
                          validated_models=sorted(maes), rejected_models=rejected, selected_model=winner,
                          forecast_kind=kind, selection_reason=sel_reason, fallback_reason=fallback,
                          alternatives=alts)


# ====================================================================== limitations
def limitation_details(ctx: dict[str, Any], caps: InputCapabilities, refused: bool) -> LimitationDetails:
    out = LimitationDetails()
    request = ctx["request"]
    absent = {s.name for s in caps.missing_inputs + caps.invalid_inputs}
    cands = ctx.get("cands") or {}
    relevant = {"ambient_temperature", "compressor_state", "door_open", "defrost", "product_temperature",
                "relative_humidity"}
    if {"compressor_state"} & absent and request.passive_since is None:
        relevant.add("power_interruption")
    if ctx.get("needs_future_ambient") or (R.FO_VARYING in cands and
                                           cands[R.FO_VARYING].outcome != CandidateOutcome.REJECTED):
        relevant.add("future_ambient")
    regime_source = ("the declared passive period" if request.passive_since is not None else
                     "the event log's power-interruption records" if ctx.get("basis") == RegimeBasis.RECORDED_EVENT
                     else None)
    for st in caps.missing_inputs + caps.invalid_inputs:
        if st.name in relevant:
            text = f"{st.name} {st.status.value}: {INPUT[st.name].when_missing}"
            if st.name == "compressor_state" and regime_source:
                text += f"; here the regime evidence comes only from {regime_source}"
            out.missing_input_impact.append(text)
    winner = None if refused else ctx.get("winner")
    if winner:
        spec = R.BY_NAME[winner]
        if spec.kind == "BASELINE":
            out.model_limitations.append(f"{winner} is a baseline: not a physical model and not a validated physical "
                                         "forecast; it is never extrapolated beyond the validated horizon")
        out.model_limitations += list(spec.limitations)
    sel_h, h = ctx.get("sel_h"), ctx["request"].horizon_min
    if sel_h is not None:
        out.horizon_limitations.append(f"held-out checks reach {sel_h:g} min; requested {h:g} min")
        if h > sel_h and not refused:
            out.horizon_limitations.append(f"the forecast beyond {sel_h:g} min is an extrapolation allowed by the user "
                                           "(physical model only); nothing beyond it was checked against data")
    if ctx.get("longest_fit_min") is not None and h > ctx["longest_fit_min"] and not refused:
        out.horizon_limitations.append(f"the horizon is longer than the longest fitted segment "
                                       f"({ctx['longest_fit_min']:.0f} min)")
    return out


# ====================================================================== recommendations
def _rec(name: str, kind: str, rationale: str, capability: str, models: list[str], guidance: str, units: str | None,
         priority: Priority, why: str, necessity: Necessity) -> Recommendation:
    extra = {"expected_benefit": NO_EFFECT_BENEFIT[necessity]} if necessity in NO_EFFECT_BENEFIT else {}
    return Recommendation(missing_input=name, kind=kind, rationale=rationale, potential_capability=capability,
                          enabled_models=models, acquisition_guidance=guidance, units_and_resolution=units,
                          priority=priority, priority_rationale=why, necessity=necessity, **extra)


def recommendations(ctx: dict[str, Any], caps: InputCapabilities, code: RefusalCode | None) -> list[Recommendation]:
    request = ctx["request"]
    missing = {s.name for s in caps.missing_inputs}
    invalid = {s.name for s in caps.invalid_inputs}
    cands = ctx.get("cands") or {}
    no_cooling = request.question == ForecastQuestion.NO_COOLING
    passive_path = ctx.get("path") != "continuation"
    recs: list[Recommendation] = []
    nec, pot = Necessity.NECESSARY_FOR_THIS_QUESTION, Necessity.POTENTIAL_IMPROVEMENT
    ctx_only, future = Necessity.OPTIONAL_CONTEXT, Necessity.FUTURE_CAPABILITY
    fitted = cands.get(R.FO_FITTED)
    disturbed = bool(ctx.get("dropped_suspicious")) or code == RefusalCode.NO_ADEQUATE_MODEL or any(
        c.rejection_code == RefusalCode.POOR_FIT for c in cands.values())

    if ctx.get("no_passive_evidence") and no_cooling:
        name = "compressor_state" if "compressor_state" not in invalid else "compressor_state (complete)"
        has_ambient = "ambient_temperature" not in missing | invalid
        enabled = [R.FO_FITTED, *([R.FO_MEASURED, R.FO_VARYING] if has_ambient else [])]
        recs.append(_rec(
            name + " or power_interruption records", "INPUT",
            "the no-cooling question needs evidence of periods without active cooling; a temperature shape is "
            "never accepted as that evidence",
            "regime evidence for the passive first-order models"
            + ("" if has_ambient else " (the ambient-based variants would also need ambient_temperature)")
            + " and for the baselines from a passive origin; recorded compressor state, not power records alone, "
            "would also allow the active-cooling hindcast", enabled,
            "log the compressor run signal from the controller output, a dry contact or a current clamp, on the same "
            "logger clock; or supply an event log of power interruptions (start, end); or state passive_since",
            "on/off at the temperature sampling interval or finer; >= 90 % of readings valid", Priority.HIGH,
            "without it this question cannot be answered on this log", nec))
    if ("ambient_temperature" in missing | invalid) and passive_path and not ctx.get("no_passive_evidence"):
        blocked = (fitted is not None and fitted.rejection_code == RefusalCode.INSUFFICIENT_IDENTIFIABILITY and not any(
            c.outcome != CandidateOutcome.REJECTED for n, c in cands.items() if n in R.BY_NAME and
            R.BY_NAME[n].kind == "PHYSICAL" and n != R.FO_ACTIVE))
        recs.append(_rec(
            "ambient_temperature", "INPUT",
            "with a measured surroundings temperature the equilibrium no longer has to be inferred from curvature, "
            "which short or nearly straight rises cannot provide",
            "measured- and varying-ambient first-order models", [R.FO_MEASURED, R.FO_VARYING],
            "a temperature sensor in the space the room exchanges heat with (corridor, machine room), logged with "
            "the room sensor; record where it is", "°C at the room sampling interval or finer",
            Priority.HIGH if blocked else Priority.MEDIUM,
            "the fitted-equilibrium model was not identifiable on this log" if blocked else
            "the equilibrium is currently fitted or assumed", nec if blocked else pot))
    if ctx.get("needs_future_ambient"):
        recs.append(_rec(
            "future_ambient", "SCENARIO", "ambient varied during the fitted period; its future value is never "
            "extrapolated", "a forecast from the varying-ambient model", [R.FO_VARYING],
            "state the expected surroundings temperature over the horizon (e.g. a weather or building forecast); "
            "it is recorded as ASSUMED", "°C, constant over the horizon in this version", Priority.HIGH,
            "the selected model cannot forecast without it", nec))
    if code in (RefusalCode.INSUFFICIENT_EVIDENCE, RefusalCode.INSUFFICIENT_DATA):
        recs.append(_rec(
            "longer usable history before the origin", "DATA",
            "held-out checks need enough data after the fitting part: horizons are validated only as far as "
            "held-out stretches reach", "validation at the requested horizon", [],
            "keep logging through longer passive periods (or longer stretches for a continuation forecast), "
            "or request a horizon within the validated one", "same sampling; longer gap-free stretches",
            Priority.HIGH, "the requested forecast was refused for lack of evidence", nec))
    for name, what, deferred in (("door_open", "door contact switch on each door (open/close timestamps)",
                                  "door_infiltration (deferred; also needs door area and humidity)"),
                                 ("defrost", "defrost start/end from the controller or its event log",
                                  "defrost_aware (deferred)")):
        if name in missing:
            recs.append(_rec(
                name + " records", "INPUT",
                f"without {name.replace('_', ' ')} records such disturbances cannot be separated from the room's own "
                "behaviour and are never attributed to a cause",
                "marking disturbed periods so they are not fitted or used as truth (implemented, via --events); "
                + deferred, [], what + "; supply as an event log (kind, start, end, provenance)",
                "timestamps to the second, in the log's time basis", Priority.MEDIUM if disturbed else Priority.LOW,
                "unexplained disturbances or misfit were seen on this log" if disturbed else
                "no unexplained disturbance was detected on this log", pot))
    if "product_temperature" in missing:
        recs.append(_rec(
            "product_temperature and product_load records", "INPUT",
            "product thermal mass and loading are absorbed into the effective k and E; they cannot be modelled "
            "separately", "multi_node_rc and product_load (both deferred)", [],
            "a product-simulant probe and a loading log (mass, entry temperature, time)",
            "°C at the room sampling interval; mass in kg", Priority.LOW, "no implemented model would use it yet",
            future))
    meta = ctx.get("metadata")
    n_sensors = len(meta.sensors) if meta else 0
    ignored_temp = any("may be other temperature sensors" in w for w in caps.quality_warnings)
    if n_sensors > 1 or ignored_temp or (meta and meta.room_volume and n_sensors <= 1):
        recs.append(_rec(
            "spatial temperature coverage", "INPUT",
            "one sensor gives the air at one point; whether it represents the room cannot be assessed from it alone",
            "per-sensor forecasts and a documented basis for any room-representative value", [],
            "review sensor placement; run the forecast per sensor (--temperature-column) and compare; add sensors "
            "where the room may stratify (near doors, high and low)", "°C, same clock and sampling for all sensors",
            Priority.MEDIUM, "several sensors exist or the room size is documented", ctx_only))
    elif "sensors" in missing:
        recs.append(_rec(
            "sensor location metadata", "INPUT", "the analysed sensor's position is unknown, so what its readings "
            "represent cannot be stated", "disclosure of the measured point", [],
            "supply metadata: sensor id, column, location, height", None, Priority.LOW,
            "documentation only", ctx_only))
    if "relative_humidity" in missing:
        recs.append(_rec(
            "relative_humidity", "INPUT", "latent loads and coil frosting cannot be assessed",
            "humidity_latent (deferred)", [], "RH sensors inside and in the adjacent space", "%, 1-5 min",
            Priority.LOW, "no implemented model would use it yet", future))
    if "cooling_power" in missing and "compressor_state" not in missing | invalid:
        recs.append(_rec(
            "cooling_power", "INPUT", "a binary compressor state gives no power: b stays an effective °C/min",
            "interpreting the cooling term physically (not implemented)", [], "electrical power meter on the "
            "condensing unit", "W at the sampling interval", Priority.LOW, "no implemented model would use it yet",
            future))
    order = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}
    return sorted(recs, key=lambda r: (order[r.priority], NECESSITY_ORDER.index(r.necessity), r.missing_input))


def annotate(ctx: dict[str, Any], refused: bool, code: RefusalCode | None, reason: str | None) -> dict:
    caps = capabilities(ctx, refused)
    return {"question": ctx["request"].question, "input_capabilities": caps,
            "model_selection": selection(ctx, refused, code, reason),
            "limitation_details": limitation_details(ctx, caps, refused),
            "recommendations": recommendations(ctx, caps, code)}
