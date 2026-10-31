"""Application entry point.

    coldsense validate LOG.csv [--json] [--export DIR]   # data-quality report only
    coldsense events LOG.csv [--upper-threshold 8] [--json]   # event segmentation
    coldsense forecast LOG.csv --horizon-min 120 [--threshold 8] [--json]   # thermal forecast (Phase 4)
    coldsense thermal-models   # Phase 4 model registry: implemented and deferred models
    coldsense analyze LOG.csv --threshold 8 [--now 3.5] [--evidence REAL]
    coldsense modules      # registered domain modules and their model cards
    coldsense doctor       # environment and configuration check
    coldsense version
    coldsense kb ingest|query|list|remove|reindex|evaluate   # technical documentation (Phase 7)

Input options shared by validate and analyze: --evidence, --unit, --timezone,
--timestamp-format, --delimiter, --encoding. Each one used is recorded in the report.

Exit codes: 0 ok, 1 error (unreadable input, bad config, invalid event settings),
2 refused (prediction refused; for validate: data not suitable for analysis; for
events: segmentation refused, including settings that cannot be applied to this log;
for forecast: structured refusal, except INVALID_REQUEST which exits 1).
A refusal is a valid outcome, so it has its own code rather than looking like a crash.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from coldsense import __version__
from coldsense.core.errors import ColdSenseError, ConfigurationError
from coldsense.core.modules import all_modules
from coldsense.domains.cold_room import analyze  # importing registers the module
from coldsense.domains.cold_room.ingest import IngestOptions
from coldsense.domains.cold_room.pipeline import segment_source, validate_source
from coldsense.domains.cold_room.schemas import EvidenceClass, SegmentationParams
from coldsense.logs import configure_logging
from coldsense.settings import get_settings

log = logging.getLogger(__name__)

EXIT_OK, EXIT_ERROR, EXIT_REFUSED = 0, 1, 2


def _options(a: argparse.Namespace) -> IngestOptions:
    return IngestOptions(encoding=a.encoding, delimiter=a.delimiter, timestamp_format=a.timestamp_format,
                         timezone=a.timezone, temperature_unit=a.unit)


def _cmd_validate(a: argparse.Namespace) -> int:
    _, obs, rep = validate_source(a.csv, source=a.csv, evidence=EvidenceClass(a.evidence), options=_options(a))
    a.result_code = EXIT_OK if rep.suitable_for_downstream else EXIT_REFUSED
    if a.export:
        _export(a.export, a.csv, obs, rep)
    if a.json:
        print(rep.model_dump_json(indent=2))
        return a.result_code
    i, c, iv = rep.input, rep.counts, rep.intervals
    print(f"QUALITY  {rep.verdict.value}  [{rep.evidence.value}]  suitable for analysis: "
          f"{'yes' if rep.suitable_for_downstream else 'NO'}")
    print(f"  input     {i.source}  sha256 {i.sha256[:16]}…  {i.size_bytes} bytes, {i.encoding}, "
          f"delimiter {i.delimiter!r}")
    print(f"  columns   {i.column_mapping}  ignored {i.ignored_columns}")
    print(f"  units     temperature °{i.temperature_unit} ({i.temperature_unit_source})")
    print(f"  time      format {i.timestamp_format} ({i.timestamp_format_source}); {i.timezone_handling}")
    print(f"  rows      read {c.data_rows_read}, usable {c.rows_usable}, excluded {c.rows_excluded}, "
          f"suspicious {c.rows_suspicious}, malformed {c.rows_malformed}, short {c.rows_short}")
    print(f"  missing   {rep.missing_by_field}")
    print(f"  invalid   {rep.invalid_by_field}")
    print(f"  span      {rep.start} → {rep.end} ({rep.duration_hours:.1f} h), coverage {rep.coverage_pct}%")
    print(f"  intervals median {iv.median_s} s, p05 {iv.p05_s}, p95 {iv.p95_s}, max {iv.max_s}; "
          f"irregular {iv.n_irregular} ({iv.irregular_pct}%), gaps {iv.n_gaps}, out of order {iv.n_out_of_order}")
    for x in rep.issues:
        print(f"  {x.severity.value:8} {x.code}: {x.message}")
    for t in rep.transformations:
        print(f"  CHANGED  {t.code} ({t.rows_affected} rows): {t.description}")
    for lim in rep.limitations:
        print(f"  LIMIT    {lim}")
    return a.result_code


def _export(out_dir: str, src: str, obs, rep) -> None:
    """Write observations, report and (if any) malformed rows' raw text; never overwrite."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    stem = Path(src).stem
    targets = [d / f"{stem}.observations.csv", d / f"{stem}.quality.json"]
    if rep.malformed_records:
        targets.append(d / f"{stem}.malformed.csv")
    for t in targets:
        if t.exists() or t.resolve() == Path(src).resolve():
            raise ColdSenseError(f"refusing to overwrite {t}")
    obs.to_csv(targets[0], index=False)
    targets[1].write_text(rep.model_dump_json(indent=2), encoding="utf-8")
    if rep.malformed_records:
        pd.DataFrame([m.model_dump() for m in rep.malformed_records]).to_csv(targets[2], index=False)
    print("exported " + ", ".join(str(t) for t in targets), file=sys.stderr)


# SegmentationParams fields that `coldsense events` exposes, and how to fix each
_EVENT_OPTIONS = {
    "upper_threshold_c": ("--upper-threshold", "a finite °C value above --lower-threshold"),
    "lower_threshold_c": ("--lower-threshold", "a finite °C value below --upper-threshold; write a negative "
                                               "value as --lower-threshold=-5"),
    "min_excursion_min": ("--min-excursion-min", "a finite number of minutes, 0 or more"),
}


def _settings_error(e: ValidationError) -> ConfigurationError:
    """Name each invalid field and its command-line option; no traceback."""
    parts = []
    for x in e.errors():
        msg = x["msg"].removeprefix("Value error, ")
        fields = [str(f) for f in x["loc"]] or [f for f in SegmentationParams.model_fields if f in msg]
        if x["loc"]:
            msg += f" (got {x['input']!r})"
        opts = [_EVENT_OPTIONS[f] for f in fields if f in _EVENT_OPTIONS]
        where = " / ".join(o for o, _ in opts) or " / ".join(fields)
        hint = "; ".join(h for _, h in opts)
        parts.append(f"{where}: {msg}" + (f" (use {hint})" if hint else ""))
    return ConfigurationError("invalid event settings: " + " | ".join(parts))


def _with_option(reason: str) -> str:
    """Point a refusal caused by a setting at the command-line option that sets it."""
    for field, (opt, _) in _EVENT_OPTIONS.items():
        if reason.startswith(f"setting {field} "):
            return f"{reason} [{opt}]"
    return reason


def _cmd_events(a: argparse.Namespace) -> int:
    try:
        params = SegmentationParams(upper_threshold_c=a.upper_threshold, lower_threshold_c=a.lower_threshold,
                                    **({"min_excursion_min": a.min_excursion_min} if a.min_excursion_min is not None
                                       else {}))
    except ValidationError as e:
        raise _settings_error(e) from None
    _, _, rep, seg = segment_source(a.csv, source=a.csv, evidence=EvidenceClass(a.evidence),
                                    options=_options(a), params=params)
    a.result_code = EXIT_REFUSED if seg.refused else EXIT_OK
    if a.json:
        print(seg.model_dump_json(indent=2))
        return a.result_code
    print(f"EVENTS  [{seg.evidence.value}]  data {rep.verdict.value}  time basis: {seg.time_basis}")
    if seg.refused:
        print("  REFUSED")
        for r in seg.reasons:
            print(f"  - {_with_option(r)}")
        return a.result_code
    print("  " + ", ".join(f"{k}={v}" for k, v in seg.summary.items()))
    for e in seg.events:
        cens = ("<" if e.start_censored else "") + ("…>" if e.end_censored else "")
        dur = "—" if e.duration_min is None else f"{e.duration_min:.1f} min"
        print(f"  {e.event_id} {e.type.value:22} {e.start:%Y-%m-%d %H:%M} → {e.end:%H:%M} {dur:>10} {cens:3} "
              f"n={e.n_observations:<4} {e.rationale}")
    for lim in seg.limitations:
        print(f"  LIMIT    {lim}")
    return a.result_code


# ForecastRequest fields that `coldsense forecast` exposes
_FORECAST_OPTIONS = {
    "horizon_min": "--horizon-min", "threshold_c": "--threshold", "direction": "--direction",
    "origin": "--origin", "passive_since": "--passive-since", "assumed_equilibrium_c": "--assume-equilibrium",
    "future_ambient_c": "--future-ambient", "step_min": "--step-min", "allow_extrapolation": "--allow-extrapolation",
    "question": "--question",
}


def _timestamp(text: str | None):
    if text is None:
        return None
    try:
        ts = pd.Timestamp(text)
    except (ValueError, TypeError) as e:
        raise ConfigurationError(f"invalid forecast request: cannot read time {text!r} ({e})") from None
    if pd.isna(ts):
        raise ConfigurationError(f"invalid forecast request: cannot read time {text!r} (not a time)")
    return ts.to_pydatetime()


def _forecast_request(a: argparse.Namespace):
    from coldsense.domains.cold_room.thermal import Direction, ForecastRequest

    raw = {"horizon_min": a.horizon_min, "threshold_c": a.threshold, "origin": _timestamp(a.origin),
           "direction": Direction.DOWNWARD if a.direction == "down" else Direction.UPWARD,
           "passive_since": _timestamp(a.passive_since), "assumed_equilibrium_c": a.assume_equilibrium,
           "future_ambient_c": a.future_ambient, "step_min": a.step_min,
           "allow_extrapolation": a.allow_extrapolation or None,
           "question": None if a.question is None else a.question.upper().replace("-", "_")}
    try:
        return ForecastRequest(**{k: v for k, v in raw.items() if v is not None})
    except ValidationError as e:
        parts = []
        for x in e.errors():
            msg = x["msg"].removeprefix("Value error, ")
            fields = [str(v) for v in x["loc"]] or [k for k in _FORECAST_OPTIONS if k in msg]
            where = " / ".join(_FORECAST_OPTIONS.get(k, k) for k in fields) or "request"
            parts.append(f"{where}: {msg}" + (f" (got {x['input']!r})" if x["loc"] else ""))
        raise ConfigurationError("invalid forecast request: " + " | ".join(parts)) from None


def _forecast_inputs(a: argparse.Namespace):
    """Optional Phase 4 adapters: event log (CSV) and room metadata (JSON). Errors exit 1, nothing is guessed."""
    from coldsense.domains.cold_room.thermal.inputs import parse_events_csv, parse_metadata_json

    events = metadata = None
    for attr, parse, label in (("events", parse_events_csv, "event log"), ("metadata", parse_metadata_json,
                                                                           "metadata")):
        path = getattr(a, attr)
        if path is None:
            continue
        try:
            text = Path(path).read_text(encoding="utf-8")
            value = parse(text)
        except FileNotFoundError:
            raise ConfigurationError(f"{label} file not found: {path}") from None
        except (OSError, UnicodeDecodeError, ValueError) as e:
            raise ConfigurationError(f"invalid {label} ({path}): {e}") from None
        if attr == "events":
            events = value
        else:
            metadata = value
    return events, metadata


def _reliability_config(a: argparse.Namespace):
    """Phase 5 configuration, or None for a plain Phase 4 forecast."""
    if not a.reliability:
        if a.interval_level is not None:
            raise ConfigurationError("--interval-level applies only with --reliability")
        return None
    from coldsense.domains.cold_room.thermal.schemas import ReliabilityConfig

    try:
        return ReliabilityConfig() if a.interval_level is None else ReliabilityConfig(interval_level=a.interval_level)
    except ValidationError as e:
        raise ConfigurationError("invalid --interval-level: " + "; ".join(
            x["msg"] for x in e.errors()) + f" (got {a.interval_level!r})") from None


def _print_reliability(res) -> None:
    """Phase 5 section, appended after the Phase 4 output (which is unchanged)."""
    rr = res.reliability_report
    ra, u, c = rr.reliability, rr.uncertainty, rr.crossing
    print(f"  RELIABILITY {ra.status.value}  [{ra.validation_scope.value}]  rule-based classification, not a "
          "probability")
    if rr.refusal_or_downgrade_reason:
        print(f"  because  {rr.refusal_or_downgrade_reason}")
    if u.interval_lower_c is not None:
        print(f"  interval {u.nominal_coverage:.0%} nominal at +{u.horizon_min:g} min: {u.interval_lower_c:.2f} .. "
              f"{u.interval_upper_c:.2f} °C ({u.status.value}; evaluation coverage {u.empirical_coverage:.3f}, "
              f"95 % CI {u.coverage_ci[0]:.3f}..{u.coverage_ci[1]:.3f}, {u.evaluation_periods} evaluation "
              "period(s)); conditional on: " + "; ".join(u.conditional_on))
    elif rr.forecast_available:
        print(f"  interval none: {u.status.value} (calibration {u.calibration_periods}, evaluation "
              f"{u.evaluation_periods} passive period(s))")
    if rr.forecast_available and c.status.value not in ("NO_THRESHOLD",):
        span = f" {c.lower_min:.1f} .. {c.upper_min:.1f} min" if c.lower_min is not None else ""
        print(f"  crossing-time {c.status.value}{span}")
    p = rr.parameter_uncertainty
    if p.status == "ASSESSED":
        print(f"  parameters envelope {p.envelope_lower_c:.2f} .. {p.envelope_upper_c:.2f} °C at +{p.horizon_min:g} "
              f"min ({p.method}; not a prediction interval)")
    for g in ra.gates:
        if g.outcome.value == "FAIL":
            print(f"  GATE FAIL {g.name} (cap {g.cap_if_failed.value}): {g.evidence}")
    if rr.causal_view.truncated:
        print(f"  causal   {rr.causal_view.note}")
    for r in rr.recommendations:
        print(f"  EVIDENCE [{r.priority.value}] {r.missing_input} ({r.necessity.value}): {r.rationale}. "
              f"Retry when: {r.retry_conditions}")


def _cmd_forecast(a: argparse.Namespace) -> int:
    from coldsense.domains.cold_room.thermal import ForecastStatus, RefusalCode, forecast_source

    request = _forecast_request(a)
    config = _reliability_config(a)
    events, metadata = _forecast_inputs(a)
    options = _options(a)
    if a.temperature_column:
        options = IngestOptions(**{**options.model_dump(), "column_map": {"temperature": a.temperature_column}})
    if config is None:
        _, _, rep, res = forecast_source(a.csv, source=a.csv, evidence=EvidenceClass(a.evidence), options=options,
                                         request=request, events=events, metadata=metadata)
    else:   # Phase 5: the same Phase 4 forecast on the causal view of the log, plus its reliability assessment
        from coldsense.domains.cold_room.thermal.reliability import assess_source

        _, _, rep, res = assess_source(a.csv, source=a.csv, evidence=EvidenceClass(a.evidence), options=options,
                                       request=request, events=events, metadata=metadata, config=config)
    if res.status == ForecastStatus.FORECAST:
        a.result_code = EXIT_OK
    else:
        a.result_code = EXIT_ERROR if res.refusal_code == RefusalCode.INVALID_REQUEST else EXIT_REFUSED
    if a.json:
        print(res.model_dump_json(indent=2))
        return a.result_code
    validated = "" if res.validated_horizon_min is None else f" (validated up to {res.validated_horizon_min:g} min)"
    print(f"FORECAST  [{res.evidence.value}]  data {rep.verdict.value}  regime {res.operating_regime.value}  "
          f"horizon {res.prediction_horizon_min:g} min{validated}")
    for ev in res.regime_evidence:
        print(f"  regime   {ev}")
    if res.status != ForecastStatus.FORECAST:
        print(f"  REFUSED  {res.refusal_code.value}: {res.refusal_reason}")
        print(f"  failed   {res.refusal_failed_condition}")
        if res.refusal_needed_to_retry:
            print(f"  needed   {res.refusal_needed_to_retry}")
    else:
        t0 = res.initial_temperature
        print(f"  model    {res.selected_model} v{res.selected_model_version}; {res.solution_method}")
        print(f"  start    {res.prediction_start}  {t0.value:.2f} °C ({t0.provenance.value}, {t0.note})")
        for p in res.parameter_estimates:
            rng = ("" if p.profile_range is None
                   else f", profile range {p.profile_range[0]:.4g}..{p.profile_range[1]:.4g}")
            val = "unavailable" if p.value is None else f"{p.value:.6g}"
            print(f"  param    {p.name} = {val} {p.unit} ({p.provenance.value}{rng}) {p.note}".rstrip())
        pts = res.trajectory
        idx = sorted({round(i * (len(pts) - 1) / 8) for i in range(9)})      # up to 9 evenly spaced points
        print(f"  trajectory ({len(pts)} points) " + ", ".join(f"+{pts[i].minutes:g} min {pts[i].temp_c:.2f} °C"
                                                               for i in idx))
        for sc in res.scenario:
            print(f"  scenario {sc}")
    if res.threshold is not None:
        when = (f" at {res.predicted_crossing_time} (+{res.predicted_crossing_minutes:.1f} min)"
                if res.predicted_crossing_time else "")
        print(f"  crossing {res.threshold.value_c:g} °C {res.threshold.direction.value}: "
              f"{res.crossing_status.value}{when}")
    vd = res.validation_diagnostics
    if vd is not None:
        print(f"  validation {vd.status}: {vd.reason}")
    for c in res.candidate_models:
        why = f" {c.rejection_code.value}: {c.rejection_reason}" if c.rejection_code else ""
        print(f"  model    {c.name:34} {c.outcome.value}{why}")
    for w in res.warnings:
        print(f"  WARNING  {w}")
    for lim in res.limitations:
        print(f"  LIMIT    {lim}")
    # ---- follow-up sections (appended: the lines above are unchanged)
    ms = res.model_selection
    if ms is not None:
        print(f"  question {ms.question.value}; forecast kind {ms.forecast_kind}")
        if ms.fallback_reason:
            print(f"  fallback {ms.fallback_reason}")
    caps = res.input_capabilities
    if caps is not None:
        print("  inputs   available: " + ", ".join(f"{i.name} ({i.status.value})" for i in caps.available_inputs))
        if caps.invalid_inputs:
            print("  inputs   invalid: " + ", ".join(i.name for i in caps.invalid_inputs))
        for w in caps.quality_warnings:
            print(f"  QUALITY  {w}")
    ld = res.limitation_details
    for x in (ld.missing_input_impact + ld.horizon_limitations) if ld else []:
        print(f"  IMPACT   {x}")
    for r in res.recommendations:
        print(f"  RECOMMEND [{r.priority.value}] {r.missing_input} ({r.necessity.value}): {r.potential_capability}. "
              f"How: {r.acquisition_guidance}")
    if res.reliability_report is not None:
        _print_reliability(res)
    return a.result_code


def _cmd_diagnose(a: argparse.Namespace) -> int:
    """Phase 6: observed facts, hypotheses (never diagnoses) and ranked next checks; exit 2 on INSUFFICIENT_EVIDENCE."""
    from coldsense.domains.cold_room.diagnostics import DiagnosisStatus, diagnose_source

    try:
        params = SegmentationParams(upper_threshold_c=a.upper_threshold, lower_threshold_c=a.lower_threshold,
                                    **({"min_excursion_min": a.min_excursion_min} if a.min_excursion_min is not None
                                       else {}))
    except ValidationError as e:
        raise _settings_error(e) from None
    if a.horizon_min is None and (a.origin or a.passive_since or a.question):
        raise ConfigurationError("--origin, --passive-since and --question apply only with --horizon-min")
    events, metadata = _forecast_inputs(a)
    request = _forecast_request(a) if a.horizon_min is not None else None
    rep, _, _, d = diagnose_source(a.csv, source=a.csv, evidence=EvidenceClass(a.evidence), options=_options(a),
                                   params=params, operational_events=events, metadata=metadata, request=request)
    a.result_code = EXIT_REFUSED if d.status == DiagnosisStatus.INSUFFICIENT_EVIDENCE else EXIT_OK
    if a.json:
        print(d.model_dump_json(indent=2))
        return a.result_code
    print(f"DIAGNOSIS {d.status.value}  [{d.data_evidence.value}]  data {rep.verdict.value}  time basis: "
          f"{d.time_basis}  (hypotheses, not diagnoses; no probabilities)")
    if d.refusal_reason:
        print(f"  because  {_with_option(d.refusal_reason)}")
    for f in d.observed_facts:
        span = (f" {f.start:%Y-%m-%d %H:%M} → {f.end:%H:%M}" if f.start and f.end and f.start.date() == f.end.date()
                else f" {f.start:%Y-%m-%d %H:%M} → {f.end:%Y-%m-%d %H:%M}" if f.start and f.end else "")
        lines = f" lines {min(f.source_lines)}–{max(f.source_lines)}" if f.source_lines else ""
        print(f"  FACT     {f.fact_id} {f.source.value} {f.kind}{span}{lines} [{f.evidence.value}]: {f.statement}")
    for f in d.findings:
        print(f"  FINDING  {f.finding_id} {f.status.value}: {f.statement}")
        for x in f.explanations:
            mark = "±" if x.supporting and x.contradicting else "+" if x.supporting else "-" if x.contradicting else "?"
            detail = "; ".join(x.supporting + x.contradicting) or ("needs " + "; ".join(x.missing) if x.missing
                                                                    else "; ".join(x.neutral))
            nxt = f"  → next check {x.required_next_check}" if x.required_next_check and f.hypothesis_ids else ""
            print(f"    {mark} {x.code.value}: {detail}{nxt}")
        if f.context_fact_ids:
            print(f"    CONTEXT {' '.join(f.context_fact_ids)} (temperature behaviour and gaps over the window; "
                  "observations, not evidence)")
        for lim in f.limitations:
            print(f"    LIMIT {lim}")
    if d.next_best_check:
        print(f"  NEXT BEST CHECK  {d.next_best_check} (rank 1 of {len(d.next_checks)}; a work priority, not a "
              "confidence in any explanation)")
    for c in d.next_checks:
        ref = f" (Phase 4 advice: {', '.join(c.phase4_recommendations)})" if c.phase4_recommendations else ""
        print(f"  CHECK {c.rank}  [{c.effort.value}] {c.check_id} ({c.open_items} open item(s); evidence value "
              f"{c.evidence_value.value}; safety {c.safety.value}): {c.action}. Needs {c.required_input}"
              + (f" [{c.units}]" if c.units else "")
              + (f", {c.resolution_or_duration}" if c.resolution_or_duration else "") + ".")
        print(f"    WHY                   {c.why_useful}{ref}")
        for label, xs in (("EVIDENCE", c.evidence), ("EXPECTED INFORMATION", c.expected_information),
                          ("SAFETY", [c.safety_note])):
            for k, x in enumerate(xs):
                print(f"    {label if k == 0 else '':21} {x}")
    for x in d.data_quality_limitations:
        print(f"  DATA     {x}")
    for x in d.limitations:
        print(f"  LIMIT    {x}")
    return a.result_code


def _cmd_thermal_models(_: argparse.Namespace) -> int:
    from coldsense.domains.cold_room.thermal import REGISTRY

    for m in REGISTRY:
        print(f"{m.name}  v{m.version}  {m.kind}  {m.status}")
        print(f"  equation  {m.equation}")
        print(f"  inputs    {'; '.join(m.required_inputs)}")
        if m.missing_for_implementation:
            print(f"  needs     {'; '.join(m.missing_for_implementation)}")
    print()
    print("availability and eligibility (canonical inputs: see docs/THERMAL_MODELS.md)")
    for m in REGISTRY:
        print(f"{m.name}  {m.availability}")
        print(f"  requires  {', '.join(m.required_input_names)}")
        if m.optional_input_names:
            print(f"  optional  {', '.join(m.optional_input_names)}")
        if m.eligibility:
            print(f"  eligible  {m.eligibility}")
        for x in m.missing_input_limitations:
            print(f"  missing   {x}")
        for x in m.refusal_conditions:
            print(f"  refuses   {x}")
    return EXIT_OK


def _add_input_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("csv")
    p.add_argument("--evidence", choices=[e.value for e in EvidenceClass], default="UNVALIDATED",
                   help="what kind of data this file is (declared by you; never inferred upward)")
    p.add_argument("--unit", choices=["C", "F", "K"], default=None, help="temperature unit, if not in the header")
    p.add_argument("--timezone", default=None, help="IANA zone of naive timestamps, e.g. Africa/Algiers")
    p.add_argument("--timestamp-format", default=None, help="strftime format, e.g. '%%d/%%m/%%Y %%H:%%M'")
    p.add_argument("--delimiter", default=None, help="column delimiter, if detection is wrong")
    p.add_argument("--encoding", default=None, help="text encoding, if detection is wrong")


def _cmd_analyze(a: argparse.Namespace) -> int:
    r = analyze(a.csv, source=a.csv, evidence=EvidenceClass(a.evidence), options=_options(a))
    p = r.predict(a.threshold, a.now)
    # decide the outcome before printing, so a broken pipe can't change it
    a.result_code = EXIT_REFUSED if p.refused else EXIT_OK
    rep = r.report
    print(f"DATA  {rep.verdict.value}  [{rep.evidence.value}]  {rep.duration_hours:.1f} h, "
          f"{rep.n_rows_clean} rows, coverage {rep.coverage_pct}%")
    for i in rep.issues:
        print(f"  {i.severity.value:8} {i.code}: {i.message}")
    clean = [s for s in r.segments if s.clean]
    print(f"SEGMENTS  {len(clean)} clean of {len(r.segments)} candidate warming periods")

    print(f"PREDICT  [{p.evidence.value}] {p.method}")
    if p.refused:
        print("  REFUSED")
        for x in p.reasons:
            print(f"  - {x}")
        return EXIT_REFUSED
    print(f"  {p.current_temp_c} °C -> {p.threshold_c} °C in ~{p.minutes_to_threshold} min "
          f"(observed-spread range {p.low_minutes}–{p.high_minutes} min)")
    for w in p.warnings:
        print(f"  WARNING: {w}")
    print("  assumptions: " + json.dumps(p.assumptions, ensure_ascii=False))
    return EXIT_OK


def _kb_store(a: argparse.Namespace):
    from coldsense.knowledge import KnowledgeStore, get_provider

    name = getattr(a, "embedding", None)
    return KnowledgeStore(Path(a.store) if a.store else get_settings().data_dir / "knowledge" / "kb.sqlite",
                          get_provider(name) if name else None)


def _cmd_kb_ingest(a: argparse.Namespace) -> int:
    """Phase 7: index technical documents (TXT, Markdown, DOCX; PDF with the optional pypdf package)."""
    from coldsense.knowledge import DocumentMetadata, IngestOutcome

    if a.doc_id and len(a.files) > 1:
        raise ConfigurationError("--doc-id applies to one file at a time")
    try:
        meta = DocumentMetadata(doc_id=a.doc_id, title=a.title, doc_type=a.doc_type, manufacturer=a.manufacturer,
                                model=a.model, revision=a.revision, date=a.date, supplied_by=a.supplied_by)
    except ValidationError as e:
        raise ConfigurationError("invalid document metadata: " + "; ".join(
            f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors())) from None
    reports = []
    with _kb_store(a) as store:
        for f in a.files:
            reports.append(store.ingest(Path(f), meta))
    a.result_code = EXIT_OK if all(r.outcome in (IngestOutcome.INDEXED, IngestOutcome.UNCHANGED)
                                   for r in reports) else EXIT_REFUSED
    if a.json:
        print(json.dumps([r.model_dump(mode="json") for r in reports], indent=2))
        return a.result_code
    for r in reports:
        extra = (f" v{r.version}" if r.version else "") + (f" (supersedes v{r.superseded_version})"
                                                           if r.superseded_version else "")
        print(f"{r.outcome.value:17} {r.filename} -> {r.doc_id}{extra}  extraction {r.extraction_status.value}"
              + (f"  chunks {r.n_chunks}, quarantined {r.n_quarantined}, trust {r.trust.value}" if r.trust else ""))
        if r.flags:
            print(f"  flags    {', '.join(f'{k} x{v}' for k, v in r.flags.items())}")
        if r.detail:
            print(f"  detail   {r.detail}")
    return a.result_code


def _cmd_kb_query(a: argparse.Namespace) -> int:
    from coldsense.knowledge import AnswerStatus, RetrievalParams, Retriever

    with _kb_store(a) as store:
        ans = Retriever(store, RetrievalParams(k=a.k)).answer(a.query)
    a.result_code = EXIT_REFUSED if ans.status == AnswerStatus.NO_RELEVANT_DOCUMENTATION else EXIT_OK
    if a.json:
        print(ans.model_dump_json(indent=2))
        return a.result_code
    print(f"ANSWER {ans.status.value}  [embedding {ans.embedding_model}; generator {ans.generator}]")
    if ans.reason:
        print(f"  because  {ans.reason}")
    for c in ans.claims:
        cit = c.citations[0]
        rev = f" rev {cit.revision}" if cit.revision else ""
        print(f"  {c.claim_type.value}  \"{c.text}\"\n      source: {cit.title}{rev}, {cit.location} [{cit.chunk_id}]")
    for x in ans.conflicts:
        srcs = "; ".join(f"{s.citations[0].title} rev {s.citations[0].revision or '?'}: \"{s.text}\""
                         for s in x.statements)
        print(f"  CONFLICT ({x.quantity_unit}) {srcs}\n      {x.note}")
    for p in ans.passages:
        cit = p.citation
        rev = f" rev {cit.revision}" if cit.revision else ""
        print(f"  PASSAGE {p.rank}  {cit.title}{rev} ({cit.doc_type.value}), {cit.location} [{cit.chunk_id}]  "
              f"trust {p.trust.value}; similarity {p.vector_similarity:.3f}, bm25 {p.lexical_score:.2f} (ranking only)"
              + (f"; admitted by {', '.join(p.relevance_basis)}" if p.relevance_basis else ""))
        print(f"      {p.excerpt}")
    print(f"  POLICY   {ans.data_policy}")
    print(f"  SCORES   {ans.score_note}")
    for x in ans.limitations:
        print(f"  LIMIT    {x}")
    return a.result_code


def _cmd_kb_list(a: argparse.Namespace) -> int:
    with _kb_store(a) as store:
        docs = store.documents(include_inactive=a.all)
    if a.json:
        print(json.dumps([d.model_dump(mode="json") for d in docs], indent=2))
        return EXIT_OK
    for d in docs:
        state = "active" if d.active else "inactive"
        meta = ", ".join(f"{k} {v}" for k, v in (("manufacturer", d.manufacturer), ("model", d.model),
                                                  ("revision", d.revision), ("date", d.date)) if v)
        print(f"{d.doc_id} v{d.version} [{state}, {d.trust.value}] {d.display_title()} ({d.doc_type.value}; "
              f"{d.format}; {d.n_chunks} chunks, {d.n_quarantined} quarantined) sha256 {d.sha256[:16]}…"
              + (f"\n  {meta} (supplied)" if meta else ""))
    if not docs:
        print("no documents")
    return EXIT_OK


def _cmd_kb_remove(a: argparse.Namespace) -> int:
    with _kb_store(a) as store:
        n = store.remove(a.doc_id)
    print(f"removed {a.doc_id}: {n} chunk(s) deleted; a tombstone keeps its id and hash")
    return EXIT_OK


def _cmd_kb_reindex(a: argparse.Namespace) -> int:
    """Copy a store into a new store with vectors from another embedding model (the source is not changed)."""
    from coldsense.knowledge import get_provider

    with _kb_store(argparse.Namespace(store=a.store)) as store:
        r = store.reindex(Path(a.to), get_provider(a.embedding), replace=a.replace)
    print(f"REINDEXED {r['documents']} active document(s), {r['chunks_embedded']} chunk(s) embedded with "
          f"{r['embedding_model']} -> {r['target']}")
    return EXIT_OK


def _cmd_kb_evaluate(a: argparse.Namespace) -> int:
    from coldsense.knowledge.evaluation import evaluate, evaluate_dataset, evaluate_split

    if a.split or a.dataset:
        s = (evaluate_dataset(a.dataset, a.embedding or "lexical-hash-v1") if a.dataset
             else evaluate_split(a.split, a.embedding or "lexical-hash-v1"))
        if a.json:
            print(s.model_dump_json(indent=2))
            return EXIT_OK
        print(f"EVALUATION  {s.split} {s.dataset} (sha256 {s.dataset_sha256[:12]}) on {s.corpus} (SYNTHETIC, "
              f"fictional); embedding {s.embedding_model}")
        groups = (list(s.metrics["retrieval"].items()) + [(f"answer/{k}", v) for k, v in
                                                          s.metrics["answerability"].items()]
                  if "answerability" in s.metrics else list(s.metrics.items()))
        print(f"  generator {s.generator}")
        for name, m in groups:
            print(f"  {name:28} " + ", ".join(f"{k} {v}" for k, v in m.items()))
        print(f"  quarantine leaks {s.quarantine_leaks}; wrong-model passages {s.wrong_model_errors}")
        print(f"  NOTE     {s.note}")
        return EXIT_OK
    r = evaluate()
    if a.json:
        print(r.model_dump_json(indent=2))
        return EXIT_OK
    print(f"EVALUATION  corpus {r.corpus} (SYNTHETIC, fictional): {r.n_documents} documents, {r.n_chunks} chunks, "
          f"{r.n_quarantined} quarantined; embedding {r.embedding_model}")
    for cat, m in r.metrics.items():
        print(f"  {cat:22} " + ", ".join(f"{k} {v}" for k, v in m.items()))
    print(f"  quarantine leaks       {r.quarantine_leaks}")
    print(f"  NOTE     {r.note}")
    return EXIT_OK


def _cmd_modules(_: argparse.Namespace) -> int:
    for m in all_modules():
        print(f"{m.key}  v{m.version}  {m.title}")
        for c in m.model_cards:
            print(f"  model {c.name} v{c.version}  validation: {c.validation_evidence.value} "
                  f"({c.validation_summary})")
    return EXIT_OK


def _cmd_doctor(_: argparse.Namespace) -> int:
    s = get_settings()
    print(f"coldsense {__version__}  python {platform.python_version()}  env={s.env}")
    ok = True
    for pkg in ("pandas", "numpy", "scipy", "pydantic", "pydantic-settings"):
        try:
            print(f"  ok       {pkg} {metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            print(f"  MISSING  {pkg}")
            ok = False
    print(f"  data_dir       {s.data_dir.resolve()}  exists={s.data_dir.exists()}")
    print(f"  real_data_dir  {s.real_data_dir.resolve()}  exists={s.real_data_dir.exists()}")
    print(f"  modules        {[m.key for m in all_modules()]}")
    print(f"  logging        level={s.log_level} format={s.log_format}")
    return EXIT_OK if ok else EXIT_ERROR


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="coldsense", description="Evidence-first cold-room decision support")
    sub = ap.add_subparsers(dest="command", required=True)

    va = sub.add_parser("validate", help="ingest and validate a log; print the data-quality report")
    _add_input_args(va)
    va.add_argument("--json", action="store_true", help="print the full report as JSON")
    va.add_argument("--export", metavar="DIR", default=None,
                    help="also write <name>.observations.csv and <name>.quality.json into DIR (never overwrites)")
    va.set_defaults(func=_cmd_validate)

    ev = sub.add_parser("events", help="validate a log and segment it into traceable events")
    _add_input_args(ev)
    ev.add_argument("--upper-threshold", type=float, default=None, help="°C; report excursions above it")
    ev.add_argument("--lower-threshold", type=float, default=None, help="°C; report excursions below it")
    ev.add_argument("--min-excursion-min", type=float, default=None,
                    help="minutes beyond a limit to count as sustained (provisional default 10)")
    ev.add_argument("--json", action="store_true", help="print the full segmentation result as JSON")
    ev.set_defaults(func=_cmd_events)

    fc = sub.add_parser("forecast", help="physics-aware forecast for a no-cooling scenario, or a structured refusal")
    _add_input_args(fc)
    fc.add_argument("--horizon-min", type=float, required=True, help="forecast horizon in minutes (explicit)")
    fc.add_argument("--threshold", type=float, default=None, help="°C; report when it is crossed")
    fc.add_argument("--direction", choices=["up", "down"], default="up",
                    help="up: crossing = rising to/above the threshold (default); down: falling to/below")
    fc.add_argument("--origin", default=None, help="forecast from this time (log's time basis); default: last reading")
    fc.add_argument("--passive-since", default=None,
                    help="declare no active cooling from this time to the origin (an assumption, e.g. a power cut)")
    fc.add_argument("--assume-equilibrium", type=float, default=None,
                    help="°C the room would settle at without cooling (assumption)")
    fc.add_argument("--future-ambient", type=float, default=None, help="°C ambient over the horizon (scenario)")
    fc.add_argument("--step-min", type=float, default=None, help="trajectory spacing in minutes")
    fc.add_argument("--allow-extrapolation", action="store_true",
                    help="allow a horizon beyond the one checked on held-out data (physical models only; flagged)")
    fc.add_argument("--question", choices=["no-cooling", "continuation"], default=None,
                    help="no-cooling (default): no active cooling from the origin; continuation: the observed "
                         "behaviour continues")
    fc.add_argument("--events", default=None, metavar="CSV",
                    help="optional event log: kind,start,end,provenance[,source,note] (door, defrost, power ...)")
    fc.add_argument("--metadata", default=None, metavar="JSON", help="optional room/sensor metadata (disclosed only)")
    fc.add_argument("--temperature-column", default=None,
                    help="header of the sensor column to analyse (one sensor per run; sensors are never averaged)")
    fc.add_argument("--reliability", action="store_true",
                    help="Phase 5: assess uncertainty and reliability (causal view of the log up to the origin)")
    fc.add_argument("--interval-level", type=float, default=None, metavar="P",
                    help="nominal level of the empirical interval with --reliability (0.5-0.99, default 0.8); an "
                         "evaluation target, not a guaranteed coverage")
    fc.add_argument("--json", action="store_true", help="print the full structured result as JSON")
    fc.set_defaults(func=_cmd_forecast)
    sub.add_parser("thermal-models", help="list Phase 4 thermal models, implemented and deferred").set_defaults(
        func=_cmd_thermal_models)

    dg = sub.add_parser("diagnose", help="facts, cautious hypotheses and ranked next checks for excursions beyond a "
                                         "limit (Phase 6; never a diagnosis of cause)")
    _add_input_args(dg)
    dg.add_argument("--upper-threshold", type=float, default=None, help="°C; diagnose sustained excursions above it")
    dg.add_argument("--lower-threshold", type=float, default=None, help="°C; diagnose sustained excursions below it")
    dg.add_argument("--min-excursion-min", type=float, default=None,
                    help="minutes beyond a limit to count as sustained (provisional default 10)")
    dg.add_argument("--events", default=None, metavar="CSV",
                    help="optional event log: kind,start,end,provenance[,source,note] (door, defrost, power ...)")
    dg.add_argument("--metadata", default=None, metavar="JSON", help="optional room/sensor metadata (disclosed only)")
    dg.add_argument("--horizon-min", type=float, default=None,
                    help="also run the Phase 4 forecast and Phase 5 reliability as context (never as evidence)")
    dg.add_argument("--origin", default=None, help="with --horizon-min: forecast origin (log's time basis)")
    dg.add_argument("--passive-since", default=None, help="with --horizon-min: declared start of no active cooling")
    dg.add_argument("--question", choices=["no-cooling", "continuation"], default=None,
                    help="with --horizon-min: the forecast question")
    dg.add_argument("--json", action="store_true", help="print the full diagnostic report as JSON")
    dg.set_defaults(func=_cmd_diagnose, threshold=None, direction="up", assume_equilibrium=None, future_ambient=None,
                    step_min=None, allow_extrapolation=False)

    an = sub.add_parser("analyze", help="validate a temperature log and predict time to a threshold")
    _add_input_args(an)
    an.add_argument("--threshold", type=float, required=True, help="°C limit for time-to-threshold")
    an.add_argument("--now", type=float, default=None, help="current temperature; default = last reading")
    an.set_defaults(func=_cmd_analyze)

    kb = sub.add_parser("kb", help="technical documentation: ingest, query with cited passages, list, remove "
                                   "(Phase 7; local, extractive)")
    kbs = kb.add_subparsers(dest="kb_command", required=True)

    def store_arg(x):
        x.add_argument("--store", metavar="FILE", default=None,
                       help="knowledge store (SQLite); default: <data_dir>/knowledge/kb.sqlite")

    ki = kbs.add_parser("ingest", help="index TXT, Markdown, DOCX or PDF documents (PDF needs pypdf)")
    ki.add_argument("files", nargs="+", metavar="FILE")
    store_arg(ki)
    ki.add_argument("--doc-id", default=None, help="stable id (default: from the file name)")
    ki.add_argument("--doc-type", default="UNSPECIFIED", help="SERVICE_MANUAL, MAINTENANCE_MANUAL, "
                    "TROUBLESHOOTING_GUIDE, INSTALLATION_MANUAL, EQUIPMENT_DATASHEET, SERVICE_BULLETIN, "
                    "OTHER_TECHNICAL")
    for opt in ("title", "manufacturer", "model", "revision", "date", "supplied-by"):
        ki.add_argument(f"--{opt}", default=None, help=f"{opt.replace('-', ' ')} as stated on the document "
                        "(never inferred)")
    ki.add_argument("--embedding", default=None, help="embedding model for a NEW store: lexical-hash-v1 "
                    "(default) or bge-small-en-v1.5 (local, pinned; needs coldsense[semantic] and the model files)")
    ki.add_argument("--json", action="store_true")
    ki.set_defaults(func=_cmd_kb_ingest)
    kq = kbs.add_parser("query", help="cited passages and a quoted answer, or NO_RELEVANT_DOCUMENTATION (exit 2)")
    kq.add_argument("query")
    store_arg(kq)
    kq.add_argument("--k", type=int, default=5, help="passages returned at most (1-20)")
    kq.add_argument("--json", action="store_true")
    kq.set_defaults(func=_cmd_kb_query)
    kl = kbs.add_parser("list", help="indexed documents with provenance")
    store_arg(kl)
    kl.add_argument("--all", action="store_true", help="include superseded and removed versions")
    kl.add_argument("--json", action="store_true")
    kl.set_defaults(func=_cmd_kb_list)
    kr = kbs.add_parser("remove", help="delete a document's chunks (all versions)")
    kr.add_argument("doc_id")
    store_arg(kr)
    kr.set_defaults(func=_cmd_kb_remove)
    kx = kbs.add_parser("reindex", help="copy a store into a NEW store with another embedding model")
    store_arg(kx)
    kx.add_argument("--to", required=True, metavar="FILE", help="the new store (never the source)")
    kx.add_argument("--embedding", required=True, help="lexical-hash-v1 or bge-small-en-v1.5")
    kx.add_argument("--replace", action="store_true", help="replace --to if it exists, once the re-index succeeds")
    kx.set_defaults(func=_cmd_kb_reindex)
    ke = kbs.add_parser("evaluate", help="retrieval metrics on the built-in SYNTHETIC corpus")
    ke.add_argument("--split", choices=("DEVELOPMENT", "HELD_OUT"), default=None,
                    help="dev-v1 / heldout-v2 (historical, extractive-v1); default: the v1 lexical regression sets")
    ke.add_argument("--dataset", choices=("dev-v1", "heldout-v2", "dev-v2", "heldout-v3"), default=None,
                    help="a labelled set by version (dev-v2 / heldout-v3: retrieval and answerability metrics)")
    ke.add_argument("--embedding", default=None, help="with --split/--dataset: lexical-hash-v1 (default) or "
                    "bge-small-en-v1.5")
    ke.add_argument("--json", action="store_true")
    ke.set_defaults(func=_cmd_kb_evaluate)

    sub.add_parser("modules", help="list domain modules and model cards").set_defaults(func=_cmd_modules)
    sub.add_parser("doctor", help="check environment and configuration").set_defaults(func=_cmd_doctor)
    sub.add_parser("version", help="print version").set_defaults(func=lambda _: print(__version__) or EXIT_OK)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        configure_logging()
        return args.func(args)
    except ColdSenseError as e:
        log.debug("command failed", exc_info=True)
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR
    except FileNotFoundError as e:
        print(f"error: file not found: {e.filename}", file=sys.stderr)
        return EXIT_ERROR
    except BrokenPipeError:
        # Reader (e.g. `head`) stopped early. Report the outcome that was already
        # decided; a refusal must never look like success. Unknown -> error.
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except (OSError, ValueError, AttributeError):
            pass  # stdout has no real file descriptor (notebooks, IDE consoles, test capture)
        return getattr(args, "result_code", EXIT_ERROR)


if __name__ == "__main__":
    raise SystemExit(main())
