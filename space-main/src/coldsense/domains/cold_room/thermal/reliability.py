"""Phase 5: uncertainty, calibration and the reliability gate.

Phase 4 supplies the forecast (the only numerical authority) and its held-out
evidence. This module adds, without re-fitting anything:

* a CAUSAL VIEW: with an explicit origin, the log is cut to what existed at the
  origin before Phase 2 validation and Phase 4 run, so no later reading can
  influence data-quality flags, fitting, selection or this assessment;
* an EMPIRICAL INTERVAL from Phase 4's final-evaluation (TEST) forecasts, made
  by models fitted on FIT+SELECT only. TEST time is split in half: the first
  half CALIBRATES (cases whose truth lies past the split are purged), the
  second half EVALUATES coverage in periods disjoint from the calibration
  ones. SELECT errors chose the model and are never used; training residuals
  are never used;
* ONE UNIT OF INFERENCE, the PASSIVE PERIOD: every held-out case of a period
  (overlapping or not) shares that thermal episode, so a period weighs 1 in
  the quantiles, the coverage estimate, its confidence interval, the bias test
  and every gate; case and window counts are descriptive only. Periods are
  assumed independent, and a lag-1 screen on period-mean errors checks it;
* a RULE-BASED GATE: every rule reports PASS / FAIL / NOT_APPLICABLE with its
  evidence; the status is the most restrictive cap among failed rules.

No number here is a probability of being right, and nothing here changes the
Phase 4 forecast, its model selection or its refusal.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from coldsense.core.evidence import EvidenceClass

from ..pipeline import validate_source
from . import limits as L
from . import registry as R
from . import reliability_limits as RL
from .baselines import recent_slope
from .engine import ENGINE_VERSION, forecast
from .schemas import (
    EMITTED_INTERVALS,
    RELIABILITY_ORDER,
    SCOPE_FOR_EVIDENCE,
    CalibrationRecord,
    CausalView,
    CrossingStatus,
    CrossingUncertainty,
    CrossingUncertaintyStatus,
    EvidenceRecommendation,
    EvidenceSummary,
    ForecastQuestion,
    ForecastRequest,
    ForecastResult,
    ForecastStatus,
    FutureInput,
    GateOutcome,
    HorizonDiagnostics,
    IntervalStatus,
    Necessity,
    ParameterUncertainty,
    Priority,
    Provenance,
    Regime,
    RegimeBasis,
    ReliabilityAssessment,
    ReliabilityConfig,
    ReliabilityGate,
    ReliabilityReport,
    ReliabilityStatus,
    UncertaintyAssessment,
    UncertaintySource,
    ValidationScope,
    min_conformal_sample,
)
from .selection import ANY_REGIME

METHOD_VERSION = "phase5-reliability-3"
INTERVAL_METHOD = ("split-conformal-style empirical interval: period-weighted order statistics of out-of-sample "
                   "signed errors (observed - forecast) at this horizon, each passive period weighing 1 and the "
                   "ranks counted in periods, from the CALIBRATION half of the Phase 4 TEST data; coverage checked "
                   "on the EVALUATION half, in periods disjoint from the calibration periods")
INFERENCE_UNIT = ("passive period = one passive run (thermal episode): Phase 4 pieces split by data gaps, flagged "
                  "readings or excluded events stay one run unless the compressor is recorded ON between them; all "
                  "held-out cases of a run form one observation of weight 1; runs are assumed independent (lag-1 "
                  "screen on period-mean errors), cases within a run are not")
GATE_METHOD = "rule-based evidence gate (no score): status = most restrictive cap among failed rules"
H, M, LO, U = (ReliabilityStatus.HIGH, ReliabilityStatus.MEDIUM, ReliabilityStatus.LOW,
               ReliabilityStatus.UNSUPPORTED)
_SCOPE_TEXT = {
    ValidationScope.SIMULATED_ONLY: "SIMULATED data: statements hold for the simulation only",
    ValidationScope.REAL_WORLD_UNVALIDATED: "REAL (measured) log, but the method has no prospective real-world "
                                            "validation",
    ValidationScope.EXPECTED_ONLY: "EXPECTED data (a model output not checked against reality): held-out errors "
                                   "measure agreement with that output, not with a room",
    ValidationScope.UNVALIDATED_PROVENANCE: "provenance UNVALIDATED (origin unknown): treated neither as real "
                                            "measurements nor as a simulation; no real-world statement is made",
}


# ====================================================================== statistics (pure, separately tested)
def order_stat_requirement(level: float) -> int:
    """Smallest number of independent units (here: passive periods) for which both conformal order statistics
    exist at this two-sided level: n >= 2/a - 1, a = 1 - level. The schema enforces the same rule."""
    return min_conformal_sample(level)


def empirical_offsets(errors: list[float], level: float) -> tuple[float, float] | None:
    """Lower / upper offsets: the floor((n+1)a/2)-th and ceil((n+1)(1-a/2))-th smallest errors (1-based).
    Infinite scores (missed or false crossings) are allowed and may give an unbounded offset."""
    e = sorted(errors)
    n, a = len(e), 1.0 - level
    j = math.floor((n + 1) * a / 2 + 1e-9)
    k = math.ceil((n + 1) * (1 - a / 2) - 1e-9)
    if j < 1 or k > n:
        return None
    return e[j - 1], e[k - 1]


def period_weighted_offsets(groups: list[list[float]], level: float) -> tuple[float, float] | None:
    """Lower / upper offsets of the period-averaged empirical distribution. Each list is one passive period and
    weighs 1 in total, shared equally by its cases; with B periods the offsets are the smallest values at which
    the cumulative weight reaches floor((B+1)a/2) and ceil((B+1)(1-a/2)). With one case per period this is exactly
    ``empirical_offsets``; with several, no finite-sample guarantee is claimed (coverage is checked separately)."""
    groups = [g for g in groups if g]
    b, a = len(groups), 1.0 - level
    j = math.floor((b + 1) * a / 2 + 1e-9)
    k = math.ceil((b + 1) * (1 - a / 2) - 1e-9)
    if j < 1 or k > b:
        return None
    lo, cum = None, 0.0
    for x, w in sorted((x, 1.0 / len(g)) for g in groups for x in g):
        cum += w
        if lo is None and cum >= j - 1e-9:
            lo = x
        if cum >= k - 1e-9:
            return lo, x
    return None                     # unreachable: the cumulative weight reaches b >= k


def period_coverage(groups: list[list[float]], lo: float, hi: float) -> float:
    """Mean over periods of the share of each period's scores inside [lo, hi]: each period weighs 1."""
    return float(np.mean([np.mean([lo <= s <= hi for s in g]) for g in groups if g]))


def period_mean_test(means: list[float]) -> tuple[float, float, float, bool] | None:
    """Two-sided one-sample t-test of the period-mean errors against 0 (n = periods, df = n - 1):
    (mean, |t|, critical t, significant); None with fewer than MIN_PERIODS_FOR_TESTS periods."""
    x = np.asarray(means, dtype=float)
    n = len(x)
    if n < RL.MIN_PERIODS_FOR_TESTS:
        return None
    m, s = float(x.mean()), float(x.std(ddof=1))
    tcrit = float(stats.t.ppf(1 - RL.BIAS_TEST_ALPHA / 2, n - 1))
    if s == 0:
        return m, (math.inf if m != 0 else 0.0), tcrit, m != 0
    t = abs(m) / (s / math.sqrt(n))
    return m, t, tcrit, t > tcrit


def dependence_screen(means: list[float]) -> tuple[float | None, float | None, bool]:
    """(lag-1 autocorrelation of the chronological period means, critical value, flagged). Flags r1 above the
    one-sided large-sample white-noise bound z(1 - alpha) / sqrt(n). A screen with little power for few periods:
    not flagging does NOT show independence. None, None, False with too few periods or no variation."""
    x = np.asarray(means, dtype=float)
    if len(x) < RL.MIN_PERIODS_FOR_TESTS:
        return None, None, False
    d = x - x.mean()
    den = float((d * d).sum())
    if den == 0:
        return None, None, False
    r1 = float((d[:-1] * d[1:]).sum() / den)
    crit = float(stats.norm.ppf(1 - RL.PERIOD_DEPENDENCE_ALPHA)) / math.sqrt(len(x))
    return r1, crit, r1 > crit


def non_overlapping_windows(items: list[tuple[int, float]], horizon: float) -> int:
    """Non-overlapping forecast windows [t, t + horizon] within each period (greedy in time). DESCRIPTIVE only:
    consecutive windows of one period share its thermal episode and are not independent evidence."""
    count, last = 0, {}
    for period, t in sorted(items):
        if period not in last or t >= last[period] + horizon - 1e-9:
            count += 1
            last[period] = t
    return count


def wilson(p: float, n: int, z: float = RL.WILSON_Z) -> tuple[float, float] | None:
    if n <= 0:
        return None
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def split_boundary(f, test_pieces) -> float | None:
    """Time (minutes) at half of the accepted TEST passive time: calibration before, evaluation after."""
    pieces = sorted(test_pieces, key=lambda p: f.t[p.rows[0]])
    durs = [p.duration(f) for p in pieces]
    total = sum(durs)
    if total <= 0:
        return None
    target, acc = total / 2, 0.0
    for p, d in zip(pieces, durs, strict=True):
        if acc + d >= target:
            return float(f.t[p.rows[0]] + (target - acc))
        acc += d
    return None


# ====================================================================== causal view
def _origin_line(clean: pd.DataFrame, origin) -> tuple[int, pd.Timestamp] | None:
    ts = clean["ts"]
    i = int(ts.searchsorted(pd.Timestamp(origin), side="right")) - 1
    if i < 0:
        return None
    return int(clean["source_line"].iloc[i]), ts.iloc[i]


def _csv_records(text: str, delim: str) -> list[tuple[int, int, bool]]:
    """(start line, end line, blank) of each physical record, with the same line accounting as Phase 2."""
    reader = csv.reader(io.StringIO(text), delimiter=delim)
    out, prev = [], 0
    for row in reader:
        start, prev = prev + 1, reader.line_num
        out.append((start, prev, not row or all(not c.strip() for c in row)))
    return out


def causal_source(src, options, request: ForecastRequest, *, evidence: EvidenceClass, limits=None):
    """(source to analyse, rows in source, rows excluded, note). Without an explicit origin nothing is cut.
    With one, only records at or before the origin reading in the file, and not timestamped after it, are
    kept, so Phase 2 flags, Phase 3 events and Phase 4 fitting see exactly what existed at the origin."""
    data = None
    if not isinstance(src, (str, Path, pd.DataFrame)):           # a stream can be read once only
        data = src.read()
        src = io.BytesIO(data) if isinstance(data, bytes) else io.StringIO(data)
    if request.origin is None:
        return src, None, 0, "no explicit origin: the forecast starts at the last reading, nothing is cut"
    full_clean, full_obs, full_rep = validate_source(src, evidence=evidence, options=options, limits=limits)
    if isinstance(src, io.IOBase):
        src.seek(0)
    total = len(full_obs)
    found = _origin_line(full_clean, request.origin) if len(full_clean) else None
    if found is None:
        return src, total, 0, "origin not inside the usable log: nothing is cut (Phase 4 decides)"
    line, t_origin = found
    ts = full_obs["ts"]
    keep = (full_obs["source_line"] <= line) & ~(ts.notna() & (ts > t_origin))
    excluded = int((~keep).sum())
    if excluded == 0:
        return src, total, 0, "no record after the origin: nothing is cut"
    kept_lines = set(full_obs.loc[keep, "source_line"].astype(int))
    if isinstance(src, pd.DataFrame):
        cut = src.iloc[sorted(ln - 1 for ln in kept_lines)]
        cut.attrs = dict(src.attrs)
        return cut, total, excluded, f"{excluded} record(s) after the origin removed before validation"
    if data is None:
        data = Path(src).read_bytes()
    if isinstance(data, bytes):
        from ..ingest import _decode  # the same decoding rules as Phase 2
        text = _decode(data, getattr(options, "encoding", None))[0]
    else:
        text = data
    lines = io.StringIO(text).readlines()
    header_done, parts = False, []
    for start, end, blank in _csv_records(text, full_rep.input.delimiter):
        if blank:
            continue
        if not header_done:
            header_done = True
            parts.append("".join(lines[start - 1:end]))
        elif start in kept_lines:
            parts.append("".join(lines[start - 1:end]))
    return (io.StringIO("".join(parts)), total, excluded,
            f"{excluded} record(s) after the origin removed before validation (hash is of the cut text)")


# ====================================================================== held-out evidence per horizon
@dataclass
class _HEval:
    diag: HorizonDiagnostics
    period_means: list[float]          # chronological, one per passive period with a held-out case at h
    lag1: float | None
    lag1_crit: float | None
    dependent: bool
    key: object                        # case -> passive run (the unit of inference)
    offsets: tuple[float, float] | None = None
    coverage: float | None = None
    ci: tuple[float, float] | None = None
    cal: list = field(default_factory=list)
    ev: list = field(default_factory=list)
    cal_windows: int = 0
    ev_windows: int = 0
    cal_periods: int = 0
    ev_periods: int = 0


def _t(f, c) -> float:
    return float(f.t[c.i0])


def passive_runs(f, pieces) -> dict[int, int]:
    """Phase 4 piece (by ``parent_first``) -> first row of its passive RUN, the unit of inference. Phase 4 splits
    one passive stretch into pieces at data gaps, flagged readings and excluded events; those pieces share one
    thermal episode. Consecutive PASSIVE pieces therefore belong to one run unless a reading with the compressor
    recorded ON lies between them (positive evidence that cooling reset the room). Without a recorded
    compressor nothing ends a run: conservative (it can only lower the number of runs). Continuation pieces
    (any regime) contain cooling themselves, so recorded cooling between them ends nothing: one run."""
    spans: dict[int, tuple[int, int]] = {}
    passive: dict[int, bool] = {}
    for pc in pieces:
        a, b = int(pc.rows[0]), int(pc.rows[-1])
        lo, hi = spans.get(int(pc.parent_first), (a, b))
        spans[int(pc.parent_first)] = (min(lo, a), max(hi, b))
        passive[int(pc.parent_first)] = getattr(pc, "regime", None) != ANY_REGIME
    out, run, prev_end = {}, None, None
    for first in sorted(spans):
        start, end = spans[first]
        if run is None or (passive[first] and passive[run]
                           and any(f.comp[i] is True for i in range(prev_end + 1, start))):
            run = first
        out[first] = run
        prev_end = max(prev_end, end) if prev_end is not None else end
    return out


def _run_key(f, cases):
    runs = passive_runs(f, {id(c.piece): c.piece for c in cases}.values())
    return lambda c: runs[int(c.piece.parent_first)]


def _groups(pairs, key) -> list[list[float]]:
    """Values of (case, value) pairs grouped by passive run, runs in chronological order."""
    by: dict[int, list[float]] = {}
    for c, v in pairs:
        by.setdefault(key(c), []).append(v)
    return [by[p] for p in sorted(by)]


def _split(f, cases, h: float, boundary: float, key) -> tuple[list, list]:
    """CALIBRATION: cases whose truth (origin + h) is at or before the boundary (purged). EVALUATION: origins at
    or after it, in runs with no calibration case, so the two halves share no passive run."""
    cal = [c for c in cases if _t(f, c) + h <= boundary + 1e-9]
    cal_runs = {key(c) for c in cal}
    return cal, [c for c in cases if _t(f, c) >= boundary - 1e-9 and key(c) not in cal_runs]


def _quantiles(x: np.ndarray, qs) -> dict[str, float]:
    return {f"p{round(q * 100):02d}": float(np.quantile(x, q)) for q in qs} if len(x) else {}


def _horizon_eval(f, cases, method: str, h: float, boundary: float | None, level: float,
                  sel_h: float | None) -> _HEval:
    at_h = [c for c in cases if c.horizon == h]
    own = [c for c in at_h if c.preds.get(method) is not None]
    e = lambda c: c.truth - c.preds[method]   # noqa: E731
    err = np.array([e(c) for c in own])
    key = _run_key(f, own)
    means = [float(np.mean(g)) for g in _groups(((c, e(c)) for c in own), key)]
    r1, crit, dependent = dependence_screen(means)
    diag = HorizonDiagnostics(
        horizon_min=h, n_cases=len(own), prediction_coverage=round(len(own) / len(at_h), 4) if at_h else None,
        non_overlapping_windows=non_overlapping_windows([(key(c), _t(f, c)) for c in own], h),
        periods=len(means), mae_c=float(np.abs(err).mean()) if len(err) else None,
        rmse_c=float(np.sqrt((err ** 2).mean())) if len(err) else None,
        bias_c=float(err.mean()) if len(err) else None,
        abs_error_quantiles_c=_quantiles(np.abs(err), RL.ABS_QUANTILES),
        signed_error_quantiles_c=_quantiles(err, RL.SIGNED_QUANTILES),
        period_mean_bias_c=float(np.mean(means)) if means else None, period_lag1_autocorrelation=r1,
        calibration_status=IntervalStatus.INSUFFICIENT_CALIBRATION_DATA)
    out = _HEval(diag, means, r1, crit, dependent, key)
    if sel_h is None or h > sel_h + 1e-9:
        diag.calibration_status = IntervalStatus.UNSUPPORTED_HORIZON
        return out
    if boundary is None:
        return out
    out.cal, out.ev = _split(f, own, h, boundary, key)
    out.cal_windows = non_overlapping_windows([(key(c), _t(f, c)) for c in out.cal], h)
    out.ev_windows = non_overlapping_windows([(key(c), _t(f, c)) for c in out.ev], h)
    gcal, gev = _groups(((c, e(c)) for c in out.cal), key), _groups(((c, e(c)) for c in out.ev), key)
    out.cal_periods, out.ev_periods = len(gcal), len(gev)
    need = order_stat_requirement(level)
    if out.cal_periods < need:
        return out
    if out.ev_periods < need:
        diag.calibration_status = IntervalStatus.UNVALIDATED
        return out
    if dependent:                    # the coverage check below assumes independent periods: contradicted
        diag.calibration_status = IntervalStatus.DEPENDENT_PERIODS
        return out
    lo, hi = period_weighted_offsets(gcal, level)                           # exists: cal_periods >= need
    cov = period_coverage(gev, lo, hi)
    ci = wilson(cov, out.ev_periods)
    out.coverage, out.ci = cov, ci
    diag.empirical_coverage = cov
    if ci[1] < level:
        diag.calibration_status = IntervalStatus.UNDER_COVERING
        return out
    diag.calibration_status = IntervalStatus.CONSERVATIVE if ci[0] > level else IntervalStatus.SUPPORTED
    out.offsets = (lo, hi)
    diag.error_offsets_c = (lo, hi)
    diag.interval_width_c = hi - lo
    return out


# ====================================================================== threshold-crossing time
def _crossing(result: ForecastResult, sink: dict, f, boundary, level: float, h: float,
              sel_h: float | None, dependent: bool) -> CrossingUncertainty:
    det, mins = result.crossing_status, result.predicted_crossing_minutes
    base = dict(deterministic_status=det, deterministic_minutes=mins,
                method="empirical interval of out-of-sample crossing-time errors (observed - predicted, minutes) "
                       "at the requested threshold and horizon; missed or false crossings count as unbounded "
                       "errors; each passive period weighs 1 (its cases share one observed crossing) in the "
                       "quantiles and the coverage; calibration and evaluation periods are disjoint")
    passthrough = {CrossingStatus.NO_THRESHOLD: CrossingUncertaintyStatus.NO_THRESHOLD,
                   CrossingStatus.ALREADY_CROSSED: CrossingUncertaintyStatus.ALREADY_CROSSED,
                   CrossingStatus.NOT_REACHABLE: CrossingUncertaintyStatus.NOT_REACHABLE,
                   CrossingStatus.NOT_WITHIN_HORIZON: CrossingUncertaintyStatus.NOT_WITHIN_HORIZON}
    if det in passthrough:
        note = ([] if det in (CrossingStatus.NO_THRESHOLD, CrossingStatus.ALREADY_CROSSED) else
                ["the Phase 4 result is deterministic under its model; no calibrated statement is made about a "
                 "crossing that model does not predict"])
        return CrossingUncertainty(status=passthrough[det], limitations=note, **base)
    if sel_h is None or h > sel_h + 1e-9:
        return CrossingUncertainty(status=CrossingUncertaintyStatus.UNSUPPORTED_HORIZON, **base,
                                   limitations=["the horizon is beyond the validated one"])
    method = sink.get("winner")
    scored = []                                     # (case, score)
    for c in sink.get("test_cases", []):
        if c.horizon != h or c.already or c.preds.get(method) is None or c.cross.get(method) is None:
            continue
        status, pm = c.cross[method]
        predicted = status == CrossingStatus.CROSSING_PREDICTED
        if c.obs_cross is None and not predicted:
            continue                                # agreement that nothing crosses: no time error to score
        score = (c.obs_cross - pm if predicted and c.obs_cross is not None else
                 -math.inf if c.obs_cross is not None else math.inf)
        scored.append((c, score))
    if boundary is None:
        return CrossingUncertainty(status=CrossingUncertaintyStatus.INSUFFICIENT_CROSSING_DATA, **base,
                                   limitations=["no held-out TEST period"])
    score = {id(c): s for c, s in scored}
    key = _run_key(f, [c for c, _ in scored])
    cal, ev = _split(f, [c for c, _ in scored], h, boundary, key)
    gcal, gev = _groups(((c, score[id(c)]) for c in cal), key), _groups(((c, score[id(c)]) for c in ev), key)
    cal_events, ev_events = len(gcal), len(gev)
    common = dict(base, nominal_coverage=level, calibration_events=cal_events, evaluation_events=ev_events)
    need = order_stat_requirement(level)
    if cal_events < need:
        return CrossingUncertainty(status=CrossingUncertaintyStatus.INSUFFICIENT_CROSSING_DATA, **common,
                                   limitations=[f"{cal_events} passive period(s) with a crossing case in "
                                                f"calibration; {need} needed at {level:.0%}"])
    offs = period_weighted_offsets(gcal, level)                             # exists: cal_events >= need
    if not all(math.isfinite(x) for x in offs):
        return CrossingUncertainty(status=CrossingUncertaintyStatus.UNBOUNDED, **common,
                                   limitations=["missed or false crossings in calibration exceed the tail level"])
    if ev_events < need:
        return CrossingUncertainty(status=CrossingUncertaintyStatus.UNVALIDATED, **common,
                                   limitations=[f"{ev_events} passive period(s) with a crossing case in "
                                                f"evaluation; {need} needed"])
    if dependent:
        return CrossingUncertainty(status=CrossingUncertaintyStatus.DEPENDENT_PERIODS, **common,
                                   limitations=["held-out errors at this horizon are serially dependent from "
                                                "period to period, so the coverage check cannot treat periods as "
                                                "independent"])
    cov = period_coverage(gev, *offs)
    ci = wilson(cov, ev_events)
    common["empirical_coverage"] = cov
    if ci[1] < level:
        return CrossingUncertainty(status=CrossingUncertaintyStatus.UNDER_COVERING, **common,
                                   limitations=["evaluation coverage significantly below nominal"])
    lower, upper = max(0.0, mins + offs[0]), mins + offs[1]
    return CrossingUncertainty(status=CrossingUncertaintyStatus.CALIBRATED_INTERVAL, lower_min=lower,
                               upper_min=upper, extends_beyond_horizon=upper > h, **common,
                               limitations=["threshold-specific; few independent passive periods; SIMULATED "
                                            "or same-log evidence only"])


# ====================================================================== parameter envelope (not a prediction interval)
def _parameter_envelope(result: ForecastResult, sink: dict, h: float) -> ParameterUncertainty:
    winner = sink.get("winner")
    t0 = result.initial_temperature.value if result.initial_temperature else None
    if winner == R.PERSISTENCE:
        return ParameterUncertainty(status="NOT_APPLICABLE", method="none", meaning="persistence has no parameters")
    if winner == R.RECENT_SLOPE:
        f, piece, i0 = sink.get("frame"), sink.get("origin_piece"), sink.get("i0")
        fit, _ = recent_slope(f.t, f.y, i0, piece.parent_first) if piece is not None else (None, None)
        if fit is None or not math.isfinite(fit.se):
            return ParameterUncertainty(status="UNAVAILABLE", method="slope standard error",
                                        meaning="slope standard error not computable")
        d = RL.SLOPE_SE_MULTIPLIER * fit.se
        lo, hi = t0 + (fit.slope - d) * h, t0 + (fit.slope + d) * h
        return ParameterUncertainty(
            status="ASSESSED", method=f"OLS slope +- {RL.SLOPE_SE_MULTIPLIER:g} standard errors, propagated linearly",
            meaning="slopes compatible with the trailing window under independent noise; residual autocorrelation "
                    "makes it optimistic", horizon_min=h, envelope_lower_c=min(lo, hi), envelope_upper_c=max(lo, hi),
            envelope_width_c=abs(hi - lo), parameters={"slope": (fit.slope - d, fit.slope + d)})
    fo = sink.get("final_fit")
    if fo is None or fo.k_range is None:
        return ParameterUncertainty(status="UNAVAILABLE", method="profile likelihood",
                                    meaning="no closed profile range for k")
    ks = (fo.k_range[0], fo.k, fo.k_range[1])
    e = sink.get("e_forecast")
    params = {"k": tuple(fo.k_range)}
    es = (e,)
    if sink.get("e_provenance") == Provenance.FITTED and fo.eq_range.get("E") is not None:
        es = (fo.eq_range["E"][0], e, fo.eq_range["E"][1])
        params["E"] = tuple(fo.eq_range["E"])
    vals = [ee + (t0 - ee) * math.exp(-kk * h) for kk in ks for ee in es]
    return ParameterUncertainty(
        status="ASSESSED", method="profile-likelihood corner propagation: T(h) at the corners of the marginal "
                                  "profile ranges (F-test level used by Phase 4 identifiability)",
        meaning="forecasts compatible with the fitted data when only the parameters vary; corners of marginal "
                "ranges are conservative for the joint region", horizon_min=h, envelope_lower_c=min(vals),
        envelope_upper_c=max(vals), envelope_width_c=max(vals) - min(vals), parameters=params)


# ====================================================================== the assessment
def _scope(evidence: EvidenceClass) -> ValidationScope:
    """Explicit mapping (schemas.SCOPE_FOR_EVIDENCE): EXPECTED and UNVALIDATED inputs are never presented as
    measured REAL data, and no label alone yields REAL_WORLD_VALIDATED."""
    return SCOPE_FOR_EVIDENCE[EvidenceClass(evidence)]


def _gate(name, ok, cap, evidence) -> ReliabilityGate:
    outcome = GateOutcome.NOT_APPLICABLE if ok is None else (GateOutcome.PASS if ok else GateOutcome.FAIL)
    return ReliabilityGate(name=name, outcome=outcome, cap_if_failed=cap, evidence=evidence)


def _assessment(gates: list[ReliabilityGate], scope: ValidationScope) -> ReliabilityAssessment:
    failed = [g for g in gates if g.outcome == GateOutcome.FAIL]
    status = max((g.cap_if_failed for g in failed), key=RELIABILITY_ORDER.index, default=H)
    return ReliabilityAssessment(
        status=status, validation_scope=scope, method=GATE_METHOD,
        decision_reasons=[f"{g.name} FAILED (cap {g.cap_if_failed.value}): {g.evidence}" for g in failed],
        gates=gates, gates_passed=[g.name for g in gates if g.outcome == GateOutcome.PASS],
        gates_failed=[g.name for g in failed])


def _rec(what, rationale, capability, guidance, retry, necessity=Necessity.NECESSARY_FOR_THIS_QUESTION,
         priority=Priority.HIGH, kind="DATA") -> EvidenceRecommendation:
    return EvidenceRecommendation(missing_input=what, kind=kind, rationale=rationale, potential_capability=capability,
                                  acquisition_guidance=guidance, priority=priority, necessity=necessity,
                                  priority_rationale="needed for the reliability level named in the rationale"
                                  if necessity == Necessity.NECESSARY_FOR_THIS_QUESTION else "a useful improvement",
                                  retry_conditions=retry)


def _refused_report(result: ForecastResult, config: ReliabilityConfig, view: CausalView,
                    scope: ValidationScope) -> ReliabilityReport:
    gate = _gate("FORECAST_AVAILABLE", False, U, f"Phase 4 refused: {result.refusal_code.value}: "
                                                 f"{result.refusal_reason}")
    retry = result.refusal_needed_to_retry or "see the Phase 4 refusal and its recommendations"
    return ReliabilityReport(
        method_version=METHOD_VERSION, forecast_reference=f"enclosing ForecastResult (Phase 4 {ENGINE_VERSION})",
        forecast_available=False, config=config, causal_view=view, reliability=_assessment([gate], scope),
        uncertainty=UncertaintyAssessment(status=IntervalStatus.NOT_APPLICABLE, method="none: no forecast",
                                          nominal_coverage=config.interval_level),
        crossing=CrossingUncertainty(status=CrossingUncertaintyStatus.NOT_APPLICABLE,
                                     deterministic_status=result.crossing_status, method="none: no forecast"),
        parameter_uncertainty=ParameterUncertainty(status="NOT_APPLICABLE", method="none",
                                                   meaning="no forecast"),
        evidence=EvidenceSummary(data_quality="see the Phase 4 refusal", model_quality="no model was used",
                                 identifiability="not assessed", regime_applicability="not assessed",
                                 horizon_applicability="not assessed", extrapolation_status="no forecast"),
        limitations=["no forecast exists, so no uncertainty or reliability is reported"],
        recommendations=[_rec("evidence asked for by the Phase 4 refusal",
                              f"Phase 4 refused ({result.refusal_code.value}); nothing can be assessed without a "
                              "forecast", "a forecast that can then be assessed", "see the Phase 4 recommendations",
                              retry)],
        refusal_or_downgrade_reason=f"no forecast: {result.refusal_code.value}: {result.refusal_reason}")


def assess(result: ForecastResult, sink: dict, report, request: ForecastRequest, config: ReliabilityConfig,
           view: CausalView, events: list | None = None) -> ReliabilityReport:
    scope = _scope(report.evidence)
    if result.status != ForecastStatus.FORECAST:
        return _refused_report(result, config, view, scope)
    f, winner, level = sink["frame"], sink["winner"], config.interval_level
    h, sel_h = request.horizon_min, result.validated_horizon_min
    cases = sink.get("test_cases", [])
    boundary = split_boundary(f, sink.get("test_pieces", []))
    evals = {hh: _horizon_eval(f, cases, winner, hh, boundary, level, sel_h) for hh in sink.get("horizons", [h])}
    he = evals.get(h) or _horizon_eval(f, cases, winner, h, boundary, level, sel_h)
    fc = float(result.trajectory[-1].temp_c)
    kind = R.BY_NAME[winner].kind
    regime, basis = sink.get("origin_regime"), sink.get("basis")
    continuation = sink.get("path") == "continuation"

    # ---- future inputs and conditions
    future: list[FutureInput] = []
    user_assumed = []
    future.append(FutureInput(name="compressor / active cooling over the horizon",
                              status="SCENARIO" if request.question == ForecastQuestion.NO_COOLING else "ASSUMED",
                              note="no active cooling from the origin (the question asked)"
                              if request.question == ForecastQuestion.NO_COOLING
                              else "the behaviour observed at the origin continues (the question asked)"))
    if kind == "PHYSICAL":
        e_prov = sink.get("e_provenance")
        if winner == R.FO_FITTED:
            future.append(FutureInput(name="equilibrium over the horizon", status="DERIVED",
                                      note="fitted effective equilibrium (not a measured ambient)"))
        elif winner == R.FO_ASSUMED:
            future.append(FutureInput(name="equilibrium over the horizon", status="ASSUMED",
                                      note="stated by the user (assumed_equilibrium_c)"))
            user_assumed.append("assumed equilibrium (user statement)")
        elif request.future_ambient_c is not None:
            future.append(FutureInput(name="ambient over the horizon", status="SCENARIO",
                                      note=f"{request.future_ambient_c:g} °C stated by the user"))
            user_assumed.append("future ambient (user scenario, not represented in calibration)")
        elif e_prov == Provenance.ASSUMED:
            future.append(FutureInput(name="ambient over the horizon", status="ASSUMED",
                                      note="held at its last measured value, the same rule the held-out "
                                           "forecasts used"))
    future.append(FutureInput(name="door activity, loading, defrost over the horizon", status="UNKNOWN",
                              note="not modelled; disturbances like those in the held-out data are reflected in "
                                   "the empirical errors, others are not"))
    conditional = [x.note if x.status != "UNKNOWN" else "no unrecorded disturbance beyond those seen in held-out "
                   "data" for x in future if x.status in ("SCENARIO", "ASSUMED", "UNKNOWN")]

    # ---- uncertainty at the requested horizon
    status = he.diag.calibration_status
    u_kw = dict(status=status, method=INTERVAL_METHOD, inference_unit=INFERENCE_UNIT, horizon_min=h,
                nominal_coverage=level, calibration_sample_count=len(he.cal),
                calibration_non_overlapping_windows=he.cal_windows, calibration_periods=he.cal_periods,
                evaluation_sample_count=len(he.ev), evaluation_non_overlapping_windows=he.ev_windows,
                evaluation_periods=he.ev_periods, supported_horizon_min=sel_h, conditional_on=conditional,
                includes=["measurement noise of the truth readings", "parameter error of a FIT+SELECT fit",
                          "model inadequacy and disturbances as they occurred in the held-out periods"],
                excludes=["future-input uncertainty (the interval is conditional on the stated scenario)",
                          "regimes, disturbances and rooms not represented in the held-out periods",
                          "the difference between the FIT+SELECT fit and the final refit used for the forecast"],
                assumptions=["passive periods are independent and exchangeable (screened by a lag-1 check on "
                             "period-mean errors, not proven); cases inside one period may depend on each other in "
                             "any way",
                             "with several cases per period no finite-sample coverage guarantee is claimed: "
                             "coverage is checked on evaluation periods disjoint from the calibration ones"],
                limitations=[f"sample sizes count passive periods ({he.cal_periods} calibration / {he.ev_periods} "
                             f"evaluation); the {len(he.cal)} / {len(he.ev)} cases and {he.cal_windows} / "
                             f"{he.ev_windows} non-overlapping {h:g}-min windows are descriptive only",
                             "pointwise at this horizon; not a band for the whole trajectory",
                             "coverage alone does not make an interval useful: a wide interval can cover; no "
                             "operational width requirement is defined, so usefulness is not judged"])
    if he.coverage is not None:
        u_kw.update(empirical_coverage=he.coverage, coverage_ci=he.ci, coverage_error=he.coverage - level)
    if status in EMITTED_INTERVALS:
        lo, hi = he.offsets
        u_kw.update(error_offsets_c=he.offsets, forecast_c=fc, interval_lower_c=fc + lo, interval_upper_c=fc + hi,
                    interval_width_c=hi - lo)
        if winner != R.PERSISTENCE and all(c.preds.get(R.PERSISTENCE) is not None for c in he.cal):
            ref = period_weighted_offsets(_groups(((c, c.truth - c.preds[R.PERSISTENCE]) for c in he.cal),
                                                  he.key), level)
            u_kw.update(reference_method=R.PERSISTENCE, reference_interval_width_c=ref[1] - ref[0])
            if hi - lo >= ref[1] - ref[0]:
                u_kw["limitations"].append(
                    f"the interval ({hi - lo:.3f} °C) is not narrower than the persistence reference interval "
                    f"({ref[1] - ref[0]:.3f} °C) calibrated the same way: it adds no precision beyond assuming the "
                    "temperature stays where it is")
    else:
        u_kw.update(forecast_c=fc)
    uncertainty = UncertaintyAssessment(**u_kw)

    # ---- other evidence
    param = _parameter_envelope(result, sink, h)
    crossing = _crossing(result, sink, f, boundary, level, h, sel_h, he.dependent)
    n_periods = len(he.period_means)
    test = period_mean_test(he.period_means)
    bias_ok, bias_ev = None, (f"{n_periods} passive period(s) with held-out cases at {h:g} min: too few for a bias "
                              f"test (need {RL.MIN_PERIODS_FOR_TESTS})")
    if test is not None:
        m, _, _, significant = test
        material = abs(m) >= RL.BIAS_MATERIAL_C
        bias_ok = not (significant and material)
        bias_ev = (f"period-weighted mean held-out error {m:+.3f} °C (observed - forecast; {n_periods} passive "
                   f"period(s) of weight 1, {he.diag.n_cases} case(s)); t-test on period means, df {n_periods - 1}, "
                   f"at {RL.BIAS_TEST_ALPHA:g}: {'significant' if significant else 'not significant'}; materiality "
                   f"{RL.BIAS_MATERIAL_C} °C" + ("; the period means are serially dependent (screen), so the "
                                                 "test's independence assumption is contradicted"
                                                 if he.dependent else ""))
    issues = [i for i in report.issues if i.code in RL.MATERIAL_QUALITY_ISSUES]
    n_test = sink.get("n_test", 0)
    manual_power = any(getattr(e, "kind", None) is not None and e.kind.value == "POWER_INTERRUPTION"
                       and e.provenance.value == "MANUAL_LOG" for e in (events or []))
    if continuation:
        regime_ok, regime_ev = None, "CONTINUATION from a non-passive origin needs no regime evidence"
        match_ok, match_ev = None, "held-out cases and origin are both 'any regime' (continuation)"
    else:
        regime_ok = basis == RegimeBasis.RECORDED_COMPRESSOR_STATE or (basis == RegimeBasis.RECORDED_EVENT
                                                                       and not manual_power)
        regime_ev = {RegimeBasis.RECORDED_COMPRESSOR_STATE: "recorded compressor state",
                     RegimeBasis.RECORDED_EVENT: "recorded power interruption"
                     + (" (MANUAL_LOG record)" if manual_power else " (system record)"),
                     RegimeBasis.DECLARED_BY_USER: "declared by the user (passive_since): an assumption"}.get(
                         basis, f"basis {getattr(basis, 'value', basis)}")
        match_ok = regime == Regime.PASSIVE
        match_ev = ("origin in a passive period, like the held-out cases" if match_ok else
                    "origin with cooling recorded ON (what-if): held-out cases all start in passive periods")
    half = (he.ci[1] - he.ci[0]) / 2 if he.ci else None
    gates = [
        _gate("FORECAST_AVAILABLE", True, U, f"Phase 4 forecast by {winner} ({kind})"),
        _gate("WITHIN_VALIDATED_HORIZON", sel_h is not None and h <= sel_h + 1e-9, U,
              f"requested {h:g} min; Phase 4 validated up to {sel_h:g} min" if sel_h is not None
              else "no validated horizon"),
        _gate("INDEPENDENT_FINAL_EVALUATION", n_test >= L.EVAL_MIN_CASES, LO,
              f"{n_test} common held-out TEST case(s) at the validated horizon (need {L.EVAL_MIN_CASES}): Phase 4's "
              "own final-evaluation check, held out from fitting; a count of cases, not of independent units"),
        _gate("INTERVAL_CALIBRATED",
              None if status == IntervalStatus.UNSUPPORTED_HORIZON else status in EMITTED_INTERVALS,
              LO if status in (IntervalStatus.UNDER_COVERING, IntervalStatus.DEPENDENT_PERIODS) else M,
              f"interval status {status.value}; calibration {he.cal_periods} / evaluation {he.ev_periods} passive "
              f"period(s) (disjoint); {order_stat_requirement(level)} needed in each at {level:.0%}"
              + (f"; period-weighted evaluation coverage {he.coverage:.3f}" if he.coverage is not None else "")),
        _gate("COVERAGE_EVIDENCE_PRECISION", None if half is None else half <= RL.COVERAGE_PRECISION_FOR_HIGH, M,
              "no coverage estimate" if half is None else
              f"coverage known to +-{half:.3f} (Wilson, n = {he.ev_periods} evaluation period(s)); need +-"
              f"{RL.COVERAGE_PRECISION_FOR_HIGH}"),
        _gate("INDEPENDENT_PERIODS", min(he.cal_periods, he.ev_periods) >= RL.MIN_INDEPENDENT_PERIODS, M,
              f"{he.cal_periods} calibration / {he.ev_periods} evaluation passive period(s); need "
              f"{RL.MIN_INDEPENDENT_PERIODS} each"),
        _gate("NO_PERIOD_DEPENDENCE_DETECTED", None if he.lag1 is None else not he.dependent, LO,
              f"{n_periods} passive period(s) at {h:g} min: too few, or no variation, for a dependence screen"
              if he.lag1 is None else
              f"lag-1 autocorrelation of {n_periods} chronological period-mean errors {he.lag1:+.3f}; flagged above "
              f"{he.lag1_crit:.3f} (one-sided {RL.PERIOD_DEPENDENCE_ALPHA:g}); a screen with little power for few "
              "periods: passing does not show independence"),
        _gate("NO_MATERIAL_BIAS", bias_ok, M, bias_ev),
        _gate("PARAMETER_UNCERTAINTY", None if param.status == "NOT_APPLICABLE" else
              param.status == "ASSESSED" and param.envelope_width_c <= RL.PARAMETER_ENVELOPE_LIMIT_C, M,
              param.meaning if param.status != "ASSESSED" else
              f"parameter envelope at {h:g} min is {param.envelope_width_c:.3f} °C wide (limit "
              f"{RL.PARAMETER_ENVELOPE_LIMIT_C} °C)"),
        _gate("REGIME_EVIDENCE_RECORDED", regime_ok, M, regime_ev),
        _gate("REGIME_MATCHES_CALIBRATION", match_ok, LO, match_ev),
        _gate("NO_USER_ASSUMED_FUTURE_INPUT", not user_assumed, M,
              "none" if not user_assumed else "; ".join(user_assumed)),
        _gate("DATA_QUALITY", not issues, M, "no material Phase 2 warning in the causal log" if not issues else
              "; ".join(f"{i.code}: {i.message}" for i in issues)),
        _gate("VALIDATION_SCOPE", scope == ValidationScope.SIMULATED_ONLY, M,
              f"{_SCOPE_TEXT[scope]} ({scope.value})"),
    ]
    assessment = _assessment(gates, scope)

    # ---- descriptive evidence and sources
    fd = result.fit_diagnostics
    evidence = EvidenceSummary(
        data_quality=f"Phase 2 verdict {report.verdict.value} on the causal log; material warnings: "
                     f"{[i.code for i in issues] or 'none'}; coverage {report.coverage_pct:.1f} %, "
                     f"{report.n_gaps} gap(s)",
        model_quality=f"{winner}: held-out MAE at {h:g} min {he.diag.mae_c:.3f} °C over {he.diag.n_cases} case(s)"
        if he.diag.mae_c is not None else f"{winner}: no held-out cases at {h:g} min",
        identifiability=("not applicable (baseline)" if kind == "BASELINE" else
                         f"{fd.identifiability.value}: {fd.identifiability_reason}" if fd else "unknown"),
        regime_applicability=match_ev, horizon_applicability=gates[1].evidence,
        extrapolation_status="extrapolated beyond the validated horizon" if sel_h is not None and h > sel_h
        else "within the validated horizon",
        future_input_assumptions=future)
    sources = [
        UncertaintySource(source="sensor noise and resolution", category="MEASUREMENT", treatment="QUALITATIVE",
                          value=fd.residual_std_c if fd else None, unit="°C" if fd else None,
                          in_empirical_interval=True,
                          note="training residual std (an upper bound: includes misfit); not used for intervals"),
        UncertaintySource(source="missing, suspicious and irregular readings", category="MEASUREMENT",
                          treatment="QUALITATIVE", in_empirical_interval=None,
                          note=f"{report.counts.rows_suspicious if report.counts else 'n/a'} suspicious, "
                               f"{report.n_gaps} gap(s), irregular "
                               f"{report.intervals.irregular_pct if report.intervals else 'n/a'} %"),
        UncertaintySource(source="parameters", category="PARAMETER", treatment="QUANTIFIED"
                          if param.status == "ASSESSED" else "NOT_ASSESSED", value=param.envelope_width_c,
                          unit="°C" if param.envelope_width_c is not None else None, in_empirical_interval=False,
                          note=param.method + "; reported beside, never added to, the empirical interval"),
        UncertaintySource(source="model inadequacy", category="MODEL_INADEQUACY", treatment="QUANTIFIED"
                          if he.diag.bias_c is not None else "NOT_ASSESSED", value=he.diag.bias_c,
                          unit="°C" if he.diag.bias_c is not None else None, in_empirical_interval=True,
                          note="held-out mean error at the horizon; only for conditions seen in held-out data"),
        UncertaintySource(source="operating regime", category="REGIME", treatment="QUALITATIVE",
                          in_empirical_interval=None, note=regime_ev + "; unseen regimes are not represented"),
        UncertaintySource(source="horizon", category="EXTRAPOLATION", treatment="QUALITATIVE",
                          in_empirical_interval=None, note=evidence.extrapolation_status),
        UncertaintySource(source="future inputs", category="FUTURE_INPUT", treatment="NOT_ASSESSED",
                          in_empirical_interval=False,
                          note="the interval is conditional on: " + "; ".join(conditional)),
        UncertaintySource(source="other rooms", category="GENERALIZATION", treatment="NOT_ASSESSED",
                          in_empirical_interval=False, note="calibration comes from this log only and is never "
                          "transferred; cross-room reliability is unvalidated"),
    ]
    test_start = float(sink["cuts"][1]) if sink.get("cuts") else None
    t_abs = lambda m: (f.ts.iloc[0] + pd.to_timedelta(m, unit="min")).floor("us").to_pydatetime()   # noqa: E731
    calibration = None
    if boundary is not None and test_start is not None:
        calibration = CalibrationRecord(
            design="chronological: FIT 60 % / SELECT 20 % (Phase 4); TEST 20 % split in half by time into "
                   "CALIBRATION and EVALUATION (Phase 5)", test_period_start=t_abs(test_start),
            boundary=t_abs(boundary),
            calibration_origins=(t_abs(min(_t(f, c) for c in he.cal)), t_abs(max(_t(f, c) for c in he.cal)))
            if he.cal else None,
            calibration_latest_truth=t_abs(max(_t(f, c) + h for c in he.cal)) if he.cal else None,
            evaluation_origins=(t_abs(min(_t(f, c) for c in he.ev)), t_abs(max(_t(f, c) for c in he.ev)))
            if he.ev else None,
            purge_rule="a calibration case is kept only if its truth (origin + horizon) is at or before the "
                       "boundary; evaluation origins are at or after it, in passive periods with no calibration "
                       "case",
            independence="unit of inference: the passive run (one thermal episode; pieces split by gaps, flagged "
                         "readings or events are merged unless cooling is recorded between them); all its cases "
                         "weigh 1 together; calibration and evaluation runs are disjoint; runs are separate passive "
                         "episodes of ONE log, assumed independent (lag-1 screen), not separate rooms",
            cross_room="UNVALIDATED: same-room temporal evaluation only; calibration is never transferred")

    recs = _recommendations(assessment, uncertainty, he, level, h, sel_h, scope, user_assumed, issues, regime)
    limitations = [
        f"validation scope {scope.value}: {_SCOPE_TEXT[scope]}",
        "the reliability status is a rule-based classification, not a probability",
        "same-log temporal evaluation only; reliability does not transfer to another room",
        "each passive period is one observation and periods are assumed independent (screened, not proven); "
        "coverage is checked on separate periods, not guaranteed",
    ]
    downgrade = None
    if assessment.status != H:
        downgrade = "; ".join(g.name for g in gates if g.outcome == GateOutcome.FAIL
                              and g.cap_if_failed == assessment.status)
    return ReliabilityReport(
        method_version=METHOD_VERSION, forecast_reference=f"enclosing ForecastResult (Phase 4 {ENGINE_VERSION})",
        forecast_available=True, config=config, causal_view=view, reliability=assessment, uncertainty=uncertainty,
        horizon_diagnostics=[evals[k].diag for k in sorted(evals)], calibration=calibration, crossing=crossing,
        parameter_uncertainty=param, uncertainty_sources=sources, evidence=evidence, limitations=limitations,
        recommendations=recs, refusal_or_downgrade_reason=downgrade)


def _recommendations(a: ReliabilityAssessment, u: UncertaintyAssessment, he: _HEval, level: float, h: float,
                     sel_h, scope, user_assumed, issues, regime) -> list[EvidenceRecommendation]:
    failed = set(a.gates_failed)
    need = order_stat_requirement(level)
    recs = []
    if "WITHIN_VALIDATED_HORIZON" in failed:
        recs.append(_rec("held-out evidence at the requested horizon", "the forecast extrapolates beyond the "
                         "validated horizon, so no reliability can be stated", "an assessable forecast",
                         "request a horizon within the validated one, or log longer passive periods",
                         f"requested horizon <= validated horizon ({sel_h:g} min)" if sel_h else "a validated horizon"))
    if "INTERVAL_CALIBRATED" in failed:
        if u.status == IntervalStatus.UNDER_COVERING:
            recs.append(_rec("more representative held-out periods", "the calibrated interval covered too few "
                             "evaluation periods: errors changed between the two halves (regime change, "
                             "disturbance or model mismatch)", "an interval whose coverage holds",
                             "check the evaluation period for unrecorded events; log more stable operation",
                             "evaluation coverage consistent with the nominal level"))
        elif u.status == IntervalStatus.DEPENDENT_PERIODS:
            recs.append(_rec("more representative held-out periods", "held-out errors drift from one passive period "
                             "to the next (serially dependent period means: regime change, drift or disturbance), "
                             "so coverage cannot be checked on independent periods", "an interval whose coverage "
                             "can be checked", "check the held-out periods for unrecorded events, a changed load or "
                             "a drifting ambient; log more stable operation",
                             "period-mean errors without detectable serial dependence"))
        else:
            recs.append(_rec("longer passive history", f"an interval at {level:.0%} needs {need} passive periods "
                             f"in each of calibration and evaluation (has {he.cal_periods} and {he.ev_periods}); "
                             "cases and windows inside one period do not count separately",
                             "a calibrated, evaluated interval", "keep logging through more passive periods "
                             "(compressor OFF cycles or recorded outages)", f">= {need} passive periods in each of "
                             "calibration and evaluation", necessity=Necessity.POTENTIAL_IMPROVEMENT,
                             priority=Priority.MEDIUM))
    if "INDEPENDENT_PERIODS" in failed:
        recs.append(_rec("more independent passive periods", "all held-out evidence comes from too few separate "
                         "passive periods to separate period-to-period variation from noise",
                         "evidence across separate periods", "record more OFF cycles or outages",
                         f">= {RL.MIN_INDEPENDENT_PERIODS} periods in each half",
                         necessity=Necessity.POTENTIAL_IMPROVEMENT, priority=Priority.MEDIUM))
    if "COVERAGE_EVIDENCE_PRECISION" in failed:
        recs.append(_rec("more evaluation periods", f"the observed coverage over {he.ev_periods} evaluation "
                         "period(s) is too imprecise for a HIGH status", "a precise coverage estimate",
                         "longer logging (more passive periods)", f"Wilson half-width <= "
                         f"{RL.COVERAGE_PRECISION_FOR_HIGH}", necessity=Necessity.POTENTIAL_IMPROVEMENT,
                         priority=Priority.LOW))
    if "NO_PERIOD_DEPENDENCE_DETECTED" in failed:
        recs.append(_rec("explanation of the period-to-period drift", "held-out errors change systematically from "
                         "one passive period to the next, so the periods are not independent evidence",
                         "independent held-out periods", "look for a regime change, a drifting ambient, sensor "
                         "drift or a changed load in the held-out periods", "period-mean errors without detectable "
                         "serial dependence", necessity=Necessity.POTENTIAL_IMPROVEMENT, priority=Priority.MEDIUM))
    if "REGIME_EVIDENCE_RECORDED" in failed:
        recs.append(_rec("compressor_state", "the passive regime rests on a declaration or a manual record",
                         "recorded regime evidence", "log the compressor run signal on the same clock",
                         "a compressor_state column or system-recorded power interruptions",
                         necessity=Necessity.POTENTIAL_IMPROVEMENT, priority=Priority.MEDIUM, kind="INPUT"))
    if "REGIME_MATCHES_CALIBRATION" in failed:
        recs.append(_rec("held-out cases that start with cooling ON", "the what-if forecast starts in a regime "
                         "the held-out evidence does not contain", "evidence for this regime",
                         "record real power cuts during operation (system-logged)",
                         "held-out cooling-loss cases from an active origin",
                         necessity=Necessity.POTENTIAL_IMPROVEMENT, priority=Priority.MEDIUM))
    if "NO_MATERIAL_BIAS" in failed:
        recs.append(_rec("explanation of the systematic error", "held-out errors have a material mean offset",
                         "an unbiased model or an explained offset", "look for unrecorded loads, doors, defrost "
                         "or a sensor offset in the held-out periods", "no material held-out bias",
                         necessity=Necessity.POTENTIAL_IMPROVEMENT, priority=Priority.MEDIUM))
    if "NO_USER_ASSUMED_FUTURE_INPUT" in failed:
        recs.append(_rec("validated future-input evidence", "the forecast depends on " + "; ".join(user_assumed)
                         + "; the interval is conditional on it", "an unconditional statement (not implemented)",
                         "document the source of the assumed value", "no user-assumed future input",
                         necessity=Necessity.FUTURE_CAPABILITY, priority=Priority.LOW, kind="SCENARIO"))
    if "DATA_QUALITY" in failed:
        recs.append(_rec("cleaner logging", "material Phase 2 warnings: " + ", ".join(i.code for i in issues),
                         "fewer data-quality limitations", "fix the logger issues named", "no material warning",
                         necessity=Necessity.POTENTIAL_IMPROVEMENT, priority=Priority.MEDIUM, kind="INPUT"))
    if "VALIDATION_SCOPE" in failed and scope == ValidationScope.REAL_WORLD_UNVALIDATED:
        recs.append(_rec("prospective real-world validation", "the method has not been checked against real "
                         "outcomes; a real log cannot be rated HIGH", "a REAL_WORLD_VALIDATED scope",
                         "record predictions before at least three controlled unplug tests and compare with the "
                         "observed outcomes and the naive baseline", "documented real-world validation evidence"))
    elif "VALIDATION_SCOPE" in failed and scope == ValidationScope.EXPECTED_ONLY:
        recs.append(_rec("measured (REAL) log of the room", "the input is a model output (EXPECTED): agreement "
                         "with it says nothing about the room", "a reliability statement about a measured room",
                         "use the logger's measured export", "a log of measurements labelled REAL"))
    elif "VALIDATION_SCOPE" in failed:
        recs.append(_rec("established data provenance", "the log's origin is unknown (UNVALIDATED), so it is treated "
                         "neither as measurements nor as a simulation", "a scope that matches the data",
                         "record where the log comes from: REAL only for a physical logger's export, SIMULATED "
                         "only for generated data", "a log with recorded provenance (REAL or SIMULATED)"))
    return recs


# ====================================================================== entry point
def assess_source(src, *, source: str | None = None, evidence: EvidenceClass = EvidenceClass.UNVALIDATED,
                  options=None, limits=None, request: ForecastRequest, events: list | None = None, metadata=None,
                  config: ReliabilityConfig | None = None):
    """Phase 2 + Phase 4 on the causal view of the log, then the Phase 5 assessment attached to the result.
    Returns (clean, observations, data-quality report, ForecastResult with reliability_report)."""
    config = config or ReliabilityConfig()
    cut, total, excluded, note = causal_source(src, options, request, evidence=evidence, limits=limits)
    clean, obs, report = validate_source(cut, source=source, evidence=evidence, options=options, limits=limits)
    sink: dict = {}
    result = forecast(clean, report, obs, request, events=events, metadata=metadata, sink=sink)
    view = CausalView(origin=request.origin, truncated=excluded > 0,
                      rows_in_source=total if total is not None else int(len(obs)),
                      rows_excluded_after_origin=excluded,
                      causal_input_sha256=report.input.sha256 if report.input else None, note=note)
    rep = assess(result, sink, report, request, config, view, events)
    # constructed (not copied) so the result-level consistency rules run on the combined object
    return clean, obs, report, ForecastResult(**{**dict(result), "reliability_report": rep})


__all__ = ["METHOD_VERSION", "assess", "assess_source", "causal_source", "dependence_screen", "empirical_offsets",
           "non_overlapping_windows", "order_stat_requirement", "passive_runs", "period_coverage", "period_mean_test",
           "period_weighted_offsets", "split_boundary", "wilson"]
