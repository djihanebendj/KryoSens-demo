"""Event segmentation: validated observations -> traceable operational events.

Input is Phase 2's validated output (the analysis frame: usable rows, sorted,
unique timestamps, with ``source_line``) and its report. Nothing here modifies
data, fills gaps or infers compressor state. Every event names the rows that
support it. Events describe the data; none is a diagnosis.

Rules (all numbers come from ``SegmentationParams``; defaults are provisional):

* Blocks. Consecutive usable observations further apart than
  ``gap_factor`` x median interval start a new block and produce a DATA_GAP.
  Nothing is ever carried across a gap. An event touching a block edge is
  *censored* there: it may have started earlier / continued later.
* Temperature trends (per block). The temperature is smoothed with a
  rolling median (``smooth_window_min``, time-based, so irregular sampling is
  handled) only to locate boundaries; statistics use the raw values. The
  smoothed curve is split into straight pieces (Douglas-Peucker: split at the
  point furthest from the line between a piece's ends until every point is
  within ``piece_tolerance_c`` of its line). Each piece is
  STABLE      if its smoothed change is below ``stable_band_c``;
  IRREGULAR   if it changes more but lasts < ``min_trend_min`` or its
              residual std around a line exceeds ``max_trend_resid_c``;
  RISING/FALLING otherwise.
  The splitter never creates a piece with fewer than ``min_points``
  observations; shorter features stay inside a piece and are disclosed as brief
  deviations. Blocks with fewer than ``min_points`` observations are not
  classified (INSUFFICIENT_DATA). Adjacent pieces with the same label are
  merged. Adjacent trend events share their boundary observation.
* Threshold excursions (only if a limit is given). A run of consecutive
  observations above the upper / below the lower limit. Sustained if its
  observed duration >= ``min_excursion_min``, else brief; if it is shorter but
  censored, it cannot be called brief and becomes INSUFFICIENT_DATA.
* Compressor (only if the column exists and >= ``compressor_min_valid`` of
  usable rows carry a valid state). Runs of ON / OFF / missing state, the
  switch between two consecutive valid observations, and complete cycles
  (an ON period then an OFF period, both bounded by observed switches).
* Settings. ``SegmentationParams`` rejects invalid values without data. Before
  any event is built, settings that cannot be applied to *this* log are
  refused with reasons (a setting that would bridge a gap validation reported,
  or a duration longer than every gap-free stretch, so the result would be
  decided by the setting rather than the data). Settings that are honoured but
  weaken a control are disclosed in the result's limitations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from coldsense.core.errors import DataError

from . import thresholds as C
from .schemas import (
    DataQualityReport,
    Event,
    EventType,
    SegmentationParams,
    SegmentationResult,
    ValidationLimits,
    Verdict,
)

TREND_TYPES = (EventType.TEMP_STABLE, EventType.TEMP_RISING, EventType.TEMP_FALLING)
_TYPE_ORDER = {t: i for i, t in enumerate(EventType)}
_GLOBAL_ISSUES = {
    "IRREGULAR_SAMPLING": "sampling is irregular; turning points and durations are less precise",
    "SUSPICIOUS_VALUES": "some readings were flagged as suspicious by validation; affected events say so",
    "TIMEZONE_UNKNOWN": "times are local clock time of unknown zone; a clock change would distort durations",
    "GAPS": "the log has gaps; events never span them and are marked censored at their edges",
}


def _fit(t_min: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    """Least-squares slope (°C/min) and residual std; None when not computable."""
    if len(y) < 3 or np.ptp(t_min) == 0:
        return None, None
    slope, icpt = np.polyfit(t_min, y, 1)
    resid = y - (slope * t_min + icpt)
    return float(slope), float(np.sqrt(np.mean(resid ** 2)))


def _runs(keys: list) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of consecutive equal keys."""
    out, start = [], 0
    for i in range(1, len(keys) + 1):
        if i == len(keys) or keys[i] != keys[start]:
            out.append((start, i - 1))
            start = i
    return out


class _Ctx:
    """Shared, read-only view of the validated data for building events."""

    def __init__(self, clean: pd.DataFrame, report: DataQualityReport, obs: pd.DataFrame | None):
        self.ts = clean["ts"].reset_index(drop=True)
        self.t_min = ((self.ts - self.ts.iloc[0]).dt.total_seconds() / 60).to_numpy()
        self.temp = clean["temp_c"].to_numpy(dtype=float)
        self.amb = clean["ambient_c"].to_numpy(dtype=float) if "ambient_c" in clean else None
        self.lines = clean["source_line"].to_numpy()
        self.evidence = report.evidence
        self.suspicious: set[int] = set()
        self.obs = obs
        if obs is not None and "flags" in obs:
            hit = obs["flags"].str.contains("TEMP_SUSPICIOUS", regex=False)
            self.suspicious = set(obs.loc[hit, "source_line"].tolist())

    def make(self, etype: EventType, a: int, b: int, *, block: tuple[int, int], rationale: str,
             observed: dict | None = None, derived: dict | None = None, limitations: list[str] | None = None,
             start_censored: bool | None = None, end_censored: bool | None = None,
             rows: list[int] | None = None) -> Event:
        """Event over clean rows a..b (inclusive); censoring defaults to touching the block edge."""
        bs, be = block
        sc = (a == bs) if start_censored is None else start_censored
        ec = (b == be) if end_censored is None else end_censored
        rows = list(range(a, b + 1)) if rows is None else rows
        lines = [int(self.lines[i]) for i in rows]
        lim = list(limitations or [])
        n_susp = sum(1 for ln in lines if ln in self.suspicious)
        if n_susp:
            lim.append(f"{n_susp} supporting reading(s) flagged TEMP_SUSPICIOUS by validation")
        if sc or ec:
            lim.append("cut off by the start/end of the log or a data gap; duration is a lower bound")
        dur = float(self.t_min[b] - self.t_min[a]) if b > a else None
        return Event(
            event_id="", type=etype,
            start=self.ts.iloc[a].to_pydatetime(), end=self.ts.iloc[b].to_pydatetime(),
            duration_min=round(dur, 4) if dur is not None else None,
            duration_is_lower_bound=bool(sc or ec), start_censored=bool(sc), end_censored=bool(ec),
            before=self.ts.iloc[a - 1].to_pydatetime() if a > bs else None,
            after=self.ts.iloc[b + 1].to_pydatetime() if b < be else None,
            n_observations=len(rows), source_lines=lines,
            observed=observed or {}, derived=derived or {}, rationale=rationale, limitations=lim,
            evidence=self.evidence,
        )

    def temp_observed(self, a: int, b: int) -> dict:
        seg = self.temp[a:b + 1]
        out = {"temp_start_c": float(seg[0]), "temp_end_c": float(seg[-1]),
               "temp_min_c": float(seg.min()), "temp_max_c": float(seg.max())}
        if self.amb is not None:
            amb = self.amb[a:b + 1]
            out["ambient_mean_c"] = float(np.nanmean(amb)) if np.isfinite(amb).any() else None
        return out


# ------------------------------------------------------------------ trends
def _smoothed(ctx: _Ctx, a0: int, b0: int, p: SegmentationParams) -> np.ndarray:
    s = pd.Series(ctx.temp[a0:b0 + 1], index=pd.DatetimeIndex(ctx.ts.iloc[a0:b0 + 1]))
    # whole nanoseconds: a text window such as "1e-05min" is not understood by pandas
    window = pd.Timedelta(round(p.smooth_window_min * 60e9), unit="ns")
    return s.rolling(window, center=True, min_periods=1).median().to_numpy()


def _trend_pieces(ctx: _Ctx, a0: int, b0: int, smooth: np.ndarray, p: SegmentationParams) -> list[tuple[int, int]]:
    """Piecewise-linear split (Douglas-Peucker) of the smoothed curve: a piece is split at
    its point furthest from the straight line between its ends until every point lies
    within ``piece_tolerance_c`` of its piece's line. Plateaus, ramps and turning points
    become piece boundaries; noise smaller than the tolerance never does. No split may
    leave a piece with fewer than ``min_points`` observations: features shorter than
    that stay inside a piece and are disclosed as brief deviations."""
    t = ctx.t_min[a0:b0 + 1]
    m = p.min_points - 1                       # minimum intervals on each side of a split
    cuts = {0, len(t) - 1}
    stack = [(0, len(t) - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2 * m:
            continue
        span = t[j] - t[i]
        line = smooth[i] + (smooth[j] - smooth[i]) * (t[i:j + 1] - t[i]) / span if span > 0 else smooth[i]
        dev = np.abs(smooth[i:j + 1] - line)
        dev[:m] = dev[len(dev) - m:] = -1.0     # splits too close to either end are not allowed
        k = int(np.argmax(dev))
        if dev[k] > p.piece_tolerance_c:
            cuts.add(i + k)
            stack += [(i, i + k), (i + k, j)]
    c = sorted(cuts)
    return [(a0 + x, a0 + y) for x, y in zip(c[:-1], c[1:], strict=True)]


def _classify(ctx: _Ctx, a: int, b: int, smooth_delta: float, p: SegmentationParams) -> tuple[EventType, str]:
    dur = ctx.t_min[b] - ctx.t_min[a]
    slope, resid = _fit(ctx.t_min[a:b + 1], ctx.temp[a:b + 1])
    if abs(smooth_delta) < p.stable_band_c:
        return EventType.TEMP_STABLE, f"smoothed change {smooth_delta:+.2f} °C is within ±{p.stable_band_c} °C"
    if dur < p.min_trend_min:
        return EventType.TEMP_IRREGULAR, (f"changed {smooth_delta:+.2f} °C in {dur:.1f} min, shorter than the "
                                          f"{p.min_trend_min} min needed to call a sustained trend")
    if resid is None or resid > p.max_trend_resid_c:
        r = "n/a" if resid is None else f"{resid:.2f}"
        return EventType.TEMP_IRREGULAR, (f"changed {smooth_delta:+.2f} °C but unevenly (residual {r} °C > "
                                          f"{p.max_trend_resid_c} °C)")
    kind = EventType.TEMP_RISING if smooth_delta > 0 else EventType.TEMP_FALLING
    return kind, (f"changed {smooth_delta:+.2f} °C over {dur:.1f} min, steadily (slope {slope:+.3f} °C/min, "
                  f"residual {resid:.2f} °C)")


def _merge(labelled: list[list]) -> list[list]:
    """Merge adjacent [a, b, type, reason] entries with the same type (they share a boundary)."""
    out: list[list] = []
    for a, b, t, why in labelled:
        if out and out[-1][2] == t:
            out[-1][1] = b
            out[-1][3] = None  # recomputed after merging
        else:
            out.append([a, b, t, why])
    return out


def _trend_events(ctx: _Ctx, a0: int, b0: int, p: SegmentationParams) -> list[Event]:
    block = (a0, b0)
    smooth = _smoothed(ctx, a0, b0, p)
    sm = lambda i: smooth[i - a0]  # noqa: E731

    labelled = [[a, b, *_classify(ctx, a, b, sm(b) - sm(a), p)]
                for a, b in _trend_pieces(ctx, a0, b0, smooth, p)]
    events = []
    for a, b, t, why in _merge(labelled):
        slope, resid = _fit(ctx.t_min[a:b + 1], ctx.temp[a:b + 1])
        delta = sm(b) - sm(a)
        if why is None:  # merged run: describe the merged period
            why = (f"adjacent pieces each changed less than ±{p.stable_band_c} °C (net {delta:+.2f} °C)"
                   if t == EventType.TEMP_STABLE else _classify(ctx, a, b, delta, p)[1] + " (merged pieces)")
        derived = {"smoothed_change_c": round(float(delta), 4),
                   "slope_c_per_min": None if slope is None else round(slope, 5),
                   "fit_residual_c": None if resid is None else round(resid, 4)}
        # brief deviations the smoothing hides: disclosed, never silently absorbed
        dev = np.abs(ctx.temp[a:b + 1] - smooth[a - a0:b - a0 + 1])
        n_brief = int((dev >= p.stable_band_c).sum())
        observed = {**ctx.temp_observed(a, b), "brief_deviations": n_brief}
        lim = []
        if n_brief:
            lim.append(f"{n_brief} reading(s) deviate ≥ {p.stable_band_c} °C from the smoothed curve for less than "
                       f"the {p.smooth_window_min} min smoothing window (brief disturbances, not part of the trend; "
                       "see temp_min_c/temp_max_c and any excursion events)")
        events.append(ctx.make(t, a, b, block=block, rationale=why, observed=observed, derived=derived,
                               limitations=lim))
    return events


# ------------------------------------------------------------------ excursions
def _excursion_events(ctx: _Ctx, a0: int, b0: int, p: SegmentationParams) -> list[Event]:
    events = []
    seg = ctx.temp[a0:b0 + 1]
    for side, limit in (("above", p.upper_threshold_c), ("below", p.lower_threshold_c)):
        if limit is None:
            continue
        beyond = (seg > limit) if side == "above" else (seg < limit)
        for x, y in _runs(beyond.tolist()):
            if not beyond[x]:
                continue
            a, b = a0 + x, a0 + y
            dur = ctx.t_min[b] - ctx.t_min[a]
            censored = a == a0 or b == b0
            peak = float(seg[x:y + 1].max() if side == "above" else seg[x:y + 1].min())
            observed = {"limit_c": limit, "side": side, "peak_c": peak, **ctx.temp_observed(a, b)}
            derived = {"max_exceedance_c": round(abs(peak - limit), 4), "observed_duration_min": round(dur, 4)}
            where = (f"{y - x + 1} consecutive observation(s) {side} {limit} °C for {dur:.1f} min "
                     f"(peak {peak:.2f} °C); the crossings happened between the bracketing observations")
            if dur >= p.min_excursion_min:
                t, why = EventType.THRESHOLD_EXCURSION, f"sustained: {where} (≥ {p.min_excursion_min} min)"
            elif censored:
                t, why = EventType.INSUFFICIENT_DATA, (f"{where}, cut off by a gap or the log edge: cannot tell a "
                                                       f"brief excursion from a sustained one")
            else:
                t, why = EventType.BRIEF_EXCURSION, f"brief: {where} (< {p.min_excursion_min} min)"
            events.append(ctx.make(t, a, b, block=(a0, b0), rationale=why, observed=observed, derived=derived))
    return events


# ------------------------------------------------------------------ compressor
def _compressor_events(ctx: _Ctx, state: pd.Series, a0: int, b0: int) -> list[Event]:
    events: list[Event] = []
    vals = state.iloc[a0:b0 + 1].tolist()
    key = ["UNKNOWN" if pd.isna(v) else ("ON" if v else "OFF") for v in vals]
    runs = [(a0 + x, a0 + y, key[x]) for x, y in _runs(key)]
    timing = ("durations run from the first observation in a state to the first in the next; the true switch "
              "times are only known to within one sampling interval")
    for k, (a, b, st) in enumerate(runs):
        prev_known = k > 0 and runs[k - 1][2] != "UNKNOWN"
        next_known = k + 1 < len(runs) and runs[k + 1][2] != "UNKNOWN"
        if st == "UNKNOWN":
            events.append(ctx.make(EventType.COMPRESSOR_UNKNOWN, a, b, block=(a0, b0),
                                   rationale=f"{b - a + 1} observation(s) with missing or invalid compressor "
                                             "state; the state is not inferred from temperature",
                                   observed={"state": None}))
            continue
        sc, ec = not prev_known, not next_known
        derived = {}
        if next_known:
            derived["duration_to_next_state_min"] = round(float(ctx.t_min[runs[k + 1][0]] - ctx.t_min[a]), 4)
        events.append(ctx.make(EventType.COMPRESSOR_ON if st == "ON" else EventType.COMPRESSOR_OFF, a, b,
                               block=(a0, b0), start_censored=sc, end_censored=ec,
                               rationale=f"recorded compressor state {st} in {b - a + 1} consecutive observation(s)",
                               observed={"state": st}, derived=derived, limitations=[timing] if derived else []))
        if prev_known:  # switch between the last row of the previous run and the first of this one
            i = a - 1
            events.append(ctx.make(EventType.COMPRESSOR_TRANSITION, i, a, block=(a0, b0), start_censored=False,
                                   end_censored=False, rows=[i, a],
                                   rationale=f"recorded state changed {runs[k - 1][2]} → {st} between these two "
                                             "observations; the exact switch time is not known",
                                   observed={"from": runs[k - 1][2], "to": st},
                                   derived={"switch_window_min": round(float(ctx.t_min[a] - ctx.t_min[i]), 4)}))
    # complete cycles: ON then OFF, both bounded by observed switches on both sides
    for k in range(1, len(runs) - 2):
        (_, _, ps), (oa, _, os_), (fa, fb, fs), (na, _, ns) = runs[k - 1], runs[k], runs[k + 1], runs[k + 2]
        if (ps, os_, fs, ns) == ("OFF", "ON", "OFF", "ON"):
            on_min = float(ctx.t_min[fa] - ctx.t_min[oa])
            off_min = float(ctx.t_min[na] - ctx.t_min[fa])
            events.append(ctx.make(
                EventType.COMPRESSOR_CYCLE, oa, fb, block=(a0, b0), start_censored=False, end_censored=False,
                rationale="complete cycle: an ON period and the following OFF period, each bounded by recorded "
                          "switches",
                observed={"states": "ON→OFF"},
                derived={"on_min": round(on_min, 4), "off_min": round(off_min, 4),
                         "period_min": round(on_min + off_min, 4),
                         "duty_cycle": round(on_min / (on_min + off_min), 4)},
                limitations=[timing]))
    return events


# ------------------------------------------------------------------ settings vs this log
def _inapplicable_settings(ctx: _Ctx, report: DataQualityReport, p: SegmentationParams, step_s: np.ndarray,
                           med: float, blocks: list[tuple[int, int]]) -> list[str]:
    """Reasons to refuse: settings whose result on this log would be decided by the
    setting, not the data, or would break the 'never bridge a gap' rule."""
    out = []
    bridged = step_s[(step_s > C.GAP_FACTOR * med) & (step_s <= p.gap_factor * med)]
    if len(bridged):
        out.append(f"setting gap_factor = {p.gap_factor}: {len(bridged)} interval(s) that validation reports as "
                   f"gaps (longer than {C.GAP_FACTOR} × the median interval of {med:.0f} s; longest "
                   f"{bridged.max() / 60:.1f} min) would not be treated as gaps, so events would span missing "
                   f"data. Use gap_factor ≤ {C.GAP_FACTOR}.")
    longest = max(float(ctx.t_min[b] - ctx.t_min[a]) for a, b in blocks)
    where = f"the longest stretch of this log without a gap ({longest:.1f} min)"
    if p.smooth_window_min > longest:
        out.append(f"setting smooth_window_min = {p.smooth_window_min} min: longer than {where}; every smoothed "
                   "value would be the median of (nearly) the whole stretch, so boundaries would not follow the data.")
    if p.min_trend_min > longest:
        out.append(f"setting min_trend_min = {p.min_trend_min} min: longer than {where}; no rise or fall could "
                   "ever count as a trend.")
    if (p.upper_threshold_c is not None or p.lower_threshold_c is not None) and p.min_excursion_min > longest:
        out.append(f"setting min_excursion_min = {p.min_excursion_min} min: longer than {where}; no excursion "
                   "could ever count as sustained, so every one would be reported as brief.")
    lim = report.limits_used or ValidationLimits()
    if p.stable_band_c >= lim.plausible_max_c - lim.plausible_min_c:
        out.append(f"setting stable_band_c = {p.stable_band_c} °C: at least the whole range validation accepts "
                   f"({lim.plausible_min_c} to {lim.plausible_max_c} °C); every period would be STABLE whatever "
                   "the readings.")
    return out


def _setting_notes(ctx: _Ctx, p: SegmentationParams, med: float) -> list[str]:
    """Limitations for settings that are honoured but weaken a control on this log."""
    notes = []
    if p.smooth_window_min * 60 <= 2 * med:
        notes.append(f"The {p.smooth_window_min} min smoothing window holds fewer than three readings at the "
                     f"median sampling interval ({med:.0f} s), so it barely smooths: boundaries come from almost "
                     "raw readings and brief deviations are disclosed only partly or not at all.")
    spread = float(ctx.temp.max() - ctx.temp.min())
    if spread > C.EVT_STABLE_BAND_C and p.stable_band_c >= spread:
        notes.append(f"stable_band_c ({p.stable_band_c} °C) is at least the whole temperature range of this log "
                     f"({spread:.2f} °C): every period is necessarily STABLE; that reflects the band chosen, "
                     "not an absence of change.")
    return notes


# ------------------------------------------------------------------ entry point
def segment_events(clean: pd.DataFrame, report: DataQualityReport, observations: pd.DataFrame | None = None,
                   params: SegmentationParams | None = None) -> SegmentationResult:
    """Segment Phase 2's analysis frame into events, or refuse with reasons."""
    # re-validate: model_copy(update=...) and model_construct() skip pydantic validators
    p = SegmentationParams.model_validate((params or SegmentationParams()).model_dump())
    basis = report.input.timezone_handling if report.input else "unknown"
    result = SegmentationResult(refused=True, params=p, evidence=report.evidence, time_basis=basis)
    result.limitations = [
        "Events describe the data; none is a diagnosis or a cause.",
        "Segmentation parameters are provisional defaults checked on SIMULATED data only.",
    ]
    if report.verdict == Verdict.REFUSED:
        result.reasons = ["validation refused this log: " + "; ".join(i.message for i in report.blocking)]
        return result
    if len(clean) < p.min_points:
        result.reasons = [f"only {len(clean)} usable observation(s); at least {p.min_points} needed"]
        return result
    ts = clean["ts"]
    if not ts.is_monotonic_increasing or not ts.is_unique or "source_line" not in clean:
        raise DataError("segment_events expects Phase 2's analysis frame (sorted, unique timestamps, source_line)")
    med = report.median_interval_s
    if not med or med <= 0:
        result.reasons = ["no usable median sampling interval"]
        return result

    ctx = _Ctx(clean, report, observations)
    events: list[Event] = []

    # --- blocks and gaps ---
    step_s = np.diff(ctx.t_min) * 60
    gap_after = np.flatnonzero(step_s > p.gap_factor * med)
    starts = [0, *(gap_after + 1).tolist()]
    ends = [*gap_after.tolist(), len(clean) - 1]
    blocks = list(zip(starts, ends, strict=True))
    refusals = _inapplicable_settings(ctx, report, p, step_s, med, blocks)
    if refusals:
        result.reasons = refusals
        return result
    for g in gap_after:
        lim = []
        if observations is not None:
            inside = observations[(observations["ts"] > ctx.ts.iloc[g]) & (observations["ts"] < ctx.ts.iloc[g + 1])]
            if len(inside):
                lim.append(f"{len(inside)} observation(s) in this interval were excluded by validation "
                           f"(lines {inside['source_line'].tolist()[:10]})")
        events.append(ctx.make(
            EventType.DATA_GAP, int(g), int(g + 1), block=(int(g), int(g + 1)), start_censored=False,
            end_censored=False, rows=[int(g), int(g + 1)],
            rationale=f"no usable observation for {step_s[g] / 60:.1f} min (> {p.gap_factor} × median "
                      f"interval {med:.0f} s); nothing is carried across it",
            observed={"last_before_c": float(ctx.temp[g]), "first_after_c": float(ctx.temp[g + 1])},
            derived={"gap_min": round(float(step_s[g] / 60), 4)}, limitations=lim))
    for e in events:  # the two rows bracket the gap; they are not "inside" it
        e.n_observations = 0

    # --- per block: trends, excursions ---
    for a0, b0 in blocks:
        if b0 - a0 + 1 < p.min_points:
            events.append(ctx.make(EventType.INSUFFICIENT_DATA, a0, b0, block=(a0, b0),
                                   rationale=f"isolated block of {b0 - a0 + 1} observation(s) between gaps or log "
                                             f"edges, fewer than {p.min_points}: not classified",
                                   observed=ctx.temp_observed(a0, b0)))
            continue
        events += _trend_events(ctx, a0, b0, p)
        events += _excursion_events(ctx, a0, b0, p)

    # --- compressor: only from valid recorded state ---
    if "compressor_on" not in clean:
        comp_status = "absent: no compressor_state column; no compressor events"
    else:
        state = clean["compressor_on"].reset_index(drop=True)
        valid = float(state.notna().mean())
        if valid < p.compressor_min_valid:
            comp_status = (f"refused: only {100 * valid:.1f}% of usable rows carry a valid compressor state "
                           f"(need {100 * p.compressor_min_valid:.0f}%)")
            result.limitations.append("Compressor state is too incomplete to segment; it is not inferred.")
        else:
            comp_status = f"segmented ({100 * valid:.1f}% of usable rows carry a valid state)"
            for a0, b0 in blocks:
                events += _compressor_events(ctx, state, a0, b0)

    # --- deterministic order and ids ---
    events.sort(key=lambda e: (e.start, _TYPE_ORDER[e.type], e.end, e.source_lines))
    for i, e in enumerate(events, 1):
        e.event_id = f"E{i:04d}"

    covered = {ln for e in events if e.type in TREND_TYPES for ln in e.source_lines}
    counts = {t.value: sum(e.type == t for e in events) for t in EventType}
    result.refused = False
    result.events = events
    result.summary = {
        "usable_observations": len(clean), "blocks": len(blocks), **{f"n_{k}": v for k, v in counts.items() if v},
        "trend_classified_pct": round(100 * len(covered) / len(clean), 2),
        "compressor": comp_status,
        "upper_threshold_c": "none" if p.upper_threshold_c is None else p.upper_threshold_c,
        "lower_threshold_c": "none" if p.lower_threshold_c is None else p.lower_threshold_c,
    }
    result.limitations += [msg for code, msg in _GLOBAL_ISSUES.items() if any(i.code == code for i in report.issues)]
    result.limitations += _setting_notes(ctx, p, med)
    return result
