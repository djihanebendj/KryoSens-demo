"""Operating-regime evidence and segment selection, built on Phase 2/3 output.

Passive (no active cooling) stretches come ONLY from evidence:
* RECORDED: Phase 3 COMPRESSOR_OFF events (recorded state, never inferred), or
* DECLARED: the user's ``passive_since`` (an assumption, e.g. a known power cut);
  contradicted -> refused if the compressor is recorded ON inside it.
A temperature shape is never taken as proof of a regime.

Every stretch ends at the forecast origin at the latest (no future data), never
spans a Phase 3 DATA_GAP, and is split at readings Phase 2 flagged
TEMP_SUSPICIOUS (unexplained disturbances: cause unknown, never attributed to a
door or defrost). Each stretch is accepted or rejected with a reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from .. import thresholds as C
from ..schemas import DataQualityReport, EventType, SegmentationResult
from . import limits as L
from .schemas import ForecastRequest, Regime, RegimeBasis, SegmentRecord

PASSIVE_RECORDED = "PASSIVE_RECORDED_OFF"
PASSIVE_DECLARED = "PASSIVE_DECLARED"
PASSIVE_EVENT = "PASSIVE_RECORDED_EVENT"      # inside a power interruption from the supplied event log
ACTIVE_RECORDED = "ACTIVE_RECORDED_ON"
ANY_REGIME = "OBSERVED_ANY_REGIME"            # continuation question: all usable stretches, whatever the regime


@dataclass
class Frame:
    """Read-only arrays of Phase 2's analysis frame."""

    ts: pd.Series
    t: np.ndarray            # minutes since the first usable row
    y: np.ndarray
    amb: np.ndarray          # NaN where missing / absent
    has_amb: bool
    comp: list               # True / False / None per row
    has_comp: bool
    lines: np.ndarray
    suspicious: set[int]     # row positions flagged TEMP_SUSPICIOUS
    gap_after: set[int]      # row g: a Phase 3 DATA_GAP lies between rows g and g+1
    med_s: float
    # ---- follow-up: operational events (empty unless an event log is supplied)
    event_excluded: dict[int, str] = field(default_factory=dict)   # row -> event kind; not fitted, not truth
    event_split_after: set[int] = field(default_factory=set)       # row g: an event boundary after row g

    @classmethod
    def build(cls, clean: pd.DataFrame, obs: pd.DataFrame | None, report: DataQualityReport,
              seg: SegmentationResult) -> Frame:
        ts = clean["ts"].reset_index(drop=True)
        t = ((ts - ts.iloc[0]).dt.total_seconds() / 60).to_numpy(dtype=float)
        lines = clean["source_line"].to_numpy()
        pos = {int(ln): i for i, ln in enumerate(lines)}
        susp: set[int] = set()
        if obs is not None and "flags" in obs:
            hit = obs["flags"].str.contains("TEMP_SUSPICIOUS", regex=False)
            susp = {pos[int(ln)] for ln in obs.loc[hit, "source_line"] if int(ln) in pos}
        gaps = {pos[e.source_lines[0]] for e in seg.of_type(EventType.DATA_GAP)}
        amb = (clean["ambient_c"].to_numpy(dtype=float) if "ambient_c" in clean else np.full(len(clean), np.nan))
        comp = ([None if pd.isna(v) else bool(v) for v in clean["compressor_on"]] if "compressor_on" in clean
                else [None] * len(clean))
        return cls(ts, t, clean["temp_c"].to_numpy(dtype=float), amb, "ambient_c" in clean, comp,
                   "compressor_on" in clean, lines, susp, gaps, float(report.median_interval_s or 0.0))


@dataclass
class Piece:
    seg_id: str
    regime: str
    rows: np.ndarray                 # consecutive usable rows (no gap, no flagged row inside)
    parent_first: int                # first row of the original stretch (for trailing windows)
    events: list[str] = field(default_factory=list)
    role: str | None = None

    def duration(self, f: Frame) -> float:
        return float(f.t[self.rows[-1]] - f.t[self.rows[0]])

    def change(self, f: Frame) -> float:
        k = min(L.SEG_END_POINTS, len(self.rows))
        y = f.y[self.rows]
        return float(np.mean(y[-k:]) - np.mean(y[:k]))

    def record(self, f: Frame, accepted: bool, reason: str | None) -> SegmentRecord:
        r = self.rows
        return SegmentRecord(segment_id=self.seg_id, regime=self.regime, start=f.ts.iloc[r[0]].to_pydatetime(),
                             end=f.ts.iloc[r[-1]].to_pydatetime(), n_observations=len(r),
                             duration_min=round(self.duration(f), 4),
                             change_c=round(self.change(f), 4) if len(r) else None,
                             first_line=int(f.lines[r[0]]), last_line=int(f.lines[r[-1]]),
                             phase3_events=list(self.events), role=self.role, accepted=accepted, reason=reason)


def origin_index(f: Frame, request: ForecastRequest) -> tuple[int | None, str | None]:
    if request.origin is None:
        return len(f.t) - 1, None
    target = pd.Timestamp(request.origin)          # naive, in the log's time basis (checked by the engine)
    ts = f.ts
    i = int(ts.searchsorted(target, side="right")) - 1
    if i < 0:
        return None, "origin is before the first usable observation"
    lag_s = (target - ts.iloc[i]).total_seconds()
    if lag_s > C.GAP_FACTOR * f.med_s:                # the validation gap rule
        return None, (f"no usable observation within {C.GAP_FACTOR:g} median intervals before the "
                      f"origin (last one {lag_s / 60:.1f} min earlier)")
    return i, None


def regime_at_origin(f: Frame, seg: SegmentationResult, i0: int,
                     request: ForecastRequest) -> tuple[Regime, RegimeBasis, list[str], str | None]:
    """(regime, basis, evidence, contradiction)."""
    comp_ok = f.has_comp and str(seg.summary.get("compressor", "")).startswith("segmented")
    if request.passive_since is not None:
        since = pd.Timestamp(request.passive_since)
        inside = [i for i in range(i0 + 1) if f.ts.iloc[i] >= since]
        on = [i for i in inside if f.comp[i] is True]
        if on:
            return (Regime.UNKNOWN, RegimeBasis.DECLARED_BY_USER, [],
                    f"passive_since declares no active cooling from {since}, but the compressor is recorded ON at "
                    f"{len(on)} reading(s) in that period (first at line {int(f.lines[on[0]])})")
        if not inside:
            return (Regime.UNKNOWN, RegimeBasis.DECLARED_BY_USER, [],
                    "passive_since is after the origin observation")
        return (Regime.PASSIVE, RegimeBasis.DECLARED_BY_USER,
                [f"declared by the user: no active cooling since {since} (ASSUMED, not measured); "
                 f"{len(inside)} usable reading(s) in that period up to the origin"], None)
    if comp_ok:
        state = f.comp[i0]
        if state is None:
            return Regime.UNKNOWN, RegimeBasis.RECORDED_COMPRESSOR_STATE, [
                "compressor state is missing at the origin reading"], None
        j = i0
        while j > 0 and f.comp[j - 1] is state and (j - 1) not in f.gap_after:
            j -= 1
        label = "ON" if state else "OFF"
        return (Regime.ACTIVE if state else Regime.PASSIVE, RegimeBasis.RECORDED_COMPRESSOR_STATE,
                [f"compressor recorded {label} at the origin, continuously for {f.t[i0] - f.t[j]:.1f} min "
                 f"({i0 - j + 1} reading(s), lines {int(f.lines[j])}-{int(f.lines[i0])})"], None)
    if f.has_comp:
        return Regime.UNKNOWN, RegimeBasis.NONE, [
            f"compressor state present but not usable: {seg.summary.get('compressor', 'not segmented')}"], None
    return Regime.UNKNOWN, RegimeBasis.NONE, [
        "no compressor_state column and no declared passive period: the regime is not known"], None


def _split_rows(f: Frame, rows: list[int]) -> tuple[list[list[int]], list[int]]:
    """Split a row run at gaps and at TEMP_SUSPICIOUS rows (which are dropped); also at rows inside
    excluding events (dropped), at event boundaries, and wherever the row index jumps."""
    runs: list[list[int]] = [[]]
    dropped = []
    prev = None
    for i in rows:
        if prev is not None and i != prev + 1:
            runs.append([])
        prev = i
        if i in f.suspicious or i in f.event_excluded:
            dropped.append(i)
            runs.append([])
            continue
        runs[-1].append(i)
        if i in f.gap_after or i in f.event_split_after:
            runs.append([])
    return [r for r in runs if r], dropped


# ---------------------------------------------------------------- follow-up: operational events
def apply_events(f: Frame, events: list) -> list[str]:
    """Mark readings inside disturbance intervals (never fitted, never used as truth) and segment
    boundaries at point events. Power interruptions are handled as regime evidence instead.
    Returns notes for the result; nothing is attributed beyond what each record states."""
    from .inputs import EXCLUDING, EventKind

    ts = f.ts
    notes, counts = [], {}
    for e in events:
        if e.kind == EventKind.POWER_INTERRUPTION:
            continue
        start = pd.Timestamp(e.start)
        end = pd.Timestamp(e.end) if e.end is not None else None
        if (end or start) < ts.iloc[0] or start > ts.iloc[-1]:
            counts.setdefault(e.kind.value, [0, 0, 0])[2] += 1
            continue
        c = counts.setdefault(e.kind.value, [0, 0, 0])
        c[0] += 1
        if e.kind in EXCLUDING and end is not None:
            rows = np.flatnonzero(((ts >= start) & (ts <= end)).to_numpy())
            for r in rows:
                f.event_excluded.setdefault(int(r), e.kind.value)
            c[1] += len(rows)
            if not len(rows):                         # the interval falls between two readings: a boundary
                j = int(ts.searchsorted(start)) - 1
                if 0 <= j < len(ts) - 1:
                    f.event_split_after.add(j)
        else:
            j = int(ts.searchsorted(start)) - 1      # last reading strictly before the event
            if 0 <= j < len(ts) - 1:
                f.event_split_after.add(j)
    for kind, (n, rows, outside) in sorted(counts.items()):
        what = ("readings inside them are not fitted and not used as held-out truth" if rows
                else "used as segment boundaries")
        notes.append(f"{n} {kind} record(s) within the log: {what} ({rows} reading(s) excluded)"
                     + (f"; {outside} record(s) outside the log ignored" if outside else ""))
    return notes


def power_windows(f: Frame, events: list) -> list[tuple[datetime, datetime | None, list[int]]]:
    """Recorded power interruptions: (start, end or None if ongoing, usable rows inside)."""
    from .inputs import EventKind

    out = []
    for e in events:
        if e.kind != EventKind.POWER_INTERRUPTION:
            continue
        start = pd.Timestamp(e.start)
        inside = f.ts >= start
        if e.end is not None:
            inside &= f.ts <= pd.Timestamp(e.end)
        out.append((e.start, e.end, [int(i) for i in np.flatnonzero(inside.to_numpy())]))
    return out


def regime_from_power(f: Frame, i0: int, windows: list) -> tuple[Regime | None, list[str], str | None]:
    """(PASSIVE if the origin lies in a recorded power interruption else None, evidence, conflict)."""
    for start, _end, rows in windows:
        on = [i for i in rows if i <= i0 and f.comp[i] is True]
        if on:
            return None, [], (f"the event log records a power interruption from {start}, but the compressor is "
                               f"recorded ON at {len(on)} reading(s) inside it (first at line {int(f.lines[on[0]])})")
    for start, end, rows in windows:
        if i0 in rows:
            n = sum(1 for i in rows if i <= i0)
            ongoing = end is None or pd.Timestamp(end) >= f.ts.iloc[i0]
            return Regime.PASSIVE, [f"power interruption recorded in the event log from {start} "
                                    f"({'ongoing at the origin' if ongoing else f'to {end}'}); "
                                    f"{n} usable reading(s) in it up to the origin"], None
    return None, [], None


def candidate_pieces(f: Frame, seg: SegmentationResult, i0: int, request: ForecastRequest,
                     basis: RegimeBasis, power: list | None = None) -> tuple[list[Piece], list[Piece], list[int]]:
    """(passive pieces, active pieces, dropped suspicious rows); all end at the origin at the latest.
    Recorded power interruptions (``power``) add passive stretches unless a period is declared."""
    passive_runs: list[tuple[list[int], list[str], str]] = []
    active_runs: list[tuple[list[int], list[str], str]] = []
    pos = {int(ln): i for i, ln in enumerate(f.lines)}
    in_power: set[int] = set()
    if power and basis != RegimeBasis.DECLARED_BY_USER:
        for _, _, rows in power:
            in_power |= {i for i in rows if i <= i0}
        for run in np.split(np.array(sorted(in_power)), np.flatnonzero(np.diff(sorted(in_power)) != 1) + 1):
            if len(run):
                passive_runs.append(([int(i) for i in run], [], PASSIVE_EVENT))
    if basis == RegimeBasis.DECLARED_BY_USER:
        since = pd.Timestamp(request.passive_since)
        rows = [i for i in range(i0 + 1) if f.ts.iloc[i] >= since]
        passive_runs.append((rows, [], PASSIVE_DECLARED))
    if f.has_comp and str(seg.summary.get("compressor", "")).startswith("segmented"):
        for e in seg.of_type(EventType.COMPRESSOR_OFF, EventType.COMPRESSOR_ON):
            rows = [pos[ln] for ln in e.source_lines if pos[ln] <= i0]
            if not rows:
                continue
            if e.type == EventType.COMPRESSOR_ON:
                active_runs.append((rows, [e.event_id], ACTIVE_RECORDED))
            elif basis != RegimeBasis.DECLARED_BY_USER:
                rows = [i for i in rows if i not in in_power]          # those are in a power interruption run
                if rows:
                    passive_runs.append((rows, [e.event_id], PASSIVE_RECORDED))
    dropped: list[int] = []
    out: list[list[Piece]] = [[], []]
    n = 0
    for k, runs in enumerate((passive_runs, active_runs)):
        for rows, ev, regime in sorted(runs, key=lambda r: r[0][0]):
            parts, drop = _split_rows(f, rows)
            dropped += drop
            for p in parts:
                n += 1
                out[k].append(Piece(f"S{n:03d}", regime, np.array(p), p[0], ev))
    return out[0], out[1], sorted(set(dropped))


def check_piece(f: Frame, p: Piece) -> str | None:
    """Reason to reject a piece for fitting, or None."""
    if len(p.rows) < L.SEG_MIN_POINTS:
        return f"{len(p.rows)} reading(s); need {L.SEG_MIN_POINTS}"
    if p.duration(f) < L.SEG_MIN_DURATION_MIN:
        return f"lasts {p.duration(f):.1f} min; need {L.SEG_MIN_DURATION_MIN:g}"
    if abs(p.change(f)) < L.SEG_MIN_CHANGE_C:
        return (f"temperature changes {p.change(f):+.2f} °C; need at least {L.SEG_MIN_CHANGE_C} °C to carry "
                "information on the rate")
    return None


def chronological_cuts(f: Frame, passive: list[Piece]) -> tuple[float, float] | None:
    """Times (minutes) at 60 % and 80 % of the accepted passive observation time."""
    durs = [p.duration(f) for p in passive]
    total = sum(durs)
    if total <= 0:
        return None
    cuts = []
    for frac in (L.SPLIT_FIT_FRACTION, L.SPLIT_FIT_FRACTION + L.SPLIT_SELECT_FRACTION):
        target, acc = frac * total, 0.0
        for p, d in zip(passive, durs, strict=True):
            if acc + d >= target:
                cuts.append(float(f.t[p.rows[0]] + (target - acc)))
                break
            acc += d
    return cuts[0], cuts[1]


def assign_roles(f: Frame, pieces: list[Piece], cuts: tuple[float, float]) -> list[Piece]:
    """Split pieces at the cut times; rows up to cut1 FIT, up to cut2 SELECT, after TEST."""
    out = []
    for p in pieces:
        t = f.t[p.rows]
        role = np.where(t <= cuts[0], "FIT", np.where(t <= cuts[1], "SELECT", "TEST"))
        start = 0
        for j in range(1, len(role) + 1):
            if j == len(role) or role[j] != role[start]:
                sub = Piece(p.seg_id + ("" if start == 0 and j == len(role) else f".{role[start][0]}"),
                            p.regime, p.rows[start:j], p.parent_first, p.events, str(role[start]))
                out.append(sub)
                start = j
    return out


def all_pieces(f: Frame, i0: int) -> tuple[list[Piece], list[int]]:
    """Continuation question: every usable stretch up to the origin, whatever the regime."""
    parts, dropped = _split_rows(f, list(range(i0 + 1)))
    return [Piece(f"S{n:03d}", ANY_REGIME, np.array(p), p[0]) for n, p in enumerate(parts, 1)], dropped
