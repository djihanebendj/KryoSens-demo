"""Held-out forecast evaluation with identical origins and horizons for every method.

Cases: in each held-out piece, an origin at the first reading at or after every
``stride`` minutes from the piece start; for each horizon h the truth is the
observed temperature at origin + h, linearly interpolated between the two
bracketing readings of the same piece (DERIVED; never across a gap or past the
piece). A method may decline a case (e.g. too few readings for a slope); its
coverage and refusal rate are reported, and head-to-head metrics use only the
cases where every compared method forecast, so no method looks better by
skipping hard cases.

Observed threshold crossings (when a threshold is set) are interpolated the
same way; a case whose origin is already beyond the threshold is not a
crossing case.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from . import limits as L
from .models import Law
from .schemas import Comparison, CrossingStatus, Direction, HorizonMetrics
from .selection import Frame, Piece

Predictor = Callable[[Piece, int], tuple[Law | None, str | None]]


@dataclass
class Case:
    piece: Piece
    i0: int
    horizon: float
    truth: float
    obs_cross: float | None = None           # minutes after origin; None if no crossing within horizon
    already: bool = False                    # origin already at/beyond the threshold
    preds: dict[str, float | None] = field(default_factory=dict)
    cross: dict[str, tuple[CrossingStatus, float | None] | None] = field(default_factory=dict)


def _beyond(v: float, th: float, d: Direction) -> bool:
    return v >= th if d == Direction.UPWARD else v <= th


def build_cases(f: Frame, pieces: list[Piece], horizons: list[float], stride: float,
                threshold: float | None, direction: Direction) -> list[Case]:
    cases = []
    for p in pieces:
        tp, yp = f.t[p.rows], f.y[p.rows]
        marks = np.arange(tp[0], tp[-1], stride)
        origins = sorted({int(np.searchsorted(tp, m - 1e-9)) for m in marks})
        for j in origins:
            for h in horizons:
                target = tp[j] + h
                if target > tp[-1] + 1e-9:
                    continue
                c = Case(p, int(p.rows[j]), h, float(np.interp(target, tp, yp)))
                if threshold is not None:
                    c.already = _beyond(yp[j], threshold, direction)
                    if not c.already:
                        for m in range(j + 1, len(tp)):
                            if tp[m] > target + 1e-9:
                                break
                            if _beyond(yp[m], threshold, direction):
                                frac = (threshold - yp[m - 1]) / (yp[m] - yp[m - 1])
                                c.obs_cross = float(tp[m - 1] + frac * (tp[m] - tp[m - 1]) - tp[j])
                                break
                cases.append(c)
    return cases


def run(cases: list[Case], methods: dict[str, Predictor], threshold: float | None, direction: Direction) -> None:
    """Fill each case's predictions in place; a law is computed once per (piece, origin)."""
    cache: dict[tuple[str, str, int], tuple[Law | None, str | None]] = {}
    for c in cases:
        for name, pred in methods.items():
            key = (name, c.piece.seg_id, c.i0)
            if key not in cache:
                try:
                    cache[key] = pred(c.piece, c.i0)
                except (ValueError, FloatingPointError, OverflowError) as e:   # a method failure is a refusal
                    cache[key] = (None, f"numerical failure: {e}")
            law = cache[key][0]
            val = None if law is None else float(law.at(c.horizon))
            c.preds[name] = val if val is not None and math.isfinite(val) else None
            if threshold is not None and law is not None and not c.already:
                c.cross[name] = law.crossing(threshold, direction, c.horizon)
            else:
                c.cross[name] = None


def _mae_rmse(err: list[float]) -> tuple[float | None, float | None]:
    if not err:
        return None, None
    e = np.abs(np.array(err))
    return float(e.mean()), float(np.sqrt((e ** 2).mean()))


def horizon_metrics(cases: list[Case], method: str, horizon: float, compared: list[str]) -> HorizonMetrics:
    hc = [c for c in cases if c.horizon == horizon]
    own = [c for c in hc if c.preds.get(method) is not None]
    common = [c for c in hc if all(c.preds.get(m) is not None for m in compared)]
    mae, rmse = _mae_rmse([c.preds[method] - c.truth for c in own])
    mae_c, rmse_c = _mae_rmse([c.preds[method] - c.truth for c in common]) if method in compared else (None, None)
    n = len(hc)
    out = HorizonMetrics(horizon_min=horizon, n_cases=n, n_forecasts=len(own),
                         coverage=round(len(own) / n, 4) if n else None,
                         refusal_rate=round(1 - len(own) / n, 4) if n else None,
                         mae_c=mae, rmse_c=rmse, n_common=len(common), mae_common_c=mae_c, rmse_common_c=rmse_c)
    cross_cases = [c for c in own if c.cross.get(method) is not None]
    if cross_cases:
        errs = []
        for c in cross_cases:
            status, mins = c.cross[method]
            predicted = status == CrossingStatus.CROSSING_PREDICTED
            if c.obs_cross is not None:
                out.n_observed_crossings += 1
                if predicted:
                    errs.append(abs(mins - c.obs_cross))
                else:
                    out.n_missed_crossings += 1
            elif predicted:
                out.n_false_crossings += 1
        out.crossing_mae_min = float(np.mean(errs)) if errs else None
    return out


def comparisons(cases: list[Case], role: str, physical: list[str], baselines: list[str],
                horizons: list[float]) -> list[Comparison]:
    out = []
    for pm in physical:
        for h in horizons:
            common = [c for c in cases if c.horizon == h and c.preds.get(pm) is not None
                      and all(c.preds.get(b) is not None for b in baselines)]
            cmp = Comparison(role=role, horizon_min=h, physical_method=pm, best_baseline=None, n_common=len(common),
                             physical_better=0, baseline_better=0, tie=0, neither_adequate=0)
            if common:
                maes = {b: float(np.mean([abs(c.preds[b] - c.truth) for c in common])) for b in baselines}
                cmp.best_baseline = min(maes, key=maes.get) if maes else None
            for c in common:
                ep = abs(c.preds[pm] - c.truth)
                eb = min(abs(c.preds[b] - c.truth) for b in baselines)
                if min(ep, eb) > L.ADEQUATE_ABS_ERROR_C:
                    cmp.neither_adequate += 1
                elif ep < eb - 1e-12:
                    cmp.physical_better += 1
                elif eb < ep - 1e-12:
                    cmp.baseline_better += 1
                else:
                    cmp.tie += 1
            out.append(cmp)
    return out
