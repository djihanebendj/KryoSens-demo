"""Transparent baselines every physical model must be compared with.

Persistence:     T_hat(t0 + h) = T_observed(t0)
    Meaningful when the temperature changes little over the horizon (a stable
    room, very short horizons). It never predicts a crossing that has not
    already happened. Horizon: any; error grows with the true change.

Recent slope:    T_hat(t0 + h) = T_observed(t0) + s * h
    s is the ordinary least-squares slope of temperature against ACTUAL elapsed
    time over the trailing window (t0 - W, t0], W = ``SLOPE_WINDOW_MIN``, using
    only usable rows of the same gap-free stretch as t0 (never across a gap,
    never after t0). Irregular sampling is handled by regressing on timestamps,
    not on row numbers. Noise sensitivity is reported as the slope's standard
    error; stability as the slopes of the window's first and second halves.
    The forecast is anchored at the observed temperature at t0 so it starts
    from the same point as persistence and the physical model.

    Refused when the window holds fewer than ``SLOPE_MIN_POINTS`` readings or
    they span less than half of it. Unsuitable for long horizons: a constant
    slope ignores that warming slows as the room approaches its surroundings,
    and it is not a physical model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import limits as L
from .models import Law

PERSISTENCE = "persistence"
RECENT_SLOPE = "recent_slope"
BASELINE_VERSION = "1"


@dataclass(frozen=True)
class SlopeFit:
    slope: float            # °C/min
    se: float | None        # standard error of the slope
    n: int
    span_min: float
    first_half_slope: float | None
    second_half_slope: float | None


def _ols(t: np.ndarray, y: np.ndarray) -> tuple[float, float | None]:
    tc = t - t.mean()
    sxx = float(tc @ tc)
    slope = float(tc @ (y - y.mean())) / sxx
    if len(t) < 3:
        return slope, None
    resid = y - y.mean() - slope * tc
    return slope, float(np.sqrt(float(resid @ resid) / (len(t) - 2) / sxx))


def recent_slope(t: np.ndarray, y: np.ndarray, i0: int, first: int,
                 window: float = L.SLOPE_WINDOW_MIN) -> tuple[SlopeFit | None, str | None]:
    """Slope over (t[i0] - window, t[i0]] using rows first..i0 only (t in minutes)."""
    sel = np.arange(first, i0 + 1)
    sel = sel[t[sel] > t[i0] - window]
    if len(sel) < L.SLOPE_MIN_POINTS:
        return None, f"{len(sel)} reading(s) in the {window:g}-min window before the origin; need {L.SLOPE_MIN_POINTS}"
    tt, yy = t[sel], y[sel]
    span = float(tt[-1] - tt[0])
    if span < L.SLOPE_MIN_SPAN_FRACTION * window:
        return None, f"readings span {span:.1f} min of the {window:g}-min window; need at least half"
    slope, se = _ols(tt, yy)
    mid = tt[0] + span / 2
    halves = []
    for part in (tt <= mid, tt > mid):
        halves.append(_ols(tt[part], yy[part])[0] if part.sum() >= 3 and np.ptp(tt[part]) > 0 else None)
    return SlopeFit(slope, se, len(sel), span, halves[0], halves[1]), None


def persistence_law(t0: float) -> Law:
    return Law("constant", t0)


def slope_law(t0: float, fit: SlopeFit) -> Law:
    return Law("linear", t0, slope=fit.slope)
