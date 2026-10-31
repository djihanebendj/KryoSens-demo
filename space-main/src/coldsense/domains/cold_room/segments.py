"""Find candidate compressor-off (warming) segments from air temperature alone.

A warming segment runs from a local minimum to the next local maximum of the
smoothed signal. It is a *hypothesis* about compressor state: door openings,
defrost cycles and sensor placement can all produce rises too, which is why
segments are filtered and the filters are reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import thresholds as C
from .schemas import WarmingSegment


def _r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3 or np.ptp(y) == 0:
        return 0.0, 0.0
    slope, icpt = np.polyfit(x, y, 1)
    resid = y - (slope * x + icpt)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(slope), float(1 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else 0.0


def detect_warming_segments(df: pd.DataFrame, median_interval_s: float) -> list[WarmingSegment]:
    if len(df) < 5:
        return []
    win = max(3, int(round(C.SMOOTH_WINDOW_MIN * 60 / median_interval_s)) | 1)
    s = df["temp_c"].rolling(win, center=True, min_periods=1).median().to_numpy()
    ts = df["ts"].to_numpy()
    t_min = (df["ts"] - df["ts"].iloc[0]).dt.total_seconds().to_numpy() / 60
    gap_after = np.r_[np.diff(t_min) * 60 > C.GAP_FACTOR * median_interval_s, False]

    # direction of the smoothed signal, ignoring flat steps
    d = np.sign(np.diff(s))
    for i in range(1, len(d)):  # carry previous direction through flats
        if d[i] == 0:
            d[i] = d[i - 1]
    turns = np.where(np.diff(d) != 0)[0] + 1  # indices where direction changes
    bounds = np.r_[0, turns, len(s) - 1]

    segs: list[WarmingSegment] = []
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        if b - a < 2 or s[b] <= s[a]:
            continue  # not a rise
        x, y = t_min[a:b + 1], df["temp_c"].to_numpy()[a:b + 1]
        slope, r2 = _r2(x, y)
        dur = float(x[-1] - x[0])
        rise = float(s[b] - s[a])
        reason = None
        if gap_after[a:b].any():
            reason = "crosses a data gap"
        elif dur < C.SEG_MIN_DURATION_MIN:
            reason = f"shorter than {C.SEG_MIN_DURATION_MIN:.0f} min"
        elif rise < C.SEG_MIN_RISE_C:
            reason = f"rise under {C.SEG_MIN_RISE_C} °C"
        elif slope > C.SEG_MAX_RATE_C_PER_MIN:
            reason = "too fast for an off-cycle (door or defrost?)"
        elif r2 < C.SEG_MIN_R2:
            reason = f"irregular rise (R² {r2:.2f} < {C.SEG_MIN_R2})"
        if dur < 2.0 and reason:
            continue  # don't clutter the report with micro-wiggles
        segs.append(WarmingSegment(
            start=pd.Timestamp(ts[a]).to_pydatetime(), end=pd.Timestamp(ts[b]).to_pydatetime(),
            duration_min=round(dur, 2), t_start_c=round(float(s[a]), 3), t_end_c=round(float(s[b]), 3),
            rate_c_per_min=round(slope, 5), r2_linear=round(r2, 4),
            clean=reason is None, reject_reason=reason,
        ))
    return segs
