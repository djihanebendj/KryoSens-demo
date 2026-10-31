"""Naive baseline: linear extrapolation of the unit's own observed warming rate.

    minutes = (threshold - T_now) / median(clean segment rate)

This is the bar every later thermal model must beat on held-out unplug tests
(kill criterion). It is deliberately simple and deliberately honest about
when it should not be used.

Known bias (stated to the user, not hidden): real off-state warming slows as
the room approaches ambient, so a constant rate measured near the setpoint
tends to *underestimate* time-to-threshold, i.e. errs early. The bias grows
with extrapolation distance.
"""

from __future__ import annotations

import numpy as np

from . import thresholds as C
from .schemas import DataQualityReport, EvidenceClass, Prediction, Verdict, WarmingSegment

METHOD = "naive_linear_v1"


def predict_baseline(report: DataQualityReport, segments: list[WarmingSegment],
                     current_temp_c: float, threshold_c: float) -> Prediction:
    ev = EvidenceClass.SIMULATED if report.evidence == EvidenceClass.SIMULATED else EvidenceClass.EXPECTED
    pred = Prediction(method=METHOD, evidence=ev, refused=True,
                      current_temp_c=current_temp_c, threshold_c=threshold_c)
    pred.assumptions = [
        "cooling is fully lost at the current moment",
        "the room warms at the median rate seen in this unit's own off-cycles",
        "door openings, loading and ambient stay as they were during the log",
        "this is air temperature at the sensor, not product-core temperature",
    ]

    if report.verdict == Verdict.REFUSED:
        pred.reasons = ["data-quality gate refused this log: "
                        + "; ".join(i.message for i in report.blocking)]
        return pred

    clean = [s for s in segments if s.clean and s.rate_c_per_min > 0]
    pred.n_segments_used = len(clean)
    if len(clean) < C.MIN_CLEAN_SEGMENTS:
        pred.reasons.append(
            f"only {len(clean)} clean warming segments found; need at least {C.MIN_CLEAN_SEGMENTS} "
            "to estimate how this unit warms")
        return pred

    rates = np.array([s.rate_c_per_min for s in clean])
    med = float(np.median(rates))
    cv = float(rates.std(ddof=1) / rates.mean()) if len(rates) > 1 else 0.0
    if cv > C.MAX_RATE_CV:
        pred.reasons.append(
            f"warming rates are inconsistent between cycles (CV {cv:.2f} > {C.MAX_RATE_CV}); "
            "the unit's behaviour changed during the log or segments are contaminated")
        return pred

    lo_obs = min(s.t_start_c for s in clean)
    hi_obs = max(s.t_end_c for s in clean)
    pred.observed_range_c = (round(lo_obs, 2), round(hi_obs, 2))
    extrap = max(0.0, threshold_c - hi_obs)
    pred.extrapolation_c = round(extrap, 2)
    if extrap > C.MAX_EXTRAPOLATION_C:
        pred.reasons.append(
            f"threshold {threshold_c} °C is {extrap:.1f} °C above the warmest off-cycle temperature "
            f"ever observed ({hi_obs:.1f} °C); a constant rate cannot be trusted that far "
            f"(limit {C.MAX_EXTRAPOLATION_C} °C). An unplug test reaching the threshold is needed.")
        return pred

    pred.refused = False
    if current_temp_c >= threshold_c:
        pred.minutes_to_threshold = pred.low_minutes = pred.high_minutes = 0.0
        pred.warnings.append("current temperature is already at or above the threshold")
        return pred

    q25, q75 = np.percentile(rates, [25, 75])
    span = threshold_c - current_temp_c
    pred.minutes_to_threshold = round(span / med, 1)
    pred.low_minutes = round(span / q75, 1)    # faster warming -> earlier
    pred.high_minutes = round(span / q25, 1)
    pred.assumptions.append(
        f"range spans the 25th–75th percentile of {len(clean)} observed warming rates; "
        "it is not a calibrated confidence interval")
    if extrap > 0:
        pred.warnings.append(
            f"threshold is {extrap:.1f} °C beyond observed off-cycle temperatures; "
            "linear extrapolation likely errs early (conservative) there")
    if current_temp_c < lo_obs - 1.0:
        pred.warnings.append("current temperature is below the range seen in off-cycles")
    if not report.ambient_available:
        pred.warnings.append("no ambient reading: a hotter day than the log would warm the room faster")
    return pred
