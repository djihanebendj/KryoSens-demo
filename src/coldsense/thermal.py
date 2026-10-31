"""Step 4: First-Order Thermal Drift Model, Uncertainty Quantification & Reliability Gate.

Physics Model (Newton's Law of Cooling / First-Order RC Thermal Drift):
    T(t) = T_amb - (T_amb - T_0) * exp(-(t - t_0) / tau)

Rules That Never Bend:
- Reliability Gate evaluates sample count, data gaps, physical monotonicity, and R^2 fit quality.
- When the Reliability Gate fails, the system returns PredictionRefusal (never a guessed curve).
- Compares the first-order thermal model against a Naive Baseline (linear initial-slope extrapolation).
"""

from __future__ import annotations

import math
import statistics
from typing import List, Optional, Tuple

from coldsense.models import (
    CaseRecord,
    EvidenceClass,
    EventType,
    ForecastPoint,
    PredictionOutcome,
    PredictionRefusal,
    PredictionSuccess,
    RefusalReasonCode,
    TemperatureEvent,
    TemperaturePoint,
    WorkflowStep,
)

MIN_DRIFT_SAMPLES = 8
MAX_ALLOWED_GAP_RATIO = 0.30
MIN_R_SQUARED = 0.85


def _select_primary_drift_event(case: CaseRecord, event_id: Optional[str] = None) -> Optional[TemperatureEvent]:
    if event_id:
        for ev in case.events:
            if ev.event_id == event_id:
                return ev
        return None
    for ev in case.events:
        if ev.event_type == EventType.POWER_CUT:
            return ev
    return case.events[0] if case.events else None


def _fit_first_order_tau(
    elapsed_sec: List[float],
    temps_c: List[float],
    t0_c: float,
    ambient_c: float,
) -> float:
    """Fit the single thermal time constant tau (in seconds) via log-linear initialization
    followed by Gauss-Newton non-linear least-squares refinement.
    """
    # Step 1: Analytical log-linear estimate on valid points
    # ln((T_amb - T(t)) / (T_amb - T0)) = -t / tau
    num = 0.0
    den = 0.0
    delta0 = ambient_c - t0_c
    if abs(delta0) < 0.1:
        return 14400.0

    for t, y in zip(elapsed_sec, temps_c):
        if t <= 0:
            continue
        ratio = (ambient_c - y) / delta0
        if 0.01 < ratio < 0.9999:
            ln_r = math.log(ratio)
            num += -t * ln_r
            den += ln_r * ln_r

    tau = (num / den) if den > 1e-9 else 14400.0
    tau = max(300.0, min(tau, 86400.0 * 3))

    # Step 2: Gauss-Newton refinement minimizing sum((f(t; tau) - y)^2)
    for _ in range(35):
        grad = 0.0
        hess = 0.0
        for t, y in zip(elapsed_sec, temps_c):
            if t <= 0:
                continue
            exp_term = math.exp(-t / tau)
            pred = ambient_c - delta0 * exp_term
            err = pred - y
            # d(pred)/d(tau) = -delta0 * exp(-t/tau) * (t / tau^2)
            j = -delta0 * exp_term * (t / (tau * tau))
            grad += j * err
            hess += j * j
        if hess < 1e-15:
            break
        step = grad / hess
        tau_new = max(300.0, min(tau - step, 86400.0 * 5))
        if abs(tau_new - tau) < 1e-3:
            tau = tau_new
            break
        tau = tau_new

    return tau


def run_thermal_prediction(
    case: CaseRecord,
    event_id: Optional[str] = None,
    horizon_sec: float = 3600.0,
) -> PredictionOutcome:
    """Step 4: Run the Reliability Gate and either produce a First-Order Thermal Forecast
    with uncertainty + Naive Baseline comparison, or return a typed PredictionRefusal.
    """
    if not case.is_step_unlocked(WorkflowStep.PREDICTION):
        raise ValueError("Step 4 (Prediction) is locked until Step 3 (Event Analysis) completes.")

    target_event = _select_primary_drift_event(case, event_id=event_id)
    if target_event is None:
        refusal = PredictionRefusal(
            response_type="REFUSAL",
            case_id=case.case_id,
            event_id=event_id,
            evidence_class=case.evidence_class,
            reason_code=RefusalReasonCode.NO_DRIFT_EVENT_FOUND,
            refusal_explanation=(
                "Reliability Gate refused prediction: no thermal drift or power-cut event window "
                "was detected in the temperature log."
            ),
            failed_checks=["NO_DRIFT_EVENT_WINDOW"],
            diagnostics_metrics={"event_count": float(len(case.events))},
        )
        case.prediction = refusal
        if case.unlocked_step < int(WorkflowStep.EVIDENCE):
            case.unlocked_step = int(WorkflowStep.EVIDENCE)
        return refusal

    segment: List[TemperaturePoint] = case.points[
        target_event.start_index : target_event.end_index + 1
    ]
    n_samples = len(segment)
    failed_checks: List[str] = []

    # Check 1: Minimum sample count in the drift window
    if n_samples < MIN_DRIFT_SAMPLES:
        failed_checks.append(
            f"Sample count n={n_samples} is below the minimum required ({MIN_DRIFT_SAMPLES}) for reliable parameter estimation."
        )

    # Check 2: Data gap ratio inside the event window
    t_start = segment[0].timestamp_sec
    t_end = segment[-1].timestamp_sec
    duration = max(1.0, t_end - t_start)
    cadence = case.data_quality.median_cadence_sec if case.data_quality else 300.0
    gap_time = 0.0
    for i in range(1, n_samples):
        dt = segment[i].timestamp_sec - segment[i - 1].timestamp_sec
        if cadence > 0 and dt > cadence * 2.5:
            gap_time += dt
    gap_ratio = gap_time / duration

    if gap_ratio > MAX_ALLOWED_GAP_RATIO:
        failed_checks.append(
            f"Missing data gap ratio ({gap_ratio:.1%}) exceeds maximum allowed ({MAX_ALLOWED_GAP_RATIO:.0%})."
        )

    # Check 3: Physical directionality / trajectory sanity
    t0_c = segment[0].temp_c
    ambient_candidates = [p.ambient_temp_c for p in segment if p.ambient_temp_c is not None]
    ambient_c = statistics.median(ambient_candidates) if ambient_candidates else 22.0

    # In a warming drift (ambient > t0), temperature should generally move toward ambient
    net_change = segment[-1].temp_c - t0_c
    expected_sign = 1.0 if ambient_c > t0_c else -1.0
    reversals = 0
    for i in range(1, n_samples):
        step_diff = segment[i].temp_c - segment[i - 1].temp_c
        if step_diff * expected_sign < -0.35:
            reversals += 1
    reversal_ratio = reversals / max(1, n_samples - 1)

    if net_change * expected_sign <= 0.2 or reversal_ratio > 0.25:
        failed_checks.append(
            f"Non-physical trajectory: net temperature change ({net_change:+.2f} C) or step reversal ratio ({reversal_ratio:.1%}) violates first-order passive drift physics."
        )

    if failed_checks:
        primary_code = (
            RefusalReasonCode.INSUFFICIENT_SAMPLES
            if n_samples < MIN_DRIFT_SAMPLES
            else (
                RefusalReasonCode.EXCESSIVE_DATA_GAPS
                if gap_ratio > MAX_ALLOWED_GAP_RATIO
                else RefusalReasonCode.NON_PHYSICAL_TRAJECTORY
            )
        )
        refusal = PredictionRefusal(
            response_type="REFUSAL",
            case_id=case.case_id,
            event_id=target_event.event_id,
            evidence_class=case.evidence_class,
            reason_code=primary_code,
            refusal_explanation="Reliability Gate refused forecast: " + " ".join(failed_checks),
            failed_checks=failed_checks,
            diagnostics_metrics={
                "sample_count": float(n_samples),
                "gap_ratio": round(gap_ratio, 4),
                "net_temp_change_c": round(net_change, 3),
                "reversal_ratio": round(reversal_ratio, 4),
            },
        )
        case.prediction = refusal
        if case.unlocked_step < int(WorkflowStep.EVIDENCE):
            case.unlocked_step = int(WorkflowStep.EVIDENCE)
        return refusal

    # Fit first-order model on calibration window (first 60% of event) and evaluate across full event
    elapsed = [p.timestamp_sec - t_start for p in segment]
    temps = [p.temp_c for p in segment]
    split_idx = max(4, int(n_samples * 0.6))
    tau_sec = _fit_first_order_tau(elapsed[:split_idx], temps[:split_idx], t0_c, ambient_c)

    # Naive Baseline: linear extrapolation from initial 25% of drift window
    init_idx = max(2, int(n_samples * 0.25))
    naive_slope = (temps[init_idx] - t0_c) / max(1.0, elapsed[init_idx])

    model_preds = [
        ambient_c - (ambient_c - t0_c) * math.exp(-t / tau_sec) for t in elapsed
    ]
    naive_preds = [t0_c + naive_slope * t for t in elapsed]

    residuals = [y - yhat for y, yhat in zip(temps, model_preds)]
    naive_errors = [y - nhat for y, nhat in zip(temps, naive_preds)]

    ss_res = sum(r * r for r in residuals)
    mean_y = statistics.mean(temps)
    ss_tot = sum((y - mean_y) ** 2 for y in temps)
    r_squared = 1.0 - (ss_res / max(ss_tot, 1e-9))

    if r_squared < MIN_R_SQUARED:
        refusal = PredictionRefusal(
            response_type="REFUSAL",
            case_id=case.case_id,
            event_id=target_event.event_id,
            evidence_class=case.evidence_class,
            reason_code=RefusalReasonCode.POOR_MODEL_FIT,
            refusal_explanation=(
                f"Reliability Gate refused forecast: first-order model goodness-of-fit "
                f"R^2={r_squared:.3f} is below required threshold ({MIN_R_SQUARED:.2f})."
            ),
            failed_checks=[f"R^2 ({r_squared:.3f}) < {MIN_R_SQUARED:.2f}"],
            diagnostics_metrics={
                "r_squared": round(r_squared, 4),
                "sample_count": float(n_samples),
            },
        )
        case.prediction = refusal
        if case.unlocked_step < int(WorkflowStep.EVIDENCE):
            case.unlocked_step = int(WorkflowStep.EVIDENCE)
        return refusal

    model_rmse = math.sqrt(ss_res / n_samples)
    naive_rmse = math.sqrt(sum(e * e for e in naive_errors) / n_samples)
    model_mae = sum(abs(r) for r in residuals) / n_samples
    naive_mae = sum(abs(e) for e in naive_errors) / n_samples

    # Uncertainty quantification: residual standard error + parameter growth with horizon
    res_std = statistics.stdev(residuals) if len(residuals) > 1 else 0.1
    base_sigma = max(res_std, 0.08)

    forecast_curve: List[ForecastPoint] = []
    for t, obs_y, m_y, n_y in zip(elapsed, temps, model_preds, naive_preds):
        horizon_factor = 1.0 + 0.45 * (t / max(duration, 1.0))
        sigma_t = base_sigma * horizon_factor
        forecast_curve.append(
            ForecastPoint(
                timestamp_sec=t_start + t,
                elapsed_sec=t,
                observed_temp_c=round(obs_y, 3),
                thermal_model_temp_c=round(m_y, 3),
                lower_95_c=round(m_y - 1.96 * sigma_t, 3),
                upper_95_c=round(m_y + 1.96 * sigma_t, 3),
                lower_50_c=round(m_y - 0.674 * sigma_t, 3),
                upper_50_c=round(m_y + 0.674 * sigma_t, 3),
                naive_baseline_temp_c=round(n_y, 3),
                evidence_class=EvidenceClass.EXPECTED,
            )
        )

    # Extend future forecast horizon beyond observed event window
    step_sec = cadence if cadence > 0 else 300.0
    extra_steps = int(round(horizon_sec / step_sec))
    for k in range(1, extra_steps + 1):
        t = elapsed[-1] + k * step_sec
        m_y = ambient_c - (ambient_c - t0_c) * math.exp(-t / tau_sec)
        n_y = t0_c + naive_slope * t
        horizon_factor = 1.0 + 0.45 * (t / max(duration, 1.0))
        sigma_t = base_sigma * horizon_factor
        forecast_curve.append(
            ForecastPoint(
                timestamp_sec=t_start + t,
                elapsed_sec=t,
                observed_temp_c=None,
                thermal_model_temp_c=round(m_y, 3),
                lower_95_c=round(m_y - 1.96 * sigma_t, 3),
                upper_95_c=round(m_y + 1.96 * sigma_t, 3),
                lower_50_c=round(m_y - 0.674 * sigma_t, 3),
                upper_50_c=round(m_y + 0.674 * sigma_t, 3),
                naive_baseline_temp_c=round(n_y, 3),
                evidence_class=EvidenceClass.EXPECTED,
            )
        )

    # Calculate time from event start until critical threshold breach
    thresh = case.critical_threshold_c
    time_to_thresh: Optional[float] = None
    if t0_c < thresh < ambient_c:
        ratio = (ambient_c - thresh) / (ambient_c - t0_c)
        if 0 < ratio < 1:
            time_to_thresh = round(-tau_sec * math.log(ratio), 1)

    halfwidth_95 = round(1.96 * base_sigma * 1.45, 3)
    success = PredictionSuccess(
        response_type="SUCCESS",
        case_id=case.case_id,
        event_id=target_event.event_id,
        evidence_class=case.evidence_class,
        baseline_evidence_class=EvidenceClass.EXPECTED,
        initial_temp_c=round(t0_c, 3),
        ambient_temp_c=round(ambient_c, 3),
        time_constant_tau_sec=round(tau_sec, 1),
        time_constant_tau_hours=round(tau_sec / 3600.0, 2),
        r_squared=round(r_squared, 4),
        thermal_model_rmse_c=round(model_rmse, 4),
        naive_baseline_rmse_c=round(naive_rmse, 4),
        thermal_model_mae_c=round(model_mae, 4),
        naive_baseline_mae_c=round(naive_mae, 4),
        critical_threshold_c=round(thresh, 2),
        estimated_time_to_threshold_sec=time_to_thresh,
        uncertainty_95_halfwidth_c=halfwidth_95,
        reliability_gate_passed=True,
        reliability_notes=[
            f"Reliability Gate passed: n={n_samples} samples, gap_ratio={gap_ratio:.1%}, R^2={r_squared:.4f}.",
            f"First-order thermal model reduced RMSE from {naive_rmse:.3f} C (naive baseline) to {model_rmse:.3f} C.",
        ],
        forecast_curve=forecast_curve,
    )
    case.prediction = success
    if case.unlocked_step < int(WorkflowStep.EVIDENCE):
        case.unlocked_step = int(WorkflowStep.EVIDENCE)
    return success
