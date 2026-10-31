"""Synthetic cold-room logs. Output is ALWAYS labelled SIMULATED.

Used for tests, for the demo sample, and for checking that detection and
refusal logic behave. Never used as evidence that a model works on a real room.

Model: lumped first-order room air,
    off:  dT/dt = (T_amb - T) / tau
    on:   dT/dt = (T_amb - T) / tau - cool_rate
with an on/off thermostat between ``setpoint_lo`` and ``setpoint_hi``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RoomParams:
    tau_min: float = 300.0       # thermal time constant of room air + load
    ambient_c: float = 25.0
    cool_rate: float = 0.25      # °C/min of refrigeration effect
    setpoint_lo: float = 2.0
    setpoint_hi: float = 5.0
    noise_c: float = 0.05
    resolution_c: float = 0.1    # logger quantisation
    door_events_per_day: float = 0.0
    door_jump_c: float = 2.0


def simulate_log(hours: float = 24.0, interval_s: int = 60, p: RoomParams | None = None,
                 seed: int = 0, start: str = "2026-10-20 00:00:00",
                 outage_at_h: float | None = None, outage_h: float = 0.0) -> pd.DataFrame:
    """Return ts, temp_c, ambient_c, compressor_on (ground truth, for tests only)."""
    p = p or RoomParams()
    rng = np.random.default_rng(seed)
    dt = interval_s / 60.0
    n = int(hours * 3600 / interval_s) + 1
    T = np.empty(n)
    on = np.zeros(n, dtype=bool)
    t, comp = p.setpoint_hi - 0.5, True
    door_p = p.door_events_per_day * dt / (24 * 60)
    for i in range(n):
        h = i * dt / 60
        in_outage = outage_at_h is not None and outage_at_h <= h < outage_at_h + outage_h
        if not in_outage:
            if t >= p.setpoint_hi:
                comp = True
            elif t <= p.setpoint_lo:
                comp = False
        powered = comp and not in_outage
        on[i] = powered
        T[i] = t
        dTdt = (p.ambient_c - t) / p.tau_min - (p.cool_rate if powered else 0.0)
        t = t + dTdt * dt
        if rng.random() < door_p:
            t += p.door_jump_c
    meas = T + rng.normal(0, p.noise_c, n)
    if p.resolution_c:
        meas = np.round(meas / p.resolution_c) * p.resolution_c
    ts = pd.date_range(start, periods=n, freq=f"{interval_s}s")
    df = pd.DataFrame({"ts": ts, "temp_c": meas, "ambient_c": p.ambient_c, "compressor_on": on})
    df.attrs["coldsense_evidence"] = "SIMULATED"  # provenance marker: may never be declared REAL
    return df


def true_minutes_to_threshold(t0: float, threshold: float, p: RoomParams) -> float:
    """Analytic ground truth for the off-state model (no noise)."""
    if t0 >= threshold:
        return 0.0
    if threshold >= p.ambient_c:
        return float("inf")
    return float(p.tau_min * np.log((p.ambient_c - t0) / (p.ambient_c - threshold)))
