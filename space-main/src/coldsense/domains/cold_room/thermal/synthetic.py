"""Synthetic ground truth for the thermal engine. Output is ALWAYS labelled SIMULATED.

Independent of the fitting code on purpose: trajectories come from numerical
integration with ``scipy.integrate.solve_ivp`` (DOP853, rtol = atol = 1e-10)
of the generating equation, never from the closed-form solutions or the
exponential integrator the engine uses. Thermostat switching is located as an
integration event; jumps and regime changes are integration breakpoints.

Generating equation (time t in minutes):

    dT/dt = k(t) * (A(t) - T) - b * u(t)

    k(t)  true exchange rate (1/min); may change once (regime change)
    A(t)  true surroundings temperature (°C), constant or a given function
    u(t)  compressor state 0/1 from an on/off thermostat (setpoint_lo/hi),
          forced 0 during a power cut; b = cooling effect (°C/min)
    plus optional instantaneous jumps (unrecorded disturbances, e.g. a door)

Measurement: T + N(0, noise_c) rounded to ``resolution_c``; optional jittered
timestamps, missing cells, removed rows (gaps) and isolated spikes. What the
log *records* is chosen separately (ambient column true/absent/other sensor;
compressor column present or not), so tests can make the recorded inputs
disagree with the truth.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

START = pd.Timestamp("2026-10-20 00:00:00")


@dataclass
class SyntheticSpec:
    hours: float = 12.0
    interval_s: float = 60.0
    t0_c: float = 3.0
    k: float = 1 / 300                       # true rate, 1/min
    ambient_c: float = 25.0                  # true constant surroundings
    ambient_fn: Callable[[float], float] | None = None   # true time-varying surroundings A(t)
    record_ambient: str = "true"             # "true" | "none" | "other"
    other_ambient_c: float = 35.0            # what an unrelated "ambient" sensor reads
    b: float = 0.0                           # cooling effect while ON, °C/min
    setpoint_lo: float = 2.0
    setpoint_hi: float = 5.0
    cooling_until_min: float | None = None   # thermostat active until then (power cut after); None = never
    record_compressor: bool = False
    noise_c: float = 0.05
    resolution_c: float = 0.1
    seed: int = 0
    jitter_s: float = 0.0                    # uniform +- jitter on every timestamp
    gaps_min: list[tuple[float, float]] = field(default_factory=list)          # rows removed: (start, duration)
    missing_temp_min: list[tuple[float, float]] = field(default_factory=list)  # temperature cell empty
    missing_ambient_min: list[tuple[float, float]] = field(default_factory=list)
    spikes: list[tuple[float, float]] = field(default_factory=list)            # (t_min, +delta °C) one reading
    jumps: list[tuple[float, float]] = field(default_factory=list)             # (t_min, +delta °C) true state
    regime_change: tuple[float, float] | None = None                           # (t_min, new k)


@dataclass
class Truth:
    spec: SyntheticSpec
    t_dense: Callable[[float], float]        # noise-free T(t), t in minutes from START
    t_end: float
    u_switch_min: list[float]

    def crossing_minutes(self, origin_min: float, threshold: float, upward: bool = True,
                         until_min: float | None = None) -> float | None:
        """First time after origin when the noise-free truth crosses (root-found on the dense solution)."""
        end = min(self.t_end, until_min if until_min is not None else self.t_end)
        grid = np.linspace(origin_min, end, max(int((end - origin_min) * 4), 2))
        g = np.array([self.t_dense(x) for x in grid]) - threshold
        g = g if upward else -g
        if g[0] >= 0:
            return 0.0
        hit = np.flatnonzero(g >= 0)
        if not len(hit):
            return None
        j = hit[0]
        f = (lambda x: self.t_dense(x) - threshold) if upward else (lambda x: threshold - self.t_dense(x))
        return float(brentq(f, grid[j - 1], grid[j], xtol=1e-9) - origin_min)


def simulate(spec: SyntheticSpec) -> tuple[pd.DataFrame, Truth]:
    t_end = spec.hours * 60
    amb = spec.ambient_fn or (lambda t: spec.ambient_c)

    def k_at(t: float) -> float:
        return spec.regime_change[1] if spec.regime_change and t >= spec.regime_change[0] else spec.k

    breaks = sorted({0.0, t_end, *[j[0] for j in spec.jumps if 0 < j[0] < t_end],
                     *([spec.regime_change[0]] if spec.regime_change else []),
                     *([spec.cooling_until_min] if spec.cooling_until_min and spec.cooling_until_min < t_end else [])})
    pieces: list[tuple[float, float, object]] = []
    switches: list[float] = []
    T = spec.t0_c
    cooling_on = spec.cooling_until_min is not None and spec.b > 0
    u = 1 if cooling_on and T >= spec.setpoint_hi else 0
    for a, b in zip(breaks[:-1], breaks[1:], strict=True):
        if any(abs(j[0] - a) < 1e-12 for j in spec.jumps):
            T += sum(j[1] for j in spec.jumps if abs(j[0] - a) < 1e-12)
        t = a
        while t < b - 1e-12:
            thermostat = cooling_on and spec.cooling_until_min is not None and t < spec.cooling_until_min
            if not thermostat:
                u = 0

            def rhs(x, s, u=u):
                return [k_at(x) * (amb(x) - s[0]) - spec.b * u]
            events = []
            if thermostat:
                def ev(x, s, u=u):
                    return s[0] - (spec.setpoint_lo if u else spec.setpoint_hi)
                ev.terminal = True
                ev.direction = -1 if u else 1
                events = [ev]
            sol = solve_ivp(rhs, (t, b), [T], method="DOP853", rtol=1e-10, atol=1e-10, dense_output=True,
                            events=events or None, max_step=5.0)
            t_stop = float(sol.t[-1])
            pieces.append((t, t_stop, sol.sol))
            T = float(sol.y[0, -1])
            if thermostat and sol.status == 1:
                u = 1 - u
                switches.append(t_stop)
            t = t_stop

    def t_dense(x: float) -> float:
        for a, b, s in pieces:
            if a - 1e-9 <= x <= b + 1e-9:
                return float(s(min(max(x, a), b))[0])
        raise ValueError(f"time {x} outside the simulated range")

    def u_at(x: float) -> int:
        if not (spec.cooling_until_min is not None and spec.b > 0 and x < spec.cooling_until_min):
            return 0
        n = sum(1 for s in switches if s <= x)
        u0 = 1 if spec.t0_c >= spec.setpoint_hi else 0
        return u0 if n % 2 == 0 else 1 - u0

    rng = np.random.default_rng(spec.seed)
    step = spec.interval_s / 60
    tm = np.arange(0.0, t_end + 1e-9, step)
    if spec.jitter_s:
        tm = tm + rng.uniform(-spec.jitter_s, spec.jitter_s, len(tm)) / 60
        tm = np.clip(np.sort(tm), 0.0, t_end)
    secs = np.unique(np.round(tm * 60).astype(np.int64))       # loggers record whole seconds
    tm = secs / 60.0
    keep = np.ones(len(tm), dtype=bool)
    for a, d in spec.gaps_min:
        keep &= ~((tm >= a) & (tm < a + d))
    tm, secs = tm[keep], secs[keep]
    true_t = np.array([t_dense(x) for x in tm])
    meas = true_t + rng.normal(0, spec.noise_c, len(tm))
    for ts_, d in spec.spikes:
        meas[int(np.argmin(np.abs(tm - ts_)))] += d
    if spec.resolution_c:
        meas = np.round(meas / spec.resolution_c) * spec.resolution_c
    temp = [f"{v:.3f}" for v in meas]
    for a, d in spec.missing_temp_min:
        for i in np.flatnonzero((tm >= a) & (tm < a + d)):
            temp[i] = ""
    df = pd.DataFrame({"ts": START + pd.to_timedelta(secs, unit="s"), "temp_c": temp})
    if spec.record_ambient != "none":
        av = np.array([amb(x) for x in tm]) if spec.record_ambient == "true" else np.full(len(tm), spec.other_ambient_c)
        acol = [f"{v:.3f}" for v in av]
        for a, d in spec.missing_ambient_min:
            for i in np.flatnonzero((tm >= a) & (tm < a + d)):
                acol[i] = ""
        df["ambient_c"] = acol
    if spec.record_compressor:
        df["compressor_state"] = ["on" if u_at(x) else "off" for x in tm]
    df.attrs["coldsense_evidence"] = "SIMULATED"
    return df, Truth(spec, t_dense, t_end, switches)


def minutes(ts) -> float:
    return (pd.Timestamp(ts) - START).total_seconds() / 60
