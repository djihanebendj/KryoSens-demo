"""Prospective calibration benchmark for Phase 5 (SIMULATED rooms only).

Each ROOM is an independent synthetic configuration: its own seed, exchange
rate k and surroundings temperature, drawn from the class's range. For each
room the log is generated past the forecast origin, Phase 5 sees only the
data up to the origin (its causal cut), and the emitted interval is compared
with the reading the room actually produced at origin + horizon. For a
no-cooling forecast from a cycling room, that reading comes from the
counterfactual run in which cooling stops at the origin (same past, same noise
draws). Rooms are independent, so the coverage counts below are independent
trials, unlike the dependent held-out cases inside one log.

These are simulated rooms, not installations; nothing here is real-world
evidence. Reproducible: fixed seeds, no wall-clock values in the output.

    python -m coldsense.domains.cold_room.thermal.reliability_benchmark [--rooms N]
"""

from __future__ import annotations

import argparse
import logging
import math
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from coldsense.core.evidence import EvidenceClass

from .reliability import assess_source, wilson
from .schemas import EMITTED_INTERVALS, CrossingUncertaintyStatus, ForecastRequest, ReliabilityConfig
from .synthetic import START, SyntheticSpec, simulate


@dataclass(frozen=True)
class RoomClass:
    name: str
    history_h: float
    horizon_min: float
    k_range: tuple[float, float]
    ambient_range: tuple[float, float]
    noise_c: float = 0.05
    cycling: bool = True
    declared: bool = False
    threshold_c: float | None = None
    future_k_factor: float | None = None          # the exchange rate changes AT the origin (unseen in the past)


CLASSES = (
    RoomClass("small room, cycling, compressor recorded, 15 min", 48, 15, (1 / 400, 1 / 200), (22, 30),
              threshold_c=4.0),
    RoomClass("small room, cycling, compressor recorded, 30 min", 48, 30, (1 / 400, 1 / 200), (22, 30)),
    RoomClass("large room, cycling, compressor recorded, 30 min", 96, 30, (1 / 1500, 1 / 900), (22, 30),
              threshold_c=4.0),
    RoomClass("noisy sensor 0.3 C, cycling, 15 min", 48, 15, (1 / 400, 1 / 200), (22, 30), noise_c=0.3),
    RoomClass("rate doubles at the origin (unseen change), 15 min", 48, 15, (1 / 400, 1 / 200), (22, 30),
              future_k_factor=2.0),
    RoomClass("declared power cut, one passive stretch, 15 min", 12, 15, (1 / 400, 1 / 200), (22, 30),
              cycling=False, declared=True),
    # long enough for 9 + 9 independent passive periods, so emitted intervals are checked prospectively
    RoomClass("small room, cycling, compressor recorded, 144 h history, 15 min", 144, 15, (1 / 400, 1 / 200),
              (22, 30), threshold_c=4.0),
)


@dataclass
class RoomOutcome:
    room: str
    status: str
    reliability: str
    interval: str
    covered: bool | None
    covered_noise_free: bool | None
    width: float | None
    crossing_status: str | None
    crossing_covered: bool | None


def _spec(rc: RoomClass, seed: int) -> tuple[SyntheticSpec, float]:
    rng = np.random.default_rng(10_000 + seed)
    k = float(rng.uniform(*rc.k_range))
    amb = float(rng.uniform(*rc.ambient_range))
    origin_min = rc.history_h * 60
    hours = rc.history_h + (rc.horizon_min + 30) / 60
    spec = SyntheticSpec(hours=hours, k=k, ambient_c=amb, noise_c=rc.noise_c, seed=seed,
                         regime_change=(origin_min, k * rc.future_k_factor) if rc.future_k_factor else None)
    if rc.cycling:
        spec = replace(spec, b=0.25, t0_c=3.0, record_compressor=True, cooling_until_min=10 ** 9)
    return spec, origin_min


def run_room(rc: RoomClass, seed: int, level: float = 0.8) -> RoomOutcome:
    spec, om = _spec(rc, seed)
    df, truth = simulate(spec)
    cf_df, cf_truth = simulate(replace(spec, cooling_until_min=om)) if rc.cycling else (df, truth)
    origin = (START + pd.to_timedelta(om, unit="min")).to_pydatetime()
    req = ForecastRequest(horizon_min=rc.horizon_min, origin=origin, threshold_c=rc.threshold_c,
                          passive_since=START.to_pydatetime() if rc.declared else None)
    res = assess_source(df, evidence=EvidenceClass.SIMULATED, request=req,
                        config=ReliabilityConfig(interval_level=level))[3]
    rr = res.reliability_report
    u = rr.uncertainty
    name = f"{rc.name} #{seed}"
    out = RoomOutcome(name, res.status.value, rr.reliability.status.value, u.status.value, None, None, None,
                      rr.crossing.status.value if rr.forecast_available else None, None)
    if u.status in EMITTED_INTERVALS:
        target = pd.Timestamp(origin) + pd.to_timedelta(rc.horizon_min, unit="min")
        row = cf_df.loc[pd.to_datetime(cf_df["ts"]) == target, "temp_c"]
        observed = float(row.iloc[0]) if len(row) and pd.notna(row.iloc[0]) else None
        noise_free = cf_truth.t_dense(om + rc.horizon_min)
        out.covered = None if observed is None else bool(u.interval_lower_c <= observed <= u.interval_upper_c)
        out.covered_noise_free = bool(u.interval_lower_c <= noise_free <= u.interval_upper_c)
        out.width = u.interval_width_c
    c = rr.crossing
    if c.status == CrossingUncertaintyStatus.CALIBRATED_INTERVAL:
        actual = cf_truth.crossing_minutes(om, rc.threshold_c, True, om + rc.horizon_min * 4)   # after origin
        out.crossing_covered = actual is not None and c.lower_min <= actual <= c.upper_min
    return out


def run(rooms: int = 20, level: float = 0.8, classes=CLASSES) -> list[tuple[RoomClass, list[RoomOutcome]]]:
    return [(rc, [run_room(rc, s, level) for s in range(rooms)]) for rc in classes]


def _fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def report(results, level: float) -> str:
    lines = [f"# Phase 5 prospective calibration benchmark (SIMULATED rooms; nominal {level:.0%})", "",
             "Independent simulated rooms per class; coverage is checked against the reading the room produced "
             "after the origin, which Phase 5 never saw. Not real-world evidence.", "",
             "| class | rooms | forecasts | intervals | covered | coverage | 95 % CI | noise-free covered | "
             "mean width °C | reliability | crossing intervals (covered) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for rc, outs in results:
        fc = [o for o in outs if o.status == "FORECAST"]
        em = [o for o in outs if o.covered is not None]
        cov = sum(o.covered for o in em)
        ci = wilson(cov / len(em), len(em)) if em else None
        nf = sum(o.covered_noise_free for o in em)
        rel = Counter(o.reliability for o in outs)
        cr = [o for o in outs if o.crossing_covered is not None]
        lines.append(
            f"| {rc.name} | {len(outs)} | {len(fc)} | {len(em)} | {cov} | {_fmt(cov / len(em) if em else None)} | "
            f"{'n/a' if ci is None else f'{ci[0]:.2f}-{ci[1]:.2f}'} | {nf} | "
            f"{_fmt(float(np.mean([o.width for o in em])) if em else None)} | "
            f"{', '.join(f'{k} {v}' for k, v in sorted(rel.items()))} | "
            f"{len(cr)} ({sum(o.crossing_covered for o in cr)}) |")
    lines += ["", "Interval statuses per class:", ""]
    for rc, outs in results:
        lines.append(f"* {rc.name}: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(o.interval for o in outs)
                                                                               .items())))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rooms", type=int, default=20)
    ap.add_argument("--level", type=float, default=0.8)
    a = ap.parse_args(argv)
    logging.disable(logging.INFO)
    print(report(run(a.rooms, a.level), a.level), end="")


if __name__ == "__main__":
    main()
