"""Synthetic ground-truth benchmark of the thermal engine. Every number it prints is SIMULATED.

    python -m coldsense.domains.cold_room.thermal.benchmark

Runs a fixed catalogue of generated logs (``synthetic.py``, fixed seeds) through the
full engine and prints, per scenario, the outcome, the true and fitted rate, the
largest forecast error against the noise-free truth, and the crossing error; then the
held-out model-versus-baseline counts by horizon. Nothing here is evidence about
a real room.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, replace

import pandas as pd

from coldsense.core.evidence import EvidenceClass

from . import registry as R
from .engine import forecast_source
from .schemas import Direction, ForecastRequest, ForecastStatus
from .synthetic import START, SyntheticSpec, minutes, simulate


@dataclass
class Scenario:
    name: str
    spec: SyntheticSpec
    origin_h: float
    horizon: float
    passive: bool = True
    request: dict | None = None


def catalogue() -> list[Scenario]:
    S = SyntheticSpec
    vary = lambda t: 25 + 6 * math.sin(2 * math.pi * t / 720)       # noqa: E731
    outage = dict(hours=22, b=0.25, t0_c=4.5, cooling_until_min=16 * 60, record_compressor=True)
    return [
        Scenario("passive warming, measured ambient", S(hours=12, seed=1), 10, 60, request={"threshold_c": 23.0}),
        Scenario("passive warming, 120 min extrapolated", S(hours=12, seed=1), 4, 120,
                 request={"threshold_c": 18.0, "allow_extrapolation": True}),
        Scenario("passive warming, 120 min default", S(hours=12, seed=1), 4, 120, request={"threshold_c": 18.0}),
        Scenario("passive cooling (downward)", S(hours=12, t0_c=25, ambient_c=5, seed=2), 10, 60,
                 request={"threshold_c": 6.0, "direction": Direction.DOWNWARD}),
        Scenario("fast room k=1/100", S(hours=12, k=1 / 100, seed=5), 8, 60),
        Scenario("slow room k=1/1500", S(hours=12, k=1 / 1500, seed=5), 8, 60),
        Scenario("no ambient column", S(hours=12, record_ambient="none", seed=7), 8, 60),
        Scenario("time-varying ambient", S(hours=12, ambient_fn=vary, seed=4), 8, 60),
        Scenario("time-varying ambient + future", S(hours=12, ambient_fn=vary, seed=4), 8, 60,
                 request={"future_ambient_c": 20.0}),
        Scenario("noisy readings 0.3 °C", S(hours=12, noise_c=0.3, seed=6), 8, 60),
        Scenario("irregular timestamps", S(hours=12, jitter_s=25, seed=9), 8, 60),
        Scenario("60-min gap", S(hours=12, gaps_min=[(200, 60)], seed=10), 8, 30),
        Scenario("sensor spikes", S(hours=12, spikes=[(150, 6.0), (320, -5.0)], seed=11), 8, 30),
        Scenario("no temperature variation", S(hours=12, t0_c=25, seed=13), 8, 60),
        Scenario("unrecorded oscillating load",
                 S(hours=12, t0_c=20, record_ambient="none", seed=14,
                   ambient_fn=lambda t: 25 + 10 * math.sin(2 * math.pi * t / 120)), 8, 60),
        Scenario("near-linear (tau 5000 min)", S(hours=8, k=1 / 5000, record_ambient="none", seed=15), 6, 30),
        Scenario("regime change at 5 h", S(hours=12, regime_change=(300, 1 / 80), seed=16), 9, 60),
        Scenario("wrong ambient sensor", S(hours=12, k=1 / 400, t0_c=2, ambient_c=12, record_ambient="other",
                                           other_ambient_c=35, seed=24), 8, 60),
        Scenario("very noisy 1.2 °C", S(hours=12, noise_c=1.2, seed=26), 8, 60),
        Scenario("recorded power cut", S(**outage, seed=19), 17, 120, passive=False,
                 request={"threshold_c": 12.0, "allow_extrapolation": True}),
        Scenario("cooling ON at origin (what-if)", S(hours=14, b=0.25, t0_c=4.5, cooling_until_min=14 * 60,
                                                     record_compressor=True, seed=21), 11.9, 15, passive=False),
    ]


def run_scenario(sc: Scenario) -> dict:
    df, truth = simulate(sc.spec)
    origin = START + pd.to_timedelta(sc.origin_h, unit="h")
    req = ForecastRequest(horizon_min=sc.horizon, origin=origin.to_pydatetime(),
                          passive_since=START.to_pydatetime() if sc.passive else None, **(sc.request or {}))
    res = forecast_source(df, evidence=EvidenceClass.SIMULATED, request=req)[3]
    om = minutes(origin)
    if sc.spec.cooling_until_min is not None and sc.spec.cooling_until_min > om:
        # the forecast answers "no cooling from the origin": compare with that counterfactual, not with the
        # cycling that really followed (same trajectory up to the origin, cooling cut there)
        truth = simulate(replace(sc.spec, cooling_until_min=om))[1]
    row = {"scenario": sc.name, "status": res.status.value, "code": res.refusal_code.value if res.refusal_code else "",
           "selected": res.selected_model or "", "validated_h": res.validated_horizon_min, "k_true": sc.spec.k,
           "k_fit": None, "k_in_range": None, "max_err_c": None, "cross_err_min": None, "result": res}
    fit = res.fit_diagnostics or next((c.fit for c in res.candidate_models if c.kind == "PHYSICAL" and c.fit
                                       and c.rejection_code is None), None)
    if fit is not None:
        k = next(p for p in fit.parameters if p.name == "k")
        row["k_fit"] = k.value
        row["k_in_range"] = k.profile_range[0] <= sc.spec.k <= k.profile_range[1] if k.profile_range else None
    if res.status == ForecastStatus.FORECAST:
        row["max_err_c"] = max(abs(p.temp_c - truth.t_dense(om + p.minutes)) for p in res.trajectory)
        if res.predicted_crossing_minutes is not None:
            up = res.threshold.direction == Direction.UPWARD
            true_x = truth.crossing_minutes(om, res.threshold.value_c, up, om + sc.horizon)
            row["cross_err_min"] = None if true_x is None else res.predicted_crossing_minutes - true_x
    return row


def main() -> None:
    previous = logging.root.manager.disable
    logging.disable(logging.INFO)                 # keep the table readable; restored below
    try:
        rows = [run_scenario(sc) for sc in catalogue()]
    finally:
        logging.disable(previous)
    f = lambda v, d=4: "" if v is None else (f"{v:.{d}g}" if isinstance(v, float) else str(v))   # noqa: E731
    print("SIMULATED benchmark (synthetic ground truth; not evidence about a real room)\n")
    print("| scenario | status | code | selected | validated h | k true | k fitted | true k in profile range "
          "| max |forecast - truth| °C | crossing error min |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['scenario']} | {r['status']} | {r['code']} | {r['selected']} | {f(r['validated_h'])} | "
              f"{f(r['k_true'])} | {f(r['k_fit'])} | {f(r['k_in_range'])} | {f(r['max_err_c'], 3)} | "
              f"{f(r['cross_err_min'], 3)} |")
    agg: dict[tuple, list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    for r in rows:
        for c in r["result"].model_comparison:
            if c.n_common:
                a = agg[(c.role, c.horizon_min, c.physical_method)]
                for i, v in enumerate((c.n_common, c.physical_better, c.baseline_better, c.tie, c.neither_adequate)):
                    a[i] += v
    print("\nHeld-out comparison with the best baseline per case, summed over scenarios "
          "(physical better / baseline better / tie / neither adequate)\n")
    print("| role | horizon min | physical model | cases | physical better | baseline better | tie "
          "| neither adequate |")
    print("|---|---|---|---|---|---|---|---|")
    for (role, h, m), a in sorted(agg.items(), key=lambda x: (x[0][0], x[0][2], x[0][1])):
        print(f"| {role} | {h:g} | {m} | {a[0]} | {a[1]} | {a[2]} | {a[3]} | {a[4]} |")
    sel = [r for r in rows if r["status"] == "FORECAST"]
    print(f"\n{len(rows)} scenarios: {len(sel)} forecasts "
          f"({sum(r['selected'] in (R.PERSISTENCE, R.RECENT_SLOPE) for r in sel)} by a baseline), "
          f"{len(rows) - len(sel)} refusals")


if __name__ == "__main__":
    main()
