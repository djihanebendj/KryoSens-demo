"""First-order thermal exchange: fitting, identifiability and forecast laws.

Model (per fitting segment s, time tau in minutes from the segment's first row):

    dT/dt = k * (E_s(t) - T)

* ``E_s`` constant and known (measured ambient mean, or an assumed value):
      T(tau) = E_s + (T0_s - E_s) * exp(-k tau)
* ``E_s`` constant and unknown (fitted effective equilibrium, shared by segments):
      same formula, E a fitted parameter
* ``E_s(t)`` a measured series (time-varying ambient): integrated exactly for
  ambient varying linearly between consecutive observations (exponential
  integrator; no step-size error, only the interpolation assumption):
      T_{i+1} = T_i e^{-x} + a_{i+1} - a_i e^{-x} - (a_{i+1} - a_i) phi(x),
      x = k (t_{i+1} - t_i),  phi(x) = (1 - e^{-x}) / x

For a fixed rate ``k`` the trajectory is LINEAR in the initial states T0_s and
in any fitted equilibria, so those are solved exactly by least squares
(variable projection) and only ``k`` is searched: a log-spaced grid over the
admissible range, then bounded refinement. The grid also gives the profile of
the sum of squared errors over ``k``, used for the identifiability check:

    keep k while SSE(k) <= SSE_min * (1 + F(0.95; 1, nu) / nu),  nu = n_eff - p

with n_eff corrected for lag-1 residual autocorrelation. The rate is called
identifiable only if that range closes inside the searched interval on both
sides and spans at most ``IDENT_MAX_K_RATIO``. The range assumes independent
Gaussian residuals and is used to judge identifiability, never reported as a
calibrated forecast uncertainty (Phase 5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats

from . import limits as L
from .schemas import CrossingStatus, Direction


# ------------------------------------------------------------------ segments
@dataclass
class FitSegment:
    """One contiguous stretch of usable observations with a single regime."""

    seg_id: str
    t: np.ndarray                 # minutes since the segment's first row (actual elapsed time)
    y: np.ndarray                 # observed room temperature, °C
    eq_kind: str                  # "known" | "param" | "series"
    eq_value: float | None = None     # known constant equilibrium (°C)
    eq_param: str | None = None       # name of a shared fitted equilibrium
    amb: np.ndarray | None = None     # ambient per row (°C) for "series"

    def __post_init__(self) -> None:
        if not (len(self.t) == len(self.y) and len(self.t) >= 3):
            raise ValueError("a fitting segment needs at least 3 aligned observations")
        if not (np.all(np.isfinite(self.t)) and np.all(np.isfinite(self.y)) and np.all(np.diff(self.t) > 0)):
            raise ValueError("segment times must be finite and strictly increasing; temperatures finite")
        if self.eq_kind == "series" and (self.amb is None or not np.all(np.isfinite(self.amb))):
            raise ValueError("a time-varying equilibrium needs a finite ambient value at every row")


def _phi(x: np.ndarray) -> np.ndarray:
    """(1 - e^-x) / x, stable as x -> 0."""
    out = np.ones_like(x)
    nz = x > 1e-12
    out[nz] = -np.expm1(-x[nz]) / x[nz]
    return out


def forced_response(t: np.ndarray, a: np.ndarray, k: float) -> np.ndarray:
    """Zero-initial-state response of dT/dt = k (a(t) - T), a linear between samples (exact)."""
    d = np.diff(t)
    x = k * d
    e = np.exp(-x)
    inc = a[1:] - a[:-1] * e - (a[1:] - a[:-1]) * _phi(x)
    g = np.empty(len(t))
    g[0] = 0.0
    acc = 0.0
    for i in range(len(d)):          # linear recursion with time-varying factor
        acc = acc * e[i] + inc[i]
        g[i + 1] = acc
    return g


@dataclass
class Projection:
    sse: float
    names: list[str]
    coef: np.ndarray
    fitted: np.ndarray
    rank: int
    n_cols: int


def project(segs: list[FitSegment], k: float) -> Projection:
    """Least-squares solution of all linear parameters for a fixed rate k.

    Each segment's initial state affects only its own rows, so it is eliminated
    exactly per segment (projection off a_s = e^{-k t}); only the few shared
    equilibrium parameters remain in a small normal system. Identical to solving
    the full block design at once, in O(n) instead of O(n x segments) memory."""
    params = sorted({s.eq_param for s in segs if s.eq_kind == "param"})
    q = len(params)
    gram = np.zeros((q, q))
    rhs = np.zeros(q)
    parts = []
    for s in segs:
        a = np.exp(-k * s.t)                                   # initial-state column
        w = -np.expm1(-k * s.t)                                # equilibrium column (1 - e^{-kt})
        if s.eq_kind == "known":
            known = s.eq_value * w
        elif s.eq_kind == "series":
            known = forced_response(s.t, s.amb, k)
        else:
            known = np.zeros(len(s.t))
        r = s.y - known
        B = np.zeros((len(s.t), q))
        if s.eq_kind == "param":
            B[:, params.index(s.eq_param)] = w
        aa = float(a @ a)                                      # > 0: a[0] = 1
        pr = r - a * (a @ r) / aa
        pb = B - np.outer(a, a @ B) / aa
        gram += pb.T @ pb
        rhs += pb.T @ pr
        parts.append((a, aa, r, B, known))
    if q:
        theta, _, rank_q, _ = np.linalg.lstsq(gram, rhs, rcond=None)
    else:
        theta, rank_q = np.zeros(0), 0
    coef, fitted, sse = [], [], 0.0
    for (a, aa, r, B, known), _ in zip(parts, segs, strict=True):
        rb = r - B @ theta
        t0 = float(a @ rb) / aa
        fit = known + B @ theta + a * t0
        coef.append(t0)
        fitted.append(fit)
        res = (r + known) - fit
        sse += float(res @ res)
    return Projection(sse, [f"T0[{s.seg_id}]" for s in segs] + params, np.array(coef + list(theta)),
                      np.concatenate(fitted), len(segs) + int(rank_q), len(segs) + q)


# ------------------------------------------------------------------ fit
@dataclass
class FitOutcome:
    status: str                              # CONVERGED | AT_SEARCH_BOUND | SINGULAR | FAILED
    k: float | None = None
    k_bounds: tuple[float, float] = (math.nan, math.nan)
    eq: dict[str, float] = field(default_factory=dict)
    eq_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    k_range: tuple[float, float] | None = None
    k_range_open: tuple[bool, bool] = (True, True)
    identifiable: bool = False
    ident_reason: str = ""
    sse: float | None = None
    n: int = 0
    p: int = 0
    n_eff: float | None = None
    rho: float | None = None
    residuals: list[np.ndarray] = field(default_factory=list)
    fitted: list[np.ndarray] = field(default_factory=list)
    message: str = ""


def _lag1(res: list[np.ndarray]) -> float | None:
    allr = np.concatenate(res)
    mu = allr.mean()
    den = float(((allr - mu) ** 2).sum())
    if den <= 0:
        return None
    num = sum(float(((r[:-1] - mu) * (r[1:] - mu)).sum()) for r in res if len(r) > 1)
    return num / den


def fit_first_order(segs: list[FitSegment], k_lo: float, k_hi: float) -> FitOutcome:
    """Profile search over k with all linear parameters solved exactly. Never raises
    for numerical trouble: failures come back as a status with a message."""
    if not (0 < k_lo < k_hi and math.isfinite(k_hi)):
        return FitOutcome("FAILED", message=f"invalid rate search interval [{k_lo}, {k_hi}]")
    out = FitOutcome("FAILED", k_bounds=(k_lo, k_hi))
    lk = np.linspace(math.log(k_lo), math.log(k_hi), L.PROFILE_GRID_POINTS)

    def sse(logk: float) -> float:
        try:
            v = project(segs, math.exp(logk)).sse
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return math.inf
        return v if math.isfinite(v) else math.inf

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        prof = np.array([sse(x) for x in lk])
        if not np.isfinite(prof).any():
            out.message = "the objective could not be evaluated for any rate"
            return out
        i = int(np.argmin(prof))
        lo_i, hi_i = max(i - 1, 0), min(i + 1, len(lk) - 1)
        res = optimize.minimize_scalar(sse, bounds=(lk[lo_i], lk[hi_i]), method="bounded",
                                       options={"xatol": 1e-7})
        lopt = float(res.x) if res.success and res.fun <= prof[i] else float(lk[i])
        best = min(float(res.fun) if res.success else math.inf, float(prof[i]))
        proj = project(segs, math.exp(lopt))

    out.k = math.exp(lopt)
    out.sse = best
    out.n = sum(len(s.y) for s in segs)
    out.p = proj.n_cols + 1
    names = proj.names
    out.eq = {nm: float(c) for nm, c in zip(names, proj.coef, strict=True) if not nm.startswith("T0[")}
    off = 0
    for s in segs:
        m = len(s.y)
        out.fitted.append(proj.fitted[off:off + m])
        out.residuals.append(s.y - proj.fitted[off:off + m])
        off += m
    if out.n - out.p < 2:
        out.status = "FAILED"
        out.message = f"{out.n} observations for {out.p} parameters"
        return out
    at_bound = i in (0, len(lk) - 1)
    out.status = "AT_SEARCH_BOUND" if at_bound else ("SINGULAR" if proj.rank < proj.n_cols else "CONVERGED")

    # ---- identifiability: profile range of k
    rho = _lag1(out.residuals)
    out.rho = rho
    n_eff = out.n * (1 - rho) / (1 + rho) if rho is not None and rho > 0 else float(out.n)
    n_eff = float(min(max(n_eff, out.p + 2), out.n))
    out.n_eff = n_eff
    nu = n_eff - out.p
    thr = best * (1 + stats.f.ppf(L.PROFILE_LEVEL, 1, nu) / nu) if best > 0 else 0.0

    def edge(direction: int) -> tuple[float, bool]:
        j = int(np.searchsorted(lk, lopt)) - (1 if direction < 0 else 0)
        j = min(max(j, 0), len(lk) - 1)
        while 0 <= j < len(lk) and prof[j] <= thr:
            j += direction
        if not 0 <= j < len(lk):
            return (k_lo if direction < 0 else k_hi), True
        a, b = sorted((lk[j], lopt))
        try:
            root = optimize.brentq(lambda x: sse(x) - thr, a, b, xtol=1e-9)
        except ValueError:
            root = lk[j]
        return math.exp(root), False

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        (klo, open_lo), (khi, open_hi) = edge(-1), edge(+1)
        out.k_range, out.k_range_open = (klo, khi), (open_lo, open_hi)
        # equilibrium values compatible with the data: profiled over the k range
        inside = [x for x in lk if math.log(klo) <= x <= math.log(khi)] + [math.log(klo), math.log(khi), lopt]
        for nm in out.eq:
            vals = []
            for x in inside:
                pr = project(segs, math.exp(x))
                vals.append(float(pr.coef[pr.names.index(nm)]))
            out.eq_range[nm] = (min(vals), max(vals))

    if out.status == "AT_SEARCH_BOUND":
        out.ident_reason = (f"best rate lies at the edge of the searched range [{k_lo:.3g}, {k_hi:.3g}] /min: "
                            "the data do not determine it")
    elif out.status == "SINGULAR":
        out.ident_reason = "the linear parameters are not separately determined (rank-deficient design)"
    elif open_lo or open_hi:
        side = " and ".join(s for s, o in (("slower", open_lo), ("faster", open_hi)) if o)
        out.ident_reason = (f"rates down to the {side} end of the searched range fit about as well "
                            f"(profile-likelihood {L.PROFILE_LEVEL:.0%}); the data do not pin the rate down")
    elif khi / klo > L.IDENT_MAX_K_RATIO:
        out.ident_reason = (f"rates from {klo:.3g} to {khi:.3g} /min fit about as well (ratio {khi / klo:.1f} > "
                            f"{L.IDENT_MAX_K_RATIO}); the time constant is not determined within about x2")
    else:
        out.identifiable = True
        out.ident_reason = (f"profile-likelihood range of the rate {klo:.3g}-{khi:.3g} /min (ratio "
                            f"{khi / klo:.2f} <= {L.IDENT_MAX_K_RATIO}), closed inside the searched range")
    return out


# ------------------------------------------------------------------ forecast laws
@dataclass(frozen=True)
class Law:
    """Closed-form trajectory from an origin: T(h) for h >= 0 minutes."""

    kind: str                      # "constant" | "linear" | "exponential"
    t0: float
    slope: float | None = None     # °C/min (linear)
    eq: float | None = None        # °C (exponential)
    k: float | None = None         # 1/min (exponential)

    def __post_init__(self) -> None:
        vals = [self.t0, self.slope, self.eq, self.k]
        if not all(v is None or math.isfinite(v) for v in vals):
            raise ValueError("forecast law parameters must be finite")
        if self.kind == "exponential" and not (self.k and self.k > 0 and self.eq is not None):
            raise ValueError("an exponential law needs k > 0 and an equilibrium")
        if self.kind == "linear" and self.slope is None:
            raise ValueError("a linear law needs a slope")

    def at(self, h: np.ndarray | float) -> np.ndarray | float:
        if self.kind == "constant":
            return self.t0 + 0 * np.asarray(h, dtype=float)
        if self.kind == "linear":
            return self.t0 + self.slope * np.asarray(h, dtype=float)
        # T0 + (E - T0)(1 - e^{-kh}), written with expm1 to avoid cancellation
        return self.t0 - (self.eq - self.t0) * np.expm1(-self.k * np.asarray(h, dtype=float))

    def crossing(self, threshold: float, direction: Direction, horizon: float) -> tuple[CrossingStatus, float | None]:
        """Analytic first crossing time (minutes) within (0, horizon]."""
        up = direction == Direction.UPWARD
        beyond = (self.t0 >= threshold) if up else (self.t0 <= threshold)
        if beyond:
            return CrossingStatus.ALREADY_CROSSED, None
        t: float | None = None
        if self.kind == "linear":
            if (up and self.slope > 0) or (not up and self.slope < 0):
                t = (threshold - self.t0) / self.slope
        elif self.kind == "exponential":
            # monotone approach to eq: reachable only if eq lies strictly beyond the threshold
            if (up and self.eq > threshold) or (not up and self.eq < threshold):
                t = math.log1p((self.t0 - threshold) / (threshold - self.eq)) / self.k
        if t is None or not math.isfinite(t):
            return CrossingStatus.NOT_REACHABLE, None
        if t > horizon:
            return CrossingStatus.NOT_WITHIN_HORIZON, None
        return CrossingStatus.CROSSING_PREDICTED, t
