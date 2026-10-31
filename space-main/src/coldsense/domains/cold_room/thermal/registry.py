"""Model registry and input-capability inventory.

Every candidate model, implemented or deferred, is described here with its
inputs, assumptions and validation rule. A DEFERRED_NOT_IMPLEMENTED entry is
documentation of what would be needed, never a capability.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import pandas as pd

from coldsense.core.evidence import EvidenceClass
from coldsense.core.governance import ModelCard

from ..schemas import DataQualityReport, SegmentationResult
from .schemas import InputItem, InputStatus

IMPLEMENTED = "IMPLEMENTED"
DEFERRED = "DEFERRED_NOT_IMPLEMENTED"

PERSISTENCE = "persistence"
RECENT_SLOPE = "recent_slope"
FO_FITTED = "first_order_fitted_equilibrium"
FO_MEASURED = "first_order_measured_ambient"
FO_ASSUMED = "first_order_assumed_equilibrium"
FO_VARYING = "first_order_varying_ambient"
FO_ACTIVE = "first_order_active_cooling"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    version: str
    kind: str                       # BASELINE | PHYSICAL
    status: str                     # IMPLEMENTED | DEFERRED_NOT_IMPLEMENTED
    equation: str
    interpretation: str
    required_inputs: tuple[str, ...]
    regimes: str
    assumptions: tuple[str, ...]
    identifiability: str
    fitting: str
    validation: str
    limitations: tuple[str, ...]
    missing_for_implementation: tuple[str, ...] = field(default_factory=tuple)
    # ---- follow-up metadata (canonical input names from inputs.CATALOGUE)
    availability: str = ""                              # ALWAYS | CONDITIONAL | DEFERRED
    required_input_names: tuple[str, ...] = ()
    optional_input_names: tuple[str, ...] = ()
    eligibility: str = ""
    missing_input_limitations: tuple[str, ...] = ()
    refusal_conditions: tuple[str, ...] = ()

    def card(self) -> ModelCard | None:
        if self.status != IMPLEMENTED:
            return None
        return ModelCard(
            name=self.name, version=self.version, purpose=self.interpretation,
            baseline="none: this IS a baseline" if self.kind == "BASELINE" else "persistence and recent_slope",
            metric="MAE / RMSE of held-out forecasts by horizon; threshold-crossing time error",
            failure_modes=list(self.limitations), product_benefit=self.regimes,
            validation_evidence=EvidenceClass.SIMULATED,
            validation_summary="checked on SIMULATED data only; no REAL room evaluation yet")


_PASSIVE = ("no active cooling over the fitted segments and over the forecast horizon",
            "room air behaves as one well-mixed node at the sensor (no stratification, no product lag)",
            "no unrecorded internal loads (lights, people, defrost heaters, fans) that change over time",
            "door openings and loading are absent or recorded as disturbances, not modelled")

_BASE: tuple[ModelSpec, ...] = (
    ModelSpec(
        PERSISTENCE, "1", "BASELINE", IMPLEMENTED, "T_hat(t0+h) = T(t0)",
        "the temperature stays at its last observed value",
        ("room temperature [°C]",), "any; meaningful when the temperature is stable over the horizon",
        ("nothing changes over the horizon",), "nothing fitted", "none",
        "held-out forecasts, same origins and horizons as every other method",
        ("never predicts a crossing that has not happened", "error grows with the true change")),
    ModelSpec(
        RECENT_SLOPE, "1", "BASELINE", IMPLEMENTED, "T_hat(t0+h) = T(t0) + s h, s = OLS slope over (t0-15 min, t0]",
        "the recent rate of change continues unchanged",
        ("room temperature [°C]", "timestamps"), "the regime of the trailing window continues",
        ("the trailing window's rate continues for the whole horizon",),
        "slope standard error and half-window slopes reported", "OLS on actual elapsed time",
        "held-out forecasts, same origins and horizons",
        ("not a physical model: ignores slowing as the room approaches its surroundings",
         "noise-sensitive with few readings; refused below 5 readings or half-window span")),
    ModelSpec(
        FO_FITTED, "1", "PHYSICAL", IMPLEMENTED, "dT/dt = k (E - T);  T(t) = E + (T0 - E) e^(-k t)",
        "lumped first-order exchange toward an EFFECTIVE equilibrium E fitted from the data "
        "(not a measured ambient)",
        ("room temperature [°C]", "passive-regime evidence (recorded compressor OFF or declared)"),
        "passive (no active cooling)", _PASSIVE + ("E stays as fitted over the horizon",),
        "k and E are separable only if the segments show curvature; profile-likelihood range of k must close "
        "within the searched range and span <= 4x",
        "variable projection: T0 per segment and E by least squares for each k; 1-D search over log k",
        "chronological hold-out: fit 60 %, select 20 %, final evaluation 20 % of passive time",
        ("a nearly straight rise cannot separate k from E -> INSUFFICIENT_IDENTIFIABILITY",
         "E is an effective value: it absorbs internal loads and is not the ambient")),
    ModelSpec(
        FO_MEASURED, "1", "PHYSICAL", IMPLEMENTED, "dT/dt = k (T_amb - T), T_amb = segment mean of measured ambient",
        "lumped first-order exchange with the space whose temperature the ambient sensor measures",
        ("room temperature [°C]", "ambient temperature [°C], steady within each segment (range <= 1 °C)",
         "passive-regime evidence"),
        "passive; ambient steady", _PASSIVE + (
            "the ambient sensor measures the space the room exchanges heat with (NOT_VERIFIED from data)",
            "over the horizon ambient stays at its last measured value unless future_ambient_c is given"),
        "k only; well determined when the room-ambient difference is large",
        "variable projection with E fixed per segment; 1-D search over log k",
        "chronological hold-out as above",
        ("wrong if the ambient sensor is not the room's real surroundings (e.g. outdoor vs corridor)",
         "internal loads bias k")),
    ModelSpec(
        FO_ASSUMED, "1", "PHYSICAL", IMPLEMENTED, "dT/dt = k (E_assumed - T)",
        "first-order exchange toward an equilibrium stated by the user",
        ("room temperature [°C]", "assumed equilibrium [°C] (user)", "passive-regime evidence"),
        "passive", _PASSIVE + ("the user's equilibrium value is right (ASSUMED)",),
        "k only", "variable projection with E fixed; 1-D search over log k", "chronological hold-out as above",
        ("only as good as the assumed equilibrium",)),
    ModelSpec(
        FO_VARYING, "1", "PHYSICAL", IMPLEMENTED, "dT/dt = k (T_amb(t) - T)",
        "first-order exchange with a measured, time-varying ambient",
        ("room temperature [°C]", "ambient temperature series [°C] covering each segment", "passive-regime evidence",
         "future ambient [°C] for forecasting (user scenario)"),
        "passive; ambient varies by more than 1 °C within a segment",
        _PASSIVE + ("ambient varies linearly between its readings",
                    "future ambient is the scenario value given (ASSUMED); never extrapolated"),
        "k only", "exact exponential integrator for piecewise-linear ambient; 1-D search over log k",
        "chronological hold-out; held-out forecasts hold ambient at its value at the origin (no future data)",
        ("missing ambient bridged only across holes <= 3 median intervals",
         "forecast refused without future_ambient_c")),
    ModelSpec(
        FO_ACTIVE, "1", "PHYSICAL", IMPLEMENTED,
        "dT/dt = k (T_amb - T) - b u,  u = recorded compressor state (0/1); E_on = T_amb - b/k",
        "first-order exchange plus an effective cooling term while the compressor is recorded ON",
        ("room temperature [°C]", "compressor state (recorded ON and OFF runs)", "ambient [°C] (else E_off fitted)"),
        "cycling (recorded ON and OFF runs)",
        _PASSIVE[1:] + ("cooling effect is constant while ON (b, °C/min); b is NOT a cooling capacity or power",),
        "needs both ON and OFF runs: OFF runs fix k, ON runs fix E_on; ON-only data cannot separate k from b",
        "variable projection: T0 per run, E_on (and E_off without ambient) linear; 1-D search over log k",
        "hindcast on held-out runs with the RECORDED compressor state",
        ("cannot forecast: the future compressor state is unknown (HINDCAST_ONLY)",
         "binary state gives no cooling power; capacity varies with temperature in reality")),
    # ---------------------------------------------------------------- deferred
    ModelSpec(
        "multi_node_rc", "-", "PHYSICAL", DEFERRED, "C_a dT_a/dt = (T_p - T_a)/R_ap + (T_amb - T_a)/R_aw + ...",
        "separate air, product and envelope nodes", ("air temperature", "product temperature", "wall/supply-air "
                                                     "temperatures"), "-", (), "-", "-", "-", (),
        ("a second measured node (product or wall temperature); one air sensor cannot identify two time constants",)),
    ModelSpec(
        "door_infiltration", "-", "PHYSICAL", DEFERRED, "Q_inf = m_dot_air (h_out - h_in)",
        "heat and moisture carried in through open doors", ("door open/closed state and timestamps",
                                                            "door geometry", "humidity inside and outside"),
        "-", (), "-", "-", "-", (),
        ("door state signal", "door area / type / air curtain", "relative humidity",
         "an airflow or infiltration model; duration alone does not give the mass flow")),
    ModelSpec(
        "product_load", "-", "PHYSICAL", DEFERRED,
        "Q = m c_p (T_entry - T_target) / t_pull-down (+ latent, respiration)",
        "heat brought in by product", ("product mass", "specific heat", "entry temperature", "loading times"),
        "-", (), "-", "-", "-", (), ("product mass, type, specific heat, entry temperature and loading timestamps",)),
    ModelSpec(
        "transmission_load", "-", "PHYSICAL", DEFERRED, "Q_dot = U A (T_out - T_in)",
        "steady heat flow through the envelope", ("U-value [W/m²K]", "surface areas [m²]",
                                                  "adjacent temperatures"),
        "-", (), "-", "-", "-", (), ("U-values or construction, surface areas, adjacent-space temperatures",)),
    ModelSpec(
        "defrost_aware", "-", "PHYSICAL", DEFERRED, "regime switch at recorded defrost start/end",
        "excludes or models defrost heating", ("defrost state / timestamps",), "-", (), "-", "-", "-", (),
        ("defrost state signal; without it defrost shows only as an unexplained disturbance",)),
    ModelSpec(
        "humidity_latent", "-", "PHYSICAL", DEFERRED, "moisture balance + latent heat on the coil",
        "latent loads and frosting", ("relative humidity inside and outside",), "-", (), "-", "-", "-", (),
        ("indoor and adjacent relative humidity",)),
)

# ---------------------------------------------------------------- follow-up: eligibility metadata
REGIME_EVIDENCE = "regime_evidence (one of: compressor_state, power_interruption, passive_since)"
_COMMON_REFUSALS = ("held-out MAE above the 0.5 °C adequacy limit (NO_ADEQUATE_MODEL)",
                    "requested horizon beyond the validated one (INSUFFICIENT_EVIDENCE)")
_PHYS_REFUSALS = ("within-segment R² < 0.9 (POOR_FIT)",
                  "rate not pinned down by the data (INSUFFICIENT_IDENTIFIABILITY)",
                  "equilibrium outside plausible surroundings (IMPLAUSIBLE_PARAMETERS)", *_COMMON_REFUSALS)
_DISTURBANCE_INPUTS = ("door_open", "defrost", "product_load", "setpoint_change", "maintenance", "equipment_fault")
_EXT: dict[str, dict] = {
    PERSISTENCE: dict(
        availability="ALWAYS", required_input_names=("room_temperature", "timestamp"),
        optional_input_names=_DISTURBANCE_INPUTS,
        eligibility="NO_COOLING: only when the origin lies in an accepted passive segment (it extrapolates the "
                    "observed regime); CONTINUATION: any regime",
        refusal_conditions=("not applicable at the origin (NO_COOLING outside a passive segment)", *_COMMON_REFUSALS,
                            "never extrapolated beyond the validated horizon")),
    RECENT_SLOPE: dict(
        availability="ALWAYS", required_input_names=("room_temperature", "timestamp"),
        optional_input_names=_DISTURBANCE_INPUTS,
        eligibility="as persistence, plus >= 5 readings spanning >= half of the 15-min window before the origin",
        refusal_conditions=("too few readings in the trailing window", "not applicable at the origin (NO_COOLING "
                            "outside a passive segment)", *_COMMON_REFUSALS, "never extrapolated")),
    FO_FITTED: dict(
        availability="CONDITIONAL", required_input_names=("room_temperature", "timestamp", REGIME_EVIDENCE),
        optional_input_names=_DISTURBANCE_INPUTS,
        eligibility="accepted passive segments before the origin; the segments must show curvature",
        missing_input_limitations=("without ambient temperature, E is identified from curvature alone and may be "
                                   "INSUFFICIENT_IDENTIFIABILITY on short or nearly straight rises",),
        refusal_conditions=_PHYS_REFUSALS),
    FO_MEASURED: dict(
        availability="CONDITIONAL",
        required_input_names=("room_temperature", "timestamp", REGIME_EVIDENCE, "ambient_temperature"),
        optional_input_names=("future_ambient", *_DISTURBANCE_INPUTS),
        eligibility="ambient in >= 90 % of a segment's rows and steady within 1 °C",
        missing_input_limitations=("ineligible without ambient temperature",
                                   "what the ambient sensor measures is NOT_VERIFIED: a wrong sensor biases k"),
        refusal_conditions=(*_PHYS_REFUSALS, "forecast: ambient varied in the last 60 min and no future_ambient")),
    FO_ASSUMED: dict(
        availability="CONDITIONAL",
        required_input_names=("room_temperature", "timestamp", REGIME_EVIDENCE, "assumed_equilibrium"),
        optional_input_names=_DISTURBANCE_INPUTS,
        eligibility="the user states the equilibrium (ASSUMED, never treated as a measured ambient)",
        missing_input_limitations=("ineligible without an assumed equilibrium",),
        refusal_conditions=_PHYS_REFUSALS),
    FO_VARYING: dict(
        availability="CONDITIONAL",
        required_input_names=("room_temperature", "timestamp", REGIME_EVIDENCE, "ambient_temperature"),
        optional_input_names=("future_ambient", *_DISTURBANCE_INPUTS),
        eligibility="ambient varies by more than 1 °C within a segment and is present at both segment ends",
        missing_input_limitations=("ineligible without ambient temperature",
                                   "forecasts need future_ambient: future ambient is never extrapolated"),
        refusal_conditions=(*_PHYS_REFUSALS, "forecast without future_ambient (MISSING_REQUIRED_INPUT)")),
    FO_ACTIVE: dict(
        availability="CONDITIONAL",
        required_input_names=("room_temperature", "timestamp", "compressor_state"),
        optional_input_names=("ambient_temperature",),
        eligibility="recorded compressor ON and OFF runs before the origin; reported as a hindcast only",
        missing_input_limitations=("ineligible without recorded compressor state",
                                   "no forecast: the future compressor state is unknown and no conditional "
                                   "future-state scenario is implemented",
                                   "b is an effective °C/min; a cooling power would need a power measurement"),
        refusal_conditions=("ON-only data: exchange and cooling not separable (INSUFFICIENT_IDENTIFIABILITY)",
                            "b <= 0 (IMPLAUSIBLE_PARAMETERS)", "never forecasts (MISSING_REQUIRED_INPUT)")),
    "multi_node_rc": dict(availability="DEFERRED", required_input_names=(
        "room_temperature", "product_temperature", "wall_temperature")),
    "door_infiltration": dict(availability="DEFERRED", required_input_names=(
        "door_open", "door_area", "relative_humidity")),
    "product_load": dict(availability="DEFERRED", required_input_names=(
        "product_mass", "product_specific_heat", "product_temperature", "product_load")),
    "transmission_load": dict(availability="DEFERRED", required_input_names=(
        "u_value", "envelope_area", "ambient_temperature")),
    "defrost_aware": dict(availability="DEFERRED", required_input_names=("defrost",)),
    "humidity_latent": dict(availability="DEFERRED", required_input_names=("relative_humidity",)),
}
REGISTRY: tuple[ModelSpec, ...] = tuple(replace(m, **_EXT.get(m.name, {})) for m in _BASE)
BY_NAME = {m.name: m for m in REGISTRY}


# ------------------------------------------------------------------ inventory
_MISSING_FAMILIES: tuple[tuple[str, str, str | None], ...] = (
    ("supply/return air temperature", "temperature and environment", "°C"),
    ("product temperature", "temperature and environment", "°C"),
    ("wall / envelope temperature", "temperature and environment", "°C"),
    ("relative humidity (room, adjacent)", "temperature and environment", "%"),
    ("sensor id / location", "temperature and environment", None),
    ("door open/closed state and timestamps", "door and infiltration", None),
    ("door area, type, seal, air curtain", "door and infiltration", None),
    ("room dimensions, surface areas, U-values, insulation", "room and insulation", None),
    ("product type, mass, specific heat, entry temperature, loading times", "product load", None),
    ("cooling capacity or electrical power", "refrigeration", "W"),
    ("setpoint and controller mode", "refrigeration", "°C"),
    ("evaporator / condenser / refrigerant measurements", "refrigeration", None),
    ("fan state", "refrigeration", None),
    ("defrost state and timestamps", "refrigeration", None),
    ("alarms and power interruptions", "refrigeration", None),
)


def audit_inputs(clean: pd.DataFrame, report: DataQualityReport, seg: SegmentationResult | None) -> list[InputItem]:
    """What this log actually provides, per input family (code inspection + data)."""
    ignored = report.input.ignored_columns if report.input else []
    items = [
        InputItem(name="room air temperature", family="temperature and environment", unit="°C",
                  status=InputStatus.AVAILABLE if len(clean) else InputStatus.MISSING,
                  detail=f"{len(clean)} usable readings (Phase 2)"),
        InputItem(name="timestamps", family="temperature and environment", unit=None,
                  status=InputStatus.AVAILABLE if len(clean) else InputStatus.MISSING,
                  detail=f"median interval {report.median_interval_s} s; "
                         f"{report.input.timezone_handling if report.input else 'unknown time basis'}"),
    ]
    tz_unknown = any(i.code == "TIMEZONE_UNKNOWN" for i in report.issues)
    items.append(InputItem(name="time zone", family="temperature and environment", unit=None,
                           status=InputStatus.NOT_VERIFIED if tz_unknown else InputStatus.AVAILABLE,
                           detail="naive local times; a clock change would distort durations" if tz_unknown
                           else "offsets or declared zone (Phase 2)"))
    items.append(InputItem(name="data-quality flags", family="temperature and environment", unit=None,
                           status=InputStatus.AVAILABLE, detail="per-row status and flags from Phase 2"))
    if "ambient_c" not in clean:
        items.append(InputItem(name="ambient temperature", family="temperature and environment", unit="°C",
                               status=InputStatus.MISSING, detail="no ambient_temperature column"))
    else:
        share = float(clean["ambient_c"].notna().mean()) if len(clean) else 0.0
        ok = report.ambient_available
        items.append(InputItem(name="ambient temperature", family="temperature and environment", unit="°C",
                               status=InputStatus.AVAILABLE if ok else InputStatus.INVALID,
                               detail=f"{100 * share:.1f}% of usable rows carry a valid reading"))
        items.append(InputItem(name="what the ambient sensor measures", family="temperature and environment",
                               unit=None, status=InputStatus.NOT_VERIFIED,
                               detail="adjacent space, machine room or outdoors is not recorded; using it as the "
                                      "room's equilibrium is an assumption"))
    comp = (seg.summary.get("compressor") if seg and not seg.refused else None) or ""
    if "compressor_on" not in clean:
        items.append(InputItem(name="compressor state", family="refrigeration", unit="on/off",
                               status=InputStatus.MISSING, detail="no compressor_state column"))
    elif str(comp).startswith("segmented"):
        items.append(InputItem(name="compressor state", family="refrigeration", unit="on/off",
                               status=InputStatus.AVAILABLE, detail=f"Phase 3: {comp}"))
        items.append(InputItem(name="meaning of compressor OFF", family="refrigeration", unit=None,
                               status=InputStatus.NOT_VERIFIED,
                               detail="OFF is taken as 'no active cooling'; evaporator fans, residual coil cold "
                                      "or defrost heaters are not recorded"))
    else:
        items.append(InputItem(name="compressor state", family="refrigeration", unit="on/off",
                               status=InputStatus.INVALID, detail=f"Phase 3: {comp or 'not segmented'}"))
    note = (f"; the file has unmapped columns {ignored}, which are not interpreted" if ignored else "")
    for name, fam, unit in _MISSING_FAMILIES:
        items.append(InputItem(name=name, family=fam, unit=unit, status=InputStatus.MISSING,
                               detail="no input role for it in the ingest contract" + note))
    return items
