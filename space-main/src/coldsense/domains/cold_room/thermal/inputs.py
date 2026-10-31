"""Input-capability layer of the thermal engine (Phase 4 follow-up).

Four input families are kept apart, because they mean different things:

* MEASUREMENT: time series read from the log through the accepted Phase 2
  contract (room temperature, timestamps, optional ambient and compressor state).
* EVENT: operational records (door, defrost, power interruption, ...) supplied as
  a separate event log through the Phase 4 adapter below. Phase 2 is unchanged.
* METADATA: static room and equipment facts supplied as a separate file. They are
  validated, kept and disclosed; no implemented model computes with them yet.
* SCENARIO: what the user states about the horizon or the past (passive_since,
  assumed equilibrium, future ambient). Always ASSUMED, never measured.

Nothing here invents, fills or converts a value. Units must be given exactly as
the catalogue states them (no conversion); timestamps must be naive, in the
log's own time basis; every value carries a provenance.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .registry import FO_ACTIVE, FO_FITTED, FO_MEASURED, FO_VARYING, PERSISTENCE, RECENT_SLOPE


class InputFamily(str, Enum):
    MEASUREMENT = "MEASUREMENT"
    EVENT = "EVENT"
    METADATA = "METADATA"
    SCENARIO = "SCENARIO"


class InputSource(str, Enum):
    LOG_COLUMN = "LOG_COLUMN"            # a column of the log, through Phase 2 ingest (unchanged)
    EVENT_LOG = "EVENT_LOG"              # separate event file, Phase 4 adapter
    METADATA_FILE = "METADATA_FILE"      # separate metadata file, Phase 4 adapter
    REQUEST = "REQUEST"                  # the forecast request (user statement)
    NOT_SUPPORTED = "NOT_SUPPORTED"      # no way to supply it in this version


@dataclass(frozen=True)
class InputSpec:
    name: str                      # canonical name
    family: InputFamily
    kind: str                      # e.g. "float time series", "interval event", "static value"
    unit: str | None
    source: InputSource
    alignment: str
    provenance: str
    required: bool
    used_by: tuple[str, ...]       # implemented models or engine capabilities that use it
    enables_deferred: tuple[str, ...]
    when_missing: str
    when_invalid: str


_TS = "on the log's timestamps (same rows)"
CATALOGUE: tuple[InputSpec, ...] = (
    # ---------------------------------------------------------------- A. measurements (log columns)
    InputSpec("room_temperature", InputFamily.MEASUREMENT, "float time series", "°C", InputSource.LOG_COLUMN, _TS,
              "OBSERVED", True, (PERSISTENCE, RECENT_SLOPE, FO_FITTED, FO_MEASURED, FO_VARYING, FO_ACTIVE), (),
              "the log cannot be read (Phase 2 IngestError)", "row EXCLUDED by Phase 2 validation, never fitted"),
    InputSpec("timestamp", InputFamily.MEASUREMENT, "datetime", None, InputSource.LOG_COLUMN,
              "naive; UTC-normalised or local time of unknown zone (Phase 2)", "OBSERVED", True, ("all",), (),
              "the log cannot be read", "row EXCLUDED by Phase 2 validation"),
    InputSpec("ambient_temperature", InputFamily.MEASUREMENT, "float time series", "°C", InputSource.LOG_COLUMN,
              _TS + "; constant-ambient variant needs >= 90 % of a segment's rows and a range <= 1 °C",
              "OBSERVED", False, (FO_MEASURED, FO_VARYING, FO_ACTIVE), (),
              "measured- and varying-ambient variants unavailable; equilibrium only fitted or assumed",
              "flagged by Phase 2 and set to missing"),
    InputSpec("compressor_state", InputFamily.MEASUREMENT, "on/off time series", "on/off", InputSource.LOG_COLUMN,
              _TS + "; Phase 3 needs >= 90 % valid states", "OBSERVED", False, ("regime evidence", FO_ACTIVE), (),
              "no recorded passive periods; needs passive_since or power-interruption events",
              "invalid states set to missing (COMPRESSOR_UNKNOWN); < 90 % valid -> not used"),
    *(InputSpec(n, InputFamily.MEASUREMENT, "float time series", u, InputSource.NOT_SUPPORTED, _TS, "OBSERVED",
                False, (), d, w, "no ingest role in this version")
      for n, u, d, w in (
          ("supply_air_temperature", "°C", ("multi_node_rc",), "coil behaviour cannot be separated from room air"),
          ("return_air_temperature", "°C", ("multi_node_rc",), "coil behaviour cannot be separated from room air"),
          ("product_temperature", "°C", ("multi_node_rc", "product_load"),
           "product thermal mass is absorbed into the effective k and E"),
          ("wall_temperature", "°C", ("multi_node_rc",), "envelope storage is absorbed into the effective k and E"),
          ("relative_humidity", "%", ("humidity_latent",), "latent loads and frosting cannot be assessed"),
          ("fan_state", "on/off", (), "fan heat and air mixing are not separated"),
          ("cooling_power", "W", (), "the active-cooling term b stays an effective °C/min, never a power"),
          ("evaporator_condenser_measurements", "°C / bar", (), "refrigeration-side behaviour is not observed"),
      )),
    # ---------------------------------------------------------------- B. operational events (event log adapter)
    InputSpec("door_open", InputFamily.EVENT, "interval event (start, end)", None, InputSource.EVENT_LOG,
              "naive timestamps in the log's time basis", "OBSERVED (event log)", False,
              ("disturbance marking: readings inside the interval are not fitted or used as truth",),
              ("door_infiltration",), "door effects cannot be separated; disturbances stay unexplained",
              "event rejected with its line number"),
    InputSpec("defrost", InputFamily.EVENT, "interval event (start, end)", None, InputSource.EVENT_LOG,
              "naive timestamps in the log's time basis", "OBSERVED (event log)", False,
              ("disturbance marking",), ("defrost_aware",),
              "defrost heating cannot be separated; it appears as an unexplained disturbance",
              "event rejected with its line number"),
    InputSpec("power_interruption", InputFamily.EVENT, "interval event (start, end or ongoing)", None,
              InputSource.EVENT_LOG, "naive timestamps in the log's time basis", "OBSERVED (event log)", False,
              ("regime evidence: a recorded period without active cooling",), (),
              "passive periods only from compressor state or passive_since", "event rejected with its line number"),
    InputSpec("product_load", InputFamily.EVENT, "point or interval event", None, InputSource.EVENT_LOG,
              "naive timestamps in the log's time basis", "OBSERVED (event log)", False,
              ("segment boundary; readings inside an interval are not fitted",), ("product_load",),
              "a load changes thermal mass unseen: the fit may span it", "event rejected with its line number"),
    InputSpec("setpoint_change", InputFamily.EVENT, "point event", None, InputSource.EVENT_LOG,
              "naive timestamps in the log's time basis", "OBSERVED (event log)", False, ("segment boundary",), (),
              "a change of control regime is not marked", "event rejected with its line number"),
    InputSpec("maintenance", InputFamily.EVENT, "interval event (start, end)", None, InputSource.EVENT_LOG,
              "naive timestamps in the log's time basis", "OBSERVED (event log)", False, ("disturbance marking",), (),
              "interventions are not marked", "event rejected with its line number"),
    InputSpec("equipment_fault", InputFamily.EVENT, "point or interval event", None, InputSource.EVENT_LOG,
              "naive timestamps in the log's time basis", "OBSERVED (event log)", False,
              ("segment boundary / disturbance marking",), (), "faults are not marked",
              "event rejected with its line number"),
    InputSpec("compressor_start_stop_events", InputFamily.EVENT, "point events", None, InputSource.NOT_SUPPORTED,
              "-", "-", False, (), (), "use the compressor_state column (Phase 2/3); events are not converted",
              "not accepted"),
    # ---------------------------------------------------------------- C. metadata (metadata adapter)
    *(InputSpec(n, InputFamily.METADATA, "static value", u, InputSource.METADATA_FILE, "applies to the whole log",
                "user-supplied with a stated provenance (NOT_VERIFIED)", False, ("disclosure only",), d,
                "not needed by any implemented model", f"rejected: value must be finite and > 0, unit exactly {u}")
      for n, u, d in (
          ("room_volume", "m3", ()), ("floor_area", "m2", ()), ("envelope_area", "m2", ("transmission_load",)),
          ("door_area", "m2", ("door_infiltration",)), ("insulation_thickness", "m", ("transmission_load",)),
          ("u_value", "W/m2K", ("transmission_load",)), ("product_mass", "kg", ("product_load",)),
          ("product_specific_heat", "J/kgK", ("product_load",)), ("refrigeration_capacity", "W", ()),
      )),
    InputSpec("sensors", InputFamily.METADATA, "list of sensor descriptions", None, InputSource.METADATA_FILE,
              "sensor.column matches a header of the log", "user-supplied (NOT_VERIFIED)", False,
              ("disclosure of the analysed sensor and of the others",), (),
              "the analysed sensor's location is unknown", "rejected: duplicate ids/columns or invalid values"),
    # ---------------------------------------------------------------- D. scenario (request)
    InputSpec("passive_since", InputFamily.SCENARIO, "datetime", None, InputSource.REQUEST,
              "naive, log time basis", "ASSUMED", False, ("regime evidence (declared)",), (),
              "-", "request rejected (INVALID_REQUEST)"),
    InputSpec("assumed_equilibrium", InputFamily.SCENARIO, "float", "°C", InputSource.REQUEST, "-", "ASSUMED", False,
              ("first_order_assumed_equilibrium",), (), "that variant is unavailable", "request rejected"),
    InputSpec("future_ambient", InputFamily.SCENARIO, "float (constant over the horizon)", "°C", InputSource.REQUEST,
              "-", "ASSUMED", False, (FO_VARYING, FO_MEASURED), (),
              "varying-ambient forecasts refused; measured-ambient holds the last steady reading (ASSUMED)",
              "request rejected"),
)
BY_NAME = {s.name: s for s in CATALOGUE}


# ====================================================================== event log adapter
class EventKind(str, Enum):
    DOOR_OPEN = "DOOR_OPEN"
    DEFROST = "DEFROST"
    POWER_INTERRUPTION = "POWER_INTERRUPTION"
    PRODUCT_LOAD = "PRODUCT_LOAD"
    SETPOINT_CHANGE = "SETPOINT_CHANGE"
    MAINTENANCE = "MAINTENANCE"
    EQUIPMENT_FAULT = "EQUIPMENT_FAULT"


class EventProvenance(str, Enum):
    RECORDED_BY_SYSTEM = "RECORDED_BY_SYSTEM"   # controller, door contact, BMS export
    MANUAL_LOG = "MANUAL_LOG"                   # written down by a person


END_REQUIRED = {EventKind.DOOR_OPEN, EventKind.DEFROST, EventKind.MAINTENANCE}
END_FORBIDDEN = {EventKind.SETPOINT_CHANGE}
EXCLUDING = {EventKind.DOOR_OPEN, EventKind.DEFROST, EventKind.MAINTENANCE, EventKind.PRODUCT_LOAD,
             EventKind.EQUIPMENT_FAULT}           # readings inside the interval are not fitted / not used as truth


def _naive(v: datetime | None, name: str) -> datetime | None:
    if v is None:
        return v
    if v != v:                                     # pandas NaT
        raise ValueError(f"{name} is not a valid time")
    if v.tzinfo is not None:
        raise ValueError(f"{name} carries a time zone; give it in the log's own time basis (naive)")
    return v


class OperationalEvent(BaseModel):
    """One operational record. Point events have no end; interval events have end >= start."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: EventKind
    start: datetime
    end: datetime | None = None
    provenance: EventProvenance
    source: str | None = Field(None, max_length=200)
    note: str | None = Field(None, max_length=500)

    @model_validator(mode="after")
    def _check(self) -> OperationalEvent:
        _naive(self.start, "start")
        _naive(self.end, "end")
        if self.kind in END_REQUIRED and self.end is None:
            raise ValueError(f"{self.kind.value} needs an end time")
        if self.kind in END_FORBIDDEN and self.end is not None:
            raise ValueError(f"{self.kind.value} is a point event and takes no end time")
        if self.end is not None and self.end < self.start:
            raise ValueError("end is before start")
        return self


EVENT_COLUMNS = ("kind", "start", "end", "provenance", "source", "note")


def parse_events_csv(text: str) -> list[OperationalEvent]:
    """Parse an event log (CSV text): header kind,start,end,provenance[,source,note]; ISO times.
    Raises ValueError naming the line of any invalid row; nothing is skipped silently."""
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise ValueError("event log is empty")
    header = [h.strip().lower() for h in rows[0]]
    missing = [c for c in ("kind", "start", "provenance") if c not in header]
    unknown = [c for c in header if c not in EVENT_COLUMNS]
    if missing or unknown:
        raise ValueError(f"event log header must be {','.join(EVENT_COLUMNS)} (kind, start, provenance required); "
                         f"missing {missing}, unknown {unknown}")
    out = []
    for n, r in enumerate(rows[1:], start=2):
        if not any(c.strip() for c in r):
            continue
        if len(r) != len(header):
            raise ValueError(f"event log line {n}: {len(r)} fields, header has {len(header)}")
        rec = {h: (v.strip() or None) for h, v in zip(header, r, strict=True)}
        try:
            out.append(OperationalEvent(**rec))
        except ValidationError as e:
            msg = "; ".join(f"{'.'.join(map(str, x['loc'])) or 'event'}: {x['msg'].removeprefix('Value error, ')}"
                            for x in e.errors())
            raise ValueError(f"event log line {n}: {msg}") from None
    return out


# ====================================================================== metadata adapter
class MetadataProvenance(str, Enum):
    MEASURED_ON_SITE = "MEASURED_ON_SITE"
    MANUFACTURER_SPECIFICATION = "MANUFACTURER_SPECIFICATION"
    DESIGN_DOCUMENT = "DESIGN_DOCUMENT"
    USER_ESTIMATE = "USER_ESTIMATE"


class MetaValue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    value: float = Field(gt=0)
    unit: str
    provenance: MetadataProvenance
    source: str | None = Field(None, max_length=200)


class SensorInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    id: str = Field(min_length=1, max_length=100)
    column: str | None = Field(None, description="header of this sensor's column in the log, if it is there")
    location: str | None = Field(None, max_length=200)
    height_m: float | None = Field(None, ge=0)
    calibration_date: date | None = None
    calibration_offset_c: float | None = Field(None, description="documented only; readings are never corrected")
    provenance: MetadataProvenance


METADATA_UNITS = {s.name: s.unit for s in CATALOGUE if s.family == InputFamily.METADATA and s.unit}


class RoomMetadata(BaseModel):
    """Optional static facts. Validated and disclosed; not used in any calculation in this version."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    room_volume: MetaValue | None = None
    floor_area: MetaValue | None = None
    envelope_area: MetaValue | None = None
    door_area: MetaValue | None = None
    insulation_thickness: MetaValue | None = None
    u_value: MetaValue | None = None
    product_mass: MetaValue | None = None
    product_specific_heat: MetaValue | None = None
    refrigeration_capacity: MetaValue | None = None
    sensors: list[SensorInfo] = Field(default_factory=list)
    notes: str | None = Field(None, max_length=2000)

    @field_validator("sensors")
    @classmethod
    def _unique(cls, v: list[SensorInfo]) -> list[SensorInfo]:
        ids = [s.id for s in v]
        cols = [s.column for s in v if s.column]
        if len(set(ids)) != len(ids) or len(set(cols)) != len(cols):
            raise ValueError("sensor ids and columns must be unique")
        return v

    @model_validator(mode="after")
    def _units(self) -> RoomMetadata:
        for name, unit in METADATA_UNITS.items():
            mv = getattr(self, name)
            if mv is not None and mv.unit != unit:
                raise ValueError(f"{name}: unit {mv.unit!r} is not accepted; give {unit} (no conversion is made)")
        return self

    def supplied(self) -> list[str]:
        return [n for n in METADATA_UNITS if getattr(self, n) is not None] + (["sensors"] if self.sensors else [])


def parse_metadata_json(text: str) -> RoomMetadata:
    """Parse room metadata (JSON text). Raises ValueError with every invalid field named."""
    try:
        return RoomMetadata.model_validate(json.loads(text))
    except json.JSONDecodeError as e:
        raise ValueError(f"metadata is not valid JSON: {e}") from None
    except ValidationError as e:
        msg = "; ".join(f"{'.'.join(map(str, x['loc'])) or 'metadata'}: {x['msg'].removeprefix('Value error, ')}"
                        for x in e.errors())
        raise ValueError(f"invalid metadata: {msg}") from None
