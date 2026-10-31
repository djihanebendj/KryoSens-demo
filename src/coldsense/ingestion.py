"""Step 1 (Upload Data / Demo Case) and Step 2 (Data Quality) implementation.

Enforces:
- User-uploaded datasets are strictly tagged EvidenceClass.UNVALIDATED.
- A user's claim that uploaded data is real ('user_claimed_real=True') is stored separately
  and NEVER upgrades the evidence_class from UNVALIDATED.
- Built-in demo datasets are explicitly tagged EvidenceClass.SIMULATED.
"""

from __future__ import annotations

import csv
import io
import json
import math
import statistics
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from coldsense.models import (
    CaseRecord,
    DataQualityIssue,
    DataQualityReport,
    EvidenceClass,
    TemperaturePoint,
    WorkflowStep,
)

# Physical sanity bounds for commercial/industrial cold rooms (-60 C to +60 C)
MIN_PHYSICAL_TEMP_C = -60.0
MAX_PHYSICAL_TEMP_C = 60.0


def _parse_iso_or_epoch(raw_ts: Any, index: int, default_step_sec: float = 300.0) -> Tuple[float, str]:
    """Parse an ISO-8601 string or numeric epoch seconds into (timestamp_sec, iso_string)."""
    if isinstance(raw_ts, (int, float)):
        epoch = float(raw_ts)
        iso = datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()
        return epoch, iso

    s = str(raw_ts).strip()
    if not s:
        raise ValueError(f"Empty timestamp at row {index}")

    try:
        epoch = float(s)
        iso = datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()
        return epoch, iso
    except ValueError:
        pass

    normalized = s.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp(), dt.isoformat()


def _parse_optional_bool(val: Any) -> Optional[bool]:
    if val is None or str(val).strip() == "":
        return None
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "on", "open", "active"):
        return True
    if s in ("0", "false", "no", "off", "closed", "inactive"):
        return False
    return None


def _parse_optional_float(val: Any) -> Optional[float]:
    if val is None or str(val).strip() == "":
        return None
    try:
        f = float(str(val).strip())
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except ValueError:
        return None


def parse_temperature_log(
    raw_content: str,
    fmt: str = "csv",
) -> Tuple[List[TemperaturePoint], int, int]:
    """Parse CSV or JSON log text into TemperaturePoints.

    Returns:
        (valid_points, total_raw_rows, missing_value_rows)
    """
    points: List[TemperaturePoint] = []
    total_rows = 0
    missing_rows = 0

    if fmt.lower() == "json":
        data = json.loads(raw_content)
        rows = data if isinstance(data, list) else data.get("readings", data.get("data", []))
        for idx, row in enumerate(rows):
            total_rows += 1
            ts_raw = row.get("timestamp") or row.get("time") or row.get("timestamp_sec")
            temp_raw = row.get("temp_c") if "temp_c" in row else row.get("temperature")
            if ts_raw is None or temp_raw is None or str(temp_raw).strip() == "":
                missing_rows += 1
                continue
            try:
                ts_sec, iso_str = _parse_iso_or_epoch(ts_raw, idx)
                temp_c = float(temp_raw)
                if math.isnan(temp_c) or math.isinf(temp_c):
                    missing_rows += 1
                    continue
            except (ValueError, TypeError):
                missing_rows += 1
                continue

            points.append(
                TemperaturePoint(
                    timestamp_sec=ts_sec,
                    iso_timestamp=iso_str,
                    temp_c=temp_c,
                    ambient_temp_c=_parse_optional_float(row.get("ambient_temp_c")),
                    door_open=_parse_optional_bool(row.get("door_open")),
                    power_on=_parse_optional_bool(row.get("power_on")),
                    defrost_active=_parse_optional_bool(row.get("defrost_active")),
                )
            )
    else:
        reader = csv.DictReader(io.StringIO(raw_content.strip()))
        for idx, row in enumerate(reader):
            total_rows += 1
            ts_raw = (
                row.get("timestamp")
                or row.get("time")
                or row.get("timestamp_sec")
                or row.get("datetime")
            )
            temp_raw = (
                row.get("temp_c")
                if row.get("temp_c") not in (None, "")
                else (row.get("temperature") or row.get("temp"))
            )
            if ts_raw is None or temp_raw is None or str(temp_raw).strip() == "":
                missing_rows += 1
                continue
            try:
                ts_sec, iso_str = _parse_iso_or_epoch(ts_raw, idx)
                temp_c = float(str(temp_raw).strip())
                if math.isnan(temp_c) or math.isinf(temp_c):
                    missing_rows += 1
                    continue
            except (ValueError, TypeError):
                missing_rows += 1
                continue

            points.append(
                TemperaturePoint(
                    timestamp_sec=ts_sec,
                    iso_timestamp=iso_str,
                    temp_c=temp_c,
                    ambient_temp_c=_parse_optional_float(row.get("ambient_temp_c")),
                    door_open=_parse_optional_bool(row.get("door_open")),
                    power_on=_parse_optional_bool(row.get("power_on")),
                    defrost_active=_parse_optional_bool(row.get("defrost_active")),
                )
            )

    # Sort chronologically
    points.sort(key=lambda p: p.timestamp_sec)
    return points, total_rows, missing_rows


def create_uploaded_case(
    owner_session_id: str,
    raw_content: str,
    fmt: str = "csv",
    title: str = "Uploaded Cold-Room Log",
    user_claimed_real: bool = False,
    critical_threshold_c: float = 8.0,
    ttl_seconds: float = 3600.0,
) -> Tuple[CaseRecord, int, int]:
    """Step 1: Create a private Case from user-uploaded data.

    Always assigns EvidenceClass.UNVALIDATED regardless of user_claimed_real.
    """
    points, total_rows, missing_rows = parse_temperature_log(raw_content, fmt=fmt)
    now = time.time()
    case = CaseRecord(
        case_id=f"case-{uuid.uuid4().hex[:12]}",
        owner_session_id=owner_session_id,
        title=title,
        evidence_class=EvidenceClass.UNVALIDATED,
        user_claimed_real=bool(user_claimed_real),
        created_at_epoch=now,
        expires_at_epoch=now + ttl_seconds,
        critical_threshold_c=critical_threshold_c,
        points=points,
        unlocked_step=int(WorkflowStep.DATA_QUALITY),
    )
    return case, total_rows, missing_rows


def generate_simulated_demo_case(
    owner_session_id: str,
    scenario: str = "power_cut",
    ttl_seconds: float = 3600.0,
) -> CaseRecord:
    """Step 1 Demo Path: Create a explicitly labelled SIMULATED cold-room dataset.

    Generates a realistic 6-hour log (5-minute cadence = 73 readings) including:
    - Normal cooling cycling around 2.0 C (00:00 to 01:30)
    - Brief door opening spike at 01:00
    - Unpowered thermal drift (power cut) starting at 01:30 where temperature drifts
      toward ambient (22.0 C) with thermal time constant tau = 4.0 hours (14,400 sec)
    - Cooling recovery after power restoration at 04:30
    """
    now = time.time()
    base_dt = datetime(2026, 10, 25, 8, 0, 0, tzinfo=timezone.utc)
    cadence_sec = 300.0  # 5 minutes
    ambient_c = 22.0
    tau_sec = 14400.0  # 4 hours
    points: List[TemperaturePoint] = []

    if scenario == "corrupted_unreliable":
        # Sparse, noisy dataset designed to trigger the Reliability Gate refusal
        for i in range(4):
            dt = base_dt + timedelta(seconds=i * 1800)
            # Non-physical oscillating jumps during an alleged outage
            temp = 2.0 if i % 2 == 0 else -4.5
            points.append(
                TemperaturePoint(
                    timestamp_sec=dt.timestamp(),
                    iso_timestamp=dt.isoformat(),
                    temp_c=temp,
                    ambient_temp_c=ambient_c,
                    door_open=False,
                    power_on=False,
                    defrost_active=False,
                )
            )
    else:
        for i in range(73):
            elapsed_sec = i * cadence_sec
            dt = base_dt + timedelta(seconds=elapsed_sec)
            #Deterministic pseudo-noise for reproducibility without randomness
            micro_ripple = 0.06 * math.sin(i * 1.1)

            if i < 18:
                # 00:00 - 01:25: Normal operation around 2.0 C, brief door open at i=10..12
                if 10 <= i <= 12:
                    temp = 2.1 + 0.55 * (i - 9) + micro_ripple
                    door = True
                else:
                    temp = 2.0 + 0.15 * math.sin(i * 0.7) + micro_ripple
                    door = False
                power = True
                defrost = False
            elif 18 <= i <= 54:
                # 01:30 - 04:30 (3 hours = 37 samples): Power cut thermal drift
                t_drift = (i - 18) * cadence_sec
                t0 = 2.1
                temp = ambient_c - (ambient_c - t0) * math.exp(-t_drift / tau_sec) + micro_ripple
                door = False
                power = False
                defrost = False
            else:
                # 04:35 - 06:00: Power restored, active cooling pull-down back toward 2.0 C
                t_rec = (i - 54) * cadence_sec
                t_peak = ambient_c - (ambient_c - 2.1) * math.exp(-(36 * cadence_sec) / tau_sec)
                temp = 2.0 + (t_peak - 2.0) * math.exp(-t_rec / 3600.0) + micro_ripple
                door = False
                power = True
                defrost = False

            points.append(
                TemperaturePoint(
                    timestamp_sec=dt.timestamp(),
                    iso_timestamp=dt.isoformat(),
                    temp_c=round(temp, 3),
                    ambient_temp_c=ambient_c,
                    door_open=door,
                    power_on=power,
                    defrost_active=defrost,
                )
            )

    return CaseRecord(
        case_id=f"sim-{uuid.uuid4().hex[:12]}",
        owner_session_id=owner_session_id,
        title=f"SIMULATED Cold-Room Demo ({scenario})",
        evidence_class=EvidenceClass.SIMULATED,
        user_claimed_real=False,
        created_at_epoch=now,
        expires_at_epoch=now + ttl_seconds,
        critical_threshold_c=8.0,
        points=points,
        unlocked_step=int(WorkflowStep.DATA_QUALITY),
    )


def evaluate_data_quality(
    case: CaseRecord,
    total_raw_rows: Optional[int] = None,
    missing_value_rows: int = 0,
) -> DataQualityReport:
    """Step 2: Inspect the uploaded or simulated log for missing data, gaps, and anomalies.

    Unlocks Step 3 (EVENT_ANALYSIS) if the log is usable.
    """
    pts = case.points
    valid_rows = len(pts)
    raw_rows = total_raw_rows if total_raw_rows is not None else (valid_rows + missing_value_rows)
    issues: List[DataQualityIssue] = []

    if valid_rows < 3:
        issues.append(
            DataQualityIssue(
                code="TOO_FEW_READINGS",
                severity="CRITICAL",
                message=f"Log contains only {valid_rows} valid readings (minimum 3 required).",
            )
        )
        report = DataQualityReport(
            case_id=case.case_id,
            evidence_class=case.evidence_class,
            total_rows=raw_rows,
            valid_rows=valid_rows,
            duplicate_timestamps=0,
            missing_value_rows=missing_value_rows,
            gap_count=0,
            max_gap_sec=0.0,
            median_cadence_sec=0.0,
            cadence_jitter_sec=0.0,
            out_of_range_count=0,
            flatline_segments=0,
            completeness_ratio=0.0 if raw_rows == 0 else round(valid_rows / raw_rows, 4),
            is_usable_for_analysis=False,
            issues=issues,
        )
        case.data_quality = report
        return report

    # Compute time deltas, duplicates, and gaps
    deltas: List[float] = []
    duplicate_count = 0
    for i in range(1, valid_rows):
        dt = pts[i].timestamp_sec - pts[i - 1].timestamp_sec
        if dt <= 0:
            duplicate_count += 1
            issues.append(
                DataQualityIssue(
                    code="DUPLICATE_TIMESTAMP",
                    severity="WARNING",
                    message=f"Duplicate or non-increasing timestamp at index {i} ({pts[i].iso_timestamp}).",
                    start_index=i - 1,
                    end_index=i,
                )
            )
        else:
            deltas.append(dt)

    median_cadence = statistics.median(deltas) if deltas else 0.0
    jitter = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0

    # Identify significant gaps (> 2.5x median cadence)
    gap_count = 0
    max_gap = max(deltas) if deltas else 0.0
    gap_threshold = max(median_cadence * 2.5, 1.0)
    for i in range(1, valid_rows):
        dt = pts[i].timestamp_sec - pts[i - 1].timestamp_sec
        if dt > gap_threshold:
            gap_count += 1
            issues.append(
                DataQualityIssue(
                    code="SAMPLING_GAP",
                    severity="WARNING" if dt < median_cadence * 6 else "CRITICAL",
                    message=f"Sampling gap of {dt:.0f}s between index {i-1} and {i} (expected ~{median_cadence:.0f}s).",
                    start_index=i - 1,
                    end_index=i,
                )
            )

    # Out-of-bounds physical temperatures
    out_of_range = 0
    for i, p in enumerate(pts):
        if p.temp_c < MIN_PHYSICAL_TEMP_C or p.temp_c > MAX_PHYSICAL_TEMP_C:
            out_of_range += 1
            issues.append(
                DataQualityIssue(
                    code="OUT_OF_PHYSICAL_RANGE",
                    severity="CRITICAL",
                    message=f"Reading {p.temp_c:.2f} C at index {i} is outside physical bounds [{MIN_PHYSICAL_TEMP_C}, {MAX_PHYSICAL_TEMP_C}] C.",
                    start_index=i,
                    end_index=i,
                )
            )

    # Detect suspicious sensor flatlines (>= 8 identical consecutive readings to 4 decimals)
    flatline_segments = 0
    run_len = 1
    for i in range(1, valid_rows):
        if abs(pts[i].temp_c - pts[i - 1].temp_c) < 1e-6:
            run_len += 1
            if run_len == 8:
                flatline_segments += 1
                issues.append(
                    DataQualityIssue(
                        code="SENSOR_FLATLINE",
                        severity="WARNING",
                        message=f"Sensor flatline detected starting at index {i - 7} ({pts[i].temp_c:.2f} C).",
                        start_index=i - 7,
                        end_index=i,
                    )
                )
        else:
            run_len = 1

    if missing_value_rows > 0:
        issues.append(
            DataQualityIssue(
                code="MISSING_VALUES",
                severity="WARNING",
                message=f"Dropped {missing_value_rows} rows with missing or malformed timestamps/temperatures.",
            )
        )

    total_span_sec = pts[-1].timestamp_sec - pts[0].timestamp_sec
    expected_samples = (
        int(round(total_span_sec / median_cadence)) + 1 if median_cadence > 0 else valid_rows
    )
    completeness = min(1.0, valid_rows / max(expected_samples, raw_rows, 1))
    usable = valid_rows >= 4 and out_of_range == 0

    report = DataQualityReport(
        case_id=case.case_id,
        evidence_class=case.evidence_class,
        total_rows=raw_rows,
        valid_rows=valid_rows,
        duplicate_timestamps=duplicate_count,
        missing_value_rows=missing_value_rows,
        gap_count=gap_count,
        max_gap_sec=round(max_gap, 2),
        median_cadence_sec=round(median_cadence, 2),
        cadence_jitter_sec=round(jitter, 2),
        out_of_range_count=out_of_range,
        flatline_segments=flatline_segments,
        completeness_ratio=round(completeness, 4),
        is_usable_for_analysis=usable,
        issues=issues,
    )
    case.data_quality = report
    if usable and case.unlocked_step < int(WorkflowStep.EVENT_ANALYSIS):
        case.unlocked_step = int(WorkflowStep.EVENT_ANALYSIS)
    return report
