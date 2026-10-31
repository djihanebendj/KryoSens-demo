"""Step 3: Temperature Timeline & Thermal Event Detection.

Identifies operational anomalies and thermal regimes in cold-room logs:
- POWER_CUT: Unpowered warming drift (via power_on=False telemetry or sustained monotonic warming).
- DOOR_OPEN: Rapid, short-duration temperature spike coinciding with door_open or steep transient slope.
- DEFROST_CYCLE: Periodic warming excursion followed by rapid active pull-down.
- COOLING_RECOVERY: Active pull-down segment following a warming excursion.
- SETPOINT_EXCURSION: Period where cold-room temperature exceeds the critical threshold.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from coldsense.models import (
    CaseRecord,
    EventType,
    TemperatureEvent,
    TemperaturePoint,
    WorkflowStep,
)


def _build_event(
    pts: List[TemperaturePoint],
    case: CaseRecord,
    event_idx: int,
    event_type: EventType,
    start_i: int,
    end_i: int,
    description: str,
) -> TemperatureEvent:
    segment = pts[start_i : end_i + 1]
    start_p = pts[start_i]
    end_p = pts[end_i]
    duration = max(1.0, end_p.timestamp_sec - start_p.timestamp_sec)
    peak_c = max(p.temp_c for p in segment)
    delta_c = end_p.temp_c - start_p.temp_c
    rate_c_hr = (delta_c / duration) * 3600.0

    return TemperatureEvent(
        event_id=f"evt-{event_idx:03d}-{event_type.value.lower()}",
        event_type=event_type,
        evidence_class=case.evidence_class,
        start_index=start_i,
        end_index=end_i,
        start_timestamp_sec=start_p.timestamp_sec,
        end_timestamp_sec=end_p.timestamp_sec,
        start_iso=start_p.iso_timestamp,
        end_iso=end_p.iso_timestamp,
        duration_sec=round(duration, 2),
        start_temp_c=round(start_p.temp_c, 3),
        end_temp_c=round(end_p.temp_c, 3),
        peak_temp_c=round(peak_c, 3),
        temp_delta_c=round(delta_c, 3),
        drift_rate_c_per_hour=round(rate_c_hr, 3),
        description=description,
    )


def _find_boolean_runs(
    pts: List[TemperaturePoint], predicate
) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    in_run = False
    start_i = 0
    for i, p in enumerate(pts):
        if predicate(p):
            if not in_run:
                in_run = True
                start_i = i
        else:
            if in_run:
                runs.append((start_i, i - 1))
                in_run = False
    if in_run:
        runs.append((start_i, len(pts) - 1))
    return runs


def detect_temperature_events(case: CaseRecord) -> List[TemperatureEvent]:
    """Step 3: Detect temperature events in the case log and unlock Step 4 (PREDICTION)."""
    if case.data_quality is None or not case.data_quality.is_usable_for_analysis:
        raise ValueError("Cannot run event analysis before Data Quality check passes.")

    pts = case.points
    n = len(pts)
    events: List[TemperatureEvent] = []
    evt_counter = 1

    # 1. Explicit telemetry-flagged power cuts (power_on is False)
    power_cut_runs = _find_boolean_runs(pts, lambda p: p.power_on is False)
    covered_power_indices = set()
    for s_i, e_i in power_cut_runs:
        if e_i > s_i:
            for k in range(s_i, e_i + 1):
                covered_power_indices.add(k)
            events.append(
                _build_event(
                    pts,
                    case,
                    evt_counter,
                    EventType.POWER_CUT,
                    s_i,
                    e_i,
                    "Unpowered thermal drift detected (power loss telemetry active).",
                )
            )
            evt_counter += 1

    # 2. Sustained warming drift detection when power_on channel is absent
    if not power_cut_runs and n >= 5:
        run_start: Optional[int] = None
        for i in range(1, n):
            dt_temp = pts[i].temp_c - pts[i - 1].temp_c
            if dt_temp > -0.05:  # non-cooling / warming step
                if run_start is None:
                    run_start = i - 1
            else:
                if run_start is not None and (i - 1 - run_start) >= 4:
                    total_rise = pts[i - 1].temp_c - pts[run_start].temp_c
                    if total_rise >= 1.5:
                        events.append(
                            _build_event(
                                pts,
                                case,
                                evt_counter,
                                EventType.POWER_CUT,
                                run_start,
                                i - 1,
                                "Sustained thermal warming drift consistent with cooling loss or power cut.",
                            )
                        )
                        for k in range(run_start, i):
                            covered_power_indices.add(k)
                        evt_counter += 1
                run_start = None
        if run_start is not None and (n - 1 - run_start) >= 4:
            total_rise = pts[n - 1].temp_c - pts[run_start].temp_c
            if total_rise >= 1.5:
                events.append(
                    _build_event(
                        pts,
                        case,
                        evt_counter,
                        EventType.POWER_CUT,
                        run_start,
                        n - 1,
                        "Sustained thermal warming drift consistent with cooling loss or power cut.",
                    )
                )
                for k in range(run_start, n):
                    covered_power_indices.add(k)
                evt_counter += 1

    # 3. Door open events
    door_runs = _find_boolean_runs(pts, lambda p: p.door_open is True)
    for s_i, e_i in door_runs:
        end_bound = min(n - 1, max(s_i + 1, e_i))
        events.append(
            _build_event(
                pts,
                case,
                evt_counter,
                EventType.DOOR_OPEN,
                s_i,
                end_bound,
                "Door open transient thermal disturbance.",
            )
        )
        evt_counter += 1

    # 4. Defrost cycles
    defrost_runs = _find_boolean_runs(pts, lambda p: p.defrost_active is True)
    for s_i, e_i in defrost_runs:
        end_bound = min(n - 1, max(s_i + 1, e_i))
        events.append(
            _build_event(
                pts,
                case,
                evt_counter,
                EventType.DEFROST_CYCLE,
                s_i,
                end_bound,
                "Scheduled or active evaporator defrost cycle.",
            )
        )
        evt_counter += 1

    # 5. Cooling recovery segments immediately following a power cut
    for s_i, e_i in power_cut_runs:
        if e_i + 2 < n:
            rec_start = e_i
            rec_end = e_i + 1
            while rec_end + 1 < n and pts[rec_end + 1].temp_c <= pts[rec_end].temp_c + 0.15:
                rec_end += 1
            if pts[rec_start].temp_c - pts[rec_end].temp_c >= 1.0:
                events.append(
                    _build_event(
                        pts,
                        case,
                        evt_counter,
                        EventType.COOLING_RECOVERY,
                        rec_start,
                        rec_end,
                        "Active refrigeration pull-down recovery following thermal excursion.",
                    )
                )
                evt_counter += 1

    # 6. Critical setpoint excursion windows
    excursion_runs = _find_boolean_runs(pts, lambda p: p.temp_c >= case.critical_threshold_c)
    for s_i, e_i in excursion_runs:
        end_bound = min(n - 1, max(s_i + 1, e_i))
        events.append(
            _build_event(
                pts,
                case,
                evt_counter,
                EventType.SETPOINT_EXCURSION,
                s_i,
                end_bound,
                f"Cold-room temperature exceeded critical threshold ({case.critical_threshold_c:.1f} C).",
            )
        )
        evt_counter += 1

    events.sort(key=lambda e: (e.start_timestamp_sec, e.event_id))
    case.events = events
    if case.unlocked_step < int(WorkflowStep.PREDICTION):
        case.unlocked_step = int(WorkflowStep.PREDICTION)
    return events
