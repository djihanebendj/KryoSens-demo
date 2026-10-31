"""Data-quality gate. Decides which observations may feed analysis, and whether
the log as a whole may be used at all.

Nothing is deleted. Every observation keeps a status (USABLE / EXCLUDED) and its
flags (see ``schemas.ROW_FLAGS``). Only rows that are impossible or unreadable
are excluded from the analysis frame; unusual-but-possible readings are
flagged and stay in. Order of decisions (kept identical to the original gate so
analysis input is unchanged):

    parse flags -> sort by time (stable) -> duplicates (first kept)
    -> plausibility range + isolated impossible spikes -> sufficiency
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import thresholds as C
from .ingest import MISSING_TOKENS, IngestResult
from .schemas import (
    ROW_FLAGS,
    DataQualityReport,
    EvidenceClass,
    IntervalStats,
    Issue,
    RecordCounts,
    Severity,
    Transformation,
    ValidationLimits,
    Verdict,
)

EXCLUDING = {code for code, (excl, _) in ROW_FLAGS.items() if excl}
ANALYSIS_COLUMNS = ("ts", "temp_c", "ambient_c", "compressor_on", "source_line")
TELEMETRY_FIELDS = ("temp_c", "ambient_c", "compressor_on")  # compared for duplicates


def _flatline_minutes(df: pd.DataFrame) -> float:
    """Longest run of identical readings, in minutes."""
    if len(df) < 2:
        return 0.0
    change = df["temp_c"].diff().ne(0).cumsum()
    runs = df.groupby(change)["ts"].agg(["min", "max"])
    return float(((runs["max"] - runs["min"]).dt.total_seconds() / 60).max())


def _suspicious(usable: pd.DataFrame, median_interval_s: float | None) -> np.ndarray:
    """Readings far from their neighbours' median, beyond both an absolute floor
    and K robust standard deviations. Flag only; never excludes."""
    n = len(usable)
    if n < 5 or not median_interval_s:
        return np.zeros(n, dtype=bool)
    win = max(5, int(round(C.SUSPECT_WINDOW_MIN * 60 / median_interval_s)) | 1)
    t = usable["temp_c"]
    resid = t - t.rolling(win, center=True, min_periods=3).median()
    mad = float((resid - resid.median()).abs().median())
    thr = max(C.SUSPECT_MIN_DEV_C, C.SUSPECT_MAD_K * 1.4826 * mad)
    return (resid.abs() > thr).fillna(False).to_numpy()


def validate(ingested: IngestResult, *, source: str | None = None,
             evidence: EvidenceClass = EvidenceClass.UNVALIDATED,
             limits: ValidationLimits | None = None,
             declared_simulated: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, DataQualityReport]:
    """Return (analysis frame, observations with status and flags, report)."""
    lim = limits or ValidationLimits()
    src = source or ingested.identity.source
    obs = ingested.observations.copy()
    flags: list[list[str]] = [f.split(";") if f else [] for f in obs["flags"]]
    issues: list[Issue] = list(ingested.issues)
    transforms: list[Transformation] = list(ingested.transformations)
    add = lambda code, sev, msg: issues.append(Issue(code=code, severity=sev, message=msg))

    def flag(idx, code: str) -> None:
        for i in idx:
            flags[i].append(code)

    def excluded(i: int) -> bool:
        return any(f in EXCLUDING for f in flags[i])

    # --- evidence: never upgrade synthetic data to REAL ---
    looks_simulated = declared_simulated or os.path.basename(src).startswith("SIMULATED_")
    if evidence == EvidenceClass.REAL and looks_simulated:
        add("EVIDENCE_CONFLICT", Severity.BLOCKING,
            "declared REAL but the input is marked as simulated (file name or generator); refusing to "
            "treat synthetic data as real")

    # --- parse-level exclusions (already flagged by ingest) ---
    parse_excl = {c: sum(c in f for f in flags) for c in ("TS_MISSING", "TS_INVALID", "TS_LOCAL_AMBIGUOUS",
                                                          "TEMP_MISSING", "TEMP_NON_NUMERIC", "TEMP_INFINITE")}
    n_parse_excl = sum(excluded(i) for i in range(len(obs)))
    if n_parse_excl:
        detail = ", ".join(f"{k}={v}" for k, v in parse_excl.items() if v)
        add("INVALID_RECORDS", Severity.WARNING,
            f"{n_parse_excl} rows cannot be used: {detail}. Kept in the observations table, "
            "excluded from analysis.")

    # --- order: rows earlier in time than a previous row in the file ---
    valid_ts = obs["ts"].notna() & ~pd.Series([any(f.startswith("TS_") for f in fl) for fl in flags])
    ts_valid = obs["ts"].where(valid_ts)
    prev_max = ts_valid.ffill().cummax().shift(1)
    ooo = (valid_ts & (ts_valid < prev_max)).to_numpy()
    flag(np.flatnonzero(ooo), "OUT_OF_ORDER")
    n_ooo = int(ooo.sum())
    if n_ooo:
        add("TIMESTAMPS_OUT_OF_ORDER", Severity.WARNING,
            f"{n_ooo} rows are earlier than a previous row (clock change, merged files or logger reset?)")
        transforms.append(Transformation(code="SORTED_BY_TIME", rows_affected=n_ooo,
                                         description="rows ordered by timestamp (stable: file order kept for ties)"))

    # --- candidates: sort (stable), then duplicate timestamps ---
    cand_idx = [i for i in range(len(obs)) if not excluded(i)]
    cand = obs.loc[cand_idx].sort_values("ts", kind="stable")
    dup = cand["ts"].duplicated(keep="first")
    n_exact = n_conflict = 0
    conflict_fields: dict[str, int] = {}
    obs["duplicate_of_line"] = pd.Series(pd.NA, index=obs.index, dtype="Int64")
    obs["duplicate_differs"] = ""
    if dup.any():
        # Exact = every canonical telemetry field present in the file is equal to the
        # kept row's (both missing counts as equal). Any difference -> conflict.
        fields = [c for c in TELEMETRY_FIELDS if c in cand]
        kept_rows = cand[~dup].set_index("ts")
        ref = kept_rows.loc[cand.loc[dup, "ts"]]
        ref.index = cand.index[dup]
        differs = pd.DataFrame(False, index=ref.index, columns=fields)
        for c in fields:
            a, b = cand.loc[dup, c], ref[c]
            same = (a == b).fillna(False).astype(bool) | (a.isna() & b.isna())
            differs[c] = ~same
        is_conflict_dup = differs.any(axis=1)
        is_conflict = pd.Series(False, index=cand.index)
        is_conflict[is_conflict_dup.index] = is_conflict_dup
        is_exact = dup & ~is_conflict
        n_exact, n_conflict = int(is_exact.sum()), int(is_conflict.sum())
        conflict_fields = {c: int(differs[c].sum()) for c in fields if differs[c].any()}
        flag(cand.index[is_exact], "DUPLICATE_EXACT")
        flag(cand.index[is_conflict], "DUPLICATE_CONFLICT")
        kept = cand.index[~dup & cand["ts"].isin(cand.loc[is_conflict, "ts"])]
        flag(kept, "DUPLICATE_CONFLICT_KEPT")
        obs.loc[ref.index, "duplicate_of_line"] = kept_rows.loc[cand.loc[dup, "ts"], "source_line"].to_numpy()
        obs.loc[differs.index, "duplicate_differs"] = [";".join(c for c in fields if row[c])
                                                       for _, row in differs.iterrows()]
        detail = ", ".join(f"{c} differs in {k}" for c, k in conflict_fields.items())
        add("DUPLICATE_TIMESTAMPS", Severity.WARNING,
            f"{n_exact + n_conflict} duplicate timestamps ({n_exact} exact repeats, {n_conflict} conflicting"
            f"{': ' + detail if detail else ''}); the first row in the file was kept for each and the others "
            "are excluded, flagged and linked to it (duplicate_of_line)")
        transforms.append(Transformation(code="DUPLICATES_FIRST_KEPT", rows_affected=n_exact + n_conflict,
                                         description="for each repeated timestamp, the first row in the file "
                                                     "is used; later rows are excluded and flagged, not merged"))
        if n_conflict / max(1, len(obs)) > C.CONFLICT_DUP_MAX_FRACTION:
            add("DUPLICATE_CONFLICTS", Severity.BLOCKING,
                f"{n_conflict} timestamps carry conflicting values ({100 * n_conflict / len(obs):.1f}% of rows, "
                f"limit {100 * C.CONFLICT_DUP_MAX_FRACTION:.0f}%): two sensors merged or a clock change?")
    cand = cand[~dup]

    # --- impossible values: outside physical range, isolated impossible spikes ---
    oor = (cand["temp_c"] < lim.plausible_min_c) | (cand["temp_c"] > lim.plausible_max_c)
    dt_min = cand["ts"].diff().dt.total_seconds() / 60
    jump = cand["temp_c"].diff().abs()
    step = jump / dt_min
    too_fast = (step > lim.max_step_c_per_min) & (jump >= lim.spike_min_jump_c)  # fast AND large
    spike = too_fast & too_fast.shift(-1, fill_value=False)  # up then straight back down
    flag(cand.index[oor], "TEMP_OUT_OF_RANGE")
    flag(cand.index[spike & ~oor], "TEMP_IMPLAUSIBLE_SPIKE")
    n_bad = int((oor | spike).sum())
    if n_bad:
        add("IMPLAUSIBLE_VALUES", Severity.WARNING,
            f"{n_bad} readings are physically implausible (outside {lim.plausible_min_c}..{lim.plausible_max_c} °C "
            f"or isolated jumps of ≥ {lim.spike_min_jump_c} °C faster than {lim.max_step_c_per_min} °C/min "
            "that come straight back); kept and flagged, excluded "
            "from analysis")
    usable_idx = cand.index[~(oor | spike)]

    # --- ambient sanity (flag only; ambient is optional context) ---
    if "ambient_c" in obs:
        a_oor = (obs["ambient_c"] < lim.ambient_min_c) | (obs["ambient_c"] > lim.ambient_max_c)
        flag(np.flatnonzero(a_oor.to_numpy()), "AMBIENT_OUT_OF_RANGE")
        n_a_bad = int(a_oor.sum()) + sum(("AMBIENT_NON_NUMERIC" in f) or ("AMBIENT_INFINITE" in f) for f in flags)
        if n_a_bad:
            add("AMBIENT_INVALID", Severity.WARNING,
                f"{n_a_bad} ambient readings are non-numeric, infinite or implausible; flagged")
    if "compressor_on" in obs:
        n_c_bad = sum("COMPRESSOR_INVALID" in f for f in flags)
        if n_c_bad:
            add("COMPRESSOR_INVALID", Severity.WARNING,
                f"{n_c_bad} compressor-state values are not recognised on/off values; set to missing")

    cols = [c for c in ANALYSIS_COLUMNS if c in obs]
    clean = obs.loc[usable_idx, cols].reset_index(drop=True)

    # --- sampling statistics and sufficiency (on usable rows) ---
    n = len(clean)
    start = clean["ts"].iloc[0] if n else None
    end = clean["ts"].iloc[-1] if n else None
    dur_h = (end - start).total_seconds() / 3600 if n > 1 else 0.0

    med_s = longest_gap = coverage = None
    n_gaps = n_irreg = 0
    d = pd.Series(dtype=float)
    if n > 2:
        d = clean["ts"].diff().dt.total_seconds().iloc[1:]
        med_s = float(d.median())
        gaps = d[d > C.GAP_FACTOR * med_s]
        n_gaps = int(len(gaps))
        longest_gap = float(gaps.max() / 60) if n_gaps else 0.0
        expected = dur_h * 3600 / med_s + 1 if med_s > 0 else n
        coverage = min(100.0, 100.0 * n / expected)
        n_irreg = int(((d - med_s).abs() > C.IRREGULAR_TOLERANCE * med_s).sum()) if med_s > 0 else 0

    if n < 3:
        add("TOO_FEW_ROWS", Severity.BLOCKING, f"only {n} usable readings")
    else:
        if med_s > C.MAX_INTERVAL_S:
            add("COARSE_SAMPLING", Severity.BLOCKING,
                f"median interval {med_s/60:.1f} min is too coarse to see compressor cycles "
                f"(need ≤ {C.MAX_INTERVAL_S/60:.0f} min)")
        if dur_h < C.MIN_DURATION_H:
            add("TOO_SHORT", Severity.BLOCKING,
                f"log covers {dur_h:.1f} h; need at least {C.MIN_DURATION_H:.0f} h to characterise the unit")
        elif dur_h < C.TARGET_DURATION_H:
            add("SHORT_LOG", Severity.WARNING,
                f"log covers {dur_h:.1f} h; {C.TARGET_DURATION_H:.0f} h recommended")
        if coverage < C.MIN_COVERAGE_PCT:
            add("LOW_COVERAGE", Severity.BLOCKING,
                f"only {coverage:.0f}% of expected samples present (need ≥ {C.MIN_COVERAGE_PCT:.0f}%)")
        elif n_gaps:
            add("GAPS", Severity.WARNING,
                f"{n_gaps} gaps; longest {longest_gap:.0f} min. Segments crossing a gap are excluded.")
        if len(d) and n_irreg / len(d) > C.IRREGULAR_MAX_FRACTION:
            add("IRREGULAR_SAMPLING", Severity.WARNING,
                f"{n_irreg} of {len(d)} intervals ({100 * n_irreg / len(d):.0f}%) differ from the median "
                f"{med_s:.0f} s by more than {100 * C.IRREGULAR_TOLERANCE:.0f}%")
        flat = _flatline_minutes(clean)
        if flat >= C.FLATLINE_MIN:
            add("FLATLINE", Severity.WARNING,
                f"temperature unchanged for {flat:.0f} min; possible stuck sensor or logger fault")

    # --- unusual but possible: flag, keep in analysis ---
    susp = _suspicious(clean, med_s)
    flag(usable_idx[susp], "TEMP_SUSPICIOUS")
    n_susp = int(susp.sum())
    if n_susp:
        add("SUSPICIOUS_VALUES", Severity.WARNING,
            f"{n_susp} readings differ sharply from neighbouring readings but are physically possible "
            "(door opening, defrost, sensor touch?); kept in analysis and flagged")

    # --- unit plausibility when the unit was only assumed ---
    if ingested.identity.temperature_unit_source == "assumed" and n:
        med_t = float(clean["temp_c"].median())
        if med_t > C.UNDECLARED_UNIT_WARM_MEDIAN_C:
            add("UNIT_SUSPECT", Severity.WARNING,
                f"unit not declared and the median reading is {med_t:.1f} °C, unusually warm for a "
                "refrigerated room. If the log is in °F, declare the unit.")

    has_amb = "ambient_c" in clean and clean["ambient_c"].notna().mean() > 0.5
    if not has_amb:
        add("NO_AMBIENT", Severity.INFO,
            "no usable ambient temperature; warming rates cannot be corrected for room surroundings")

    if any(i.severity == Severity.BLOCKING for i in issues):
        verdict = Verdict.REFUSED
    elif any(i.severity == Severity.WARNING for i in issues):
        verdict = Verdict.LIMITED
    else:
        verdict = Verdict.USABLE

    # --- observations: final status per row ---
    obs["flags"] = [";".join(f) for f in flags]
    obs["status"] = "EXCLUDED"
    obs.loc[usable_idx, "status"] = "USABLE"
    flag_counts: dict[str, int] = {}
    for fl in flags:
        for f in fl:
            flag_counts[f] = flag_counts.get(f, 0) + 1

    def n_missing(raw_col: str) -> int:
        return int(obs[raw_col].str.strip().str.lower().isin(MISSING_TOKENS).sum()) if raw_col in obs else 0

    fc = flag_counts.get
    missing_by_field = {"timestamp": fc("TS_MISSING", 0), "temperature": fc("TEMP_MISSING", 0)}
    invalid_by_field = {
        "timestamp": fc("TS_INVALID", 0) + fc("TS_LOCAL_AMBIGUOUS", 0),
        "temperature": fc("TEMP_NON_NUMERIC", 0) + fc("TEMP_INFINITE", 0) + fc("TEMP_OUT_OF_RANGE", 0)
        + fc("TEMP_IMPLAUSIBLE_SPIKE", 0),
    }
    if "ambient_c" in obs:
        missing_by_field["ambient_temperature"] = n_missing("raw_ambient_temperature")
        invalid_by_field["ambient_temperature"] = (fc("AMBIENT_NON_NUMERIC", 0) + fc("AMBIENT_INFINITE", 0)
                                                   + fc("AMBIENT_OUT_OF_RANGE", 0))
    if "compressor_on" in obs:
        missing_by_field["compressor_state"] = n_missing("raw_compressor_state")
        invalid_by_field["compressor_state"] = fc("COMPRESSOR_INVALID", 0)

    limitations = [
        "Validation checks the file's internal consistency only. It cannot confirm sensor calibration, "
        "sensor placement, or that the log comes from the unit it is said to come from.",
        "Air temperature at the sensor is not product-core temperature.",
    ]
    if evidence == EvidenceClass.UNVALIDATED:
        limitations.append("Provenance not declared: results from this file are UNVALIDATED.")
    if evidence == EvidenceClass.SIMULATED:
        limitations.append("SIMULATED data: says nothing about how a real room behaves.")
    if any(i.code == "TIMEZONE_UNKNOWN" for i in issues):
        limitations.append("Time zone unknown: intervals assume the logger clock never changed.")
    if ingested.identity.temperature_unit_source == "assumed":
        limitations.append("Temperature unit was not declared; °C was assumed.")
    if "compressor_on" in obs:
        limitations.append("compressor_state is validated and carried, but not used by analysis yet.")

    counts = RecordCounts(
        data_rows_read=len(obs), rows_malformed=ingested.rows_malformed,
        malformed_lines=ingested.malformed_lines, rows_short=ingested.rows_short,
        blank_lines=ingested.blank_lines, rows_usable=n, rows_excluded=len(obs) - n, rows_suspicious=n_susp,
    )
    q = d.quantile([0.05, 0.95]) if len(d) else None
    intervals = IntervalStats(
        n_intervals=len(d),
        min_s=float(d.min()) if len(d) else None, p05_s=float(q[0.05]) if q is not None else None,
        median_s=med_s, p95_s=float(q[0.95]) if q is not None else None,
        max_s=float(d.max()) if len(d) else None,
        n_irregular=n_irreg, irregular_pct=round(100 * n_irreg / len(d), 2) if len(d) else 0.0,
        n_gaps=n_gaps, longest_gap_min=round(longest_gap or 0.0, 1), n_out_of_order=n_ooo,
    )

    report = DataQualityReport(
        source=src, evidence=evidence,
        n_rows_raw=len(obs), n_rows_clean=n,
        start=start, end=end, duration_hours=round(dur_h, 3),
        median_interval_s=med_s, coverage_pct=round(coverage or 0.0, 1),
        n_gaps=n_gaps, longest_gap_min=round(longest_gap or 0.0, 1),
        n_implausible=n_bad, n_duplicates=n_exact + n_conflict,
        temp_min_c=float(clean["temp_c"].min()) if n else None,
        temp_max_c=float(clean["temp_c"].max()) if n else None,
        ambient_available=bool(has_amb), issues=issues, verdict=verdict,
        suitable_for_downstream=verdict != Verdict.REFUSED,
        input=ingested.identity, counts=counts,
        missing_by_field=missing_by_field, invalid_by_field=invalid_by_field,
        flag_counts=dict(sorted(flag_counts.items())),
        n_duplicates_exact=n_exact, n_duplicates_conflicting=n_conflict,
        duplicate_conflict_fields=conflict_fields,
        intervals=intervals, malformed_records=ingested.malformed_records, transformations=transforms,
        limits_used=lim, limitations=limitations,
    )
    return clean, obs, report
