"""Read a CSV temperature log into an observations table, one row per data row.

Nothing is dropped here. Every data row becomes an observation that keeps its
file line number and its raw text, next to the parsed value and any parse
flags (see ``schemas.ROW_FLAGS``). Deciding what is usable is ``validate``'s job.

``IngestError`` is raised only when the file cannot be interpreted at all:
empty, binary, undecodable, no data rows, required column missing or
ambiguous, unrecognised or ambiguous timestamp format, mixed time-zone styles.

The source is opened read-only and never modified.

Preservation contract: every data row that has at most as many fields as the
header becomes an observation. Rows with MORE fields than the header cannot be
assigned to columns; they are NOT observations. They are kept separately as
``malformed_records`` (file line + exact raw text) in the result and report.

DataFrame input (``read_dataframe``): cells are converted to text one by one
(no CSV round-trip) and then parsed by the same rules as file cells. File-only
options (encoding, delimiter) are rejected. Decimal commas in text cells are
accepted, because a cell boundary is never ambiguous in a DataFrame.
``source_line`` is the 1-based row position; ``source_index`` is the caller's
index label. The caller's DataFrame is not modified.

Input column roles (case, accents and unit suffixes are ignored when matching):
    timestamp            required
    temperature          required, room air
    ambient_temperature  optional
    compressor_state     optional, on/off; carried and validated, not yet used
Internal names: ts, temp_c, ambient_c, compressor_on.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from coldsense.core.errors import DataError

from . import thresholds as C
from .schemas import InputIdentity, Issue, MalformedRecord, Severity, TemperatureUnit, Transformation


class IngestError(DataError, ValueError):
    """The file cannot be interpreted as a temperature log at all."""


ROLES = ("timestamp", "temperature", "ambient_temperature", "compressor_state")
REQUIRED_ROLES = ("timestamp", "temperature")
ALIASES: dict[str, set[str]] = {
    "timestamp": {"timestamp", "time_stamp", "time", "datetime", "date_time", "ts", "date", "date_heure",
                  "horodatage"},
    "temperature": {"temperature", "temp", "air_temp", "air_temperature", "room_temp", "room_temperature",
                    "t_air"},
    "ambient_temperature": {"ambient_temperature", "ambient", "ambient_temp", "outside_temp",
                            "outside_temperature", "t_amb", "temperature_ambiante", "t_ext"},
    "compressor_state": {"compressor_state", "compressor", "compressor_on", "comp_state", "compresseur",
                         "etat_compresseur"},
}
MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "--", "?"}
INF_TOKENS = {"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
ON_TOKENS = {"1", "1.0", "true", "on", "yes", "run", "running", "marche"}
OFF_TOKENS = {"0", "0.0", "false", "off", "no", "stop", "stopped", "arret"}
DELIMITERS = (";", "\t", ",", "|")  # ';' first: decimal commas can fake ',' consistency
_NUM_RE = r"^([-+]?(?:\d+(?:[.,]\d+)?|[.,]\d+)(?:[eE][-+]?\d+)?)\s*(?:°\s*)?([CFKcfk])?$"
_TZ_RE = r"\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*(?:Z|[+-]\d{2}:?\d{2})$"
_DATE_ORDERS = (("%d", "%m"), ("%m", "%d"))
CANDIDATE_FORMATS = [f"{a}{s}{b}{s}%Y{t}" for s in "/-." for a, b in _DATE_ORDERS for t in (" %H:%M:%S", " %H:%M")]


class IngestOptions(BaseModel):
    """Explicit choices that override detection. Every one used is recorded."""

    model_config = ConfigDict(frozen=True)

    encoding: str | None = None
    delimiter: str | None = None
    timestamp_format: str | None = None
    timezone: str | None = None          # IANA name for naive local timestamps, e.g. "Africa/Algiers"
    temperature_unit: TemperatureUnit | None = None
    column_map: dict[str, str] = {}      # role -> exact header


@dataclass
class IngestResult:
    observations: pd.DataFrame
    identity: InputIdentity
    issues: list[Issue] = field(default_factory=list)
    transformations: list[Transformation] = field(default_factory=list)
    rows_malformed: int = 0
    malformed_lines: list[int] = field(default_factory=list)
    malformed_records: list[MalformedRecord] = field(default_factory=list)
    rows_short: int = 0
    blank_lines: int = 0


# ---------------------------------------------------------------- bytes / text
def _read_source(src: str | Path | io.IOBase) -> tuple[bytes | str, str]:
    if isinstance(src, (str, Path)):
        p = Path(src)
        if not p.is_file():
            raise FileNotFoundError(2, "No such file", str(p))
        with p.open("rb") as fh:  # read-only; the source is never written
            return fh.read(), str(p)
    data = src.read()
    return data, getattr(src, "name", f"<{type(src).__name__}>")


def _decode(data: bytes, forced: str | None) -> tuple[str, str, list[Issue]]:
    if forced:
        try:
            return data.decode(forced), forced, []
        except (UnicodeDecodeError, LookupError) as e:
            raise IngestError(f"cannot decode file as {forced}: {e}") from e
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig"), "utf-8-sig", []
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16"), "utf-16", []
    if b"\x00" in data[:4096]:
        raise IngestError("file contains NUL bytes: binary, or UTF-16 without a byte-order mark. "
                          "Export as CSV (UTF-8) or pass the encoding explicitly.")
    try:
        return data.decode("utf-8"), "utf-8", []
    except UnicodeDecodeError:
        pass
    try:
        text = data.decode("cp1252")
    except UnicodeDecodeError as e:
        raise IngestError("file is neither UTF-8 nor Windows-1252 text; pass the encoding explicitly") from e
    return text, "cp1252", [Issue(code="ENCODING_FALLBACK", severity=Severity.WARNING,
                                  message="file is not valid UTF-8; decoded as Windows-1252 (cp1252). "
                                          "Check accented characters in headers and notes.")]


def _detect_delimiter(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()][:25]
    best, best_score = None, -1.0
    for d in DELIMITERS:
        counts = [len(next(csv.reader([ln], delimiter=d))) for ln in lines]
        if counts[0] < 2:
            continue
        score = sum(c == counts[0] for c in counts[1:]) / max(1, len(counts) - 1)
        if score > best_score:  # strict '>' keeps the earlier delimiter on ties
            best, best_score = d, score
    if best is None:
        raise IngestError("could not find a column delimiter (tried ; tab , |): the header has a single "
                          f"column: {lines[0][:80]!r}")
    return best


# --------------------------------------------------------------- CSV structure
def _parse_rows(text: str, delim: str) -> tuple[list[str], list[tuple[int, list[str]]], dict]:
    reader = csv.reader(io.StringIO(text), delimiter=delim)
    lines = io.StringIO(text).readlines()  # same line splitting as the csv reader
    header: list[str] | None = None
    rows: list[tuple[int, list[str]]] = []
    stats = {"blank": 0, "short": 0, "malformed": 0, "malformed_lines": [], "malformed_records": []}
    prev_line = 0
    try:
        for row in reader:
            start_line = prev_line + 1
            prev_line = reader.line_num
            if not row or all(not c.strip() for c in row):
                if header is not None:
                    stats["blank"] += 1
                continue
            if header is None:
                header = [c.strip() for c in row]
                continue
            n = len(header)
            if len(row) > n:
                if all(not c.strip() for c in row[n:]):
                    row = row[:n]  # trailing empty fields (e.g. "a,b,") are not data
                else:
                    stats["malformed"] += 1
                    stats["malformed_lines"].append(start_line)
                    raw = "".join(lines[start_line - 1:reader.line_num]).rstrip("\r\n")
                    stats["malformed_records"].append(
                        MalformedRecord(line=start_line, n_fields=len(row), expected_fields=n, raw_text=raw))
                    continue
            if len(row) < n:
                stats["short"] += 1
                row = row + [""] * (n - len(row))
            rows.append((start_line, row))
    except csv.Error as e:
        raise IngestError(f"malformed CSV near line {reader.line_num}: {e}") from e
    if header is None:
        raise IngestError("file is empty")
    if not rows:
        raise IngestError("no data rows below the header")
    return header, rows, stats


# --------------------------------------------------------------- column roles
_UNIT_PATTERNS = (
    r"[\(\[]\s*°?\s*([cfk])\s*[\)\]]",   # (°C) [F] (K)
    r"°\s*([cfk])\b",                     # °C
    r"(?:^|_)deg_?([cfk])$",              # deg_c, degc
    r"_([cfk])$",                         # temp_c, temperature_f
)
_UNIT_WORDS = {"celsius": "C", "fahrenheit": "F", "kelvin": "K"}


def _normalise_header(h: str) -> tuple[str, TemperatureUnit | None]:
    s = unicodedata.normalize("NFKD", h)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower().strip()
    unit = None
    for word, u in _UNIT_WORDS.items():
        if word in s:
            unit, s = u, s.replace(word, "")
    s = re.sub(r"[\s\-/\.]+", "_", s)
    for pat in _UNIT_PATTERNS:
        m = re.search(pat, s)
        if m:
            unit = unit or m.group(1).upper()
            s = s[:m.start()] + s[m.end():]
            break
    s = re.sub(r"_+", "_", s).strip("_")
    return s, unit


def _map_columns(header: list[str], opts: IngestOptions) -> tuple[dict[str, int], dict[str, TemperatureUnit | None]]:
    mapping: dict[str, int] = {}
    units: dict[str, TemperatureUnit | None] = {}
    for role, name in opts.column_map.items():
        if role not in ROLES:
            raise IngestError(f"unknown column role '{role}'; roles are {ROLES}")
        if name not in header:
            raise IngestError(f"column '{name}' given for '{role}' is not in the header {header}")
        mapping[role] = header.index(name)
    for role in ROLES:
        if role in mapping:
            units[role] = _normalise_header(header[mapping[role]])[1]
            continue
        hits = [i for i, h in enumerate(header) if _normalise_header(h)[0] in ALIASES[role]
                and i not in mapping.values()]
        if len(hits) > 1:
            raise IngestError(f"several columns could be '{role}': {[header[i] for i in hits]}. "
                              "Choose one explicitly (column_map).")
        if hits:
            mapping[role] = hits[0]
            units[role] = _normalise_header(header[hits[0]])[1]
    missing = [r for r in REQUIRED_ROLES if r not in mapping]
    if missing:
        raise IngestError(
            f"need a timestamp and a temperature column; missing {missing}. Found {header}. "
            f"Accepted names include {sorted(ALIASES['timestamp'])[:6]}... and "
            f"{sorted(ALIASES['temperature'])[:6]}..., or set column_map.")
    return mapping, units


# --------------------------------------------------------------- values
def _parse_numbers(raw: pd.Series, decimal_comma: bool) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (values, status, unit). status in ok/missing/non_numeric/infinite."""
    s = raw.str.strip()
    low = s.str.lower()
    status = pd.Series("non_numeric", index=s.index, dtype=object)
    status[low.isin(MISSING_TOKENS)] = "missing"
    status[low.isin(INF_TOKENS)] = "infinite"
    m = s.str.extract(_NUM_RE)
    num_txt, unit = m[0], m[1].str.upper()
    has_num = num_txt.notna() & (status == "non_numeric")
    if not decimal_comma:
        has_num &= ~num_txt.fillna("").str.contains(",", regex=False)
    status[has_num] = "ok"
    values = pd.Series(np.nan, index=s.index, dtype=float)
    # float() not pd.to_numeric: pandas 2.2 raises on "1e999" while pandas 3 returns inf.
    # Every string here already matched _NUM_RE, so float() cannot fail; overflow gives ±inf.
    values[has_num] = [float(x) for x in num_txt[has_num].str.replace(",", ".", regex=False)]
    overflow = np.isinf(values)  # e.g. "1e999": written as a number, but not a finite one
    status[overflow] = "infinite"
    values[overflow] = np.nan
    return values, status, unit.where(has_num & ~overflow)


def _resolve_unit(role: str, header_unit, cell_units: pd.Series, option, fallback: TemperatureUnit,
                  issues: list[Issue]) -> tuple[TemperatureUnit, str]:
    seen = sorted(set(cell_units.dropna()))
    if len(seen) > 1:
        issues.append(Issue(code="UNIT_INCONSISTENT", severity=Severity.BLOCKING,
                            message=f"{role}: cells carry different units {seen}; values were not converted"))
    declared = [(src, u) for src, u in (("option", option), ("header", header_unit),
                                         ("cells", seen[0] if len(seen) == 1 else None)) if u]
    if len({u for _, u in declared}) > 1:
        issues.append(Issue(code="UNIT_INCONSISTENT", severity=Severity.BLOCKING,
                            message=f"{role}: unit declarations disagree: {declared}"))
    if declared:
        return declared[0][1], declared[0][0]
    return fallback, "assumed"


def _to_celsius(v: pd.Series, unit: str) -> pd.Series:
    if unit == "F":
        return (v - 32.0) * 5.0 / 9.0
    if unit == "K":
        return v - 273.15
    return v


def _parses_with_any(s: pd.Series, fmts: list[str], *, utc: bool) -> pd.Series:
    """True where a cell parses with at least one of ``fmts`` (NaN cells are False)."""
    ok = pd.Series(False, index=s.index)
    for f in fmts:
        try:
            ok |= pd.to_datetime(s, format=f, errors="coerce", utc=utc).notna()
        except ValueError:  # an unusable explicit format is reported by the caller
            continue
    return ok


def _detect_naive(s: pd.Series, rows: pd.Series) -> tuple[pd.Series, str, str]:
    """Format detection for naive rows (unchanged rules): ISO 8601, else the best
    day/month-first candidate; ambiguous day/month or no usable format -> error."""
    n = int(rows.sum())
    iso = pd.to_datetime(s.where(rows), format="ISO8601", errors="coerce")
    if iso.notna().sum() / n >= C.TS_FORMAT_MIN_SUCCESS:
        return iso, "ISO8601", "iso8601"
    parsed = {f: pd.to_datetime(s.where(rows), format=f, errors="coerce") for f in CANDIDATE_FORMATS}
    rate = {f: p.notna().sum() / n for f, p in parsed.items()}
    fmt = max(CANDIDATE_FORMATS, key=lambda f: rate[f])  # first wins ties
    if rate[fmt] < C.TS_FORMAT_MIN_SUCCESS:
        examples = list(s[rows].head(3))
        raise IngestError(f"unrecognised timestamp format (e.g. {examples}); pass timestamp_format, "
                          "for example '%d/%m/%Y %H:%M'")
    twin = fmt.replace("%d", "@").replace("%m", "%d").replace("@", "%m")
    both = parsed[fmt].notna() & parsed[twin].notna()
    if rate[twin] >= rate[fmt] and (parsed[fmt][both] != parsed[twin][both]).any():
        raise IngestError(f"timestamps are ambiguous: they parse as both {fmt!r} and {twin!r} "
                          f"(e.g. {s[both].iloc[0]!r}). Pass timestamp_format to say which.")
    return parsed[fmt], fmt, "detected"


def _parse_timestamps(raw: pd.Series, opts: IngestOptions, issues: list[Issue],
                      transforms: list[Transformation]) -> tuple[pd.Series, pd.Series, str, str, str]:
    """Return (ts naive UTC-or-local datetime64, flag per row or '', format, format source, tz note).

    Rows with an explicit UTC offset (Z / ±hh:mm after a time) and rows without one
    are parsed separately. An offset is authoritative for its own row. Naive rows
    are interpreted only in the declared ``timezone``; with none, they stay local
    clock time and may not be mixed with offset rows (nothing is reinterpreted as UTC).
    """
    s = raw.str.strip()
    present = ~s.str.lower().isin(MISSING_TOKENS)
    flag = pd.Series("", index=s.index, dtype=object)
    flag[~present] = "TS_MISSING"
    if not present.any():
        raise IngestError("every timestamp cell is empty")
    tz = opts.timezone
    if tz:
        try:
            ZoneInfo(tz)
        except (ZoneInfoNotFoundError, ValueError) as e:
            raise IngestError(f"unknown time zone {tz!r}") from e

    # An explicit format is never ignored: with %z it describes offset rows (and,
    # minus %z, naive rows); without %z it describes naive rows.
    opt_fmt = opts.timestamp_format
    has_z = bool(opt_fmt) and "%z" in opt_fmt
    aware_fmt = opt_fmt if has_z else "ISO8601"
    naive_opt_fmt = re.sub(r"\s*%z", "", opt_fmt).strip() if has_z else opt_fmt

    aware = present & s.str.contains(_TZ_RE, regex=True)
    naive = present & ~aware
    naive_fmts = [naive_opt_fmt] if naive_opt_fmt else ["ISO8601", *CANDIDATE_FORMATS]
    aware_valid = _parses_with_any(s.where(aware), [aware_fmt], utc=True)
    naive_valid = _parses_with_any(s.where(naive), naive_fmts, utc=False)

    if aware_valid.any() and naive_valid.any() and not tz:
        raise IngestError(
            f"{int(aware_valid.sum())} valid timestamps carry a UTC offset and {int(naive_valid.sum())} valid "
            f"timestamps do not (e.g. {s[aware_valid].iloc[0]!r} and {s[naive_valid].iloc[0]!r}). Naive times "
            "are never assumed to be UTC: pass the time zone they were recorded in (timezone= / --timezone, "
            "e.g. 'Africa/Algiers'), or export the log with one convention.")
    if opt_fmt and has_z and not aware_valid.any():
        raise IngestError(f"timestamp_format {opt_fmt!r} includes %z but no timestamp matches it with an "
                          "offset; remove %z or check the format")
    if opt_fmt and not has_z and aware_valid.any() and not naive_valid.any():
        raise IngestError(f"timestamps carry UTC offsets but timestamp_format {opt_fmt!r} has no %z; "
                          "add %z or omit the option")

    ts = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    notes: list[str] = []
    fmt_parts: list[str] = []

    # --- rows with an explicit offset: authoritative, normalised to UTC ---
    if aware_valid.any():
        try:
            parsed = pd.to_datetime(s.where(aware), format=aware_fmt, errors="coerce", utc=True)
        except ValueError as e:
            raise IngestError(f"timestamp_format {aware_fmt!r} is not usable: {e}") from e
        utc = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
        ts[aware] = utc[aware].astype("datetime64[ns]")
        offsets = sorted(set(s[aware_valid].str.extract(r"(Z|[+-]\d{2}:?\d{2})$")[0].dropna()))
        note = f"{int(utc.notna().sum())} timestamps carried UTC offsets {offsets}; converted to UTC"
        notes.append(note)
        transforms.append(Transformation(code="TZ_TO_UTC", description=note, rows_affected=int(utc.notna().sum())))
        fmt_parts.append("ISO8601 with UTC offset" if aware_fmt == "ISO8601" else aware_fmt)

    # --- rows without an offset: declared zone, or local clock time of unknown zone ---
    run_naive = naive.any() and (naive_valid.any() or not aware_valid.any())  # junk next to offset rows: flag only
    if run_naive:
        if naive_opt_fmt:
            try:
                local = pd.to_datetime(s.where(naive), format=naive_opt_fmt, errors="coerce")
            except ValueError as e:
                raise IngestError(f"timestamp_format {naive_opt_fmt!r} is not usable: {e}") from e
            nfmt, nsrc = naive_opt_fmt, "option"
        else:
            local, nfmt, nsrc = _detect_naive(s, naive)
        fmt_parts.append(nfmt)
        if tz:
            before = local.notna() & naive
            loc = local.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
            flag[before & loc.isna()] = "TS_LOCAL_AMBIGUOUS"
            local = loc.dt.tz_convert("UTC").dt.tz_localize(None)
            n_conv = int((local.notna() & naive).sum())
            note = f"{n_conv} naive timestamps declared as {tz} (option); converted to UTC"
            notes.append(note)
            transforms.append(Transformation(code="TZ_TO_UTC", description=note, rows_affected=n_conv))
        else:
            notes.append("naive local timestamps, time zone unknown; no conversion applied")
            issues.append(Issue(code="TIMEZONE_UNKNOWN", severity=Severity.INFO,
                                message="timestamps have no time zone; treated as local clock time with no "
                                        "conversion. Clock changes in the log would appear as gaps or duplicates."))
        ts[naive] = local[naive].astype("datetime64[ns]")
    elif tz:
        issues.append(Issue(code="TIMEZONE_OPTION_IGNORED", severity=Severity.WARNING,
                            message=f"timezone {tz!r} not used: no valid timestamp lacks an offset"))

    flag[present & ts.isna() & (flag == "")] = "TS_INVALID"
    if opt_fmt:
        fmt, src = opt_fmt, "option"
    elif run_naive and aware_valid.any():
        fmt, src = "; ".join(fmt_parts), "detected" if nsrc == "detected" else "iso8601"
    elif run_naive:
        fmt, src = fmt_parts[-1], nsrc
    else:
        fmt, src = fmt_parts[0] if fmt_parts else "ISO8601", "iso8601"
    return ts.astype("datetime64[ns]"), flag, fmt, src, "; ".join(notes)


def _parse_compressor(raw: pd.Series) -> tuple[pd.Series, pd.Series]:
    low = raw.str.strip().str.lower()
    out = pd.Series(pd.NA, index=raw.index, dtype="boolean")
    out[low.isin(ON_TOKENS)] = True
    out[low.isin(OFF_TOKENS)] = False
    invalid = ~(low.isin(ON_TOKENS) | low.isin(OFF_TOKENS) | low.isin(MISSING_TOKENS))
    return out, invalid


# --------------------------------------------------------------- entry point
def _build_observations(header: list[str], rows: list[tuple[int, list[str]]], opts: IngestOptions,
                        decimal_comma: bool, issues: list[Issue], transforms: list[Transformation],
                        source_index: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Shared by file and DataFrame input: text cells -> observations with parse flags."""
    mapping, header_units = _map_columns(header, opts)

    def col(role: str) -> pd.Series:
        return pd.Series([r[mapping[role]] for _, r in rows], dtype=object)

    obs = pd.DataFrame({"source_line": [ln for ln, _ in rows]})
    if source_index is not None:
        obs["source_index"] = source_index
    flags: list[list[str]] = [[] for _ in rows]

    def add_flags(mask: pd.Series, code: str) -> None:
        for i in np.flatnonzero(mask.to_numpy()):
            flags[i].append(code)

    # timestamps
    obs["raw_timestamp"] = col("timestamp")
    ts, ts_flag, fmt, fmt_src, tz_note = _parse_timestamps(obs["raw_timestamp"], opts, issues, transforms)
    obs["ts"] = ts
    for code in ("TS_MISSING", "TS_INVALID", "TS_LOCAL_AMBIGUOUS"):
        add_flags(ts_flag == code, code)

    # temperature
    obs["raw_temperature"] = col("temperature")
    v, status, cell_units = _parse_numbers(obs["raw_temperature"], decimal_comma)
    unit, unit_src = _resolve_unit("temperature", header_units.get("temperature"), cell_units,
                                   opts.temperature_unit, "C", issues)
    temp_inconsistent = any(i.code == "UNIT_INCONSISTENT" for i in issues)
    if unit_src == "assumed" and not temp_inconsistent:
        issues.append(Issue(code="UNIT_ASSUMED", severity=Severity.INFO,
                            message="temperature unit is not declared (header, cells or option); assumed °C"))
    if unit != "C" and not temp_inconsistent:
        v = _to_celsius(v, unit)
        transforms.append(Transformation(code="UNIT_TO_CELSIUS",
                                         description=f"temperature converted from °{unit} to °C ({unit_src})",
                                         rows_affected=int(v.notna().sum())))
    obs["temp_c"] = v
    add_flags(status == "missing", "TEMP_MISSING")
    add_flags(status == "non_numeric", "TEMP_NON_NUMERIC")
    add_flags(status == "infinite", "TEMP_INFINITE")
    n_comma = int((obs["raw_temperature"].str.contains(",", regex=False) & (status == "ok")).sum())

    # ambient (optional)
    if "ambient_temperature" in mapping:
        obs["raw_ambient_temperature"] = col("ambient_temperature")
        av, astatus, acell = _parse_numbers(obs["raw_ambient_temperature"], decimal_comma)
        n_before = len(issues)
        aunit, asrc = _resolve_unit("ambient_temperature", header_units.get("ambient_temperature"), acell,
                                    None, unit, issues)
        ambient_inconsistent = any(i.code == "UNIT_INCONSISTENT" for i in issues[n_before:])
        if aunit != "C" and not ambient_inconsistent:
            av = _to_celsius(av, aunit)
            transforms.append(Transformation(
                code="UNIT_TO_CELSIUS", rows_affected=int(av.notna().sum()),
                description=f"ambient converted from °{aunit} to °C "
                            f"({'same as temperature' if asrc == 'assumed' else asrc})"))
        obs["ambient_c"] = av
        add_flags(astatus == "non_numeric", "AMBIENT_NON_NUMERIC")
        add_flags(astatus == "infinite", "AMBIENT_INFINITE")
        n_comma += int((obs["raw_ambient_temperature"].str.contains(",", regex=False) & (astatus == "ok")).sum())

    # compressor (optional)
    if "compressor_state" in mapping:
        obs["raw_compressor_state"] = col("compressor_state")
        comp, invalid = _parse_compressor(obs["raw_compressor_state"])
        obs["compressor_on"] = comp
        add_flags(invalid, "COMPRESSOR_INVALID")

    if n_comma:
        transforms.append(Transformation(code="DECIMAL_COMMA", rows_affected=n_comma,
                                         description="numbers written with a decimal comma read as decimals"))
    obs["flags"] = [";".join(f) for f in flags]

    info = dict(
        column_mapping={r: header[i] for r, i in mapping.items()},
        ignored_columns=[h for i, h in enumerate(header) if i not in mapping.values()],
        temperature_unit=unit, temperature_unit_source=unit_src,
        timestamp_format=fmt, timestamp_format_source=fmt_src, timezone_handling=tz_note,
    )
    return obs, info


def read_csv(src: str | Path | io.IOBase, options: IngestOptions | None = None) -> IngestResult:
    opts = options or IngestOptions()
    data, source = _read_source(src)
    issues: list[Issue] = []
    transforms: list[Transformation] = []
    if isinstance(data, str):
        encoded = data.encode("utf-8")
        text, encoding = data, "text (already decoded; hash is of its UTF-8 form)"
        sha, size = hashlib.sha256(encoded).hexdigest(), len(encoded)
    else:
        sha, size = hashlib.sha256(data).hexdigest(), len(data)
        if not data.strip():
            raise IngestError("file is empty")
        text, encoding, enc_issues = _decode(data, opts.encoding)
        issues += enc_issues
    if not text.strip():
        raise IngestError("file is empty")

    delim = opts.delimiter or _detect_delimiter(text)
    header, rows, st = _parse_rows(text, delim)
    obs, info = _build_observations(header, rows, opts, delim != ",", issues, transforms)
    if st["malformed"]:
        issues.append(Issue(code="MALFORMED_ROWS", severity=Severity.WARNING,
                            message=f"{st['malformed']} rows have more fields than the header and cannot be "
                                    f"assigned to columns (lines {st['malformed_lines'][:10]}). They are not "
                                    "observations; their raw text is kept in malformed_records."))
    if st["short"]:
        issues.append(Issue(code="SHORT_ROWS", severity=Severity.INFO,
                            message=f"{st['short']} rows have fewer fields than the header; missing fields "
                                    "are treated as empty"))

    identity = InputIdentity(
        source=source, sha256=sha, size_bytes=size, encoding=encoding, delimiter=delim,
        decimal_separator=". or ," if delim != "," else ".", header=header, **info)
    return IngestResult(obs, identity, issues, transforms, st["malformed"], st["malformed_lines"],
                        st["malformed_records"], st["short"], st["blank"])


def _cell_text(v: object) -> str:
    """One DataFrame cell as the text a CSV cell would hold."""
    if v is None or v is pd.NaT or v is pd.NA:
        return ""
    if isinstance(v, float | np.floating) and np.isnan(v):
        return ""
    if isinstance(v, bool | np.bool_):
        return "true" if v else "false"
    return str(v)


def read_dataframe(df: pd.DataFrame, options: IngestOptions | None = None,
                   source: str = "<DataFrame>") -> IngestResult:
    """Observations from an in-memory table. The caller's DataFrame is not modified."""
    opts = options or IngestOptions()
    file_only = [k for k in ("encoding", "delimiter") if getattr(opts, k)]
    if file_only:
        raise IngestError(f"option(s) {file_only} apply to CSV files only, not to DataFrame input")
    if df.shape[1] == 0:
        raise IngestError("DataFrame has no columns")
    if df.shape[0] == 0:
        raise IngestError("no data rows in the DataFrame")
    header = [str(c).strip() for c in df.columns]
    cols = [[_cell_text(v) for v in df.iloc[:, j].tolist()] for j in range(df.shape[1])]
    cells = [list(r) for r in zip(*cols, strict=True)]
    rows = [(i + 1, r) for i, r in enumerate(cells)]
    issues: list[Issue] = []
    transforms: list[Transformation] = []
    obs, info = _build_observations(header, rows, opts, True, issues, transforms,
                                    source_index=[str(x) for x in df.index])
    canon = "\n".join("\x1f".join(r) for r in [header, *cells]).encode("utf-8")
    identity = InputIdentity(
        source=source, sha256=hashlib.sha256(canon).hexdigest(), size_bytes=len(canon),
        encoding="n/a (DataFrame; hash is of its cell text)", delimiter="n/a (DataFrame)",
        decimal_separator=". or ,", header=header, **info)
    return IngestResult(obs, identity, issues, transforms)
