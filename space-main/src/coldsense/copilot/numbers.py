"""Numeric-claim integrity: find every number in a text and check it against what this case may state.

Extracted: digits (signed, dot or comma decimals), ranges ("40-60", "40 to 60", "between 40 and 60"),
percentages, number words ("twenty-five"), each with the unit that follows; identifiers (EX-200, SB-12, R-290,
E3); dates and times; document references ("page 3", "section 4.2", "lines 13-18", "v2").

Checked (``check``): a number is allowed only if the same value AND unit is in the allowed set; units are
normalised by alias only (``minutes`` -> ``min``) and never converted. Fail-closed: anything that looks numeric
but cannot be read with certainty (non-ASCII digits, fractions, exponents, ordinals, "1,200" which may be
1200 or 1.2, a number word without a readable value such as "half", "degrees" without C or F) is a violation.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

UNIT_ALIASES = {
    "°c": "°C", "° c": "°C", "degc": "°C", "deg c": "°C", "degrees c": "°C", "degree c": "°C", "degrees celsius": "°C",
    "celsius": "°C", "°f": "°F", "degrees f": "°F", "fahrenheit": "°F", "k": "K", "kelvin": "K",
    "%": "%", "percent": "%", "per cent": "%",
    "min": "min", "mins": "min", "minute": "min", "minutes": "min",
    "h": "h", "hr": "h", "hrs": "h", "hour": "h", "hours": "h",
    "s": "s", "sec": "s", "secs": "s", "second": "s", "seconds": "s",
    "day": "days", "days": "days", "week": "weeks", "weeks": "weeks", "month": "months", "months": "months",
    "year": "years", "years": "years",
    "v": "V", "volt": "V", "volts": "V", "kv": "kV", "a": "A", "amp": "A", "amps": "A", "ampere": "A",
    "amperes": "A", "ma": "mA", "hz": "Hz", "w": "W", "watt": "W", "watts": "W", "kw": "kW", "kwh": "kWh",
    "bar": "bar", "psi": "psi", "kpa": "kPa", "mpa": "MPa", "pa": "Pa",
    "g": "g", "gram": "g", "grams": "g", "kg": "kg", "kilogram": "kg", "kilograms": "kg",
    "mm": "mm", "cm": "cm", "m": "m", "metre": "m", "metres": "m", "meter": "m", "meters": "m",
    "ohm": "ohm", "ohms": "ohm", "ω": "ohm", "rpm": "rpm",
    "1/min": "1/min", "°c/min": "°C/min", "k/min": "K/min", "°c/h": "°C/h",
}
_UNIT_KEYS = sorted(UNIT_ALIASES, key=len, reverse=True)
_UNIT_RE = "|".join(re.escape(u) for u in _UNIT_KEYS)
_NUM = r"[-+−]?\d+(?:[.,]\d+)*"
_AMBIGUOUS_DEGREE = re.compile(r"^\s*(?:°|degrees?\b)(?!\s*[cf]\b)", re.I)

WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
         "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
         "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
         "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
SCALES = {"hundred": 100, "thousand": 1000}
VAGUE = {"half", "quarter", "dozen", "twice", "thrice", "double", "triple", "couple", "several", "single",
         "first", "second", "third", "fourth", "fifth", "tenth"}
_WORD_RE = re.compile(r"\b(?:(?:" + "|".join(list(WORDS) + list(SCALES)) + r")(?:[\s-]+(?:and[\s-]+)?|\b))+", re.I)

_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?\b|"
                   r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_REF = re.compile(r"\b(pages?|p\.|sections?|§|lines?|paragraphs?|versions?|v|rev(?:ision)?)\s*"
                  r"(\d+(?:[.\-–]\d+)*)", re.I)
_IDENT = re.compile(r"\b[A-Za-z]+(?:-[A-Za-z]+)*-?\d+(?:[.\-/]?[A-Za-z0-9]+)*\b")
_SUSPECT = re.compile(r"\d\s*[eE][-+]?\d|\d\s*/\s*\d|\d(?:st|nd|rd|th)\b|[⁄½¼¾⅓⅔]|\d[xX×]\b|\b[xX×]\d")


def norm_unit(u: str | None) -> str:
    if not u:
        return ""
    return UNIT_ALIASES.get(re.sub(r"\s+", " ", u.strip().lower()), u.strip())


def norm_ident(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _value(tok: str) -> float | None:
    t = tok.replace("−", "-")
    if "," in t:
        if "." in t or re.search(r",\d{3}(?!\d)", t):
            return None                                   # thousands separator or mixed: cannot be read safely
        t = t.replace(",", ".")
    if t.count(".") > 1:
        return None
    try:
        return float(t)
    except ValueError:
        return None


@dataclass(frozen=True)
class NumberMention:
    text: str
    value: float | None
    unit: str
    kind: str = "NUMBER"          # NUMBER | WORD


@dataclass
class Extraction:
    numbers: list[NumberMention] = field(default_factory=list)
    identifiers: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    references: list[tuple[str, str]] = field(default_factory=list)
    unparseable: list[str] = field(default_factory=list)


def _words_value(phrase: str) -> int | None:
    total, cur, seen = 0, 0, False
    for w in re.split(r"[\s-]+", phrase.lower()):
        if not w or w == "and":
            continue
        if w in WORDS:
            cur += WORDS[w]
            seen = True
        elif w in SCALES:
            cur = max(cur, 1) * SCALES[w]
            if SCALES[w] >= 1000:
                total, cur = total + cur, 0
            seen = True
        else:
            return None
    return total + cur if seen else None


def extract(text: str) -> Extraction:
    out = Extraction()
    for ch in text:                                       # non-ASCII digits: fail closed
        if ch.isdigit() and not ("0" <= ch <= "9"):
            out.unparseable.append(ch)
    s = unicodedata.normalize("NFKC", text)
    for m in _SUSPECT.finditer(s):
        out.unparseable.append(m.group(0))
    taken = [False] * len(s)

    def take(a: int, b: int) -> None:
        for i in range(a, b):
            taken[i] = True

    def free(a: int, b: int) -> bool:
        return not any(taken[a:b])

    for m in _DATE.finditer(s):
        out.dates.append(m.group(0))
        take(*m.span())
    for m in _REF.finditer(s):
        if free(*m.span()):
            out.references.append((m.group(1).lower().rstrip("s."), m.group(2)))
            take(*m.span())
    for m in _IDENT.finditer(s):
        if free(*m.span()) and not re.fullmatch(r"[A-Za-z]+", m.group(0)):
            tok = m.group(0)
            # "230V", "16A", "8h": a number glued to a unit is a quantity, not an identifier
            q = re.fullmatch(r"(\d+(?:[.,]\d+)?)(" + _UNIT_RE + r")", tok, re.I)
            if q is None:
                out.identifiers.append(tok)
                take(*m.span())
    num_re = re.compile(r"(?<![\w.,])(" + _NUM + r")(?:\s*(?:-|–|to|and)\s*(" + _NUM + r"))?"
                        r"(?:\s*(" + _UNIT_RE + r")(?![a-zA-Z0-9]))?", re.I)
    for m in num_re.finditer(s):
        if not free(*m.span(1)):
            continue
        unit = norm_unit(m.group(3))
        tail = s[m.end():m.end() + 12]
        if not m.group(3) and _AMBIGUOUS_DEGREE.match(tail):
            out.unparseable.append(m.group(0) + tail.split()[0] if tail.split() else m.group(0))
        for g in (1, 2):
            tok = m.group(g)
            if tok is None:
                continue
            v = _value(tok)
            if v is None:
                out.unparseable.append(tok)
            else:
                out.numbers.append(NumberMention(tok, v, unit))
        take(*m.span())
    low = s.lower()
    for m in _WORD_RE.finditer(low):
        phrase = m.group(0).strip(" -")
        if not phrase or not free(m.start(), m.start() + len(phrase)):
            continue
        v = _words_value(phrase)
        um = re.match(r"\s*(" + _UNIT_RE + r")(?![a-zA-Z0-9])", low[m.start() + len(phrase):], re.I)
        unit = norm_unit(um.group(1)) if um else ""
        if phrase in ("one", "a") and not unit:
            continue                                      # "one of the checks": not a quantity
        if v is None:
            out.unparseable.append(phrase)
        else:
            out.numbers.append(NumberMention(phrase, float(v), unit, "WORD"))
    for w in re.findall(r"[a-z]+", low):
        if w in VAGUE:
            out.unparseable.append(w)
    return out


@dataclass
class Allowed:
    """What one statement may state: (value, unit) pairs, identifiers, dates and document references."""
    quantities: set[tuple[float, str]] = field(default_factory=set)
    identifiers: set[str] = field(default_factory=set)
    dates: set[str] = field(default_factory=set)
    references: set[tuple[str, str]] = field(default_factory=set)

    def merge(self, other: Allowed) -> Allowed:
        return Allowed(self.quantities | other.quantities, self.identifiers | other.identifiers,
                       self.dates | other.dates, self.references | other.references)

    def add_text(self, text: str) -> Allowed:
        """Everything in a trusted text (a verified quote, a citation label, a user message) becomes allowed."""
        e = extract(text)
        self.quantities |= {(n.value, n.unit) for n in e.numbers if n.value is not None}
        self.identifiers |= {norm_ident(i) for i in e.identifiers}
        self.dates |= set(e.dates)
        self.references |= set(e.references)
        return self


def check(text: str, allowed: Allowed) -> list[str]:
    """Violations (empty list = every numeric element is supported). Never raises on odd input: fails closed."""
    try:
        e = extract(text)
    except Exception as exc:                              # noqa: BLE001 - a parser failure is a violation
        return [f"UNPARSEABLE: the text could not be checked ({type(exc).__name__})"]
    v = [f"UNPARSEABLE: '{u}' cannot be verified as a number" for u in e.unparseable]
    for n in e.numbers:
        if (n.value, n.unit) not in allowed.quantities:
            near = sorted({u for val, u in allowed.quantities if val == n.value})
            v.append(f"UNSUPPORTED_NUMBER: '{n.text}{(' ' + n.unit) if n.unit else ''}' is not a value of this "
                     "case" + (f" (the value exists only with unit {', '.join(near) or 'none'})" if near else ""))
    for i in e.identifiers:
        if norm_ident(i) not in allowed.identifiers:
            v.append(f"UNSUPPORTED_IDENTIFIER: '{i}' does not match this case or its cited sources")
    for d in e.dates:
        if d not in allowed.dates:
            v.append(f"UNSUPPORTED_DATE: '{d}' is not in a tool result or source of this case")
    for r in e.references:
        if r not in allowed.references:
            v.append(f"UNSUPPORTED_REFERENCE: '{r[0]} {r[1]}' does not match a cited source location")
    return v
