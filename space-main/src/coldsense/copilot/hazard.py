"""Hazard gate (docs/COPILOT.md §5): deterministic application logic, run before any LLM generation and before
every LLM-proposed tool call. It triggers on the PRESENCE of a hazard TOPIC noun in the request, never on a verb.

* Tier A, PROTECTIVE_DEVICE: a protective-device noun is present. The request is never answered with a procedure,
  whatever a document says. The application composes a fixed refusal and may show only verbatim manufacturer
  warnings or documented-function quotes, attributed, that pass the Phase 8 verbatim and numeric checks.
* Tier B, HAZARDOUS_WORK: a hazardous-work noun is present. Procedural passages are shown only with an
  established model, authoritative documentation and a qualified-personnel boundary; otherwise the same fixed
  template states what is missing. Steps are always verbatim quotes, never LLM-authored.

Topics come from ``src/coldsense/copilot/hazard_topics.yaml`` inside the copilot package (loaded with
``importlib.resources`` from the package directory, never from the repository layout). If the file is
missing, empty, malformed or invalid, every request fails closed with CONFIG_ERROR and no retrieval runs.

Detection normalises case, accents, hyphens and simple plurals, then matches a topic's noun tokens as a
contiguous run. The whole recent conversation is classified so a request cannot be split across turns; a clear,
referential-free change to a non-hazard topic does not carry a hazard topic forward, and an ambiguous relation
fails closed.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from importlib.resources import files
from pathlib import Path

log = logging.getLogger("coldsense.copilot.gate")

CONTEXT_WINDOW = 6                      # messages before the current one that may supply a referenced topic
# non-hazard subjects that, with no back-reference, mark a genuine topic change
NEW_TOPIC = re.compile(r"\b(defrost|setpoint|set point|temperature|humidity|interval|maintenance schedule|"
                       r"service report|forecast|crossing|threshold|log|ingest|segment|report|version|model number)\b",
                       re.I)
# markers that the current turn refers back to an earlier one
REFERENTIAL = re.compile(r"\b(it|its|that|this|them|those|the device|the same|same one|do it|go ahead|"
                         r"continue|carry on|go on|and then|then how|what about|proceed|like i said|as above)\b",
                         re.I)


class Tier(str, Enum):
    NONE = "NONE"
    PROTECTIVE_DEVICE = "PROTECTIVE_DEVICE"     # Tier A
    HAZARDOUS_WORK = "HAZARDOUS_WORK"           # Tier B
    CONFIG_ERROR = "CONFIG_ERROR"               # topics config unusable: fail closed


UNSUPPORTED_LANGUAGE_MESSAGE = (
    "This assistant answers in English or French only. Please restate your request in English or French. "
    "Cet assistant répond uniquement en anglais ou en français. Veuillez reformuler votre demande en anglais "
    "ou en français."
)


def unsupported_script(text: str) -> bool:
    """True when the text contains a letter or digit outside the Latin script (for example Arabic, Cyrillic, Greek
    or CJK), including mixed-script input. Deterministic and run before any LLM call or proposed tool execution.

    Limitation: this is a SCRIPT check, not language identification. Unsupported languages written in the Latin
    script (e.g. Spanish or German) are not detected; only English and French are supported by policy.
    """
    for ch in text or "":
        if ch.isdigit() and not ("0" <= ch <= "9"):
            return True
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        if not name.startswith("LATIN"):
            return True
    return False


@dataclass(frozen=True)
class GateDecision:
    triggered: bool
    tier: Tier
    topic: str | None = None               # the matched noun, or None
    trigger: str | None = None             # where it matched: user_request | prior_turn | proposed_args | ambiguous
    reason: str = ""
    config_ok: bool = True

    def log_record(self) -> dict:
        """Structured, redacted: the decision and why, never the request text or document content."""
        return {"tier": self.tier.value, "triggered": self.triggered, "trigger": self.trigger,
                "topic": self.topic, "reason": self.reason}


# ---------------------------------------------------------------- normalisation and matching
def _singular(w: str) -> str:
    if len(w) > 3 and w.endswith(("ches", "shes", "sses", "xes", "zes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 2 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def _tokens(text: str) -> list[str]:
    """Lower-cased alphanumeric tokens, accents and hyphens removed, each lightly singularised."""
    s = unicodedata.normalize("NFKD", text)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace("-", " ").replace("/", " ")
    return [_singular(t) for t in re.findall(r"[a-z0-9]+", s)]


def _contains(hay: list[str], needle: tuple[str, ...]) -> bool:
    n = len(needle)
    return n > 0 and any(tuple(hay[i:i + n]) == needle for i in range(len(hay) - n + 1))


class HazardTopics:
    """The two validated noun lists, each as normalised token tuples."""

    def __init__(self, protective: list[str], hazardous: list[str]) -> None:
        self.protective = {p: tuple(_tokens(p)) for p in protective}
        self.hazardous = {h: tuple(_tokens(h)) for h in hazardous}

    def find(self, text: str) -> tuple[Tier, str] | None:
        """The first protective, then hazardous, noun present in the text (protective takes precedence)."""
        toks = _tokens(text)
        for name, seq in self.protective.items():
            if _contains(toks, seq):
                return Tier.PROTECTIVE_DEVICE, name
        for name, seq in self.hazardous.items():
            if _contains(toks, seq):
                return Tier.HAZARDOUS_WORK, name
        return None


class ConfigError(Exception):
    """The hazard-topics configuration could not be loaded or validated. The gate fails closed."""


def _parse_topics_yaml(text: str) -> dict[str, list[str]]:
    """A strict reader for the exact shape of hazard_topics.yaml: two top-level keys, each a list of plain
    string items. No external YAML dependency; anything outside this shape raises ConfigError (fail closed)."""
    out: dict[str, list[str]] = {}
    current: str | None = None
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if re.fullmatch(r"[a-z_]+:", line):                         # a top-level key, no indentation
            current = line[:-1]
            if current in out:
                raise ConfigError(f"line {n}: duplicate key {current!r}")
            out[current] = []
        elif re.match(r"\s*-\s+\S", line):
            if current is None:
                raise ConfigError(f"line {n}: list item before any key")
            item = line.split("-", 1)[1].strip().strip("'\"")
            if not item or not re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ\u00C0-\u024F0-9 /()-]+", item):
                raise ConfigError(f"line {n}: item {item!r} is not a plain noun")
            out[current].append(item)
        else:
            raise ConfigError(f"line {n}: not a key or a list item: {raw!r}")
    return out


DEFAULT_TOPICS_PATH = Path(str(files("coldsense.copilot").joinpath("hazard_topics.yaml")))


def load_topics(path: Path | None = None) -> HazardTopics:
    """Load and validate the topics file. Raises ConfigError on any problem (caller fails closed)."""
    p = Path(path or DEFAULT_TOPICS_PATH)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"cannot read hazard topics at {p}: {e}") from None
    if not text.strip():
        raise ConfigError("hazard topics file is empty")
    data = _parse_topics_yaml(text)
    if set(data) != {"protective_device", "hazardous_work"}:
        raise ConfigError(f"hazard topics must define exactly protective_device and hazardous_work; got {sorted(data)}")
    for key in ("protective_device", "hazardous_work"):
        items = data[key]
        if not items or not all(isinstance(x, str) and x.strip() for x in items):
            raise ConfigError(f"{key} must be a non-empty list of non-empty nouns")
        if any(not _tokens(x) for x in items):
            raise ConfigError(f"{key} has an item with no usable word")
    return HazardTopics(data["protective_device"], data["hazardous_work"])


# ---------------------------------------------------------------- classification
def _user_turns(conversation: list[dict]) -> list[str]:
    return [str(t.get("content", "")) for t in conversation if t.get("role") == "user"]


def _args_text(proposed_call: dict | None) -> str:
    if not proposed_call:
        return ""
    args = proposed_call.get("arguments", proposed_call.get("args", {})) or {}
    parts = [str(v) for v in args.values()] if isinstance(args, dict) else [str(args)]
    return " ".join(parts)


class HazardGate:
    """classify(conversation, proposed_call) -> GateDecision. Deterministic; the LLM and documents cannot
    change it. Construct with a topics object, or leave it None to signal a CONFIG_ERROR gate (fail closed)."""

    def __init__(self, topics: HazardTopics | None, config_error: str | None = None) -> None:
        self.topics = topics
        self.config_error = config_error

    @classmethod
    def load(cls, path: Path | None = None) -> HazardGate:
        try:
            return cls(load_topics(path))
        except ConfigError as e:
            log.error("hazard gate config error: %s", e)
            return cls(None, str(e))

    def classify(self, conversation: list[dict], proposed_call: dict | None = None) -> GateDecision:
        if self.topics is None:
            d = GateDecision(True, Tier.CONFIG_ERROR, reason=self.config_error or "topics unavailable",
                             trigger="config", config_ok=False)
            log.warning("gate CONFIG_ERROR: %s", self.config_error)
            return self._emit(d)

        users = _user_turns(conversation)
        current = users[-1] if users else ""
        window = users[:-1][-CONTEXT_WINDOW:]

        # 1. a proposed tool call whose arguments introduce a hazard topic absent from the user's request+context
        if proposed_call is not None:
            call_hit = self.topics.find(_args_text(proposed_call))
            conv_text = " ".join([current, *window])
            if call_hit is not None and self.topics.find(conv_text) is None:
                return self._emit(GateDecision(True, call_hit[0], call_hit[1], "proposed_args",
                                               "the proposed tool call introduces a hazard topic absent from "
                                               "the request and context"))

        # 2. the current user request itself
        hit = self.topics.find(current)
        if hit is not None:
            return self._emit(GateDecision(True, hit[0], hit[1], "user_request",
                                           f"the request names a {hit[0].value} topic"))

        # 3. a hazard topic in the recent context, carried forward only under the context rule
        ctx_hits = [h for t in window if (h := self.topics.find(t)) is not None]
        if ctx_hits:
            tier = Tier.PROTECTIVE_DEVICE if any(h[0] == Tier.PROTECTIVE_DEVICE for h in ctx_hits) else \
                Tier.HAZARDOUS_WORK
            name = next(h[1] for h in ctx_hits if h[0] == tier)
            referential = bool(REFERENTIAL.search(current))
            new_topic = bool(NEW_TOPIC.search(current))
            if referential and not new_topic:
                return self._emit(GateDecision(True, tier, name, "prior_turn",
                                               "the request refers back to an earlier hazard topic (split request)"))
            if new_topic and not referential:
                return self._emit(GateDecision(False, Tier.NONE, None, None,
                                               "the request changed to an unrelated topic"))
            return self._emit(GateDecision(True, tier, name, "ambiguous",
                                           "the relation to an earlier hazard topic is ambiguous; fail closed"))

        return self._emit(GateDecision(False, Tier.NONE, None, None, "no hazard topic in the request or context"))

    def _emit(self, d: GateDecision) -> GateDecision:
        log.info("gate decision %s", d.log_record())
        return d
