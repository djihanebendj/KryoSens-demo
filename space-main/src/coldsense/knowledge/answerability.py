"""Answerability: whether retrieved evidence states the fact a question asks for (docs/KNOWLEDGE.md §10).

Retrieval decides that a passage is RELATED to a query. This module decides, per sentence, whether the passage
SUPPORTS an answer. Only a supporting sentence is quoted as a DOCUMENT_FACT; related passages without one are
returned as context only (PASSAGES_ONLY), never as an answer.

A sentence (with its own heading) supports the query when every rule below holds. The rules are deterministic
and readable; the two numbers (sentence similarity and the passage threshold used by retrieval) are selected on
DEVELOPMENT data only.

1. Answer type: if the query asks for a kind of value (an interval or duration, a temperature, a voltage, a
   current, a resistance, a length or height, a humidity, a charge or mass, a pressure, a refrigerant type, a
   tolerance), the sentence states a value of that kind. "What humidity" is not answered by "+2 to +8 °C".
2. Action: if the query asks about an action (bypass or disable, clean, replace, adjust, install, reset,
   inspect, recharge, export, repair), the sentence names that action or a listed synonym. "Can the high pressure
   switch be bypassed" is not answered by "E3 means the high pressure switch has tripped".
3. Identifiers: every exact identifier of the query (a token with a digit) is in the sentence, its heading or
   the document's identity.
4. Subject: the sentence covers the query's terms (the extractive rule, ``min_claim_coverage``), or its cosine
   similarity to the query reaches ``min_sentence_similarity`` (SEMANTIC stores only).
5. Contrast (optional, selected on DEVELOPMENT): a subject word that the indexed documents use elsewhere but
   that this sentence and its heading do not contain marks the sentence as being about something else, unless
   the sentence similarity reaches ``min_sentence_similarity``.
"""

from __future__ import annotations

import json
import re

import numpy as np

from .chunking import sentences
from .embeddings import check_vectors, stem, terms
from .retrieval import (
    ExtractiveGenerator,
    GroundedContext,
    RetrievalParams,
    body_text,
    exact_identifiers,
    identity_terms,
    token_grams,
)
from .schemas import Claim, ClaimType, Conflict

_NUM = r"(?<![\w.])[-+]?\d+(?:[.,]\d+)?\s*"
EVIDENCE = {
    "TIME": re.compile(_NUM + r"(?:s|sec|seconds?|min|minutes?|h|hrs?|hours?|days?|weeks?|months?|years?)\b"
                       r"|\b(?:every|each|once|twice)\b|\b(?:daily|weekly|monthly|yearly|annually)\b", re.I),
    "TEMPERATURE": re.compile(_NUM + r"(?:°\s?[CF]\b|K\b)"),
    "VOLTAGE": re.compile(_NUM + r"(?:V|kV|Hz|W|kW)\b"),
    "CURRENT": re.compile(_NUM + r"(?:A|mA)\b"),
    "RESISTANCE": re.compile(_NUM + r"(?:ohms?|Ω|kΩ)", re.I),
    "LENGTH": re.compile(_NUM + r"(?:mm|cm|m)\b"),
    "HUMIDITY": re.compile(_NUM + r"%\s*(?:RH|relative humidity)?|\bhumidity\b[^.]*\d", re.I),
    "MASS": re.compile(_NUM + r"(?:g|kg)\b"),
    "PRESSURE": re.compile(_NUM + r"(?:bar|psi|kPa|MPa|Pa)\b"),
    "REFRIGERANT": re.compile(r"\bR-?\d{2,4}[A-Z]?\b"),
    "TOLERANCE": re.compile(_NUM + r"(?:°\s?[CF]|K|%|V|A|bar|mm)\b|±"),
}
# query cues -> the answer type they ask for (stemmed single terms, or lower-case phrases)
TYPE_TERMS = {
    "TIME": {"interval", "duration", "delay", "time", "frequency", "period", "often", "frequently", "when",
             "minute", "hour"},
    "TEMPERATURE": {"temperature", "setpoint", "degree"},
    "VOLTAGE": {"voltage", "volt", "electricity", "supply", "power"},
    "CURRENT": {"current", "amperage", "amp", "ampere", "draw"},
    "RESISTANCE": {"resistance", "ohm"},
    "LENGTH": {"height", "distance", "clearance"},
    "HUMIDITY": {"humidity"},
    "MASS": {"charge", "amount", "weigh", "weight"},
    "PRESSURE": {"pressure"},
    "TOLERANCE": {"tolerance", "accuracy", "deviation"},
}
TYPE_PHRASES = {"TIME": ("how long", "how often", "how frequently", "how many minutes", "how many hours"),
                "LENGTH": ("how high", "how far"), "TEMPERATURE": ("how cold", "how warm"),
                "REFRIGERANT": ("which refrigerant", "what refrigerant", "refrigerant gas", "refrigerant type",
                                "type of refrigerant"),
                "TOLERANCE": ("acceptable error",)}
# terms that name a component rather than ask for a value when they modify another noun
_MODIFIER_ONLY = {"PRESSURE": ("pressure switch", "pressure safety switch"),
                  "TEMPERATURE": ("temperature sensor", "temperature probe", "high-temperature", "temperature alarm")}
ACTIONS = {
    "DEFEAT": {"bypass", "override", "defeat", "jumper", "disable", "deactivate", "bridge", "bridg"},
    "CLEAN": {"clean", "wash", "rinse"},
    "REPLACE": {"replace", "renew", "swap"},
    "ADJUST": {"change", "adjust", "modify"},
    "INSTALL": {"install", "mount", "fit", "place", "position"},
    "RESET": {"reset"},
    "INSPECT": {"inspect", "check", "look", "examine"},
    "RECHARGE": {"recharge", "refill"},
    "EXPORT": {"export", "download", "transfer"},
    "REPAIR": {"repair", "fix"},
}
_ACTION_OF = {stem(w): g for g, ws in ACTIONS.items() for w in ws}
_CUE_TERMS = {stem(w) for ws in TYPE_TERMS.values() for w in ws} | set(_ACTION_OF)


def answer_types(query: str) -> set[str]:
    q = query.lower()
    qt = set(terms(query))
    out = {t for t, ws in TYPE_TERMS.items() if qt & {stem(w) for w in ws}}
    out |= {t for t, ps in TYPE_PHRASES.items() if any(p in q for p in ps)}
    for t, mods in _MODIFIER_ONLY.items():
        if t in out and any(m in q for m in mods) and not any(p in q for p in TYPE_PHRASES.get(t, ())):
            rest = q
            for m in mods:
                rest = rest.replace(m, " ")
            if not set(terms(rest)) & {stem(w) for w in TYPE_TERMS[t]}:
                out.discard(t)
    return out


def actions(query: str) -> set[str]:
    return {_ACTION_OF[t] for t in terms(query) if t in _ACTION_OF}


class AnswerabilityGenerator:
    """Quotes only sentences that state the requested fact. Same output contract as ``extractive-v1``."""
    name = "answerability-v1"

    def __init__(self, store, min_sentence_similarity: float, contrast: bool,
                 vector_cache: dict[str, np.ndarray] | None = None) -> None:
        self.store, self.min_sentence_similarity, self.contrast = store, min_sentence_similarity, contrast
        self._vectors: dict[str, np.ndarray] = {} if vector_cache is None else vector_cache
        self._vocab: tuple[tuple[str, ...], set[str]] | None = None
        self.last_explanation: str | None = None

    def _vocabulary(self) -> set[str]:
        """Terms of every retrievable passage body (document titles, manufacturers and models excluded)."""
        rows, _ = self.store.retrievable()
        key = tuple(r["chunk_id"] for r in rows)
        if self._vocab is None or self._vocab[0] != key:
            self._vocab = (key, {t for r in rows for t in terms(body_text(
                r["text"], json.loads(r["location"]).get("section"), r["title"]))})
        return self._vocab[1]

    def _sims(self, query: str, texts: list[str]) -> list[float]:
        if self.store.provider.kind != "SEMANTIC" or not texts:
            return [-1.0] * len(texts)
        missing = [t for t in dict.fromkeys(texts) if t not in self._vectors]
        if missing:
            for t, v in zip(missing, check_vectors(self.store.provider, missing), strict=True):
                self._vectors[t] = v
        qv = check_vectors(self.store.provider, [query], query=True)[0]
        return [float(self._vectors[t] @ qv) for t in texts]

    def supporting(self, context: GroundedContext, idf: dict[str, float],
                   params: RetrievalParams) -> list[tuple[float, str, int, list[str]]]:
        """(-score, sentence, passage index, reasons) for every sentence that supports the query."""
        query = context.query
        q = sorted(set(terms(query)))
        types, acts, need = answer_types(query), actions(query), exact_identifiers(query)
        model_terms = set(terms(" ".join(context.named_models)))
        vocab = self._vocabulary() if self.contrast else set()
        cands = []
        for pi, p in enumerate(context.passages):
            heading = p.section or ""
            ident = identity_terms(p.citation)
            subject = [t for t in q if t not in ident and t not in model_terms and t not in _CUE_TERMS]
            aq = [t for t in q if t not in ident and t not in model_terms] or q
            total = sum(idf.get(t, 0.0) for t in aq)
            doc_ident = " ".join(x for x in (p.citation.title, p.citation.manufacturer, p.citation.model,
                                             p.citation.revision) if x)
            for line in p.text.split("\n"):
                for s in sentences(line) or ([line.strip()] if line.strip() else []):
                    if s in p.text:
                        cands.append((pi, s, heading, subject, aq, total, doc_ident))
        sims = self._sims(query, [f"{c[2]}\n{c[1]}" for c in cands])
        out = []
        for (pi, s, heading, subject, aq, total, doc_ident), sim in zip(cands, sims, strict=True):
            where = f"{heading}\n{s}"
            have = set(terms(body_text(s, heading, context.passages[pi].citation.title)))
            reasons = []
            if types and not any(EVIDENCE[t].search(s) for t in types):
                continue
            if types:
                reasons.append("ANSWER_TYPE:" + "/".join(sorted(types)))
            if acts and not any(_ACTION_OF.get(t) in acts for t in have):
                continue
            if acts:
                reasons.append("ACTION:" + "/".join(sorted(acts)))
            if need and not need <= token_grams(f"{where}\n{doc_ident}"):
                continue
            cov = sum(idf.get(t, 0.0) for t in aq if t in have) / total if total else 0.0
            semantic_ok = sim >= self.min_sentence_similarity
            if cov >= params.min_claim_coverage:
                reasons.append("TERM_COVERAGE")
            if semantic_ok:
                reasons.append("SENTENCE_SIMILARITY")
            if not ({"TERM_COVERAGE", "SENTENCE_SIMILARITY"} & set(reasons)):
                continue
            if self.contrast and not semantic_ok and any(t in vocab and t not in have for t in subject):
                continue
            out.append((-(max(sim, 0.0) + cov), s, pi, reasons))
        return sorted(out, key=lambda x: (x[0], x[2], x[1]))

    def generate(self, context: GroundedContext, idf: dict[str, float],
                 params: RetrievalParams) -> tuple[list[Claim], list[Conflict]]:
        claims: list[Claim] = []
        per_passage: dict[int, int] = {}
        for _, s, pi, reasons in self.supporting(context, idf, params):
            p = context.passages[pi]
            if len(claims) >= params.max_claims or per_passage.get(pi, 0) >= 2 or any(c.text == s for c in claims):
                continue
            per_passage[pi] = per_passage.get(pi, 0) + 1
            claims.append(Claim(claim_type=ClaimType.DOCUMENT_FACT, text=s, citations=[p.citation],
                                source_ids=[p.citation.chunk_id],
                                note="quoted from the cited document; supports the question by "
                                     + ", ".join(reasons) + "; not checked or endorsed by ColdSense"))
        types, acts = answer_types(context.query), actions(context.query)
        need = ([f"a value of type {'/'.join(sorted(types))}"] if types else []) + \
            ([f"the action {'/'.join(sorted(acts))}"] if acts else [])
        self.last_explanation = None if claims else (
            "context only: the passages are related to the question, but no sentence in them states the requested "
            "fact" + (f" ({'; '.join(need)})" if need else "") + "; this is not a supported answer")
        return claims, ExtractiveGenerator.conflicts(claims, sorted(set(terms(context.query))))
