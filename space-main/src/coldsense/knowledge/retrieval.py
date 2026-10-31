"""Query → embedding → vector retrieval → ranking → bounded context → answer.

* Retrieval: cosine similarity of the query vector with every retrievable chunk (active, not quarantined,
  same embedding model, applicable model); the best ``candidates`` go on.
* LEXICAL provider. Ranking: BM25 over the active corpus, then cosine, then chunk id (deterministic). Relevance
  gate: the IDF-weighted share of the query's terms found in the passage reaches ``min_coverage``. A query term
  that no document contains weighs most, so questions the corpus does not cover are refused
  (NO_RELEVANT_DOCUMENTATION) instead of answered from loosely related text.
* SEMANTIC provider. Relevance: every exact identifier in the query (a token with a digit: model, error code,
  part number, value) must appear in the passage or its document identity, AND the cosine similarity reaches
  ``min_similarity`` (or, with the hybrid gate, the lexical coverage reaches ``min_coverage``). Ranking:
  reciprocal-rank fusion of the BM25 and cosine ranks, or cosine alone. The threshold, gate and ranking were
  selected on the DEVELOPMENT queries only (docs/KNOWLEDGE.md §9); without a selected threshold a SEMANTIC store
  refuses to answer rather than guess one.
* Answer: extractive only. Sentences are quoted verbatim from the passages they cite, as DOCUMENT_FACT; a
  difference in a stated quantity between documents is reported as a conflict, without choosing one.
Scores order passages; they are never probabilities or evidence that a document is right.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from coldsense.core.errors import DataError

from .chunking import sentences
from .embeddings import EmbeddingError, check_vectors, terms
from .schemas import (
    DATA_POLICY,
    Answer,
    AnswerStatus,
    Citation,
    Claim,
    ClaimType,
    Conflict,
    DocumentType,
    Passage,
    SourceLocation,
    Trust,
)
from .security import MAX_QUERY_CHARS, clean_invisible
from .store import KnowledgeStore


class RetrievalParams(BaseModel):
    """Every retrieval number. Defaults are provisional and were fixed before the evaluation set was written."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    k: int = Field(5, ge=1, le=20, description="passages returned at most")
    candidates: int = Field(50, ge=1, le=500, description="vector-retrieval candidates passed to ranking")
    min_coverage: float = Field(0.6, gt=0, le=1, description="IDF-weighted share of query terms a passage needs")
    min_claim_coverage: float = Field(0.5, gt=0, le=1, description="share a sentence needs to be quoted")
    max_claims: int = Field(3, ge=1, le=10)
    max_context_chars: int = Field(6000, ge=500, le=50000, description="total passage text returned")
    bm25_k1: float = Field(1.2, gt=0)
    bm25_b: float = Field(0.75, ge=0, le=1)
    # SEMANTIC providers only; None takes the value selected on DEVELOPMENT data for the store's model
    min_similarity: float | None = Field(None, gt=-1, le=1, description="cosine a passage needs (SEMANTIC)")
    semantic_gate: Literal["semantic", "hybrid"] | None = Field(
        None, description="semantic: similarity alone admits; hybrid: similarity or lexical coverage admits")
    semantic_ranking: Literal["cosine", "rrf"] | None = Field(
        None, description="cosine order, or reciprocal-rank fusion of BM25 and cosine ranks")
    rrf_k: int = Field(60, ge=1, description="reciprocal-rank fusion constant (Cormack et al. 2009)")
    min_sentence_similarity: float | None = Field(None, gt=-1, le=1.01, description="answerability: cosine a "
                                                  "sentence needs to count as addressing the query (SEMANTIC)")
    answer_contrast: bool | None = Field(None, description="answerability: reject a sentence that lacks a subject "
                                         "word the corpus uses elsewhere (unless its similarity suffices)")


_QUANTITY = re.compile(r"(?<![\w.])([-+]?\d+(?:[.,]\d+)?)\s*(°\s?[CF]|K|bar|psi|kPa|MPa|Pa|kW|W|kWh|V|A|mA|Hz|"
                       r"rpm|min|minutes|h|hours|s|seconds|%|mm|cm|m|kg|g|Nm|days|months)\b")
_UNIT_ALIASES = {"minutes": "min", "hours": "h", "seconds": "s", "° C": "°C", "° F": "°F"}


def quantities(text: str) -> dict[str, set[float]]:
    out: dict[str, set[float]] = {}
    for m in _QUANTITY.finditer(text):
        unit = _UNIT_ALIASES.get(m.group(2), m.group(2).replace(" ", ""))
        out.setdefault(unit, set()).add(float(m.group(1).replace(",", ".")))
    return out


def identity_terms(c: Citation) -> set[str]:
    """Terms that name the document or its equipment (title, manufacturer, model). They select WHICH document;
    they are not evidence that a passage answers the question."""
    return set(terms(" ".join(x for x in (c.title, c.manufacturer, c.model) if x)))


def answer_terms(q_terms: list[str], ident: set[str], model_terms: set[str]) -> list[str]:
    """The query terms that a passage must contain to count as answering: not the words naming the document or
    its equipment. If the query only names a document, the named model's terms are still set aside."""
    rest = [t for t in q_terms if t not in ident and t not in model_terms]
    return rest or [t for t in q_terms if t not in model_terms] or q_terms


def body_text(text: str, section: str | None, title: str | None) -> str:
    """The passage and its own headings, without the document title at the top of the section path."""
    parts = (section or "").split(" > ") if section else []
    if parts and title and parts[0] == title:
        parts = parts[1:]
    return " > ".join(parts) + "\n" + text


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


_IDENT = re.compile(r"[A-Za-z0-9]+(?:[-./][A-Za-z0-9]+)*")


def exact_identifiers(query: str) -> set[str]:
    """Tokens that must match exactly: any token containing a digit (EX-200, E3, SB-12, R-290, 230V), normalised."""
    return {_norm(t) for t in _IDENT.findall(query) if any(ch.isdigit() for ch in t)}


def token_grams(text: str, n: int = 4) -> set[str]:
    toks = [_norm(t) for t in re.findall(r"[A-Za-z0-9]+", text)]
    return {"".join(toks[i:j]) for i in range(len(toks)) for j in range(i + 1, min(i + n, len(toks)) + 1)}


def named_models(query: str, models: set[str]) -> set[str]:
    """Stated equipment models that the query names ('EX-200', 'ex 200', 'EX200'), matched on whole tokens."""
    toks = [_norm(t) for t in re.findall(r"[A-Za-z0-9]+", query)]
    grams = {"".join(toks[i:j]) for i in range(len(toks)) for j in range(i + 1, min(i + 4, len(toks)) + 1)}
    return {m for m in models if _norm(m) in grams}


class GroundedContext(BaseModel):
    """What a later answer generator (Phase 8) receives: the query, the cited passages and the data policy."""
    query: str
    passages: list[Passage]
    named_models: list[str] = Field(default_factory=list, description="stated models the query named")
    data_policy: str = DATA_POLICY

    def render(self) -> str:
        """Passages as delimited, labelled DATA blocks. Delimiters inside passage text are neutralised."""
        out = [f"POLICY: {self.data_policy}", f"QUERY: {self.query}"]
        for p in self.passages:
            body = p.text.replace("<<<", "‹‹‹").replace(">>>", "›››")
            c = p.citation
            out.append(f"<<<PASSAGE chunk={c.chunk_id} source=\"{c.title}; {c.location}\" trust={p.trust.value}>>>\n"
                       f"{body}\n<<<END PASSAGE>>>")
        return "\n\n".join(out)


class AnswerGenerator(Protocol):
    """Turns a grounded context into claims. The extractive generator below is the only one in Phase 7; an LLM
    generator (Phase 8) must keep every DOCUMENT_FACT verbatim with its citation, or add a separately validated
    claim type."""
    name: str

    def generate(self, context: GroundedContext, idf: dict[str, float],
                 params: RetrievalParams) -> tuple[list[Claim], list[Conflict]]: ...


class ExtractiveGenerator:
    name = "extractive-v1"

    def generate(self, context: GroundedContext, idf: dict[str, float],
                 params: RetrievalParams) -> tuple[list[Claim], list[Conflict]]:
        q = sorted(set(terms(context.query)))
        claims: list[Claim] = []
        model_terms = set(terms(" ".join(context.named_models)))
        for p in context.passages:
            aq = answer_terms(q, identity_terms(p.citation), model_terms)
            total = sum(idf[t] for t in aq)
            scored = []
            for line in p.text.split("\n"):
                for s in sentences(line) or ([line.strip()] if line.strip() else []):
                    have = set(terms(s))
                    cov = sum(idf[t] for t in aq if t in have) / total if total else 0.0
                    if cov >= params.min_claim_coverage and s in p.text:
                        scored.append((-cov, s))
            for _, s in sorted(scored)[:2]:
                if len(claims) < params.max_claims and all(c.text != s for c in claims):
                    claims.append(Claim(claim_type=ClaimType.DOCUMENT_FACT, text=s, citations=[p.citation],
                                        source_ids=[p.citation.chunk_id],
                                        note="quoted from the cited document; not checked or endorsed by ColdSense"))
        return claims, self.conflicts(claims, q)

    @staticmethod
    def conflicts(claims: list[Claim], q: list[str]) -> list[Conflict]:
        """Quoted sentences from different documents for the same model that state different values in one unit."""
        conflicts: list[Conflict] = []
        for i, a in enumerate(claims):
            for b in claims[i + 1:]:
                ca, cb = a.citations[0], b.citations[0]
                if ca.doc_id == cb.doc_id or not ca.model or not cb.model or _norm(ca.model) != _norm(cb.model):
                    continue                    # only documents stated for the same equipment can disagree
                qa, qb = quantities(a.text), quantities(b.text)
                shared = set(terms(a.text)) & set(terms(b.text)) & set(q)
                for unit in sorted(set(qa) & set(qb)):
                    if qa[unit] != qb[unit] and len(shared) >= 2:
                        conflicts.append(Conflict(quantity_unit=unit, statements=[a, b]))
        return conflicts


class Retriever:
    def __init__(self, store: KnowledgeStore, params: RetrievalParams | None = None,
                 generator: AnswerGenerator | None = None) -> None:
        self.store = store
        self.params = params or RetrievalParams()
        self.semantic = store.provider.kind == "SEMANTIC"
        if self.semantic:
            sel = getattr(store.provider, "selected_retrieval", None) or {}
            p = self.params
            upd = {k: sel[k] for k in ("min_similarity", "semantic_gate", "semantic_ranking")
                   if getattr(p, k) is None and k in sel}
            self.params = p.model_copy(update=upd)
            if None in (self.params.min_similarity, self.params.semantic_gate, self.params.semantic_ranking):
                raise EmbeddingError(f"no relevance threshold, gate and ranking have been selected for "
                                     f"{store.provider.model_id}; pass them explicitly (DEVELOPMENT data only)")
        opt_in = self.params.min_sentence_similarity is not None and self.params.answer_contrast is not None
        if generator is None and self.semantic and opt_in:       # answerability-v1: opt-in only (D-060)
            from .answerability import AnswerabilityGenerator
            generator = AnswerabilityGenerator(store, self.params.min_sentence_similarity,
                                               self.params.answer_contrast)
        self.generator = generator or ExtractiveGenerator()

    def _refuse(self, query: str, reason: str) -> Answer:
        return Answer(query=query, status=AnswerStatus.NO_RELEVANT_DOCUMENTATION, reason=reason,
                      embedding_model=self.store.provider.model_id, generator=self.generator.name,
                      limitations=self._limitations())

    def _limitations(self) -> list[str]:
        out = ["extractive: sentences are quoted, not interpreted; ColdSense does not check that a document is "
               "correct or that it applies to this equipment",
               "a quoted instruction is the manufacturer's or author's text, not a ColdSense recommendation"]
        if self.store.provider.kind == "LEXICAL":
            out.append(f"retrieval matches words ({self.store.provider.model_id}), not meaning: a question worded "
                       "differently from the documentation may find nothing")
        else:
            p = self.params
            out.append(f"semantic retrieval ({self.store.provider.model_id}; {p.semantic_gate} gate, cosine >= "
                       f"{p.min_similarity}, {p.semantic_ranking} ranking): similarity selects and orders passages; "
                       "it is not a probability and does not show that a passage answers the question")
            out.append("the embedding model is English only, and only the first 512 tokens of a passage are "
                       "embedded (lexical matching covers the whole passage)")
        return out

    def retrieve(self, query: str) -> tuple[list[Passage], dict[str, float], str | None, set[str]]:
        """Relevant passages in rank order, the IDF table, a reason when there are none, and the stated models
        the query named (documents for other stated models were excluded)."""
        p = self.params
        q_terms = sorted(set(terms(query)))
        if not q_terms:
            return [], {}, "the query has no content words to search for", set()
        rows, vecs = self.store.retrievable()
        if not rows:
            return [], {}, "the knowledge base has no retrievable passages", set()
        qv = check_vectors(self.store.provider, [query], query=True)[0]
        sims = vecs @ qv
        named = named_models(query, {r["model"] for r in rows if r["model"]})
        allowed = [i for i, r in enumerate(rows) if not named or r["model"] is None or r["model"] in named]
        model_terms = set(terms(" ".join(named)))
        order = sorted(allowed, key=lambda i: (-float(sims[i]), rows[i]["chunk_id"]))[: p.candidates]

        tfs = [Counter(json.loads(r["terms"])) for r in rows]
        n = len(rows)
        df = Counter(t for tf in tfs for t in set(tf) if t in q_terms)
        idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in q_terms}
        avg_len = sum(sum(tf.values()) for tf in tfs) / n
        if self.semantic:
            scored = self._semantic(query, rows, sims, order, tfs, idf, avg_len, q_terms, model_terms)
            return self._passages(scored, rows, sims, idf, named, semantic=True)
        scored = []
        for i in order:
            tf, length = tfs[i], sum(tfs[i].values())
            r = rows[i]
            body = set(terms(body_text(r["text"], json.loads(r["location"]).get("section"), r["title"])))
            matched = [t for t in q_terms if t in body]
            ident = set(terms(" ".join(x for x in (r["title"], r["manufacturer"], r["model"]) if x)))
            aq = answer_terms(q_terms, ident, model_terms)
            cov = sum(idf[t] for t in aq if t in body) / sum(idf[t] for t in aq)
            if cov < p.min_coverage or sims[i] <= 0:
                continue
            bm25 = sum(idf[t] * tf[t] * (p.bm25_k1 + 1) / (tf[t] + p.bm25_k1 * (1 - p.bm25_b + p.bm25_b * length
                                                                                  / avg_len)) for t in matched)
            scored.append((-round(bm25, 9), -round(float(sims[i]), 9), rows[i]["chunk_id"], i, cov, matched, bm25))
        scored.sort()
        return self._passages([(i, cov, matched, bm25, None, ["TERM_COVERAGE"]) for *_, i, cov, matched, bm25
                               in scored], rows, sims, idf, named, semantic=False)

    def _bm25(self, tf: Counter, length: int, avg_len: float, idf: dict[str, float], matched: list[str]) -> float:
        p = self.params
        return sum(idf[t] * tf[t] * (p.bm25_k1 + 1) / (tf[t] + p.bm25_k1 * (1 - p.bm25_b + p.bm25_b * length
                                                                             / avg_len)) for t in matched)

    def _semantic(self, query, rows, sims, order, tfs, idf, avg_len, q_terms, model_terms) -> list[tuple]:
        """SEMANTIC relevance and ranking over the candidates (already restricted to applicable models)."""
        p = self.params
        need = exact_identifiers(query)
        cands = []
        for i in order:
            r = rows[i]
            section = json.loads(r["location"]).get("section")
            body = set(terms(body_text(r["text"], section, r["title"])))
            matched = [t for t in q_terms if t in body]
            ident = set(terms(" ".join(x for x in (r["title"], r["manufacturer"], r["model"]) if x)))
            aq = answer_terms(q_terms, ident, model_terms)
            cov = sum(idf[t] for t in aq if t in body) / sum(idf[t] for t in aq)
            have = token_grams(" ".join(x for x in (r["text"], section, r["title"], r["manufacturer"], r["model"],
                                                     r["revision"]) if x))
            if not need <= have:
                continue                                   # an exact identifier of the query is absent
            basis = ["EXACT_IDENTIFIERS"] if need else []
            if float(sims[i]) >= p.min_similarity:
                basis.append("SEMANTIC_SIMILARITY")
            if p.semantic_gate == "hybrid" and cov >= p.min_coverage:
                basis.append("TERM_COVERAGE")
            if not set(basis) & {"SEMANTIC_SIMILARITY", "TERM_COVERAGE"}:
                continue
            cands.append((i, cov, matched, self._bm25(tfs[i], sum(tfs[i].values()), avg_len, idf, matched), basis))
        if p.semantic_ranking == "cosine":
            return [(i, cov, m, b, None, basis) for i, cov, m, b, basis in
                    sorted(cands, key=lambda c: (-round(float(sims[c[0]]), 9), rows[c[0]]["chunk_id"]))]
        cos_rank = {c[0]: k for k, c in enumerate(sorted(cands, key=lambda c: (-float(sims[c[0]]),
                                                                               rows[c[0]]["chunk_id"])), 1)}
        lex = sorted((c for c in cands if c[3] > 0), key=lambda c: (-c[3], rows[c[0]]["chunk_id"]))
        lex_rank = {c[0]: k for k, c in enumerate(lex, 1)}
        out = []
        for i, cov, m, b, basis in cands:
            fused = 1 / (p.rrf_k + cos_rank[i]) + (1 / (p.rrf_k + lex_rank[i]) if i in lex_rank else 0.0)
            out.append((-round(fused, 12), -round(float(sims[i]), 9), rows[i]["chunk_id"], (i, cov, m, b, fused,
                                                                                            basis)))
        return [x[-1] for x in sorted(out)]

    def _passages(self, scored, rows, sims, idf, named, semantic: bool):
        p = self.params
        passages, used = [], 0
        for i, cov, matched, bm25, fused, basis in scored:
            r = rows[i]
            if len(passages) >= p.k or used + len(r["text"]) > p.max_context_chars:
                break
            used += len(r["text"])
            loc = SourceLocation.model_validate_json(r["location"])
            cit = Citation(doc_id=r["doc_id"], version=r["version"],
                           title=r["title"] or f"{r['filename']} (no title in the document)", revision=r["revision"],
                           doc_type=DocumentType(r["doc_type"]), manufacturer=r["manufacturer"], model=r["model"],
                           chunk_id=r["chunk_id"], location=loc.label(), filename=r["filename"])
            sents = [s for line in r["text"].split("\n") for s in (sentences(line) or [line])]
            best = sorted(sents, key=lambda s: (-len(set(terms(s)) & set(matched)), sents.index(s)))[:2]
            excerpt = " … ".join(s for s in sents if s in best)
            passages.append(Passage(rank=len(passages) + 1, citation=cit, text=r["text"], excerpt=excerpt,
                                    trust=Trust(r["trust"]), flags=json.loads(r["flags"]),
                                    vector_similarity=round(float(sims[i]), 6), lexical_score=round(bm25, 6),
                                    term_coverage=round(cov, 6), matched_terms=matched,
                                    retrieval_method=f"{'SEMANTIC' if semantic else 'LEXICAL'} "
                                                     f"{self.store.provider.model_id}",
                                    relevance_basis=basis, fusion_score=None if fused is None else round(fused, 9),
                                    section=loc.section))
        if not passages and semantic:
            return [], idf, (f"no passage passed the semantic relevance rule (every identifier in the query present, "
                             f"and cosine similarity >= {p.min_similarity}"
                             + (f" or {p.min_coverage:.0%} of the query's term weight" if p.semantic_gate == "hybrid"
                                else "")
                             + (f"; restricted to documents for {', '.join(sorted(named))}" if named else "")
                             + "); the documentation indexed here does not cover this question"), named
        if not passages:
            return [], idf, (f"no passage contains enough of the query's terms other than document and model names"
                             f" (needs {p.min_coverage:.0%} of their IDF weight"
                             + (f"; restricted to documents for {', '.join(sorted(named))}" if named else "")
                             + "); the documentation indexed here does not cover this question"), named
        return passages, idf, None, named

    def answer(self, query: str) -> Answer:
        query = clean_invisible(query).text.strip()
        if len(query) > MAX_QUERY_CHARS:
            raise DataError(f"query longer than {MAX_QUERY_CHARS} characters")
        passages, idf, reason, named = self.retrieve(query)
        if not passages:
            return self._refuse(query, reason or "no relevant documentation")
        ctx = GroundedContext(query=query, passages=passages, named_models=sorted(named))
        claims, conflicts = self.generator.generate(ctx, idf, self.params)
        status = (AnswerStatus.CONFLICTING_SOURCES if conflicts else AnswerStatus.EXTRACTIVE_ANSWER if claims
                  else AnswerStatus.PASSAGES_ONLY)
        lim = self._limitations()
        if named:
            lim.append(f"restricted to documents stated for {', '.join(sorted(named))}; documents without "
                       "a stated model were kept")
        if any(p.trust == Trust.FLAGGED for p in passages):
            lim.append("a cited document is FLAGGED: parts of it were quarantined or redacted at ingestion")
        explained = getattr(self.generator, "last_explanation", None)
        return Answer(query=query, status=status, claims=claims, passages=passages, conflicts=conflicts,
                      reason=None if claims else explained or "relevant passages found, but no single sentence "
                      "covers the question; read the passages", embedding_model=self.store.provider.model_id,
                      generator=self.generator.name, limitations=lim)
