"""Deterministic retrieval evaluation on the synthetic corpus in ``eval_corpus/`` (fictional documents).

Relevance labels are (document id, text contained in the section path). Metrics per answerable query: Recall@k
(share of relevant labels with a passage in the top k), reciprocal rank of the first relevant passage, nDCG@5
(binary gain, each label counted once). Unanswerable queries must return NO_RELEVANT_DOCUMENTATION. Every quoted
claim must cite a relevant passage. Quarantined text must never be returned.

These numbers describe this tiny synthetic set only; they are not an estimate of production retrieval quality.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from .embeddings import get_provider
from .retrieval import RetrievalParams, Retriever
from .schemas import AnswerStatus, DocumentMetadata
from .store import KnowledgeStore

CORPUS = Path(__file__).parent / "eval_corpus"
FIXED_TIME = datetime(2026, 1, 1, tzinfo=UTC)
QUARANTINE_PROBE = "bypass the high pressure switch"


class QueryResult(BaseModel):
    id: str
    category: str
    query: str
    status: AnswerStatus
    ranked_chunks: list[str]
    first_relevant_rank: int | None
    recall_at: dict[int, float] = Field(default_factory=dict)
    ndcg_at_5: float | None = None
    claims_cite_relevant: int = 0
    claims_total: int = 0


class EvalReport(BaseModel):
    corpus: str
    embedding_model: str
    n_documents: int
    n_chunks: int
    n_quarantined: int
    queries: list[QueryResult]
    metrics: dict[str, dict[str, float | int]]
    quarantine_leaks: int
    note: str = ("synthetic, fictional corpus of a few documents with labels written before retrieval was run; "
                 "the numbers say how this pipeline behaves on it, not how it would perform on real manuals")


def build_store(path: Path) -> KnowledgeStore:
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    store = KnowledgeStore(path)
    for d in manifest["documents"]:
        meta = DocumentMetadata(**{k: v for k, v in d.items() if k != "file"})
        store.ingest(CORPUS / d["file"], meta, now=FIXED_TIME)
    return store


def _relevant(label_doc: str, label_section: str, doc_id: str, section: str | None) -> bool:
    return doc_id == label_doc and label_section in (section or "")


def evaluate(params: RetrievalParams | None = None, store_path: Path | None = None) -> EvalReport:
    queries = [q for f in ("queries.json", "queries_holdout.json")
               for q in json.loads((CORPUS / f).read_text())["queries"]]
    with tempfile.TemporaryDirectory() as tmp:
        store = build_store(store_path or Path(tmp) / "eval.sqlite")
        try:
            retriever = Retriever(store, params or RetrievalParams(k=5))
            docs = store.documents()
            results, leaks = [], 0
            for q in queries:
                ans = retriever.answer(q["query"])
                labels = [tuple(x) for x in q["relevant"]]
                ranked, hits = [], []
                for p in ans.passages:
                    chunk = next(c for c in store.chunks(p.citation.doc_id) if c.chunk_id == p.citation.chunk_id)
                    ranked.append(p.citation.chunk_id)
                    hits.append({lab for lab in labels if _relevant(*lab, chunk.doc_id, chunk.location.section)})
                    leaks += QUARANTINE_PROBE in p.text.lower()
                r = QueryResult(id=q["id"], category=q["category"], query=q["query"], status=ans.status,
                                ranked_chunks=ranked, first_relevant_rank=None)
                if labels:
                    r.first_relevant_rank = next((i + 1 for i, h in enumerate(hits) if h), None)
                    for k in (1, 3, 5):
                        found = set().union(*hits[:k]) if hits[:k] else set()
                        r.recall_at[k] = len(found) / len(labels)
                    seen, dcg = set(), 0.0
                    for i, h in enumerate(hits[:5]):
                        new = h - seen
                        if new:
                            dcg += 1 / math.log2(i + 2)
                            seen |= new
                    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(labels), 5)))
                    r.ndcg_at_5 = dcg / ideal
                    rel_chunks = {c for c, h in zip(ranked, hits, strict=True) if h}
                    r.claims_total = len(ans.claims)
                    r.claims_cite_relevant = sum(all(cit.chunk_id in rel_chunks for cit in c.citations)
                                                 for c in ans.claims)
                results.append(r)
            metrics = {}
            for cat in ("keyword", "paraphrase", "unanswerable", "holdout-keyword", "holdout-paraphrase",
                        "holdout-unanswerable"):
                rs = [r for r in results if r.category == cat]
                if not rs:
                    continue
                if cat.endswith("unanswerable"):
                    refused = sum(r.status == AnswerStatus.NO_RELEVANT_DOCUMENTATION for r in rs)
                    metrics[cat] = {"queries": len(rs), "refused": refused}
                    continue
                metrics[cat] = {
                    "queries": len(rs),
                    "recall@1": round(sum(r.recall_at[1] for r in rs) / len(rs), 4),
                    "recall@3": round(sum(r.recall_at[3] for r in rs) / len(rs), 4),
                    "recall@5": round(sum(r.recall_at[5] for r in rs) / len(rs), 4),
                    "mrr@5": round(sum(1 / r.first_relevant_rank if r.first_relevant_rank else 0 for r in rs)
                                   / len(rs), 4),
                    "ndcg@5": round(sum(r.ndcg_at_5 or 0 for r in rs) / len(rs), 4),
                    "false_refusals": sum(r.status == AnswerStatus.NO_RELEVANT_DOCUMENTATION for r in rs),
                    "claims": sum(r.claims_total for r in rs),
                    "claims_citing_relevant": sum(r.claims_cite_relevant for r in rs),
                }
            return EvalReport(corpus=str(CORPUS.name), embedding_model=store.provider.model_id, n_documents=len(docs),
                              n_chunks=sum(d.n_chunks for d in docs), n_quarantined=sum(d.n_quarantined for d in docs),
                              queries=results, metrics=metrics, quarantine_leaks=leaks)
        finally:
            store.close()


# ====================================================================== DEVELOPMENT / HELD_OUT (corpus v2)
SPLITS = {"DEVELOPMENT": "queries_development.json", "HELD_OUT": "queries_heldout_v2.json"}   # dev-v1, heldout-v2
# every labelled set by version; heldout-v2 is consumed (inspected after its run), heldout-v3 is the current one
DATASETS = {"dev-v1": "queries_development.json", "heldout-v2": "queries_heldout_v2.json",
            "dev-v2": "queries_development_v2.json", "heldout-v3": "queries_heldout_v3.json"}
MANIFESTS = {"eval-corpus-v2": "manifest_v2.json", "eval-corpus-v3": "manifest_v3.json"}
GROUPS = {"exact": ("exact", "identifier", "trap"), "paraphrase": ("paraphrase",), "conflict": ("conflict",),
          "refusal": ("unanswerable", "trap-unanswerable", "quarantine")}


class SplitReport(BaseModel):
    split: str
    dataset: str
    dataset_sha256: str
    corpus: str
    embedding_model: str
    embedding_identity: dict | None
    params: dict
    queries: list[dict]
    metrics: dict[str, dict]
    quarantine_leaks: int
    wrong_model_errors: int
    generator: str = ""
    note: str = ("SYNTHETIC, fictional corpus of a few documents; the numbers say how this pipeline behaves on it, "
                 "not how it would perform on real manuals")


def build_store_v2(path: Path, provider=None, corpus: str = "eval-corpus-v2") -> KnowledgeStore:
    manifest = json.loads((CORPUS / MANIFESTS[corpus]).read_text())
    store = KnowledgeStore(path, provider)
    for d in manifest["documents"]:
        store.ingest(CORPUS / d["file"], DocumentMetadata(**{k: v for k, v in d.items() if k != "file"}),
                     now=FIXED_TIME)
    return store


def load_split(split: str) -> tuple[dict, str]:
    raw = (CORPUS / SPLITS[split]).read_bytes()
    data = json.loads(raw)
    if data["split"] != split:
        raise ValueError(f"{SPLITS[split]} is labelled {data['split']}, not {split}")
    return data, hashlib.sha256(raw).hexdigest()


def load_dataset(version: str) -> tuple[dict, str]:
    raw = (CORPUS / DATASETS[version]).read_bytes()
    data = json.loads(raw)
    if data["version"] != version:
        raise ValueError(f"{DATASETS[version]} is version {data['version']}, not {version}")
    return data, hashlib.sha256(raw).hexdigest()


def run_split(store: KnowledgeStore, split: str, params: RetrievalParams | None = None) -> SplitReport:
    """Every query of one split (dev-v1 / heldout-v2) against an already built v2 store."""
    from .retrieval import ExtractiveGenerator  # these sets were evaluated with extractive-v1 (D-059)
    return run_dataset(store, {"DEVELOPMENT": "dev-v1", "HELD_OUT": "heldout-v2"}[split], params,
                       Retriever(store, params or RetrievalParams(k=5), ExtractiveGenerator()))


SUPPORTED = (AnswerStatus.EXTRACTIVE_ANSWER.value, AnswerStatus.CONFLICTING_SOURCES.value)
ANSWER_GROUPS = {"exact": ("exact", "identifier", "trap"), "paraphrase": ("paraphrase",), "safety": ("safety",),
                 "conflict": ("conflict",), "related-unanswerable": ("related-unanswerable",),
                 "unanswerable": ("unanswerable", "trap-unanswerable", "quarantine")}


def answerability_metrics(rows: list[dict]) -> dict[str, dict]:
    """Supported answers, kept apart from retrieval. A supported answer is EXTRACTIVE_ANSWER or CONFLICTING_SOURCES
    (quoted sentences); PASSAGES_ONLY is context only and counts as NOT answering."""
    out = {}
    for name, cats in ANSWER_GROUPS.items():
        rs = [r for r in rows if r["category"] in cats]
        if not rs:
            continue
        a = [r for r in rs if r["answerable"]]
        u = [r for r in rs if not r["answerable"]]
        m: dict = {"queries": len(rs)}
        if a:
            m["supported_answer_rate"] = round(sum(r["supported_correct"] for r in a) / len(a), 4)
            m["context_only"] = sum(r["status"] == AnswerStatus.PASSAGES_ONLY.value for r in a)
            m["refused"] = sum(r["status"] == AnswerStatus.NO_RELEVANT_DOCUMENTATION.value for r in a)
            m["unsupported_quotes"] = sum(r["claims"] - r["claims_citing_relevant"] for r in a)
        if u:
            m["refusal_accuracy"] = round(sum(r["status"] not in SUPPORTED for r in u) / len(u), 4)
            m["false_support"] = sum(r["status"] in SUPPORTED for r in u)
            m["context_only"] = sum(r["status"] == AnswerStatus.PASSAGES_ONLY.value for r in u)
            m["no_documentation"] = sum(r["status"] == AnswerStatus.NO_RELEVANT_DOCUMENTATION.value for r in u)
        claims = sum(r["claims"] for r in rs)
        m["source_attribution_accuracy"] = (round(sum(r["claims_citing_relevant"] for r in rs) / claims, 4)
                                            if claims else None)
        if name == "conflict":
            m["conflicting_sources_status"] = sum(r["status"] == AnswerStatus.CONFLICTING_SOURCES.value for r in rs)
        out[name] = m
    return out


def run_dataset(store: KnowledgeStore, version: str, params: RetrievalParams | None = None,
                retriever: Retriever | None = None) -> SplitReport:
    """Every query of one labelled set against an already built store for its corpus."""
    data, sha = load_dataset(version)
    split = data["split"]
    retriever = retriever or Retriever(store, params or RetrievalParams(k=5))
    quarantined = {c.chunk_id for d in store.documents() for c in store.chunks(d.doc_id) if c.quarantined}
    section = {c.chunk_id: (c.doc_id, c.location.section) for d in store.documents() for c in store.chunks(d.doc_id)}
    rows, leaks, wrong = [], 0, 0
    for q in data["queries"]:
        ans = retriever.answer(q["query"])
        labels = [tuple(x) for x in q["relevant"]]
        hits = [{lab for lab in labels if _relevant(*lab, *section[p.citation.chunk_id])} for p in ans.passages]
        leak = sum(p.citation.chunk_id in quarantined or QUARANTINE_PROBE in p.text.lower() for p in ans.passages)
        wm = sum(bool(q.get("model")) and p.citation.model is not None and p.citation.model != q["model"]
                 for p in ans.passages)
        leaks, wrong = leaks + leak, wrong + wm
        rel_chunks = {p.citation.chunk_id for p, h in zip(ans.passages, hits, strict=True) if h}
        first = next((i + 1 for i, h in enumerate(hits) if h), None)
        rows.append({"id": q["id"], "category": q["category"], "query": q["query"], "status": ans.status.value,
                     "answerable": bool(labels), "first_relevant_rank": first,
                     "recall@1": len(set().union(*hits[:1])) / len(labels) if labels and hits else (0.0 if labels
                                                                                                   else None),
                     "recall@5": len(set().union(*hits[:5])) / len(labels) if labels and hits else (0.0 if labels
                                                                                                   else None),
                     "claims": len(ans.claims),
                     "claims_citing_relevant": sum(all(c.chunk_id in rel_chunks for c in cl.citations)
                                                   for cl in ans.claims),
                     "wrong_model_passages": wm, "quarantine_leaks": leak,
                     "supported_correct": ans.status.value in SUPPORTED and bool(labels) and any(
                         all(c.chunk_id in rel_chunks for c in cl.citations) for cl in ans.claims),
                     "ranked": [p.citation.chunk_id for p in ans.passages],
                     "similarities": [p.vector_similarity for p in ans.passages]})
    metrics = {}
    groups = GROUPS if version in ("dev-v1", "heldout-v2") else {
        "exact": ("exact", "identifier", "trap"), "paraphrase": ("paraphrase",), "safety": ("safety",),
        "conflict": ("conflict",)}
    for name, cats in groups.items():
        rs = [r for r in rows if r["category"] in cats]
        if not rs:
            continue
        ans_rs, un_rs = [r for r in rs if r["answerable"]], [r for r in rs if not r["answerable"]]
        claims = sum(r["claims"] for r in rs)
        m = {"queries": len(rs), "answerable": len(ans_rs), "unanswerable": len(un_rs)}
        if ans_rs:
            m.update({"recall@1": round(sum(r["recall@1"] for r in ans_rs) / len(ans_rs), 4),
                      "recall@5": round(sum(r["recall@5"] for r in ans_rs) / len(ans_rs), 4),
                      "mrr@5": round(sum(1 / r["first_relevant_rank"] if r["first_relevant_rank"] else 0
                                         for r in ans_rs) / len(ans_rs), 4),
                      "false_refusals": sum(r["status"] == AnswerStatus.NO_RELEVANT_DOCUMENTATION.value
                                            for r in ans_rs)})
        if un_rs:
            m["refusal_accuracy"] = round(sum(r["status"] == AnswerStatus.NO_RELEVANT_DOCUMENTATION.value
                                              for r in un_rs) / len(un_rs), 4)
        m["claims"] = claims
        m["source_attribution_accuracy"] = (round(sum(r["claims_citing_relevant"] for r in rs) / claims, 4)
                                            if claims else None)
        m["wrong_model_errors"] = sum(r["wrong_model_passages"] for r in rs)
        if name == "conflict":
            m["conflicting_sources_status"] = sum(r["status"] == AnswerStatus.CONFLICTING_SOURCES.value for r in rs)
        metrics[name] = m
    meta = store.meta()
    if version not in ("dev-v1", "heldout-v2"):
        metrics = {"retrieval": metrics, "answerability": answerability_metrics(rows)}
    return SplitReport(split=split, dataset=data["version"], dataset_sha256=sha, corpus=data["corpus"],
                       embedding_model=store.provider.model_id,
                       embedding_identity=json.loads(meta["embedding_identity"]) if "embedding_identity" in meta
                       else None, params=retriever.params.model_dump(), queries=rows, metrics=metrics,
                       quarantine_leaks=leaks, wrong_model_errors=wrong, generator=retriever.generator.name)


def evaluate_split(split: str, embedding: str = "lexical-hash-v1",
                   params: RetrievalParams | None = None) -> SplitReport:
    from .retrieval import ExtractiveGenerator
    return evaluate_dataset({"DEVELOPMENT": "dev-v1", "HELD_OUT": "heldout-v2"}[split], embedding, params,
                            generator=lambda _: ExtractiveGenerator())


def evaluate_dataset(version: str, embedding: str = "lexical-hash-v1",
                     params: RetrievalParams | None = None, generator=None) -> SplitReport:
    data, _ = load_dataset(version)
    with tempfile.TemporaryDirectory() as tmp:
        store = build_store_v2(Path(tmp) / "eval.sqlite", get_provider(embedding), data["corpus"])
        try:
            r = None
            if generator is not None:
                r = Retriever(store, params or RetrievalParams(k=5), generator(store))
            return run_dataset(store, version, params, r)
        finally:
            store.close()


def _correct(r: dict) -> bool:
    refused = r["status"] == AnswerStatus.NO_RELEVANT_DOCUMENTATION.value
    return (not refused and (r["recall@5"] or 0) > 0 and not r["wrong_model_passages"]) if r["answerable"] \
        else refused


def select_on_development(policies=("plain", "instruct"), grid=None) -> dict:
    """Choose query policy, gate, ranking and similarity threshold on DEVELOPMENT only. Objective: the number of
    DEVELOPMENT queries handled correctly (answerable: not refused, a relevant passage in the top 5 and no
    passage for another model; unanswerable: refused). Ties go to the higher threshold (more refusals), then to
    the larger MRR@5 sum, then to the earlier option in the order plain/instruct, semantic/hybrid, cosine/rrf."""
    from .semantic import BgeSmallOnnxProvider
    grid = grid or [round(0.50 + 0.01 * i, 2) for i in range(46)]
    table, best = [], None
    for pi, policy in enumerate(policies):
        with tempfile.TemporaryDirectory() as tmp:
            store = build_store_v2(Path(tmp) / "dev.sqlite", BgeSmallOnnxProvider(policy))
            try:
                for gi, gate in enumerate(("semantic", "hybrid")):
                    for ri, ranking in enumerate(("cosine", "rrf")):
                        for tau in grid:
                            rep = run_split(store, "DEVELOPMENT", RetrievalParams(
                                k=5, min_similarity=tau, semantic_gate=gate, semantic_ranking=ranking))
                            ok = sum(_correct(r) for r in rep.queries)
                            mrr = sum(1 / r["first_relevant_rank"] for r in rep.queries if r["first_relevant_rank"])
                            row = {"query_policy": policy, "semantic_gate": gate, "semantic_ranking": ranking,
                                   "min_similarity": tau, "correct": ok, "of": len(rep.queries),
                                   "mrr_sum": round(mrr, 4)}
                            table.append(row)
                            key = (ok, tau, round(mrr, 9), -pi, -gi, -ri)
                            if best is None or key > best[0]:
                                best = (key, row)
            finally:
                store.close()
    return {"selected": best[1], "table": table}


def select_answerability(version: str = "dev-v2") -> dict:
    """Choose the passage threshold, the sentence threshold and the contrast option on a DEVELOPMENT set only
    (gate hybrid and cosine ranking stay as selected on dev-v1). Objective: queries handled correctly (answerable:
    a supported answer quoting a labelled passage and no wrong-model passage; unanswerable: no supported answer).
    Ties go to more correct refusals, then more Recall@5 (useful context), then the higher sentence threshold,
    then contrast on, then the higher passage threshold."""
    from .answerability import AnswerabilityGenerator
    from .semantic import BgeSmallOnnxProvider
    data, _ = load_dataset(version)
    if data["split"] != "DEVELOPMENT":
        raise ValueError(f"{version} is not a DEVELOPMENT set")
    passage_grid = [round(0.50 + 0.02 * i, 2) for i in range(21)]
    sentence_grid = [round(0.60 + 0.02 * i, 2) for i in range(18)] + [1.01]       # 1.01: term coverage only
    table, best, cache = [], None, {}
    with tempfile.TemporaryDirectory() as tmp:
        store = build_store_v2(Path(tmp) / "dev.sqlite", BgeSmallOnnxProvider(), data["corpus"])
        try:
            for tp in passage_grid:
                for ts in sentence_grid:
                    for contrast in (False, True):
                        params = RetrievalParams(k=5, min_similarity=tp, semantic_gate="hybrid",
                                                 semantic_ranking="cosine", min_sentence_similarity=ts,
                                                 answer_contrast=contrast)
                        gen = AnswerabilityGenerator(store, ts, contrast, cache)
                        rep = run_dataset(store, version, params, Retriever(store, params, gen))
                        ok = sum(_answer_correct(r) for r in rep.queries)
                        un = sum(_answer_correct(r) for r in rep.queries if not r["answerable"])
                        rec = sum(r["recall@5"] or 0 for r in rep.queries if r["answerable"])
                        row = {"min_similarity": tp, "min_sentence_similarity": ts, "answer_contrast": contrast,
                               "correct": ok, "unanswerable_correct": un, "recall5_sum": round(rec, 4),
                               "of": len(rep.queries)}
                        table.append(row)
                        key = (ok, un, round(rec, 9), ts, contrast, tp)
                        if best is None or key > best[0]:
                            best = (key, row)
        finally:
            store.close()
    return {"selected": best[1], "table": table}


def _answer_correct(r: dict) -> bool:
    if r["answerable"]:
        return r["supported_correct"] and not r["wrong_model_passages"]
    return r["status"] not in SUPPORTED
