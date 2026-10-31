"""Persistent local knowledge store: one SQLite file holding documents, chunks and their vectors.

Versioning rules (docs/KNOWLEDGE.md §4):
* the same bytes under the same document id: UNCHANGED, nothing is duplicated;
* the same bytes under another document id: DUPLICATE_CONTENT, not indexed twice;
* new bytes under an existing id: a new version; the previous version's chunks become inactive (kept for audit,
  never retrieved);
* removing a document deletes its chunks (text and vectors); a tombstone keeps its id, hash and dates;
* a failed extraction or embedding changes nothing that was already indexed.
Every ingestion attempt is written to an audit table (file name, hash, outcome; never content).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .chunking import ChunkingParams, chunk_blocks
from .embeddings import EmbeddingError, EmbeddingProvider, check_vectors, get_provider, terms
from .extract import extract
from .schemas import (
    QUARANTINE_FLAGS,
    Chunk,
    ChunkFlag,
    DocumentMetadata,
    DocumentRecord,
    ExtractionStatus,
    IngestOutcome,
    IngestReport,
    SourceLocation,
    Trust,
)
from .security import Rejected, check_file, clean_metadata_from_document, instruction_patterns, screen

log = logging.getLogger(__name__)
SCHEMA_VERSION = "1"

_DDL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT NOT NULL, version INTEGER NOT NULL, title TEXT, title_source TEXT NOT NULL, doc_type TEXT NOT NULL,
  manufacturer TEXT, model TEXT, revision TEXT, date TEXT, metadata_source TEXT NOT NULL, supplied_by TEXT,
  filename TEXT NOT NULL, source_path TEXT NOT NULL, format TEXT NOT NULL, size_bytes INTEGER NOT NULL,
  sha256 TEXT NOT NULL, ingested_at TEXT NOT NULL, extraction_status TEXT NOT NULL, extraction_detail TEXT,
  trust TEXT NOT NULL, active INTEGER NOT NULL, deleted INTEGER NOT NULL DEFAULT 0, n_chunks INTEGER NOT NULL,
  n_quarantined INTEGER NOT NULL, embedding_model TEXT, PRIMARY KEY (doc_id, version));
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, version INTEGER NOT NULL, idx INTEGER NOT NULL,
  text TEXT NOT NULL, location TEXT NOT NULL, content_hash TEXT NOT NULL, embedding_model TEXT NOT NULL,
  dim INTEGER NOT NULL, vector BLOB, terms TEXT NOT NULL, flags TEXT NOT NULL, flag_details TEXT NOT NULL,
  quarantined INTEGER NOT NULL, active INTEGER NOT NULL,
  FOREIGN KEY (doc_id, version) REFERENCES documents (doc_id, version));
CREATE INDEX IF NOT EXISTS chunks_active ON chunks (active, quarantined);
CREATE TABLE IF NOT EXISTS ingestions (
  at TEXT NOT NULL, filename TEXT NOT NULL, sha256 TEXT, doc_id TEXT, outcome TEXT NOT NULL,
  extraction_status TEXT NOT NULL, detail TEXT);
"""


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-.")[:64]
    return s if s and s[0].isalnum() else ""


class KnowledgeStore:
    def __init__(self, path: Path, provider: EmbeddingProvider | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(_DDL)
        row = self.db.execute("SELECT value FROM meta WHERE key = 'embedding_model'").fetchone()
        stored = row["value"] if row else None
        if provider is None:
            try:
                provider = get_provider(stored) if stored else get_provider()
            except EmbeddingError:
                self.db.close()
                raise
        if stored and stored != provider.model_id:
            self.db.close()
            raise EmbeddingError(f"this store holds {stored} vectors; {provider.model_id} vectors cannot be compared "
                                 "with them (re-index into a new store)")
        self.provider = provider
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO meta VALUES ('schema_version', ?)", (SCHEMA_VERSION,))
            self.db.execute("INSERT OR IGNORE INTO meta VALUES ('embedding_model', ?)", (provider.model_id,))
            ident = getattr(provider, "identity", None)
            if ident is not None:
                self.db.execute("INSERT OR IGNORE INTO meta VALUES ('embedding_identity', ?)",
                                (json.dumps(ident(), sort_keys=True),))

    def meta(self) -> dict[str, str]:
        return {r["key"]: r["value"] for r in self.db.execute("SELECT key, value FROM meta ORDER BY key")}

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> KnowledgeStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------------------------------------------------------- ingestion
    def _audit(self, at: str, filename: str, sha: str | None, doc_id: str | None, outcome: IngestOutcome,
               status: ExtractionStatus, detail: str | None) -> None:
        with self.db:
            self.db.execute("INSERT INTO ingestions VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (at, filename, sha, doc_id, outcome.value, status.value, detail))
        log.info("knowledge ingest file=%s doc_id=%s outcome=%s extraction=%s", filename, doc_id, outcome.value,
                 status.value)

    def ingest(self, path: Path, metadata: DocumentMetadata | None = None, params: ChunkingParams | None = None,
               now: datetime | None = None) -> IngestReport:
        path = Path(path)
        meta = metadata or DocumentMetadata()
        at = (now or datetime.now(UTC)).isoformat()
        fname = path.name
        doc_id = meta.doc_id or slug(path.stem) or None
        try:
            fmt, data = check_file(path)
        except Rejected as e:
            self._audit(at, fname, None, doc_id, IngestOutcome.NOT_INDEXED, ExtractionStatus.REJECTED, str(e))
            return IngestReport(filename=fname, doc_id=doc_id, outcome=IngestOutcome.NOT_INDEXED,
                                extraction_status=ExtractionStatus.REJECTED, detail=str(e))
        sha = hashlib.sha256(data).hexdigest()
        doc_id = doc_id or f"doc-{sha[:12]}"
        current = self.db.execute("SELECT * FROM documents WHERE doc_id = ? AND active = 1", (doc_id,)).fetchone()
        if current is not None and current["sha256"] == sha:
            self._audit(at, fname, sha, doc_id, IngestOutcome.UNCHANGED, ExtractionStatus.EXTRACTED, None)
            return IngestReport(filename=fname, doc_id=doc_id, outcome=IngestOutcome.UNCHANGED,
                                extraction_status=ExtractionStatus(current["extraction_status"]),
                                version=current["version"], n_chunks=current["n_chunks"],
                                detail="the same content is already indexed under this id",
                                trust=Trust(current["trust"]))
        other = self.db.execute("SELECT doc_id FROM documents WHERE sha256 = ? AND active = 1 AND doc_id != ?",
                                (sha, doc_id)).fetchone()
        if other is not None:
            detail = f"the same content is already indexed as '{other['doc_id']}'; not indexed twice"
            self._audit(at, fname, sha, doc_id, IngestOutcome.DUPLICATE_CONTENT, ExtractionStatus.EXTRACTED, detail)
            return IngestReport(filename=fname, doc_id=doc_id, outcome=IngestOutcome.DUPLICATE_CONTENT,
                                extraction_status=ExtractionStatus.EXTRACTED, duplicate_of=other["doc_id"],
                                detail=detail)
        try:
            ex = extract(fmt, data)
        except Rejected as e:
            ex_status, ex_detail, ex = ExtractionStatus.REJECTED, str(e), None
        else:
            ex_status, ex_detail = ex.status, ex.detail
        keep = (f"; version {current['version']} stays active" if current is not None else "")
        if ex is None or ex.status != ExtractionStatus.EXTRACTED:
            self._audit(at, fname, sha, doc_id, IngestOutcome.NOT_INDEXED, ex_status, (ex_detail or "") + keep)
            return IngestReport(filename=fname, doc_id=doc_id, outcome=IngestOutcome.NOT_INDEXED,
                                extraction_status=ex_status, detail=(ex_detail or ex_status.value) + keep)

        # screen every block (headings included) before chunking: cleaned, redacted text only from here on
        screened = {}
        for b in ex.blocks:
            s = screen(b.text)
            b.text = s.text
            screened[id(b)] = s
        drafts = chunk_blocks([b for b in ex.blocks if b.text.strip()], params)
        rows, flag_count = [], Counter()
        for d in drafts:
            flags, details = [], []
            for b in d.sources:
                s = screened.get(id(b))
                for f, det in zip(s.flags, s.details, strict=False) if s else []:
                    if f not in flags:
                        flags.append(f)
                        details.append(det)
            joined = f"{d.location.section or ''}\n{d.text}"
            if ChunkFlag.INSTRUCTION_PATTERN not in flags and instruction_patterns(joined):
                flags.append(ChunkFlag.INSTRUCTION_PATTERN)
                details.append("text addressed to an AI system: " + ", ".join(instruction_patterns(joined)))
            rows.append((d, flags, details))
            flag_count.update(f.value for f in flags)
        if not rows:
            detail = "no text left after screening" + keep
            self._audit(at, fname, sha, doc_id, IngestOutcome.NOT_INDEXED, ExtractionStatus.EMPTY_TEXT, detail)
            return IngestReport(filename=fname, doc_id=doc_id, outcome=IngestOutcome.NOT_INDEXED,
                                extraction_status=ExtractionStatus.EMPTY_TEXT, detail=detail)

        live = [i for i, (_, f, _) in enumerate(rows) if not set(f) & QUARANTINE_FLAGS]
        vectors = check_vectors(self.provider, [f"{rows[i][0].location.section or ''}\n{rows[i][0].text}"
                                                for i in live]) if live else np.zeros((0, self.provider.dim))
        vec_of = dict(zip(live, vectors, strict=True))

        title, title_source = meta.title, "SUPPLIED" if meta.title else "NONE"
        if title is None:
            title, dropped = clean_metadata_from_document(ex.title)
            title_source = "DOCUMENT" if title else "NONE"
            if dropped:
                flag_count["UNSAFE_TITLE_DROPPED"] += 1
        n_q = len(rows) - len(live)
        trust = Trust.FLAGGED if n_q or flag_count.get(ChunkFlag.SECRET_REDACTED.value) or \
            flag_count.get("UNSAFE_TITLE_DROPPED") else Trust.UNREVIEWED
        last = self.db.execute("SELECT MAX(version) AS v FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()["v"]
        version = (last or 0) + 1
        supplied = any(getattr(meta, k) for k in ("manufacturer", "model", "revision", "date"))
        with self.db:
            if current is not None:
                self.db.execute("UPDATE documents SET active = 0 WHERE doc_id = ?", (doc_id,))
                self.db.execute("UPDATE chunks SET active = 0 WHERE doc_id = ?", (doc_id,))
            self.db.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1,0,?,?,?)",
                (doc_id, version, title, title_source, meta.doc_type.value, meta.manufacturer, meta.model,
                 meta.revision, meta.date, "SUPPLIED" if supplied else "NONE", meta.supplied_by, fname,
                 str(path.resolve()), fmt, len(data), sha, at, ExtractionStatus.EXTRACTED.value, ex.detail,
                 trust.value, len(rows), n_q, self.provider.model_id))
            for i, (d, flags, details) in enumerate(rows):
                chunk = Chunk(chunk_id=f"{doc_id}@v{version}#c{i:04d}", doc_id=doc_id, version=version, index=i,
                              text=d.text, location=d.location,
                              content_hash=hashlib.sha256(d.text.encode()).hexdigest(),
                              embedding_model=self.provider.model_id, flags=flags, flag_details=details,
                              quarantined=bool(set(flags) & QUARANTINE_FLAGS))
                v = vec_of.get(i)
                self.db.execute(
                    "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)",
                    (chunk.chunk_id, doc_id, version, i, chunk.text, chunk.location.model_dump_json(),
                     chunk.content_hash, chunk.embedding_model, self.provider.dim,
                     None if v is None else np.asarray(v, dtype=np.float32).tobytes(),
                     json.dumps(dict(sorted(Counter(terms(f"{d.location.section or ''}\n{d.text}")).items()))),
                     json.dumps([f.value for f in flags]), json.dumps(details), int(chunk.quarantined)))
        report = IngestReport(filename=fname, doc_id=doc_id, outcome=IngestOutcome.INDEXED,
                              extraction_status=ExtractionStatus.EXTRACTED, detail=ex.detail, version=version,
                              superseded_version=current["version"] if current is not None else None,
                              n_chunks=len(rows), n_quarantined=n_q, flags=dict(sorted(flag_count.items())),
                              trust=trust)
        self._audit(at, fname, sha, doc_id, IngestOutcome.INDEXED, ExtractionStatus.EXTRACTED, ex.detail)
        return report

    # ---------------------------------------------------------------- management
    def remove(self, doc_id: str) -> int:
        """Delete a document's chunks (all versions). A tombstone keeps id, hash and dates. Returns chunks deleted."""
        with self.db:
            n = self.db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,)).rowcount
            self.db.execute("UPDATE documents SET active = 0, deleted = 1, n_chunks = 0 WHERE doc_id = ?", (doc_id,))
        log.info("knowledge remove doc_id=%s chunks=%d", doc_id, n)
        return n

    def documents(self, include_inactive: bool = False) -> list[DocumentRecord]:
        q = "SELECT * FROM documents" + ("" if include_inactive else " WHERE active = 1") + " ORDER BY doc_id, version"
        return [_doc(r) for r in self.db.execute(q)]

    def chunks(self, doc_id: str, include_inactive: bool = False) -> list[Chunk]:
        q = "SELECT * FROM chunks WHERE doc_id = ?" + ("" if include_inactive else " AND active = 1") + \
            " ORDER BY version, idx"
        return [_chunk(r) for r in self.db.execute(q, (doc_id,))]

    def ingestions(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM ingestions ORDER BY rowid")]

    def reindex(self, target: Path, provider: EmbeddingProvider, replace: bool = False,
                now: datetime | None = None) -> dict:
        """Copy this store into a new store whose vectors come from ``provider``. Document ids, versions, chunk ids,
        text, locations, provenance, trust, flags, quarantine state, tombstones and the audit trail are copied
        unchanged; only the vectors of active, non-quarantined chunks are recomputed. Inactive versions keep their
        text for audit, without a vector. The new store is written to a temporary file and moved into place only
        when complete, so a failure leaves this store and any existing target untouched."""
        target = Path(target)
        if target.resolve() == self.path.resolve():
            raise EmbeddingError("re-index into a new store: a store never holds vectors from two models")
        if target.exists() and not replace:
            raise EmbeddingError(f"{target} exists; pass replace=True to replace it once the re-index succeeds")
        rows = self.db.execute("SELECT c.* FROM chunks c JOIN documents d ON c.doc_id = d.doc_id AND "
                               "c.version = d.version ORDER BY c.chunk_id").fetchall()
        live = [r for r in rows if r["active"] and not r["quarantined"]]
        texts = [f"{json.loads(r['location']).get('section') or ''}\n{r['text']}" for r in live]
        vectors = check_vectors(provider, texts) if live else np.zeros((0, provider.dim), dtype=np.float32)
        vec_of = {r["chunk_id"]: v for r, v in zip(live, vectors, strict=True)}
        tmp = target.with_name(target.name + ".reindex-tmp")
        tmp.unlink(missing_ok=True)
        new = None
        try:
            new = KnowledgeStore(tmp, provider)
            with new.db:
                for d in self.db.execute("SELECT * FROM documents ORDER BY doc_id, version"):
                    d = dict(d)
                    if d["active"]:
                        d["embedding_model"] = provider.model_id
                    new.db.execute(f"INSERT INTO documents ({', '.join(d)}) VALUES ({', '.join('?' * len(d))})",
                                   tuple(d.values()))
                for r in rows:
                    r = dict(r)
                    v = vec_of.get(r["chunk_id"])
                    if r["active"] and not r["quarantined"]:
                        r.update(embedding_model=provider.model_id, dim=provider.dim,
                                 vector=np.asarray(v, dtype=np.float32).tobytes())
                    else:
                        r["vector"] = None
                    new.db.execute(f"INSERT INTO chunks ({', '.join(r)}) VALUES ({', '.join('?' * len(r))})",
                                   tuple(r.values()))
                for a in self.db.execute("SELECT * FROM ingestions ORDER BY rowid"):
                    new.db.execute("INSERT INTO ingestions VALUES (?, ?, ?, ?, ?, ?, ?)", tuple(a))
                new.db.execute("INSERT INTO meta VALUES ('reindexed_from', ?)", (json.dumps(
                    {"source_embedding_model": self.provider.model_id, "source_path": str(self.path.resolve()),
                     "at": (now or datetime.now(UTC)).isoformat()}, sort_keys=True),))
            new.close()
            tmp.replace(target)
        except BaseException:
            if new is not None:
                new.close()
            tmp.unlink(missing_ok=True)
            raise
        report = {"target": str(target), "embedding_model": provider.model_id, "documents": len(
            self.documents()), "chunks_embedded": len(live), "chunks_copied": len(rows)}
        log.info("knowledge reindex target=%s model=%s chunks=%d", target.name, provider.model_id, len(live))
        return report

    def retrievable(self):
        """Active, non-quarantined chunks of active documents, in chunk-id order, with their vectors and terms."""
        rows = self.db.execute(
            "SELECT c.*, d.title, d.title_source, d.doc_type, d.revision, d.filename, d.trust, d.manufacturer, "
            "d.model FROM chunks c JOIN documents d ON c.doc_id = d.doc_id AND c.version = d.version "
            "WHERE c.active = 1 AND c.quarantined = 0 AND d.active = 1 AND c.vector IS NOT NULL "
            "ORDER BY c.chunk_id").fetchall()
        for r in rows:
            if r["embedding_model"] != self.provider.model_id:
                raise EmbeddingError(f"chunk {r['chunk_id']} holds {r['embedding_model']} vectors; the store "
                                     f"uses {self.provider.model_id}")
        vecs = (np.frombuffer(b"".join(r["vector"] for r in rows), dtype=np.float32).reshape(len(rows),
                                                                                             self.provider.dim)
                if rows else np.zeros((0, self.provider.dim), dtype=np.float32))
        return rows, vecs


def _doc(r) -> DocumentRecord:
    return DocumentRecord(
        doc_id=r["doc_id"], version=r["version"], title=r["title"], title_source=r["title_source"],
        doc_type=r["doc_type"], manufacturer=r["manufacturer"], model=r["model"], revision=r["revision"],
        date=r["date"], metadata_source=r["metadata_source"], supplied_by=r["supplied_by"], filename=r["filename"],
        source_path=r["source_path"], format=r["format"], size_bytes=r["size_bytes"], sha256=r["sha256"],
        ingested_at=datetime.fromisoformat(r["ingested_at"]), extraction_status=r["extraction_status"],
        extraction_detail=r["extraction_detail"], trust=r["trust"], active=bool(r["active"]),
        n_chunks=r["n_chunks"], n_quarantined=r["n_quarantined"], embedding_model=r["embedding_model"])


def _chunk(r) -> Chunk:
    return Chunk(chunk_id=r["chunk_id"], doc_id=r["doc_id"], version=r["version"], index=r["idx"], text=r["text"],
                 location=SourceLocation.model_validate_json(r["location"]), content_hash=r["content_hash"],
                 embedding_model=r["embedding_model"], flags=json.loads(r["flags"]),
                 flag_details=json.loads(r["flag_details"]), quarantined=bool(r["quarantined"]))
