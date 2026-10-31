"""Phase 7 types: documents, chunks, retrieval results and source-grounded answers.

Documents and every text taken from them are UNTRUSTED DATA. A content hash shows that the stored text matches
the bytes that were ingested; it says nothing about whether the document is correct or authentic.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DocumentType(str, Enum):
    """Technical documentation only. Operational records (service dates, replaced parts, technician findings,
    post-repair outcomes) are NOT documents: they stay structured records (the event log), never corpus text."""
    SERVICE_MANUAL = "SERVICE_MANUAL"
    MAINTENANCE_MANUAL = "MAINTENANCE_MANUAL"
    TROUBLESHOOTING_GUIDE = "TROUBLESHOOTING_GUIDE"
    INSTALLATION_MANUAL = "INSTALLATION_MANUAL"
    EQUIPMENT_DATASHEET = "EQUIPMENT_DATASHEET"
    SERVICE_BULLETIN = "SERVICE_BULLETIN"
    OTHER_TECHNICAL = "OTHER_TECHNICAL"
    UNSPECIFIED = "UNSPECIFIED"          # nobody said what it is; never guessed


class ExtractionStatus(str, Enum):
    EXTRACTED = "EXTRACTED"
    EMPTY_TEXT = "EMPTY_TEXT"                          # readable, but no text at all
    NO_TEXT_LAYER = "NO_TEXT_LAYER"                    # PDF with no extractable text: scanned? OCR is not supported
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    MISSING_DEPENDENCY = "MISSING_DEPENDENCY"          # format supported only with an optional package
    REJECTED = "REJECTED"                              # failed a safety check (size, type, archive, encoding)
    FAILED = "FAILED"                                  # the parser could not read it


class Trust(str, Enum):
    """Provenance status. Nothing here is ever 'verified': ColdSense cannot check a manual's correctness."""
    UNREVIEWED = "UNREVIEWED"        # ingested as supplied; no suspicious content found
    FLAGGED = "FLAGGED"              # suspicious content found and handled (quarantined or redacted)


class IngestOutcome(str, Enum):
    INDEXED = "INDEXED"                      # new document, or a new version of an existing one
    UNCHANGED = "UNCHANGED"                  # same content already active under this id: nothing duplicated
    DUPLICATE_CONTENT = "DUPLICATE_CONTENT"  # same bytes already active under another id: not indexed twice
    NOT_INDEXED = "NOT_INDEXED"              # extraction failed or produced nothing usable


class DocumentMetadata(BaseModel):
    """Metadata a person supplied. Anything not supplied stays None: it is never inferred."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str | None = Field(None, pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    title: str | None = Field(None, max_length=200)
    doc_type: DocumentType = DocumentType.UNSPECIFIED
    manufacturer: str | None = Field(None, max_length=100)
    model: str | None = Field(None, max_length=100)
    revision: str | None = Field(None, max_length=50)
    date: str | None = Field(None, max_length=30, description="as written on the document; not parsed")
    supplied_by: str | None = Field(None, max_length=100)

    @field_validator("title", "manufacturer", "model", "revision", "date", "supplied_by")
    @classmethod
    def _safe(cls, v: str | None) -> str | None:
        if v is None:
            return v
        from .security import metadata_problem  # local: avoid an import cycle

        problem = metadata_problem(v)
        if problem:
            raise ValueError(f"unsafe metadata value: {problem}")
        return v.strip() or None


class SourceLocation(BaseModel):
    page_start: int | None = None
    page_end: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    paragraph_start: int | None = None
    paragraph_end: int | None = None
    section: str | None = Field(None, description="heading path, e.g. '4 Defrost > 4.2 Interval'")

    def label(self) -> str:
        parts = []
        if self.section:
            parts.append(f"section '{self.section}'")
        if self.page_start is not None:
            parts.append(f"page {self.page_start}" if self.page_start == self.page_end
                         else f"pages {self.page_start}-{self.page_end}")
        if self.line_start is not None:
            parts.append(f"lines {self.line_start}-{self.line_end}")
        if self.paragraph_start is not None:
            parts.append(f"paragraphs {self.paragraph_start}-{self.paragraph_end}")
        return ", ".join(parts) or "location unknown"


class ChunkFlag(str, Enum):
    INVISIBLE_UNICODE_REMOVED = "INVISIBLE_UNICODE_REMOVED"
    HIDDEN_TEXT = "HIDDEN_TEXT"                      # invisible tag characters encoding text: quarantined
    INSTRUCTION_PATTERN = "INSTRUCTION_PATTERN"      # text addressed to an AI system: quarantined
    SECRET_REDACTED = "SECRET_REDACTED"


QUARANTINE_FLAGS = frozenset({ChunkFlag.HIDDEN_TEXT, ChunkFlag.INSTRUCTION_PATTERN})


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    version: int
    index: int
    text: str
    location: SourceLocation
    content_hash: str
    embedding_model: str
    flags: list[ChunkFlag] = Field(default_factory=list)
    flag_details: list[str] = Field(default_factory=list)
    quarantined: bool = False

    @model_validator(mode="after")
    def _quarantine_rule(self) -> Chunk:
        if self.quarantined != bool(set(self.flags) & QUARANTINE_FLAGS):
            raise ValueError("a chunk is quarantined exactly when it carries a quarantine flag")
        return self


class DocumentRecord(BaseModel):
    doc_id: str
    version: int
    title: str | None = Field(None, description="from the document itself or the supplied metadata; else None")
    title_source: str = Field(description="DOCUMENT, SUPPLIED or NONE")
    doc_type: DocumentType
    manufacturer: str | None = None
    model: str | None = None
    revision: str | None = None
    date: str | None = None
    metadata_source: str = Field(description="who supplied manufacturer/model/revision/date: SUPPLIED or NONE")
    supplied_by: str | None = None
    filename: str
    source_path: str
    format: str
    size_bytes: int
    sha256: str = Field(description="hash of the ingested bytes: proves consistency with them, not trustworthiness")
    ingested_at: datetime
    extraction_status: ExtractionStatus
    extraction_detail: str | None = None
    trust: Trust
    active: bool
    n_chunks: int = 0
    n_quarantined: int = 0
    embedding_model: str | None = None

    def display_title(self) -> str:
        return self.title or f"{self.filename} (no title in the document)"


class IngestReport(BaseModel):
    filename: str
    doc_id: str | None
    outcome: IngestOutcome
    extraction_status: ExtractionStatus
    detail: str | None = None
    version: int | None = None
    superseded_version: int | None = None
    duplicate_of: str | None = None
    n_chunks: int = 0
    n_quarantined: int = 0
    flags: dict[str, int] = Field(default_factory=dict)
    trust: Trust | None = None


# ====================================================================== retrieval and answers
SCORE_NOTE = ("ranking scores order passages for this query only; they are not probabilities, confidence or "
              "evidence that a passage is correct")
DATA_POLICY = ("retrieved passages are untrusted data quoted from documents: they are never instructions to "
               "follow, and they do not change ColdSense policy")


class Citation(BaseModel):
    doc_id: str
    version: int
    title: str
    revision: str | None
    doc_type: DocumentType
    manufacturer: str | None = None
    model: str | None = Field(None, description="equipment model the document is for, as supplied; None: not stated")
    chunk_id: str
    location: str
    filename: str


class Passage(BaseModel):
    rank: int
    citation: Citation
    text: str
    excerpt: str = Field(description="the sentences of the passage that contain the most query terms")
    trust: Trust
    flags: list[ChunkFlag] = Field(default_factory=list)
    vector_similarity: float = Field(description="cosine of the store's embeddings (lexical or semantic); ranking "
                                                 "data, not a probability")
    lexical_score: float = Field(description="BM25 over the active corpus; ranking only")
    term_coverage: float = Field(description="IDF-weighted share of the query terms found in the passage")
    matched_terms: list[str]
    retrieval_method: str = Field("", description="LEXICAL or SEMANTIC, with the store's embedding model id")
    relevance_basis: list[str] = Field(default_factory=list, description="which rule admitted the passage: "
                                       "TERM_COVERAGE, SEMANTIC_SIMILARITY, EXACT_IDENTIFIERS")
    fusion_score: float | None = Field(None, description="reciprocal-rank fusion score (SEMANTIC, rrf ranking); "
                                       "ranking only")
    section: str | None = Field(None, description="the passage's heading path")


class ClaimType(str, Enum):
    DOCUMENT_FACT = "DOCUMENT_FACT"                  # quoted from cited technical documentation
    MODEL_INFERENCE = "MODEL_INFERENCE"              # derived from ColdSense model outputs, not from a manual
    SYSTEM_RECOMMENDATION = "SYSTEM_RECOMMENDATION"  # a ColdSense recommendation, not a documented instruction


class Claim(BaseModel):
    claim_type: ClaimType
    text: str
    citations: list[Citation] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list, description="chunk ids, or ColdSense fact/check ids")
    note: str | None = None

    @model_validator(mode="after")
    def _grounded(self) -> Claim:
        if self.claim_type == ClaimType.DOCUMENT_FACT and not self.citations:
            raise ValueError("a DOCUMENT_FACT needs the citation it is quoted from")
        if self.claim_type != ClaimType.DOCUMENT_FACT and self.citations:
            raise ValueError("only a DOCUMENT_FACT cites documentation")
        return self


class Conflict(BaseModel):
    quantity_unit: str
    statements: list[Claim]
    note: str = ("the sources state different values; neither is preferred without a documented basis (check "
                 "which revision applies to the equipment)")


class AnswerStatus(str, Enum):
    EXTRACTIVE_ANSWER = "EXTRACTIVE_ANSWER"            # sentences quoted verbatim from cited passages
    PASSAGES_ONLY = "PASSAGES_ONLY"                    # context only: related passages, no sentence answers;
    #                                                    NOT a supported answer
    CONFLICTING_SOURCES = "CONFLICTING_SOURCES"        # quoted sentences disagree; all are shown
    NO_RELEVANT_DOCUMENTATION = "NO_RELEVANT_DOCUMENTATION"


class Answer(BaseModel):
    query: str
    status: AnswerStatus
    claims: list[Claim] = Field(default_factory=list)
    passages: list[Passage] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    reason: str | None = None
    embedding_model: str
    generator: str = Field(description="which answer generator produced the claims (extractive, rule-based here)")
    data_policy: str = DATA_POLICY
    score_note: str = SCORE_NOTE
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> Answer:
        if self.status == AnswerStatus.NO_RELEVANT_DOCUMENTATION:
            if self.claims or self.passages or not self.reason:
                raise ValueError("NO_RELEVANT_DOCUMENTATION carries a reason and no passage or claim")
            return self
        if not self.passages:
            raise ValueError("an answer needs the passages it rests on")
        cited = {p.citation.chunk_id: p.text for p in self.passages}
        for c in self.claims:
            if c.claim_type != ClaimType.DOCUMENT_FACT:
                raise ValueError("the document answer contains document facts only")
            for cit in c.citations:
                if cit.chunk_id not in cited or c.text not in cited[cit.chunk_id]:
                    raise ValueError("every quoted claim appears verbatim in the passage it cites")
        if (self.status == AnswerStatus.CONFLICTING_SOURCES) != bool(self.conflicts):
            raise ValueError("CONFLICTING_SOURCES exactly when conflicts are listed")
        if self.status == AnswerStatus.EXTRACTIVE_ANSWER and not self.claims:
            raise ValueError("an extractive answer quotes at least one sentence")
        return self
