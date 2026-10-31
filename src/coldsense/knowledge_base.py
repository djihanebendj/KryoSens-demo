"""Phase 7: Production Technical Knowledge Base (SQLite Hybrid Vector + BM25 RRF,
Trust-Boundary Quarantine, Equipment/Identifier Filtering & Answerability Verification).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from coldsense.models import CopilotCitation


@dataclass(frozen=True)
class KnowledgeDocument:
    """Single technical manual section/chunk in the ColdSense knowledge base with full provenance."""

    doc_id: str
    title: str
    section: str
    content: str
    topic_tags: Tuple[str, ...]
    supported_intents: Tuple[str, ...]
    source_doc_id: str = ""
    equipment_model: Optional[str] = None
    manufacturer: str = "ColdSense Standard"
    doc_type: str = "TECHNICAL_MANUAL"
    revision: str = "1"
    is_active: bool = True
    quarantined: bool = False
    trust_status: str = "UNREVIEWED"

    def __post_init__(self) -> None:
        if not self.source_doc_id:
            object.__setattr__(self, "source_doc_id", self.doc_id.split("#", 1)[0])


DEFAULT_KNOWLEDGE_CORPUS: Tuple[KnowledgeDocument, ...] = (
    KnowledgeDocument(
        doc_id="KB-CR-101",
        title="Cold-Room Thermal Envelope & Door Gasket Maintenance Manual",
        section="Section 4.2: Door Perimeter Seal & Paper-Drag Inspection Procedure",
        content=(
            "A well-insulated walk-in cold room operating at a setpoint of 2.0 C should exhibit "
            "a passive thermal time constant tau between 3.5 hours and 6.0 hours during an unpowered drift test. "
            "When Perform Perimeter Door Gasket inspection is required, technicians must use a 0.1 mm paper strip "
            "every 30 cm around the door perimeter. Any gap exceeding 1.5 mm or localized frost accumulation "
            "requires immediate gasket replacement."
        ),
        topic_tags=("gasket", "door", "seal", "tau", "time", "constant", "insulation", "drift", "thermal", "paper"),
        supported_intents=("inspect", "gasket", "tau", "time constant", "seal", "tolerance", "gap", "hours", "paper"),
    ),
    KnowledgeDocument(
        doc_id="KB-CR-102",
        title="Commercial Refrigeration Excursion & Food/Pharma Cold-Chain Thresholds",
        section="Section 2.1: Critical Temperature Thresholds and Holdover Limits",
        content=(
            "Chilled storage cold rooms must maintain internal air temperature between 0.0 C and 4.0 C. "
            "The critical excursion threshold is 8.0 C. If internal air temperature exceeds 8.0 C for more "
            "than 120 minutes during a power cut, quality assurance quarantine protocols must be initiated "
            "and core product temperature probes must be verified within 15 minutes."
        ),
        topic_tags=("threshold", "excursion", "power", "cut", "outage", "critical", "temperature", "quarantine", "minutes", "holdover"),
        supported_intents=("threshold", "critical", "excursion", "power cut", "how long", "minutes", "limit", "8.0", "holdover"),
    ),
    KnowledgeDocument(
        doc_id="KB-CR-103",
        title="Evaporator Defrost & Airflow Diagnostics Guide",
        section="Section 6.4: Electric & Hot-Gas Defrost Termination Parameters",
        content=(
            "Standard electric defrost cycles run every 6 hours for a maximum duration of 30 minutes and "
            "terminate when the evaporator coil sensor reaches 12.0 C. Post-defrost fan delay should be set "
            "to 3 minutes to prevent warm moisture blow-off into the cold room. If recovery to 4.0 C takes "
            "longer than 45 minutes after defrost termination, inspect evaporator coil icing and fan airflow."
        ),
        topic_tags=("defrost", "evaporator", "coil", "fan", "recovery", "termination", "icing", "drain"),
        supported_intents=("defrost", "termination", "duration", "recovery", "evaporator", "fan", "delay", "coil"),
    ),
    KnowledgeDocument(
        doc_id="KB-CR-104",
        title="Telemetry Data Logger & Temperature Probe Calibration Standard",
        section="Section 3.1: Probe Placement, Cadence, and Ice-Bath Calibration",
        content=(
            "Cold-room temperature loggers must record at a sampling cadence of 300 seconds (5 minutes) "
            "or faster. Probes must be mounted at least 50 cm away from door openings and outside the direct "
            "evaporator discharge airstream. Annual ice-bath calibration at 0.0 C requires accuracy within "
            "plus or minus 0.5 C; probes drifting beyond 0.5 C must be recalibrated or replaced."
        ),
        topic_tags=("probe", "sensor", "logger", "calibration", "cadence", "ice", "bath", "accuracy", "placement"),
        supported_intents=("probe", "sensor", "calibration", "cadence", "accuracy", "ice bath", "placement", "sampling"),
    ),
    KnowledgeDocument(
        doc_id="KB-CR-105",
        title="Compressor & Pressure Switch Fault Codes Reference",
        section="Section 8.1: High/Low Pressostat Trip Indicators (Read-Only Diagnostics)",
        content=(
            "Controller alarm code E04 indicates that the low-pressure pressostat has tripped at 0.8 bar "
            "due to suction pressure drop. Alarm code E05 indicates that the high-pressure pressostat has "
            "tripped at 24.0 bar. Technicians may read the controller alarm log and check condenser airflow, "
            "but must never alter protection setpoints without certified commissioning instruments."
        ),
        topic_tags=("alarm", "code", "e04", "e05", "pressure", "trip", "tripped", "bar", "controller", "superheat", "txv"),
        supported_intents=("alarm", "code", "e04", "e05", "tripped", "meaning", "what does", "bar", "pressure", "superheat"),
    ),
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_NUMBER_RE = re.compile(r"(?<![a-zA-Z])[-+]?\d+(?:\.\d+)?(?![a-zA-Z])")

# Identifiers are names, not quantities. "EX-200" is a model, "SB-12" a bulletin, "R-290" a
# refrigerant, "E3" an alarm code, "section 4.2" a reference: none of them is a measured value, and the
# numeric audit must not treat them as claims. _NUMBER_RE already rejects a digit glued to a letter
# (E3), but a hyphen resets the lookbehind, so "EX-200" yielded 200 and a quoted model name made the
# audit fail closed on its own correct answer. These patterns are masked out before extraction; a bare
# quantity ("-5.5 C", "30 min", "4.2 bar") is untouched and still audited.
_IDENTIFIER_PATTERNS = (
    re.compile(r"\b[A-Za-z]{1,6}-\d+(?:\.\d+)?[A-Za-z]?\b"),        # EX-200, SB-12, FB-3, R-290, TF-90
    re.compile(r"\b[A-Za-z]{1,6}\d+(?:\.\d+)?\b"),                  # E3, P1, v2 (already safe, kept explicit)
    re.compile(r"\b(?:section|sec\.|clause|page|p\.|rev(?:ision)?|v)\s*\d+(?:\.\d+)*\b", re.IGNORECASE),
)


def mask_identifiers(text: str) -> str:
    """Replace identifier tokens with a placeholder so they are never read as quantities."""
    masked = text
    for pattern in _IDENTIFIER_PATTERNS:
        masked = pattern.sub("<ID>", masked)
    return masked


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 or t.isdigit()]


def extract_numbers(text: str) -> List[float]:
    """Extract all numeric values from text for the Numeric Integrity Audit."""
    vals: List[float] = []
    for m in _NUMBER_RE.findall(mask_identifiers(text)):
        try:
            vals.append(float(m))
        except ValueError:
            continue
    return vals


class LocalKnowledgeBase:
    """Production RAG Knowledge Base backed by `SQLiteHybridVectorStore` and `ColdSenseHybridRetriever`."""

    def __init__(
        self,
        documents: Optional[Sequence[KnowledgeDocument]] = None,
        semantic_available: bool = True,
        sqlite_path: str = ":memory:",
        allow_network_embeddings: bool = False,
        manuals_dir: Optional[Path] = None,
    ) -> None:
        from coldsense.rag_pipeline import (
            ColdSenseHybridRetriever,
            GeminiEmbeddingClient,
            SQLiteHybridVectorStore,
            load_and_chunk_manuals,
        )

        if documents is not None:
            self.documents = tuple(documents)
        else:
            disk_chunks = load_and_chunk_manuals(manuals_dir=manuals_dir)
            seen_ids = {d.doc_id for d in DEFAULT_KNOWLEDGE_CORPUS}
            merged = list(DEFAULT_KNOWLEDGE_CORPUS)
            for c in disk_chunks:
                if c.doc_id not in seen_ids:
                    merged.append(c)
                    seen_ids.add(c.doc_id)
            self.documents = tuple(merged)

        self.semantic_available = semantic_available
        self.embedder = GeminiEmbeddingClient(allow_network=allow_network_embeddings)
        self.vector_store = SQLiteHybridVectorStore(
            sqlite_path=sqlite_path,
            embedding_client=self.embedder,
            documents=self.documents,
        )
        self.retriever = ColdSenseHybridRetriever(
            vector_store=self.vector_store,
            top_k=3,
            semantic_enabled=semantic_available,
        )

    def retrieve(
        self, query: str, top_k: int = 3
    ) -> Tuple[List[Tuple[KnowledgeDocument, CopilotCitation]], str]:
        """Retrieve top_k verified documents via the LangChain Hybrid Retriever."""
        self.retriever.top_k = top_k
        self.retriever.semantic_enabled = self.semantic_available
        lc_docs = self.retriever.invoke(query)
        mode = "SEMANTIC_HYBRID" if self.semantic_available else "LEXICAL_FALLBACK"
        results: List[Tuple[KnowledgeDocument, CopilotCitation]] = []
        for lcd in lc_docs:
            kdoc: KnowledgeDocument = lcd.metadata["knowledge_doc"]
            cit: CopilotCitation = lcd.metadata["citation"]
            results.append((kdoc, cit))
        return results, mode

    def answer_query(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Execute full Phase 7 retrieval + answerability + conflict detection and return structured status:
        - `EXTRACTIVE_ANSWER`: 1+ verified quotes answering the query
        - `CONFLICTING_SOURCES`: 2+ active documents for the same model state conflicting values in the same unit
        - `NO_RELEVANT_DOCUMENTATION`: nothing passed the relevance/identifier/answerability gates
        """
        from coldsense.rag_pipeline import detect_conflicting_sources

        hits, mode = self.retrieve(query, top_k=top_k)
        if not hits:
            return {
                "status": "NO_RELEVANT_DOCUMENTATION",
                "retrieval_mode": mode,
                "citations": [],
                "conflicts": [],
            }

        conflicts = detect_conflicting_sources(query, hits)
        if conflicts:
            return {
                "status": "CONFLICTING_SOURCES",
                "retrieval_mode": mode,
                "citations": [cit.to_dict() for _, cit in hits],
                "conflicts": conflicts,
            }

        return {
            "status": "EXTRACTIVE_ANSWER",
            "retrieval_mode": mode,
            "citations": [cit.to_dict() for _, cit in hits],
            "conflicts": [],
        }


def verify_quote_and_semantic_support(
    query: str,
    doc: KnowledgeDocument,
    verbatim_quote: str,
) -> Tuple[bool, str]:
    """Enforce Rule 4: Verbatim quote substring check + production answerability verification."""
    from coldsense.rag_pipeline import verify_production_answerability

    if not verbatim_quote or verbatim_quote not in doc.content:
        return False, "Verbatim quote check failed: quote is not an exact substring of the source document."

    ok, reason = verify_production_answerability(query, doc, verbatim_quote)
    if not ok:
        # Check if full section content satisfies unit/action context when quote is a single sentence
        ok_full, _ = verify_production_answerability(query, doc, doc.content)
        if not ok_full:
            return False, f"Retrieved passage is an on-topic non-answer: {reason}"

    return True, "Verified verbatim quote and semantic support."
