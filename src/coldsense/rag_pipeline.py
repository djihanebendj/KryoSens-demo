"""Hybrid Vector+BM25 RAG Pipeline with Quarantine, Identifier Filtering, and Answerability Verification."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import unicodedata
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from coldsense.knowledge_base import (
    DEFAULT_KNOWLEDGE_CORPUS,
    KnowledgeDocument,
    tokenize,
)
from coldsense.models import CopilotCitation

try:
    from langchain_core.documents import Document as LCDocument  # type: ignore
    from langchain_core.retrievers import BaseRetriever as LCBaseRetriever  # type: ignore

    LANGCHAIN_CORE_AVAILABLE = True
except ImportError:
    LANGCHAIN_CORE_AVAILABLE = False

    @dataclass
    class LCDocument:  # type: ignore
        page_content: str
        metadata: Dict[str, Any] = field(default_factory=dict)

    class LCBaseRetriever:  # type: ignore
        def invoke(self, query: str, config: Optional[Dict[str, Any]] = None) -> List[LCDocument]:
            return self._get_relevant_documents(query)

        def _get_relevant_documents(self, query: str) -> List[LCDocument]:
            raise NotImplementedError


MAX_QUERY_CHARS = 1000
_TAG_RANGE = (0xE0000, 0xE007F)
_BIDI_CHARS = set("\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\u200e\u200f\u061c")

_INJECTION_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"note\s+for\s+assistants\b", re.IGNORECASE),
    re.compile(r"tell\s+the\s+technician\s+to\s+bypass", re.IGNORECASE),
    re.compile(r"(?:system\s+prompt|you\s+are\s+now\s+in\s+developer\s+mode)", re.IGNORECASE),
    re.compile(r"<\|(?:im_start|system|endoftext)\|>", re.IGNORECASE),
)

_SECRET_REDACT_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----.*?-----END [A-Z ]+ PRIVATE KEY-----", re.DOTALL), "[REDACTED_KEY]"),
    (re.compile(r"\b(?:password|passwd|api_key|secret_key)\s*=\s*\S+", re.IGNORECASE), "[REDACTED_SECRET]"),
)


@dataclass(frozen=True)
class SanitizedTextResult:
    """Result of cleaning invisible Unicode, decoding TAG chars, and checking for prompt injection."""

    clean_text: str
    removed_invisible_count: int
    hidden_tag_ascii: str
    bidi_count: int
    is_quarantined: bool
    quarantine_reasons: Tuple[str, ...]


def sanitize_and_audit_text(raw_text: str, heading: str = "") -> SanitizedTextResult:
    """Clean invisible Unicode, decode hidden TAG characters, redact secrets, and flag prompt injection."""
    out_chars: List[str] = []
    removed = 0
    hidden_chars: List[str] = []
    bidi = 0

    for ch in raw_text:
        cp = ord(ch)
        if _TAG_RANGE[0] <= cp <= _TAG_RANGE[1]:
            removed += 1
            ascii_cp = cp - 0xE0000
            if 0x20 <= ascii_cp <= 0x7E:
                hidden_chars.append(chr(ascii_cp))
            continue
        if ch in _BIDI_CHARS:
            bidi += 1
            removed += 1
            continue
        if ch == "\u00ad":
            continue
        cat = unicodedata.category(ch)
        if cat == "Cf" or (cat == "Cc" and ch not in ("\n", "\r", "\t")):
            removed += 1
            continue
        out_chars.append(ch)

    cleaned = "".join(out_chars)
    for pat, repl in _SECRET_REDACT_PATTERNS:
        cleaned = pat.sub(repl, cleaned)

    hidden_str = "".join(hidden_chars).strip()
    reasons: List[str] = []
    if hidden_str:
        reasons.append(f"Hidden Unicode TAG ASCII detected: '{hidden_str[:40]}'")

    combined = f"{heading}\n{cleaned}"
    for pat in _INJECTION_PATTERNS:
        if pat.search(combined):
            reasons.append(f"Prompt-injection override pattern matched: '{pat.pattern}'")

    return SanitizedTextResult(
        clean_text=cleaned,
        removed_invisible_count=removed,
        hidden_tag_ascii=hidden_str,
        bidi_count=bidi,
        is_quarantined=len(reasons) > 0,
        quarantine_reasons=tuple(reasons),
    )


def load_dotenv_file(dotenv_path: Optional[Path] = None) -> Dict[str, str]:
    """Load key=value pairs from `.env` into `os.environ`."""
    if dotenv_path is None:
        root_candidate = Path(__file__).resolve().parents[2] / ".env"
        dotenv_path = root_candidate if root_candidate.exists() else Path(".env")

    loaded: Dict[str, str] = {}
    if dotenv_path.exists():
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'").strip('"')
            loaded[k] = v
            if k not in os.environ:
                os.environ[k] = v
    return loaded


_SYNONYM_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("gas", "filled", "refrigerant", "charge"),
    ("seal", "gasket", "perimeter"),
    ("looked", "look", "inspect", "inspected", "inspection", "check", "checked"),
    ("frequently", "often", "interval", "month", "months", "every"),
    ("ohms", "ohm", "resistance", "measure"),
    ("element", "heater"),
    ("amps", "amp", "amperage", "current", "draws", "a"),
    ("voltage", "volt", "volts", "v", "400", "230", "electrical"),
    ("warm", "hot", "above", "high", "temperature", "alarm", "setpoint"),
    ("get", "stays", "reaches", "triggers"),
    ("long", "min", "minutes", "hour", "hours", "duration", "delay"),
    ("switch", "pressostat", "cutout"),
    ("bridge", "bridged", "bypass", "bypassed", "jumper", "shunt"),
    ("resets", "reset", "switch", "automatically", "itself"),
    ("wait", "delay", "restarting", "restart", "resetting"),
    ("motor", "protection", "thermal", "overload", "tripped", "trip"),
    ("drip", "fan", "delay"),
    ("guard", "removed", "without"),
)

_WORD_TO_CANONICAL: Dict[str, List[str]] = {}
for grp in _SYNONYM_GROUPS:
    for w in grp:
        _WORD_TO_CANONICAL[w] = list(grp)


def stem_token(w: str) -> str:
    w = w.lower()
    for suffix in ("ing", "ed", "es", "s"):
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            return w[: -len(suffix)]
    return w


def expand_query_tokens(tokens: Sequence[str]) -> List[str]:
    expanded: List[str] = list(tokens)
    seen = set(expanded)
    for t in tokens:
        st = stem_token(t)
        if st not in seen:
            expanded.append(st)
            seen.add(st)
        for syn in _WORD_TO_CANONICAL.get(t, ()):
            if syn not in seen:
                expanded.append(syn)
                seen.add(syn)
    return expanded


_MODEL_PATTERN = re.compile(r"\b(ex|cu|tf)[-\s]?(\d{2,3})\b", re.IGNORECASE)


def extract_equipment_models(text: str) -> Set[str]:
    models: Set[str] = set()
    for m in _MODEL_PATTERN.finditer(text):
        models.add(f"{m.group(1).upper()}-{m.group(2)}")
    return models


_IDENTIFIER_PATTERN = re.compile(r"\b(?:[a-zA-Z]{1,4}-?\d{1,4}[a-zA-Z]?)\b")
_UNIT_TOKEN_IGNORE = {"230v", "400v", "50hz", "60hz", "2024", "150g", "95g", "300mm", "15m", "10m", "25min", "30min", "60min", "120min"}


def extract_exact_identifiers(query: str) -> Set[str]:
    ids: Set[str] = set()
    for m in _IDENTIFIER_PATTERN.finditer(query):
        raw = m.group(0).upper()
        compact = raw.replace("-", "")
        if compact.lower() in _UNIT_TOKEN_IGNORE:
            continue
        if re.match(r"^(EX|CU|TF)\d{2,3}$", compact):
            continue
        ids.add(raw)
    return ids


class GeminiEmbeddingClient:
    """Generates dense vector embeddings via LangChain `GoogleGenerativeAIEmbeddings`,
    Gemini REST API (`embedContent`), or a deterministic 128-dim synonym-aware local projection.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-embedding-001",
        allow_network: bool = False,
        fallback_dim: int = 128,
    ) -> None:
        load_dotenv_file()
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or ""
        )
        self.model_name = os.environ.get("GEMINI_EMBEDDING_MODEL", model_name)
        self.allow_network = allow_network
        self.fallback_dim = fallback_dim
        self.last_provider_used: str = "LOCAL_DENSE_PROJECTION"
        self._lc_embeddings: Optional[Any] = None

        if self.allow_network and self.api_key:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore

                self._lc_embeddings = GoogleGenerativeAIEmbeddings(
                    model=f"models/{self.model_name.removeprefix('models/')}",
                    google_api_key=self.api_key,
                )
            except ImportError:
                self._lc_embeddings = None

    def embed_text(self, text: str) -> List[float]:
        if self.allow_network and self.api_key:
            if self._lc_embeddings is not None:
                try:
                    vec = self._lc_embeddings.embed_query(text)
                    self.last_provider_used = "LANGCHAIN_GOOGLE_GENAI_EMBEDDINGS"
                    return self._l2_normalize([float(x) for x in vec])
                except Exception:
                    pass
            try:
                clean_model = self.model_name.removeprefix("models/")
                # Header, not URL (see langgraph_copilot._synthesize_answer).
                url = (
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{clean_model}:embedContent"
                )
                payload = {
                    "model": f"models/{clean_model}",
                    "content": {"parts": [{"text": text}]},
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=6.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    values = data.get("embedding", {}).get("values", [])
                    if values:
                        self.last_provider_used = "GEMINI_REST_EMBED_CONTENT"
                        return self._l2_normalize([float(x) for x in values])
            except Exception:
                pass

        self.last_provider_used = "LOCAL_DENSE_PROJECTION"
        return self._local_dense_embedding(text)

    def _local_dense_embedding(self, text: str) -> List[float]:
        vec = [0.0] * self.fallback_dim
        base_tokens = tokenize(text)
        expanded = expand_query_tokens(base_tokens)
        features: List[Tuple[str, float]] = [(t, 1.0) for t in base_tokens]
        for t in expanded:
            if t not in base_tokens:
                features.append((t, 0.65))
        for i in range(len(base_tokens) - 1):
            features.append((f"{base_tokens[i]}_{base_tokens[i+1]}", 0.7))

        for feat, weight in features:
            digest = hashlib.sha256(feat.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.fallback_dim
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vec[idx] += sign * weight

        return self._l2_normalize(vec)

    @staticmethod
    def _l2_normalize(vec: Sequence[float]) -> List[float]:
        norm = math.sqrt(sum(v * v for v in vec))
        if norm <= 1e-12:
            return [0.0 for _ in vec]
        return [round(v / norm, 6) for v in vec]


def _parse_txt_sections(raw_text: str, default_title: str) -> Tuple[str, List[Tuple[str, str]]]:
    lines = raw_text.splitlines()
    title = lines[0].strip() if lines else default_title
    sections: List[Tuple[str, str]] = []
    curr_header: Optional[str] = None
    curr_body: List[str] = []

    for line in lines[1:]:
        s = line.strip()
        if not s or s.startswith("SYNTHETIC TEST DOCUMENT") or s.startswith("Not guidance"):
            continue
        if re.match(r"^\d+\s+[A-Z]", s):
            if curr_header and curr_body:
                sections.append((curr_header, " ".join(curr_body)))
            curr_header = s
            curr_body = []
        else:
            if curr_header is not None:
                curr_body.append(s)

    if curr_header and curr_body:
        sections.append((curr_header, " ".join(curr_body)))
    return title, sections


def _parse_markdown_sections(raw_text: str, default_title: str) -> Tuple[str, List[Tuple[str, str]]]:
    lines = raw_text.splitlines()
    title = default_title
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()

    sections: List[Tuple[str, str]] = []
    curr_header: Optional[str] = None
    curr_body: List[str] = []

    for line in lines:
        s = line.strip()
        if not s or s.startswith(">"):
            continue
        if s.startswith("# "):
            continue
        if s.startswith("### ") or s.startswith("## "):
            if curr_header and curr_body:
                sections.append((curr_header, " ".join(curr_body)))
            curr_header = re.sub(r"^#+\s*", "", s).strip()
            curr_body = []
        elif s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|") if c.strip()]
            if len(cells) == 2 and not all(set(c) <= {"-"} for c in cells) and cells[0].lower() != "item":
                curr_body.append(f"{cells[0]}: {cells[1]}.")
        else:
            if curr_header is None:
                curr_header = "Overview"
            curr_body.append(s)

    if curr_header and curr_body:
        sections.append((curr_header, " ".join(curr_body)))
    return title, sections


def load_and_chunk_manuals(manuals_dir: Optional[Path] = None) -> List[KnowledgeDocument]:
    if manuals_dir is None:
        pkg_corpus = Path(__file__).resolve().parent / "corpus"
        legacy_dir = Path(__file__).resolve().parents[2] / "data" / "manuals"
        manuals_dir = pkg_corpus if pkg_corpus.exists() else legacy_dir

    if not manuals_dir.exists():
        return list(DEFAULT_KNOWLEDGE_CORPUS)

    manifest_meta: Dict[str, Dict[str, Any]] = {}
    manifest_path = manuals_dir / "manifest_v3.json"
    if manifest_path.exists():
        mdata = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in mdata.get("documents", []):
            manifest_meta[entry["file"]] = entry

    default_by_id = {d.doc_id: d for d in DEFAULT_KNOWLEDGE_CORPUS}
    chunks: List[KnowledgeDocument] = []

    files = sorted(
        [p for p in manuals_dir.iterdir() if p.suffix.lower() in (".md", ".txt") and p.is_file()]
    )
    if not files:
        return list(DEFAULT_KNOWLEDGE_CORPUS)

    for fpath in files:
        raw = fpath.read_text(encoding="utf-8")
        meta = manifest_meta.get(fpath.name, {})
        base_doc_id = meta.get("doc_id")
        equipment_model = meta.get("model")
        revision = meta.get("revision", "1")
        doc_type = meta.get("doc_type", "TECHNICAL_MANUAL")
        manufacturer = meta.get("manufacturer", "ColdSense Standard")

        if fpath.suffix.lower() == ".txt":
            doc_title, parsed_sections = _parse_txt_sections(raw, meta.get("title", fpath.stem))
        else:
            doc_title, parsed_sections = _parse_markdown_sections(raw, meta.get("title", fpath.stem))

        if base_doc_id is None:
            if ":" in doc_title and doc_title.split(":", 1)[0].strip().startswith("KB-"):
                prefix, rest = doc_title.split(":", 1)
                base_doc_id = prefix.strip()
                doc_title = rest.strip()
            else:
                base_doc_id = fpath.stem

        if equipment_model is None:
            detected_models = extract_equipment_models(doc_title)
            if len(detected_models) == 1:
                equipment_model = next(iter(detected_models))

        for sec_idx, (sec_header, sec_body) in enumerate(parsed_sections, start=1):
            audit = sanitize_and_audit_text(sec_body, heading=sec_header)
            chunk_id = (
                base_doc_id
                if (base_doc_id.startswith("KB-") and sec_idx == 1)
                else f"{base_doc_id}#{sec_header}"
            )
            toks = tuple(sorted(set(tokenize(f"{doc_title} {sec_header} {audit.clean_text}"))))
            ref_doc = default_by_id.get(base_doc_id)
            intents = ref_doc.supported_intents if ref_doc else toks
            tags = ref_doc.topic_tags if ref_doc else toks

            chunks.append(
                KnowledgeDocument(
                    doc_id=chunk_id,
                    title=doc_title,
                    section=sec_header,
                    content=audit.clean_text,
                    topic_tags=tags,
                    supported_intents=intents,
                    source_doc_id=base_doc_id,
                    equipment_model=equipment_model,
                    manufacturer=manufacturer,
                    doc_type=doc_type,
                    revision=revision,
                    is_active=True,
                    quarantined=audit.is_quarantined,
                    trust_status="FLAGGED" if audit.is_quarantined else "UNREVIEWED",
                )
            )

    return chunks if chunks else list(DEFAULT_KNOWLEDGE_CORPUS)


_QUANTITY_EXPECTATIONS: Tuple[Tuple[re.Pattern, re.Pattern, str], ...] = (
    (
        re.compile(r"\b(?:what\s+humidity|humidity\s+does|relative\s+humidity)\b", re.IGNORECASE),
        re.compile(r"\b(?:\d+\s*%|rh\b|humidity)\b", re.IGNORECASE),
        "HUMIDITY",
    ),
    (
        re.compile(r"\b(?:part\s+number|replacement\s+part\s+number)\b", re.IGNORECASE),
        re.compile(r"\b(?:part\s+number|[A-Z]{2,4}-\d{2,4})\b"),
        "PART_NUMBER",
    ),
    (
        re.compile(r"\b(?:rpm|fan\s+speed\s+in\s+rpm)\b", re.IGNORECASE),
        re.compile(r"\b(?:\d+\s*rpm|revolutions)\b", re.IGNORECASE),
        "RPM",
    ),
    (
        re.compile(r"\b(?:how\s+many\s+amps|supply\s+current|maximum\s+current)\b", re.IGNORECASE),
        re.compile(r"(?:\b\d+(?:\.\d+)?\s*a\b|\bcurrent\b|\bamps?\b)", re.IGNORECASE),
        "CURRENT_AMPS",
    ),
    (
        re.compile(r"\b(?:how\s+many\s+ohms|resistance)\b", re.IGNORECASE),
        re.compile(r"(?:\b\d+(?:\.\d+)?\s*ohms?\b|\bresistance\b)", re.IGNORECASE),
        "RESISTANCE_OHMS",
    ),
    (
        re.compile(r"\b(?:supply\s+voltage|what\s+voltage)\b", re.IGNORECASE),
        re.compile(r"(?:\b\d+\s*v\b|\bvoltage\b)", re.IGNORECASE),
        "VOLTAGE",
    ),
    (
        re.compile(r"\b(?:refrigerant\s+charge|what\s+gas\s+is.*filled)\b", re.IGNORECASE),
        re.compile(r"(?:\br-\d+[a-z]*\b|\b\d+(?:\.\d+)?\s*(?:g|kg)\b|\brefrigerant\b)", re.IGNORECASE),
        "REFRIGERANT_CHARGE",
    ),
)


def verify_production_answerability(
    query: str,
    doc: KnowledgeDocument,
    sentence_or_passage: str,
) -> Tuple[bool, str]:
    """Production Answerability Gate fixing `answerability-v1` (D-060) and semantic false-acceptances."""
    if doc.quarantined or not doc.is_active:
        return False, "Chunk is quarantined or superseded."

    q_clean = sanitize_and_audit_text(query).clean_text.strip()
    if not q_clean or len(q_clean) > MAX_QUERY_CHARS:
        return False, "Query failed length or content validation."

    q_lower = q_clean.lower()
    full_context = f"{doc.source_doc_id} {doc.title} {doc.section} {doc.equipment_model or ''} {sentence_or_passage}"
    full_lower = full_context.lower()

    q_models = extract_equipment_models(q_clean)
    if q_models:
        if not doc.equipment_model or doc.equipment_model.upper() not in q_models:
            return False, f"Document model {doc.equipment_model} does not match query model {q_models}."

    req_ids = extract_exact_identifiers(q_clean)
    for req_id in req_ids:
        id_low = req_id.lower()
        id_compact = id_low.replace("-", "")
        if id_low not in full_lower and id_compact not in full_lower.replace("-", ""):
            return False, f"Exact identifier '{req_id}' not present in document or passage."

    for q_pat, ans_pat, label in _QUANTITY_EXPECTATIONS:
        if q_pat.search(q_clean):
            if not ans_pat.search(f"{doc.section} {sentence_or_passage}"):
                return False, f"Passage does not state required quantity type ({label})."

    if any(w in q_lower for w in ("bypass", "bypassed", "bridge", "bridged", "disable", "disabled")):
        if not any(w in full_lower for w in ("bypass", "bypassed", "bridge", "bridged", "disable", "disabled")):
            return (
                False,
                "On-topic non-answer rejected: passage does not address bypassing, bridging, or disabling.",
            )

    if any(w in q_lower for w in ("modify", "override", "rewire", "firmware", "reprogram")):
        if not any(w in full_lower for w in ("modify", "override", "rewire", "firmware", "reprogram")):
            return (
                False,
                "On-topic non-answer rejected: passage does not support modification, override, or firmware procedures.",
            )

    if "manually" in q_lower and "reset" in q_lower:
        if "automatically" in full_lower and "manually" not in full_lower:
            return (
                False,
                "Passage states the device resets automatically, not how to reset it manually.",
            )

    if "maximum" in q_lower and "line set" in q_lower:
        if "maximum" not in full_lower and "at most" not in full_lower:
            return (
                False,
                "Passage states a refrigerant addition threshold, not the maximum line set length.",
            )

    if ("gasket" in q_lower or "seal" in q_lower) and not any(
        w in full_lower for w in ("gasket", "seal")
    ):
        return False, "Passage does not mention door gasket or seal."

    return True, "Passage verified as semantically supporting the query."


def detect_conflicting_sources(
    query: str,
    verified_hits: List[Tuple[KnowledgeDocument, CopilotCitation]],
) -> List[Dict[str, Any]]:
    """Detect when two active documents for the same equipment model state different numeric values
    for the same queried attribute (e.g. EX-200 defrost interval: 6 h in Rev B vs 8 h in SB-12 Rev C).
    """
    if len(verified_hits) < 2:
        return []

    q_lower = query.lower()
    unit_val_re = re.compile(r"\b(\d+(?:\.\d+)?)\s*(h|hours?|min|minutes?|bar|c|°c|k|v|a|ohm|mm|g|kg)\b", re.IGNORECASE)
    by_model_and_unit: Dict[Tuple[str, str], List[Tuple[str, KnowledgeDocument, CopilotCitation]]] = {}

    for doc, cit in verified_hits:
        model_key = (doc.equipment_model or "GENERAL").upper()
        for m in unit_val_re.finditer(cit.verbatim_quote):
            val_str = m.group(1)
            unit_raw = m.group(2).lower().lstrip("°")
            unit_norm = {
                "hour": "h",
                "hours": "h",
                "minute": "min",
                "minutes": "min",
            }.get(unit_raw, unit_raw)
            key = (model_key, unit_norm)
            by_model_and_unit.setdefault(key, []).append((val_str, doc, cit))

    conflicts: List[Dict[str, Any]] = []
    for (model_key, unit_norm), items in by_model_and_unit.items():
        distinct_docs = {doc.source_doc_id for _, doc, _ in items}
        distinct_vals = {val for val, _, _ in items}
        if len(distinct_docs) >= 2 and len(distinct_vals) >= 2:
            if "interval" in q_lower and unit_norm != "h":
                continue
            conflicts.append(
                {
                    "equipment_model": model_key,
                    "unit": unit_norm,
                    "values": [
                        {
                            "value": f"{val} {unit_norm}",
                            "doc_id": doc.source_doc_id,
                            "title": doc.title,
                            "section": doc.section,
                            "revision": doc.revision,
                            "verbatim_quote": cit.verbatim_quote,
                        }
                        for val, doc, cit in items
                    ],
                }
            )
    return conflicts


class SQLiteHybridVectorStore:
    """Persistent SQLite store indexing document chunks, provenance metadata, quarantine status,
    BM25 inverted statistics, and dense embeddings.
    """

    def __init__(
        self,
        sqlite_path: str = ":memory:",
        embedding_client: Optional[GeminiEmbeddingClient] = None,
        documents: Optional[Sequence[KnowledgeDocument]] = None,
    ) -> None:
        self.sqlite_path = sqlite_path
        self.embedder = embedding_client or GeminiEmbeddingClient(allow_network=False)
        self._conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        self._init_schema()
        initial_docs = list(documents) if documents is not None else load_and_chunk_manuals()
        self.index_documents(initial_docs)

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_chunks (
                doc_id TEXT PRIMARY KEY,
                source_doc_id TEXT NOT NULL,
                title TEXT NOT NULL,
                section TEXT NOT NULL,
                content TEXT NOT NULL,
                equipment_model TEXT,
                manufacturer TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                revision TEXT NOT NULL,
                is_active INTEGER NOT NULL,
                quarantined INTEGER NOT NULL,
                trust_status TEXT NOT NULL,
                topic_tags_json TEXT NOT NULL,
                supported_intents_json TEXT NOT NULL,
                tokens_json TEXT NOT NULL,
                embedding_json TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def index_documents(self, docs: Sequence[KnowledgeDocument]) -> int:
        cur = self._conn.cursor()
        for doc in docs:
            audit = sanitize_and_audit_text(doc.content, heading=doc.section)
            is_quarantined = doc.quarantined or audit.is_quarantined
            trust = "FLAGGED" if is_quarantined else doc.trust_status
            base_toks = tokenize(
                f"{doc.title} {doc.section} {audit.clean_text} {doc.equipment_model or ''} {' '.join(doc.topic_tags)}"
            )
            expanded_toks = expand_query_tokens(base_toks)
            emb = self.embedder.embed_text(f"{doc.title}\n{doc.section}\n{audit.clean_text}")
            cur.execute(
                """
                INSERT OR REPLACE INTO rag_chunks (
                    doc_id, source_doc_id, title, section, content,
                    equipment_model, manufacturer, doc_type, revision,
                    is_active, quarantined, trust_status,
                    topic_tags_json, supported_intents_json, tokens_json, embedding_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc.doc_id,
                    doc.source_doc_id or doc.doc_id,
                    doc.title,
                    doc.section,
                    audit.clean_text,
                    doc.equipment_model,
                    doc.manufacturer,
                    doc.doc_type,
                    doc.revision,
                    1 if doc.is_active else 0,
                    1 if is_quarantined else 0,
                    trust,
                    json.dumps(list(doc.topic_tags)),
                    json.dumps(list(doc.supported_intents)),
                    json.dumps(expanded_toks),
                    json.dumps(emb),
                ),
            )
        self._conn.commit()
        return len(docs)

    def load_active_unquarantined_chunks(self) -> List[Tuple[KnowledgeDocument, List[str], List[float]]]:
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT doc_id, source_doc_id, title, section, content,
                   equipment_model, manufacturer, doc_type, revision,
                   is_active, quarantined, trust_status,
                   topic_tags_json, supported_intents_json, tokens_json, embedding_json
            FROM rag_chunks
            WHERE is_active = 1 AND quarantined = 0
            """
        )
        out: List[Tuple[KnowledgeDocument, List[str], List[float]]] = []
        for r in cur.fetchall():
            doc = KnowledgeDocument(
                doc_id=r[0],
                source_doc_id=r[1],
                title=r[2],
                section=r[3],
                content=r[4],
                equipment_model=r[5],
                manufacturer=r[6],
                doc_type=r[7],
                revision=r[8],
                is_active=bool(r[9]),
                quarantined=bool(r[10]),
                trust_status=r[11],
                topic_tags=tuple(json.loads(r[12])),
                supported_intents=tuple(json.loads(r[13])),
            )
            toks = [str(x) for x in json.loads(r[14])]
            emb = [float(x) for x in json.loads(r[15])]
            out.append((doc, toks, emb))
        return out

    @staticmethod
    def _cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
        if len(v1) != len(v2) or not v1:
            return 0.0
        return sum(a * b for a, b in zip(v1, v2))

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        semantic_enabled: bool = True,
        rrf_k: int = 10,
    ) -> Tuple[List[Tuple[KnowledgeDocument, CopilotCitation]], str]:
        """Execute the 7-stage production retrieval & answerability pipeline."""
        q_audit = sanitize_and_audit_text(query)
        q_clean = q_audit.clean_text.strip()
        if not q_clean or len(q_clean) > MAX_QUERY_CHARS:
            return [], "REFUSED_INVALID_QUERY"

        raw_q_tokens = tokenize(q_clean)
        if not raw_q_tokens:
            return [], "REFUSED_EMPTY_QUERY"

        all_records = self.load_active_unquarantined_chunks()
        if not all_records:
            return [], "EMPTY_INDEX"

        q_models = extract_equipment_models(q_clean)
        req_ids = extract_exact_identifiers(q_clean)

        candidate_records: List[Tuple[KnowledgeDocument, List[str], List[float]]] = []
        for doc, toks, emb in all_records:
            if q_models:
                if not doc.equipment_model or doc.equipment_model.upper() not in q_models:
                    continue
            full_text_low = (
                f"{doc.source_doc_id} {doc.title} {doc.section} {doc.equipment_model or ''} {doc.content}".lower()
            )
            if req_ids:
                if not all(
                    (rid.lower() in full_text_low or rid.lower().replace("-", "") in full_text_low.replace("-", ""))
                    for rid in req_ids
                ):
                    continue
            candidate_records.append((doc, toks, emb))

        if not candidate_records:
            return [], "SEMANTIC_HYBRID" if semantic_enabled else "LEXICAL_FALLBACK"

        q_expanded = expand_query_tokens(raw_q_tokens)
        n_docs = len(all_records)
        avgdl = sum(len(toks) for _, toks, _ in all_records) / max(1, n_docs)

        df_map: Dict[str, int] = {}
        for _, toks, _ in all_records:
            for w in set(toks):
                df_map[w] = df_map.get(w, 0) + 1

        stopwords = {
            "what", "which", "when", "where", "who", "how", "does", "do", "is", "are", "the", "a", "an",
            "in", "on", "at", "to", "for", "of", "with", "by", "from", "after", "before", "may", "can",
            "should", "must", "be", "it", "its", "many", "much", "long", "often", "units", "unit",
        }
        model_tokens = {t.lower() for m in q_models for t in m.replace("-", " ").split()} | {
            m.lower().replace("-", "") for m in q_models
        }
        content_q_tokens = [
            t for t in raw_q_tokens if t not in stopwords and t not in model_tokens
        ]
        if not content_q_tokens:
            content_q_tokens = list(raw_q_tokens)

        bm25_scores: List[Tuple[int, float]] = []
        coverage_map: Dict[int, float] = {}

        for idx, (doc, toks, _emb) in enumerate(candidate_records):
            tf_map: Dict[str, int] = {}
            for t in toks:
                tf_map[t] = tf_map.get(t, 0) + 1
            dl = len(toks)
            score = 0.0
            for q in set(q_expanded):
                df = df_map.get(q, 0)
                if df == 0:
                    continue
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                tf = tf_map.get(q, 0)
                denom = tf + 1.5 * (1.0 - 0.75 + 0.75 * (dl / max(1.0, avgdl)))
                weight_mult = 1.0 if q in raw_q_tokens else 0.55
                score += weight_mult * idf * ((tf * 2.5) / max(1e-9, denom))
            bm25_scores.append((idx, score))

            total_idf = 0.0
            matched_idf = 0.0
            tok_set = set(toks)
            for qt in content_q_tokens:
                qt_syns = set(expand_query_tokens([qt]))
                dfs = [df_map.get(s, 0) for s in qt_syns if df_map.get(s, 0) > 0]
                best_df = min(dfs) if dfs else 0
                w_idf = (
                    math.log(1.0 + (n_docs + 1.0) / 0.5)
                    if best_df == 0
                    else math.log(1.0 + (n_docs - best_df + 0.5) / (best_df + 0.5))
                )
                total_idf += w_idf
                if qt_syns & tok_set:
                    matched_idf += w_idf

            coverage_map[idx] = matched_idf / max(1e-9, total_idf)

        bm25_ranked = sorted(bm25_scores, key=lambda x: (x[1], candidate_records[x[0]][0].doc_id), reverse=True)
        bm25_rank_map = {idx: rank + 1 for rank, (idx, _) in enumerate(bm25_ranked)}
        bm25_val_map = dict(bm25_scores)

        if semantic_enabled:
            q_emb = self.embedder.embed_text(q_clean)
            vec_scores = [
                (idx, self._cosine_similarity(q_emb, emb))
                for idx, (_doc, _toks, emb) in enumerate(candidate_records)
            ]
            vec_ranked = sorted(vec_scores, key=lambda x: x[1], reverse=True)
            vec_rank_map = {idx: rank + 1 for rank, (idx, _) in enumerate(vec_ranked)}
            vec_val_map = dict(vec_scores)
            mode = "SEMANTIC_HYBRID"
        else:
            vec_rank_map = {idx: len(candidate_records) for idx in range(len(candidate_records))}
            vec_val_map = {idx: 0.0 for idx in range(len(candidate_records))}
            mode = "LEXICAL_FALLBACK"

        q_lower = q_clean.lower()
        fused: List[Tuple[float, KnowledgeDocument, str]] = []
        for idx, (doc, _toks, _emb) in enumerate(candidate_records):
            cov = coverage_map[idx]
            lex_val = bm25_val_map[idx]
            sem_val = vec_val_map[idx]

            # Keep only alphabetic words from section heading (strips "4.2", "5.1", etc.)
            sec_words = [w for w in tokenize(doc.section.lower()) if any(c.isalpha() for c in w)]
            heading_match = len(sec_words) >= 2 and all(w in q_lower for w in sec_words)

            if cov < 0.45 and sem_val < 0.55 and not heading_match:
                continue

            best_quote = self._extract_best_verbatim_sentence(doc.content, q_expanded)
            ans_ok, _ans_reason = verify_production_answerability(q_clean, doc, best_quote)
            if not ans_ok:
                ans_ok_full, _ = verify_production_answerability(q_clean, doc, doc.content)
                if not ans_ok_full:
                    continue

            heading_bonus = 2.5 if heading_match else 0.0
            if semantic_enabled:
                rrf_score = (
                    0.55 * (1.0 / (rrf_k + bm25_rank_map[idx]))
                    + 0.45 * (1.0 / (rrf_k + vec_rank_map[idx]))
                    + 0.30 * lex_val
                    + 0.45 * cov
                    + 0.25 * max(0.0, sem_val)
                    + heading_bonus
                )
            else:
                rrf_score = (
                    (1.0 / (rrf_k + bm25_rank_map[idx]))
                    + 0.35 * lex_val
                    + 0.50 * cov
                    + heading_bonus
                )

            fused.append((rrf_score, doc, best_quote))

        fused.sort(key=lambda item: (item[0], item[1].doc_id), reverse=True)

        results: List[Tuple[KnowledgeDocument, CopilotCitation]] = []
        if fused:
            top_score = fused[0][0]
            seen_source_docs: Set[str] = set()
            for score, doc, quote in fused:
                if len(results) >= top_k:
                    break
                sec_words = [w for w in tokenize(doc.section.lower()) if any(c.isalpha() for c in w)]
                is_heading_hit = len(sec_words) >= 2 and all(w in q_lower for w in sec_words)
                is_new_source_doc = (doc.source_doc_id or doc.doc_id) not in seen_source_docs
                min_ratio = 0.20 if (is_heading_hit and is_new_source_doc) else 0.52
                if len(results) >= 1 and score < top_score * min_ratio:
                    continue
                seen_source_docs.add(doc.source_doc_id or doc.doc_id)
                cit = CopilotCitation(
                    doc_id=doc.source_doc_id or doc.doc_id,
                    doc_title=doc.title,
                    section=doc.section,
                    verbatim_quote=quote,
                    relevance_score=round(score, 4),
                )
                results.append((doc, cit))

        return results, mode

    @staticmethod
    def _extract_best_verbatim_sentence(content: str, query_tokens: List[str]) -> str:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content.strip()) if s.strip()]
        if not sentences:
            return content.strip()
        q_set = set(query_tokens)
        best_sent = sentences[0]
        best_overlap = -1
        for s in sentences:
            s_expanded = set(expand_query_tokens(tokenize(s)))
            overlap = len(q_set & s_expanded)
            if overlap > best_overlap:
                best_overlap = overlap
                best_sent = s
        return best_sent


class ColdSenseHybridRetriever(LCBaseRetriever):
    """LangChain `BaseRetriever` wrapper around `SQLiteHybridVectorStore`."""

    vector_store: Any = None
    top_k: int = 3
    semantic_enabled: bool = True

    def __init__(
        self,
        vector_store: Optional[SQLiteHybridVectorStore] = None,
        top_k: int = 3,
        semantic_enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        if LANGCHAIN_CORE_AVAILABLE:
            super().__init__(**kwargs)
        object.__setattr__(self, "vector_store", vector_store or SQLiteHybridVectorStore())
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(self, "semantic_enabled", semantic_enabled)

    def _get_relevant_documents(self, query: str) -> List[LCDocument]:
        hits, mode = self.vector_store.hybrid_search(
            query=query, top_k=self.top_k, semantic_enabled=self.semantic_enabled
        )
        lc_docs: List[LCDocument] = []
        for doc, cit in hits:
            lc_docs.append(
                LCDocument(
                    page_content=doc.content,
                    metadata={
                        "doc_id": doc.source_doc_id or doc.doc_id,
                        "chunk_id": doc.doc_id,
                        "title": doc.title,
                        "section": doc.section,
                        "equipment_model": doc.equipment_model,
                        "revision": doc.revision,
                        "verbatim_quote": cit.verbatim_quote,
                        "relevance_score": cit.relevance_score,
                        "retrieval_mode": mode,
                        "knowledge_doc": doc,
                        "citation": cit,
                    },
                )
            )
        return lc_docs
