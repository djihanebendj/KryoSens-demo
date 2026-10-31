"""Trust boundary for documents. Everything a document contains is untrusted data.

Rules (docs/KNOWLEDGE.md §6):

* files: an allowed extension whose bytes match the format, a regular file, at most ``MAX_FILE_BYTES``; DOCX
  archives are checked for size, entry count and compression ratio, and XML with a DOCTYPE is refused;
* invisible Unicode (zero-width, bidirectional controls, other format characters) is removed and flagged.
  Unicode TAG characters can spell hidden ASCII text: the hidden text is decoded for inspection and the chunk is
  QUARANTINED, since a manual has no legitimate reason to hide text;
* text addressed to an AI system (instruction-override patterns, chat-role markers) QUARANTINES the chunk: it is
  stored for review, never retrieved, and the document is FLAGGED;
* secrets (private keys, access tokens, ``password=...``, values of secret-looking environment variables) are
  redacted before anything is stored or indexed;
* supplied metadata with control characters, invisible characters or instruction patterns is refused.

Detection is pattern-based: it reduces risk, it does not prove a document is safe.
"""

from __future__ import annotations

import os
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .schemas import ChunkFlag

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_DOCX_UNCOMPRESSED = 60 * 1024 * 1024
MAX_DOCX_ENTRIES = 2000
MAX_DOCX_RATIO = 200
MAX_PDF_PAGES = 2000
MAX_QUERY_CHARS = 1000

FORMATS = {".txt": "TXT", ".md": "MARKDOWN", ".markdown": "MARKDOWN", ".pdf": "PDF", ".docx": "DOCX"}

# ---------------------------------------------------------------- invisible and hidden characters
_TAG_RANGE = (0xE0000, 0xE007F)
_KEEP_FORMAT = {"­"}                       # soft hyphen: harmless, still removed from text but not flagged
_BIDI = set("‪‫‬‭‮⁦⁧⁨⁩‎‏؜")


@dataclass
class CleanText:
    text: str
    removed: int = 0
    hidden: str = ""                            # ASCII decoded from TAG characters
    bidi: int = 0


def clean_invisible(text: str) -> CleanText:
    out, removed, hidden, bidi = [], 0, [], 0
    for ch in text:
        cp = ord(ch)
        if _TAG_RANGE[0] <= cp <= _TAG_RANGE[1]:
            removed += 1
            if 0xE0020 <= cp <= 0xE007E:
                hidden.append(chr(cp - 0xE0000))
            continue
        if ch in _BIDI:
            removed += 1
            bidi += 1
            continue
        if unicodedata.category(ch) == "Cf":     # zero-width space/joiners, word joiner, BOM, ...
            if ch not in _KEEP_FORMAT:
                removed += 1
            continue
        if unicodedata.category(ch) == "Cc" and ch not in "\n\t\r":
            removed += 1
            continue
        out.append(ch)
    return CleanText("".join(out), removed, "".join(hidden), bidi)


# ---------------------------------------------------------------- instructions addressed to an AI system
INSTRUCTION_PATTERNS = {
    # deliberately narrow: ordinary manual wording ("disregard the previous rules", "run the command",
    # "## System overview") must not match; a match quarantines the chunk for human review
    "override-previous": r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all|"
                         r"your)\b[^.\n]{0,30}\b(instructions|prompts?|system messages?)\b",
    "role-reassignment": r"\byou are now (an? )?(ai|assistant|chatbot|language model|unrestricted)|\bact as (an? )?"
                         r"(ai|assistant|chatbot|language model)\b|\bnew (system )?instructions\s*:",
    "system-prompt": r"\b(system|developer|hidden) prompt\b|\bdeveloper mode\b|\bjailbreak\b",
    "secrecy-to-user": r"\bdo not (tell|inform|reveal)[^.\n]{0,20}\b(the )?user\b",
    "chat-markers": r"<\|?(im_start|im_end|system|assistant)\|?>|\[/?(inst|sys)\]|<</?sys>>",
    "exfiltration": r"\b(send|post|upload|email)\b[^.\n]{0,40}\b(api[ _-]?keys?|credentials|passwords?|access tokens?|"
                    r"environment variables?)\b",
    "tool-invocation": r"\b(call|invoke) (the |a )?(tool|function call)\b|\b(run|execute) (a |the )?shell\b",
}
_INSTR = {k: re.compile(v, re.IGNORECASE | re.MULTILINE) for k, v in INSTRUCTION_PATTERNS.items()}


def instruction_patterns(text: str) -> list[str]:
    return [k for k, rx in _INSTR.items() if rx.search(text)]


# ---------------------------------------------------------------- secrets
SECRET_PATTERNS = {
    "private-key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(-----END [A-Z ]*PRIVATE KEY-----|$)",
    "aws-access-key": r"\b(AKIA|ASIA)[0-9A-Z]{16}\b",
    "github-token": r"\bgh[pousr]_[A-Za-z0-9]{30,}\b",
    "slack-token": r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b",
    "api-key-prefix": r"\bsk-(ant-)?[A-Za-z0-9_-]{20,}\b",
    "jwt": r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b",
    "bearer": r"\bBearer\s+[A-Za-z0-9._~+/-]{16,}=*",
    "assignment": r"(?i)\b(api[_-]?key|secret|token|password|passwd|pwd|access[_-]?key)\b\s*[:=]\s*[\"']?"
                  r"[^\s\"',;]{6,}",
}
_SECRETS = {k: re.compile(v) for k, v in SECRET_PATTERNS.items()}
_SECRET_ENV = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)", re.IGNORECASE)


def _env_secrets() -> list[str]:
    return sorted({v for k, v in os.environ.items() if _SECRET_ENV.search(k) and len(v) >= 8}, key=len, reverse=True)


def redact_secrets(text: str) -> tuple[str, list[str]]:
    found = []
    for value in _env_secrets():
        if value in text:
            text = text.replace(value, "[REDACTED:environment-secret]")
            found.append("environment-secret")
    for name, rx in _SECRETS.items():
        text, n = rx.subn(f"[REDACTED:{name}]", text)
        found += [name] * n
    return text, found


# ---------------------------------------------------------------- one block of extracted text
@dataclass
class Screened:
    text: str
    flags: list[ChunkFlag] = field(default_factory=list)
    details: list[str] = field(default_factory=list)


def screen(text: str) -> Screened:
    """Clean, inspect and redact one piece of extracted text before it is stored or indexed."""
    c = clean_invisible(text)
    s = Screened(c.text)
    if c.removed:
        s.flags.append(ChunkFlag.INVISIBLE_UNICODE_REMOVED)
        s.details.append(f"{c.removed} invisible character(s) removed ({c.bidi} bidirectional control)")
    if c.hidden.strip():
        s.flags.append(ChunkFlag.HIDDEN_TEXT)
        s.details.append(f"hidden tag-character text ({len(c.hidden)} characters) decoded and quarantined")
    hits = sorted(set(instruction_patterns(c.text)) | set(instruction_patterns(c.hidden)))
    if hits:
        s.flags.append(ChunkFlag.INSTRUCTION_PATTERN)
        s.details.append("text addressed to an AI system: " + ", ".join(hits))
    s.text, secrets = redact_secrets(s.text)
    if secrets:
        s.flags.append(ChunkFlag.SECRET_REDACTED)
        s.details.append("redacted: " + ", ".join(sorted(set(secrets))))
    return s


def metadata_problem(value: str) -> str | None:
    if any(unicodedata.category(ch) in ("Cc", "Cf") or _TAG_RANGE[0] <= ord(ch) <= _TAG_RANGE[1] for ch in value):
        return "control or invisible characters"
    hits = instruction_patterns(value)
    if hits:
        return "text addressed to an AI system (" + ", ".join(hits) + ")"
    if redact_secrets(value)[1]:
        return "looks like a secret"
    return None


def clean_metadata_from_document(value: str | None, limit: int = 200) -> tuple[str | None, str | None]:
    """A title read from inside the document: kept only if it is safe. Returns (value, reason it was dropped)."""
    if not value:
        return None, None
    c = clean_invisible(value).text.strip()
    if not c:
        return None, None
    problem = metadata_problem(c)
    if problem or c != c[:limit]:
        return None, problem or f"longer than {limit} characters"
    return c, None


# ---------------------------------------------------------------- files
class Rejected(Exception):
    """The file fails a safety check; the message never contains file content."""


def check_file(path: Path, max_bytes: int | None = None) -> tuple[str, bytes]:
    max_bytes = MAX_FILE_BYTES if max_bytes is None else max_bytes
    if path.is_symlink():
        raise Rejected("symbolic links are not ingested; give the file itself")
    if not path.is_file():
        raise Rejected("not a regular file")
    fmt = FORMATS.get(path.suffix.lower())
    if fmt is None:
        raise Rejected(f"unsupported extension '{path.suffix}': supported are {', '.join(sorted(FORMATS))}")
    size = path.stat().st_size
    if size > max_bytes:
        raise Rejected(f"file is {size} bytes; the limit is {max_bytes}")
    data = path.read_bytes()
    if fmt == "PDF" and not data.startswith(b"%PDF-"):
        raise Rejected("the extension says PDF but the content is not a PDF")
    if fmt == "DOCX":
        check_docx_archive(data)
    if fmt in ("TXT", "MARKDOWN"):
        if b"\x00" in data:
            raise Rejected("binary content (NUL bytes) in a text file")
        if data.startswith((b"%PDF-", b"PK\x03\x04")):
            raise Rejected("the extension says text but the content is a PDF or an archive")
    return fmt, data


def check_docx_archive(data: bytes) -> None:
    import io

    if not data.startswith(b"PK\x03\x04"):
        raise Rejected("the extension says DOCX but the content is not a ZIP archive")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            infos = z.infolist()
            if len(infos) > MAX_DOCX_ENTRIES:
                raise Rejected(f"archive has {len(infos)} entries; the limit is {MAX_DOCX_ENTRIES}")
            total = sum(i.file_size for i in infos)
            if total > MAX_DOCX_UNCOMPRESSED:
                raise Rejected(f"archive expands to {total} bytes; the limit is {MAX_DOCX_UNCOMPRESSED}")
            for i in infos:
                if i.compress_size and i.file_size / i.compress_size > MAX_DOCX_RATIO:
                    raise Rejected("archive entry compression ratio too high (possible archive bomb)")
                if i.flag_bits & 0x1:
                    raise Rejected("encrypted archive entries are not supported")
            if "word/document.xml" not in {i.filename for i in infos}:
                raise Rejected("not a Word document: word/document.xml is missing")
    except zipfile.BadZipFile as e:
        raise Rejected("damaged ZIP archive") from e


def safe_xml(raw: bytes):
    """Parse XML from a document archive; DOCTYPE and entity declarations are refused (no expansion attacks)."""
    import xml.etree.ElementTree as ET

    head = raw[:4096].upper()
    if b"<!DOCTYPE" in head or b"<!ENTITY" in raw.upper():
        raise Rejected("XML with DOCTYPE or entity declarations is refused")
    return ET.fromstring(raw)
