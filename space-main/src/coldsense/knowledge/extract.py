"""Text extraction with source locations: TXT and Markdown (lines), DOCX (paragraphs; standard library only),
PDF (pages; needs the optional ``pypdf`` package). Nothing is OCR'd: a PDF without a text layer is reported as
NO_TEXT_LAYER, never guessed at."""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field

from .schemas import ExtractionStatus
from .security import MAX_PDF_PAGES, Rejected, safe_xml


@dataclass
class Block:
    kind: str                      # heading | text | table
    text: str
    level: int = 0                 # headings only
    page: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    paragraph: int | None = None


@dataclass
class Extraction:
    status: ExtractionStatus
    blocks: list[Block] = field(default_factory=list)
    title: str | None = None       # as found inside the document; screened later
    detail: str | None = None


_NUMBERED = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,4})\.?\s+[A-Za-z].{0,78}$")
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_LIST_ITEM = re.compile(r"^\s*([-*+•]|\d{1,2}[.)])\s+")


def _plain_heading(line: str) -> int:
    """Heading level of a plain-text line, or 0. Numbered ('4.2 Defrost interval') or short UPPER CASE lines."""
    s = line.strip()
    if not s or s.endswith((".", ",", ";", ":")) or len(s) > 80:
        return 0
    m = _NUMBERED.match(s)
    if m:
        return m.group(1).count(".") + 1
    letters = [c for c in s if c.isalpha()]
    if 3 <= len(letters) and s.upper() == s and len(s) <= 60 and not re.search(r"\d{3,}", s):
        return 1
    return 0


def _lines_to_blocks(lines: list[str], *, page: int | None = None, markdown: bool = False,
                     first_line: int = 1) -> list[Block]:
    blocks: list[Block] = []
    buf: list[str] = []
    start = None
    kind = "text"
    fence = False

    def flush(end: int) -> None:
        nonlocal buf, start, kind
        if kind == "text" and not (markdown and buf and buf[0].lstrip().startswith("```")):
            text = ""
            for line in buf:                       # wrapped lines form one paragraph; list items keep a line each
                sep = "\n" if _LIST_ITEM.match(line) else " "
                text += (sep if text else "") + line.strip()
            text = text.strip()
        else:
            text = "\n".join(buf).strip()
        if text:
            blocks.append(Block(kind, text, page=page, line_start=start, line_end=end))
        buf, start, kind = [], None, "text"

    for i, line in enumerate(lines, first_line):
        s = line.rstrip()
        if markdown and s.lstrip().startswith("```"):
            if not fence:
                flush(i - 1)
                start, fence = i, True
                buf.append(s)
                continue
            buf.append(s)
            fence = False
            flush(i)
            continue
        if fence:
            buf.append(s)
            continue
        is_table = markdown and s.lstrip().startswith("|")
        if markdown:
            m = _MD_HEADING.match(s)
            level, title = (len(m.group(1)), m.group(2)) if m else (0, None)
        else:
            level, title = _plain_heading(s), s.strip()
        numbered_short = (not markdown and _NUMBERED.match(s.strip()) is not None and len(s.split()) <= 8
                          and s.strip().split(None, 1)[1][:1].isupper())
        if level and (markdown or not buf or numbered_short):
            flush(i - 1)
            blocks.append(Block("heading", title, level=level, page=page, line_start=i, line_end=i))
            continue
        if not s.strip():
            flush(i - 1)
            continue
        if buf and (kind == "table") != is_table:
            flush(i - 1)
        if not buf:
            start = i
            kind = "table" if is_table else "text"
        buf.append(s)
    flush(first_line + len(lines) - 1)
    return blocks


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise Rejected("text is not valid UTF-8 (convert it to UTF-8 first; no encoding is guessed)") from e


def extract_text(data: bytes, markdown: bool) -> Extraction:
    lines = _decode(data).replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks = _lines_to_blocks(lines, markdown=markdown)
    title = None
    if markdown:
        title = next((b.text for b in blocks if b.kind == "heading" and b.level == 1), None)
    if not any(b.kind != "heading" for b in blocks):
        return Extraction(ExtractionStatus.EMPTY_TEXT, blocks, title, "no text outside headings")
    return Extraction(ExtractionStatus.EXTRACTED, blocks, title)


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_DC = "{http://purl.org/dc/elements/1.1/}"


def _w_text(el) -> str:
    out = []
    for node in el.iter():
        if node.tag == _W + "t" and node.text:
            out.append(node.text)
        elif node.tag == _W + "tab":
            out.append("\t")
        elif node.tag in (_W + "br", _W + "cr"):
            out.append("\n")
    return "".join(out)


def extract_docx(data: bytes) -> Extraction:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        root = safe_xml(z.read("word/document.xml"))
        title = None
        if "docProps/core.xml" in z.namelist():
            core = safe_xml(z.read("docProps/core.xml"))
            node = core.find(_DC + "title")
            title = node.text if node is not None and node.text else None
    body = root.find(_W + "body")
    blocks: list[Block] = []
    para = 0
    for el in (list(body) if body is not None else []):
        if el.tag == _W + "p":
            para += 1
            text = _w_text(el).strip()
            if not text:
                continue
            style = el.find(f"{_W}pPr/{_W}pStyle")
            sval = style.get(_W + "val", "") if style is not None else ""
            m = re.match(r"(?i)^heading\s*(\d)$", sval)
            if m or sval.lower() == "title":
                level = int(m.group(1)) if m else 1
                blocks.append(Block("heading", text, level=level, paragraph=para))
                if sval.lower() == "title" and title is None:
                    title = text
            else:
                blocks.append(Block("text", text, paragraph=para))
        elif el.tag == _W + "tbl":
            para += 1
            rows = []
            for tr in el.iter(_W + "tr"):
                cells = [" ".join(_w_text(tc).split()) for tc in tr.iter(_W + "tc")]
                rows.append("| " + " | ".join(cells) + " |")
            if rows:
                blocks.append(Block("table", "\n".join(rows), paragraph=para))
    if not any(b.kind != "heading" for b in blocks):
        return Extraction(ExtractionStatus.EMPTY_TEXT, blocks, title, "the document body has no text")
    return Extraction(ExtractionStatus.EXTRACTED, blocks, title)


def extract_pdf(data: bytes) -> Extraction:
    try:
        from pypdf import PdfReader
    except ImportError:
        return Extraction(ExtractionStatus.MISSING_DEPENDENCY,
                          detail="PDF extraction needs the optional 'pypdf' package "
                                 "(pip install 'coldsense[knowledge]')")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            return Extraction(ExtractionStatus.FAILED, detail="encrypted PDF: not supported")
        n = len(reader.pages)
        if n > MAX_PDF_PAGES:
            raise Rejected(f"PDF has {n} pages; the limit is {MAX_PDF_PAGES}")
        title = None
        try:
            title = reader.metadata.title if reader.metadata else None
        except Exception:                                          # malformed metadata: ignored, not guessed
            title = None
        blocks: list[Block] = []
        empty_pages = 0
        for p, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if not text.strip():
                empty_pages += 1
                continue
            blocks += _lines_to_blocks(text.replace("\r", "\n").split("\n"), page=p)
    except Rejected:
        raise
    except Exception as e:                                         # parser failure: reported, never partial
        return Extraction(ExtractionStatus.FAILED, detail=f"the PDF could not be read ({type(e).__name__})")
    if not any(b.kind != "heading" for b in blocks):
        return Extraction(ExtractionStatus.NO_TEXT_LAYER, blocks, title,
                          f"no extractable text on any of {n} page(s): a scanned or image-only PDF? OCR is not "
                          "supported")
    detail = (f"{empty_pages} of {n} page(s) have no text layer (scanned or image-only?); their content is not "
              "indexed" if empty_pages else None)
    return Extraction(ExtractionStatus.EXTRACTED, blocks, title, detail)


def extract(fmt: str, data: bytes) -> Extraction:
    if fmt == "TXT":
        return extract_text(data, markdown=False)
    if fmt == "MARKDOWN":
        return extract_text(data, markdown=True)
    if fmt == "DOCX":
        return extract_docx(data)
    if fmt == "PDF":
        return extract_pdf(data)
    return Extraction(ExtractionStatus.UNSUPPORTED_FORMAT, detail=f"format {fmt} is not supported")
