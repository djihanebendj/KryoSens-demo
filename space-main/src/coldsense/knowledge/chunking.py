"""Section-aware chunking. A chunk never crosses a heading; tables stay whole up to ``max_chunk_chars``; overlap
is whole sentences from the same section; a long sentence is never split between a number and its unit."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .extract import Block
from .schemas import SourceLocation


class ChunkingParams(BaseModel):
    """Every chunking number. Defaults are provisional; change them per corpus, not per environment."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_chars: int = Field(1200, ge=200, le=8000, description="target chunk length, characters")
    overlap_chars: int = Field(150, ge=0, le=2000, description="sentences carried into the next chunk, characters")
    max_chunk_chars: int = Field(3000, ge=400, le=16000, description="hard limit; tables up to it stay whole")
    max_chunks_per_document: int = Field(5000, ge=1, le=100000)

    @model_validator(mode="after")
    def _consistent(self) -> ChunkingParams:
        if self.overlap_chars >= self.chunk_chars // 2:
            raise ValueError("overlap_chars must be less than half of chunk_chars")
        if self.max_chunk_chars < self.chunk_chars:
            raise ValueError("max_chunk_chars must be at least chunk_chars")
        return self


@dataclass
class Draft:
    text: str
    location: SourceLocation
    sources: list[Block] = field(default_factory=list)   # the blocks it was built from, headings above it included


@dataclass
class _Piece:
    text: str
    block: Block
    sentence: bool = True


@dataclass
class _Acc:
    pieces: list[_Piece] = field(default_factory=list)
    fresh: int = 0                                  # pieces added since the last chunk (overlap excluded)

    def size(self) -> int:
        return sum(len(p.text) + 1 for p in self.pieces)


_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_NUMBER_END = re.compile(r"[-+]?\d+(?:[.,]\d+)?$")


def sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE.split(text.strip()) if s.strip()]


def _hard_split(text: str, limit: int) -> list[str]:
    """Split an over-long sentence at spaces, never right after a number (keeps '5 °C', '12 bar' together)."""
    out, rest = [], text
    while len(rest) > limit:
        cut = rest.rfind(" ", 0, limit)
        while cut > 0 and _NUMBER_END.search(rest[:cut]):
            cut = rest.rfind(" ", 0, cut)
        if cut <= 0:
            cut = limit
        out.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        out.append(rest)
    return out


def _location(pieces: list[_Piece], section: str | None) -> SourceLocation:
    def span(attr_start, attr_end):
        vals = [getattr(p.block, a) for p in pieces for a in (attr_start, attr_end) if getattr(p.block, a) is not None]
        return (min(vals), max(vals)) if vals else (None, None)

    ps, pe = span("page", "page")
    ls, le = span("line_start", "line_end")
    gs, ge = span("paragraph", "paragraph")
    return SourceLocation(page_start=ps, page_end=pe, line_start=ls, line_end=le, paragraph_start=gs,
                          paragraph_end=ge, section=section)


def chunk_blocks(blocks: list[Block], params: ChunkingParams | None = None) -> list[Draft]:
    p = params or ChunkingParams()
    drafts: list[Draft] = []
    stack: list[tuple[int, str, Block]] = []
    section: str | None = None
    acc = _Acc()

    def close(carry: bool) -> None:
        nonlocal acc
        if not acc.fresh:                           # nothing new since the last chunk: no overlap-only chunk
            acc = _Acc()
            return
        text = ""
        for k, x in enumerate(acc.pieces):
            text += ("" if k == 0 else " " if x.block is acc.pieces[k - 1].block and x.sentence else "\n") + x.text
        srcs = [h for _, _, h in stack] + [x.block for x in acc.pieces]
        drafts.append(Draft(text.strip(), _location(acc.pieces, section), list({id(b): b for b in srcs}.values())))
        keep: list[_Piece] = []
        if carry and p.overlap_chars:
            size = 0
            for x in reversed(acc.pieces):
                if not x.sentence or size + len(x.text) > p.overlap_chars:
                    break
                keep.insert(0, x)
                size += len(x.text) + 1
        acc = _Acc(keep)

    for b in blocks:
        if b.kind == "heading":
            close(carry=False)
            while stack and stack[-1][0] >= b.level:
                stack.pop()
            stack.append((b.level, " ".join(b.text.split())[:120], b))
            section = " > ".join(t for _, t, _ in stack)
            continue
        if b.kind == "table":
            rows = b.text.split("\n")
            parts = [b.text] if len(b.text) <= p.max_chunk_chars else []
            if not parts:                                   # an over-long table: whole rows, header repeated
                head, cur = rows[0], [rows[0]]
                for r in rows[1:]:
                    if len("\n".join(cur + [r])) > p.max_chunk_chars and len(cur) > 1:
                        parts.append("\n".join(cur))
                        cur = [head]
                    cur.append(r)
                parts.append("\n".join(cur))
            for t in parts:
                if acc.fresh and acc.size() + len(t) > p.chunk_chars:
                    close(carry=True)
                acc.pieces.append(_Piece(t, b, sentence=False))
                acc.fresh += 1
                if len(t) > p.chunk_chars:
                    close(carry=False)
            continue
        for s in sentences(b.text):
            for part in (_hard_split(s, p.chunk_chars) if len(s) > p.chunk_chars else [s]):
                if acc.fresh and acc.size() + len(part) > p.chunk_chars:
                    close(carry=True)
                acc.pieces.append(_Piece(part, b))
                acc.fresh += 1
        if len(drafts) > p.max_chunks_per_document:
            break
    close(carry=False)
    return drafts[: p.max_chunks_per_document]
