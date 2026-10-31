"""Turn a Phase 7 retrieval Answer into copilot evidence, and enforce the verbatim + supported-answer rules
(docs/COPILOT.md §3). A sentence may be shown as a DOCUMENT_FACT only when BOTH hold:

1. Phase 7 gave the answer a supported-answer status (EXTRACTIVE_ANSWER or CONFLICTING_SOURCES), and
2. the quote appears verbatim in the passage it cites.

The verbatim check is necessary but not sufficient: a quote can be word-for-word yet not answer the question
(the E3 / pressure-switch-bypass case). Anything not meeting both is shown only as
RELATED CONTEXT - NOT A CONFIRMED ANSWER, never as a confirmed answer, and PASSAGES_ONLY / insufficient /
conflicting results are never upgraded.
"""

from __future__ import annotations

from dataclasses import dataclass

from coldsense.knowledge.schemas import Answer, AnswerStatus, Citation

from .tools import ToolResult, ToolStatus

# Phase 7 statuses that supply a supported answer Phase 8 can verify (inspected from knowledge/schemas.py).
SUPPORTED = (AnswerStatus.EXTRACTIVE_ANSWER, AnswerStatus.CONFLICTING_SOURCES)

RELATED_LABEL = "RELATED CONTEXT - NOT A CONFIRMED ANSWER"


@dataclass(frozen=True)
class VerifiedQuote:
    text: str
    citation: Citation
    is_document_fact: bool          # True only when supported-answer status AND verbatim
    label: str                      # "DOCUMENT_FACT" or RELATED_LABEL

    def as_related(self) -> VerifiedQuote:
        """Downgrade to related context: used on a gated (refused) path, where nothing retrieved can be a
        confirmed answer to the question even when Phase 7 gave it a supported status (the E3 case)."""
        return VerifiedQuote(self.text, self.citation, False, RELATED_LABEL)

    def attribution(self) -> str:
        c = self.citation
        bits = [c.title]
        if c.revision:
            bits.append(f"rev {c.revision}")
        bits.append(c.location)
        if c.model:
            bits.append(f"model {c.model}")
        bits.append(f"[{c.chunk_id}]")
        return ", ".join(bits)


def _passage_for(answer: Answer, chunk_id: str):
    return next((p for p in answer.passages if p.citation.chunk_id == chunk_id), None)


def verbatim_in_passage(answer: Answer, quote: str, chunk_id: str) -> bool:
    p = _passage_for(answer, chunk_id)
    return p is not None and quote in p.text


def verified_quotes(answer: Answer, model: str | None = None) -> list[VerifiedQuote]:
    """Document facts if the status is supported and each quote is verbatim; otherwise related-context passages.
    When ``model`` is given, drop any citation for a different stated model (equipment-identity filter)."""
    def ok_model(c: Citation) -> bool:
        return model is None or c.model is None or c.model == model

    out: list[VerifiedQuote] = []
    if answer.status in SUPPORTED:
        for claim in answer.claims:
            cit = claim.citations[0]
            if ok_model(cit) and verbatim_in_passage(answer, claim.text, cit.chunk_id):
                out.append(VerifiedQuote(claim.text, cit, True, "DOCUMENT_FACT"))
        if out:
            return out
    # not supported, or no verbatim claim survived: show related context only
    for p in answer.passages:
        if ok_model(p.citation):
            # the full passage text is verbatim by construction (the excerpt uses " … " joins and is not a
            # substring); shown as related context, never as a confirmed answer
            out.append(VerifiedQuote(p.text, p.citation, False, RELATED_LABEL))
    return out


def answer_to_result(answer: Answer, model: str | None = None) -> ToolResult:
    """search_service_manual's structured result. It never asserts an answer: it reports the Phase 7 status and
    the verified quotes with their labels."""
    quotes = verified_quotes(answer, model)
    from .numbers import extract
    from .tools import Quantity
    qs: list[Quantity] = []
    seen = set()
    for vq in quotes:
        for n in extract(vq.text).numbers:
            if n.value is not None and (n.value, n.unit) not in seen:
                seen.add((n.value, n.unit))
                qs.append(Quantity(value=n.value, unit=n.unit, label="document quote"))
    status = ToolStatus.OK if answer.status != AnswerStatus.NO_RELEVANT_DOCUMENTATION else ToolStatus.REFUSED
    return ToolResult(
        tool="search_service_manual", status=status,
        summary=f"Phase 7 status {answer.status.value}; {len(quotes)} verified quote(s)",
        data={"retrieval_status": answer.status.value, "reason": answer.reason,
              "quotes": [{"text": q.text, "label": q.label, "document_fact": q.is_document_fact,
                          "attribution": q.attribution(), "chunk_id": q.citation.chunk_id,
                          "model": q.citation.model} for q in quotes]},
        quantities=qs,
        identifiers=[q.citation.chunk_id for q in quotes] + [q.citation.model for q in quotes if q.citation.model],
        provenance="DOCUMENT")
