"""Integration path for Phase 8 and the operational interface: documentation, ColdSense diagnostics and
operational history side by side, each with its own provenance, never merged.

* documentation: the source-grounded Answer (DOCUMENT_FACT claims quoting cited passages);
* diagnostics: Phase 6 output as MODEL_INFERENCE (hypotheses, forecast context) and SYSTEM_RECOMMENDATION
  (ranked next checks), referenced by their ids; Phase 6 observed facts are listed as observations with their
  evidence class;
* operational history: structured records (event log: maintenance, faults, setpoint changes, ...) exactly as
  supplied, with their provenance. They are never indexed as document text, and no manual passage replaces them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from .schemas import Answer, Claim, ClaimType

if TYPE_CHECKING:                                    # types only: no import-time dependency on the domain
    from coldsense.domains.cold_room.diagnostics import DiagnosticReport


class ObservationRef(BaseModel):
    fact_id: str
    kind: str
    statement: str
    evidence: str = Field(description="evidence class of the underlying data (REAL, SIMULATED, ...)")


class OperationalRecordItem(BaseModel):
    kind: str
    start: str
    end: str | None
    provenance: str = Field(description="RECORDED_BY_SYSTEM or MANUAL_LOG, as supplied")
    source: str | None = None
    note: str | None = None
    label: str = "OPERATIONAL_RECORD: an observed event as recorded; not documentation, not a ColdSense inference"


class CombinedContext(BaseModel):
    documentation: Answer | None = None
    diagnostics: list[Claim] = Field(default_factory=list)
    diagnostic_observations: list[ObservationRef] = Field(default_factory=list)
    operational_history: list[OperationalRecordItem] = Field(default_factory=list)
    separation_note: str = ("documentation, ColdSense diagnostics and operational records are separate sources; "
                            "a manual passage never overrides a recorded event, and a ColdSense recommendation is "
                            "not a documented instruction")


def diagnostic_claims(report: DiagnosticReport) -> tuple[list[Claim], list[ObservationRef]]:
    claims = [Claim(claim_type=ClaimType.MODEL_INFERENCE, text=h.statement, source_ids=[h.id],
                    note=f"PROPOSED hypothesis from ColdSense rules ({h.generated_by}); not a confirmed fault")
              for h in report.hypotheses]
    claims += [Claim(claim_type=ClaimType.MODEL_INFERENCE, text=f.statement, source_ids=[f.fact_id],
                     note=f"ColdSense model output, evidence {f.evidence.value}; context, not documentation")
               for f in report.observed_facts if f.source.value in ("PHASE4_FORECAST", "PHASE5_RELIABILITY")]
    claims += [Claim(claim_type=ClaimType.SYSTEM_RECOMMENDATION, text=c.action, source_ids=[c.check_id],
                     note=f"ColdSense next check, rank {c.rank} (a work priority); not a manufacturer instruction")
               for c in report.next_checks]
    obs = [ObservationRef(fact_id=f.fact_id, kind=f.kind, statement=f.statement, evidence=f.evidence.value)
           for f in report.observed_facts if f.source.value not in ("PHASE4_FORECAST", "PHASE5_RELIABILITY")]
    return claims, obs


def combine(documentation: Answer | None = None, report: DiagnosticReport | None = None,
            operational_events: list | None = None) -> CombinedContext:
    claims, obs = diagnostic_claims(report) if report is not None else ([], [])
    history = [OperationalRecordItem(kind=e.kind.value, start=e.start.isoformat(),
                                     end=e.end.isoformat() if e.end else None, provenance=e.provenance.value,
                                     source=e.source, note=e.note) for e in (operational_events or [])]
    return CombinedContext(documentation=documentation, diagnostics=claims, diagnostic_observations=obs,
                           operational_history=history)
