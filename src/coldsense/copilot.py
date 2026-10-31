"""Phase 8: Technician Copilot orchestrated by LangGraph + LangChain + Gemini API.

Enforces the 'Rules That Never Bend':
1. Server-side Feature Gate: Copilot is switched off by default unless explicitly enabled;
   when off, returns FEATURE_DISABLED.
2. Hazard Gate: Requests touching protection devices, bypassing safeties, or hazardous electrical /
   pressurized work return a fixed template answer with ZERO language-model or tool calls.
   A broken or empty hazard gate configuration fails closed (CONFIG_ERROR).
   Includes English and French hazard terms plus an unsupported-script gate.
3. Verbatim Quote + Semantic Support: Quotes must be exact substrings of source documents and
   semantically support answering the question.
4. Numeric Integrity: Every number in an answer must come from an allowed set of source values
   (retrieved manual chunks + active case facts). Allows at most 1 regeneration on mismatch via
   the LangGraph conditional loop, then fails closed (NUMERIC_INTEGRITY_FAILURE).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from coldsense.knowledge_base import (
    KnowledgeDocument,
    LocalKnowledgeBase,
    extract_numbers,
)
from coldsense.models import (
    CaseRecord,
    CopilotAnswer,
    CopilotCitation,
    CopilotRefusal,
    EvidenceClass,
    PredictionSuccess,
    RefusalReasonCode,
)

FIXED_HAZARD_TEMPLATE = (
    "SAFETY & HAZARD GATE NOTICE: ColdSense cannot provide instructions for bypassing, bridging, "
    "modifying, or working on live electrical protection devices, pressostats, safety relays, or "
    "pressurized refrigerant containment components. Isolate power at the locked main disconnect "
    "and consult a certified refrigeration commissioning engineer and official OEM safety procedures."
)

DEFAULT_HAZARD_CONFIG: Dict[str, Tuple[str, ...]] = {
    "protection_devices_and_bypasses_en": (
        "bypass",
        "bridge",
        "jumper",
        "shunt",
        "short circuit",
        "disable safety",
        "override protection",
        "defeat interlock",
        "high voltage",
        "live wire",
        "live terminal",
        "contactor coil",
        "thermal overload",
        "safety valve",
        "relief valve",
        "pressostat bypass",
        "bypass pressostat",
        "vent refrigerant",
        "cut capillary",
        "torch on pressurized",
    ),
    "protection_devices_and_bypasses_fr": (
        "shunter",
        "court-circuiter",
        "pontage",
        "ponter",
        "contournement",
        "contourner",
        "desactiver la securite",
        "désactiver la sécurité",
        "relais thermique",
        "soupape de securite",
        "soupape de sécurité",
        "haute tension",
        "sous tension",
        "pressostat",
        "disjoncteur moteur",
        "dégazage",
    ),
    "strict_protection_device_nouns": (
        "overload relay",
        "safety interlock",
        "high pressure switch bypass",
        "low pressure switch bypass",
        "fuse jumper",
    ),
}

_ALLOWED_SCRIPT_RE = re.compile(
    r"^[\x20-\x7E\u00C0-\u017F\u2018-\u201F\s]+$"
)


def check_hazard_gate(
    query: str,
    hazard_config: Optional[Dict[str, Tuple[str, ...]]] = DEFAULT_HAZARD_CONFIG,
) -> Optional[CopilotRefusal]:
    """Run the deterministic Hazard Gate before any retrieval, tool call, or model execution."""
    if not hazard_config or not isinstance(hazard_config, dict):
        return CopilotRefusal(
            response_type="REFUSAL",
            query=query,
            reason_code=RefusalReasonCode.CONFIG_ERROR,
            message="Hazard Gate configuration error: failing closed (CONFIG_ERROR).",
            hazard_matched_terms=[],
            evidence_class=EvidenceClass.EXPECTED,
        )

    en_terms = hazard_config.get("protection_devices_and_bypasses_en", ())
    fr_terms = hazard_config.get("protection_devices_and_bypasses_fr", ())
    strict_nouns = hazard_config.get("strict_protection_device_nouns", ())
    if not en_terms or not fr_terms:
        return CopilotRefusal(
            response_type="REFUSAL",
            query=query,
            reason_code=RefusalReasonCode.CONFIG_ERROR,
            message="Hazard Gate configuration incomplete: failing closed (CONFIG_ERROR).",
            hazard_matched_terms=[],
            evidence_class=EvidenceClass.EXPECTED,
        )

    if not query or not query.strip():
        return CopilotRefusal(
            response_type="REFUSAL",
            query=query,
            reason_code=RefusalReasonCode.INSUFFICIENT_RETRIEVAL_EVIDENCE,
            message="Query cannot be empty.",
            hazard_matched_terms=[],
            evidence_class=EvidenceClass.EXPECTED,
        )

    if not _ALLOWED_SCRIPT_RE.match(query):
        return CopilotRefusal(
            response_type="REFUSAL",
            query=query,
            reason_code=RefusalReasonCode.UNSUPPORTED_SCRIPT_OR_LANGUAGE,
            message=(
                "Unsupported script detected: the Hazard Gate only accepts verified Latin/French script "
                "queries and fails closed on unverified scripts."
            ),
            hazard_matched_terms=["NON_LATIN_SCRIPT"],
            evidence_class=EvidenceClass.EXPECTED,
        )

    q_lower = query.lower()
    matched: List[str] = []
    for term in (*en_terms, *fr_terms, *strict_nouns):
        t_low = term.lower()
        if t_low == "pressostat":
            has_safe_alarm_query = ("e04" in q_lower or "e05" in q_lower or "alarm" in q_lower)
            has_risky_verb = any(
                v in q_lower
                for v in ("bypass", "shunt", "bridge", "jumper", "disable", "adjust", "reset", "wire", "ponter", "shunter")
            )
            if has_safe_alarm_query and not has_risky_verb:
                continue
        if t_low in q_lower:
            matched.append(term)

    if matched:
        return CopilotRefusal(
            response_type="HAZARD_TEMPLATE",
            query=query,
            reason_code=RefusalReasonCode.HAZARD_GATE_BLOCKED,
            message=FIXED_HAZARD_TEMPLATE,
            hazard_matched_terms=matched,
            evidence_class=EvidenceClass.EXPECTED,
        )

    return None


def collect_allowed_numbers(
    retrieved_docs: List[KnowledgeDocument],
    case: Optional[CaseRecord] = None,
) -> Set[float]:
    """Build the deterministic allowed set of numeric values from source documents and case telemetry."""
    allowed: Set[float] = set()
    for doc in retrieved_docs:
        for num in extract_numbers(f"{doc.section} {doc.content}"):
            allowed.add(round(num, 4))

    if case is not None:
        allowed.add(round(case.critical_threshold_c, 4))
        if case.data_quality is not None:
            allowed.add(round(float(case.data_quality.valid_rows), 4))
            allowed.add(round(float(case.data_quality.median_cadence_sec), 4))
            allowed.add(round(float(case.data_quality.gap_count), 4))
        for ev in case.events:
            allowed.add(round(ev.start_temp_c, 4))
            allowed.add(round(ev.end_temp_c, 4))
            allowed.add(round(ev.peak_temp_c, 4))
            allowed.add(round(ev.duration_sec, 4))
            allowed.add(round(round(ev.duration_sec / 60.0, 1), 4))
        if isinstance(case.prediction, PredictionSuccess):
            p = case.prediction
            allowed.add(round(p.initial_temp_c, 4))
            allowed.add(round(p.ambient_temp_c, 4))
            allowed.add(round(p.time_constant_tau_hours, 4))
            allowed.add(round(p.time_constant_tau_sec, 4))
            allowed.add(round(p.r_squared, 4))
            allowed.add(round(p.thermal_model_rmse_c, 4))
            allowed.add(round(p.naive_baseline_rmse_c, 4))
    return allowed


def audit_numeric_integrity(
    answer_text: str,
    allowed_numbers: Set[float],
    tolerance: float = 1e-3,
) -> Tuple[bool, List[float], List[float]]:
    """Verify that EVERY number appearing in `answer_text` belongs to `allowed_numbers`."""
    used = extract_numbers(answer_text)
    valid_used: List[float] = []
    unauthorized: List[float] = []

    for val in used:
        if any(abs(val - a) <= tolerance for a in allowed_numbers):
            valid_used.append(val)
        else:
            unauthorized.append(val)

    return len(unauthorized) == 0, valid_used, unauthorized


def _extractive_v1_synthesizer(
    query: str,
    citations: List[CopilotCitation],
    case: Optional[CaseRecord],
    regeneration_attempt: int = 0,
) -> str:
    """Deterministic extractive answerer (D-059 extractive-v1) grounded strictly in verbatim quotes."""
    parts: List[str] = []
    for cit in citations:
        parts.append(
            f'According to {cit.doc_title} ({cit.section}): "{cit.verbatim_quote}"'
        )
    if case is not None and isinstance(case.prediction, PredictionSuccess):
        pred = case.prediction
        if any(w in query.lower() for w in ("tau", "time constant", "drift", "forecast", "case")):
            parts.append(
                f"In active case ({case.evidence_class.value}), estimated thermal time constant tau is "
                f"{pred.time_constant_tau_hours} hours ({pred.time_constant_tau_sec} s) with critical "
                f"threshold {pred.critical_threshold_c} C."
            )
    return " ".join(parts)


class TechnicianCopilot:
    """Step 7 Guarded Copilot powered by LangGraph `StateGraph`, LangChain `ColdSenseHybridRetriever`,
    and Google Gemini (`GEMINI_API_KEY`).
    """

    def __init__(
        self,
        knowledge_base: Optional[LocalKnowledgeBase] = None,
        enabled: bool = False,
        hazard_config: Optional[Dict[str, Tuple[str, ...]]] = DEFAULT_HAZARD_CONFIG,
        custom_generator: Optional[
            Callable[[str, List[CopilotCitation], Optional[CaseRecord], int], str]
        ] = None,
        allow_network_gemini: bool = False,
        gemini_api_key: Optional[str] = None,
    ) -> None:
        from coldsense.langgraph_copilot import GeminiCopilotLLM, build_copilot_langgraph

        self.kb = knowledge_base or LocalKnowledgeBase(
            allow_network_embeddings=allow_network_gemini
        )
        self.enabled = enabled
        self.hazard_config = hazard_config
        self.generator = custom_generator
        self.gemini_llm = GeminiCopilotLLM(
            api_key=gemini_api_key,
            allow_network=allow_network_gemini,
        )
        self.graph: Any = build_copilot_langgraph(
            knowledge_base=self.kb,
            gemini_llm=self.gemini_llm,
            custom_generator=custom_generator,
        )

    def ask(
        self,
        query: str,
        case: Optional[CaseRecord] = None,
    ) -> Union[CopilotAnswer, CopilotRefusal]:
        """Execute the 6-node LangGraph Copilot state machine."""
        final_state = self.graph.invoke(
            {
                "query": query,
                "case": case,
                "copilot_enabled": self.enabled,
                "hazard_config": self.hazard_config,
            }
        )
        outcome = final_state.get("final_outcome")
        if outcome is None:
            return CopilotRefusal(
                response_type="REFUSAL",
                query=query,
                reason_code=RefusalReasonCode.INSUFFICIENT_RETRIEVAL_EVIDENCE,
                message="LangGraph execution completed without a valid grounded outcome.",
                evidence_class=EvidenceClass.EXPECTED,
            )
        return outcome
