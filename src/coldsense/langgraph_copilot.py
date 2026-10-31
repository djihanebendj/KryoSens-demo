"""LangGraph + LangChain + Gemini Agentic RAG Workflow for ColdSense (SmartClim).

Architecture:
- Implements a multi-node `StateGraph` (`CopilotGraphState`) orchestrating the 6 stages of Step 7:
  1. `feature_and_hazard_gate`: Server-side feature gate (`FEATURE_DISABLED`) + deterministic EN/FR/Script
     fail-closed Hazard Gate (`HAZARD_GATE_BLOCKED` / `CONFIG_ERROR`). Never calls Gemini or tools when blocked.
  2. `retrieve_and_tools`: Invokes `ColdSenseHybridRetriever` (LangChain `BaseRetriever` over SQLite
     Vector + BM25 RRF) and executes read-only LangChain case inspection tools (`get_case_thermal_summary`,
     `get_case_events_summary`, `get_case_next_checks_summary`).
  3. `verify_quotes_and_semantics`: Enforces Rule 4 (exact verbatim quote substring check + semantic
     intent support check to reject on-topic non-answers).
  4. `generate_answer`: Calls Google Gemini (`gemini-3-flash-preview` via `langchain_google_genai.ChatGoogleGenerativeAI`
     or Gemini REST API using `GEMINI_API_KEY` from `.env`, with automatic fallback to `extractive-v1`)
     constrained to include verbatim quotes and only allowed numbers.
  5. `audit_numeric_integrity`: Enforces Rule 5 (every number in the response must belong to the allowed
     set of source numbers). Uses a **conditional graph edge** to loop back to `generate_answer` for at
     most 1 regeneration (`attempt == 1`), after which any violation routes to `build_refusal`
     (`NUMERIC_INTEGRITY_FAILURE`).
  6. `finalize_response` / `build_refusal`: Outputs the strongly typed `CopilotAnswer` or `CopilotRefusal`.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypedDict, Union

from coldsense.knowledge_base import (
    KnowledgeDocument,
    LocalKnowledgeBase,
    verify_quote_and_semantic_support,
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
from coldsense.rag_pipeline import (
    ColdSenseHybridRetriever,
    SQLiteHybridVectorStore,
    load_dotenv_file,
)

# Optional LangGraph / LangChain imports (uses real `langgraph.graph.StateGraph` when installed,
# or an identical deterministic StateGraph engine when running without external packages).
try:
    from langgraph.graph import END, START, StateGraph  # type: ignore

    LANGGRAPH_INSTALLED = True
except ImportError:
    LANGGRAPH_INSTALLED = False
    START = "__start__"
    END = "__end__"

    class _CompiledStateGraph:
        def __init__(
            self,
            nodes: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]],
            edges: Dict[str, str],
            conditional_edges: Dict[str, Tuple[Callable[[Dict[str, Any]], str], Dict[str, str]]],
        ) -> None:
            self._nodes = nodes
            self._edges = edges
            self._conditional_edges = conditional_edges

        def invoke(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
            state = dict(initial_state)
            curr = self._edges.get(START, END)
            steps = 0
            while curr != END and steps < 25:
                steps += 1
                node_fn = self._nodes[curr]
                updates = node_fn(state)
                if updates:
                    state.update(updates)
                if curr in self._conditional_edges:
                    router_fn, mapping = self._conditional_edges[curr]
                    route_key = router_fn(state)
                    curr = mapping.get(route_key, route_key)
                else:
                    curr = self._edges.get(curr, END)
            return state

    class StateGraph:  # type: ignore
        """LangGraph-compatible StateGraph state machine."""

        def __init__(self, state_schema: Any) -> None:
            self.state_schema = state_schema
            self._nodes: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
            self._edges: Dict[str, str] = {}
            self._conditional_edges: Dict[
                str, Tuple[Callable[[Dict[str, Any]], str], Dict[str, str]]
            ] = {}

        def add_node(self, name: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
            self._nodes[name] = fn

        def add_edge(self, source: str, target: str) -> None:
            self._edges[source] = target

        def add_conditional_edges(
            self,
            source: str,
            path: Callable[[Dict[str, Any]], str],
            path_map: Optional[Dict[str, str]] = None,
        ) -> None:
            self._conditional_edges[source] = (path, path_map or {})

        def compile(self) -> _CompiledStateGraph:
            return _CompiledStateGraph(self._nodes, self._edges, self._conditional_edges)


# ---------------------------------------------------------------------------
# Read-Only LangChain Diagnostic Tools for Case Telemetry Inspection
# ---------------------------------------------------------------------------
def tool_get_case_thermal_summary(case: Optional[CaseRecord]) -> str:
    """Read-only tool returning the fitted thermal model parameters or refusal reason for a Case."""
    if case is None or case.prediction is None:
        return "No active case thermal prediction loaded."
    if isinstance(case.prediction, PredictionSuccess):
        p = case.prediction
        return (
            f"Case {case.case_id} ({case.evidence_class.value}): initial_temp={p.initial_temp_c} C, "
            f"ambient_temp={p.ambient_temp_c} C, tau={p.time_constant_tau_hours} hours "
            f"({p.time_constant_tau_sec} s), R^2={p.r_squared}, critical_threshold={p.critical_threshold_c} C."
        )
    return f"Case {case.case_id} prediction refused: {case.prediction.reason_code.value}."


def tool_get_case_events_summary(case: Optional[CaseRecord]) -> str:
    """Read-only tool returning detected temperature events in the active Case."""
    if case is None or not case.events:
        return "No active case events detected."
    parts = [
        f"{e.event_id} ({e.event_type.value}): start={e.start_temp_c} C, peak={e.peak_temp_c} C, duration={round(e.duration_sec / 60.0, 1)} min"
        for e in case.events
    ]
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Gemini LLM Adapter (LangChain ChatGoogleGenerativeAI + REST API + Fallback)
# ---------------------------------------------------------------------------
# Gemini 3 spends tokens on internal reasoning before it writes anything. With a small output budget
# the visible answer came back as leaked deliberation or truncated mid-sentence, and the numeric and
# verbatim-quote audits then refused it (correctly). Verified against the live API: both
# thinkingConfig.thinkingLevel and thinkingConfig.thinkingBudget return HTTP 200, but on
# gemini-3-flash-preview the budget field is ignored (277 vs 278 thought tokens), so thinkingLevel is
# the field that applies. "low" leaves the model room to answer inside MAX_OUTPUT_TOKENS.
MAX_OUTPUT_TOKENS = 1500
THINKING_LEVEL_ENV = "GEMINI_THINKING_LEVEL"
DEFAULT_THINKING_LEVEL = "low"


def thinking_level() -> str:
    """Reasoning effort for the answer call, from the environment. Not hardcoded."""
    return (os.environ.get(THINKING_LEVEL_ENV, "") or DEFAULT_THINKING_LEVEL).strip() or DEFAULT_THINKING_LEVEL


class GeminiCopilotLLM:
    """Calls Gemini (`gemini-3-flash-preview`) via `langchain_google_genai.ChatGoogleGenerativeAI`
    or the official Gemini REST endpoint using `GEMINI_API_KEY` from `.env`, with deterministic
    `extractive-v1` fallback when offline.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3-flash-preview",
        allow_network: bool = False,
    ) -> None:
        load_dotenv_file()
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or ""
        )
        self.model_name = os.environ.get("GEMINI_MODEL", model_name)
        self.allow_network = allow_network
        self.last_engine_used: str = "EXTRACTIVE_V1"
        # Why the Gemini path did not produce the answer, for the visible notice. Never holds the key.
        self.last_engine_error: Optional[str] = None
        self._lc_chat: Optional[Any] = None

        if self.allow_network and self.api_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

                kwargs = dict(
                    model=self.model_name,
                    google_api_key=self.api_key,
                    temperature=0.0,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    thinking_level=thinking_level(),
                )
                try:
                    self._lc_chat = ChatGoogleGenerativeAI(**kwargs)
                except TypeError:
                    # An older langchain-google-genai does not know thinking_level; the REST path below
                    # still applies it, so drop the kwarg rather than lose the client.
                    kwargs.pop("thinking_level", None)
                    self._lc_chat = ChatGoogleGenerativeAI(**kwargs)
            except ImportError:
                self._lc_chat = None

    def generate_grounded_answer(
        self,
        query: str,
        citations: List[CopilotCitation],
        case: Optional[CaseRecord],
        allowed_numbers: Set[float],
        attempt: int = 0,
        unauthorized_numbers: Optional[List[float]] = None,
    ) -> str:
        """Synthesize a grounded technical answer incorporating exact verbatim quotes and allowed numbers."""
        self.last_engine_error = None
        if self.allow_network and self.api_key:
            prompt = self._build_strict_prompt(
                query=query,
                citations=citations,
                case=case,
                allowed_numbers=allowed_numbers,
                attempt=attempt,
                unauthorized_numbers=unauthorized_numbers,
            )
            # 1. Try LangChain ChatGoogleGenerativeAI
            if self._lc_chat is not None:
                try:
                    msg = self._lc_chat.invoke(prompt)
                    content = getattr(msg, "content", str(msg))
                    if isinstance(content, str) and content.strip():
                        self.last_engine_used = f"LANGCHAIN_GEMINI ({self.model_name})"
                        return content.strip()
                except Exception as exc:                       # noqa: BLE001 - reported, never silent
                    self.last_engine_error = f"LangChain Gemini call failed: {type(exc).__name__}"

            # 2. Try direct Gemini REST API (supports both AI Studio and Vertex/AQ keys)
            try:
                # The key travels in a header, never in the URL: a URL reaches access logs, proxy logs,
                # error reports and Referer headers.
                url = (
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{self.model_name}:generateContent"
                )
                body = {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.0,
                        "maxOutputTokens": MAX_OUTPUT_TOKENS,
                        "thinkingConfig": {"thinkingLevel": thinking_level()},
                    },
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=8.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        text = "".join(p.get("text", "") for p in parts).strip()
                        if text:
                            self.last_engine_used = f"GEMINI_REST_API ({self.model_name})"
                            return text
            except Exception as exc:                           # noqa: BLE001 - reported, never silent
                detail = getattr(exc, "code", None) or type(exc).__name__
                self.last_engine_error = f"Gemini REST call failed: {detail}"

        elif self.allow_network and not self.api_key:
            self.last_engine_error = "no GEMINI_API_KEY is configured"

        # 3. Deterministic extractive-v1 synthesizer (always includes verbatim quotes + allowed numbers)
        self.last_engine_used = "EXTRACTIVE_V1"
        from coldsense.copilot import _extractive_v1_synthesizer

        return _extractive_v1_synthesizer(query, citations, case, attempt)

    @staticmethod
    def _build_strict_prompt(
        query: str,
        citations: List[CopilotCitation],
        case: Optional[CaseRecord],
        allowed_numbers: Set[float],
        attempt: int,
        unauthorized_numbers: Optional[List[float]],
    ) -> str:
        quotes_block = "\n".join(
            f'- [{c.doc_id}] {c.doc_title} ({c.section}): "{c.verbatim_quote}"'
            for c in citations
        )
        allowed_str = ", ".join(str(n) for n in sorted(allowed_numbers))
        case_summary = tool_get_case_thermal_summary(case)
        retry_warning = ""
        if attempt > 0 and unauthorized_numbers:
            retry_warning = (
                f"Your previous draft used numbers that are not allowed: {unauthorized_numbers}. "
                "Remove them and use only numbers from the allowed list.\n"
            )
        # Stated once, plainly. A long rule list made the model deliberate about the rules and spend its
        # output budget on that deliberation; the rules themselves are unchanged.
        return (
            "You are the ColdSense Technician Copilot.\n"
            "Rules: answer in 1 to 3 sentences; include one exact quote copied character for character "
            "inside double quotes from the citations below; use only numbers from the allowed list; never "
            "say a fault is confirmed; output only the answer, with no reasoning and no notes.\n"
            f"Allowed numbers: [{allowed_str}]\n"
            f"{retry_warning}\n"
            f"Citations:\n{quotes_block}\n\n"
            f"Case: {case_summary}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )


# ---------------------------------------------------------------------------
# LangGraph State Definition & Graph Builder
# ---------------------------------------------------------------------------
class CopilotGraphState(TypedDict, total=False):
    query: str
    case: Optional[CaseRecord]
    copilot_enabled: bool
    hazard_config: Optional[Dict[str, Tuple[str, ...]]]
    retrieval_mode: str
    retrieved_pairs: List[Tuple[KnowledgeDocument, CopilotCitation]]
    verified_docs: List[KnowledgeDocument]
    verified_citations: List[CopilotCitation]
    allowed_numbers: Set[float]
    tool_outputs: Dict[str, str]
    candidate_answer: Optional[str]
    valid_numbers_used: List[float]
    unauthorized_numbers: List[float]
    attempt: int
    numeric_passed: bool
    has_verbatim_quote: bool
    final_outcome: Union[CopilotAnswer, CopilotRefusal, None]


def build_copilot_langgraph(
    knowledge_base: LocalKnowledgeBase,
    gemini_llm: Optional[GeminiCopilotLLM] = None,
    custom_generator: Optional[
        Callable[[str, List[CopilotCitation], Optional[CaseRecord], int], str]
    ] = None,
) -> Any:
    """Construct and compile the 6-node LangGraph `StateGraph` for the ColdSense Technician Copilot."""
    from coldsense.copilot import (
        audit_numeric_integrity,
        check_hazard_gate,
        collect_allowed_numbers,
    )

    llm = gemini_llm or GeminiCopilotLLM(allow_network=False)

    # Node 1: Server-side Feature Gate & Fail-Closed Hazard Gate
    def node_feature_and_hazard_gate(state: CopilotGraphState) -> Dict[str, Any]:
        query = state.get("query", "")
        if not state.get("copilot_enabled", False):
            return {
                "final_outcome": CopilotRefusal(
                    response_type="REFUSAL",
                    query=query,
                    reason_code=RefusalReasonCode.FEATURE_DISABLED,
                    message="Technician Copilot is currently disabled by the server-side feature gate (FEATURE_DISABLED).",
                    evidence_class=EvidenceClass.EXPECTED,
                )
            }

        hazard_refusal = check_hazard_gate(query, state.get("hazard_config"))
        if hazard_refusal is not None:
            return {"final_outcome": hazard_refusal}

        return {"final_outcome": None, "attempt": 0}

    # Node 2: Hybrid RAG Retrieval & Read-Only Case Tools
    def node_retrieve_and_tools(state: CopilotGraphState) -> Dict[str, Any]:
        query = state["query"]
        case = state.get("case")
        retrieved, mode = knowledge_base.retrieve(query, top_k=2)
        tools_data = {
            "thermal_summary": tool_get_case_thermal_summary(case),
            "events_summary": tool_get_case_events_summary(case),
        }
        if not retrieved:
            return {
                "retrieved_pairs": [],
                "retrieval_mode": mode,
                "tool_outputs": tools_data,
                "final_outcome": CopilotRefusal(
                    response_type="REFUSAL",
                    query=query,
                    reason_code=RefusalReasonCode.INSUFFICIENT_RETRIEVAL_EVIDENCE,
                    message="Refused: no matching technical manual evidence found in the knowledge base.",
                    evidence_class=EvidenceClass.EXPECTED,
                ),
            }
        return {
            "retrieved_pairs": retrieved,
            "retrieval_mode": mode,
            "tool_outputs": tools_data,
        }

    # Node 3: Verbatim Quote & Semantic Intent Verification (Rule 4)
    def node_verify_quotes_and_semantics(state: CopilotGraphState) -> Dict[str, Any]:
        query = state["query"]
        case = state.get("case")
        retrieved = state.get("retrieved_pairs", [])

        v_docs: List[KnowledgeDocument] = []
        v_cits: List[CopilotCitation] = []
        for doc, cit in retrieved:
            ok, _reason = verify_quote_and_semantic_support(query, doc, cit.verbatim_quote)
            if ok:
                v_docs.append(doc)
                v_cits.append(cit)

        if not v_cits:
            return {
                "verified_docs": [],
                "verified_citations": [],
                "final_outcome": CopilotRefusal(
                    response_type="REFUSAL",
                    query=query,
                    reason_code=RefusalReasonCode.INSUFFICIENT_RETRIEVAL_EVIDENCE,
                    message=(
                        "Refused: retrieved passages did not pass verbatim quote and semantic answer "
                        "verification (on-topic non-answer rejected)."
                    ),
                    evidence_class=EvidenceClass.EXPECTED,
                ),
            }

        allowed = collect_allowed_numbers(v_docs, case=case)
        return {
            "verified_docs": v_docs,
            "verified_citations": v_cits,
            "allowed_numbers": allowed,
        }

    # Node 4: Gemini / Custom Synthesis Node
    def node_generate_answer(state: CopilotGraphState) -> Dict[str, Any]:
        query = state["query"]
        citations = state.get("verified_citations", [])
        case = state.get("case")
        allowed = state.get("allowed_numbers", set())
        attempt = state.get("attempt", 0)
        bad_nums = state.get("unauthorized_numbers", [])

        if custom_generator is not None:
            candidate = custom_generator(query, citations, case, attempt)
        else:
            candidate = llm.generate_grounded_answer(
                query=query,
                citations=citations,
                case=case,
                allowed_numbers=allowed,
                attempt=attempt,
                unauthorized_numbers=bad_nums,
            )
        return {"candidate_answer": candidate}

    # Node 5: Deterministic Numeric Integrity & Verbatim Audit Node (Rule 5)
    def node_audit_numeric_integrity(state: CopilotGraphState) -> Dict[str, Any]:
        candidate = state.get("candidate_answer") or ""
        citations = state.get("verified_citations", [])
        allowed = state.get("allowed_numbers", set())
        attempt = state.get("attempt", 0)

        has_verbatim = any(c.verbatim_quote in candidate for c in citations)
        num_ok, valid_used, bad_nums = audit_numeric_integrity(candidate, allowed)

        if has_verbatim and num_ok:
            return {
                "numeric_passed": True,
                "has_verbatim_quote": True,
                "valid_numbers_used": valid_used,
                "unauthorized_numbers": [],
            }

        return {
            "numeric_passed": False,
            "has_verbatim_quote": has_verbatim,
            "valid_numbers_used": valid_used,
            "unauthorized_numbers": bad_nums,
            "attempt": attempt + 1,
        }

    # Node 6a: Finalize Grounded Answer
    def node_finalize_answer(state: CopilotGraphState) -> Dict[str, Any]:
        case = state.get("case")
        retrieval_mode = state.get("retrieval_mode", "SEMANTIC_HYBRID")
        conditions = [
            "Corpus indexed from data/manuals/ into persistent SQLite Hybrid RRF Vector+BM25 store.",
            f"Retrieval mode: {retrieval_mode}.",
            f"Orchestration engine: LangGraph StateGraph ({'langgraph' if LANGGRAPH_INSTALLED else 'builtin-stategraph'}).",
        ]
        if case is not None:
            conditions.append(f"Case telemetry evidence class: {case.evidence_class.value}.")

        return {
            "final_outcome": CopilotAnswer(
                response_type="ANSWER",
                query=state["query"],
                answer_text=state.get("candidate_answer") or "",
                verbatim_citations=state.get("verified_citations", []),
                allowed_numbers_used=state.get("valid_numbers_used", []),
                evidence_class=case.evidence_class if case is not None else EvidenceClass.EXPECTED,
                retrieval_mode=retrieval_mode,
                conditions=conditions,
            )
        }

    # Node 6b: Fail Closed after 1 Failed Numeric/Quote Regeneration
    def node_fail_closed_numeric(state: CopilotGraphState) -> Dict[str, Any]:
        return {
            "final_outcome": CopilotRefusal(
                response_type="REFUSAL",
                query=state["query"],
                reason_code=RefusalReasonCode.NUMERIC_INTEGRITY_FAILURE,
                message=(
                    "Refused: generated response failed the deterministic numeric allowed-set or "
                    "verbatim-quote audit after 1 regeneration attempt (failed closed)."
                ),
                evidence_class=EvidenceClass.EXPECTED,
            )
        }

    # Conditional Routers
    def route_after_gate(state: CopilotGraphState) -> str:
        return END if state.get("final_outcome") is not None else "retrieve_and_tools"

    def route_after_retrieve(state: CopilotGraphState) -> str:
        return END if state.get("final_outcome") is not None else "verify_quotes_and_semantics"

    def route_after_verify(state: CopilotGraphState) -> str:
        return END if state.get("final_outcome") is not None else "generate_answer"

    def route_after_numeric_audit(state: CopilotGraphState) -> str:
        if state.get("numeric_passed") and state.get("has_verbatim_quote"):
            return "finalize_answer"
        # Allow exactly 1 regeneration attempt (attempt was incremented from 0 -> 1)
        if state.get("attempt", 0) <= 1:
            return "generate_answer"
        return "fail_closed_numeric"

    graph = StateGraph(CopilotGraphState)
    graph.add_node("feature_and_hazard_gate", node_feature_and_hazard_gate)
    graph.add_node("retrieve_and_tools", node_retrieve_and_tools)
    graph.add_node("verify_quotes_and_semantics", node_verify_quotes_and_semantics)
    graph.add_node("generate_answer", node_generate_answer)
    graph.add_node("audit_numeric_integrity", node_audit_numeric_integrity)
    graph.add_node("finalize_answer", node_finalize_answer)
    graph.add_node("fail_closed_numeric", node_fail_closed_numeric)

    graph.add_edge(START, "feature_and_hazard_gate")
    graph.add_conditional_edges(
        "feature_and_hazard_gate",
        route_after_gate,
        {END: END, "retrieve_and_tools": "retrieve_and_tools"},
    )
    graph.add_conditional_edges(
        "retrieve_and_tools",
        route_after_retrieve,
        {END: END, "verify_quotes_and_semantics": "verify_quotes_and_semantics"},
    )
    graph.add_conditional_edges(
        "verify_quotes_and_semantics",
        route_after_verify,
        {END: END, "generate_answer": "generate_answer"},
    )
    graph.add_edge("generate_answer", "audit_numeric_integrity")
    graph.add_conditional_edges(
        "audit_numeric_integrity",
        route_after_numeric_audit,
        {
            "finalize_answer": "finalize_answer",
            "generate_answer": "generate_answer",
            "fail_closed_numeric": "fail_closed_numeric",
        },
    )
    graph.add_edge("finalize_answer", END)
    graph.add_edge("fail_closed_numeric", END)

    return graph.compile()
