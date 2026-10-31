"""Phase 8: typed tools, an LLM provider interface with a deterministic FakeLLM, the technician copilot
orchestrator, the hazard gate, the numeric-claim validator and the trusted-confirmation findings store.

Behaviour is verified with a scripted FakeLLM only (no real LLM is wired in Phase 8). The hazard gate and the
numeric validator are deterministic application logic enforced in code, not only in the system prompt.
"""

from .copilot import Copilot, CopilotResponse, Status
from .findings import FindingStore, Outcome
from .hazard import GateDecision, HazardGate, Tier
from .llm import FakeLLM, LLMResponse, ToolCall
from .numbers import Allowed, check, extract
from .tools import CaseContext, ToolResult, ToolStatus, run_tool, tool_schemas

__all__ = ["Allowed", "CaseContext", "Copilot", "CopilotResponse", "FakeLLM", "FindingStore", "GateDecision",
           "HazardGate", "LLMResponse", "Outcome", "Status", "Tier", "ToolCall", "ToolResult", "ToolStatus",
           "check", "extract", "run_tool", "tool_schemas"]
