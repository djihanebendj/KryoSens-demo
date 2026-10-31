"""LLM provider interface for the copilot (docs/COPILOT.md §2).

The orchestrator talks only to this interface. Every automated test uses ``FakeLLM``, a deterministic, scripted
provider that makes no network call. One real adapter (``AnthropicAdapter``) is provided but DISABLED by default
and only constructed when configuration explicitly enables it; its API key comes from the environment, never from
code, and it is never used in tests.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict


@dataclass(frozen=True)
class LLMResponse:
    """Either a final answer (text) or one tool call. Never both; the orchestrator decides what to do next."""
    text: str | None = None
    tool_call: ToolCall | None = None

    def __post_init__(self) -> None:
        if (self.text is None) == (self.tool_call is None):
            raise ValueError("an LLM response is exactly one of: final text, or a tool call")


@dataclass(frozen=True)
class LLMRequest:
    system: str
    messages: list[dict]                 # [{role, content}]; tool results included as role="tool"
    tools: list[dict]                    # tool schemas offered this turn


class LLMProvider(Protocol):
    name: str

    def complete(self, request: LLMRequest) -> LLMResponse: ...


@dataclass
class FakeLLM:
    """A scripted provider. ``script`` is a list of LLMResponse, returned in order, one per ``complete`` call.
    Alternatively ``responder`` maps a request to a response for context-dependent scripts. Counts calls so a
    test can assert the LLM was never called on a gated path."""
    script: list[LLMResponse] = field(default_factory=list)
    responder: Callable[[LLMRequest], LLMResponse] | None = None
    name: str = "fake-llm"
    calls: int = 0
    seen: list[LLMRequest] = field(default_factory=list)

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        self.seen.append(request)
        if self.responder is not None:
            return self.responder(request)
        if not self.script:
            raise AssertionError("FakeLLM script exhausted: the orchestrator made an unexpected LLM call")
        return self.script.pop(0)


class AnthropicAdapter:
    """DISABLED by default. A thin adapter to the Claude Messages API, constructed only when the application
    explicitly enables a real LLM. The key is read from the environment at call time and never logged. No test
    constructs this. Documented data flow (docs/COPILOT.md §2): when enabled, the system prompt, the
    conversation and tool *results* for the current case are sent to the configured endpoint; logs and documents
    remain untrusted data and the safety gate and validators run on the response exactly as with FakeLLM."""

    name = "anthropic"

    def __init__(self, *, enabled: bool = False, model: str = "", api_key_env: str = "ANTHROPIC_API_KEY") -> None:
        if not enabled:
            raise RuntimeError("the real LLM adapter is disabled; enable it explicitly in configuration to use it")
        self.model = model
        self._api_key_env = api_key_env

    def complete(self, request: LLMRequest) -> LLMResponse:        # pragma: no cover - never run in tests
        key = os.environ.get(self._api_key_env)
        if not key:
            raise RuntimeError(f"{self._api_key_env} is not set; the real LLM adapter cannot run")
        raise NotImplementedError("the real Anthropic adapter is intentionally not wired in Phase 8; "
                                  "Phase 8 is verified with FakeLLM only")
