"""Server-side Copilot lock (spec §6, amendment A).

The copilot is disabled by default and is not a UI toggle. Opening it needs all three of:

1. explicit deployment configuration (``COLDSENSE_API_ENABLE_COPILOT``);
2. an LLM provider configured for the Phase 8 adapter (``ANTHROPIC_API_KEY`` present; its value is never logged);
3. a verified Phase 8 acceptance record at ``COLDSENSE_API_PHASE8_ACCEPTANCE`` proving the Rule 7 hardening passed.

No acceptance record exists in this repository and Phase 9 does not create one. The gate therefore fails closed,
and both health and the copilot endpoint report the first missing condition without constructing a provider or
calling an LLM/tool. The record is read as data only; it can never be supplied by a request.

This Phase 9 slice also does not wire a conversation provider. Even if all three preconditions were present, the
gate keeps the state ``FEATURE_DISABLED`` and says so, so health never advertises a capability that could call an
LLM. No conversation, retrieval or tool call exists in this slice.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .config import PROVIDER_ENV, ResolvedApiConfig
from .schemas import FeatureState

log = logging.getLogger(__name__)

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
RECORD_REQUIRED = ("phase", "status", "rule7_hardening", "commit")


@dataclass(frozen=True)
class Acceptance:
    verified: bool
    reason: str


@dataclass(frozen=True)
class CopilotGate:
    state: FeatureState
    reason: str


def verify_acceptance(path: Path | None) -> Acceptance:
    """Read and validate the acceptance record. Any missing, unreadable or incomplete record is unverified."""
    if path is None:
        return Acceptance(False, "no Phase 8 acceptance record path is configured")
    if not path.is_file():
        return Acceptance(False, "no Phase 8 acceptance record exists at the configured path")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Acceptance(False, "the Phase 8 acceptance record is unreadable")
    if not isinstance(data, dict) or any(k not in data for k in RECORD_REQUIRED):
        return Acceptance(False, "the Phase 8 acceptance record is incomplete")
    if data.get("phase") != 8:
        return Acceptance(False, "the record is not for Phase 8")
    if data.get("status") != "ACCEPTED":
        return Acceptance(False, "the record does not show an ACCEPTED Phase 8")
    if data.get("rule7_hardening") != "VERIFIED":
        return Acceptance(False, "the record does not show the Rule 7 hardening as VERIFIED")
    commit = data.get("commit")
    if not isinstance(commit, str) or not _HEX40.match(commit):
        return Acceptance(False, "the record does not identify the verified commit")
    return Acceptance(True, f"the Phase 8 acceptance record verifies commit {commit}")


def provider_configured() -> bool:
    """Presence only; the key is never read into a response, a log line or a case."""
    return bool(os.environ.get(PROVIDER_ENV, "").strip())


def evaluate(config: ResolvedApiConfig) -> CopilotGate:
    if not config.enable_copilot:
        return CopilotGate(FeatureState.FEATURE_DISABLED, (
            "disabled at the API boundary: it needs explicit deployment configuration AND a verified Phase 8 "
            "acceptance record for the Rule 7 hardening; neither is configured (spec §6, amendment A)"))
    if not provider_configured():
        return CopilotGate(FeatureState.FEATURE_DISABLED, (
            "disabled at the API boundary: no LLM provider is configured; a verified Phase 8 acceptance record is "
            "also required (spec §6, amendment A)"))
    acceptance = verify_acceptance(config.phase8_acceptance_path)
    if not acceptance.verified:
        return CopilotGate(FeatureState.FEATURE_DISABLED, (
            "disabled at the API boundary: it needs explicit deployment configuration AND a verified Phase 8 "
            f"acceptance record for the Rule 7 hardening; {acceptance.reason} (spec §6, amendment A)"))
    # All three preconditions are present, but this Phase 9 slice deliberately does not wire a conversation provider.
    # The state stays disabled so health never advertises a capability that would make an LLM call.
    return CopilotGate(FeatureState.FEATURE_DISABLED, (
        f"all three preconditions are present ({acceptance.reason}), but this Phase 9 slice does not wire a "
        "conversation provider; no LLM or tool call is made (spec §6, amendment A)"))
