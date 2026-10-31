"""CaseStore with TTL expiry, RateLimiter, and RedactedLogger for ColdSense."""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from coldsense.models import CaseRecord


class CaseNotFoundError(Exception):
    """Raised when a case ID does not exist, has expired, or belongs to another visitor session."""


class RateLimitExceededError(Exception):
    """Raised when a client exceeds the configured request rate limit."""


class CaseStore:
    """Session-isolated in-memory CaseStore with automatic TTL expiration."""

    def __init__(self, default_ttl_sec: float = 3600.0) -> None:
        self.default_ttl_sec = default_ttl_sec
        self._cases: Dict[str, CaseRecord] = {}

    def purge_expired(self, now_epoch: Optional[float] = None) -> int:
        now = now_epoch if now_epoch is not None else time.time()
        expired_ids = [cid for cid, c in self._cases.items() if now > c.expires_at_epoch]
        for cid in expired_ids:
            del self._cases[cid]
        return len(expired_ids)

    def put(self, case: CaseRecord) -> None:
        self.purge_expired()
        self._cases[case.case_id] = case

    def get(
        self,
        case_id: str,
        owner_session_id: str,
        now_epoch: Optional[float] = None,
    ) -> CaseRecord:
        self.purge_expired(now_epoch=now_epoch)
        case = self._cases.get(case_id)
        if case is None:
            raise CaseNotFoundError(f"Case '{case_id}' not found or has expired.")
        if case.owner_session_id != owner_session_id:
            raise CaseNotFoundError(
                f"Access denied: Case '{case_id}' is private to another visitor session."
            )
        return case

    def list_for_session(
        self,
        owner_session_id: str,
        now_epoch: Optional[float] = None,
    ) -> List[CaseRecord]:
        self.purge_expired(now_epoch=now_epoch)
        return [
            c
            for c in self._cases.values()
            if c.owner_session_id == owner_session_id
        ]


class RateLimiter:
    """Sliding-window rate limiter per session/IP key."""

    def __init__(self, max_requests: int = 60, window_sec: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_sec = window_sec
        self._hits: Dict[str, List[float]] = {}

    def check(self, client_key: str, now_epoch: Optional[float] = None) -> None:
        now = now_epoch if now_epoch is not None else time.time()
        cutoff = now - self.window_sec
        history = [t for t in self._hits.get(client_key, []) if t >= cutoff]
        if len(history) >= self.max_requests:
            raise RateLimitExceededError(
                f"Rate limit exceeded ({self.max_requests} requests per {self.window_sec:.0f}s)."
            )
        history.append(now)
        self._hits[client_key] = history


_SENSITIVE_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"(tct-[a-f0-9]{16,32})", re.IGNORECASE), "[REDACTED_TOKEN]"),
    (re.compile(r"(csrf-[a-f0-9]{16,64})", re.IGNORECASE), "[REDACTED_CSRF]"),
    (re.compile(r"(sess-[a-f0-9]{12,64}(?:\.[a-f0-9]{16,64})?)", re.IGNORECASE), "[REDACTED_SESSION]"),
    (re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)"), "[REDACTED_EMAIL]"),
)


class RedactedLogger:
    """Audit logger that automatically redacts tokens, session cookies, CSRF secrets, and emails."""

    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []

    @staticmethod
    def redact_text(val: str) -> str:
        out = val
        for pat, repl in _SENSITIVE_PATTERNS:
            out = pat.sub(repl, out)
        return out

    def log(self, event: str, details: str, status_code: int = 200) -> Dict[str, Any]:
        entry = {
            "timestamp_epoch": time.time(),
            "event": event,
            "details": self.redact_text(details),
            "status_code": status_code,
        }
        self.entries.append(entry)
        return entry
