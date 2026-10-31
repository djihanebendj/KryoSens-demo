"""Technician findings with trusted confirmation (docs/COPILOT.md §9).

The LLM can only create a PENDING draft; a draft is not part of history. A write happens only through a trusted,
application-level channel: the application (never the LLM) renders the exact record from structured data, asks
the user on the CLI, and issues a single-use, expiring token bound to a hash of that record. ``confirm`` writes
only when given a valid token for identical content. An ``confirmed=true`` flag in an LLM tool argument, a
document, or a tool result is NOT consent and never reaches this module as proof.

Records are append-only and kept apart from manuals/RAG: the original statement, its source and timestamp are
preserved, and confirmation adds an audit entry (how and when). Outcome defaults to UNKNOWN and changes only from
a user-entered value; causation is never inferred from timing.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

log = logging.getLogger("coldsense.copilot.findings")
TOKEN_TTL_SECONDS = 300


class Outcome(str, Enum):
    UNKNOWN = "UNKNOWN"
    RESOLVED = "RESOLVED"
    NOT_RESOLVED = "NOT_RESOLVED"
    PENDING = "PENDING"


def _content_hash(asset_id: str, statement: str, event_window: str | None) -> str:
    h = hashlib.sha256()
    for part in (asset_id, statement, event_window or ""):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


@dataclass(frozen=True)
class FindingDraft:
    draft_id: str
    asset_id: str
    statement: str
    event_window: str | None
    content_hash: str
    created_at: datetime


@dataclass(frozen=True)
class FindingRecord:
    record_id: str
    asset_id: str
    statement: str
    event_window: str | None
    content_hash: str
    created_at: datetime
    confirmed_at: datetime
    confirmed_via: str
    outcome: Outcome = Outcome.UNKNOWN
    source: str = "technician via copilot (trusted confirmation)"


class ConfirmationError(Exception):
    """The confirmation token was missing, invalid, expired, replayed, or for different content. No write."""


class FindingStore:
    """Append-only store. In memory by design; it is never the knowledge base and is never retrieved as text."""

    def __init__(self, secret: bytes | None = None) -> None:
        self._secret = secret or secrets.token_bytes(32)
        self._drafts: dict[str, FindingDraft] = {}
        self._records: list[FindingRecord] = []
        self._used_tokens: set[str] = set()
        self._n = 0

    # ---- LLM-reachable: draft only, never a write
    def draft(self, asset_id: str, statement: str, event_window: str | None = None) -> FindingDraft:
        self._n += 1
        d = FindingDraft(draft_id=f"draft-{self._n:04d}", asset_id=asset_id, statement=statement,
                         event_window=event_window,
                         content_hash=_content_hash(asset_id, statement, event_window), created_at=datetime.now(UTC))
        self._drafts[d.draft_id] = d
        log.info("finding draft created draft_id=%s asset=%s hash=%s", d.draft_id, asset_id, d.content_hash[:12])
        return d

    def get_draft(self, draft_id: str) -> FindingDraft | None:
        return self._drafts.get(draft_id)

    # ---- trusted application channel: issue a token, then confirm
    def issue_token(self, draft_id: str, ttl: int = TOKEN_TTL_SECONDS) -> str:
        """Called by the APPLICATION after it has shown the exact record to the user and the user agreed. Binds a
        single-use, expiring token to this draft's content hash. Never call this from inside the LLM loop."""
        d = self._drafts.get(draft_id)
        if d is None:
            raise ConfirmationError(f"no such draft {draft_id}")
        expiry = int(time.time()) + ttl
        nonce = secrets.token_hex(8)
        body = f"{draft_id}.{d.content_hash}.{expiry}.{nonce}"
        sig = hmac.new(self._secret, body.encode(), "sha256").hexdigest()
        return f"{body}.{sig}"

    def confirm(self, draft_id: str, token: str, outcome: Outcome = Outcome.UNKNOWN) -> FindingRecord:
        """Append the record iff the token is valid, unused, unexpired, and for this draft's exact content."""
        d = self._drafts.get(draft_id)
        if d is None:
            raise ConfirmationError(f"no such draft {draft_id}")
        parts = token.split(".")
        if len(parts) != 5:
            raise ConfirmationError("malformed token")
        t_draft, t_hash, t_expiry, t_nonce, t_sig = parts
        body = f"{t_draft}.{t_hash}.{t_expiry}.{t_nonce}"
        expected = hmac.new(self._secret, body.encode(), "sha256").hexdigest()
        if not hmac.compare_digest(expected, t_sig):
            raise ConfirmationError("bad token signature")
        if token in self._used_tokens:
            raise ConfirmationError("token already used (replay)")
        if t_draft != draft_id or t_hash != d.content_hash:
            raise ConfirmationError("token is for different content")
        if int(t_expiry) < int(time.time()):
            raise ConfirmationError("token expired")
        self._used_tokens.add(token)
        rec = FindingRecord(record_id=f"finding-{len(self._records) + 1:04d}", asset_id=d.asset_id,
                            statement=d.statement, event_window=d.event_window, content_hash=d.content_hash,
                            created_at=d.created_at, confirmed_at=datetime.now(UTC),
                            confirmed_via="trusted CLI confirmation token", outcome=outcome)
        self._records.append(rec)
        log.info("finding confirmed record_id=%s draft_id=%s outcome=%s", rec.record_id, draft_id, outcome.value)
        return rec

    def records(self, asset_id: str | None = None) -> list[FindingRecord]:
        return [r for r in self._records if asset_id is None or r.asset_id == asset_id]

    def drafts(self) -> list[FindingDraft]:
        return list(self._drafts.values())
