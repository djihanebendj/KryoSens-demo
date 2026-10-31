"""CaseStore: server-side sessions and the cases they own.

Stage 1-2 ship ``InMemoryCaseStore``: one process, nothing written to disk, everything gone on restart. That is a
deliberate privacy property (uploads are never persisted) and a deployment constraint (one instance; see
docs/PHASE9_ARCHITECTURE.md §5). Another implementation (SQLite, Redis) must keep the same contract:

* The cookie holds only an opaque session id; the store decides whether that session exists, has expired or was
  revoked, and which cases it owns (spec §4).
* ``get`` returns a case only to the session that created it. A case owned by someone else and a case that never
  existed are indistinguishable (both CASE_NOT_FOUND), so case ids cannot be probed.
* A case past its lifetime is deleted; its owner is told CASE_EXPIRED, everyone else CASE_NOT_FOUND.
* Revoking a session deletes its cases at once.
* Limits per session and per process are enforced at creation.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from . import limits as L
from .errors import ApiError, ErrorCode
from .session import new_id


class SessionState(str, Enum):
    VALID = "VALID"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"
    UNKNOWN = "UNKNOWN"       # signed by this server but not in the store (e.g. after a restart)


@dataclass
class SessionRecord:
    session_id: str
    created_at: float
    expires_at: float
    revoked_at: float | None = None


@dataclass
class CaseRecord:
    case_id: str
    owner: str                                   # session id
    created_at: float
    expires_at: float
    filename: str
    evidence: str                                # engine evidence class (UNVALIDATED for anonymous uploads)
    declared_evidence: str | None                # the uploader's unverified claim, stored separately (amendment G)
    source_kind: str                             # ANONYMOUS_UPLOAD or BUNDLED_SAMPLE
    content: bytes
    sha256: str
    options: dict[str, Any]
    results: dict[str, Any] = field(default_factory=dict)   # cached quality / analysis / prediction
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    findings: Any = field(default=None, repr=False)          # Phase 9 findings state; see app/api/findings.py


class CaseStore(Protocol):
    def create_session(self) -> SessionRecord: ...

    def session_state(self, session_id: str) -> SessionState: ...

    def revoke_session(self, session_id: str) -> None: ...

    def extend_session(self, session_id: str, until: float) -> SessionRecord: ...

    def create(self, *, owner: str, filename: str, evidence: str, declared_evidence: str | None, source_kind: str,
               content: bytes, sha256: str, options: dict[str, Any]) -> CaseRecord: ...

    def get(self, case_id: str, owner: str) -> CaseRecord: ...

    def delete(self, case_id: str, owner: str) -> None: ...

    def sweep(self) -> int: ...


def new_case_id() -> str:
    # Letters only, deliberately: the Phase 8 report checker scans the report's "asset <id>" line for numbers and
    # identifiers. An opaque id containing digits, '-' or 'e'-like sequences can be misread as an external number or
    # identifier and refuse an otherwise valid report (docs/PHASE9_LIMITATIONS.md). 32 letters carry 128 random bits.
    return "case_" + "".join("abcdefghijklmnop"[int(c, 16)] for c in secrets.token_hex(16))


class InMemoryCaseStore:
    _TOMBSTONE_S = 24 * 3600     # how long an expired case / ended session is remembered (ids and times only, no data)

    def __init__(self, *, ttl_s: float = L.CASE_TTL_S, session_ttl_s: float = L.SESSION_MAX_AGE_S,
                 per_visitor: int = L.CASES_PER_VISITOR, total: int = L.CASES_TOTAL,
                 sessions_total: int = L.SESSIONS_TOTAL, clock=time.time) -> None:
        self.ttl_s, self.session_ttl_s, self.per_visitor, self.total = ttl_s, session_ttl_s, per_visitor, total
        self.sessions_total, self._clock = sessions_total, clock
        self._cases: dict[str, CaseRecord] = {}
        self._sessions: dict[str, SessionRecord] = {}
        self._expired: dict[str, tuple[str, float]] = {}       # case_id -> (owner, expired_at); holds no data
        self._lock = threading.Lock()

    # ---- housekeeping
    def _sweep_locked(self, now: float) -> int:
        gone = [cid for cid, c in self._cases.items() if c.expires_at <= now]
        for cid in gone:
            c = self._cases.pop(cid)
            self._expired[cid] = (c.owner, c.expires_at)
        for cid in [k for k, (_, at) in self._expired.items() if now - at > self._TOMBSTONE_S]:
            del self._expired[cid]
        for sid in [k for k, s in self._sessions.items()
                    if now - min(s.expires_at, s.revoked_at or s.expires_at) > self._TOMBSTONE_S]:
            del self._sessions[sid]
        return len(gone)

    def sweep(self) -> int:
        with self._lock:
            return self._sweep_locked(self._clock())

    def snapshot(self) -> tuple:
        """Everything a request could change, for tests that check read-only methods change nothing."""
        with self._lock:
            return (tuple(sorted((k, v.expires_at, v.revoked_at) for k, v in self._sessions.items())),
                    tuple(sorted((k, v.expires_at, tuple(sorted(v.results))) for k, v in self._cases.items())))

    # ---- sessions
    def create_session(self) -> SessionRecord:
        with self._lock:
            now = self._clock()
            self._sweep_locked(now)
            if sum(s.revoked_at is None and s.expires_at > now for s in self._sessions.values()) \
                    >= self.sessions_total:
                raise ApiError(503, ErrorCode.CASE_LIMIT_REACHED, "the service is holding its maximum number of "
                               "sessions; try again later", retryable=True)
            rec = SessionRecord(new_id(), now, now + self.session_ttl_s)
            self._sessions[rec.session_id] = rec
            return rec

    def session_state(self, session_id: str) -> SessionState:
        with self._lock:
            s = self._sessions.get(session_id)
            if s is None:
                return SessionState.UNKNOWN
            if s.revoked_at is not None:
                return SessionState.REVOKED
            return SessionState.VALID if self._clock() < s.expires_at else SessionState.EXPIRED

    def extend_session(self, session_id: str, until: float) -> SessionRecord:
        with self._lock:
            s = self._sessions[session_id]
            s.expires_at = max(s.expires_at, until)
            return s

    def revoke_session(self, session_id: str) -> None:
        with self._lock:
            s = self._sessions.get(session_id)
            if s is not None and s.revoked_at is None:
                s.revoked_at = self._clock()
            for cid in [k for k, c in self._cases.items() if c.owner == session_id]:
                del self._cases[cid]                               # revocation deletes the session's data now

    # ---- cases
    def create(self, *, owner: str, filename: str, evidence: str, declared_evidence: str | None, source_kind: str,
               content: bytes, sha256: str, options: dict[str, Any]) -> CaseRecord:
        with self._lock:
            now = self._clock()
            self._sweep_locked(now)
            if sum(c.owner == owner for c in self._cases.values()) >= self.per_visitor:
                raise ApiError(429, ErrorCode.CASE_LIMIT_REACHED,
                               f"at most {self.per_visitor} live cases per visitor; delete one or wait for one to "
                               "expire")
            if len(self._cases) >= self.total:
                raise ApiError(503, ErrorCode.CASE_LIMIT_REACHED, "the service is holding its maximum number of "
                               "cases; try again later", retryable=True)
            cid = new_case_id()
            rec = CaseRecord(case_id=cid, owner=owner, created_at=now, expires_at=now + self.ttl_s,
                             filename=filename, evidence=evidence, declared_evidence=declared_evidence,
                             source_kind=source_kind, content=content, sha256=sha256, options=options)
            self._cases[cid] = rec
            return rec

    def get(self, case_id: str, owner: str) -> CaseRecord:
        with self._lock:
            self._sweep_locked(self._clock())
            rec = self._cases.get(case_id)
            if rec is not None and secrets.compare_digest(rec.owner, owner):
                return rec
            tomb = self._expired.get(case_id)
            if rec is None and tomb is not None and secrets.compare_digest(tomb[0], owner):
                raise ApiError(410, ErrorCode.CASE_EXPIRED, "this case expired and its data was deleted; upload "
                               "the file again to start a new case")
        raise ApiError(404, ErrorCode.CASE_NOT_FOUND, "no such case for this session")

    def delete(self, case_id: str, owner: str) -> None:
        with self._lock:
            rec = self._cases.get(case_id)
            if rec is None or not secrets.compare_digest(rec.owner, owner):
                raise ApiError(404, ErrorCode.CASE_NOT_FOUND, "no such case for this session")
            del self._cases[case_id]

    def __len__(self) -> int:
        with self._lock:
            return len(self._cases)
