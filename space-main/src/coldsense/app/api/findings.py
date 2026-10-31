"""Stage 3: technician findings, incident state and service reports.

The Phase 8 modules own every trust rule: ``FindingStore`` creates drafts and writes only when it is given a
single-use, expiring token bound to the exact content hash; ``build_report`` composes text from confirmed records
only and runs the deterministic numeric-claim check. This module is the application channel around them. It adds
the per-case draft state (including cancel), copies the Phase 8 records into API views, and keeps the report
snapshot. It makes no causal or numeric decision.

Cancel is a Stage 3 state change only: a cancelled draft is never passed to ``confirm`` and nothing is appended to
the Phase 8 store. The Phase 8 store is append-only, so its own draft object remains as an excluded draft; the
report builder already excludes every draft whose content hash matches a confirmed record and counts the rest as
drafts, not history.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from coldsense.copilot.findings import (
    ConfirmationError,
    FindingDraft,
    FindingRecord,
    FindingStore,
    Outcome,
)
from coldsense.copilot.report import build_report
from coldsense.copilot.tools import ToolStatus

from . import service as S
from .errors import ApiError, ErrorCode
from .schemas import (
    FindingConfirmRequest,
    FindingCounts,
    FindingDraftCreated,
    FindingDraftRequest,
    FindingDraftView,
    FindingRecordView,
    FindingsView,
    IncidentView,
    ReportView,
)

log = logging.getLogger(__name__)

DraftStatus = Literal["PENDING", "CANCELLED", "CONFIRMED"]


@dataclass
class _DraftMeta:
    status: DraftStatus = "PENDING"
    token: str | None = None
    expires_at: float = 0.0
    record_id: str | None = None


@dataclass
class FindingState:
    """Per-case state. The Phase 8 store is per case, so one case's records can never be read through another."""
    store: FindingStore = field(default_factory=FindingStore)
    drafts: dict[str, _DraftMeta] = field(default_factory=dict)
    report: ReportView | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


def state(rec) -> FindingState:
    st = rec.findings
    if st is None:
        with rec.lock:
            if rec.findings is None:
                rec.findings = FindingState()
            st = rec.findings
    return st


def _draft_view(d: FindingDraft, status: Literal["PENDING", "CANCELLED"]) -> FindingDraftView:
    return FindingDraftView(draft_id=d.draft_id, asset_id=d.asset_id, statement=d.statement,
                            event_window=d.event_window, content_hash=d.content_hash, created_at=d.created_at,
                            status=status)


def _record_view(r: FindingRecord) -> FindingRecordView:
    return FindingRecordView(record_id=r.record_id, asset_id=r.asset_id, statement=r.statement,
                             event_window=r.event_window, content_hash=r.content_hash, created_at=r.created_at,
                             confirmed_at=r.confirmed_at, confirmed_via=r.confirmed_via, outcome=r.outcome,
                             source=r.source, confirmation_channel="WEB_APPLICATION")


def _visible_drafts(st: FindingState, rec) -> list[FindingDraftView]:
    out = []
    for d in st.store.drafts():
        if d.asset_id != rec.case_id:
            continue
        meta = st.drafts.get(d.draft_id)
        status = meta.status if meta is not None else "PENDING"
        if status == "CONFIRMED":
            continue                      # it is in the records now, not a draft
        out.append(_draft_view(d, status))
    return out


def _records(st: FindingState, rec) -> list[FindingRecord]:
    return st.store.records(rec.case_id)


def _counts(st: FindingState, rec) -> FindingCounts:
    records = _records(st, rec)
    drafts = _visible_drafts(st, rec)
    return FindingCounts(
        confirmed=len(records),
        drafts_pending=sum(d.status == "PENDING" for d in drafts),
        drafts_cancelled=sum(d.status == "CANCELLED" for d in drafts),
        outcomes_unknown=sum(r.outcome == Outcome.UNKNOWN for r in records),
        outcomes_verified=sum(r.outcome in (Outcome.RESOLVED, Outcome.NOT_RESOLVED) for r in records),
    )


def _finding_provenance(rec):
    return S.provenance(rec, method="Phase 8 trusted-confirmation finding store (technician-entered statements)")


def create_draft(rec, req: FindingDraftRequest) -> FindingDraftCreated:
    st = state(rec)
    with st.lock:
        d = st.store.draft(rec.case_id, req.statement, req.event_window)
        token = st.store.issue_token(d.draft_id)
        expires = int(token.split(".")[2])       # the token's own expiry, so the view and the store agree
        st.drafts[d.draft_id] = _DraftMeta(status="PENDING", token=token, expires_at=float(expires))
        st.report = None                      # a report built before this finding is not the current report
        log.info("web finding draft created draft_id=%s case=%s", d.draft_id, rec.case_id)
        return FindingDraftCreated(**_draft_view(d, "PENDING").model_dump(),
                                   confirmation_token=token,
                                   token_expires_at=datetime.fromtimestamp(expires, UTC))


def confirm_draft(rec, draft_id: str, body: FindingConfirmRequest) -> FindingRecordView:
    st = state(rec)
    with st.lock:
        d = st.store.get_draft(draft_id)
        if d is None or d.asset_id != rec.case_id:
            raise ApiError(404, ErrorCode.FINDING_DRAFT_NOT_FOUND, "no such finding draft for this case")
        meta = st.drafts.get(draft_id)
        if meta is None:
            raise ApiError(409, ErrorCode.FINDING_NOT_READY, "prepare this draft again before confirming it")
        if meta.status == "CANCELLED":
            raise ApiError(409, ErrorCode.FINDING_CANCELLED,
                           "this draft was cancelled; nothing was recorded and it cannot be confirmed")
        if meta.status == "CONFIRMED":
            if meta.token is not None and body.token == meta.token:
                raise ApiError(409, ErrorCode.FINDING_CONFIRMATION_REPLAY,
                               "this confirmation was already used; nothing was written twice")
            raise ApiError(409, ErrorCode.FINDING_ALREADY_CONFIRMED, "this draft is already confirmed")
        if meta.token is None:
            raise ApiError(409, ErrorCode.FINDING_NOT_READY, "prepare this draft again before confirming it")
        if body.token != meta.token:
            raise ApiError(409, ErrorCode.FINDING_CONFIRMATION_INVALID,
                           "the confirmation token does not match this draft")
        if int(time.time()) > int(meta.expires_at):
            raise ApiError(409, ErrorCode.FINDING_CONFIRMATION_EXPIRED,
                           "the confirmation token expired; prepare the draft again")
        try:
            record = st.store.confirm(draft_id, body.token, Outcome(body.outcome))
        except ConfirmationError as e:
            message = str(e)
            code = (ErrorCode.FINDING_CONFIRMATION_EXPIRED if "expired" in message else
                    ErrorCode.FINDING_CONFIRMATION_REPLAY if "replay" in message else
                    ErrorCode.FINDING_CONFIRMATION_INVALID)
            raise ApiError(409, code, "the confirmation was refused; nothing was recorded") from None
        meta.status, meta.record_id = "CONFIRMED", record.record_id
        st.report = None                      # a report built before this record is stale; regenerate explicitly
        log.info("web finding confirmed record_id=%s case=%s outcome=%s", record.record_id, rec.case_id,
                 record.outcome.value)
        return _record_view(record)


def cancel_draft(rec, draft_id: str) -> FindingDraftView:
    st = state(rec)
    with st.lock:
        d = st.store.get_draft(draft_id)
        if d is None or d.asset_id != rec.case_id:
            raise ApiError(404, ErrorCode.FINDING_DRAFT_NOT_FOUND, "no such finding draft for this case")
        meta = st.drafts.get(draft_id)
        if meta is None:
            raise ApiError(409, ErrorCode.FINDING_NOT_READY, "this draft has no live confirmation state")
        if meta.status == "CONFIRMED":
            raise ApiError(409, ErrorCode.FINDING_ALREADY_CONFIRMED,
                           "this draft is already confirmed; cancelling it would change nothing")
        if meta.status == "CANCELLED":
            return _draft_view(d, "CANCELLED")       # idempotent: the first cancel did not write
        meta.status, meta.token = "CANCELLED", None
        st.report = None                      # the excluded-draft count changed; regenerate explicitly
        log.info("web finding draft cancelled draft_id=%s case=%s (no write)", draft_id, rec.case_id)
        return _draft_view(d, "CANCELLED")


def findings_view(rec) -> FindingsView:
    st = state(rec)
    with st.lock:
        return FindingsView(case_id=rec.case_id, drafts=_visible_drafts(st, rec),
                            records=[_record_view(r) for r in _records(st, rec)],
                            counts=_counts(st, rec), provenance=_finding_provenance(rec))


def incident_view(rec) -> IncidentView:
    st = state(rec)
    with st.lock:
        return IncidentView(case_id=rec.case_id, drafts=_visible_drafts(st, rec),
                            confirmed=[_record_view(r) for r in _records(st, rec)],
                            counts=_counts(st, rec), provenance=_finding_provenance(rec))


@dataclass(frozen=True)
class _ReportContext:
    findings: FindingStore
    asset_id: str
    model: str | None = None


def _report_drafts_excluded(st: FindingState, rec) -> list[FindingDraftView]:
    records = _records(st, rec)
    hashes = {r.content_hash for r in records}
    return [_draft_view(d, (st.drafts[d.draft_id].status if d.draft_id in st.drafts else "PENDING"))
            for d in st.store.drafts() if d.asset_id == rec.case_id and d.content_hash not in hashes]


def build_report_view(rec) -> ReportView:
    st = state(rec)
    with st.lock:
        result = build_report(_ReportContext(findings=st.store, asset_id=rec.case_id))
        records = [_record_view(r) for r in _records(st, rec)]
        drafts = _report_drafts_excluded(st, rec)
        if result.status == ToolStatus.OK:
            view = ReportView(
                case_id=rec.case_id, status="OK", report=result.data["report"],
                n_confirmed=result.data["n_confirmed"], n_drafts_excluded=result.data["n_drafts_excluded"],
                violations=[], confirmed=records, drafts_excluded=drafts, numeric_check="PASSED",
                generated_at=datetime.now(UTC),
                provenance=S.provenance(rec, method="Phase 8 service report builder and numeric-claim check"))
        else:
            violations = [str(v) for v in result.data.get("violations", [])]
            view = ReportView(
                case_id=rec.case_id, status="REFUSED", report=None,
                n_confirmed=len(records), n_drafts_excluded=len(drafts), violations=violations,
                confirmed=records, drafts_excluded=drafts, numeric_check="REFUSED",
                generated_at=datetime.now(UTC),
                provenance=S.provenance(rec, method="Phase 8 service report builder and numeric-claim check"))
        st.report = view
        log.info("service report built case=%s status=%s confirmed=%d drafts_excluded=%d", rec.case_id, view.status,
                 view.n_confirmed, view.n_drafts_excluded)
        return view


def latest_report(rec) -> ReportView:
    st = state(rec)
    with st.lock:
        if st.report is None:
            raise ApiError(404, ErrorCode.REPORT_NOT_GENERATED, "generate a report for this case first")
        return st.report
