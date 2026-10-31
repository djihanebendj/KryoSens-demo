"""HTTP routes, version 1.

Every state-changing route (POST, DELETE) passes ``require_csrf`` first: an allowed ``Origin`` header and a valid
``X-CSRF-Token`` bound to this browser's ``cs_csrf`` cookie (spec §4, amendment G). Every case route then checks the
server-side session and the case's owner. GET routes never change server state.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, Response
from starlette.concurrency import run_in_threadpool

from coldsense import __version__
from coldsense.core.evidence import EvidenceClass
from coldsense.knowledge.security import MAX_QUERY_CHARS

from . import findings as F
from . import limits as L
from . import service as S
from .config import normalize_origin
from .copilot_gate import evaluate as copilot_gate
from .errors import ApiError, ErrorCode, responses
from .knowledge import KnowledgeUnavailable
from .schemas import (
    AnalysisRequest,
    AnalysisView,
    CaseStatus,
    CopilotRequest,
    CsrfToken,
    DeclaredEvidence,
    EvidenceView,
    FeatureState,
    FindingConfirmRequest,
    FindingDraftCreated,
    FindingDraftRequest,
    FindingDraftView,
    FindingRecordView,
    FindingsView,
    Health,
    IncidentView,
    InvestigationView,
    KnowledgeView,
    NextCheckView,
    PredictionRequest,
    PredictionResponse,
    QualityView,
    ReportView,
    SampleInfo,
    TimelineView,
)
from .session import COOKIE_NAME, CSRF_COOKIE, CSRF_HEADER, new_id
from .store import SessionState

router = APIRouter(prefix="/api/v1")

# Bundled, explicitly labelled ColdSense samples: the only data the public flow analyses as SIMULATED (amendment G).
SAMPLES = {
    "power-cut-22h": ("SIMULATED_power_cut_22h_1min.csv",
                      "SIMULATED: 16 h of thermostat cycling, then a recorded power cut (compressor OFF) for 6 h"),
    "cycling-24h": ("SIMULATED_coldroom_24h_1min.csv",
                    "SIMULATED: 24 h of thermostat cycling with no compressor column (a forecast refuses)"),
}


@dataclass(frozen=True)
class Visitor:
    session_id: str


def _state(request: Request):
    return request.app.state


def set_session_cookie(app_state, response: Response, session_id: str, max_age: int) -> None:
    response.set_cookie(COOKIE_NAME, app_state.signer.cookie_for(session_id), max_age=max_age, httponly=True,
                        samesite="lax", secure=app_state.config.cookie_secure, path="/")


# ---------------------------------------------------------------- CSRF
def require_csrf(request: Request) -> None:
    st = _state(request)
    origin = request.headers.get("origin")
    if not origin or normalize_origin(origin) not in st.config.allowed_origins:
        raise ApiError(403, ErrorCode.CSRF_ORIGIN_REJECTED,
                       "state-changing requests must come from an allowed origin" if origin
                       else "state-changing requests must send an Origin header")
    nonce, token = request.cookies.get(CSRF_COOKIE), request.headers.get(CSRF_HEADER)
    if not nonce or not token:
        raise ApiError(403, ErrorCode.CSRF_TOKEN_MISSING,
                       f"get a token from GET /api/v1/csrf and send it as {CSRF_HEADER}")
    if not st.signer.csrf_valid(nonce, token):
        raise ApiError(403, ErrorCode.CSRF_TOKEN_INVALID, "the CSRF token does not match this browser's CSRF cookie")


CsrfDep = [Depends(require_csrf)]


# ---------------------------------------------------------------- sessions
def _session_error(state: SessionState) -> ApiError:
    if state == SessionState.EXPIRED:
        return ApiError(401, ErrorCode.SESSION_EXPIRED, "your session expired; upload the file again",
                        clear_session=True)
    if state == SessionState.REVOKED:
        return ApiError(401, ErrorCode.SESSION_REVOKED, "this session was ended; upload the file again",
                        clear_session=True)
    return ApiError(401, ErrorCode.SESSION_INVALID, "the session cookie is not valid for this server",
                    clear_session=True)


def require_visitor(request: Request) -> Visitor:
    st = _state(request)
    raw = request.cookies.get(COOKIE_NAME)
    if not raw:
        raise ApiError(401, ErrorCode.SESSION_REQUIRED, "no session: upload a file to start a case")
    sid = st.signer.session_id(raw)
    if sid is None:
        raise _session_error(SessionState.UNKNOWN)
    state = st.store.session_state(sid)
    if state != SessionState.VALID:
        raise _session_error(state)
    return Visitor(sid)


VisitorDep = Annotated[Visitor, Depends(require_visitor)]
_CASE_ERRORS = responses(401, 404, 410)
_WRITE_ERRORS = (401, 403, 404, 410, 422, 429, 503)


def _case(request: Request, case_id: str, visitor: Visitor):
    return _state(request).store.get(case_id, visitor.session_id)


def _rate(request: Request, bucket: str, *, session_id: str | None = None) -> None:
    """Rate-limit before the work. A visitor session is the key; before one exists, the browser's CSRF nonce
    (a per-browser HttpOnly value, set by GET /csrf before any valid write) is used, so clients behind the same
    same-origin proxy do not share one bucket. Only a missing cookie falls back to the connection address. No
    request content is used."""
    key = session_id or request.cookies.get(CSRF_COOKIE) or (request.client.host if request.client is not None
                                                             else "unknown")
    _state(request).limiter.check(bucket, key)


# ---------------------------------------------------------------- service
@router.get("/health", response_model=Health, tags=["service"])
def health(request: Request) -> Health:
    """The Copilot state comes from the server-side gate; the frontend never decides it."""
    gate = copilot_gate(_state(request).config)
    return Health(coldsense_version=__version__, copilot=gate.state, copilot_reason=gate.reason)


@router.get("/knowledge", response_model=KnowledgeView, tags=["knowledge"], responses=responses(422, 429))
def knowledge(
    request: Request,
    q: Annotated[str, Query(min_length=1, max_length=MAX_QUERY_CHARS,
                            description="a technical question; quoted passages only, never instructions")],
    k: Annotated[int, Query(ge=1, le=5, description="maximum cited passages")] = 5,
) -> KnowledgeView:
    """Read-only Phase 7 retrieval over the local SQLite knowledge store.

    The retrieval is deliberately separate from diagnosis: a quoted manual passage never changes a refusal or a
    finding. If the index is missing the response is a clear ``NOT_AVAILABLE`` result, not a 500.
    """
    query = q.strip()
    if not query:
        raise ApiError(422, ErrorCode.INVALID_REQUEST, "query must not be blank")
    _rate(request, "knowledge")
    try:
        answer = _state(request).knowledge.query(query, k)
    except KnowledgeUnavailable as e:
        return KnowledgeView(query=query, status="NOT_AVAILABLE", reason=str(e))
    return KnowledgeView(query=query, answer=answer)


@router.get("/csrf", response_model=CsrfToken, tags=["service"])
def csrf(request: Request, response: Response) -> CsrfToken:
    """Read-only: no server state. Sets (or keeps) the ``cs_csrf`` nonce cookie and returns its token."""
    st = _state(request)
    nonce = request.cookies.get(CSRF_COOKIE)
    if not st.signer.valid_nonce(nonce):
        nonce = new_id()
        response.set_cookie(CSRF_COOKIE, nonce, httponly=True, samesite="strict", secure=st.config.cookie_secure,
                            path="/")
    return CsrfToken(csrf_token=st.signer.csrf_token(nonce))


@router.delete("/session", status_code=204, tags=["service"], dependencies=CsrfDep, responses=responses(401, 403))
def end_session(request: Request, visitor: VisitorDep) -> Response:
    """Revoke this session and delete every case it owns, now."""
    _state(request).store.revoke_session(visitor.session_id)
    resp = Response(status_code=204)
    resp.delete_cookie(COOKIE_NAME, path="/", secure=_state(request).config.cookie_secure, httponly=True,
                       samesite="lax")
    return resp


@router.get("/samples", response_model=list[SampleInfo], tags=["cases"])
def samples() -> list[SampleInfo]:
    return [SampleInfo(sample_id=k, filename=f, description=d) for k, (f, d) in SAMPLES.items()]


# ---------------------------------------------------------------- case creation
def _validate_in_slot(slots, content, name, evidence, options):
    with slots:                       # waits in a worker thread, never on the event loop
        return S.validate_upload(content, name, evidence, options)


async def _start_case(request: Request, response: Response, *, name: str, content: bytes, evidence: EvidenceClass,
                      declared: str | None, source_kind: str, options) -> CaseStatus:
    st = _state(request)
    report = await run_in_threadpool(_validate_in_slot, st.slots, content, name, evidence, options)
    if report.n_rows_raw > L.UPLOAD_MAX_ROWS:
        raise ApiError(413, ErrorCode.UPLOAD_TOO_MANY_ROWS, f"at most {L.UPLOAD_MAX_ROWS} data rows are accepted")
    sid = st.signer.session_id(request.cookies.get(COOKIE_NAME))
    if sid is None or st.store.session_state(sid) != SessionState.VALID:
        sid = st.store.create_session().session_id        # missing, invalid, expired or revoked: a new session
    rec = st.store.create(owner=sid, filename=name, evidence=evidence.value, declared_evidence=declared,
                          source_kind=source_kind, content=content, sha256=hashlib.sha256(content).hexdigest(),
                          options={"ingest": options})
    rec.results["quality_report"] = report
    rec.findings = F.FindingState()          # per-case findings state; created here so a GET never mutates the case
    session = st.store.extend_session(sid, rec.expires_at)  # the session outlives its newest case
    set_session_cookie(st, response, sid, max_age=int(session.expires_at - rec.created_at))
    return S.case_status(rec)


async def _read_body_capped(request: Request) -> bytes:
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > L.UPLOAD_MAX_BYTES:
        raise ApiError(413, ErrorCode.UPLOAD_TOO_LARGE, f"the file exceeds {L.UPLOAD_MAX_BYTES} bytes")
    chunks, total = [], 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > L.UPLOAD_MAX_BYTES:
            raise ApiError(413, ErrorCode.UPLOAD_TOO_LARGE, f"the file exceeds {L.UPLOAD_MAX_BYTES} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


_UPLOAD_BODY = {"requestBody": {"required": True, "content": {
    "text/csv": {"schema": {"type": "string", "format": "binary"}},
    "application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}}}


@router.post("/cases", status_code=201, response_model=CaseStatus, tags=["cases"], openapi_extra=_UPLOAD_BODY,
             dependencies=CsrfDep, responses=responses(403, 413, 415, 422, 429, 503))
async def create_case(
    request: Request, response: Response,
    filename: Annotated[str, Query(min_length=1, max_length=255, description="original file name (display only)")],
    declared_evidence: Annotated[DeclaredEvidence | None, Query(
        description="what you say this file is: stored and shown as 'declared by uploader, not verified'; it never "
                    "changes the evidence class (always UNVALIDATED for uploads) or the Phase 5 scope")] = None,
    unit: Annotated[str | None, Query(pattern="^[CFK]$")] = None,
    timezone: Annotated[str | None, Query(max_length=64)] = None,
    timestamp_format: Annotated[str | None, Query(max_length=64)] = None,
    delimiter: Annotated[str | None, Query(max_length=4)] = None,
    encoding: Annotated[str | None, Query(max_length=32)] = None,
) -> CaseStatus:
    """Upload a log (raw CSV body, not multipart: nothing is spooled to disk) and start a case. The data is analysed
    as UNVALIDATED. Runs the Phase 2 data-quality check immediately."""
    _rate(request, "upload")
    name = S.sanitize_filename(filename)
    content = await _read_body_capped(request)
    S.check_upload(name, content)
    options = S.ingest_options(unit=unit, timezone=timezone, timestamp_format=timestamp_format,
                               delimiter=delimiter, encoding=encoding)
    return await _start_case(request, response, name=name, content=content, evidence=EvidenceClass.UNVALIDATED,
                             declared=declared_evidence, source_kind="ANONYMOUS_UPLOAD", options=options)


@router.post("/cases/samples/{sample_id}", status_code=201, response_model=CaseStatus, tags=["cases"],
             dependencies=CsrfDep, responses=responses(403, 404, 429, 503))
async def create_sample_case(request: Request, response: Response, sample_id: str) -> CaseStatus:
    """Start a case from a bundled, labelled ColdSense SIMULATED sample (a server-side file; no visitor data)."""
    _rate(request, "sample")
    if sample_id not in SAMPLES:
        raise ApiError(404, ErrorCode.SAMPLE_NOT_FOUND, "no such bundled sample")
    filename = SAMPLES[sample_id][0]
    path = Path(_state(request).samples_dir) / filename
    try:
        content = path.read_bytes()
    except OSError:
        raise ApiError(404, ErrorCode.SAMPLE_NOT_FOUND, "the bundled sample is not installed on this server") from None
    return await _start_case(request, response, name=filename, content=content, evidence=EvidenceClass.SIMULATED,
                             declared=None, source_kind="BUNDLED_SAMPLE",
                             options=S.ingest_options(unit=None, timezone=None, timestamp_format=None, delimiter=None,
                                                      encoding=None))


# ---------------------------------------------------------------- case reads and writes
@router.get("/cases/{case_id}", response_model=CaseStatus, tags=["cases"], responses=_CASE_ERRORS)
def get_case(request: Request, case_id: str, visitor: VisitorDep) -> CaseStatus:
    return S.case_status(_case(request, case_id, visitor))


@router.delete("/cases/{case_id}", status_code=204, tags=["cases"], dependencies=CsrfDep,
               responses=responses(401, 403, 404))
def delete_case(request: Request, case_id: str, visitor: VisitorDep) -> Response:
    _state(request).store.delete(case_id, visitor.session_id)
    return Response(status_code=204)


@router.get("/cases/{case_id}/quality", response_model=QualityView, tags=["cases"], responses=_CASE_ERRORS)
def get_quality(request: Request, case_id: str, visitor: VisitorDep) -> QualityView:
    return S.quality_view(_case(request, case_id, visitor))


@router.post("/cases/{case_id}/analysis", response_model=AnalysisView, tags=["cases"], dependencies=CsrfDep,
             responses=responses(*_WRITE_ERRORS))
def run_analysis(request: Request, case_id: str, body: AnalysisRequest, visitor: VisitorDep) -> AnalysisView:
    """Phase 3 event segmentation (+ Stage 2: Phase 6 diagnosis) with the limits you set; replaces the previous run."""
    rec = _case(request, case_id, visitor)
    _rate(request, "analysis", session_id=visitor.session_id)
    with rec.lock:
        return S.run_analysis(rec, body, _state(request).slots)


@router.get("/cases/{case_id}/analysis", response_model=AnalysisView, tags=["cases"],
            responses=responses(401, 404, 410))
def get_analysis(request: Request, case_id: str, visitor: VisitorDep) -> AnalysisView:
    rec = _case(request, case_id, visitor)
    return S.cached(rec, "analysis", ErrorCode.ANALYSIS_NOT_RUN, "run the analysis for this case first")


@router.post("/cases/{case_id}/prediction", response_model=PredictionResponse, tags=["cases"], dependencies=CsrfDep,
             responses=responses(*_WRITE_ERRORS))
def create_prediction(request: Request, case_id: str, body: PredictionRequest, visitor: VisitorDep):
    """Phase 4 forecast (+ Phase 5 reliability). A refusal is a normal 200 result with ``outcome: REFUSED``."""
    rec = _case(request, case_id, visitor)
    _rate(request, "prediction", session_id=visitor.session_id)
    out = S.predict(rec, body, _state(request).slots)
    rec.results["prediction"] = out
    return out


@router.get("/cases/{case_id}/prediction", response_model=PredictionResponse, tags=["cases"],
            responses=responses(401, 404, 410))
def get_prediction(request: Request, case_id: str, visitor: VisitorDep):
    rec = _case(request, case_id, visitor)
    return S.cached(rec, "prediction", ErrorCode.PREDICTION_NOT_RUN, "no prediction has been requested for this case")


# ---------------------------------------------------------------- Stage 2: analysis workspace (read-only views)
_VIEW_ERRORS = responses(401, 404, 410)


@router.get("/cases/{case_id}/timeline", response_model=TimelineView, tags=["workspace"], responses=_VIEW_ERRORS)
def get_timeline(request: Request, case_id: str, visitor: VisitorDep) -> TimelineView:
    """Usable readings split at Phase 3 data gaps, with the Phase 3 events. Needs POST .../analysis first."""
    return S.timeline_view(_case(request, case_id, visitor))


@router.get("/cases/{case_id}/evidence", response_model=EvidenceView, tags=["workspace"], responses=_VIEW_ERRORS)
def get_evidence(request: Request, case_id: str, visitor: VisitorDep) -> EvidenceView:
    return S.evidence_view(_case(request, case_id, visitor))


@router.get("/cases/{case_id}/next-check", response_model=NextCheckView, tags=["workspace"], responses=_VIEW_ERRORS)
def get_next_check(request: Request, case_id: str, visitor: VisitorDep) -> NextCheckView:
    return S.next_check_view(_case(request, case_id, visitor))


@router.get("/cases/{case_id}/investigation", response_model=InvestigationView, tags=["workspace"],
            responses=_VIEW_ERRORS)
def get_investigation(request: Request, case_id: str, visitor: VisitorDep) -> InvestigationView:
    return S.investigation_view(_case(request, case_id, visitor))


@router.get("/cases/{case_id}/incident", response_model=IncidentView, tags=["workspace"], responses=_VIEW_ERRORS)
def get_incident(request: Request, case_id: str, visitor: VisitorDep) -> IncidentView:
    """Drafts, confirmed findings and their outcome state. Read-only; nothing is written by this route."""
    return S.incident_view(_case(request, case_id, visitor))


# ---------------------------------------------------------------- Stage 3: findings and reports
_FINDING_WRITE_ERRORS = responses(401, 403, 404, 409, 410, 422, 429)
_FINDING_READ_ERRORS = responses(401, 404, 410)


@router.post("/cases/{case_id}/findings/drafts", status_code=201, response_model=FindingDraftCreated,
             tags=["findings"], dependencies=CsrfDep, responses=_FINDING_WRITE_ERRORS)
def create_finding_draft(request: Request, case_id: str, body: FindingDraftRequest,
                         visitor: VisitorDep) -> FindingDraftCreated:
    """Create a PENDING draft from the technician's own statement. A draft is not history; it writes no record."""
    rec = _case(request, case_id, visitor)
    _rate(request, "findings", session_id=visitor.session_id)
    return F.create_draft(rec, body)


@router.post("/cases/{case_id}/findings/drafts/{draft_id}/confirm", response_model=FindingRecordView,
             tags=["findings"], dependencies=CsrfDep, responses=_FINDING_WRITE_ERRORS)
def confirm_finding(request: Request, case_id: str, draft_id: str, body: FindingConfirmRequest,
                    visitor: VisitorDep) -> FindingRecordView:
    """Explicit user confirmation with the single-use token from the draft. This is the only route that appends."""
    rec = _case(request, case_id, visitor)
    _rate(request, "findings", session_id=visitor.session_id)
    return F.confirm_draft(rec, draft_id, body)


@router.post("/cases/{case_id}/findings/drafts/{draft_id}/cancel", response_model=FindingDraftView,
             tags=["findings"], dependencies=CsrfDep, responses=_FINDING_WRITE_ERRORS)
def cancel_finding(request: Request, case_id: str, draft_id: str, visitor: VisitorDep) -> FindingDraftView:
    """Cancel a PENDING draft. Nothing is appended to the Phase 8 store and no report content changes."""
    rec = _case(request, case_id, visitor)
    _rate(request, "findings", session_id=visitor.session_id)
    return F.cancel_draft(rec, draft_id)


@router.get("/cases/{case_id}/findings", response_model=FindingsView, tags=["findings"],
            responses=_FINDING_READ_ERRORS)
def list_findings(request: Request, case_id: str, visitor: VisitorDep) -> FindingsView:
    return F.findings_view(_case(request, case_id, visitor))


@router.post("/cases/{case_id}/reports", response_model=ReportView, tags=["reports"], dependencies=CsrfDep,
             responses=responses(401, 403, 404, 410, 429))
def create_report(request: Request, case_id: str, visitor: VisitorDep) -> ReportView:
    """Compose the report with the Phase 8 builder from confirmed records only; the numeric check can refuse it."""
    rec = _case(request, case_id, visitor)
    _rate(request, "report", session_id=visitor.session_id)
    return F.build_report_view(rec)


@router.get("/cases/{case_id}/reports/latest", response_model=ReportView, tags=["reports"],
            responses=responses(401, 404, 410))
def get_latest_report(request: Request, case_id: str, visitor: VisitorDep) -> ReportView:
    return F.latest_report(_case(request, case_id, visitor))


@router.post("/cases/{case_id}/copilot", tags=["cases"], dependencies=CsrfDep, responses=responses(401, 403, 404, 410,
                                                                                                    503))
def copilot(request: Request, case_id: str, body: CopilotRequest, visitor: VisitorDep) -> None:
    """Technician copilot: the server gate decides. While disabled no LLM provider is constructed, no LLM is called
    and no tool runs, even for a direct API request."""
    _case(request, case_id, visitor)
    gate = copilot_gate(_state(request).config)
    if gate.state is not FeatureState.ENABLED:
        raise ApiError(503, ErrorCode.FEATURE_DISABLED, "the technician copilot is " + gate.reason)
    raise ApiError(503, ErrorCode.FEATURE_DISABLED,
                   "the technician copilot gate is open, but this Phase 9 slice does not wire a conversation "
                   "provider; no LLM or tool call was made")
