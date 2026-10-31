"""Error envelope of the HTTP service.

Every non-2xx response has the same body: ``{"error": {"code", "message", "retryable"}}``. Codes are stable and
machine-readable; messages are for people and never contain a traceback, a file path or another visitor's data.
A refusal by the domain (data unsuitable, forecast refused) is NOT an error: it is a 200 result (D-003).
"""

from __future__ import annotations

import logging
from enum import Enum

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger(__name__)


class ErrorCode(str, Enum):
    SESSION_REQUIRED = "SESSION_REQUIRED"            # no session cookie on a request that needs one
    SESSION_INVALID = "SESSION_INVALID"              # cookie present but not signed by this server
    SESSION_EXPIRED = "SESSION_EXPIRED"              # the session passed its lifetime
    SESSION_REVOKED = "SESSION_REVOKED"              # the session was ended (its cases were deleted)
    CSRF_ORIGIN_REJECTED = "CSRF_ORIGIN_REJECTED"    # state-changing request without an allowed Origin header
    CSRF_TOKEN_MISSING = "CSRF_TOKEN_MISSING"        # state-changing request without X-CSRF-Token / cs_csrf cookie
    CSRF_TOKEN_INVALID = "CSRF_TOKEN_INVALID"        # token does not match this browser's CSRF cookie
    CASE_NOT_FOUND = "CASE_NOT_FOUND"                # unknown, deleted, or owned by another visitor (not told apart)
    CASE_EXPIRED = "CASE_EXPIRED"                    # this visitor's case passed its lifetime; its data is deleted
    CASE_LIMIT_REACHED = "CASE_LIMIT_REACHED"        # per-visitor or per-process case limit
    UPLOAD_EMPTY = "UPLOAD_EMPTY"
    UPLOAD_TOO_LARGE = "UPLOAD_TOO_LARGE"
    UPLOAD_TYPE_NOT_ALLOWED = "UPLOAD_TYPE_NOT_ALLOWED"
    UPLOAD_TOO_MANY_ROWS = "UPLOAD_TOO_MANY_ROWS"
    DATA_UNREADABLE = "DATA_UNREADABLE"              # the file cannot be interpreted at all (core DataError)
    INVALID_REQUEST = "INVALID_REQUEST"              # request body/parameters fail validation
    ANALYSIS_NOT_RUN = "ANALYSIS_NOT_RUN"            # a view that needs POST .../analysis first
    PREDICTION_NOT_RUN = "PREDICTION_NOT_RUN"        # GET of a prediction that was never requested for this case
    SAMPLE_NOT_FOUND = "SAMPLE_NOT_FOUND"            # unknown bundled sample id
    FEATURE_DISABLED = "FEATURE_DISABLED"            # the capability exists but is switched off in this deployment
    NOT_AVAILABLE = "NOT_AVAILABLE"                  # no backend exists for this capability yet
    BUSY = "BUSY"                                    # every analysis slot is in use
    RATE_LIMITED = "RATE_LIMITED"                    # too many requests from this client/session in the window
    REQUEST_TOO_LARGE = "REQUEST_TOO_LARGE"          # whole request exceeds the global cap (before parsing)
    REQUEST_TIMEOUT = "REQUEST_TIMEOUT"              # the server-side request guard expired
    FINDING_DRAFT_NOT_FOUND = "FINDING_DRAFT_NOT_FOUND"
    FINDING_NOT_READY = "FINDING_NOT_READY"          # the draft has no live confirmation token
    FINDING_CANCELLED = "FINDING_CANCELLED"          # a cancelled draft can never be confirmed
    FINDING_ALREADY_CONFIRMED = "FINDING_ALREADY_CONFIRMED"
    FINDING_CONFIRMATION_INVALID = "FINDING_CONFIRMATION_INVALID"
    FINDING_CONFIRMATION_EXPIRED = "FINDING_CONFIRMATION_EXPIRED"
    FINDING_CONFIRMATION_REPLAY = "FINDING_CONFIRMATION_REPLAY"   # the same single-use token was submitted twice
    REPORT_NOT_GENERATED = "REPORT_NOT_GENERATED"    # GET latest before POST .../reports
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ErrorBody(BaseModel):
    code: ErrorCode
    message: str
    retryable: bool = False


class ErrorEnvelope(BaseModel):
    error: ErrorBody


class ApiError(Exception):
    def __init__(self, status: int, code: ErrorCode, message: str, *, retryable: bool = False,
                 clear_session: bool = False) -> None:
        super().__init__(message)
        self.status, self.code, self.message, self.retryable = status, code, message, retryable
        self.clear_session = clear_session


def error_response(status: int, code: ErrorCode, message: str, retryable: bool = False) -> JSONResponse:
    body = ErrorEnvelope(error=ErrorBody(code=code, message=message, retryable=retryable))
    return JSONResponse(status_code=status, content=body.model_dump(mode="json"),
                        headers={"Cache-Control": "no-store"})


# Documented in OpenAPI for every route that can return them
def responses(*statuses: int) -> dict:
    return {s: {"model": ErrorEnvelope} for s in statuses}


def install_handlers(app: FastAPI, *, clear_cookie) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, e: ApiError) -> JSONResponse:
        resp = error_response(e.status, e.code, e.message, e.retryable)
        if e.clear_session:
            clear_cookie(resp)
        return resp

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, e: RequestValidationError) -> JSONResponse:
        # Field names and messages only: the offending input is not echoed back.
        parts = []
        for x in e.errors():
            loc = ".".join(str(p) for p in x.get("loc", ()) if p not in ("body", "query", "path"))
            parts.append(f"{loc or 'request'}: {x.get('msg', 'invalid')}")
        return error_response(422, ErrorCode.INVALID_REQUEST, "; ".join(parts) or "invalid request")

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, e: StarletteHTTPException) -> JSONResponse:
        code = {404: ErrorCode.NOT_AVAILABLE, 405: ErrorCode.INVALID_REQUEST}.get(e.status_code,
                                                                                 ErrorCode.INVALID_REQUEST)
        msg = "no such endpoint" if e.status_code == 404 else str(e.detail)
        return error_response(e.status_code, code, msg)

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, e: Exception) -> JSONResponse:
        log.exception("unhandled error in API request: %s", type(e).__name__)
        return error_response(500, ErrorCode.INTERNAL_ERROR, "unexpected server error; nothing was changed")
