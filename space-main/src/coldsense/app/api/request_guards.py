"""Whole-request size and time guards (spec §7).

The upload route already streams at most 10 MiB; this middleware stops an obviously oversized request before a
JSON body is parsed and caps how long any request may occupy the event loop. A timeout does not undo work already
running in a thread: it returns a safe 504 and the request's own locks/state rules still apply. The Next.js proxy
has the tighter 90 s budget in front of this guard.
"""

from __future__ import annotations

import asyncio

from fastapi import FastAPI, Request

from . import limits as L
from .errors import ErrorCode, error_response


def install_request_guards(app: FastAPI) -> None:
    @app.middleware("http")
    async def _request_guard(request: Request, call_next):
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > L.REQUEST_MAX_BYTES:
            return error_response(413, ErrorCode.REQUEST_TOO_LARGE,
                                  f"the request body exceeds {L.REQUEST_MAX_BYTES} bytes")
        try:
            return await asyncio.wait_for(call_next(request), timeout=L.REQUEST_TIMEOUT_S)
        except TimeoutError:
            return error_response(504, ErrorCode.REQUEST_TIMEOUT,
                                  "the request took too long and its response was stopped; refresh the case before "
                                  "retrying, because the server may still finish the work", retryable=False)
