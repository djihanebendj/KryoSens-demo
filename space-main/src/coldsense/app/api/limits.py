"""Operational limits of the HTTP service (Phase 9).

These protect a small single-instance deployment; they are not engineering parameters and do not change any
refusal rule (those stay in ``domains/<name>/thresholds.py``). Like the refusal thresholds they live in code, so a
deployment cannot loosen them by environment variable (D-002); a change is a reviewed code change.
"""

from __future__ import annotations

UPLOAD_MAX_BYTES = 10 * 1024 * 1024         # 10 MiB (spec §7 default): ~120 days of 1-minute data, four columns
UPLOAD_MAX_ROWS = 200_000                   # data rows after ingest; the domain reads every row into memory
UPLOAD_ALLOWED_SUFFIXES = (".csv", ".txt")  # what the ingest layer reads; anything else is refused before parsing
FILENAME_MAX_CHARS = 120                    # display only; the name is never used as a path

SESSION_MAX_AGE_S = 8 * 3600                # a visitor session lives 8 h (extended to cover its newest case)
CASE_TTL_S = 2 * 3600                       # a case lives 2 h from creation, then its data is deleted
CASES_PER_VISITOR = 5                       # live cases per visitor
CASES_TOTAL = 100                           # live cases per process (memory bound: <=10 MiB raw + results each)
SESSIONS_TOTAL = 5000                       # live sessions per process (a session record holds no data)

MAX_CONCURRENT_ANALYSES = 2                 # forecasts take seconds of CPU; more requests wait, then get BUSY
ANALYSIS_QUEUE_WAIT_S = 10.0                # how long a request waits for a free analysis slot

# --- Stage 3 request guards. These bound abuse of a small public instance; they change no result.
REQUEST_MAX_BYTES = UPLOAD_MAX_BYTES + 64 * 1024   # whole request (the upload route enforces the exact 10 MiB)
REQUEST_TIMEOUT_S = 120.0                         # server-side guard; the Next.js proxy gives up at 90 s first
RATE_RULES = {                                    # bucket: (requests, window seconds); in-process, per key
    "upload": (30, 60.0),                          # per browser CSRF nonce; connection address if absent
    "sample": (30, 60.0),                          # per browser CSRF nonce; connection address if absent
    "analysis": (60, 60.0),                        # per visitor session
    "prediction": (60, 60.0),                      # per visitor session
    "findings": (60, 60.0),                        # per visitor session
    "report": (30, 60.0),                          # per visitor session
    "knowledge": (30, 60.0),                        # read-only retrieval; per browser CSRF nonce
}
