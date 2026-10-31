"""In-process sliding-window rate limits for the public demo (spec §7).

A limit is a code constant (``limits.RATE_RULES``), never an environment variable (D-002). The limiter is per
process; with the one-instance deployment in docs/PHASE9_DEPLOYMENT.md that is the whole service. Keys are the
visitor session (for case routes), or the browser CSRF nonce before a session exists, with the client address only
as a fallback. No request content, statement, token or file name is ever used as a key or logged.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

from . import limits as L
from .errors import ApiError, ErrorCode


@dataclass(frozen=True)
class Rule:
    limit: int
    window_s: float


DEFAULT_RULES: dict[str, Rule] = {name: Rule(limit, window) for name, (limit, window) in L.RATE_RULES.items()}


class RateLimiter:
    """One deque of timestamps per (bucket, key). ``check`` either records the hit or raises 429."""

    def __init__(self, rules: dict[str, Rule] | None = None, *, clock=time.monotonic) -> None:
        self._rules = dict(DEFAULT_RULES if rules is None else rules)
        self._clock = clock
        self._hits: dict[tuple[str, str], deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, bucket: str, key: str) -> None:
        rule = self._rules.get(bucket)
        if rule is None or rule.limit <= 0:
            return
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault((bucket, key), deque())
            cutoff = now - rule.window_s
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= rule.limit:
                retry = max(1, int(rule.window_s - (now - hits[0])) + 1)
                raise ApiError(429, ErrorCode.RATE_LIMITED,
                               f"too many requests; try again in about {retry} s", retryable=True)
            hits.append(now)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
