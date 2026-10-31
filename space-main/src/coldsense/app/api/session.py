"""Session cookie and CSRF token signing.

Session cookie ``cs_session = v2.<session_id>.<sig>``: an opaque random session id plus HMAC-SHA256(secret,
"v2.<session_id>"). The cookie carries nothing else; whether the session exists, has expired or was revoked, and
which cases it owns, is server-side state in the CaseStore (spec §4). A cookie whose signature does not verify is
rejected before the store is consulted, so session ids cannot be probed.

CSRF (spec §4, amendment G) uses the signed double-submit pattern and keeps no server state, so the token endpoint
is a read-only GET: ``cs_csrf`` (HttpOnly, SameSite=Strict) holds a random nonce, and the token the page sends in
``X-CSRF-Token`` must equal HMAC-SHA256(secret, "csrf.<nonce>"). Combined with the Origin allowlist in routes.py.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets

COOKIE_NAME = "cs_session"
CSRF_COOKIE = "cs_csrf"
CSRF_HEADER = "X-CSRF-Token"
_VERSION = "v2"
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{22}$")      # 16 random bytes, base64url without padding


def new_id() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(16)).rstrip(b"=").decode()


class SessionSigner:
    def __init__(self, secret: bytes) -> None:
        if len(secret) < 32:
            raise ValueError("the session secret must be at least 32 bytes")
        self._secret = secret

    def _mac(self, payload: str) -> str:
        mac = hmac.new(self._secret, payload.encode(), hashlib.sha256).digest()
        return base64.urlsafe_b64encode(mac).rstrip(b"=").decode()

    # ---- session cookie
    def cookie_for(self, session_id: str) -> str:
        payload = f"{_VERSION}.{session_id}"
        return f"{payload}.{self._mac(payload)}"

    def session_id(self, value: str | None) -> str | None:
        """The session id of a cookie this server signed, else None. Says nothing about expiry or revocation."""
        if not value:
            return None
        parts = value.split(".")
        if len(parts) != 3 or parts[0] != _VERSION or not _ID_RE.match(parts[1]):
            return None
        return parts[1] if hmac.compare_digest(self._mac(f"{parts[0]}.{parts[1]}"), parts[2]) else None

    # ---- CSRF
    def csrf_token(self, nonce: str) -> str:
        return self._mac(f"csrf.{nonce}")

    def csrf_valid(self, nonce: str | None, token: str | None) -> bool:
        if not nonce or not token or not _ID_RE.match(nonce):
            return False
        return hmac.compare_digest(self.csrf_token(nonce), token)

    @staticmethod
    def valid_nonce(nonce: str | None) -> bool:
        return bool(nonce and _ID_RE.match(nonce))
