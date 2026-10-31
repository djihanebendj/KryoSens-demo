"""Security design for the ColdSense application:
- Signed HttpOnly session cookie (HMAC-SHA256)
- Synchronizer/HMAC CSRF token bound to the session ID
- Strict Origin check on all state-changing requests (POST, PUT, PATCH, DELETE)
"""

from __future__ import annotations

import hashlib
import os
import warnings
import hmac
import secrets
from typing import Dict, Optional, Set, Tuple


class SecurityError(Exception):
    """Raised when session cookie signature, CSRF token, or Origin check fails."""


def resolve_secret(env_var: str, explicit: Optional[str], purpose: str) -> bytes:
    """The signing secret, in order: the caller's value, then ``env_var``, then a random one.

    A random secret is correct but ephemeral: every restart invalidates the cookies and tokens signed
    with the previous one, so a deployment must set ``env_var``. The warning says so and never prints
    the value.
    """
    value = explicit or os.environ.get(env_var, "").strip()
    if value:
        return value.encode("utf-8")
    generated = secrets.token_urlsafe(32)
    warnings.warn(
        f"{env_var} is not set: generated a random {purpose} secret for this process only. "
        "Sessions and confirmation tokens will not survive a restart; set "
        f"{env_var} in the deployment environment.",
        RuntimeWarning,
        stacklevel=2,
    )
    print(f"[coldsense] WARNING: {env_var} not set; using a random {purpose} secret for this process only.")
    return generated.encode("utf-8")


def extra_allowed_origins_from_env() -> Set[str]:
    """Origins added through COLDSENSE_ALLOWED_ORIGINS (comma-separated, trailing slash dropped, lowercased)."""
    raw = os.environ.get("COLDSENSE_ALLOWED_ORIGINS", "")
    return {o.strip().rstrip("/").lower() for o in raw.split(",") if o.strip()}


class SessionSecurityManager:
    """Validates signed HttpOnly session cookies, CSRF tokens, and Origin headers."""

    def __init__(
        self,
        secret_key: Optional[str] = None,
        allowed_origins: Optional[Set[str]] = None,
    ) -> None:
        self._secret = resolve_secret("COLDSENSE_SESSION_SECRET", secret_key, "session and CSRF signing")
        self.allowed_origins: Set[str] = allowed_origins or {
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        }
        # Deployment glue: COLDSENSE_ALLOWED_ORIGINS (comma-separated) ADDS origins, e.g. the public
        # https://<owner>-<space>.hf.space URL; unset leaves the defaults unchanged.
        self.allowed_origins.update(extra_allowed_origins_from_env())

    def _sign_value(self, payload: str) -> str:
        return hmac.new(self._secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def create_session(self) -> Tuple[str, str, str]:
        """Mint a new session_id, signed cookie string, and session-bound CSRF token.

        Returns:
            (session_id, signed_cookie_value, csrf_token)
        """
        session_id = f"sess-{secrets.token_hex(12)}"
        cookie_sig = self._sign_value(f"cookie:{session_id}")
        signed_cookie = f"{session_id}.{cookie_sig}"

        csrf_nonce = secrets.token_hex(8)
        csrf_sig = self._sign_value(f"csrf:{session_id}:{csrf_nonce}")
        csrf_token = f"csrf-{csrf_nonce}.{csrf_sig}"
        return session_id, signed_cookie, csrf_token

    def verify_session_cookie(self, signed_cookie: Optional[str]) -> str:
        """Verify a signed session cookie and return the raw session_id."""
        if not signed_cookie or "." not in signed_cookie:
            raise SecurityError("Missing or malformed signed HttpOnly session cookie.")
        session_id, sig = signed_cookie.split(".", 1)
        expected = self._sign_value(f"cookie:{session_id}")
        if not hmac.compare_digest(expected, sig):
            raise SecurityError("Invalid signature on session cookie.")
        return session_id

    def verify_csrf_token(self, session_id: str, csrf_token: Optional[str]) -> None:
        """Verify that the X-CSRF-Token header is cryptographically bound to session_id."""
        if not csrf_token or "." not in csrf_token or not csrf_token.startswith("csrf-"):
            raise SecurityError("Missing or malformed CSRF token.")
        prefix, sig = csrf_token.split(".", 1)
        nonce = prefix[len("csrf-") :]
        expected = self._sign_value(f"csrf:{session_id}:{nonce}")
        if not hmac.compare_digest(expected, sig):
            raise SecurityError("CSRF token signature mismatch for active session.")

    def verify_origin(self, origin: Optional[str]) -> None:
        """Verify that state-changing requests originate from an allowed same-origin host."""
        if not origin or origin.strip() not in self.allowed_origins:
            raise SecurityError(
                f"Origin check failed: '{origin}' is not in allowed origins."
            )

    def enforce_state_changing_request(
        self,
        signed_cookie: Optional[str],
        csrf_token: Optional[str],
        origin: Optional[str],
    ) -> str:
        """Enforce signed cookie + CSRF token + Origin check on POST/PUT/DELETE requests.

        Returns the verified session_id.
        """
        self.verify_origin(origin)
        session_id = self.verify_session_cookie(signed_cookie)
        self.verify_csrf_token(session_id, csrf_token)
        return session_id
