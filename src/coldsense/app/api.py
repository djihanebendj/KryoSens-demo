"""HTTP / REST API Router & Static Web App Server for ColdSense (SmartClim).

Supports:
1. Serving the production frontend (`src/coldsense/web/index.html`, `styles.css`, `app.js`)
   at `/` and `/assets/*` on the exact same origin as `/api/*`.
2. Zero-dependency in-process `ColdSenseTestClient` and `ColdSenseHTTPServer` (`ThreadingHTTPServer`).
3. Optional `create_fastapi_app()` factory when `fastapi` is installed.
"""

from __future__ import annotations

import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

from coldsense.app.case_store import CaseNotFoundError, RateLimitExceededError
from coldsense.app.security import SecurityError
from coldsense.app.limits import UPLOAD_MAX_BYTES, UPLOAD_MAX_ROWS
from coldsense.app.service import ColdSenseService, copilot_enabled_from_env, WorkflowGateError
from coldsense.confirmation import ConfirmationError

WEB_DIR = Path(__file__).resolve().parents[1] / "web"


# Sent on every response. The UI is one origin serving its own HTML, CSS and JSON, so a strict policy
# costs nothing: no framing (clickjacking), no MIME sniffing, no referrer leakage, and scripts and
# styles only from this origin. 'unsafe-inline' is required because index.html carries inline handlers
# and a <style> block; removing those is a UI change, not a security fix, so it is left alone.
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'; "
        "img-src 'self' data:; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
    ),
}


class ColdSenseAPIDispatcher:
    """Transport-independent HTTP request dispatcher enforcing cookies, CSRF, Origin, and status codes."""

    def __init__(self, service: Optional[ColdSenseService] = None) -> None:
        # The copilot is a server-side decision only: no route and no visitor action can enable it.
        self.service = service or ColdSenseService(copilot_enabled=copilot_enabled_from_env())

    def handle_request(
        self,
        method: str,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Dict[str, str], Dict[str, Any]]:
        """Dispatch an HTTP API request and return (status_code, response_headers, json_body)."""
        hdrs = {k.lower(): v for k, v in (headers or {}).items()}
        payload = body or {}
        method_up = method.upper()
        parsed_path = urlparse(path).path.rstrip("/") or "/"

        # 1. Session bootstrap endpoint (GET /api/session)
        if method_up == "GET" and parsed_path == "/api/session":
            # Reuse existing valid session cookie if present so browser refreshes preserve cases
            cookie_hdr = hdrs.get("cookie", "")
            existing_cookie: Optional[str] = None
            for part in cookie_hdr.split(";"):
                part = part.strip()
                if part.startswith("coldsense_session="):
                    existing_cookie = part[len("coldsense_session=") :]
                    break
            if existing_cookie:
                try:
                    session_id = self.service.security.verify_session_cookie(existing_cookie)
                    csrf_nonce = "active"
                    csrf_sig = self.service.security._sign_value(f"csrf:{session_id}:{csrf_nonce}")
                    csrf_token = f"csrf-{csrf_nonce}.{csrf_sig}"
                    return (
                        200,
                        {"Content-Type": "application/json"},
                        {
                            "session_id": session_id,
                            "signed_cookie": existing_cookie,
                            "csrf_token": csrf_token,
                            "copilot_enabled": self.service.copilot.enabled,
                        },
                    )
                except SecurityError:
                    pass

            session_id, signed_cookie, csrf_token = self.service.security.create_session()
            resp_headers = {
                "Set-Cookie": f"coldsense_session={signed_cookie}; HttpOnly; SameSite=Strict; Path=/",
                "Content-Type": "application/json",
            }
            return (
                200,
                resp_headers,
                {
                    "session_id": session_id,
                    "signed_cookie": signed_cookie,
                    "csrf_token": csrf_token,
                    "copilot_enabled": self.service.copilot.enabled,
                },
            )

        # Extract signed session cookie from Cookie header
        cookie_hdr = hdrs.get("cookie", "")
        signed_cookie = None
        for part in cookie_hdr.split(";"):
            part = part.strip()
            if part.startswith("coldsense_session="):
                signed_cookie = part[len("coldsense_session=") :]
                break

        try:
            if method_up in ("POST", "PUT", "PATCH", "DELETE"):
                session_id = self.service.security.enforce_state_changing_request(
                    signed_cookie=signed_cookie,
                    csrf_token=hdrs.get("x-csrf-token"),
                    origin=hdrs.get("origin"),
                )
            else:
                session_id = self.service.security.verify_session_cookie(signed_cookie)

            # Route matching
            if method_up == "GET" and parsed_path == "/api/dashboard":
                return 200, {"Content-Type": "application/json"}, self.service.get_dashboard(session_id)

            if method_up == "POST" and parsed_path == "/api/cases/upload":
                raw_content = str(payload.get("raw_content", ""))
                content_bytes = len(raw_content.encode("utf-8"))
                if content_bytes > UPLOAD_MAX_BYTES:
                    return (
                        413,
                        {"Content-Type": "application/json"},
                        {
                            "error": "UPLOAD_TOO_LARGE",
                            "message": f"uploaded content is {content_bytes} bytes; the limit is "
                                       f"{UPLOAD_MAX_BYTES} bytes ({UPLOAD_MAX_BYTES // (1024 * 1024)} MiB).",
                        },
                    )
                row_count = max(0, raw_content.count("\n") + (1 if raw_content and not raw_content.endswith("\n") else 0) - 1)
                if row_count > UPLOAD_MAX_ROWS:
                    return (
                        413,
                        {"Content-Type": "application/json"},
                        {
                            "error": "UPLOAD_TOO_MANY_ROWS",
                            "message": f"uploaded log has about {row_count} data rows; the limit is "
                                       f"{UPLOAD_MAX_ROWS}.",
                        },
                    )
                res = self.service.upload_case(
                    session_id=session_id,
                    raw_content=raw_content,
                    fmt=str(payload.get("format", "csv")),
                    title=str(payload.get("title", "Uploaded Cold-Room Log")),
                    user_claimed_real=bool(payload.get("user_claimed_real", False)),
                    critical_threshold_c=float(payload.get("critical_threshold_c", 8.0)),
                )
                return 201, {"Content-Type": "application/json"}, res

            if method_up == "POST" and parsed_path == "/api/cases/demo":
                res = self.service.create_demo_case(
                    session_id=session_id,
                    scenario=str(payload.get("scenario", "power_cut")),
                )
                return 201, {"Content-Type": "application/json"}, res

            # Case-scoped routes: /api/cases/{case_id}/...
            m_case = re.match(r"^/api/cases/([^/]+)/([a-z-]+(?:/[a-z-]+)?)$", parsed_path)
            if m_case:
                case_id, action = m_case.group(1), m_case.group(2)
                if method_up == "POST" and action == "data-quality":
                    return (
                        200,
                        {"Content-Type": "application/json"},
                        self.service.run_step2_data_quality(session_id, case_id),
                    )
                if method_up == "POST" and action == "events":
                    return (
                        200,
                        {"Content-Type": "application/json"},
                        self.service.run_step3_timeline_events(session_id, case_id),
                    )
                if method_up == "POST" and action == "prediction":
                    return (
                        200,
                        {"Content-Type": "application/json"},
                        self.service.run_step4_prediction(
                            session_id,
                            case_id,
                            event_id=payload.get("event_id"),
                            horizon_sec=float(payload.get("horizon_sec", 3600.0)),
                        ),
                    )
                if method_up == "GET" and action == "evidence":
                    return (
                        200,
                        {"Content-Type": "application/json"},
                        self.service.run_step5_evidence(session_id, case_id),
                    )
                if method_up == "GET" and action == "next-best-check":
                    return (
                        200,
                        {"Content-Type": "application/json"},
                        self.service.run_step6_next_best_check(session_id, case_id),
                    )
                if method_up == "GET" and action == "investigation":
                    return (
                        200,
                        {"Content-Type": "application/json"},
                        self.service.get_investigation_view(session_id, case_id),
                    )
                if method_up == "POST" and action == "findings/stage":
                    return (
                        200,
                        {"Content-Type": "application/json"},
                        self.service.stage_finding_for_confirmation(
                            session_id=session_id,
                            case_id=case_id,
                            hypothesis_id=str(payload.get("hypothesis_id", "")),
                            check_id=str(payload.get("check_id", "")),
                            proposed_status=str(payload.get("proposed_status", "")),
                            technician_id=str(payload.get("technician_id", "")),
                            technician_observation=str(payload.get("technician_observation", "")),
                        ),
                    )
                if method_up == "POST" and action == "findings/confirm":
                    return (
                        200,
                        {"Content-Type": "application/json"},
                        self.service.confirm_staged_finding(
                            session_id=session_id,
                            case_id=case_id,
                            hypothesis_id=str(payload.get("hypothesis_id", "")),
                            check_id=str(payload.get("check_id", "")),
                            proposed_status=str(payload.get("proposed_status", "")),
                            technician_id=str(payload.get("technician_id", "")),
                            technician_observation=str(payload.get("technician_observation", "")),
                            token_id=str(payload.get("token_id", "")),
                            signature=str(payload.get("signature", "")),
                            caller_role=str(payload.get("caller_role", "technician")),
                        ),
                    )
                if method_up == "GET" and action == "incident":
                    return (
                        200,
                        {"Content-Type": "application/json"},
                        self.service.get_incident_view(session_id, case_id),
                    )
                if method_up == "GET" and action == "report":
                    return (
                        200,
                        {"Content-Type": "application/json"},
                        self.service.get_step10_report(session_id, case_id),
                    )

            if method_up == "POST" and parsed_path == "/api/copilot/ask":
                res = self.service.run_step7_copilot(
                    session_id=session_id,
                    query=str(payload.get("query", "")),
                    case_id=payload.get("case_id"),
                )
                status = 503 if not self.service.copilot.enabled else 200
                return status, {"Content-Type": "application/json"}, res

            if method_up == "POST" and parsed_path == "/api/kb/query":
                query_str = str(payload.get("query", ""))
                kb_res = self.service.kb.answer_query(query_str, top_k=3)
                return 200, {"Content-Type": "application/json"}, kb_res

            return (
                404,
                {"Content-Type": "application/json"},
                {"error": "NOT_FOUND", "message": f"Route {method_up} {parsed_path} not found."},
            )

        except SecurityError as exc:
            return 403, {"Content-Type": "application/json"}, {"error": "SECURITY_ERROR", "message": str(exc)}
        except RateLimitExceededError as exc:
            return 429, {"Content-Type": "application/json"}, {"error": "RATE_LIMIT_EXCEEDED", "message": str(exc)}
        except CaseNotFoundError:
            # One fixed message for "not yours" and "does not exist". The store distinguishes them and
            # says so in the exception, which is useful in the server log; echoing it to the client let
            # anyone probe which case ids are live.
            return (
                404,
                {"Content-Type": "application/json"},
                {"error": "CASE_NOT_FOUND", "message": "No such case for this visitor session."},
            )
        except WorkflowGateError as exc:
            return 409, {"Content-Type": "application/json"}, {"error": "WORKFLOW_GATE_LOCKED", "message": str(exc)}
        except ConfirmationError as exc:
            return 400, {"Content-Type": "application/json"}, {"error": "CONFIRMATION_ERROR", "message": str(exc)}
        except ValueError as exc:
            return 400, {"Content-Type": "application/json"}, {"error": "VALIDATION_ERROR", "message": str(exc)}


class ColdSenseTestClient:
    """Ergonomic test client for invoking ColdSense REST API routes with cookie/CSRF management."""

    def __init__(self, service: Optional[ColdSenseService] = None) -> None:
        self.dispatcher = ColdSenseAPIDispatcher(service=service)
        self.signed_cookie: Optional[str] = None
        self.csrf_token: Optional[str] = None
        self.origin: str = "http://localhost:3000"

    def init_session(self) -> Dict[str, Any]:
        status, _hdrs, data = self.dispatcher.handle_request("GET", "/api/session")
        assert status == 200
        self.signed_cookie = data["signed_cookie"]
        self.csrf_token = data["csrf_token"]
        return data

    def request(
        self,
        method: str,
        path: str,
        json_body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        req_headers: Dict[str, str] = {}
        if self.signed_cookie:
            req_headers["Cookie"] = f"coldsense_session={self.signed_cookie}"
        if self.csrf_token and method.upper() in ("POST", "PUT", "PATCH", "DELETE"):
            req_headers["X-CSRF-Token"] = self.csrf_token
            req_headers["Origin"] = self.origin
        if headers:
            req_headers.update(headers)
        status, _resp_hdrs, resp_json = self.dispatcher.handle_request(
            method, path, headers=req_headers, body=json_body
        )
        return status, resp_json


class ColdSenseHTTPServer:
    """Full-stack HTTP server serving both the ColdSense UI (`src/coldsense/web/`) and `/api/*`."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        service: Optional[ColdSenseService] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.dispatcher = ColdSenseAPIDispatcher(service=service)
        # Ensure dynamic server port is in allowed_origins
        self.dispatcher.service.security.allowed_origins.update(
            {
                f"http://{host}:{port}",
                f"http://localhost:{port}",
                f"http://127.0.0.1:{port}",
            }
        )
        self._httpd: Optional[ThreadingHTTPServer] = None

    def create_server(self) -> ThreadingHTTPServer:
        dispatcher = self.dispatcher

        class Handler(BaseHTTPRequestHandler):
            def _serve_static(self, req_path: str) -> None:
                clean = urlparse(req_path).path
                if clean in ("/", "/index.html"):
                    target = WEB_DIR / "index.html"
                else:
                    rel = clean.lstrip("/")
                    target = (WEB_DIR / rel).resolve()
                    if not str(target).startswith(str(WEB_DIR.resolve())):
                        self.send_error(403, "Forbidden")
                        return
                if not target.exists() or not target.is_file():
                    # SPA fallback to index.html for client-side routes
                    target = WEB_DIR / "index.html"
                if not target.exists():
                    self.send_error(404, "Frontend file not found")
                    return
                ctype, _ = mimetypes.guess_type(str(target))
                raw = target.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", f"{ctype or 'text/html'}; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self._send_security_headers()
                self.end_headers()
                self.wfile.write(raw)

            def _send_security_headers(self) -> None:
                """Baseline hardening on every response: the app serves HTML and JSON from one origin."""
                for name, value in SECURITY_HEADERS.items():
                    self.send_header(name, value)

            def _send_json(self, status: int, body: Dict[str, Any]) -> None:
                payload_bytes = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload_bytes)))
                self._send_security_headers()
                self.end_headers()
                self.wfile.write(payload_bytes)

            def _dispatch_api(self, method: str) -> None:
                # The length is checked before any byte is read: an unbounded or absent length would
                # otherwise let one request pull the whole socket into memory.
                raw_length = self.headers.get("Content-Length")
                if method.upper() in ("POST", "PUT", "PATCH"):
                    if raw_length is None or not str(raw_length).strip().isdigit():
                        self._send_json(400, {
                            "error": "INVALID_CONTENT_LENGTH",
                            "message": "a POST needs a Content-Length header holding a non-negative integer.",
                        })
                        return
                    length = int(raw_length)
                    if length > UPLOAD_MAX_BYTES:
                        self._send_json(413, {
                            "error": "UPLOAD_TOO_LARGE",
                            "message": f"request body is {length} bytes; the limit is {UPLOAD_MAX_BYTES} bytes "
                                       f"({UPLOAD_MAX_BYTES // (1024 * 1024)} MiB).",
                        })
                        return
                else:
                    text = str(raw_length or "0").strip()
                    length = int(text) if text.isdigit() else 0
                raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else ""
                try:
                    body_dict = json.loads(raw_body) if raw_body.strip() else {}
                except ValueError:
                    self._send_json(400, {"error": "INVALID_JSON", "message": "request body is not valid JSON."})
                    return
                hdrs = {k: v for k, v in self.headers.items()}
                status, resp_hdrs, resp_body = dispatcher.handle_request(
                    method, self.path, headers=hdrs, body=body_dict
                )
                payload_bytes = json.dumps(resp_body).encode("utf-8")
                self.send_response(status)
                for hk, hv in resp_hdrs.items():
                    self.send_header(hk, hv)
                self.send_header("Content-Length", str(len(payload_bytes)))
                self._send_security_headers()
                self.end_headers()
                self.wfile.write(payload_bytes)

            def do_GET(self) -> None:
                if self.path.startswith("/api/"):
                    self._dispatch_api("GET")
                else:
                    self._serve_static(self.path)

            def do_POST(self) -> None:
                self._dispatch_api("POST")

            def log_message(self, format: str, *args: Any) -> None:
                return

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        return self._httpd


def create_fastapi_app(service: Optional[ColdSenseService] = None) -> Any:
    """Create a FastAPI application if `fastapi` is installed in the environment."""
    try:
        from fastapi import FastAPI, Request, Response
        from fastapi.responses import FileResponse
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI is not installed in this Python environment. Use ColdSenseHTTPServer or install fastapi."
        ) from exc

    app = FastAPI(title="ColdSense (SmartClim) API", version="1.1.0")
    dispatcher = ColdSenseAPIDispatcher(service=service)

    @app.api_route("/api/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def proxy_all(request: Request, full_path: str) -> Response:
        raw_bytes = await request.body()
        body_dict = json.loads(raw_bytes.decode("utf-8")) if raw_bytes else {}
        hdrs = dict(request.headers)
        status, resp_hdrs, resp_json = dispatcher.handle_request(
            request.method,
            f"/api/{full_path}",
            headers=hdrs,
            body=body_dict,
        )
        return Response(
            content=json.dumps(resp_json),
            status_code=status,
            headers=resp_hdrs,
            media_type="application/json",
        )

    @app.get("/{static_path:path}")
    async def serve_spa(static_path: str) -> Any:
        target = (WEB_DIR / static_path).resolve() if static_path else (WEB_DIR / "index.html")
        if not str(target).startswith(str(WEB_DIR.resolve())) or not target.is_file():
            target = WEB_DIR / "index.html"
        return FileResponse(str(target))

    return app
