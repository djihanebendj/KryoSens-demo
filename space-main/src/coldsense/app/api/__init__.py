"""Phase 9 HTTP service: a thin FastAPI layer over the accepted Phase 2-5 results (docs/PHASE9_ARCHITECTURE.md).

Run locally:  uvicorn coldsense.app.api:create_app --factory --host 127.0.0.1 --port 8000
The service is meant to sit behind the Next.js same-origin proxy; it serves no CORS headers.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request

from coldsense import __version__
from coldsense.settings import get_settings

from .config import ResolvedApiConfig, load_api_config
from .errors import install_handlers
from .knowledge import KnowledgeService
from .ratelimit import RateLimiter
from .request_guards import install_request_guards
from .routes import router
from .service import AnalysisSlots
from .session import COOKIE_NAME, SessionSigner
from .store import CaseStore, InMemoryCaseStore


def create_app(config: ResolvedApiConfig | None = None, *, store: CaseStore | None = None,
               signer: SessionSigner | None = None, slots: AnalysisSlots | None = None,
               samples_dir: str | Path | None = None, rate_limiter: RateLimiter | None = None,
               knowledge_db: str | Path | None = None) -> FastAPI:
    config = config or load_api_config()
    docs = config.serve_docs
    app = FastAPI(title="ColdSense API", version=__version__,
                  description="Evidence-first cold-room decision support. Refusals are results, not errors.",
                  openapi_url="/api/v1/openapi.json" if docs else None,
                  docs_url="/api/v1/docs" if docs else None, redoc_url=None)
    app.state.config = config
    app.state.store = store if store is not None else InMemoryCaseStore()   # an empty store is falsy
    app.state.signer = signer if signer is not None else SessionSigner(config.secret)
    app.state.slots = slots if slots is not None else AnalysisSlots()
    app.state.samples_dir = Path(samples_dir) if samples_dir is not None else get_settings().data_dir / "sample"
    app.state.limiter = rate_limiter if rate_limiter is not None else RateLimiter()
    app.state.knowledge = KnowledgeService(knowledge_db if knowledge_db is not None else config.knowledge_db)

    def clear_cookie(resp) -> None:
        resp.delete_cookie(COOKIE_NAME, path="/", secure=config.cookie_secure, httponly=True, samesite="lax")

    install_handlers(app, clear_cookie=clear_cookie)
    app.include_router(router)
    install_request_guards(app)          # size/time guard; added before _headers so _headers wraps its responses

    @app.middleware("http")
    async def _headers(request: Request, call_next):
        resp = await call_next(request)
        resp.headers.setdefault("Cache-Control", "no-store")
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        return resp

    return app


__all__ = ["create_app"]
