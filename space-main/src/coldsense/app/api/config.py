"""Deployment settings of the HTTP service (prefix ``COLDSENSE_API_``), read once at start-up.

Only deployment concerns live here: the signing secret, cookie flags, docs, the Origin allowlist and the
provenance-trust switch. Operational limits are code constants (``limits.py``); refusal rules are never
configurable (D-002). Nothing here can be changed by a request: the resolved config is fixed for the process.

Provenance trust (amendment G): ``COLDSENSE_API_TRUST_DECLARED_PROVENANCE`` defaults to OFF. It may only ever apply to
a separately authenticated, operator-controlled ingestion path. Phase 9 has no such path, so the setting has no
effect on any upload: anonymous uploads are always analysed as UNVALIDATED, whatever is declared or configured.
The copilot has a server-side gate (spec §6, amendment A). It is disabled by default and becomes available only
when deployment configuration, an LLM provider and a verified Phase 8 acceptance record for the Rule 7 hardening
are all present. No acceptance record exists in this repository, so the public deployment stays disabled.
"""

from __future__ import annotations

import logging
import secrets
from pathlib import Path

from pydantic import SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from coldsense.core.errors import ConfigurationError
from coldsense.settings import get_settings

log = logging.getLogger(__name__)

DEV_ORIGINS = ("http://localhost:3000", "http://127.0.0.1:3000")

# The acceptance record is read, never created, by the application (docs/PHASE9_DEPLOYMENT.md).
DEFAULT_PHASE8_ACCEPTANCE = Path(__file__).resolve().parents[4] / "docs" / "PHASE8_ACCEPTANCE.json"
PROVIDER_ENV = "ANTHROPIC_API_KEY"


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COLDSENSE_API_", env_file=".env", env_file_encoding="utf-8",
                                      extra="ignore", frozen=True)

    session_secret: SecretStr | None = None   # required when COLDSENSE_ENV=prod; >= 32 characters
    cookie_secure: bool | None = None         # default: True in prod, False otherwise (plain-HTTP localhost)
    serve_docs: bool | None = None            # /api/v1/docs and /api/v1/openapi.json; default: off in prod
    allowed_origins: str | None = None        # comma-separated, e.g. "https://coldsense.example"; required in prod
    trust_declared_provenance: bool = False   # operator ingestion only; no such path exists, so no effect (above)
    enable_copilot: bool = False              # explicit operator opt-in; still requires provider + acceptance record
    phase8_acceptance: str | None = None       # path to the verified Phase 8 acceptance record (read-only)
    knowledge_db: str | None = None            # Phase 7 SQLite store used by the Evidence retrieval panel


def normalize_origin(origin: str) -> str:
    return origin.strip().rstrip("/").lower()


class ResolvedApiConfig:
    def __init__(self, secret: bytes, cookie_secure: bool, serve_docs: bool, env: str, *,
                 allowed_origins: tuple[str, ...] = DEV_ORIGINS, trust_declared_provenance: bool = False,
                 enable_copilot: bool = False, phase8_acceptance_path: str | Path | None = None,
                 knowledge_db: str | Path | None = None) -> None:
        self.secret, self.cookie_secure, self.serve_docs, self.env = secret, cookie_secure, serve_docs, env
        self.allowed_origins = frozenset(normalize_origin(o) for o in allowed_origins if o.strip())
        self.trust_declared_provenance = trust_declared_provenance
        self.enable_copilot = enable_copilot
        self.phase8_acceptance_path = (Path(phase8_acceptance_path) if phase8_acceptance_path is not None
                                       else DEFAULT_PHASE8_ACCEPTANCE)
        self.knowledge_db = (Path(knowledge_db) if knowledge_db is not None
                             else get_settings().data_dir / "knowledge" / "kb.sqlite")

    def __setattr__(self, name, value):                 # fixed for the life of the process
        if name in self.__dict__:
            raise AttributeError(f"ResolvedApiConfig.{name} cannot be changed after start-up")
        super().__setattr__(name, value)


def load_api_config() -> ResolvedApiConfig:
    env = get_settings().env
    try:
        s = ApiSettings()
    except ValidationError as e:
        raise ConfigurationError(f"invalid ColdSense API settings:\n{e}") from e
    prod = env == "prod"
    if s.session_secret is not None:
        secret = s.session_secret.get_secret_value().encode()
        if len(secret) < 32:
            raise ConfigurationError("COLDSENSE_API_SESSION_SECRET must be at least 32 characters")
    elif prod:
        raise ConfigurationError("COLDSENSE_API_SESSION_SECRET is required when COLDSENSE_ENV=prod")
    else:
        secret = secrets.token_bytes(32)
        log.warning("no COLDSENSE_API_SESSION_SECRET: using a random per-process secret; sessions end on restart")
    if s.allowed_origins:
        origins = tuple(o for o in s.allowed_origins.split(",") if o.strip())
    elif prod:
        raise ConfigurationError("COLDSENSE_API_ALLOWED_ORIGINS is required when COLDSENSE_ENV=prod")
    else:
        origins = DEV_ORIGINS
    if s.trust_declared_provenance:
        log.warning("COLDSENSE_API_TRUST_DECLARED_PROVENANCE is set, but there is no operator ingestion path: it has "
                    "no effect and anonymous uploads stay UNVALIDATED")
    if s.enable_copilot:
        log.warning("COLDSENSE_API_ENABLE_COPILOT is set; the copilot still needs an LLM provider and a verified "
                    "Phase 8 acceptance record, and no acceptance record exists in this repository")
    return ResolvedApiConfig(secret=secret, cookie_secure=prod if s.cookie_secure is None else s.cookie_secure,
                             serve_docs=(not prod) if s.serve_docs is None else s.serve_docs, env=env,
                             allowed_origins=origins, trust_declared_provenance=s.trust_declared_provenance,
                             enable_copilot=s.enable_copilot,
                             phase8_acceptance_path=s.phase8_acceptance or DEFAULT_PHASE8_ACCEPTANCE,
                             knowledge_db=s.knowledge_db)
