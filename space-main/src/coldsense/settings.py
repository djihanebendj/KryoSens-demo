"""Deployment settings, read from environment variables (prefix ``COLDSENSE_``)
and an optional ``.env`` file.

What belongs here: where things are, how loudly to log, which environment
this is, and (later) secrets.

What does NOT belong here: refusal thresholds or model parameters. Those are
engineering decisions; letting an environment variable change them would let
ColdSense stop refusing without a test noticing.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from coldsense.core.errors import ConfigurationError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="COLDSENSE_", env_file=".env", env_file_encoding="utf-8",
        extra="ignore", frozen=True,
    )

    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"
    data_dir: Path = Path("data")
    real_data_dir: Path = Path("data/real")

    @field_validator("log_level")
    @classmethod
    def _valid_level(cls, v: str) -> str:
        v = v.upper()
        if v not in logging.getLevelNamesMapping():
            raise ValueError(f"unknown log level '{v}'")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load once per process. Call ``get_settings.cache_clear()`` in tests."""
    try:
        return Settings()
    except ValidationError as e:
        raise ConfigurationError(f"invalid ColdSense settings:\n{e}") from e
