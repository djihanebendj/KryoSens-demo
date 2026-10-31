"""Domain-agnostic result vocabulary: issues, severities, verdicts.

A refusal is a normal *result* carrying reasons, never an exception.
Exceptions (see ``core.errors``) are reserved for genuine failures.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    BLOCKING = "BLOCKING"  # any blocking issue forces a refusal


class Issue(BaseModel):
    code: str
    severity: Severity
    message: str


class Verdict(str, Enum):
    USABLE = "USABLE"
    LIMITED = "LIMITED"   # usable with stated warnings
    REFUSED = "REFUSED"
