"""Exception hierarchy.

Rule: raise only when something is genuinely broken (unreadable input,
bad configuration, a bug). When the data is readable but insufficient,
return a refusal result instead; the caller must be able to show *why*.
"""

from __future__ import annotations


class ColdSenseError(Exception):
    """Base class for every error ColdSense raises on purpose."""


class ConfigurationError(ColdSenseError):
    """Settings are missing or invalid."""


class DataError(ColdSenseError):
    """Input cannot be interpreted at all (not: input is poor quality)."""


class ModuleError(ColdSenseError):
    """A domain module is unknown, misregistered or failed to load."""
