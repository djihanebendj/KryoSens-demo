"""Domain-agnostic building blocks. Must not import from ``coldsense.domains``."""

from .errors import ColdSenseError, ConfigurationError, DataError, ModuleError
from .evidence import EvidenceClass, derived_evidence
from .governance import Hypothesis, HypothesisStatus, ModelCard
from .modules import DomainModule
from .results import Issue, Severity, Verdict

__all__ = [
    "ColdSenseError", "ConfigurationError", "DataError", "ModuleError",
    "EvidenceClass", "derived_evidence",
    "Hypothesis", "HypothesisStatus", "ModelCard",
    "DomainModule",
    "Issue", "Severity", "Verdict",
]
