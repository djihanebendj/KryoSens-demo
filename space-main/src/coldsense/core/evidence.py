"""Evidence classification. Shared by every domain module.

Every number ColdSense shows must carry one of these classes.
"""

from __future__ import annotations

from enum import Enum


class EvidenceClass(str, Enum):
    """Where a number came from. Never shown without one."""

    REAL = "REAL"                # measured on a physical unit
    SIMULATED = "SIMULATED"      # produced by a synthetic generator
    EXPECTED = "EXPECTED"        # a model output not yet checked against reality
    UNVALIDATED = "UNVALIDATED"  # provenance unknown (e.g. user upload, unlabelled)


def derived_evidence(input_class: EvidenceClass) -> EvidenceClass:
    """Evidence class of a model output computed from data of ``input_class``.

    A prediction is never REAL: even from real data it is EXPECTED until an
    outcome confirms it. Simulated inputs keep their SIMULATED label.
    """
    return EvidenceClass.SIMULATED if input_class == EvidenceClass.SIMULATED else EvidenceClass.EXPECTED
