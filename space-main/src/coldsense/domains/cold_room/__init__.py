"""Cold-room module: temperature-log analysis and time-to-threshold prediction.

Status: data foundation (ingest, quality gate, warming segments, naive
baseline). Built before the platform foundation; moved here unchanged.
"""

from coldsense.core.evidence import EvidenceClass
from coldsense.core.governance import ModelCard
from coldsense.core.modules import DomainModule, register

from .baseline import METHOD as BASELINE_METHOD
from .pipeline import AnalysisResult, analyze

NAIVE_BASELINE_CARD = ModelCard(
    name=BASELINE_METHOD,
    version="1",
    purpose="time-to-threshold after loss of cooling, from the unit's own observed warming rate",
    baseline="none: this IS the baseline every thermal model must beat",
    metric="absolute threshold-crossing time error (min and %) on held-out unplug tests",
    failure_modes=[
        "errs early far above the setpoint: constant rate ignores slowing as room nears ambient",
        "percentile range reflects cycle spread only, not extrapolation error",
        "refuses in rooms with frequent door openings (too few clean segments)",
    ],
    product_benefit="a transparent, explainable floor for predictions and the kill-criterion reference",
    validation_evidence=EvidenceClass.SIMULATED,
    validation_summary="checked on SIMULATED logs only; no REAL unplug test yet",
)

MODULE = register(DomainModule(
    key="cold_room",
    title="Cold-room temperature behaviour",
    version="0.1.0",
    description="Ingest and validate cold-room temperature logs; predict time to a threshold, or refuse.",
    model_cards=(NAIVE_BASELINE_CARD,),
))

__all__ = ["MODULE", "NAIVE_BASELINE_CARD", "AnalysisResult", "analyze"]
