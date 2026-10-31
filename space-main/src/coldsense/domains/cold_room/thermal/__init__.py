"""Phase 4: physics-aware thermal modelling and forecasting for the cold-room module.

Separate from the accepted Phase 1-3 code, which it consumes unchanged:
Phase 2 validation (analysis frame, report, row flags) and Phase 3 events
(gap-free blocks, recorded compressor runs). See docs/THERMAL_MODELS.md.
"""

from .engine import ENGINE_VERSION, forecast, forecast_source
from .registry import REGISTRY, audit_inputs
from .schemas import (
    CrossingStatus,
    Direction,
    ForecastRequest,
    ForecastResult,
    ForecastStatus,
    Provenance,
    RefusalCode,
    Regime,
)

__all__ = [
    "ENGINE_VERSION", "REGISTRY", "CrossingStatus", "Direction", "ForecastRequest", "ForecastResult",
    "ForecastStatus", "Provenance", "RefusalCode", "Regime", "audit_inputs", "forecast", "forecast_source",
]
