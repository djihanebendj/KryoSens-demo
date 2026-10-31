"""Phase 5 limits. PROVISIONAL defaults, documented in docs/RELIABILITY.md.

Phase 4 limits (limits.py) are reused unchanged; nothing here relaxes them.
"""

from __future__ import annotations

from . import limits as L

DEFAULT_INTERVAL_LEVEL = 0.80          # nominal level of the empirical interval: an evaluation target only
WILSON_Z = 1.959963984540054           # two-sided 95 % Wilson interval for an observed coverage
COVERAGE_PRECISION_FOR_HIGH = 0.15     # HIGH needs the coverage estimate known to +-0.15 (Wilson half-width)
MIN_INDEPENDENT_PERIODS = 2            # distinct passive periods needed in each of calibration and evaluation
MIN_PERIODS_FOR_TESTS = 3              # period means needed for the bias t-test and the dependence screen
BIAS_TEST_ALPHA = 0.05                 # two-sided t-test on PERIOD-MEAN held-out errors (n = periods, df = n - 1)
BIAS_MATERIAL_C = 0.10                 # ... and the period-weighted mean error must be at least this large (°C)
PERIOD_DEPENDENCE_ALPHA = 0.05         # one-sided screen: lag-1 autocorrelation of period means > z(1-a)/sqrt(n)
PARAMETER_ENVELOPE_LIMIT_C = L.ADEQUATE_ABS_ERROR_C   # parameter envelope wider than this at the horizon: material
SLOPE_SE_MULTIPLIER = 2.0              # recent-slope envelope: slope +- 2 standard errors
ABS_QUANTILES = (0.5, 0.8, 0.9, 0.95)
SIGNED_QUANTILES = (0.05, 0.1, 0.5, 0.9, 0.95)

# Phase 2 warnings that bear on forecast reliability (others: SHORT_LOG is covered by the evidence counts,
# NO_AMBIENT by model eligibility, TIMEZONE_UNKNOWN is disclosed)
MATERIAL_QUALITY_ISSUES = frozenset({
    "INVALID_RECORDS", "MALFORMED_ROWS", "TIMESTAMPS_OUT_OF_ORDER", "DUPLICATE_TIMESTAMPS", "IMPLAUSIBLE_VALUES",
    "AMBIENT_INVALID", "COMPRESSOR_INVALID", "GAPS", "IRREGULAR_SAMPLING", "FLATLINE", "SUSPICIOUS_VALUES",
    "UNIT_SUSPECT",
})
