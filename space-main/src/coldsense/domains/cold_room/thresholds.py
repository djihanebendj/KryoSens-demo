"""Every threshold that can make ColdSense refuse lives here, with its reason.

These are provisional engineering defaults, not validated constants. They were
checked against simulated logs only and have NOT been validated against real
logger exports. Change them only with a recorded reason; tests pin the
current behaviour.
"""

# --- plausibility of air temperature in a chilled/frozen room (°C) ---
PLAUSIBLE_MIN_C = -40.0   # below typical blast-freezer air; likely sensor fault
PLAUSIBLE_MAX_C = 60.0    # above any credible room air; likely sensor fault
MAX_STEP_C_PER_MIN = 5.0  # faster change than room air can do; spike / glitch
SPIKE_MIN_JUMP_C = 2.0    # ...and the jump must also be this large: at 1 s sampling a 0.1 °C
                          # logger rounding step reads as 6 °C/min, which is noise, not a spike.
                          # For intervals >= 24 s the rate limit already implies > 2 °C.

# --- measurement assumptions that a caller MAY override per file (recorded in
#     the report as ``limits_used``); the defaults below apply otherwise ---
UNDECLARED_UNIT_WARM_MEDIAN_C = 20.0  # undeclared unit + median above this: maybe °F
AMBIENT_PLAUSIBLE_MIN_C = -40.0
AMBIENT_PLAUSIBLE_MAX_C = 70.0        # machine-room air can be hot; still a sanity bound

# --- suspicious-but-possible readings (flagged, never excluded) ---
SUSPECT_WINDOW_MIN = 15       # rolling-median neighbourhood
SUSPECT_MIN_DEV_C = 1.5       # deviation below this is never called suspicious
SUSPECT_MAD_K = 8.0           # ...and it must also exceed K robust standard deviations

# --- timestamps and records ---
TS_FORMAT_MIN_SUCCESS = 0.5   # a timestamp format must parse at least half the rows
IRREGULAR_TOLERANCE = 0.5     # interval deviating > 50 % from the median is irregular
IRREGULAR_MAX_FRACTION = 0.10 # more irregular intervals than this -> warning
CONFLICT_DUP_MAX_FRACTION = 0.05  # conflicting duplicate timestamps above this -> refuse
                                  # (two sensors merged, or a clock change)

# --- sampling ---
GAP_FACTOR = 3.0          # a gap is > GAP_FACTOR x median interval
MAX_INTERVAL_S = 600      # coarser than 10 min cannot resolve off-cycles
MIN_DURATION_H = 6.0      # below this, too few cycles to characterise the unit
TARGET_DURATION_H = 24.0  # Wed 21 Oct checkpoint asks for 24 h
MIN_COVERAGE_PCT = 80.0   # below this, too much is missing
FLATLINE_MIN = 120        # identical readings this long suggest a stuck sensor

# --- warming-segment detection ---
SMOOTH_WINDOW_MIN = 5         # rolling median window
SEG_MIN_DURATION_MIN = 8.0    # shorter rises are noise / door events
SEG_MIN_RISE_C = 0.5
SEG_MIN_R2 = 0.80             # rise must be reasonably steady
SEG_MAX_RATE_C_PER_MIN = 1.0  # faster suggests door open / defrost, not off-cycle

# --- naive baseline prediction ---
MIN_CLEAN_SEGMENTS = 5        # Wed 21 Oct checkpoint: >= 5 clean off-segments
MAX_RATE_CV = 0.5             # warming rates too inconsistent -> refuse
MAX_EXTRAPOLATION_C = 5.0     # threshold this far beyond observed warming -> refuse

# --- event segmentation (Phase 3). PROVISIONAL defaults, chosen from engineering
#     judgement and checked on SIMULATED logs only; every one can be overridden per
#     call through SegmentationParams and the values used are recorded. ---
EVT_GAP_FACTOR = GAP_FACTOR      # same gap rule as validation, so both layers agree on what a gap is
EVT_SMOOTH_WINDOW_MIN = 5.0      # rolling-median window used only to find turning points;
                                 # removes logger rounding wiggles, keeps multi-minute behaviour
EVT_STABLE_BAND_C = 0.3          # a piece changing less than this is "stable": about 3x a 0.1 °C
                                 # logger resolution, so rounding noise alone cannot make a trend
EVT_PIECE_TOLERANCE_C = EVT_STABLE_BAND_C / 2  # max distance of the smoothed curve from its
                                 # piece's straight line before the piece is split
EVT_MIN_TREND_MIN = 5.0          # a rise/fall must last this long to be called sustained
EVT_MAX_TREND_RESID_C = 0.3      # rise/fall must follow a line this closely (residual std), else irregular
EVT_MIN_POINTS = 5               # fewer observations than this cannot be classified at all
EVT_MIN_EXCURSION_MIN = 10.0     # a threshold excursion shorter than this is "brief", longer is
                                 # "sustained". Product and regulation dependent: set it per site
EVT_COMPRESSOR_MIN_VALID = 0.9   # compressor events only if >= 90 % of usable rows carry a valid state
