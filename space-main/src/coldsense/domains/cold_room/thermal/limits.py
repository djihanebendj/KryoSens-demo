"""Every number that can make the thermal engine refuse or reject, with its reason.

PROVISIONAL engineering defaults, checked on SIMULATED data only; none has been
validated against real rooms. They live here, in versioned code, for the same
reason as ``domains/cold_room/thresholds.py`` (DECISIONS D-002): a value that
decides when ColdSense says "no" must be reviewed and covered by tests. They
are kept in this module rather than appended to ``thresholds.py`` so the
accepted Phase 1-3 files stay untouched (D-032).
"""

# --- fitting segments -----------------------------------------------------------
SEG_MIN_POINTS = 10            # fewer observations cannot separate a curve from noise
SEG_MIN_DURATION_MIN = 10.0    # shorter stretches carry almost no information on a rate
SEG_MIN_CHANGE_C = 0.5         # same as the Phase 1 warming-segment rule (SEG_MIN_RISE_C): a
                               # smaller change is dominated by 0.1 °C logger resolution and noise
SEG_END_POINTS = 3             # change is measured between the means of the first and last 3 readings

# --- measured ambient -----------------------------------------------------------
AMBIENT_MIN_COVERAGE = 0.9     # share of a segment's rows that must carry a valid ambient reading
AMBIENT_STEADY_RANGE_C = 1.0   # ambient range within a segment above which it is not "constant"
                               # and only the time-varying model may use it
AMBIENT_MAX_INTERP_FACTOR = 3.0  # missing ambient bridged by linear interpolation only across holes
                                 # up to this x median interval (the validation gap rule); else split

# --- parameter search and identifiability ---------------------------------------
K_MIN_SPAN_FACTOR = 1000.0     # slowest rate searched: time constant 1000 x the fitted time span.
                               # Slower is indistinguishable from a straight line; a fit that ends
                               # at this bound is not identified
PROFILE_GRID_POINTS = 241      # log-spaced rates evaluated before local refinement
PROFILE_LEVEL = 0.95           # profile-likelihood level used to judge identifiability (not a
                               # calibrated forecast uncertainty; see Phase 5)
IDENT_MAX_K_RATIO = 4.0        # upper/lower end of the rate's profile range: above this the time
                               # constant is not pinned down to within about a factor of two
FIT_MIN_R2 = 0.90              # within-segment R² below which the model does not describe the data
RESID_AUTOCORR_WARN = 0.5      # lag-1 residual autocorrelation above this -> systematic misfit warning

# --- evaluation and selection ---------------------------------------------------
SPLIT_FIT_FRACTION = 0.6       # chronological: first 60 % of eligible passive time fits,
SPLIT_SELECT_FRACTION = 0.2    # next 20 % selects the method, last 20 % is final evaluation
EVAL_HORIZONS_MIN = (5.0, 15.0, 30.0, 60.0, 120.0, 240.0, 480.0)
EVAL_ORIGIN_STRIDE_MIN = 5.0   # forecast origins every 5 min (or every reading, if coarser)
EVAL_MIN_CASES = 5             # common origin-horizon cases needed to compare methods
SELECTION_MARGIN_C = 0.02      # a more complex method is chosen over a simpler one only if its held-out
                               # MAE is lower by more than this: a fifth of the 0.1 °C logger resolution
ADEQUATE_ABS_ERROR_C = 0.5     # a forecast error above this is "not adequate": half of a typical
                               # 1 °C alarm margin. Applied to MAE (selection) and per case

# --- baselines ------------------------------------------------------------------
SLOPE_WINDOW_MIN = 15.0        # trailing window of the recent-slope baseline
SLOPE_MIN_POINTS = 5
SLOPE_MIN_SPAN_FRACTION = 0.5  # observations must span at least half the window

# --- requests -------------------------------------------------------------------
MAX_HORIZON_MIN = 7 * 24 * 60.0    # one week; longer is not a room-air forecast
MAX_TRAJECTORY_POINTS = 5000
