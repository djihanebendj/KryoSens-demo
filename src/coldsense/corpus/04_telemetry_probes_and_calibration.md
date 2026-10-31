# KB-CR-104: Telemetry Data Logger & Temperature Probe Calibration Standard

## Section 3.1: Probe Placement, Cadence, and Ice-Bath Calibration
Cold-room temperature loggers must record at a sampling cadence of 300 seconds (5 minutes) or faster. Probes must be mounted at least 50 cm away from door openings and outside the direct evaporator discharge airstream. Annual crushed ice-bath calibration at 0.0 C requires accuracy within plus or minus 0.5 C; probes drifting beyond 0.5 C must be recalibrated or replaced.

## Section 3.2: Data Quality & Reliability Gate Criteria
Automated thermal drift forecasting requires a minimum of 8 valid readings inside the drift window, a missing-data gap ratio of at most 30 percent, and a first-order model goodness-of-fit R-squared of at least 0.85. When sensor flatlines of 8 or more identical consecutive samples occur, the probe ADC buffer must be inspected before trusting model parameters.
