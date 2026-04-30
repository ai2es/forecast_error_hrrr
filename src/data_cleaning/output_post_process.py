"""Post-processing stub for inference output (placeholder).

This module is intentionally minimal: it is a placeholder for any
post-processing transforms (e.g. linear calibration, bias correction,
unit conversion) that should be applied to the model output parquets
*after* `lstm_s2s_engine.py` writes them.

The current calibration is handled inline inside
`lstm_s2s_engine.linear_transform` and reads from the per-`(climdiv,
metvar)` lookup CSVs under `MODELS/`.  Add new post-processing steps
to this file (or call them from `pipeline.py`) as the analytics
needs grow.
"""

import pandas as pd  # noqa: F401  (kept for downstream extension)
