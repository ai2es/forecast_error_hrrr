# -*- coding: utf-8 -*-
"""
Helpers for the per-window z-score normalization that the sequencer
performs.

The dataframe coming out of `prepare_lstm_data*` is intentionally left in
raw units.  The sequencer self-normalizes each (sequence_length +
forecast_steps) window using statistics computed on its OWN past
portion, then applies the architecture's NYSM persistence overwrite on
top.  The only thing this module provides is a list of which feature
columns get normalized vs. left alone (cyclic time encodings, geo /
categorical features, the target itself, and image references are all
left alone).
"""

from typing import Iterable, List

import numpy as np


# Substrings that mark a feature as already on a sensible scale.
SKIP_PREFIXES = (
    "valid_time_cos",
    "valid_time_sin",
    "valid_time_cos_clock",
    "valid_time_sin_clock",
    "lat",
    "lon",
    "elev",
    "lulc",
    "slope",
)
SKIP_EXACT = {"target_error"}


def _should_skip(col_name: str) -> bool:
    if col_name in SKIP_EXACT:
        return True
    if "images" in col_name:
        return True
    return any(sub in col_name for sub in SKIP_PREFIXES)


def get_normalize_mask(feature_names: Iterable[str]) -> np.ndarray:
    """Return a boolean numpy array, True for feature columns that the
    sequencer should z-score."""
    return np.asarray(
        [not _should_skip(c) for c in feature_names], dtype=bool
    )


def get_normalize_indices(feature_names: Iterable[str]) -> List[int]:
    """Same information as `get_normalize_mask` but as a list of
    integer column indices."""
    return [i for i, c in enumerate(feature_names) if not _should_skip(c)]
