"""Cyclic time encodings (sin / cos) for `valid_time`.

Adds four columns to a dataframe:

* `{col}_sin`, `{col}_cos`         - day-of-year on a `max_val` cycle
* `{col}_sin_clock`, `{col}_cos_clock` - second-of-day on a 24h cycle

Encoding both with sin and cos lets a model learn smooth, continuous
seasonal / diurnal patterns without the discontinuity at year/day
boundaries that a raw integer day-of-year would impose.

These columns are intentionally listed in
`model_data.normalization.SKIP_PREFIXES` so the per-window z-score in
the sequencer leaves them untouched (they're already on `[-1, 1]`).
"""

import numpy as np


def encode(data, col, max_val):
    """Add sin/cos cyclic encodings of `data['valid_time']`.

    Parameters
    ----------
    data : pandas.DataFrame
        Must contain a datetime column named ``valid_time``.
    col : str
        Prefix for the new columns (typically ``"valid_time"``).
    max_val : int
        Period for the day-of-year encoding (use 366 to handle leap
        years gracefully).

    Returns
    -------
    pandas.DataFrame
        The input dataframe with four new columns inserted at the
        front and the temporary helper columns dropped.
    """
    # Day-of-year encoding (annual cycle).
    data["day_of_year"] = data["valid_time"].dt.dayofyear
    sin = np.sin(2 * np.pi * data["day_of_year"] / max_val)
    data.insert(loc=0, column=f"{col}_sin", value=sin)
    cos = np.cos(2 * np.pi * data["day_of_year"] / max_val)
    data.insert(loc=0, column=f"{col}_cos", value=cos)
    data = data.drop(columns=["day_of_year"])

    # Time-of-day encoding (diurnal cycle).
    seconds_in_day = 24 * 60 * 60
    data["seconds_of_day"] = (
        data["valid_time"].dt.hour * 3600
        + data["valid_time"].dt.minute * 60
        + data["valid_time"].dt.second
    )
    sin = np.sin(2 * np.pi * data["seconds_of_day"] / seconds_in_day)
    data.insert(loc=0, column=f"{col}_sin_clock", value=sin)
    cos = np.cos(2 * np.pi * data["seconds_of_day"] / seconds_in_day)
    data.insert(loc=0, column=f"{col}_cos_clock", value=cos)
    data = data.drop(columns=["seconds_of_day"])

    return data
