"""Compute the realised forecast-error column."""

import pandas as pd


# Map NWP forecast variable names (HRRR convention) to the matching
# NYSM observation column names.  Extend this dict as new target
# variables are added to the pipeline.
VARS_DICT = {
    "t2m": "tair",
    "mslma": "pres",
    "tp": "precip_total",
    "u_total": "wspd_sonic_mean",
}


def nwp_error(target, station, df):
    """Insert a `target_error` column equal to ``NWP - NYSM``.

    Parameters
    ----------
    target : str
        NWP variable name (e.g. ``t2m``, ``u_total``, ``tp``).
    station : str
        Station id whose suffixed columns to use (the wide dataframes
        produced by ``prepare_lstm_data*`` carry `_{station}` suffixes).
    df : pandas.DataFrame
        Wide HRRR + NYSM dataframe.  Must contain
        ``{target}_{station}`` and the matching NYSM column for the
        same station.

    Returns
    -------
    pandas.DataFrame
        `df` with `target_error = df[f"{target}_{station}"] -
        df[f"{nysm_var}_{station}"]` inserted near the front.

    Notes
    -----
    The model is trained to predict this raw error in physical units.
    It is intentionally never normalized so the model can learn the
    true error distribution.
    """
    nysm_var = VARS_DICT.get(target)
    if nysm_var is None:
        raise KeyError(
            f"No NYSM mapping registered for NWP variable {target!r}; "
            f"add it to `VARS_DICT` in model_data.get_error."
        )

    target_error = df[f"{target}_{station}"] - df[f"{nysm_var}_{station}"]
    df.insert(loc=1, column="target_error", value=target_error)
    return df
