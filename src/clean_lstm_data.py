"""Refresh cleaned HRRR + NYSM parquet files for a given forecast hour.

Two stages run for the requested forecast hour `fh`:

1. `forecast_hr_parquet_builder.main(start, end, fh)`
   - Reads the raw per-init-time HRRR parquets for the requested
     window and stitches them into a single `valid_time`-indexed
     parquet for that forecast hour.
2. `all_models_comparison_to_mesos_lstm.main(month, year, model, fh)`
   - Cross-references the freshly built HRRR parquet against the
     NYSM observations and emits the cleaned, NYSM-aligned
     forecast/observation parquet that the LSTM data prep consumes.

Date window
-----------
Given a `(year, month, day)`:

    start = first day of the previous month  (or Dec 1 of `year - 1`
                                              when `month == 1`)
    end   = `min(day + 1, last day of `month`)` at 23:59:59

This rolling window keeps the most recent ~30 days of cleaned data
fresh, which is what `prepare_lstm_data*.py` expects when it pulls
the past 30 hours at inference time.

CLI
---
    python clean_lstm_data.py --fh 6 --year 2025 --month 4 --day 30 --model hrrr
"""

import argparse
import sys
import time
from calendar import monthrange
from datetime import datetime

import numpy as np

sys.path.append("..")

from data_cleaning import (
    all_models_comparison_to_mesos_lstm,
    forecast_hr_parquet_builder,
    get_resampled_nysm_data,
)


def run_forecast_hour_tasks(fh, year, month, day, model):
    """Run the two cleaning steps for a single forecast hour.

    Parameters
    ----------
    fh : int
        HRRR forecast hour (1..18 in the supported configurations).
    year, month, day : int
        Anchor date used to derive the rolling cleaning window.
    model : str
        Free-form NWP model identifier passed through to the cleaning
        helpers (only "hrrr" is currently supported by the inference
        pipeline).
    """
    fh = int(fh)
    start_time = time.time()

    # Build the cleaning window: start = first day of previous month,
    # end = `day + 1` (clamped to the last valid day of `month`).
    if month == 1:
        start_dt = datetime(int(year - 1), 12, 1)
    else:
        start_dt = datetime(year, int(month - 1), 1)

    end_day = min(int(day + 1), monthrange(year, month)[1])
    end_dt = datetime(year, month, end_day, 23, 59, 59)

    print(f"[INFO] Running forecast hour {fh} from {start_dt} to {end_dt}")
    t0 = time.time()
    print(f"Setup took {time.time() - start_time:.2f} seconds")

    # Step 1: stitch raw HRRR parquets into a `valid_time`-indexed
    # parquet for this forecast hour.
    forecast_hr_parquet_builder.main(start_dt, end_dt, fh)
    t1 = time.time()
    print(f"Parquet builder took {time.time() - t0:.2f} seconds")

    # Step 2: align the freshly built HRRR parquet with the NYSM
    # observation parquets for the requested month/year.
    all_models_comparison_to_mesos_lstm.main(
        str(month).zfill(2), year, model, str(fh).zfill(2)
    )
    print(f"All models comparison took {time.time() - t1:.2f} seconds")


def main(now, fh_range=range(1, 19), model="hrrr"):
    """Refresh cleaned input parquets for every forecast hour.

    Designed to be called from `pipeline.py` once an hour.  The current
    wall-clock `datetime` is passed in; the cleaning window is derived
    from `(now.year, now.month, now.day)`.

    Parameters
    ----------
    now : datetime.datetime
        Anchor timestamp (typically the current wall-clock time).
    fh_range : iterable of int
        Forecast hours to refresh (default: 1..18).
    model : str
        NWP model identifier (default: "hrrr").
    """
    for fh in fh_range:
        try:
            run_forecast_hour_tasks(fh, now.year, now.month, now.day, model)
        except Exception as exc:
            print(f"[clean_lstm_data] fh={fh} failed: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run forecast hour tasks for a given date and model."
    )
    parser.add_argument("--fh", type=int, required=True, help="Forecast hour")
    parser.add_argument("--year", type=int, required=True, help="Year")
    parser.add_argument("--month", type=int, required=True, help="Month")
    parser.add_argument("--day", type=int, required=True, help="Day")
    parser.add_argument(
        "--model", type=str, required=True, help="Model name (e.g. 'hrrr')"
    )

    args = parser.parse_args()
    print(args.fh, args.year, args.month, args.day, args.model)
    run_forecast_hour_tasks(args.fh, args.year, args.month, args.day, args.model)
