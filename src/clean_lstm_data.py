import sys

sys.path.append("..")
from datetime import datetime
from calendar import monthrange
import numpy as np

from data_cleaning import (
    get_resampled_nysm_data,
    all_models_comparison_to_mesos_lstm,
    forecast_hr_parquet_builder,
)
import time
import argparse


def run_forecast_hour_tasks(fh, year, month, day, model):
    # Ensure the forecast hour is treated as an integer
    fh = int(fh)
    start_time = time.time()

    # Determine the start date:
    # If the current month is January, the start date is December 1 of the previous year
    if month == 1:
        start_dt = datetime(int(year - 1), 12, 1)
    else:
        # Otherwise, use the 1st day of the previous month in the same year
        start_dt = datetime(year, int(month - 1), 1)

    # Determine the end date:
    # The end day is either the next day or the last day of the current month, whichever is smaller
    end_day = min(int(day + 1), monthrange(year, month)[1])
    # Set end datetime to the end of that day (23:59:59)
    end_dt = datetime(year, month, end_day, 23, 59, 59)

    # Log the time range for this forecast hour's processing
    print(f"[INFO] Running forecast hour {fh} from {start_dt} to {end_dt}")
    t0 = time.time()
    print(f"Setup took {time.time() - start_time:.2f} seconds")
    # Run the function to build forecast hour data from Parquet files
    forecast_hr_parquet_builder.main(start_dt, end_dt, fh)
    t1 = time.time()
    print(f"Parquet builder took {time.time() - t0:.2f} seconds")

    # Run the comparison between all models and the LSTM mesoscale model
    # Ensures that month and forecast hour are zero-padded to two digits
    all_models_comparison_to_mesos_lstm.main(
        str(month).zfill(2), year, model, str(fh).zfill(2)
    )
    t2 = time.time()
    print(f"All models comparison took {time.time() - t1:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run forecast hour tasks for a given date and model."
    )
    parser.add_argument("--fh", type=int, required=True, help="Forecast hour")
    parser.add_argument("--year", type=int, required=True, help="Year")
    parser.add_argument("--month", type=int, required=True, help="Month")
    parser.add_argument("--day", type=int, required=True, help="Day")
    parser.add_argument(
        "--model", type=str, required=True, help="Model name (e.g., hrrr)"
    )

    args = parser.parse_args()

    print(args.fh, args.year, args.month, args.day, args.model)
    run_forecast_hour_tasks(args.fh, args.year, args.month, args.day, args.model)
