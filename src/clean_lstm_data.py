import sys
from datetime import datetime
from calendar import monthrange
import numpy as np

from data_cleaning import (
    get_resampled_nysm_data,
    all_models_comparison_to_mesos_lstm,
    forecast_hour_parquet_builder,
)


def run_forecast_hour_tasks(fh, year, month, day, model):
    # Ensure the forecast hour is treated as an integer
    fh = int(fh)

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

    # Run the function to build forecast hour data from Parquet files
    forecast_hour_parquet_builder.main(start_dt, end_dt, fh)

    # Run the comparison between all models and the LSTM mesoscale model
    # Ensures that month and forecast hour are zero-padded to two digits
    all_models_comparison_to_mesos_lstm.main(
        str(month).zfill(2), year, model, str(fh).zfill(2)
    )


if __name__ == "__main__":
    fh = int(sys.argsv[1])
    year = int(sys.argv[2])
    month = int(sys.argv[3])
    day = int(sys.argv[4])
    model = sys.argv[5]

    # Clean NYSM data only once
    get_resampled_nysm_data.main(year, month)
    run_forecast_hour_tasks(fh, year, month, day, model)
