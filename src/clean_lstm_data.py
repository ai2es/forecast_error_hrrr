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
    fh = int(fh)

    if month == 1:
        start_dt = datetime(year - 1, 12, 1)
    else:
        start_dt = datetime(year, month - 1, 1)

    end_day = min(day + 1, monthrange(year, month)[1])
    end_dt = datetime(year, month, end_day, 23, 59, 59)

    print(f"[INFO] Running forecast hour {fh} from {start_dt} to {end_dt}")
    forecast_hour_parquet_builder.main(start_dt, end_dt, fh)
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
