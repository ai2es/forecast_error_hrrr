# run_one_fh.py

import sys
from datetime import datetime
from data_cleaning import (
    get_resampled_nysm_data,
    all_models_comparison_to_mesos_lstm,
    forecast_hour_parquet_builder,
)

def run_forecast_hour_tasks(fh, year, month, day, model):
    fh = int(fh)
    start_dt = datetime(year - 1, month - 1 or 12, 1)
    end_dt = datetime(year, month, min(day + 1, 28), 23, 59, 59)

    print(f"[INFO] Running forecast hour {fh} from {start_dt} to {end_dt}")
    forecast_hour_parquet_builder.main(start_dt, end_dt, fh)
    all_models_comparison_to_mesos_lstm.main(
        str(month).zfill(2), year, model, str(fh).zfill(2)
    )

if __name__ == "__main__":
    fh = sys.argv[1]
    year = int(sys.argv[2])
    month = int(sys.argv[3])
    day = int(sys.argv[4])
    model = sys.argv[5]

    # Clean NYSM data only once — you could move this elsewhere for batch-wide reuse
    get_resampled_nysm_data.main(year, month)

    run_forecast_hour_tasks(fh, year, month, day, model)