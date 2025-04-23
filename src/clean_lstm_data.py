import sys
from datetime import datetime
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial

sys.path.append("..")

from data_cleaning import (
    get_resampled_nysm_data,
    all_models_comparison_to_mesos_lstm,
    forecast_hour_parquet_builder,
)


def run_forecast_hour_tasks(fh, start_dt, end_dt, year, month, day, model):
    # Step 1: Clean NWP data for this forecast hour
    forecast_hour_parquet_builder.main(start_dt, end_dt, fh)

    # Step 2: Collate model output with NYSM
    all_models_comparison_to_mesos_lstm.main(
        str(month).zfill(2), year, model, str(fh).zfill(2)
    )


def main(now):
    year = now.year
    month = now.month
    day = now.day

    # Step 0: Clean NYSM data once
    get_resampled_nysm_data.main(year, month)

    model = "hrrr"

    # Setup shared datetime values
    start_dt = datetime(year - 1, month - 1 or 12, 1)
    end_dt = datetime(year, month, min(day + 1, 28), 23, 59, 59)

    # Run each forecast hour in parallel
    with ProcessPoolExecutor() as executor:
        executor.map(
            partial(
                run_forecast_hour_tasks,
                start_dt=start_dt,
                end_dt=end_dt,
                year=year,
                month=month,
                day=day,
                model=model,
            ),
            range(1, 19),
        )


if __name__ == "__main__":
    main()