import sys

sys.path.append("..")

# imports
import datetime

from data_cleaning import (
    get_resampled_nysm_data,
    all_models_comparison_to_mesos_lstm,
    forecast_hour_parquet_builder,
)


def main(now):
    """
    GET CURRENT LOCAL TIME TO RUN INFERENCE
    """
    year = now.year
    month = now.month
    day = now.day

    get_resampled_nysm_data.main(year, month)  # clean nysm data

    # clean nwp data
    model = "hrrr"
    for fh in np.arange(1, 19):
        forecast_hour_parquet_builder.main(
            datetime(int(year - 1), int(month - 1), 1, 0, 0, 0),
            datetime(year, month, int(day + 1), 23, 59, 59),
            fh,
        )
        all_models_comparison_to_mesos_lstm.main(
            str(month).zfill(2), year, model, str(fh).zfill(2)
        )


if __name__ == "__main__":
    main()
