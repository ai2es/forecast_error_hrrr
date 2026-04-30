"""Stitch per-init-time HRRR parquets into one parquet per `valid_time`.

Raw HRRR forecast parquets are produced by the upstream ingest as
``{savedir}{year}/{month}/{YYYYMMDD}_hrrr.t{init}z_{fh}.parquet`` -
one file per (init time, forecast hour).  For training and inference
we want a single parquet per day where each row is keyed by
`valid_time = init_time + fh`.

`main(start_date, end_date, fh)` iterates day by day across the date
window, picks the row with the matching `valid_time` from each of
the 24 init-time files, and concatenates them into a single per-day
parquet at:

    /home/aevans/ai2es/lstm/HRRR/fh_{fh}/{year}/{month}/{YYYYMMDD}_hrrr_fh{fh}.parquet

Run via `clean_lstm_data.py` (preferred) or directly from the CLI.
"""

import argparse
import os
from datetime import datetime, timedelta

import cudf


def make_dirs(year, month, day, fh):
    """Create the per-day output directory if it doesn't already exist."""
    base_path = f"/home/aevans/ai2es/lstm/HRRR/fh_{fh}/{year}/{month}/"
    os.makedirs(base_path, exist_ok=True)


def main(start_date, end_date, fh):
    """Build one stitched parquet per day in `[start_date, end_date]`.

    Parameters
    ----------
    start_date, end_date : datetime.datetime
        Inclusive day range to process.
    fh : int or str
        HRRR forecast hour (1..18).  Will be zero-padded internally.
    """
    savedir = "/home/aevans/ai2es/cleaned/HRRR/"
    delta = timedelta(days=1)
    fh = str(fh).zfill(2)

    while start_date <= end_date:
        the_df = cudf.DataFrame()
        my_date = start_date
        my_time = my_date + timedelta(hours=int(fh))

        # Walk the 24 init times for this day; for each, take the
        # single row whose valid_time matches our target.
        for i in range(24):
            init = str(i).zfill(2)
            month = my_date.strftime("%m")
            year = my_date.strftime("%Y")
            day = my_date.strftime("%d")
            my_time_str = str(my_time)

            file_path = (
                f"{savedir}{year}/{month}/"
                f"{year}{month}{day}_hrrr.t{init}z_{fh}.parquet"
            )

            if not os.path.exists(file_path):
                continue

            try:
                df = cudf.read_parquet(file_path).reset_index()
            except Exception:
                print(f"Failed to open {file_path}")
                start_date += delta
                continue

            new_df = df[df["valid_time"] == my_time_str]
            the_df = cudf.concat([new_df, the_df])

            my_time += timedelta(hours=1)

        start_date += delta
        make_dirs(year, month, day, fh)

        # The accumulator was built bottom-up; reverse to chronological.
        the_df = the_df[::-1]
        out_path = (
            f"/home/aevans/ai2es/lstm/HRRR/fh_{fh}/{year}/{month}/"
            f"{year}{month}{day}_hrrr_fh{fh}.parquet"
        )
        the_df.to_parquet(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--time1",
        type=datetime.fromisoformat,
        required=True,
        help="Inclusive start date (ISO-8601)",
    )
    parser.add_argument(
        "--time2",
        type=datetime.fromisoformat,
        required=True,
        help="Inclusive end date (ISO-8601)",
    )
    parser.add_argument(
        "--fh", type=int, required=True, help="HRRR forecast hour (1..18)"
    )
    args = parser.parse_args()

    main(args.time1, args.time2, args.fh)
