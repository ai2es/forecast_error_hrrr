import pandas as pd
from datetime import datetime, timedelta, date
import numpy as np
import multiprocessing as mp
import os
import cudf  # RAPIDS cuDF


def make_dirs(year, month, day, fh):
    base_path = f"/home/aevans/ai2es/lstm/HRRR/fh_{fh}/{year}/{month}/"
    os.makedirs(base_path, exist_ok=True)


def main(start_date, end_date, fh):
    savedir = "/home/aevans/ai2es/cleaned/HRRR/"
    delta = timedelta(days=1)
    fh = str(fh).zfill(2)

    while start_date <= end_date:
        the_df = cudf.DataFrame()
        my_date = start_date
        my_time = my_date + timedelta(hours=int(fh))

        for i in range(24):
            init = str(i).zfill(2)
            month = my_date.strftime("%m")
            year = my_date.strftime("%Y")
            day = my_date.strftime("%d")
            my_time_str = str(my_time)

            file_path = (
                f"{savedir}{year}/{month}/{year}{month}{day}_hrrr.t{init}z_{fh}.parquet"
            )

            if not os.path.exists(file_path):
                continue

            try:
                df = cudf.read_parquet(file_path).reset_index()
            except Exception:
                print(f"Failed to open {file_path}")
                start_date += delta
                continue

            # Filter for the exact valid_time
            new_df = df[df["valid_time"] == my_time_str]
            the_df = cudf.concat([new_df, the_df])

            my_time += timedelta(hours=1)

        start_date += delta
        make_dirs(year, month, day, fh)

        # Reverse and save
        the_df = the_df[::-1]
        out_path = f"/home/aevans/ai2es/lstm/HRRR/fh_{fh}/{year}/{month}/{year}{month}{day}_hrrr_fh{fh}.parquet"
        the_df.to_parquet(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--time1",
        type=datetime,
        required=True,
        help="Datetime to START pulling data from",
    )
    parser.add_argument(
        "--time2",
        type=datetime,
        required=True,
        help="Datetime to STOP pulling data from",
    )
    parser.add_argument(
        "--fh", type=int, required=True, help="Forecast Hour-- to grab data for"
    )
    args = parser.parse_args()

    main(args.time1, args.time2, args.fh)
