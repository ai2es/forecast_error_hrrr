"""Top-level real-time inference pipeline (LSTM model).

Run this at the top of every wall-clock hour to:

1. Refresh the cleaned HRRR + NYSM parquets for the previous month so
   the model has up-to-date inputs (`clean_lstm_data.main`).
2. Score every NYSM station for every supported variable / forecast
   hour with the trained LSTM and write the predictions to disk
   (`lstm_s2s_engine.main`).

Usage
-----
    python pipeline.py

The current wall-clock time is used as the inference anchor (`now`).
All paths are resolved by the modules called here; see their
docstrings for the exact filesystem layout that is expected.
"""

import sys

sys.path.append("..")

import datetime
import time
import traceback

import clean_lstm_data
import lstm_s2s_engine


def main():
    now = datetime.datetime.now()

    # 1) Refresh cleaned input parquets so the model has the most
    #    recent HRRR forecasts and NYSM observations available on
    #    disk before we score.
    clean_lstm_data.main(now)

    # 2) Run inference for every (station, metvar, fh) combination
    #    and append predictions to the output parquets.
    lstm_s2s_engine.main(now)


if __name__ == "__main__":
    main()
