import sys

sys.path.append("..")

import clean_lstm_data
import lstm_s2s_engine
import datetime
import time
import traceback


def main():
    # get time for inference
    now = datetime.datetime.now()

    # clean nysm + nwp data
    clean_lstm_data.main(now)

    # run models
    lstm_s2s_engine.main(now)


if __name__ == "__main__":
    main()
