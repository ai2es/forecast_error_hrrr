"""Top-level entry point for *training* the LSTM models.

This is the training counterpart to `pipeline.py` (which runs
inference).  It delegates to `TRAINING.engine_lstm_training.main`,
which iterates over climate divisions / stations / variables /
forecast hours and trains one encoder-decoder per combination.

Usage
-----
    python training_pipeline.py

All hyper-parameters and date ranges are set inside
`TRAINING/engine_lstm_training.py`'s `__main__` block.
"""

import sys

sys.path.append("..")

from datetime import datetime

from TRAINING import engine_lstm_training


def main():
    engine_lstm_training.main()
    print("Training Finished")
    print(datetime.now())
    print(" -- Closing Training --")


if __name__ == "__main__":
    main()
