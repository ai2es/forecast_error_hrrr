import sys

sys.path.append("..")

from TRAINING import engine_lstm_training

from datetime import datetime

def main():
    engine_lstm_training.main()
    print("Training Finished")
    print(datetime.datetime.now())
    print(" -- Closing Training --")


if __name__ == "__main__":
    main()

