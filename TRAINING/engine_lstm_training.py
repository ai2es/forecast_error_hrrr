import sys

sys.path.append(".")

from comet_ml import Experiment, Artifact
from comet_ml.integration.pytorch import log_model

import os
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pandas as pd
import numpy as np
import gc
from datetime import datetime
import random

from model_architecture import encode_decode_lstm, sequencer
from model_data import nysm_data, hrrr_data, prepare_lstm_data


print("imports loaded")


def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None  # Return None if the batch is empty
    return torch.utils.data.default_collate(batch)

def date_filter(ldf, time1, time2):
    ldf = ldf[ldf["valid_time"] > time1]
    ldf = ldf[ldf["valid_time"] < time2]

    return ldf

class EarlyStopper:
    def __init__(self, patience, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class OutlierFocusedLoss(nn.Module):
    def __init__(self, alpha, device):
        super(OutlierFocusedLoss, self).__init__()
        self.alpha = alpha
        self.device = device

    def forward(self, y_pred, y_true):
        y_true = y_true.to(self.device)
        y_pred = y_pred.to(self.device)

        # Calculate the error
        error = y_true - y_pred

        # Calculate the base loss (Mean Absolute Error in this case)
        base_loss = torch.abs(error)

        # Apply a weighting function to give more focus to outliers
        weights = (torch.abs(error) + 1).pow(self.alpha)

        # Calculate the weighted loss
        weighted_loss = weights * base_loss

        # Return the mean of the weighted loss
        return weighted_loss.mean()


def get_model_file_size(file_path):
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    print(f"Model file size: {size_mb:.2f} MB")


def save_model_weights(model, encoder_path, decoder_path):
    torch.save(model.encoder.state_dict(), f"{encoder_path}")
    torch.save(model.decoder.state_dict(), decoder_path)


def main(
    start_time,
    end_time,
    batch_size,
    station,
    num_layers,
    epochs,
    weight_decay,
    fh,
    clim_div,
    nwp_model,
    exclusion_buffer,
    metvar,
    filtered_df,
    hrrr_df,
    sequence_length=30,
    target="target_error",
    learning_rate=5e-5,
    save_model=True,
):
    print("Am I using GPUS ???", torch.cuda.is_available())
    print("Number of gpus: ", torch.cuda.device_count())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(device)
    torch.manual_seed(101)
    print(" *********")
    print("::: In Main :::")

    filtered_df = date_filter(filtered_df, start_time, end_time)
    hrrr_df = date_filter(hrrr_df, start_time, end_time)

    decoder_path_og = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/s2s/{clim_div}/{clim_div}_{metvar}_{station}_decoder.pth"
    encoder_path_og = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/s2s/{clim_div}/{clim_div}_{metvar}_{station}_encoder.pth"

    decoder_path = (
        f"/home/aevans/inference/MODELS/{clim_div}_{metvar}_{station}_decoder.pth"
    )
    encoder_path = (
        f"/home/aevans/inference/MODELS/{clim_div}_{metvar}_{station}_encoder.pth"
    )

    # prepare data for LSTM
    (lstm_df, features, stations, target, valid_times) = (
        prepare_lstm_data.prepare_lstm_data(filtered_df, hrrr_df, train=True)
    )

    (
        df_train,
        features,
        stations,
        target,
        vt,
    ) = create_data_for_lstm.create_data_for_model(
        station, fh, today_date, metvar
    )  # to change which model you are matching for you need to chage which
    print("FEATURES", features)
    print()
    print("TARGET", target)
    print()

    experiment = Experiment(
        api_key="leAiWyR5Ck7tkdiHIT7n6QWNa",
        project_name="inference_training",
        workspace="shmaronshmevans",
    )

    train_dataset = sequencer.SequenceDatasetMultiTask(
        dataframe=df_train,
        target=target,
        features=features,
        sequence_length=sequence_length,
        forecast_steps=fh,
        device=device,
        nwp_model=nwp_model,
        metvar=metvar,
    )

    train_kwargs = {
        "batch_size": batch_size,
        "pin_memory": False,
        "shuffle": True,
        "collate_fn": custom_collate,
    }

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    print("!! Data Loaders Succesful !!")

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    num_sensors = int(len(features))
    hidden_units = int(12 * len(features))

    # Initialize multi-task learning model with one encoder and decoders for each station
    model = encode_decode_lstm.ShallowLSTM_seq2seq_multi_task(
        num_sensors=num_sensors,
        hidden_units=hidden_units,
        num_layers=num_layers,
        mlp_units=1500,
        device=device,
        num_stations=len(stations),
    ).to(device)

    if fh == 1:
        if os.path.exists(encoder_path_og):
            print("Loading Encoder Model")
            model.encoder.load_state_dict(torch.load(encoder_path_og), strict=False)
            # Example usage for encoder and decoder
            get_model_file_size(encoder_path_og)

        if os.path.exists(decoder_path_og):
            print("Loading Decoder Model")
            model.decoder.load_state_dict(torch.load(decoder_path_og), strict=False)
            get_model_file_size(decoder_path_og)
    else:
        if os.path.exists(encoder_path):
            print("Loading Encoder Model")
            model.encoder.load_state_dict(torch.load(encoder_path), strict=False)
            # Example usage for encoder and decoder
            get_model_file_size(encoder_path)

        if os.path.exists(decoder_path):
            print("Loading Decoder Model")
            model.decoder.load_state_dict(torch.load(decoder_path), strict=False)
            get_model_file_size(decoder_path)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    loss_function = OutlierFocusedLoss(2.0, device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=4
    )

    hyper_params = {
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "sequence_length": sequence_length,
        "num_hidden_units": hidden_units,
        "batch_size": batch_size,
        "station": station,
        "regularization": weight_decay,
        "forecast_hour": fh,
        "climate_div": clim_div,
        "metvar": metvar,
    }
    print("--- Training LSTM ---")

    early_stopper = EarlyStopper(10)

    init_start_event.record()
    train_loss_ls = []
    for ix_epoch in range(1, epochs + 1):
        gc.collect()
        train_loss = model.train_model(
            data_loader=train_loader,
            loss_func=loss_function,
            optimizer=optimizer,
            epoch=ix_epoch,
            training_prediction="recursive",
            teacher_forcing_ratio=0.5,
        )
        scheduler.step(train_loss)
        print(" ")
        train_loss_ls.append(train_loss)
        # log info for comet and loss curves
        experiment.set_epoch(ix_epoch)
        experiment.log_metric("train_loss", train_loss)
        experiment.log_metrics(hyper_params, epoch=ix_epoch)
        if early_stopper.early_stop(train_loss):
            print(f"Early stopping at epoch {ix_epoch}")
            break
        if train_loss <= min(train_loss_ls) and ix_epoch > 5:
            print(f"Saving Model Weights... EPOCH {ix_epoch}")
            save_model_weights(model, encoder_path, decoder_path)
            save_model = False

    init_end_event.record()

    if save_model == True:
        states = model.state_dict()
        torch.save(model.encoder.state_dict(), f"{encoder_path}")
        torch.save(model.decoder.state_dict(), decoder_path)

    print("Successful Experiment")
    # Seamlessly log your Pytorch model
    # log_model(experiment, model, model_name="v9")
    experiment.end()
    print("... completed ...")
    gc.collect()
    torch.cuda.empty_cache()
    # End of MAIN


if __name__ == "__main__":
    # get time for inference
    now = datetime.datetime.now()
    year = now.year
    month = now.month
    day = now.day 

    # start_time = datetime(int(year-1), int(month-1), 1, 0, 0, 0)
    # end_time = date_time(year, int(month-1), 28, 23, 59, 0)
    start_time = datetime(2023, 10, 1, 0, 0, 0)
    end_time = date_time(2025, 3, 31, 23, 59, 0)

    metvar_ls = ["t2m", "u_total", "tp"]
    nwp_model = "HRRR"

    nysm_clim = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/nysm.csv")
    # load NYSM data
    nysm_df = nysm_data.load_nysm_data(year)
    nysm_df.reset_index(inplace=True)
    nysm_network = nysm_df["station"].unique().tolist()

    for c in nysm_clim["climate_division_name"].unique():
        df = nysm_clim[nysm_clim["climate_division_name"] == c]
        stations = df["stid"].unique()
        fh_all = np.arange(1, 19)
        for metvar in metvar_ls:
            for s in stations:
                filtered_df = nysm_df[nysm_df["station"] == s]
                fh = fh_all.copy()
                while len(fh) > 0:
                    fh_r = random.choice(fh)
                    try:
                        print(f"-- Loading data from HRRR for FH {fh_r} --")
                        hrrr_df = hrrr_data.read_hrrr_data(str(fh_r).zfill(2), year)
                        main(
                            start_time=start_time,
                            end_time=end_time,
                            batch_size=1000,
                            station=s,
                            num_layers=3,
                            epochs=5000,
                            weight_decay=1e-15,
                            fh=fh_r,
                            clim_div=c,
                            nwp_model=nwp_model,
                            exclusion_buffer=exclude,
                            metvar=metvar,
                            filtered_df=filtered_df,
                            hrrr_df=hrrr_df,
                        )
                        gc.collect()
                        fh = fh[fh != fh_r]  # removes used FH by value
                    except Exception as e:
                        print(f"-- ERROR ERROR --")
                        print(f"Climate Div: {c}, Station: {s}, FH: {fh_r}")
                        print(e)
