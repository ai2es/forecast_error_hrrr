import sys

sys.path.append("..")

import datetime
import panas as pd
import numpy as np
import os

from model_data import nysm_data_rapids, hrrr_data, prepare_lstm_data

from model_architecture import encode_decode_lstm, sequencer
import pickle


def main(now):
    nwp_model = "HRRR"
    start_time = init_start_event.record()
    # initiate NVIDIA GPU
    print("Number of gpus: ", torch.cuda.device_count())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(device)
    torch.manual_seed(101)

    # get time for inference
    year = now.year
    month = now.month
    day = now.day
    outpath = "/home/aevans/inference/FINAL_OUTPUT"

    # load NYSM data
    nysm_df = nysm_data_rapids.load_nysm_data(year)
    nysm_df.reset_index(inplace=True)

    nysm_network = nysm_df["station"].unique().tolist()

    # load clim_div_dict
    with open(
        "/home/aevans/inference_ai2es_forecast_err/MODELS/lookups/station_to_climdiv.pkl",
        "rb",
    ) as f:
        loaded_dict = pickle.load(f)

    for stid in nysm_network:
        filtered_df = nysm_df[nysm_df["station"] == stid]
        for metvar in ["t2m", "u_total", "tp"]:
            # grab climate division
            clim_div = station_to_climdiv.get(stid)
            # load models
            decoder_path = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/exclusion_buffer/{clim_div}_{metvar}_{stid}_decoder.pth"
            encoder_path = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/exclusion_buffer{clim_div}_{metvar}_{stid}_encoder.pth"

            # load model
            # Initialize multi-task learning model with one encoder and decoders for each station
            model = encode_decode_lstm.ShallowLSTM_seq2seq_multi_task(
                num_sensors=144,
                hidden_units=1728,
                num_layers=3,
                mlp_units=1500,
                device=device,
                num_stations=4,
            ).to(device)

            if os.path.exists(encoder_path):
                print("Loading Encoder Model")
                model.encoder.load_state_dict(torch.load(encoder_path), strict=False)

            if os.path.exists(decoder_path):
                print("Loading Decoder Model")
                model.decoder.load_state_dict(torch.load(decoder_path), strict=False)

            for fh in np.arange(1, 19):
                # load nwp data
                print("-- loading data from HRRR --")
                hrrr_df = hrrr_data_rapids.read_hrrr_data(str(fh).zfill(2), year)

                # prepare data for LSTM
                (lstm_df, features, stations, target, valid_times) = (
                    prepare_lstm_data_rapids.prepare_lstm_data(filtered_df, hrrr_df)
                )

                lstm_dataset = sequencer.SequenceDatasetMultiTask(
                    dataframe=lstm_df,
                    target=target,
                    features=features,
                    sequence_length=30,
                    forecast_steps=fh,
                    device=device,
                    nwp_model=nwp_model,
                    metvar=metvar,
                )
                lstm_kwargs = {
                    "batch_size": 1,
                    "pin_memory": False,
                    "shuffle": False,
                }
                lstm_loader = torch.utils.data.DataLoader(lstm_dataset, **lstm_kwargs)

                # predict for model
                lstm_output = model.predict(data_loader=lstm_loader)

                # augment lstm output

                # save_lstm_output

    end_time = init_end_event.record()
