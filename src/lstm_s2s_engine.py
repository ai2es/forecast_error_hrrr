import sys
sys.path.append("..")

import datetime
import cudf
import cupy as cp
import numpy as np
import os
import torch
import pickle
import argparse

from torch.utils.dlpack import from_dlpack
from model_data import nysm_data_rapids, hrrr_data_rapids, prepare_lstm_data_rapids
from model_architecture import encode_decode_lstm, sequencer

def main(now, fh):
    nwp_model = "HRRR"
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    print("Number of GPUs:", torch.cuda.device_count())
    device_id = 0
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    torch.manual_seed(101)

    year = now.year
    outpath = "/home/aevans/inference/FINAL_OUTPUT"

    # Load NYSM data using cuDF
    nysm_df = nysm_data_rapids.load_nysm_data(year)
    nysm_df = nysm_df.reset_index(drop=True)

    nysm_network = nysm_df["station"].unique().to_pandas().tolist()

    hrrr_df = hrrr_data_rapids.read_hrrr_data(str(fh).zfill(2), year)

    # Load station-to-climdiv mapping
    with open("/home/aevans/inference_ai2es_forecast_err/MODELS/lookups/station_to_climdiv.pkl", "rb") as f:
        station_to_climdiv = pickle.load(f)

    for stid in nysm_network:
        for metvar in ["t2m", "u_total", "tp"]:
            clim_div = station_to_climdiv.get(stid)
            decoder_path = f"/home/aevans/inference_ai2es_forecast_err/MODELS/{clim_div}_{metvar}_{stid}_decoder.pth"
            encoder_path = f"/home/aevans/inference_ai2es_forecast_err/MODELS/{clim_div}_{metvar}_{stid}_encoder.pth"

            model = encode_decode_lstm.ShallowLSTM_seq2seq_multi_task(
                num_sensors=144,
                hidden_units=1728,
                num_layers=3,
                mlp_units=1500,
                device=device,
                num_stations=4,
            ).to(device)

            if os.path.exists(encoder_path):
                print(f"Loading Encoder Model for {stid} / {metvar}")
                model.encoder.load_state_dict(torch.load(encoder_path), strict=False)

            if os.path.exists(decoder_path):
                print(f"Loading Decoder Model for {stid} / {metvar}")
                model.decoder.load_state_dict(torch.load(decoder_path), strict=False)

            lstm_df, features, stations, target, valid_times = prepare_lstm_data_rapids.prepare_lstm_data(
                nysm_df, hrrr_df
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

            lstm_loader = torch.utils.data.DataLoader(
                lstm_dataset,
                batch_size=32,
                pin_memory=True,
                shuffle=False,
            )

            lstm_output = model.predict(data_loader=lstm_loader)

            # TODO: Add logic to augment or save output here


    end_event.record()
    torch.cuda.synchronize()
    print(f"Inference Completed in {start_event.elapsed_time(end_event) / 1000:.2f} seconds")

    '''
    END OF MAIN
    '''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fh", type=int, required=True, help="Target Forecast Hour"
    )
    parser.add_argument(
        "--device_id",
        type=int,
        required=True,
        help="Device to use (e.g., 'cuda:0', 'cuda:1', 'cpu')",
    )
    args = parser.parse_args()
    now = datetime.now()
    # try:
    main(now, args.fh)
    # except Exception as e:
    print("🔥 Runtime Error:", str(e))
