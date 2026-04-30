"""Hybrid (LSTM + ViT) inference driver.

Mirrors `lstm_s2s_engine.py`.  Differences:

    * builds `hybrid_vit_lstm.LSTM_Encoder_Decoder_with_ViT`
    * uses `prepare_hybrid_data_rapids` so radiometer image paths get
      attached to each row of the inference dataframe
    * uses `SequenceDatasetMultiTaskHybrid` with
      `return_valid_times=True` so each batch is `(X, P, y, v)` -
      what `LSTM_Encoder_Decoder_with_ViT.predict` expects
"""

import sys

sys.path.append("..")

import argparse
import os
import pickle
from datetime import datetime

import cudf
import torch

from model_architecture import hybrid_vit_lstm, sequencer
from model_data import (
    hrrr_data_rapids,
    nysm_data_rapids,
    prepare_hybrid_data,
)


MODEL_DIR = os.environ.get(
    "HYBRID_MODEL_DIR",
    "/home/aevans/inference_ai2es_forecast_err/MODELS/HYBRID",
)
OUT_DIR = os.environ.get(
    "HYBRID_OUTPUT_DIR",
    "/home/aevans/inference/FINAL_OUTPUT/HYBRID",
)


HYBRID_DEFAULTS = dict(
    past_timesteps=1,
    future_timesteps=1,
    pos_embedding=0.5,
    time_embedding=0.5,
    vit_num_layers=3,
    num_heads=11,
    hidden_dim=7260,
    mlp_dim=1032,
    output_dim=1,
    dropout=1e-15,
    attention_dropout=1e-12,
)


def main(now, fh, hybrid_kwargs=None):
    nwp_model = "HRRR"
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    print("Number of GPUs:", torch.cuda.device_count())
    device_id = 0
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    torch.manual_seed(101)

    hybrid_kwargs = {**HYBRID_DEFAULTS, **(hybrid_kwargs or {})}
    year = now.year

    nysm_df = nysm_data_rapids.load_nysm_data(year)
    nysm_df = nysm_df.reset_index(drop=True)
    nysm_network = nysm_df["station"].unique().tolist()

    hrrr_df = hrrr_data_rapids.read_hrrr_data(str(fh).zfill(2), year)

    with open(
        "/home/aevans/inference_ai2es_forecast_err/MODELS/lookups/station_to_climdiv.pkl",
        "rb",
    ) as f:
        station_to_climdiv = pickle.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)

    for metvar in ["t2m", "u_total", "tp"]:
        outs = []
        for stid in nysm_network:
            clim_div = station_to_climdiv.get(stid)
            encoder_path = f"{MODEL_DIR}/{clim_div}_{metvar}_{stid}_fh{fh}_encoder.pth"
            decoder_path = f"{MODEL_DIR}/{clim_div}_{metvar}_{stid}_fh{fh}_decoder.pth"
            vit_path = f"{MODEL_DIR}/{clim_div}_{metvar}_{stid}_fh{fh}_vit.pth"

            if not (
                os.path.exists(encoder_path)
                and os.path.exists(decoder_path)
                and os.path.exists(vit_path)
            ):
                print(f"[skip] missing weights for {clim_div} / {metvar} / {stid}")
                continue

            (
                lstm_df,
                features,
                stations,
                target,
                valid_times,
                image_list_cols,
            ) = prepare_hybrid_data.prepare_hybrid_data_rapids(
                nysm_df, hrrr_df, stid, metvar, fh, now=now
            )

            num_sensors = int(len(features))
            hidden_units = int(12 * len(features))

            model = hybrid_vit_lstm.LSTM_Encoder_Decoder_with_ViT(
                num_sensors=num_sensors,
                hidden_units=hidden_units,
                num_layers=3,
                mlp_units=1500,
                device=device,
                num_stations=len(image_list_cols),
                **hybrid_kwargs,
            ).to(device)
            model.encoder.load_state_dict(torch.load(encoder_path), strict=False)
            model.decoder.load_state_dict(torch.load(decoder_path), strict=False)
            model.ViT.load_state_dict(torch.load(vit_path), strict=False)

            lstm_dataset = sequencer.SequenceDatasetMultiTaskHybrid(
                dataframe=lstm_df,
                target=target,
                features=features,
                sequence_length=30,
                forecast_steps=fh,
                device=device,
                metvar=metvar,
                nwp_model=nwp_model,
                image_list_cols=image_list_cols,
                valid_times=valid_times,
                return_valid_times=True,
            )
            lstm_loader = torch.utils.data.DataLoader(
                lstm_dataset, batch_size=2, shuffle=False
            )

            outputs, vts = model.predict(data_loader=lstm_loader)
            # Take only the most-recent forecast (i = 0).
            out0 = outputs[0].detach().cpu().numpy()
            outs.append(
                {
                    "valid_time": now,
                    "stid": stid,
                    "model_output": out0.tolist(),
                }
            )

        if not outs:
            print(f"[{metvar}] no stations produced output for fh={fh}")
            continue

        out_path = f"{OUT_DIR}/fh{fh}_{metvar}_hybrid_inference_out.parquet"
        new_df = cudf.DataFrame(outs)
        if os.path.exists(out_path):
            existing = cudf.read_parquet(out_path).reset_index()
            new_df = cudf.concat([existing, new_df], ignore_index=True)
        new_df.set_index(["valid_time", "stid"], inplace=True)
        new_df.to_parquet(out_path)

    end_event.record()
    torch.cuda.synchronize()
    print(
        f"Hybrid inference completed in "
        f"{start_event.elapsed_time(end_event) / 1000:.2f} seconds"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fh", type=int, required=True, help="Target Forecast Hour")
    parser.add_argument(
        "--device_id",
        type=int,
        required=True,
        help="Device to use (e.g., 0 for cuda:0)",
    )
    args = parser.parse_args()
    now = datetime.now()
    now = now.replace(minute=0, second=0, microsecond=0)
    try:
        main(now, args.fh)
    except Exception as e:
        print("Runtime Error:", str(e))
        raise
