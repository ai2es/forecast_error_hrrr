"""LSTM seq2seq inference driver.

For a given inference time (`now`) and forecast hour (`fh`):

1. Load the year's HRRR forecasts (for that `fh`) and NYSM
   observations onto the GPU via cuDF.
2. For each NYSM station and each supported variable
   (`t2m`, `u_total`, `tp`):
     a. Build the inference dataframe (past 30 hours of HRRR + NYSM,
        plus future HRRR forecasts out to `now + fh`) via
        `prepare_lstm_data_rapids`.
     b. Wrap it in `SequenceDatasetMultiTask` (per-window z-score,
        NYSM persistence on the future portion).
     c. Load the trained encoder/decoder weights for that
        `(climdiv, metvar, station)` and run `model.predict`.
     d. Optionally apply the post-hoc linear calibration if a lookup
        table is available for that `(climdiv, metvar)`.
3. Append the predictions to the per-(`fh`, `metvar`) output parquet
   under `OUT_DIR`.

CLI
---
    python lstm_s2s_engine.py --fh 6 --device_id 0
"""

import argparse
import os
import pickle
import sys
from datetime import datetime

import cudf
import torch

sys.path.append("..")

from model_architecture import encode_decode_lstm, sequencer
from model_data import (
    hrrr_data_rapids,
    nysm_data_rapids,
    prepare_lstm_data_rapids,
)


# -- Paths -----------------------------------------------------------
# Override these via environment variables if your filesystem layout
# differs from the defaults.
MODEL_DIR = os.environ.get(
    "LSTM_MODEL_DIR",
    "/home/aevans/inference_ai2es_forecast_err/MODELS",
)
OUT_DIR = os.environ.get(
    "LSTM_OUTPUT_DIR",
    "/home/aevans/inference/FINAL_OUTPUT",
)


def linear_transform(station, clim_div, metvar, fh, lstm_output):
    """Apply the optional post-hoc linear calibration.

    Looks up `(station, fh)` in a per-`(clim_div, metvar)` CSV.  If the
    file is missing, or the table has no row for this combination,
    returns `lstm_output` unchanged.

    The calibration formula is::

        calibrated = (lstm_output - diff) * alpha
    """
    lookup_path = (
        f"{MODEL_DIR}/{clim_div}_{metvar}_HRRR_lookup_linear.csv"
    )
    if not os.path.exists(lookup_path):
        return lstm_output
    linear_tbl = cudf.read_csv(lookup_path)
    row = linear_tbl[
        (linear_tbl["station"] == station) & (linear_tbl["forecast_hour"] == fh)
    ]
    if row.shape[0] == 0:
        return lstm_output
    alpha1 = row["alpha"].iloc[0]
    diff1 = row["diff"].iloc[0]
    return (lstm_output - diff1) * alpha1


def main(now, fh):
    """Score every NYSM station for forecast hour `fh`.

    Parameters
    ----------
    now : datetime.datetime
        The inference anchor time (typically rounded to the hour).
    fh : int
        HRRR forecast hour to score (1..18).
    """
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

    # NYSM observations for the year (cuDF -> pandas hand-off).
    nysm_df = nysm_data_rapids.load_nysm_data(year).reset_index(drop=True)
    nysm_network = nysm_df["station"].unique().tolist()

    # HRRR forecasts for this `fh` for the year.
    hrrr_df = hrrr_data_rapids.read_hrrr_data(str(fh).zfill(2), year)

    # Map from station id -> climate division (used to locate the
    # right model weights).
    with open(
        "/home/aevans/inference_ai2es_forecast_err/MODELS/lookups/"
        "station_to_climdiv.pkl",
        "rb",
    ) as f:
        station_to_climdiv = pickle.load(f)

    for metvar in ["t2m", "u_total", "tp"]:
        outs = []
        for stid in nysm_network:
            clim_div = station_to_climdiv.get(stid)
            decoder_path = (
                f"{MODEL_DIR}/{clim_div}_{metvar}_{stid}_decoder.pth"
            )
            encoder_path = (
                f"{MODEL_DIR}/{clim_div}_{metvar}_{stid}_encoder.pth"
            )

            # Build the model with the same architecture used during
            # training (these constants come from the original LSTM
            # training run).
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
                model.encoder.load_state_dict(
                    torch.load(encoder_path), strict=False
                )

            if os.path.exists(decoder_path):
                print(f"Loading Decoder Model for {stid} / {metvar}")
                model.decoder.load_state_dict(
                    torch.load(decoder_path), strict=False
                )

            (
                lstm_df,
                features,
                stations,
                target,
                valid_times,
            ) = prepare_lstm_data_rapids.prepare_lstm_data(
                nysm_df, hrrr_df, stid, metvar, fh, now=now
            )

            lstm_dataset = sequencer.SequenceDatasetMultiTask(
                dataframe=lstm_df,
                target=target,
                features=features,
                sequence_length=30,
                forecast_steps=fh,
                device=device,
                metvar=metvar,
                nwp_model=nwp_model,
            )

            lstm_loader = torch.utils.data.DataLoader(
                lstm_dataset, batch_size=2, shuffle=False
            )

            lstm_output = model.predict(data_loader=lstm_loader)
            lstm_output = linear_transform(
                stid, clim_div, metvar, fh, lstm_output
            )

            outs.append(
                {
                    "valid_time": now,
                    "stid": stid,
                    "model_output": lstm_output,
                }
            )

        # Append the new rows to the per-(fh, metvar) output parquet.
        out_path = f"{OUT_DIR}/fh{fh}_{metvar}_inference_out.parquet"
        muthr = cudf.read_parquet(out_path).reset_index()
        child = cudf.DataFrame(outs)
        muthr = cudf.concat([muthr, child], ignore_index=True)
        muthr.set_index(["valid_time", "stid"], inplace=True)
        muthr.to_parquet(out_path)

    end_event.record()
    torch.cuda.synchronize()
    print(
        f"Inference Completed in "
        f"{start_event.elapsed_time(end_event) / 1000:.2f} seconds"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fh", type=int, required=True, help="Target Forecast Hour"
    )
    parser.add_argument(
        "--device_id",
        type=int,
        required=True,
        help="GPU device id (e.g. 0 for cuda:0)",
    )
    args = parser.parse_args()

    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    try:
        main(now, args.fh)
    except Exception as e:
        print("Runtime Error:", str(e))
        raise
