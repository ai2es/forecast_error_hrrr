"""BNN inference driver.

Mirrors `lstm_s2s_engine.py` but loads the BNN model and exposes the
full Bayesian uncertainty decomposition from
`bnn.ShallowLSTM_seq2seq_multi_task_bnn.predict`:

* `mu`              - predictive mean (point estimate)
* `epistemic_var`   - model uncertainty (shrinks with more data)
* `aleatoric_var`   - irreducible data noise the model predicts
* `total_var`       - `epistemic_var + aleatoric_var` (law of total
                       variance), i.e. the full predictive variance

`mc_samples` controls how many Monte-Carlo draws from the variational
posterior are used to estimate `epistemic_var`.  20-50 is typical;
higher values give a less noisy estimate at linearly higher cost.

The output parquet schema is:

    valid_time, stid, mu, epistemic_var, aleatoric_var, total_var

Each value column holds a list (length = output_dim) of per-target
floats.

CLI
---
    python bnn_s2s_engine.py --fh 6 --device_id 0 --mc_samples 30
"""

import sys

sys.path.append("..")

import argparse
import os
import pickle
from datetime import datetime

import cudf
import torch

from model_architecture import bnn, sequencer
from model_data import hrrr_data_rapids, nysm_data_rapids, prepare_lstm_data_rapids


MODEL_DIR = os.environ.get(
    "BNN_MODEL_DIR",
    "/home/aevans/inference_ai2es_forecast_err/MODELS/BNN",
)
OUT_DIR = os.environ.get(
    "BNN_OUTPUT_DIR",
    "/home/aevans/inference/FINAL_OUTPUT/BNN",
)


def main(now, fh, mc_samples=30):
    """Score every NYSM station with the Bayesian model.

    Parameters
    ----------
    now : datetime.datetime
        Inference anchor time (typically rounded to the hour).
    fh : int
        HRRR forecast hour to score (1..18).
    mc_samples : int
        Number of MC draws from the BNN's variational posterior
        (default 30).  Drives the noise level of `epistemic_var`.
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
            bnn_path = f"{MODEL_DIR}/{clim_div}_{metvar}_{stid}_fh{fh}_bnn.pth"

            if not (
                os.path.exists(encoder_path)
                and os.path.exists(decoder_path)
                and os.path.exists(bnn_path)
            ):
                print(f"[skip] missing weights for {clim_div} / {metvar} / {stid}")
                continue

            (
                lstm_df,
                features,
                stations,
                target,
                valid_times,
            ) = prepare_lstm_data_rapids.prepare_lstm_data(
                nysm_df, hrrr_df, stid, metvar, fh, now=now
            )

            num_sensors = int(len(features))
            hidden_units = int(12 * len(features))

            model = bnn.ShallowLSTM_seq2seq_multi_task_bnn(
                num_sensors=num_sensors,
                hidden_units=hidden_units,
                num_layers=3,
                mlp_units=1500,
                device=device,
                num_stations=len(stations),
                seq_len=30,
                input_dim=num_sensors,
            ).to(device)
            model.encoder.load_state_dict(torch.load(encoder_path), strict=False)
            model.decoder.load_state_dict(torch.load(decoder_path), strict=False)
            model.bnn.load_state_dict(torch.load(bnn_path), strict=False)

            lstm_dataset = sequencer.SequenceDatasetMultiTask(
                dataframe=lstm_df,
                target=target,
                features=features,
                sequence_length=30,
                forecast_steps=fh,
                device=device,
                metvar=metvar,
                nwp_model=nwp_model,
                valid_times=valid_times,
                return_valid_times=True,
            )
            lstm_loader = torch.utils.data.DataLoader(
                lstm_dataset, batch_size=2, shuffle=False
            )

            mu, epistemic_var, aleatoric_var, vts = model.predict(
                data_loader=lstm_loader,
                mc_samples=mc_samples,
            )
            total_var = epistemic_var + aleatoric_var

            # Take only the most-recent inference window (i = 0); the
            # rest of the items slide off the end of the inference
            # window and aren't valid at `now`.
            mu0 = mu[0].detach().cpu().numpy()
            epi0 = epistemic_var[0].detach().cpu().numpy()
            ale0 = aleatoric_var[0].detach().cpu().numpy()
            tot0 = total_var[0].detach().cpu().numpy()

            outs.append(
                {
                    "valid_time": now,
                    "stid": stid,
                    "mu": mu0.tolist(),
                    "epistemic_var": epi0.tolist(),
                    "aleatoric_var": ale0.tolist(),
                    "total_var": tot0.tolist(),
                }
            )

        if not outs:
            print(f"[{metvar}] no stations produced output for fh={fh}")
            continue

        out_path = f"{OUT_DIR}/fh{fh}_{metvar}_bnn_inference_out.parquet"
        new_df = cudf.DataFrame(outs)
        if os.path.exists(out_path):
            existing = cudf.read_parquet(out_path).reset_index()
            new_df = cudf.concat([existing, new_df], ignore_index=True)
        new_df.set_index(["valid_time", "stid"], inplace=True)
        new_df.to_parquet(out_path)

    end_event.record()
    torch.cuda.synchronize()
    print(
        f"BNN inference completed in "
        f"{start_event.elapsed_time(end_event) / 1000:.2f} seconds"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fh", type=int, required=True, help="Target Forecast Hour")
    parser.add_argument(
        "--device_id",
        type=int,
        required=True,
        help="GPU device id (e.g. 0 for cuda:0)",
    )
    parser.add_argument(
        "--mc_samples",
        type=int,
        default=30,
        help=(
            "Number of Monte-Carlo draws from the BNN posterior used to "
            "estimate epistemic uncertainty (default 30)."
        ),
    )
    args = parser.parse_args()
    now = datetime.now()
    now = now.replace(minute=0, second=0, microsecond=0)
    try:
        main(now, args.fh, mc_samples=args.mc_samples)
    except Exception as e:
        print("Runtime Error:", str(e))
        raise
