import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap


def plot_combined_heatmap(
    clim_div,
    base_path,
    hrrr_file="ALL_u_total_hourly_abs_error_hrrr.csv",
    pers_file="ALL_u_total_hourly_abs_error_persistence.csv",
    lstm_file="ALL_u_total_hourly_error.csv",
    title="Combined Error Heatmap",
):

    def load_and_pivot(file):
        all_data = []
        for div in clim_div:
            csv_path = os.path.join(base_path, div, file)
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df["Hour"] = df["Hour"].str.slice(0, 2).astype(int)
                df.rename(columns={"Mean_Absolute_Error": "error"}, inplace=True)
                df["Division"] = div
                all_data.append(df)
            else:
                print(f"Missing file: {csv_path}")

        df_all = pd.concat(all_data, ignore_index=True)

        heat = df_all.pivot_table(
            index="Division", columns="Hour", values="error", aggfunc="mean"
        )
        return heat.loc[clim_div]

    # Load
    hrrr_heat = load_and_pivot(hrrr_file)
    pers_heat = load_and_pivot(pers_file)
    lstm_heat = load_and_pivot(lstm_file)

    station_counts = nysm_df.groupby("Climate_division")["stid"].nunique().to_dict()

    # Build rows
    weights = np.array([station_counts[div] for div in clim_div])
    weights = weights / weights.sum()  # normalize to sum to 1

    # Weighted averages
    hrrr_row = np.average(hrrr_heat, axis=0, weights=weights)
    pers_row = np.average(pers_heat, axis=0, weights=weights)
    lstm_row = np.average(lstm_heat, axis=0, weights=weights)

    # Convert back to Series for consistent indexing
    hrrr_row = pd.Series(hrrr_row, index=hrrr_heat.columns)
    pers_row = pd.Series(pers_row, index=pers_heat.columns)
    lstm_row = pd.Series(lstm_row, index=lstm_heat.columns)

    diff_lstm_row = hrrr_row - pers_row
    diff_lstm_row_perc = (diff_lstm_row / hrrr_row) * 100

    top = pd.DataFrame(
        [hrrr_row, pers_row, diff_lstm_row_perc],
        index=["HRRR (MAE)", "LSTM (MAE)", "LSTM % Improvement"],
    )

    # Add name to LSTM row
    lstm_row.name = "All Stations"

    # Build final table
    final = pd.concat([top, lstm_heat, lstm_row.to_frame().T], axis=0)

    ##################################################################
    # SINGLE HEATMAP TRICK: we mask each row group for different cmap
    ##################################################################

    mask_grey = np.ones_like(final, dtype=bool)  # Row 0
    mask_row2 = np.ones_like(final, dtype=bool)  # Row 1 (unique cmap)
    mask_diff = np.ones_like(final, dtype=bool)  # Row 2 (difference)
    mask_main = np.ones_like(final, dtype=bool)  # Rows 3+

    mask_grey[0] = False  # Only row 0
    mask_row2[1] = False  # Only row 1
    mask_diff[2] = False  # Only row 2
    mask_main[3:] = False  # All remaining

    fig, ax = plt.subplots(figsize=(26, 12))

    # -------------------------------------------------------------
    # 1. MAIN PiYG heatmap (bottom layer)
    # -------------------------------------------------------------
    hm_main = sns.heatmap(
        final,
        cmap="Purples",
        mask=mask_main,
        linewidths=0.4,
        ax=ax,
        cbar=False,
        vmin=3.5,
        vmax=5,
    )

    # Grab the *correct* PiYG QuadMesh
    quad_piyg = hm_main.collections[0]
    # -------------------------------------------------------------
    # 4. RIGHT PiYG colorbar (main errors)
    # -------------------------------------------------------------
    cbar_right = plt.colorbar(quad_piyg, ax=ax, location="right", pad=0.02, shrink=0.75)
    cbar_right.set_label(r"Mean Error (mm hr$^{-1}$)", fontsize=16)

    # (m s$^{-1}$)
    # (°C)
    # -------------------------------------------------------------
    # 2. GREYS top row
    # -------------------------------------------------------------
    # Extract only the visible values in the GREYS row
    grey_values = final.iloc[0][~mask_grey[0]]

    # Compute dynamic bounds
    vmin_grey = grey_values.min()
    vmax_grey = grey_values.max()

    sns.heatmap(
        final,
        cmap="Greys",
        mask=mask_grey,
        ax=ax,
        cbar=False,
        vmin=vmin_grey,
        vmax=vmax_grey,
    )

    # -------------------------------------------------------------
    # 2B. ROW 1: Persistence values, but colored using % difference
    # -------------------------------------------------------------
    # Build temp matrix full of NaNs
    temp = final.copy() * np.nan

    # Put only diff_lstm_row into row 1 (color will come from this)
    temp.iloc[1] = diff_lstm_row.values

    # Draw heatmap for ROW 1 but colored by difference
    sns.heatmap(
        temp,
        cmap="RdBu",
        mask=mask_row2,  # mask everything except row 1
        center=0,
        ax=ax,
        cbar=False,
        vmin=-1,
        vmax=1,
    )

    # -------------------------------------------------------------
    # 3. DIFFERENCE RdBu heatmap (top of PiYG except row 0)
    # -------------------------------------------------------------
    hm_diff = sns.heatmap(
        final,
        cmap="RdBu",
        mask=mask_diff,
        center=0,
        ax=ax,
        cbar=True,
        vmin=-50,
        vmax=50,
        cbar_kws={
            "orientation": "vertical",
            "location": "left",
            "pad": 0.2,
            "label": "← Worse | % Diff. b/w Persistence and LSTM | Better →",
            "fraction": 0.01,  # shrink colorbar (smaller = shorter)
            "aspect": 30,  # optional: change thickness
        },
    )
    # >>> ADD THESE LINES <<<
    hm_diff.collections[0].colorbar.ax.tick_params(labelsize=16)
    hm_diff.collections[0].colorbar.ax.yaxis.label.set_size(16)

    # Add an emphasized line between row 2 and the rest (i.e., between row index 1 and 2)
    ax.hlines(
        y=3,  # boundary between second and third row
        xmin=0,
        xmax=final.shape[1],  # number of columns
        color="black",
        linewidth=3,
    )

    for i in range(final.shape[0]):
        for j in range(final.shape[1]):

            if not (
                mask_grey[i, j]
                and mask_row2[i, j]
                and mask_diff[i, j]
                and mask_main[i, j]
            ):

                val = final.values[i, j]
                if pd.notna(val):

                    # --------------------------------------------------
                    # Choose correct colormap + normalization per row
                    # --------------------------------------------------
                    if i == 0:
                        cmap = plt.get_cmap("Greys")
                        norm = mcolors.Normalize(vmin=vmin_grey, vmax=vmax_grey)
                        val_for_color = val

                    elif i == 1:
                        cmap = plt.get_cmap("RdBu")
                        norm = mcolors.Normalize(vmin=-1, vmax=1)
                        val_for_color = diff_lstm_row.iloc[j]

                    elif i == 2:
                        cmap = plt.get_cmap("RdBu")
                        norm = mcolors.Normalize(vmin=-50, vmax=50)
                        val_for_color = val

                    else:
                        cmap = plt.get_cmap("PiYG")
                        norm = mcolors.Normalize(vmin=-1, vmax=1)
                        val_for_color = val

                    rgba = cmap(norm(val_for_color))
                    r, g, b = rgba[:3]

                    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                    text_color = "white" if luminance < 0.5 else "black"

                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=12,
                        color=text_color,
                        zorder=20,
                    )

    plt.title(title, fontsize=22)
    plt.xlabel("Local Hour of Day", fontsize=18)
    plt.ylabel("Climate Divisions", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16, rotation=0)

    plt.tight_layout()
    plt.show()

    return final


final_df = plot_combined_heatmap(
    clim_div=clim_div,
    base_path="/home/aevans/nwp_bias/src/machine_learning/data/error_visuals/dataframes",
    title="OKSM: Wind Error Predictions\n LSTM Error Grouped By Local Time of Day",
)
