import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import importlib
import functions as f
from scintkit.preprocessing.format import temp_formating
from scintkit.services.phase_detrend import detect_sampling_rate
from pathlib import Path
import time
import CONFIG as cf

from concurrent.futures import ProcessPoolExecutor

importlib.reload(f)
importlib.reload(cf)


# ============================================================
# PROCESS ONE FILE PAIR
# ============================================================

def process_s4_file_pair(pair):

    fileA, fileB = pair

    results = []

    # ========================================================
    # TIMERS FOR THIS FILE PAIR
    # ========================================================

    total_read_time = 0
    total_preprocess_time = 0
    total_s4_time = 0
    total_grouping_time = 0
    total_merge_time = 0
    total_corr_time = 0

    file_start = time.time()

    print(f"\nProcessing:")
    print(fileA)
    print(fileB)


    # ========================================================
    # 1. READ FILES
    # ========================================================

    read_start = time.time()

    dfa = pd.read_parquet(fileA)
    dfb = pd.read_parquet(fileB)

    rAloc = f.extract_coord(fileA)
    rBloc = f.extract_coord(fileB)

    read_time = time.time() - read_start
    total_read_time += read_time

    print(f"Read files: {read_time:.3f} seconds")


    # ========================================================
    # 2. PREPROCESSING
    # ========================================================

    preprocess_start = time.time()

    dfa = dfa.sort_values("datetime").reset_index(drop=True)
    dfb = dfb.sort_values("datetime").reset_index(drop=True)

    # Filter elevation
    dfa = dfa[dfa['elev'] > cf.elevation_filter].copy()
    dfb = dfb[dfb['elev'] > cf.elevation_filter].copy()

    # Format timestamps
    dfa = temp_formating(dfa)
    dfb = temp_formating(dfb)

    # Sampling rate
    samp_ra = detect_sampling_rate(dfa)
    dt = 1 / samp_ra

    preprocess_time = time.time() - preprocess_start
    total_preprocess_time += preprocess_time

    print(f"Preprocessing: {preprocess_time:.3f} seconds")


    # ========================================================
    # 3. S4 CALCULATION
    # ========================================================

    s4_start = time.time()

    print("Adding S4")

    dfa = f.add_s4(dfa)
    dfb = f.add_s4(dfb)

    s4A = (
        dfa[
            (dfa.s4_1 > cf.thresh) &
            (dfa.s4_2 > cf.thresh)
        ][
            ["minbin", "svid", "cons", "s4_1", "s4_2"]
        ].drop_duplicates()
    )

    s4B = (
        dfb[
            (dfb.s4_1 > cf.thresh) &
            (dfb.s4_2 > cf.thresh)
        ][
            ["minbin", "svid", "cons", "s4_1", "s4_2"]
        ].drop_duplicates()
    )

    s4_time = time.time() - s4_start
    total_s4_time += s4_time

    print(f"S4 calculation/filtering: {s4_time:.3f} seconds")


    # ========================================================
    # 4. MERGE S4 SUMMARIES
    # ========================================================

    merge_start = time.time()

    s4_summary = s4A.merge(
        s4B,
        on=["minbin", "svid", "cons"],
        suffixes=("_A", "_B")
    )

    interesting = s4_summary

    merge_time = time.time() - merge_start
    total_merge_time += merge_time

    print(f"S4 summary merge: {merge_time:.3f} seconds")
    print(f"Interesting scintillation events: {len(interesting)}")


    # ========================================================
    # 5. CREATE GROUPS
    # ========================================================

    grouping_start = time.time()

    dfa["min_floor"] = dfa["datetime"].dt.floor("min")
    dfb["min_floor"] = dfb["datetime"].dt.floor("min")

    groupsA = dict(
        list(
            dfa.groupby(
                ["svid", "cons", "min_floor"]
            )
        )
    )

    groupsB = dict(
        list(
            dfb.groupby(
                ["svid", "cons", "min_floor"]
            )
        )
    )

    grouping_time = time.time() - grouping_start
    total_grouping_time += grouping_time

    print(f"Grouping data: {grouping_time:.3f} seconds")


    # ========================================================
    # 6. CROSS CORRELATION
    # ========================================================

    for _, event in interesting.iterrows():

        # ----------------------------------------------------
        # Get event information
        # ----------------------------------------------------

        minute = event["minbin"]
        svid = event["svid"]
        cons = event["cons"]


        # ----------------------------------------------------
        # Extract groups
        # ----------------------------------------------------

        groupA = groupsA.get(
            (svid, cons, minute)
        )

        groupB = groupsB.get(
            (svid, cons, minute)
        )

        if groupA is None or groupB is None:
            continue


        # ----------------------------------------------------
        # Check minimum data
        # ----------------------------------------------------

        if len(groupA) < 10 or len(groupB) < 10:
            continue


        # ====================================================
        # MERGE RECEIVER A AND B
        # ====================================================

        event_merge_start = time.time()

        group = groupA.merge(
            groupB,
            on=["datetime", "svid", "cons"],
            suffixes=("_A", "_B")
        )

        group = f.handle_nan(
            group,
            cf.nan_method,
            sig_columns=[
                'snr1_A',
                'snr1_B',
                'snr2_A',
                'snr2_B'
            ]
        )

        event_merge_time = time.time() - event_merge_start
        total_merge_time += event_merge_time


        if len(group) < 10:
            continue


        # ====================================================
        # CROSS CORRELATION
        # ====================================================

        corr_start = time.time()


        # ----------------------------------------------------
        # SNR 1
        # ----------------------------------------------------

        correlation, lag_b, cor_norm, lag_norm = (
            f.cross_correlation(
                group["snr1_A"],
                group["snr1_B"]
            )
        )

        autoA_max, autoA_lagb, autoA_cor, autoA_lags = (
            f.cross_correlation(
                group["snr1_A"],
                group["snr1_A"]
            )
        )

        autoB_max, autoB_lagb, autoB_cor, autoB_lags = (
            f.cross_correlation(
                group["snr1_B"],
                group["snr1_B"]
            )
        )


        # ----------------------------------------------------
        # SNR 2
        # ----------------------------------------------------

        correlation2, lag2, cor2, lag_n_2 = (
            f.cross_correlation(
                group["snr2_A"],
                group["snr2_B"]
            )
        )

        autoA2_max, autoA_lag2, autoA_cor2, autoA_lags2 = (
            f.cross_correlation(
                group["snr2_A"],
                group["snr2_A"]
            )
        )

        autoB2_max, autoB_lag2, autoB_cor2, autoB_lags2 = (
            f.cross_correlation(
                group["snr2_B"],
                group["snr2_B"]
            )
        )


        # ----------------------------------------------------
        # Time delays
        # ----------------------------------------------------

        time_delay = lag_b * dt
        time_delay2 = lag2 * dt


        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        results.append({

            'minute': minute,

            'prn': group['prn_B'].iloc[0],

            # S4
            's4_1_A': event['s4_1_A'],
            's4_1_B': event['s4_1_B'],
            's4_2_A': event['s4_2_A'],
            's4_2_B': event['s4_2_B'],

            # Geometry
            'elev': group['elev_A'].mean(),
            'azim': group['azim_A'].mean(),

            # Receiver locations
            'r_A': rAloc,
            'r_B': rBloc,


            # =================================================
            # SNR 1 AUTO CORRELATION
            # =================================================

            'auto_cor_A_1': autoA_cor,
            'auto_cor_A_max_1': autoA_max,

            'auto_cor_B_1': autoB_cor,
            'auto_cor_B_max_1': autoB_max,


            # =================================================
            # SNR 1 CROSS CORRELATION
            # =================================================

            'corr_norm_1': cor_norm,
            'lag_norm_1': lag_norm,

            'max_corr_1': correlation,
            'best_lag_1': lag_b,

            'time_delay_1': time_delay,


            # =================================================
            # SNR 2 AUTO CORRELATION
            # =================================================

            'auto_cor_A_2': autoA_cor2,
            'auto_cor_A_max_2': autoA2_max,

            'auto_cor_B_2': autoB_cor2,
            'auto_cor_B_max_2': autoB2_max,


            # =================================================
            # SNR 2 CROSS CORRELATION
            # =================================================

            'corr_norm_2': cor2,
            'lag_norm_2': lag_n_2,

            'max_corr_2': correlation2,
            'best_lag_2': lag2,

            'time_delay_2': time_delay2
        })


        corr_time = time.time() - corr_start
        total_corr_time += corr_time


    # ========================================================
    # FINISHED FILE
    # ========================================================

    total_file_time = time.time() - file_start

    print(f"Finished {Path(fileA).name}")

    print(
        f"File total: {total_file_time:.3f} sec | "
        f"Read: {total_read_time:.3f} | "
        f"Preprocess: {total_preprocess_time:.3f} | "
        f"S4: {total_s4_time:.3f} | "
        f"Grouping: {total_grouping_time:.3f} | "
        f"Merge: {total_merge_time:.3f} | "
        f"Correlation: {total_corr_time:.3f}"
    )


    # ========================================================
    # RETURN RESULTS + TIMINGS
    # ========================================================

    return {
        'results': results,

        'read_time': total_read_time,
        'preprocess_time': total_preprocess_time,
        's4_time': total_s4_time,
        'grouping_time': total_grouping_time,
        'merge_time': total_merge_time,
        'corr_time': total_corr_time,

        'file_time': total_file_time
    }


# ============================================================
# MAIN
# ============================================================

def main():

    start = time.time()


    # ========================================================
    # TOTAL TIMERS
    # ========================================================

    total_read_time = 0
    total_preprocess_time = 0
    total_s4_time = 0
    total_grouping_time = 0
    total_merge_time = 0
    total_corr_time = 0


    # ========================================================
    # 1. FILE ORGANIZATION
    # ========================================================

    file_org_start = time.time()

    files = f.find_files(cf.input_directory)

    receiverA_files, receiverB_files = f.org_receivers(
        files,
        cf.r_latitude,
        cf.r_longitude,
        cf.lat_tol,
        cf.lon_tol
    )

    file_org_time = time.time() - file_org_start


    print(f"\nFound {len(files)} total files")
    print(f"Receiver A: {len(receiverA_files)} files")
    print(f"Receiver B: {len(receiverB_files)} files")

    if len(receiverA_files) == 0:
        raise ValueError(
            "No files were assigned to Receiver A."
        )

    if len(receiverB_files) == 0:
        raise ValueError(
            "No files were assigned to Receiver B."
        )

    print(
        f"File organization: "
        f"{file_org_time:.3f} seconds"
    )


    # ========================================================
    # 2. FILE PAIRING
    # ========================================================

    pair_start = time.time()

    paired_files = f.pair_receiver_files(
        receiverA_files,
        receiverB_files,
        cf
    )

    pair_time = time.time() - pair_start

    print(
        f"Created {len(paired_files)} valid file pairs"
    )

    print(
        f"File pairing: "
        f"{pair_time:.3f} seconds"
    )


    # ========================================================
    # 3. PARALLEL PROCESSING
    # ========================================================

    all_scint = []

    parallel_start = time.time()

    with ProcessPoolExecutor(
        max_workers=cf.max_workers
    ) as executor:

        results = executor.map(
            process_s4_file_pair,
            paired_files
        )

        for result in results:

            # ----------------------------------------------
            # Collect scintillation results
            # ----------------------------------------------

            all_scint.extend(
                result['results']
            )


            # ----------------------------------------------
            # Collect timing results
            # ----------------------------------------------

            total_read_time += (
                result['read_time']
            )

            total_preprocess_time += (
                result['preprocess_time']
            )

            total_s4_time += (
                result['s4_time']
            )

            total_grouping_time += (
                result['grouping_time']
            )

            total_merge_time += (
                result['merge_time']
            )

            total_corr_time += (
                result['corr_time']
            )


    parallel_time = time.time() - parallel_start


    # ========================================================
    # 4. CREATE DATAFRAME
    # ========================================================

    dataframe_start = time.time()

    cross_cor = pd.DataFrame(all_scint)

    dataframe_time = time.time() - dataframe_start

    print(
        f"\nProcessed {len(cross_cor)} "
        f"scintillation events"
    )

    print(
        f"DataFrame creation: "
        f"{dataframe_time:.3f} seconds"
    )


    # ========================================================
    # 5. DISTANCE CALCULATION
    # ========================================================

    distance_start = time.time()

    cross_cor['distance (km)'] = cross_cor.apply(
        lambda row: f.calc_dist(
            row['r_A'],
            row['r_B']
        ),
        axis=1
    )

    distance_time = time.time() - distance_start

    print(
        f"Distance calculation: "
        f"{distance_time:.3f} seconds"
    )


    # ========================================================
    # 6. SAVE OUTPUT FILES
    # ========================================================

    save_start = time.time()

    base_name = Path(
        cf.cross_correlation_file
    ).stem

    output_folder = Path(
        cf.output_folder
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Hemisphere determination
    # --------------------------------------------------------

    lat_letter = (
        'N'
        if cf.r_latitude >= 0
        else 'S'
    )

    lon_letter = (
        'E'
        if cf.r_longitude >= 0
        else 'W'
    )

    lat = abs(cf.r_latitude)
    lon = abs(cf.r_longitude)


    # --------------------------------------------------------
    # Save each day
    # --------------------------------------------------------

    for day, day_df in cross_cor.groupby(
        cross_cor['minute'].dt.date
    ):

        day_str = pd.Timestamp(day).strftime(
            '%Y%m%d'
        )

        output_path = (
            output_folder /
            (
                f'{base_name}{day_str}'
                f'{lat:.3f}{lat_letter}'
                f'{lon:.3f}{lon_letter}.pq'
            )
        )

        day_df.to_parquet(
            output_path,
            index=False
        )

        print(
            f"Saved to {output_path.name}"
        )


    save_time = time.time() - save_start


    # ========================================================
    # 7. TOTAL RUNTIME
    # ========================================================

    total_runtime = time.time() - start


    print("\n")
    print("==============================================")
    print("             RUNTIME BREAKDOWN")
    print("==============================================")

    print(
        f"File organization:      "
        f"{file_org_time:.3f} sec"
    )

    print(
        f"File pairing:           "
        f"{pair_time:.3f} sec"
    )

    print(
        f"Reading files:          "
        f"{total_read_time:.3f} sec"
    )

    print(
        f"Preprocessing:          "
        f"{total_preprocess_time:.3f} sec"
    )

    print(
        f"S4 calculation:         "
        f"{total_s4_time:.3f} sec"
    )

    print(
        f"Grouping:               "
        f"{total_grouping_time:.3f} sec"
    )

    print(
        f"Merge:                  "
        f"{total_merge_time:.3f} sec"
    )

    print(
        f"Cross-correlation:      "
        f"{total_corr_time:.3f} sec"
    )

    print(
        f"DataFrame creation:     "
        f"{dataframe_time:.3f} sec"
    )

    print(
        f"Distance calculation:   "
        f"{distance_time:.3f} sec"
    )

    print(
        f"Saving files:           "
        f"{save_time:.3f} sec"
    )

    print("----------------------------------------------")

    print(
        f"Parallel processing:    "
        f"{parallel_time:.3f} sec"
    )

    print(
        f"TOTAL WALL TIME:        "
        f"{total_runtime:.3f} sec"
    )

    print("==============================================")


    # ========================================================
    # 8. PIE CHART
    # ========================================================

    processes = [
        "File organization",
        "File pairing",
        "Reading files",
        "Preprocessing",
        "S4 calculation",
        "Grouping",
        "Merge",
        "Cross-correlation",
        "DataFrame creation",
        "Distance calculation",
        "Saving files"
    ]

    times = [
        file_org_time,
        pair_time,
        total_read_time,
        total_preprocess_time,
        total_s4_time,
        total_grouping_time,
        total_merge_time,
        total_corr_time,
        dataframe_time,
        distance_time,
        save_time
    ]


if __name__ == '__main__':
    main()