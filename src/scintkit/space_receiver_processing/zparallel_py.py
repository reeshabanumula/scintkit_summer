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
import os

#
from concurrent.futures import ProcessPoolExecutor
#

importlib.reload(f)
importlib.reload(cf)



#parallel process function
def process_file_pair(pair):

    rstart = time.time()

    fileA, fileB = pair

    print(f"\nProcessing:")
    print(fileA)
    print(fileB)

    #read single files # import pqs
    dfa = pd.read_parquet(fileA)
    dfb = pd.read_parquet(fileB)

    rAloc = f.extract_coord(fileA)
    rBloc = f.extract_coord(fileB)


    dfa = dfa.sort_values("datetime").reset_index(drop=True)
    dfb = dfb.sort_values("datetime").reset_index(drop=True)

    # filter dfs to contain certain elevation
    dfa = dfa[dfa['elev'] > cf.elevation_filter].copy()
    dfb = dfb[dfb['elev'] > cf.elevation_filter].copy()


    # individual sampling rates
    dfa = temp_formating(dfa)
    dfb = temp_formating(dfb)


    samp_ra = detect_sampling_rate(dfa)
    dt = 1 / samp_ra

    #add s4 filtering here in method 2

    # merge 2 receiver dfs
    merged = dfa.merge(dfb, on=["datetime", "svid", "cons"], suffixes=("_A", "_B"))
    merged['snr_diff'] = abs(merged['snr1_A'] - merged['snr1_B'])


    thresh = cf.thresh  # threshold for s4 scintillation measurement

    #### start cross corr file creation


    scint = []

    sat_groups = merged.groupby(['svid', 'cons'])

    for (svid, cons), sat_group in sat_groups:

        # group the data into different satellites
        sat_group = f.datetime_to_seconds(sat_group)

        min_groups = sat_group.groupby(sat_group['datetime'].dt.floor('min'))

        # with the chosen satellite for this iteration
        # find s4 and then determine scintillation
        for min, group in min_groups:
            
            group = f.handle_nan(group, cf.nan_method, sig_columns = ['snr1_A', 'snr1_B'])
            #add nan processing
            if len(group) < 10: #makes sure there is enough samples to actually process data
                continue

            #snr 1
            sig_1_lina = f.db2lin(group['snr1_A'])
            s4_1_a = np.std(sig_1_lina) / np.mean(sig_1_lina)

            sig_1_linb = f.db2lin(group['snr1_B'])
            s4_1_b = np.std(sig_1_linb) / np.mean(sig_1_linb)

            #snr 2 (snr 3 is all 0 or nan)
            sig_2_lina = f.db2lin(group['snr2_A'])
            s4_2_a = np.std(sig_2_lina) / np.mean(sig_2_lina)

            sig_2_linb = f.db2lin(group['snr2_B'])
            s4_2_b = np.std(sig_2_linb) / np.mean(sig_2_linb)




            if s4_1_a > thresh or s4_1_b > thresh:

                # check threshold of scintillation and compute correlation and run auto correlation
                correlation, lag_b, cor_norm, lag_norm = f.cross_correlation(group["snr1_A"], group["snr1_B"])
                autoA_max, autoA_lagb, autoA_cor, autoA_lags = f.cross_correlation(group['snr1_A'], group['snr1_A'])
                autoB_max, autoB_lagb, autoB_cor, autoB_lags = f.cross_correlation(group['snr1_B'], group['snr1_B'])

                time_delay = lag_b * dt

                scint.append({
                    'minute': min,
                    'prn' : group['prn_B'].iloc[0],
                    's4_1_a': s4_1_a,
                    's4_1_b': s4_1_b,
                    's4_2_a': s4_2_a,
                    's4_2_b': s4_2_b,
                    

                    #adding elev and azim
                    'elev' : group['elev_A'].mean(),
                    'azim' : group['azim_A'].mean(),
                    
                    #adding location, using the location at the start of each minute not the mean, can be changed
                    'r_a' : rAloc,
                    'r_b' : rBloc,

                    'auto_cor_a' : autoA_cor,
                    'auto_cor_a_max' : autoA_max,
                    'auto_cor_b' : autoB_cor,
                    'auto_cor_b_max' : autoB_max,


                    'corr_norm': cor_norm,
                    'lag_norm': lag_norm,
                    'max_corr': correlation,
                    'best_lag': lag_b,
                    'time_delay': time_delay
                })

    print(f"Finished {Path(fileA).name}")

    return scint


def main():
    
    start = time.time()

    print(f'started at {start- start}')
    #create file organization code:

    files = f.find_files(cf.input_directory)

    receiverA_files, receiverB_files = f.org_receivers(files, cf.r_latitude, cf.r_longitude, cf.lat_tol, cf.lon_tol)

        #finally have all the receiver data organized into 2 seperate files 

    # ----- Verification -----
    print(f"Found {len(files)} total files")
    print(f"Receiver A: {len(receiverA_files)} files")
    print(f"Receiver B: {len(receiverB_files)} files")

    if len(receiverA_files) == 0:
        raise ValueError("No files were assigned to Receiver A.")

    if len(receiverB_files) == 0:
        raise ValueError("No files were assigned to Receiver B.")

    print(f"time to load in files: {time.time() - start:.3f} seconds")


    #create file pairing code:

    paired_files = f.pair_receiver_files(receiverA_files, receiverB_files, cf)
    print(f"Created {len(paired_files)} valid file pairs")


    #code

    all_scint = []

    print(f"Using {os.cpu_count()} worker processes")

    with ProcessPoolExecutor(max_workers = 2) as executor:
        results = executor.map(process_file_pair, paired_files)

        for scint in results:
            all_scint.extend(scint)

    # create dataframe storing all scintillation events
    cross_cor = pd.DataFrame(all_scint)

    print(f"Processed {len(cross_cor)} scintillation events")
    print(f"Runtime: {time.time() - start:.3f} seconds")


    # add distance
    cross_cor['distance (km)'] = cross_cor.apply(lambda row: f.calc_dist(row['r_a'], row['r_b']),axis=1)

    print("calculated distances")
    print(f"Runtime: {time.time() - start:.3f} seconds")


    #file creation
    #change to file for each day

    base_name = Path(cf.cross_correlation_file).stem


    output_folder = Path(cf.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)


    for day, day_df in cross_cor.groupby(cross_cor['minute'].dt.date):
        day_str = pd.Timestamp(day).strftime('%Y%m%d')

        output_path = output_folder / (f'{base_name}_{day_str}'
                                    f'_A_{cf.r_latitude:.5f}_{cf.r_longitude:.5f}.pq')
        day_df.to_parquet(output_path, index = False)

        print(f'Saved to {output_path.name}')



    print(f"Total Time: {time.time() - start:.3f} seconds")


if __name__ == '__main__':
    main()






