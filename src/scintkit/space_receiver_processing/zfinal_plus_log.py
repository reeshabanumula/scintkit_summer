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

#
from concurrent.futures import ProcessPoolExecutor
#



importlib.reload(f)
importlib.reload(cf)


def process_s4_file_pair(pair):

    fileA, fileB = pair

    results = []

    print(f"\nProcessing:")
    print(fileA)
    print(fileB)


####### LOAD AND PREPROCESS FILES #########
    #read single files # import pqs
    dfa = pd.read_parquet(fileA)
    dfb = pd.read_parquet(fileB)

    rAloc = f.extract_coord(fileA)
    rBloc = f.extract_coord(fileB)


    # filter dfs to contain certain elevation
    dfa = dfa[dfa['elev'] > cf.elevation_filter].copy()
    dfb = dfb[dfb['elev'] > cf.elevation_filter].copy()

    # individual sampling rates
    dfa = temp_formating(dfa)
    dfb = temp_formating(dfb)

    samp_ra = detect_sampling_rate(dfa)
    dt = 1 / samp_ra

####### FIND SCINTILLATION WITH s4 #############
    x_s4_time = time.time()

    print('adding s4')
    dfa = f.add_s4(dfa)
    dfb = f.add_s4(dfb)

    s4A = (dfa[(dfa.s4_1 > cf.thresh) & (dfa.s4_2 > cf.thresh)] [["minbin", "svid", "cons", "s4_1", "s4_2"]].drop_duplicates())
    s4B = (dfb[(dfb.s4_1 > cf.thresh) &(dfb.s4_2 > cf.thresh)] [["minbin", "svid", "cons", "s4_1", "s4_2"]].drop_duplicates())

    print(f'done adding s4, time: {time.time() - x_s4_time}s')


###### MERGE FILES ########
    s4_summary = s4A.merge(s4B, on=["minbin", "svid", "cons"], suffixes=("_A", "_B"))

    interesting = s4_summary
    print(f'intersting (scintillation) events: {len(interesting)}')

    #changes to make looping using interesting as a bookmark more efficent
    dfa["min_floor"] = dfa["datetime"].dt.floor("min")
    dfb["min_floor"] = dfb["datetime"].dt.floor("min")

    groupsA = dict(list(dfa.groupby(["svid", "cons", "min_floor"])))
    groupsB = dict(list(dfb.groupby(["svid", "cons", "min_floor"])))



###### RUN CROSS CORRELATION #######

    for _, event in interesting.iterrows():

        # Get event information
        minute = event["minbin"]
        svid = event["svid"]
        cons = event["cons"]

        # Extract Receiver A data for this event #changing the extraction so that it doesnt need to loop through the whole df, makes it faster

        groupA = groupsA.get((svid, cons, minute))
        groupB = groupsB.get((svid, cons, minute))

        if groupA is None or groupB is None:
            continue


        #groupA = dfa[(dfa["svid"] == svid) & (dfa["cons"] == cons) & (dfa["datetime"].dt.floor("min") == minute)]
        # Extract Receiver B data for this event
        #groupB = dfb[(dfb["svid"] == svid) & (dfb["cons"] == cons) & (dfb["datetime"].dt.floor("min") == minute)]

        if len(groupA) < 10 or len(groupB) < 10:
            continue


        group = groupA.merge(groupB, on=["datetime", "svid", "cons"], suffixes=("_A", "_B"))
        group = f.handle_nan(group, cf.nan_method, sig_columns = ['snr1_A', 'snr1_B', 'snr2_A', 'snr2_B'])

        if len(group) < 10:
            continue


            #SNR 1
        correlation, lag_b, cor_norm, lag_norm = f.cross_correlation(group["snr1_A"], group["snr1_B"])
        autoA_max, autoA_lagb, autoA_cor, autoA_lags = f.cross_correlation(group['snr1_A'], group['snr1_A'])
        autoB_max, autoB_lagb, autoB_cor, autoB_lags = f.cross_correlation(group['snr1_B'], group['snr1_B'])

            #SNR 2
        correlation2, lag2, cor2, lag_n_2 = f.cross_correlation(group['snr2_A'], group['snr2_B'])
        autoA2_max, autoA_lag2, autoA_cor2, nautoA_lag2 = f.cross_correlation(group['snr2_A'], group['snr2_A'])
        autoB2_max, autoB_lag2, autoB_cor2, nautoB_lag2 = f.cross_correlation(group['snr2_B'], group['snr2_B'])


        # check threshold of scintillation and compute correlation and run auto correlation

        time_delay = lag_b * dt
        time_delay2 = lag2 * dt

        results.append({
            'minute': minute,
            'prn' : group['prn_B'].iloc[0],
            's4_1_A': event['s4_1_A'],
            's4_1_B': event['s4_1_B'],
            's4_2_A': event['s4_2_A'],
            's4_2_B': event['s4_2_B'],
            

            #adding elev and azim
            'elev' : group['elev_A'].mean(),
            'azim' : group['azim_A'].mean(),
            
            #adding location, using the location at the start of each minute not the mean, can be changed
            'r_A' : rAloc,
            'r_B' : rBloc,

            #snr 1 auto
            'auto_cor_A_1' : autoA_cor,
            'auto_cor_A_max_1' : autoA_max,
            'auto_cor_B_1' : autoB_cor,
            'auto_cor_B_max_1' : autoB_max,

            #SNR 1
            'corr_norm_1': cor_norm,
            'lag_norm_1': lag_norm,
            'max_corr_1': correlation,
            'best_lag_1': lag_b,
            'time_delay_1': time_delay,


            #snr 2 auto
            'auto_cor_A_2' : autoA_cor2,
            'auto_cor_A_max_2' : autoA2_max,
            'auto_cor_B_2' : autoB_cor2,
            'auto_cor_B_max_2' : autoB2_max,

            #SNR 2
            'corr_norm_2': cor2,
            'lag_norm_2': lag_n_2,
            'max_corr_2': correlation2,
            'best_lag_2': lag2,
            'time_delay_2': time_delay2
            })

    print(f"Finished {Path(fileA).name}")

    return results, fileA, fileB



#########################################################################################################################

def main():
    start = time.time()

    f.initialize_log(cf.log_file)

    processing_folder = f.create_processing_scratch(cf.storage_folder, cf.scratch_folder, cf.input_pattern, cf.temp_root, cf.max_workers, cf.verbose)
    print(f'processing folder returned: \n {processing_folder}')




    #create file organization code:

    files = f.find_files(str(processing_folder))

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


    #file pairing code


    paired_files = f.pair_receiver_files(receiverA_files, receiverB_files, cf)
    print(f"Created {len(paired_files)} valid file pairs")

    all_scint = []


    with ProcessPoolExecutor(max_workers = cf.max_workers) as executor:

        results = executor.map(
            process_s4_file_pair,
            paired_files
        )

        for result, fileA, fileB in results:
            all_scint.extend(result)
            f.log_processed_pair(fileA, fileB, cf.log_file)

    cross_cor = pd.DataFrame(all_scint)
    print(f"Processed {len(cross_cor)} scintillation events")
    print(f"Runtime: {time.time() - start:.3f} seconds")

    cross_cor['distance (km)'] = cross_cor.apply(lambda row: f.calc_dist(row['r_A'], row['r_B']), axis=1)
    
    print("Calculated distances")
    print(f"Runtime: {time.time()-start:.3f} seconds")

    
    #file creation
    #change to file for each day

    base_name = Path(cf.cross_correlation_file).stem


    output_folder = Path(cf.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    #hemisphere determination
    lat_letter = 'N' if cf.r_latitude >= 0 else 'S'
    lon_letter = 'E' if cf.r_longitude >= 0 else 'W'
    lat = abs(cf.r_latitude)
    lon = abs(cf.r_longitude)

    for day, day_df in cross_cor.groupby(cross_cor['minute'].dt.date):
        day_str = pd.Timestamp(day).strftime('%Y%m%d')

        output_path = output_folder / (f'{base_name}_{day_str}_'
                                    f'{lat:.3f}{lat_letter}_{lon:.3f}{lon_letter}.pq')
        day_df.to_parquet(output_path, index = False)

        print(f'Saved to {output_path.name}')

    f.cleanup_processing_scratch(processing_folder, cf.scratch_folder)

    print(f"Total Time: {time.time() - start:.3f} seconds")


if __name__ == '__main__':
    main()