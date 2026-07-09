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

importlib.reload(f)
importlib.reload(cf)

start = time.time()

print(f'started at {start}')
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



# import pqs
dfa = f.load_receiver(receiverA_files)
dfb = f.load_receiver(receiverB_files)

print(f"time to read files: {time.time() - start:.3f} seconds")

dfa = dfa.sort_values("datetime").reset_index(drop=True)
dfb = dfb.sort_values("datetime").reset_index(drop=True)


# filter dfs to contain certain elevation
dfa = dfa[dfa['elev'] > 20].copy()
dfb = dfb[dfb['elev'] > 20].copy()



# individual sampling rates
dfa = temp_formating(dfa)
dfb = temp_formating(dfb)


samp_ra = detect_sampling_rate(dfa)
samp_rb = detect_sampling_rate(dfb)

dt = 1 / samp_ra

# merge 2 receiver dfs
merged = dfa.merge(dfb, on=["datetime", "svid", "cons"], suffixes=("_A", "_B"))
merged['snr_diff'] = abs(merged['snr1_A'] - merged['snr1_B'])


print(f"time to merge dfs: {time.time() - start:.3f} seconds")


print(f"time to create new time: {time.time() - start:.3f} seconds")

thresh = cf.thresh  # threshold for s4 scintillation measurement


#### start cross corr file creation

rstart = time.time()

scint = []

sat_groups = merged.groupby(['svid', 'cons'])

for (svid, cons), sat_group in sat_groups:

    # group the data into different satellites
    sat_group = f.datetime_to_seconds(sat_group)

    min_groups = sat_group.groupby(sat_group['datetime'].dt.floor('min'))

    # with the chosen satellite for this iteration
    # find s4 and then determine scintillation
    for min, group in min_groups:
        
        group = f.handle_nan(group, cf.nan_method)
        #add nan processing
        if len(group) < 10: #makes sure there is enough samples to actually process data
            continue

        sig_lina = f.db2lin(group['snr1_A'])
        s4a = np.std(sig_lina) / np.mean(sig_lina)

        sig_linb = f.db2lin(group['snr1_B'])
        s4b = np.std(sig_linb) / np.mean(sig_linb)

        if s4a > thresh or s4b > thresh:

            # check threshold of scintillation and compute correlation and run auto correlation
            correlation, lag_b, cor_norm, lag_norm = f.cross_correlation(group["snr1_A"], group["snr1_B"])
            autoA_max, autoA_lagb, autoA_cor, autoA_lags = f.cross_correlation(group['snr1_A'], group['snr1_A'])
            autoB_max, autoB_lagb, autoB_cor, autoB_lags = f.cross_correlation(group['snr1_B'], group['snr1_B'])

            # dt = group['time_sec'].diff().median()
            time_delay = lag_b * dt

            scint.append({
                'minute': min,
                'prn' : group['prn_B'].iloc[0],
                's4A': s4a,
                's4B': s4b,
                

                #adding elev and azim
                'elev' : group['elev_A'].mean(),
                'azim' : group['azim_A'].mean(),
                
                #adding location, using the location at the start of each minute not the mean, can be changed
                'rAloc' : (group['lat_A'].iloc[0]/10000, group['lon_A'].iloc[0]/10000, group['hei_A'].mean()/1000),
                'rBloc' : (group['lat_B'].iloc[0]/10000, group['lon_B'].iloc[0]/10000, group['hei_B'].mean()/1000),

                #'auto_cor_A' : autoA_cor,
                'auto_cor_Amax' : autoA_max,
                #'auto_cor_B' : autoB_cor,
                'auto_cor_Bmax' : autoB_max,


                'corr_norm': cor_norm,
                'lag_norm': lag_norm,
                'max_corr': correlation,
                #'best_lag': lag_b,
                'time_delay': time_delay
            })

# create dataframe storing all scintillation events with distance
cross_cor = pd.DataFrame(scint)

print(f"Processed {len(cross_cor)} scintillation events")
print(f"Runtime: {time.time() - rstart:.3f} seconds")
# add distance
cross_cor['distance (km)'] = cross_cor.apply(lambda row: f.calc_dist(row['rAloc'], row['rBloc']), axis = 1)

print(f"calculated distances")
print(f"Runtime: {time.time() - rstart:.3f} seconds")


output_path = Path(cf.output_folder) / cf.cross_correlation_file

print(f"Total Time: {time.time() - start:.3f} seconds")

#fix the output of the df below

output_path = Path(__file__).parent / "_corrs.pq"

cross_cor.to_parquet(output_path, index=False)


