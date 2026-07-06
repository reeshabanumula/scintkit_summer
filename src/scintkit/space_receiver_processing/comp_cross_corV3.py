import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import importlib
import functions as f
from scintkit.preprocessing.format import temp_formating
from scintkit.services.phase_detrend import detect_sampling_rate
import time
import CONFIG as cf


start = time.time()

# import pqs
dfa = pd.read_parquet(cf.Data_folder1)
dfb = pd.read_parquet(cf.Data_folder2)

print(f"time to read files: {time.time() - start:.3f} seconds")

# filter dfs to contain certain elevation
dfa = dfa[dfa['elev'] > 20].copy()
dfb = dfb[dfb['elev'] > 20].copy()

# fit distance into here?

print(dfa)

# individual sampling rates
dfa = temp_formating(dfa)
dfb = temp_formating(dfb)


print(dfa)


samp_ra = detect_sampling_rate(dfa)
samp_rb = detect_sampling_rate(dfb)

# print(samp_ra)
# print(samp_rb)

dt = 1 / samp_ra

# merge 2 receiver dfs
merged = dfa.merge(dfb, on=["datetime", "svid", "cons"], suffixes=("_A", "_B"))
merged['snr_diff'] = abs(merged['snr1_A'] - merged['snr1_B'])

# need to normalize time from datetime to just time in s

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

        sig_lina = f.db2lin(group['snr1_A'])
        s4a = np.std(sig_lina) / np.mean(sig_lina)

        sig_linb = f.db2lin(group['snr1_B'])
        s4b = np.std(sig_linb) / np.mean(sig_linb)

        if s4a > thresh or s4b > thresh:

            # check threshold of scintillation and compute correlation
            correlation, lag_b, cor_norm, lag_norm = f.cross_correlation(group["snr1_A"], group["snr1_B"])

            # dt = group['time_sec'].diff().median()
            time_delay = lag_b * dt

            scint.append({
                'svid': svid,
                'cons': cons,
                'minute': min,
                's4A': s4a,
                's4B': s4b,
                

                #adding elev and azim
                'elev' : group['elev_A'].mean(),
                'azim' : group['azim_A'].mean(),
                
                #adding location, using the location at the start of each minute not the mean, can be changed
                'r1loc' : (group['lat_A'].iloc[0]/10000, group['lon_A'].iloc[0]/10000, group['hei_A'].mean()/1000),
                'r2loc' : (group['lat_B'].iloc[0]/10000, group['lon_B'].iloc[0]/10000, group['hei_B'].mean()/1000),

                'corr_norm': cor_norm,
                'lag_norm': lag_norm,
                'max_corr': correlation,
                #'best_lag': lag_b,
                'time_delay': time_delay
            })

# create dataframe storing all scintillation events
cross_cor = pd.DataFrame(scint)

#add elevation and azimuth to df here and locations r1 and r2

print(f"Processed {len(cross_cor)} scintillation events")
print(f"Runtime: {time.time() - rstart:.3f} seconds")


#add elev and azimuth #going to use average/mean value for each minute Added them into df in the loop as we can just take the average of each minute
#done

cross_cor.to_parquet('src/scintkit/space_receiver_processing/_corrs.pq')




