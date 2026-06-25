import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import importlib
import functions as f
from scintkit.preprocessing.format import temp_formating
from scintkit.services.phase_detrend import detect_sampling_rate
import time

start = time.time()

# import pqs
dfa = pd.read_parquet(r'C:\Users\irees\Downloads\Summer_learning\research26\Brazil_22_lvl0\scintpi3_20221004_2000_359060.7812W_72122.4141S_v325_lvl0.pq')

dfb = pd.read_parquet(r'C:\Users\irees\Downloads\Summer_learning\research26\Brazil_22_lvl0\scintpi3_20221004_2000_359072.7500W_72126.9375S_v325_lvl0.pq')

print(f"time to read files: {time.time() - start:.3f} seconds")

# filter dfs to contain certain elevation
dfa = dfa[dfa['elev'] > 20]
dfb = dfb[dfb['elev'] > 20]

# fit distance into here?

# individual sampling rates
dfa = temp_formating(dfa)
dfb = temp_formating(dfb)

samp_ra = detect_sampling_rate(dfa)
samp_rb = detect_sampling_rate(dfb)

print(samp_ra)
print(samp_rb)

# merge 2 receiver dfs
merged = dfa.merge(dfb, on=["datetime", "svid", "cons"], suffixes=("_A", "_B"))
merged['snr_diff'] = abs(merged['snr1_A'] - merged['snr1_B'])

# add temp formatting and detect sampling rate
merged = temp_formating(merged)

print(merged.columns.tolist())

# need to normalize time from datetime to just time in s


print(f"time to create new time: {time.time() - start:.3f} seconds")

thresh = 0.2  # threshold for s4 scintillation measurement

samp_r = detect_sampling_rate(merged)
print(samp_r)

dt = 1 / samp_r

########################
1 / 0

rstart = time.time()

scint = []

sat_groups = merged.groupby(['svid', 'cons'])

for (svid, cons), sat_group in sat_groups:

    # group the data into different satellites
    sat_group = f.datetime_to_seconds(sat_group)

    min_groups = sat_group.groupby(
        sat_group['datetime'].dt.floor('min')
    )

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
                'corr_norm': cor_norm,
                'lag_norm': lag_norm,
                'max_corr': correlation,
                'best_lag': lag_b,
                'time_delay': time_delay
            })

# create dataframe storing all scintillation events
cross_cor = pd.DataFrame(scint)

print(f"Processed {len(cross_cor)} scintillation events")
print(f"Runtime: {time.time() - rstart:.3f} seconds")

# print(cross_cor)

##################
# plotting method of normal cross correlation
# plot by finding 1 satellite from df and then choosing 1 event

sat = cross_cor[
    (cross_cor['svid'] == 10) &
    (cross_cor['cons'] == 0)
]

event = sat.iloc[10]

print(sat['max_corr'].iloc[10])
print(sat['best_lag'].iloc[10])
print(sat['time_delay'].iloc[10])

plt.plot(event['lag_norm'], event['corr_norm'])
plt.xlabel('lag')
plt.ylabel('correlation (normalized)')
plt.title('Normalized Cross Correlation')
plt.show()

# currently does not have distance calculations in this file