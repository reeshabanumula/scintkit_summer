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

importlib.reload(f)
importlib.reload(cf)
start = time.time()

#prepping test example dfs

results = []
# import pqs
dfa = pd.read_parquet(r'C:\Users\irees\Downloads\Summer_learning\research26\Brazil_22_lvl0\scintpi3_20221004_2000_359072.7500W_72126.9375S_v325_lvl0.pq')
dfb = pd.read_parquet(r'C:\Users\irees\Downloads\Summer_learning\research26\Brazil_22_lvl0\scintpi3_20221004_2000_359060.7812W_72122.4141S_v325_lvl0.pq')

print(f"time to read files: {time.time() - start:.3f} seconds")

#rAloc = f.extract_coord(dfa)
#rBloc = f.extract_coord(dfb)

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



def add_s4(df):
    df = temp_formating(df).copy()

    agg_dict = {}

    for i in ("1", "2", "3"):
        snr_col = f"snr{i}"

        if snr_col in df.columns:
            agg_dict[f"s4_{i}"] = (snr_col, compute_s4)

    if not agg_dict:
        return df

    s4_products = (
        df.groupby(["prn", "minbin"], sort=False)
        .agg(**agg_dict)
        .reset_index()
    )

    return df.merge(
        s4_products,
        on=["prn", "minbin"],
        how="left",
    )


def compute_s4(snr):
    snr = snr.dropna()
    if len(snr) == 0:
        return np.nan

    lin_snr = 10 ** (snr / 10)
    mean = np.mean(lin_snr)
    std = np.std(lin_snr)

    return std / mean if mean > 0 else np.nan


####### FIND SCINTILLATION WITH s4 #############
x_s4_time = time.time()

print('adding s4')
dfa = add_s4(dfa)
dfb = add_s4(dfb)

dfa_int = dfa[(dfa.s4_1 > cf.thresh) & (dfa.s4_2 > cf.thresh)]
dfb_int = dfb[(dfb.s4_1 > cf.thresh) &(dfb.s4_2 > cf.thresh)]

print(f'done adding s4 {time.time() - x_s4_time}')


s4_time = time.time()
print('starting merge')
###### MERGE FILES ########
s4_summary = dfa_int.merge(dfb_int, on=["datetime", "svid", "cons"], suffixes=("_A", "_B"))
interesting = s4_summary

print(f'ending merge time taken = {time.time() - s4_time}')
print(f'intersting (scintillation) events: {len(interesting)}')

#changes to make looping using interesting as a bookmark more efficent
dfa["min_floor"] = dfa["datetime"].dt.floor("min")
dfb["min_floor"] = dfb["datetime"].dt.floor("min")

groupsA = dict(dfa.groupby(["svid", "cons", "min_floor"]))
groupsB = dict(dfb.groupby(["svid", "cons", "min_floor"]))


###### RUN CROSS CORRELATION #######

for _, event in interesting.iterrows():

    # Get event information
    minute = event["minute"]
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
    if len(group) < 10:
        continue

    correlation, lag_b, cor_norm, lag_norm = f.cross_correlation(group["snr1_A"], group["snr1_B"])
    autoA_max, autoA_lagb, autoA_cor, autoA_lags = f.cross_correlation(group['snr1_A'], group['snr1_A'])
    autoB_max, autoB_lagb, autoB_cor, autoB_lags = f.cross_correlation(group['snr1_B'], group['snr1_B'])

    # check threshold of scintillation and compute correlation and run auto correlation

    time_delay = lag_b * dt

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

        'auto_cor_A' : autoA_cor,
        'auto_cor_A_max' : autoA_max,
        'auto_cor_B' : autoB_cor,
        'auto_cor_B_max' : autoB_max,


        'corr_norm': cor_norm,
        'lag_norm': lag_norm,
        'max_corr': correlation,
        'best_lag': lag_b,
        'time_delay': time_delay
    })

# create dataframe storing all scintillation events
cross_cor = pd.DataFrame(results)

print(f"Processed {len(cross_cor)} scintillation events")


print(f"Runtime: {time.time() - start:.3f} seconds")









