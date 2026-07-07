import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import CONFIG as cf

def cross_correlation(sig1, sig2):
    norm1 = (sig1 - np.mean(sig1))/np.std(sig1)
    norm2 = (sig2 - np.mean(sig2))/np.std(sig2)

    cor = np.correlate(norm1, norm2, mode = 'full')
    cor_norm = cor/ (np.linalg.norm(norm1) * np.linalg.norm(norm2))

    lag_norm = sp.correlation_lags(len(norm1),len(norm2), mode = 'full')
    
    #identify : max correlation, and lag
    #figure out how to correlate lag to time shift

    max_corr = np.max(cor_norm)
    best_lag = lag_norm[np.argmax(cor_norm)]
    
    return max_corr, best_lag, cor_norm, lag_norm


# helps create s4
def db2lin(sig_db):
    return 10 ** (sig_db / 10)


def datetime_to_seconds(df, time_column="datetime"):
    df = df.copy()

    # Seconds since first measurement
    df["time_sec"] = (df[time_column] - df[time_column].iloc[0]).dt.total_seconds()
    return df

#works for 1 row of r1loc and r2loc make it work for entire column
def haversine(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    R = cf.R_earth #km


    lamba = lon2 - lon1
    deta = lat2 - lat1

    a = (np.sin(deta/2))**2 + np.cos(lat1)*np.cos(lat2)*(np.sin(lamba/2))**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c
    
    return(d)

def calc_dist(rloc1, rloc2):
    lat1, lon1, hei1 = rloc1
    lat2, lon2, hei2 = rloc2

    distances_2d = haversine(lon1, lat1, lon2, lat2)
    heights = (hei2 - hei1)

#3D distance now using pythagorean theorem and curvature haversine distance
    distance = np.sqrt(distances_2d**2 + heights**2)
    #print(f'{distance} km')
    return distance

def handle_nan(df, method):
    if method == 'interpolate':
        #interpolate nan values using previous points
        df["snr1_A"] = df["snr1_A"].ffill()
        df["snr1_B"] = df["snr1_B"].ffill()
    elif method == 'drop':
        #drop all nan values
        df = df.dropna(subset=["snr1_A", "snr1_B"])
    elif method == 'none':
        #leave data unprocessed
        pass
    else:
        #error
        raise ValueError(
            f"Invalid NaN handling method: '{method}'. "
            "Choose 'drop', 'interpolate', or 'none'."
        )
    return df