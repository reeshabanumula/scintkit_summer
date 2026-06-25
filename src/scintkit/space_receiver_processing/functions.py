import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp


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
