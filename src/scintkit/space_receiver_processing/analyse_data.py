# %% Imports and plotting settings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd


filedir = Path('/Users/isaac/Downloads/corss/')

# Figure 4-style limits from Gomez Socola et al. (2025).
UT_START = 20
UT_END = 24
S4_LIMITS = (0, 1)
VELOCITY_LIMITS = (20, 180)


# %% Load the daily cross-correlation files
files = sorted(filedir.glob('*.pq'))
if not files:
    raise FileNotFoundError(f'No .pq files found in {filedir}')

frames = []
for file in files:
    print(file)
    frames.append(pd.read_parquet(file))

df = pd.concat(frames, ignore_index=True)
df['minute'] = pd.to_datetime(df['minute'])


# %% Keep SBAS PRN 136 and apply the original quality/elevation thresholds
# The explicit constellation check prevents another constellation's SVID 136
# from being included if a future file uses overlapping numeric identifiers.
cons = df['cons'].astype(str).str.upper()
prn = df['prn'].astype(str).str.upper().str.replace('PRN', '', regex=False)
svid = pd.to_numeric(df['svid'], errors='coerce')

is_sbas_136 = (
    (cons.isin(['SBAS', 'SBS']) & svid.eq(136))
    | prn.isin(['G10', 'SBAS136'])
)

df_sbas136 = df[is_sbas_136].copy()
if df_sbas136.empty:
    available = sorted(df['prn'].dropna().astype(str).unique())
    raise ValueError(
        'No SBAS PRN 136 rows were found. '
        f'The loaded files contain these PRNs: {available}'
    )

# Keep these cuts unchanged: they are the original analysis selection.
df1 = df_sbas136[
    (df_sbas136.max_auto_cross_corr_1 > 0.8)
    & (df_sbas136.decorrelation_time_A_1 < 50)
    & (df_sbas136.elev > 30)
    & (df_sbas136.elev < 90)
].copy()

# Use only 20:00 <= UT < 24:00 on every day.
df1['ut_hour'] = (
    df1['minute'].dt.hour
    + df1['minute'].dt.minute / 60
    + df1['minute'].dt.second / 3600
)
df1 = df1[df1['ut_hour'].between(UT_START, UT_END, inclusive='left')].copy()

print(f'Loaded rows: {len(df):,}')
print(f'SBAS PRN 136 rows: {len(df_sbas136):,}')
print(f'Rows after quality, elevation, and UT filters: {len(df1):,}')


# %% Calculate average S4, apparent velocity, and Briggs true velocity
# Figure 4(a) uses the average S4 measured by the two spaced receivers.
df1['s4_mean'] = df1[['s4_1_A', 's4_1_B']].mean(axis=1)

# Simple apparent drift: receiver separation divided by the peak time lag.
tau0 = pd.to_numeric(df1['time_delay_1'], errors='coerce')
distance_m = pd.to_numeric(df1['distance (km)'], errors='coerce') * 1000
df1['apparent_velocity_ms'] = distance_m / tau0

# Briggs et al. (1950): v_true = v_apparent / (1 + t0^2 / tau0^2).
# t0 is the receiver-A ACF time at the measured maximum cross-correlation.
t0 = pd.to_numeric(
    df1['auto_cor_A_time_at_max_corr_1'], errors='coerce'
)
df1['true_velocity_briggs_ms'] = df1['apparent_velocity_ms'] / (
    1 + (t0 / tau0) ** 2
)

df1.replace([np.inf, -np.inf], np.nan, inplace=True)


# %% Figure helper: date versus UT, matching the paper's color scales
def plot_date_ut(
    data,
    value_column,
    colorbar_label,
    panel_label,
    cmap,
    limits,
    title,
    marker_size=8,
    extend='neither',
    colorbar_ticks=None,
):
    plot_data = data.dropna(subset=['minute', 'ut_hour', value_column])
    if plot_data.empty:
        print(f'No finite values available for {title}')
        return None, None

    fig, ax = plt.subplots(figsize=(12, 4))
    points = ax.scatter(
        plot_data['minute'].dt.normalize(),
        plot_data['ut_hour'],
        c=plot_data[value_column],
        cmap=cmap,
        vmin=limits[0],
        vmax=limits[1],
        marker='s',
        s=marker_size,
        linewidths=0,
        rasterized=True,
    )

    ax.set_ylim(UT_START, UT_END)
    ax.set_ylabel('Time (UT)')
    ax.set_title(title)
    ax.text(
        0.015,
        0.95,
        panel_label,
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=13,
        fontstyle='italic',
    )

    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda hour, _: f'{int(hour):02d}:00')
    )
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    fig.autofmt_xdate(rotation=0, ha='center')

    colorbar = fig.colorbar(
        points,
        ax=ax,
        pad=0.015,
        extend=extend,
        ticks=colorbar_ticks,
    )
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    plt.show()
    return fig, ax


# %% S4 for SBAS PRN 136: average of receivers A and B
fig_s4, ax_s4 = plot_date_ut(
    df1,
    value_column='s4_mean',
    colorbar_label=r'$S_4$',
    panel_label='(a)',
    cmap='viridis',
    limits=S4_LIMITS,
    title='SBAS PRN 136: average $S_4$',
    colorbar_ticks=np.arange(0, 1.01, 0.2),
)


# %% Simple apparent drift = distance / time lag
fig_apparent, ax_apparent = plot_date_ut(
    df1,
    value_column='apparent_velocity_ms',
    colorbar_label=r"$v'_{scint}$ (m/s)",
    panel_label='(b)',
    cmap='jet',
    limits=VELOCITY_LIMITS,
    title='SBAS PRN 136: apparent drift',
    extend='both',
    colorbar_ticks=np.arange(20, 181, 20),
)


# %% True drift after the Briggs decorrelation correction
fig_true, ax_true = plot_date_ut(
    df1,
    value_column='true_velocity_briggs_ms',
    colorbar_label=r'$v_{scint}$ (m/s)',
    panel_label='(c)',
    cmap='jet',
    limits=VELOCITY_LIMITS,
    title='SBAS PRN 136: Briggs-corrected true drift',
    extend='both',
    colorbar_ticks=np.arange(20, 181, 20),
)


# %% Three-minute bins, retained as a separate dataframe
df1['minute_3min'] = df1['minute'].dt.floor('3min')

df_3min = (
    df1.groupby('minute_3min', as_index=False)
    .agg(
        s4_mean=('s4_mean', 'mean'),
        apparent_velocity_ms=('apparent_velocity_ms', 'mean'),
        apparent_velocity_std_ms=('apparent_velocity_ms', 'std'),
        true_velocity_briggs_ms=('true_velocity_briggs_ms', 'mean'),
        true_velocity_briggs_std_ms=('true_velocity_briggs_ms', 'std'),
        sample_count=('true_velocity_briggs_ms', 'count'),
    )
    .rename(columns={'minute_3min': 'minute'})
)
df_3min['ut_hour'] = (
    df_3min['minute'].dt.hour
    + df_3min['minute'].dt.minute / 60
    + df_3min['minute'].dt.second / 3600
)


# %% Three-minute mean apparent drift
fig_apparent_3min, ax_apparent_3min = plot_date_ut(
    df_3min,
    value_column='apparent_velocity_ms',
    colorbar_label=r"3 min mean $v'_{scint}$ (m/s)",
    panel_label='(d)',
    cmap='jet',
    limits=VELOCITY_LIMITS,
    title='SBAS PRN 136: 3 min mean apparent drift',
    marker_size=12,
    extend='both',
    colorbar_ticks=np.arange(20, 181, 20),
)


# %% Three-minute mean Briggs-corrected true drift
fig_true_3min, ax_true_3min = plot_date_ut(
    df_3min,
    value_column='true_velocity_briggs_ms',
    colorbar_label=r'3 min mean $v_{scint}$ (m/s)',
    panel_label='(e)',
    cmap='jet',
    limits=VELOCITY_LIMITS,
    title='SBAS PRN 136: 3 min mean Briggs-corrected true drift',
    marker_size=12,
    extend='both',
    colorbar_ticks=np.arange(20, 181, 20),
)

# %%
