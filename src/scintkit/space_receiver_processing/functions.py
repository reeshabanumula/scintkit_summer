import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import CONFIG as cf
import re
from datetime import datetime
import pandas as pd
from scintkit.preprocessing.format import temp_formating
from scintkit.pipelines.lvl0_convert_to_pq import run_conversion
from scintkit.services.phase_detrend import detect_sampling_rate
from pathlib import Path
import shutil
import time


#cross correlation function to normalized outputs
def cross_correlation(sig1, sig2):
    norm1 = (sig1 - np.mean(sig1))/np.std(sig1)
    norm2 = (sig2 - np.mean(sig2))/np.std(sig2)

    cor = sp.correlate(norm1, norm2, mode = 'full')
    cor_norm = cor/ (np.linalg.norm(norm1) * np.linalg.norm(norm2))

    lag_norm = sp.correlation_lags(len(norm1),len(norm2), mode = 'full')
    
    #identify : max correlation, and lag
    #figure out how to correlate lag to time shift

    max_corr = np.max(cor_norm)
    best_lag = lag_norm[np.argmax(cor_norm)]
    
    return max_corr, best_lag, cor_norm, lag_norm


# helps create s4 changes snr to linear
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

def handle_nan(df, method, sig_columns):
    if method == 'interpolate':

        for column in sig_columns:
            df[column] = df[column].ffill()

    elif method == 'drop':
        #drop all nan values
        df = df.dropna(subset = sig_columns) #run it on sat 10
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


#for organizing the files from a bigger folder

def extract_coord (file, height = 0):

    pattern = r'_(\d+\.\d+)([EW])_(\d+\.\d+)([NS])_'

    match = re.search(pattern, file.name)

    if match is None:
        raise ValueError(f"Could not read coordinates from {file.name}")

    longitude = float(match.group(1)) / 10000
    latitude = float(match.group(3)) / 10000

    return latitude, longitude, height

def org_receivers(files, reference_lat, reference_lon, lat_tol, lon_tol):
    
    receiverA = []
    receiverB =[]

    for file in files:
        lat , lon, hei = extract_coord (file, cf.r_height)

        if (abs(lat - reference_lat) <= lat_tol and abs(lon - reference_lon) <= lon_tol):
            receiverA.append(file)
        else:
            receiverB.append(file)

    return receiverA, receiverB


from pathlib import Path


def find_files(input_directory):

    input_directory = Path(input_directory)

    files = sorted(input_directory.glob("*.pq"))

    return files


#file pairing

def extract_file_datetime(file):
    """
    Returns the datetime contained in a ScintPi filename.

    Example:
    scintpi3_20250325_1552_359062.0938W...
            ↓
    datetime(2025, 3, 25, 15, 52)
    """

    match = re.search(r'_(\d{8})_(\d{4})_', file.name)

    if match is None:
        raise ValueError(f"Could not extract datetime from {file.name}")

    return datetime.strptime(match.group(1) + match.group(2),"%Y%m%d%H%M") #changes string to a datetime, last part tells python how to read this new value




def pair_receiver_files(receiverA_files, receiverB_files, cf): #takes in configuration file as a parameter

    receiverB_dict = {
        extract_file_datetime(file): file
        for file in receiverB_files
    }

    paired_files = []

    for fileA in receiverA_files:

        datetimeA = extract_file_datetime(fileA)

        # ---------------- EXACT ----------------

        if cf.pairing_mode == "exact":

            if datetimeA in receiverB_dict:
                paired_files.append((fileA, receiverB_dict[datetimeA]))

            elif cf.unpaired_file_action == "skip":
                print(f"Skipping {fileA.name}")

            else:
                raise ValueError(f"No matching file for {fileA.name}")

        # ---------------- NEAREST ----------------

        elif cf.pairing_mode == "nearest":

            closest = min(receiverB_dict.keys(), key=lambda t: abs((t - datetimeA).total_seconds()))

            difference = abs((closest - datetimeA).total_seconds()) / 60

            if difference <= cf.pairing_tolerance:
                paired_files.append((fileA, receiverB_dict[closest]))

            elif cf.unpaired_file_action == "skip":
                print(f"Skipping {fileA.name}")

            else:
                raise ValueError(f"No nearby match for {fileA.name}")

        else:
            raise ValueError("Invalid pairing_mode")

    return paired_files



def add_s4(df):

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






def initialize_log(log_file):
    """
    Create the processing log if it does not already exist.
    """

    log_file = Path(log_file)

    # Make sure the directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Create empty log if necessary
    if not log_file.exists():
        log_file.touch()


def log_processed_pair(fileA, fileB, log_file):
    """
    Append a successfully processed file pair to the log.

    Format:
        fileA,fileB
    """

    fileA = Path(fileA).name
    fileB = Path(fileB).name

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{fileA},{fileB}\n")


def create_processing_scratch(
    storage,
    scratch,
    input_pattern,
    temp_root=None,
    n_workers=1,
    verbose=True
):

    storage = Path(storage).resolve()
    scratch = Path(scratch).resolve()

    if not storage.exists():
        raise FileNotFoundError(
            f"Storage folder does not exist:\n{storage}"
        )


    # =========================================================
    # 1. CREATE UNIQUE FOLDER FOR EACH RUN
    # =========================================================

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    scratch_root = (
        scratch /
        f"run_{timestamp}"
    )

    scratch_input = (
        scratch_root /
        "input"
    )

    scratch_pq = (
        scratch_root /
        "pq"
    )


    scratch_input.mkdir(
        parents=True,
        exist_ok=False
    )

    scratch_pq.mkdir(
        parents=True,
        exist_ok=True
    )


    print("\n==============================================")
    print("CREATING SCRATCH WORKSPACE")
    print("==============================================")

    print(
        f"Scratch run folder:\n"
        f"{scratch_root}"
    )


    # =========================================================
    # 2. COPY STORAGE FILES
    # =========================================================

    storage_files = [
        file
        for file in storage.iterdir()
        if file.is_file()
    ]

    print(
        f"\nFound {len(storage_files)} "
        f"files in storage"
    )


    for source_file in storage_files:

        destination = (
            scratch_input /
            source_file.name
        )

        shutil.copy2(
            source_file,
            destination
        )

        print(
            f"Copied: {source_file}"
        )


    # =========================================================
    # 3. COPY EXISTING PQ FILES
    # =========================================================

    existing_pq_files = list(
        scratch_input.glob("*.pq")
    )

    print(
        f"\nFound {len(existing_pq_files)} "
        f"existing parquet files."
    )


    for pq_file in existing_pq_files:

        shutil.copy2(
            pq_file,
            scratch_pq / pq_file.name
        )

        print(
            f"Existing PQ copied: "
            f"{pq_file.name}"
        )


    # =========================================================
    # 4. CREATE SCRATCH INPUT PATTERN
    # =========================================================

    # IMPORTANT:
    #
    # run_conversion() uses:
    #
    # glob.glob(input_pattern)
    #
    # Therefore input_pattern must contain
    # the COMPLETE scratch path.
    #
    # Example:
    #
    # C:\...\scratch\run_123\input\*.bin.zip

    scratch_input_pattern = str(
        scratch_input /
        input_pattern
    )


    print(
        f"\nScratch conversion pattern:"
    )

    print(
        f"    {scratch_input_pattern}"
    )


    # =========================================================
    # 5. CHECK BIN.ZIP FILES
    # =========================================================

    conversion_files = list(
        scratch_input.glob(
            input_pattern
        )
    )


    print(
        f"\nFound {len(conversion_files)} "
        f"files matching:"
    )

    print(
        f"    {input_pattern}"
    )


    # =========================================================
    # 6. CONVERT BIN.ZIP → PQ
    # =========================================================

    if conversion_files:

        print(
            "\nStarting bin.zip conversion..."
        )


        run_conversion(
            mode="single",

            # THIS is the important change
            input_pattern=scratch_input_pattern,

            input_root=str(
                scratch_input
            ),

            output_root=str(
                scratch_pq
            ),

            infer_missing=False,

            n_workers=n_workers,

            temp_root=(
                str(temp_root)
                if temp_root is not None
                else None
            ),

            verbose=verbose
        )


    else:

        print(
            "\nNo bin.zip files found "
            "for conversion."
        )


    # =========================================================
    # 7. CHECK THAT PQ FILES EXIST
    # =========================================================

    pq_files = list(
        scratch_pq.glob("*.pq")
    )


    print(
        f"\nProcessing-ready parquet files: "
        f"{len(pq_files)}"
    )


    if len(pq_files) == 0:

        raise RuntimeError(
            "No parquet files were created "
            "or found in the scratch "
            "processing folder."
        )


    print(
        "\nScratch preparation complete."
    )

    print(
        f"Processing folder:\n"
        f"{scratch_pq}"
    )


    return scratch_pq



def cleanup_processing_scratch(
    processing_folder,
    scratch_folder
):
    """
    Delete the temporary run folder after
    successful processing.

    Original storage files are not touched.
    """

    # =========================================================
    # 1. CONVERT PATHS
    # =========================================================

    processing_folder = Path(processing_folder).resolve()

    scratch_folder = Path(scratch_folder).resolve()

    # =========================================================
    # 2. FIND THE RUN FOLDER
    # =========================================================

    # processing_folder looks like:
    #
    # scratch/run_20260727_153522/pq
    #
    # .parent gives:
    #
    # scratch/run_20260727_153522

    scratch_run_folder = (processing_folder.parent)

    # =========================================================
    # 3. SAFETY CHECK
    # =========================================================

    if scratch_run_folder.parent != scratch_folder:

        raise RuntimeError(
            "SAFETY ERROR: The folder being "
            "deleted is not directly inside "
            "the configured scratch folder.\n"
            f"Refusing to delete:\n"
            f"{scratch_run_folder}"
        )


    # =========================================================
    # 4. MAKE SURE RUN FOLDER EXISTS
    # =========================================================

    if not scratch_run_folder.exists():

        print("Scratch run folder already " "does not exist.")

        return

    print("\n==============================================")
    print("CLEANING SCRATCH WORKSPACE")
    print("==============================================")

    print(f"Deleting:\n" f"{scratch_run_folder}")

    shutil.rmtree(scratch_run_folder)

    print("Scratch workspace deleted.")