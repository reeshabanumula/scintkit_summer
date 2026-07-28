"""Production receiver cross-correlation using NumPy and configured workers.

Configuration is read from ``configurations.txt`` through ``CONFIG.py``.
The original files in ``storage_folder`` are copied and converted inside a
timestamped ``scratch_folder/run_*`` directory. That exact run directory is
deleted in ``finally`` after processing succeeds or fails.

This is the winning "parallel + baseline" method:

- S4 event selection occurs after the full receiver merge.
- Cross-correlation uses ``numpy.correlate``.
- ``parallel_process_max_workers`` controls the worker count.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
import argparse

import numpy as np
import pandas as pd
from scipy import signal

import CONFIG as cf
import functions as f
from scintkit.preprocessing.format import temp_formating
from scintkit.services.phase_detrend import detect_sampling_rate


MERGE_KEYS = ["datetime", "svid", "cons"]
MIN_SAMPLES = 10


def normalized_numpy_correlation(
    first: pd.Series,
    second: pd.Series,
) -> tuple[float, int, np.ndarray, np.ndarray]:
    """Return normalized full correlation and its lag coordinates."""

    first_values = np.asarray(first, dtype=float)
    second_values = np.asarray(second, dtype=float)

    first_std = np.std(first_values)
    second_std = np.std(second_values)
    if (
        not np.all(np.isfinite(first_values))
        or not np.all(np.isfinite(second_values))
        or first_std == 0
        or second_std == 0
    ):
        raise ValueError("Correlation inputs must be finite and non-constant.")

    first_normalized = (
        first_values - np.mean(first_values)
    ) / first_std
    second_normalized = (
        second_values - np.mean(second_values)
    ) / second_std

    correlation = np.correlate(
        first_normalized,
        second_normalized,
        mode="full",
    )
    correlation = correlation / (
        np.linalg.norm(first_normalized)
        * np.linalg.norm(second_normalized)
    )
    lags = signal.correlation_lags(
        len(first_normalized),
        len(second_normalized),
        mode="full",
    )
    best_index = int(np.argmax(correlation))

    return (
        float(correlation[best_index]),
        int(lags[best_index]),
        correlation,
        lags,
    )


def process_file_pair(
    pair: tuple[Path, Path],
) -> tuple[list[dict[str, object]], Path, Path]:
    """Process one receiver pair with the unchanged NumPy baseline logic."""

    file_a, file_b = pair
    receiver_a = pd.read_parquet(file_a)
    receiver_b = pd.read_parquet(file_b)

    receiver_a = receiver_a[
        receiver_a["elev"] > cf.elevation_filter
    ].copy()
    receiver_b = receiver_b[
        receiver_b["elev"] > cf.elevation_filter
    ].copy()

    receiver_a = temp_formating(receiver_a)
    receiver_b = temp_formating(receiver_b)
    sampling_rate = detect_sampling_rate(receiver_a)
    seconds_per_sample = 1 / sampling_rate

    merged = receiver_a.merge(
        receiver_b,
        on=MERGE_KEYS,
        suffixes=("_A", "_B"),
    )
    merged["_event_minute"] = merged["datetime"].dt.floor("min")

    location_a = f.extract_coord(file_a)
    location_b = f.extract_coord(file_b)
    distance = f.calc_dist(location_a, location_b)

    rows: list[dict[str, object]] = []
    for (svid, cons, minute), unclean_group in merged.groupby(
        ["svid", "cons", "_event_minute"],
        sort=False,
    ):
        group = f.handle_nan(
            unclean_group.copy(),
            cf.nan_method,
            sig_columns=["snr1_A", "snr1_B"],
        )
        if len(group) < MIN_SAMPLES:
            continue

        linear_a = f.db2lin(group["snr1_A"])
        linear_b = f.db2lin(group["snr1_B"])
        s4_a = np.std(linear_a) / np.mean(linear_a)
        s4_b = np.std(linear_b) / np.mean(linear_b)
        if not (s4_a > cf.thresh or s4_b > cf.thresh):
            continue

        values_a = np.asarray(group["snr1_A"], dtype=float)
        values_b = np.asarray(group["snr1_B"], dtype=float)
        if (
            not np.all(np.isfinite(values_a))
            or not np.all(np.isfinite(values_b))
            or np.std(values_a) == 0
            or np.std(values_b) == 0
        ):
            continue

        max_cross, best_lag, cross, lags = normalized_numpy_correlation(
            group["snr1_A"],
            group["snr1_B"],
        )
        max_auto_a, _, auto_a, _ = normalized_numpy_correlation(
            group["snr1_A"],
            group["snr1_A"],
        )
        max_auto_b, _, auto_b, _ = normalized_numpy_correlation(
            group["snr1_B"],
            group["snr1_B"],
        )

        rows.append(
            {
                "source_a": file_a.name,
                "source_b": file_b.name,
                "minute": minute,
                "prn": group["prn_B"].iloc[0],
                "svid": svid,
                "cons": cons,
                "s4_1_A": s4_a,
                "s4_1_B": s4_b,
                "elev": group["elev_A"].mean(),
                "azim": group["azim_A"].mean(),
                "r_A": location_a,
                "r_B": location_b,
                "distance (km)": distance,
                "auto_cor_A_1": auto_a,
                "auto_cor_A_max_1": max_auto_a,
                "auto_cor_B_1": auto_b,
                "auto_cor_B_max_1": max_auto_b,
                "corr_norm_1": cross,
                "lag_norm_1": lags,
                "max_corr_1": max_cross,
                "best_lag_1": best_lag,
                "time_delay_1": best_lag * seconds_per_sample,
            }
        )

    return rows, file_a, file_b


def find_file_pairs(processing_folder: Path) -> list[tuple[Path, Path]]:
    """Find the same receiver pairs for every production run."""

    files = f.find_files(processing_folder)
    receiver_a, receiver_b = f.org_receivers(
        files,
        cf.r_latitude,
        cf.r_longitude,
        cf.lat_tol,
        cf.lon_tol,
    )
    if not receiver_a or not receiver_b:
        raise ValueError("Processing data must contain both receivers.")

    pairs = f.pair_receiver_files(receiver_a, receiver_b, cf)
    if not pairs:
        raise ValueError("No receiver file pairs matched the configuration.")
    return pairs


def run_pairs(
    paired_files: list[tuple[Path, Path]],
) -> tuple[list[dict[str, object]], list[tuple[Path, Path]]]:
    """Run serially or in parallel according to the configured worker count."""

    all_rows: list[dict[str, object]] = []
    processed_pairs: list[tuple[Path, Path]] = []

    if cf.max_workers == 1:
        batches = map(process_file_pair, paired_files)
        for rows, file_a, file_b in batches:
            all_rows.extend(rows)
            processed_pairs.append((file_a, file_b))
        return all_rows, processed_pairs

    with ProcessPoolExecutor(max_workers=cf.max_workers) as executor:
        batches = executor.map(
            process_file_pair,
            paired_files,
            chunksize=1,
        )
        for rows, file_a, file_b in batches:
            all_rows.extend(rows)
            processed_pairs.append((file_a, file_b))

    return all_rows, processed_pairs


def save_correlations(rows: list[dict[str, object]]) -> list[Path]:
    """Atomically save one Parquet correlation product per UTC day."""

    correlations = pd.DataFrame(rows)
    if correlations.empty:
        raise RuntimeError("No valid scintillation correlations were produced.")

    output_folder = Path(cf.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    base_name = Path(cf.cross_correlation_file).stem

    latitude_letter = "N" if cf.r_latitude >= 0 else "S"
    longitude_letter = "E" if cf.r_longitude >= 0 else "W"
    latitude = abs(cf.r_latitude)
    longitude = abs(cf.r_longitude)

    output_paths: list[Path] = []
    for day, day_frame in correlations.groupby(
        correlations["minute"].dt.date,
    ):
        day_string = pd.Timestamp(day).strftime("%Y%m%d")
        output_path = output_folder / (
            f"{base_name}_{day_string}_"
            f"{latitude:.3f}{latitude_letter}_"
            f"{longitude:.3f}{longitude_letter}.pq"
        )
        temporary_path = output_path.with_suffix(".pq.tmp")
        try:
            day_frame.to_parquet(temporary_path, index=False)
            temporary_path.replace(output_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        output_paths.append(output_path)

    return output_paths


def write_processed_pairs_log(
    processed_pairs: list[tuple[Path, Path]],
) -> Path:
    """Write a fresh log only after all correlation files are saved."""

    log_path = Path(cf.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        f"{file_a.name},{file_b.name}\n"
        for file_a, file_b in processed_pairs
    )
    log_path.write_text(text, encoding="utf-8")
    return log_path


def create_cleanable_processing_scratch() -> Path:
    """Create scratch data and remove a partial run if conversion fails."""

    scratch_root = Path(cf.scratch_folder).resolve()
    existing_runs = {
        path.resolve()
        for path in scratch_root.glob("run_*")
        if path.is_dir()
    }

    try:
        return f.create_processing_scratch(
            storage=cf.storage_folder,
            scratch=cf.scratch_folder,
            input_pattern=cf.input_pattern,
            temp_root=cf.temp_root,
            n_workers=cf.max_workers,
            verbose=cf.verbose,
        )
    except Exception:
        new_runs = [
            path.resolve()
            for path in scratch_root.glob("run_*")
            if path.is_dir() and path.resolve() not in existing_runs
        ]
        for run_folder in new_runs:
            f.cleanup_processing_scratch(
                processing_folder=run_folder / "pq",
                scratch_folder=scratch_root,
            )
        raise


def run(
    existing_processing_folder: Path | None = None,
) -> tuple[list[Path], Path]:
    """Create outputs and always remove the exact scratch run afterward."""

    start = perf_counter()
    processing_folder: Path | None = None

    try:
        if existing_processing_folder is None:
            processing_folder = create_cleanable_processing_scratch()
        else:
            processing_folder = existing_processing_folder.resolve()

        paired_files = find_file_pairs(processing_folder)
        rows, processed_pairs = run_pairs(paired_files)
        output_paths = save_correlations(rows)
        log_path = write_processed_pairs_log(processed_pairs)

        print(f"Processed {len(processed_pairs)} file pairs")
        print(f"Saved {len(rows)} correlation rows")
        for output_path in output_paths:
            print(f"Saved correlations to {output_path}")
        print(f"Saved processed-pairs log to {log_path}")
        print(f"Runtime: {perf_counter() - start:.3f} seconds")
        return output_paths, log_path
    finally:
        if processing_folder is not None:
            f.cleanup_processing_scratch(
                processing_folder=processing_folder,
                scratch_folder=cf.scratch_folder,
            )


def main() -> None:
    """Run from configuration, with an optional existing scratch recovery path."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processing-folder",
        type=Path,
        help=(
            "Use an existing scratch run's pq folder, then delete that run "
            "after processing. Normally omitted."
        ),
    )
    arguments = parser.parse_args()
    run(arguments.processing_folder)


if __name__ == "__main__":
    main()
