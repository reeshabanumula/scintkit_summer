"""Production receiver cross-correlation using NumPy and configured workers.

Configuration is read from ``configurations.txt`` through ``CONFIG.py``.
Source files are grouped by the date in each filename. Slurm array tasks can
split those dates into disjoint shards. Within each shard, one date at a time
is converted into a unique ``scratch_folder/run_<date>_*`` directory,
correlated, saved as one daily result, and removed before the next date begins.
Conversion and correlation both use the configured worker count, but only the
parent process combines rows and writes the daily result.

This is the winning "parallel + baseline" method:

- S4 event selection occurs after the full receiver merge.
- Cross-correlation uses ``numpy.correlate``.
- ``parallel_process_max_workers`` controls the worker count.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path
import tempfile
from time import perf_counter
import traceback

import numpy as np
import pandas as pd
from scipy import signal

import CONFIG as cf
import functions as f
from scintkit.pipelines.lvl0_convert_to_pq import run_conversion
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


def group_files_by_filename_day(
    files: list[Path],
) -> dict[date, list[Path]]:
    """Group files by the YYYYMMDD field in each ScintPi filename."""

    grouped: dict[date, list[Path]] = {}
    for file in sorted(files):
        filename_day = f.extract_file_datetime(file).date()
        grouped.setdefault(filename_day, []).append(file)
    return dict(sorted(grouped.items()))


def discover_source_files_by_day() -> dict[date, list[Path]]:
    """Discover configured source files without copying or opening them."""

    storage_folder = Path(cf.storage_folder).resolve()
    if not storage_folder.exists():
        raise FileNotFoundError(
            f"Storage folder does not exist:\n{storage_folder}"
        )

    source_files = sorted(
        file
        for file in storage_folder.glob(cf.input_pattern)
        if file.is_file()
    )
    if not source_files:
        raise FileNotFoundError(
            f"No files matching {cf.input_pattern!r} in {storage_folder}"
        )

    return group_files_by_filename_day(source_files)


def select_date_shard(
    files_by_day: dict[date, list[Path]],
    shard_index: int,
    shard_count: int,
) -> dict[date, list[Path]]:
    """Select one deterministic, round-robin shard of whole filename dates."""

    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            f"shard_index must be between 0 and {shard_count - 1}; "
            f"received {shard_index}"
        )

    ordered_days = list(sorted(files_by_day.items()))
    return dict(ordered_days[shard_index::shard_count])


def receiver_files_or_skip(
    files: list[Path],
) -> tuple[list[Path], list[Path]] | None:
    """Classify receiver files and honor the configured incomplete-file rule."""

    receiver_a, receiver_b = f.org_receivers(
        files,
        cf.r_latitude,
        cf.r_longitude,
        cf.lat_tol,
        cf.lon_tol,
    )
    if receiver_a and receiver_b:
        return receiver_a, receiver_b

    filename_days = sorted(
        {f.extract_file_datetime(file).strftime("%Y-%m-%d") for file in files}
    )
    locations = sorted(
        {f.extract_coord(file)[:2] for file in files}
    )
    location_preview = ", ".join(
        f"({latitude:.8f}, {longitude:.8f})"
        for latitude, longitude in locations[:6]
    )
    if len(locations) > 6:
        location_preview += f", ... {len(locations) - 6} more"

    message = (
        f"filename date(s) {', '.join(filename_days)} contain "
        f"Receiver A files={len(receiver_a)} and Receiver B candidates="
        f"{len(receiver_b)}; parsed coordinate magnitudes: {location_preview}"
    )
    if cf.unpaired_file_action == "skip":
        print(f"Skipping incomplete date batch: {message}")
        return None
    raise ValueError(f"Processing data must contain both receivers; {message}")


def find_file_pairs(
    processing_folder: Path,
    files: list[Path] | None = None,
) -> list[tuple[Path, Path]]:
    """Find the same receiver pairs for every production run."""

    if files is None:
        files = f.find_files(processing_folder)
    receiver_files = receiver_files_or_skip(files)
    if receiver_files is None:
        return []
    receiver_a, receiver_b = receiver_files

    pairs = f.pair_receiver_files(receiver_a, receiver_b, cf)
    if not pairs:
        if cf.unpaired_file_action == "skip":
            print("Skipping date batch: no receiver file pairs matched.")
            return []
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


def save_correlations(
    rows: list[dict[str, object]],
    filename_day: date | None = None,
) -> list[Path]:
    """Atomically save correlations, optionally as one filename-date batch."""

    correlations = pd.DataFrame(rows)
    if correlations.empty:
        raise RuntimeError("No valid scintillation correlations were produced.")

    if filename_day is None:
        daily_frames = correlations.groupby(correlations["minute"].dt.date)
    else:
        # A daily production batch is defined by the date embedded in its
        # source filenames. Keep all worker results together and write once.
        daily_frames = ((filename_day, correlations),)

    output_paths: list[Path] = []
    for output_day, day_frame in daily_frames:
        output_path = daily_correlation_path(output_day)
        temporary_path = output_path.with_suffix(".pq.tmp")
        try:
            day_frame.to_parquet(temporary_path, index=False)
            temporary_path.replace(output_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

        error_path = daily_error_path(output_day)
        if error_path.exists():
            error_path.unlink()
        output_paths.append(output_path)

    return output_paths


def daily_correlation_path(filename_day: date) -> Path:
    """Return the configured correlation filename for one filename date."""

    output_folder = Path(cf.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    base_name = Path(cf.cross_correlation_file).stem
    latitude_letter = "N" if cf.r_latitude >= 0 else "S"
    longitude_letter = "E" if cf.r_longitude >= 0 else "W"

    return output_folder / (
        f"{base_name}_{filename_day:%Y%m%d}_"
        f"{abs(cf.r_latitude):.3f}{latitude_letter}_"
        f"{abs(cf.r_longitude):.3f}{longitude_letter}.pq"
    )


def daily_error_path(filename_day: date) -> Path:
    """Return the daily diagnostic path matching the correlation filename."""

    correlation_path = daily_correlation_path(filename_day)
    return correlation_path.with_name(f"{correlation_path.stem}_err.txt")


def write_daily_error(
    filename_day: date,
    error: Exception,
    source_files: list[Path],
) -> Path:
    """Atomically write a traceback for one failed date batch."""

    error_path = daily_error_path(filename_day)
    temporary_path = error_path.with_suffix(".txt.tmp")
    source_text = "\n".join(f"  {file}" for file in source_files)
    traceback_text = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    text = (
        f"Filename date: {filename_day:%Y-%m-%d}\n"
        f"Exception: {type(error).__name__}: {error}\n"
        f"Source files ({len(source_files)}):\n{source_text}\n\n"
        f"Traceback:\n{traceback_text}"
    )

    try:
        temporary_path.write_text(text, encoding="utf-8")
        temporary_path.replace(error_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return error_path


def write_processed_pairs_log(
    processed_pairs: list[tuple[Path, Path]],
    shard_index: int = 0,
    shard_count: int = 1,
) -> Path:
    """Write a fresh, collision-free log for one date shard."""

    log_path = Path(cf.log_file)
    if shard_count > 1:
        log_path = log_path.with_name(
            f"{log_path.stem}_shard_{shard_index + 1:02d}_"
            f"of_{shard_count:02d}{log_path.suffix}"
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        f"{file_a.name},{file_b.name}\n"
        for file_a, file_b in processed_pairs
    )
    log_path.write_text(text, encoding="utf-8")
    return log_path


def create_cleanable_processing_scratch(
    filename_day: date,
    source_files: list[Path],
) -> Path:
    """Convert one filename-date batch into an isolated scratch directory."""

    if not source_files:
        raise ValueError(f"No source files supplied for {filename_day}")

    scratch_folder = Path(cf.scratch_folder).resolve()
    scratch_folder.mkdir(parents=True, exist_ok=True)
    run_folder = Path(
        tempfile.mkdtemp(
            prefix=f"run_{filename_day:%Y%m%d}_",
            dir=scratch_folder,
        )
    )
    processing_folder = run_folder / "pq"
    conversion_temp_folder = run_folder / "conversion_temp"
    processing_folder.mkdir()
    conversion_temp_folder.mkdir()

    print("\n==============================================")
    print(f"STAGING DATE {filename_day:%Y-%m-%d}")
    print("==============================================")
    print(f"Source files in batch: {len(source_files)}")
    print(f"Scratch run folder: {run_folder}")

    try:
        converted_files = run_conversion(
            mode="single",
            flist=[str(file) for file in source_files],
            input_root=str(Path(cf.storage_folder).resolve()),
            output_root=str(processing_folder),
            infer_missing=False,
            n_workers=cf.max_workers,
            temp_root=str(conversion_temp_folder),
            verbose=cf.verbose,
        )
        failed_files = [
            source_file
            for source_file, converted_file in zip(
                source_files,
                converted_files,
                strict=True,
            )
            if not converted_file or not Path(converted_file).exists()
        ]
        if failed_files:
            names = "\n".join(f"  {file.name}" for file in failed_files)
            raise RuntimeError(
                f"Conversion failed for {len(failed_files)} files:\n{names}"
            )

        print(f"Converted Parquet files: {len(converted_files)}")
        return processing_folder
    except Exception:
        f.cleanup_processing_scratch(
            processing_folder=processing_folder,
            scratch_folder=scratch_folder,
        )
        raise


def process_one_day(
    filename_day: date,
    processing_folder: Path,
    files: list[Path] | None = None,
) -> tuple[list[Path], list[tuple[Path, Path]], int]:
    """Correlate and save one date after all workers return to the parent."""

    paired_files = find_file_pairs(processing_folder, files=files)
    if not paired_files:
        print(f"No correlation output for skipped date {filename_day}")
        return [], [], 0

    rows, processed_pairs = run_pairs(paired_files)
    output_paths = save_correlations(rows, filename_day=filename_day)

    print(f"Processed {len(processed_pairs)} pairs for {filename_day}")
    print(f"Saved {len(rows)} correlation rows for {filename_day}")
    for output_path in output_paths:
        print(f"Saved correlations to {output_path}")

    return output_paths, processed_pairs, len(rows)


def run(
    existing_processing_folder: Path | None = None,
    *,
    shard_index: int = 0,
    shard_count: int = 1,
) -> tuple[list[Path], Path]:
    """Process one shard of dates sequentially with parallel work per date."""

    start = perf_counter()
    all_output_paths: list[Path] = []
    all_error_paths: list[Path] = []
    all_processed_pairs: list[tuple[Path, Path]] = []
    total_rows = 0

    # Existing scratch recovery owns and removes one shared run directory, so
    # it must remain a single-process operation.
    if existing_processing_folder is not None and shard_count != 1:
        raise ValueError(
            "Date sharding cannot be combined with --processing-folder."
        )

    if existing_processing_folder is not None:
        processing_folder = existing_processing_folder.resolve()
        try:
            parquet_files_by_day = group_files_by_filename_day(
                f.find_files(processing_folder)
            )
            if not parquet_files_by_day:
                raise FileNotFoundError(
                    f"No Parquet files found in {processing_folder}"
                )

            for filename_day, parquet_files in parquet_files_by_day.items():
                try:
                    outputs, pairs, row_count = process_one_day(
                        filename_day,
                        processing_folder,
                        files=parquet_files,
                    )
                    all_output_paths.extend(outputs)
                    all_processed_pairs.extend(pairs)
                    total_rows += row_count
                except Exception as error:
                    error_path = write_daily_error(
                        filename_day,
                        error,
                        parquet_files,
                    )
                    all_error_paths.append(error_path)
                    print(
                        f"Date {filename_day} failed; wrote diagnostics to "
                        f"{error_path}"
                    )
        finally:
            f.cleanup_processing_scratch(
                processing_folder=processing_folder,
                scratch_folder=cf.scratch_folder,
            )
    else:
        all_source_files_by_day = discover_source_files_by_day()
        source_files_by_day = select_date_shard(
            all_source_files_by_day,
            shard_index,
            shard_count,
        )
        print(
            f"Discovered {sum(map(len, all_source_files_by_day.values()))} "
            f"source files across {len(all_source_files_by_day)} dates"
        )
        print(
            f"Date shard {shard_index + 1}/{shard_count} was assigned "
            f"{len(source_files_by_day)} dates and "
            f"{sum(map(len, source_files_by_day.values()))} source files"
        )

        if not source_files_by_day:
            log_path = write_processed_pairs_log(
                [],
                shard_index=shard_index,
                shard_count=shard_count,
            )
            print("This shard has no filename dates to process.")
            print(f"Saved empty processed-pairs log to {log_path}")
            return [], log_path

        for batch_number, (filename_day, source_files) in enumerate(
            source_files_by_day.items(),
            start=1,
        ):
            print(
                f"\nStarting date batch {batch_number}/"
                f"{len(source_files_by_day)}: {filename_day}"
            )
            processing_folder: Path | None = None
            try:
                if receiver_files_or_skip(source_files) is None:
                    continue

                processing_folder = create_cleanable_processing_scratch(
                    filename_day,
                    source_files,
                )
                outputs, pairs, row_count = process_one_day(
                    filename_day,
                    processing_folder,
                )
                all_output_paths.extend(outputs)
                all_processed_pairs.extend(pairs)
                total_rows += row_count
            except Exception as error:
                error_path = write_daily_error(
                    filename_day,
                    error,
                    source_files,
                )
                all_error_paths.append(error_path)
                print(
                    f"Date {filename_day} failed; wrote diagnostics to "
                    f"{error_path}"
                )
            finally:
                if processing_folder is not None:
                    f.cleanup_processing_scratch(
                        processing_folder=processing_folder,
                        scratch_folder=cf.scratch_folder,
                    )

    if not all_output_paths and not all_error_paths:
        raise RuntimeError(
            "No daily correlation files were saved. All filename-date "
            "batches were incomplete, unpaired, or produced no valid rows."
        )

    log_path = write_processed_pairs_log(
        all_processed_pairs,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    print(f"\nProcessed {len(all_processed_pairs)} total file pairs")
    print(f"Saved {total_rows} total correlation rows")
    print(f"Saved {len(all_output_paths)} daily correlation files")
    print(f"Saved {len(all_error_paths)} daily error files")
    print(f"Saved processed-pairs log to {log_path}")
    print(f"Runtime: {perf_counter() - start:.3f} seconds")
    return all_output_paths, log_path


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
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based date-shard index. Slurm array tasks pass their task ID.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Total number of disjoint date shards.",
    )
    arguments = parser.parse_args()
    run(
        arguments.processing_folder,
        shard_index=arguments.shard_index,
        shard_count=arguments.shard_count,
    )


if __name__ == "__main__":
    main()
