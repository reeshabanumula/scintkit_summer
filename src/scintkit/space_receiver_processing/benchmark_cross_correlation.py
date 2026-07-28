"""Factorial timing experiment for receiver cross-correlation processing.

This script varies exactly three implementation choices:

1. Calculate S4 and select events before the full receiver-data merge.
2. Use ``scipy.signal.correlate(method="auto")`` instead of ``numpy.correlate``.
3. Process file pairs with two workers instead of one.

All scientific settings and output fields are shared by all eight experiments.
The benchmark also checks that every experiment produces the same rounded
numerical result as the baseline.

The timed region includes reading, preprocessing, merging, S4 calculation,
correlation, multiprocessing startup, and result construction. File discovery,
file pairing, and writing the timing CSV/plot are deliberately outside it.

Edit the settings below, then run this file only when test data are available.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from itertools import product
from pathlib import Path
from time import perf_counter
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

import CONFIG as cf
import functions as f
from scintkit.preprocessing.format import temp_formating
from scintkit.services.phase_detrend import detect_sampling_rate


# ---------------------------------------------------------------------------
# Fixed scientific settings (not benchmark factors)
# ---------------------------------------------------------------------------

DATA_FOLDER = Path("/Users/isaac/Documents/crosscorr/data")
SCRATCH_FOLDER = Path("/Users/isaac/Documents/crosscorr/scratch")
CORRELATION_OUTPUT_FOLDER = Path("/Users/isaac/Documents/crosscorr/corrs")

# Keep only snr1 to reproduce the original baseline's event selection and
# correlation product. To expand the product later, add "snr2" here; that
# single fixed setting will then apply to all eight experiments.
SNR_COLUMNS = ("snr1",)
MIN_SAMPLES = 10
S4_THRESHOLD = cf.thresh
ELEVATION_FILTER = cf.elevation_filter
NAN_METHOD = cf.nan_method

# One repeat is convenient for a first data check. Use at least three repeats
# for a timing result that will be reported.
REPEATS = 1
RANDOM_SEED = 20260728

BENCHMARK_OUTPUT_FOLDER = CORRELATION_OUTPUT_FOLDER
TIMING_CSV = BENCHMARK_OUTPUT_FOLDER / "cross_correlation_timings.csv"
TIMING_PLOT = BENCHMARK_OUTPUT_FOLDER / "cross_correlation_timings.png"

MERGE_KEYS = ["datetime", "svid", "cons"]
EVENT_KEYS = ["svid", "cons", "_event_minute"]


@dataclass(frozen=True)
class ExperimentConfig:
    """The only three settings allowed to vary in the experiment."""

    early_s4_filter: bool
    correlation_engine: str
    workers: int

    @property
    def label(self) -> str:
        changes: list[str] = []
        if self.early_s4_filter:
            changes.append("S4 before merge")
        if self.correlation_engine == "scipy_auto":
            changes.append("SciPy auto")
        if self.workers == 2:
            changes.append("2 workers")
        return "Baseline" if not changes else " + ".join(changes)


EXPERIMENTS = tuple(
    ExperimentConfig(early_s4, engine, workers)
    for early_s4, engine, workers in product(
        (False, True),
        ("numpy", "scipy_auto"),
        (1, 2),
    )
)

BASELINE = ExperimentConfig(
    early_s4_filter=False,
    correlation_engine="numpy",
    workers=1,
)

# Guard against accidentally dropping or duplicating a combination.
assert len(EXPERIMENTS) == 8
assert len(set(EXPERIMENTS)) == 8
assert BASELINE in EXPERIMENTS


def _clean_group(group: pd.DataFrame) -> pd.DataFrame:
    """Apply the same NaN rule to every experiment and every SNR channel."""

    signal_columns = [
        f"{snr}_{receiver}"
        for snr in SNR_COLUMNS
        for receiver in ("A", "B")
    ]
    return f.handle_nan(group.copy(), NAN_METHOD, sig_columns=signal_columns)


def _s4_summary(group: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]] | None:
    """Clean one event group and calculate all receiver/channel S4 values."""

    group = _clean_group(group)
    if len(group) < MIN_SAMPLES:
        return None

    summary: dict[str, float] = {}
    for snr in SNR_COLUMNS:
        for receiver in ("A", "B"):
            column = f"{snr}_{receiver}"
            linear_snr = f.db2lin(group[column])
            summary[f"s4_{snr}_{receiver}"] = (
                np.std(linear_snr) / np.mean(linear_snr)
            )

    return group, summary


def _is_event_of_interest(s4: dict[str, float]) -> bool:
    """Use one fixed event rule in all experiments.

    A minute is selected when any configured SNR channel at either receiver
    exceeds the S4 threshold. This avoids the old inconsistency where one
    method used OR while another required every channel and receiver.
    """

    return any(value > S4_THRESHOLD for value in s4.values())


def _has_valid_correlation_signals(group: pd.DataFrame) -> bool:
    """Reject groups that cannot be normalized into finite correlations."""

    for snr in SNR_COLUMNS:
        for receiver in ("A", "B"):
            values = np.asarray(group[f"{snr}_{receiver}"], dtype=float)
            if not np.all(np.isfinite(values)) or np.std(values) == 0:
                return False
    return True


def _normalized_correlation(
    first: pd.Series,
    second: pd.Series,
    engine: str,
) -> tuple[float, int, np.ndarray, np.ndarray]:
    """Run the same normalization with either correlation implementation."""

    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)

    first_normalized = (
        first_array - np.mean(first_array)
    ) / np.std(first_array)
    second_normalized = (
        second_array - np.mean(second_array)
    ) / np.std(second_array)

    if engine == "numpy":
        correlation = np.correlate(
            first_normalized,
            second_normalized,
            mode="full",
        )
    elif engine == "scipy_auto":
        correlation = signal.correlate(
            first_normalized,
            second_normalized,
            mode="full",
            method="auto",
        )
    else:
        raise ValueError(f"Unknown correlation engine: {engine}")

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


def _find_early_events(
    receiver_a: pd.DataFrame,
    receiver_b: pd.DataFrame,
) -> dict[tuple[object, ...], dict[str, float]]:
    """Find events from a narrow merge before merging the full dataframes.

    The narrow merge contains only timestamp, satellite identifiers, and the
    SNR columns. It preserves the baseline's timestamp intersection and NaN
    behavior while avoiding an early merge of unused wide columns.
    """

    narrow_columns = MERGE_KEYS + list(SNR_COLUMNS)
    narrow = receiver_a[narrow_columns].merge(
        receiver_b[narrow_columns],
        on=MERGE_KEYS,
        suffixes=("_A", "_B"),
    )
    narrow["_event_minute"] = narrow["datetime"].dt.floor("min")

    selected: dict[tuple[object, ...], dict[str, float]] = {}
    for event_key, group in narrow.groupby(EVENT_KEYS, sort=False):
        calculated = _s4_summary(group)
        if calculated is None:
            continue
        _, s4 = calculated
        if _is_event_of_interest(s4):
            selected[event_key] = s4

    return selected


def _filter_to_event_keys(
    dataframe: pd.DataFrame,
    selected: dict[tuple[object, ...], dict[str, float]],
) -> pd.DataFrame:
    """Restrict one receiver dataframe to the selected satellite-minutes."""

    if not selected:
        return dataframe.iloc[0:0].copy()

    selected_index = pd.MultiIndex.from_tuples(
        selected,
        names=EVENT_KEYS,
    )
    dataframe_index = pd.MultiIndex.from_frame(dataframe[EVENT_KEYS])
    return dataframe.loc[dataframe_index.isin(selected_index)].copy()


def _process_file_pair(
    pair: tuple[Path, Path],
    experiment: ExperimentConfig,
) -> list[dict[str, object]]:
    """Process one file pair using only the choices in ``experiment``."""

    file_a, file_b = pair
    receiver_a = pd.read_parquet(file_a)
    receiver_b = pd.read_parquet(file_b)

    receiver_a = receiver_a[
        receiver_a["elev"] > ELEVATION_FILTER
    ].copy()
    receiver_b = receiver_b[
        receiver_b["elev"] > ELEVATION_FILTER
    ].copy()

    receiver_a = temp_formating(receiver_a)
    receiver_b = temp_formating(receiver_b)

    sampling_rate = detect_sampling_rate(receiver_a)
    seconds_per_sample = 1 / sampling_rate

    receiver_a["_event_minute"] = receiver_a["datetime"].dt.floor("min")
    receiver_b["_event_minute"] = receiver_b["datetime"].dt.floor("min")

    early_s4: dict[tuple[object, ...], dict[str, float]] = {}
    if experiment.early_s4_filter:
        early_s4 = _find_early_events(receiver_a, receiver_b)
        receiver_a = _filter_to_event_keys(receiver_a, early_s4)
        receiver_b = _filter_to_event_keys(receiver_b, early_s4)

    # The full merge is deliberately identical in all experiments. The early
    # S4 variants simply reach it with fewer rows.
    merged = receiver_a.merge(
        receiver_b,
        on=MERGE_KEYS + ["_event_minute"],
        suffixes=("_A", "_B"),
    )

    receiver_a_location = f.extract_coord(file_a)
    receiver_b_location = f.extract_coord(file_b)
    receiver_distance = f.calc_dist(
        receiver_a_location,
        receiver_b_location,
    )

    results: list[dict[str, object]] = []
    for event_key, unclean_group in merged.groupby(EVENT_KEYS, sort=False):
        if experiment.early_s4_filter:
            group = _clean_group(unclean_group)
            if len(group) < MIN_SAMPLES:
                continue
            s4 = early_s4[event_key]
        else:
            calculated = _s4_summary(unclean_group)
            if calculated is None:
                continue
            group, s4 = calculated
            if not _is_event_of_interest(s4):
                continue

        if not _has_valid_correlation_signals(group):
            continue

        svid, cons, minute = event_key
        row: dict[str, object] = {
            "source_a": Path(file_a).name,
            "source_b": Path(file_b).name,
            "minute": minute,
            "svid": svid,
            "cons": cons,
            "elev": group["elev_A"].mean(),
            "azim": group["azim_A"].mean(),
            "r_A": receiver_a_location,
            "r_B": receiver_b_location,
            "distance (km)": receiver_distance,
            **s4,
        }

        for snr in SNR_COLUMNS:
            channel = snr.removeprefix("snr")

            max_cross, best_lag, cross, lags = _normalized_correlation(
                group[f"{snr}_A"],
                group[f"{snr}_B"],
                experiment.correlation_engine,
            )
            max_auto_a, _, auto_a, _ = _normalized_correlation(
                group[f"{snr}_A"],
                group[f"{snr}_A"],
                experiment.correlation_engine,
            )
            max_auto_b, _, auto_b, _ = _normalized_correlation(
                group[f"{snr}_B"],
                group[f"{snr}_B"],
                experiment.correlation_engine,
            )

            row.update(
                {
                    f"corr_norm_{channel}": cross,
                    f"lag_norm_{channel}": lags,
                    f"max_corr_{channel}": max_cross,
                    f"best_lag_{channel}": best_lag,
                    f"time_delay_{channel}": best_lag * seconds_per_sample,
                    f"auto_cor_A_{channel}": auto_a,
                    f"auto_cor_A_max_{channel}": max_auto_a,
                    f"auto_cor_B_{channel}": auto_b,
                    f"auto_cor_B_max_{channel}": max_auto_b,
                }
            )

        results.append(row)

    return results


def _run_experiment(
    paired_files: list[tuple[Path, Path]],
    experiment: ExperimentConfig,
) -> list[dict[str, object]]:
    """Run one configuration over the exact same ordered file-pair list."""

    process = partial(_process_file_pair, experiment=experiment)

    if experiment.workers == 1:
        batches = map(process, paired_files)
        return [row for batch in batches for row in batch]

    with ProcessPoolExecutor(max_workers=experiment.workers) as executor:
        batches = executor.map(process, paired_files)
        return [row for batch in batches for row in batch]


def _result_signature(rows: list[dict[str, object]]) -> str:
    """Create a compact, tolerance-aware signature for consistency checks."""

    digest = sha256()
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            row["source_a"],
            row["source_b"],
            row["minute"],
            row["svid"],
            row["cons"],
        ),
    )

    for row in ordered_rows:
        identity = (
            row["source_a"],
            row["source_b"],
            pd.Timestamp(row["minute"]).isoformat(),
            row["svid"],
            row["cons"],
        )
        digest.update(repr(identity).encode())

        for snr in SNR_COLUMNS:
            channel = snr.removeprefix("snr")
            for receiver in ("A", "B"):
                s4 = round(float(row[f"s4_{snr}_{receiver}"]), 9)
                digest.update(repr(s4).encode())

            digest.update(repr(int(row[f"best_lag_{channel}"])).encode())
            for name in (
                f"corr_norm_{channel}",
                f"auto_cor_A_{channel}",
                f"auto_cor_B_{channel}",
            ):
                rounded = np.round(
                    np.asarray(row[name], dtype=float),
                    decimals=9,
                )
                digest.update(rounded.tobytes())

    return digest.hexdigest()


def _plot_timings(timings: pd.DataFrame) -> None:
    """Save a bar chart in the style of the original timing notebook."""

    label_order = [experiment.label for experiment in EXPERIMENTS]
    summary = (
        timings.groupby("method", sort=False)["seconds"]
        .agg(["mean", "std"])
        .reindex(label_order)
    )

    minutes = summary["mean"] / 60
    errors = summary["std"].fillna(0) / 60

    figure, axis = plt.subplots(figsize=(12, 7))
    bars = axis.bar(
        summary.index,
        minutes,
        yerr=errors if REPEATS > 1 else None,
        capsize=4,
    )
    axis.set_ylabel("Runtime (minutes)")
    axis.set_xlabel("Method")
    axis.set_title("Runtime Comparison of Cross-Correlation Methods")
    axis.tick_params(axis="x", rotation=45)

    for bar, runtime in zip(bars, minutes):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{runtime:.1f}m",
            ha="center",
            va="bottom",
        )

    figure.tight_layout()
    figure.savefig(TIMING_PLOT, dpi=200)
    plt.close(figure)


def _prepare_processing_folder() -> Path:
    """Return Parquet input, converting raw files into scratch when needed."""

    parquet_files = list(DATA_FOLDER.glob("*.pq"))
    if parquet_files:
        return DATA_FOLDER

    raw_files = list(DATA_FOLDER.glob("*.bin.zip"))
    if not raw_files:
        raise ValueError(
            f"No .pq or .bin.zip files were found in {DATA_FOLDER}"
        )

    # Reuse the newest complete conversion so interrupted benchmark attempts do
    # not copy and convert the same raw files again.
    scratch_candidates = sorted(
        SCRATCH_FOLDER.glob("run_*/pq"),
        reverse=True,
    )
    for candidate in scratch_candidates:
        if len(list(candidate.glob("*.pq"))) == len(raw_files):
            return candidate

    return f.create_processing_scratch(
        storage=DATA_FOLDER,
        scratch=SCRATCH_FOLDER,
        input_pattern="*.bin.zip",
        temp_root=SCRATCH_FOLDER,
        n_workers=2,
        verbose=cf.verbose,
    )


def _find_file_pairs(processing_folder: Path) -> list[tuple[Path, Path]]:
    """Discover and pair files once so every experiment gets the same input."""

    files = f.find_files(processing_folder)
    receiver_a, receiver_b = f.org_receivers(
        files,
        cf.r_latitude,
        cf.r_longitude,
        cf.lat_tol,
        cf.lon_tol,
    )
    if not receiver_a or not receiver_b:
        raise ValueError(
            "The configured input directory must contain files from both "
            "receivers."
        )

    pairs = f.pair_receiver_files(receiver_a, receiver_b, cf)
    if not pairs:
        raise ValueError("No receiver file pairs matched the pairing settings.")
    return pairs


def main() -> None:
    """Run all eight configurations, validate them, and plot their timings."""

    processing_folder = _prepare_processing_folder()
    print(f"Using processing data in {processing_folder}")
    paired_files = _find_file_pairs(processing_folder)
    random_generator = random.Random(RANDOM_SEED)
    records: list[dict[str, object]] = []
    signatures: dict[tuple[int, ExperimentConfig], str] = {}

    for repeat in range(1, REPEATS + 1):
        run_order = list(EXPERIMENTS)
        random_generator.shuffle(run_order)

        for order, experiment in enumerate(run_order, start=1):
            print(
                f"Repeat {repeat}/{REPEATS}, "
                f"experiment {order}/8: {experiment.label}"
            )
            start = perf_counter()
            rows = _run_experiment(paired_files, experiment)
            elapsed = perf_counter() - start
            signature = _result_signature(rows)
            signatures[(repeat, experiment)] = signature

            records.append(
                {
                    "repeat": repeat,
                    "run_order": order,
                    "method": experiment.label,
                    "early_s4_filter": experiment.early_s4_filter,
                    "correlation_engine": experiment.correlation_engine,
                    "workers": experiment.workers,
                    "seconds": elapsed,
                    "event_count": len(rows),
                    "result_signature": signature,
                }
            )

    for repeat in range(1, REPEATS + 1):
        baseline_signature = signatures[(repeat, BASELINE)]
        inconsistent = [
            experiment.label
            for experiment in EXPERIMENTS
            if signatures[(repeat, experiment)] != baseline_signature
        ]
        if inconsistent:
            raise RuntimeError(
                "These experiments did not match the baseline result: "
                + ", ".join(inconsistent)
            )

    BENCHMARK_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    timings = pd.DataFrame(records)
    timings["consistent_with_baseline"] = True
    timings.to_csv(TIMING_CSV, index=False)
    _plot_timings(timings)

    print(f"Saved timings to {TIMING_CSV}")
    print(f"Saved timing plot to {TIMING_PLOT}")


if __name__ == "__main__":
    main()
