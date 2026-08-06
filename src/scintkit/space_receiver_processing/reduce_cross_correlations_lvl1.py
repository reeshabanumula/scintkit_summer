"""Reduce raw SC003 correlation Parquets to compact scalar level-1 files.

Each input file is reduced independently. Correlation/list columns are read one
at a time so a worker does not need to materialize every large array column at
once. Receiver coordinate lists are expanded to latitude, longitude, and
height columns. After every input has a successful level-1 output, the level-1
files from the current run are filtered to mutual S4 events on either channel
and concatenated into one Parquet file.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import gc
from pathlib import Path
import re
import traceback

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import signal


DEFAULT_INPUT_DIR = Path("/titan/frodrigues/corrs_sc003")
DEFAULT_PATTERN = "sc003_corrs_*.pq"
DEFAULT_COMBINED_NAME = "sc003_corrs_all_lvl1.pq"
DEFAULT_SECONDS_PER_SAMPLE = 0.05
S4_THRESHOLD = 0.1
S4_FILTER_COLUMNS = ("s4_1_A", "s4_1_B", "s4_2_A", "s4_2_B")
COORDINATE_COLUMNS = ("r_A", "r_B")
CHANNEL_PATTERN = re.compile(r"^(?:auto_cor_[AB]|corr_norm)_(\d+)$")


@dataclass(frozen=True)
class ReductionResult:
    """Serializable result returned by one worker."""

    source: Path
    output: Path
    row_count: int
    status: str
    error: str | None = None


def is_list_type(data_type: pa.DataType) -> bool:
    """Return whether an Arrow type stores a list in each row."""

    return (
        pa.types.is_list(data_type)
        or pa.types.is_large_list(data_type)
        or pa.types.is_fixed_size_list(data_type)
    )


def discover_input_files(input_dir: Path, pattern: str) -> list[Path]:
    """Return source Parquets, excluding already reduced products."""

    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = sorted(
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and not path.name.endswith("_lvl1.pq")
    )
    if not files:
        raise FileNotFoundError(
            f"No source files matching {pattern!r} in {input_dir}"
        )
    return files


def output_path_for(source: Path, output_dir: Path) -> Path:
    """Return the individual level-1 path for one source file."""

    return output_dir / f"{source.stem}_lvl1.pq"


def error_path_for(output: Path) -> Path:
    """Return the per-file diagnostic path for an output."""

    return output.with_name(f"{output.stem}_err.txt")


def channel_suffixes(schema: pa.Schema) -> list[str]:
    """Discover every numbered SNR channel represented by array columns."""

    channels = {
        match.group(1)
        for name in schema.names
        if (match := CHANNEL_PATTERN.match(name)) is not None
    }
    return sorted(channels, key=int)


def coordinate_triplet(value: object) -> tuple[float, float, float]:
    """Convert one receiver coordinate list to three nullable scalars."""

    if value is None:
        return (np.nan, np.nan, np.nan)
    values = np.asarray(value, dtype=float).reshape(-1)
    if len(values) != 3:
        raise ValueError(
            f"Receiver coordinate must contain exactly 3 values; got {len(values)}"
        )
    return (float(values[0]), float(values[1]), float(values[2]))


def read_scalar_frame(parquet_file: pq.ParquetFile) -> pd.DataFrame:
    """Read scalar columns plus the two small receiver-coordinate lists."""

    selected_columns = [
        field.name
        for field in parquet_file.schema_arrow
        if not is_list_type(field.type) or field.name in COORDINATE_COLUMNS
    ]
    frame = parquet_file.read(
        columns=selected_columns,
        use_threads=False,
    ).to_pandas()

    for coordinate in COORDINATE_COLUMNS:
        if coordinate not in frame:
            continue
        expanded = pd.DataFrame(
            [coordinate_triplet(value) for value in frame[coordinate]],
            columns=(
                f"{coordinate}_lat",
                f"{coordinate}_lon",
                f"{coordinate}_height",
            ),
            index=frame.index,
        )
        frame = pd.concat(
            [frame.drop(columns=coordinate), expanded],
            axis=1,
        )

    return frame


def filter_mutual_s4(
    frame: pd.DataFrame,
    threshold: float = S4_THRESHOLD,
) -> pd.DataFrame:
    """Keep rows where A and B exceed the threshold on channel 1 or 2."""

    missing = [column for column in S4_FILTER_COLUMNS if column not in frame]
    if missing:
        raise ValueError(
            "Cannot apply the level-1 S4 filter; missing columns: "
            f"{missing}"
        )

    values = {
        column: pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        for column in S4_FILTER_COLUMNS
    }
    keep = (
        (values["s4_1_A"] > threshold)
        & (values["s4_1_B"] > threshold)
    ) | (
        (values["s4_2_A"] > threshold)
        & (values["s4_2_B"] > threshold)
    )
    return frame.loc[keep].reset_index(drop=True)


def _iter_list_values(
    parquet_file: pq.ParquetFile,
    column: str,
    batch_size: int,
):
    """Yield Python list values from one Parquet column in row order."""

    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=[column],
        use_threads=False,
    ):
        yield from batch.column(0).to_pylist()


def cross_metrics_from_array(
    parquet_file: pq.ParquetFile,
    column: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate maximum cross-correlation and best lag in samples."""

    maxima: list[float] = []
    best_lags: list[float] = []
    for value in _iter_list_values(parquet_file, column, batch_size):
        if value is None:
            maxima.append(np.nan)
            best_lags.append(np.nan)
            continue
        correlation = np.asarray(value, dtype=float)
        if len(correlation) % 2 != 1:
            raise ValueError(
                f"{column} must have odd-length full correlations; "
                f"got {len(correlation)}"
            )
        finite = np.isfinite(correlation)
        if not finite.any():
            maxima.append(np.nan)
            best_lags.append(np.nan)
            continue
        best_index = int(np.nanargmax(correlation))
        maxima.append(float(correlation[best_index]))
        best_lags.append(float(best_index - len(correlation) // 2))
    return np.asarray(maxima), np.asarray(best_lags)


def seconds_per_sample_values(
    frame: pd.DataFrame,
    channel: str,
    default_seconds_per_sample: float,
) -> np.ndarray:
    """Infer per-row cadence from stored lag/time scalars, with a fallback."""

    if default_seconds_per_sample <= 0:
        raise ValueError("default_seconds_per_sample must be greater than zero")

    result = np.full(len(frame), default_seconds_per_sample, dtype=float)
    lag_column = f"best_lag_{channel}"
    time_column = f"time_delay_{channel}"
    if lag_column not in frame or time_column not in frame:
        return result

    lags = pd.to_numeric(frame[lag_column], errors="coerce").to_numpy(float)
    delays = pd.to_numeric(frame[time_column], errors="coerce").to_numpy(float)
    ratios = np.full(len(frame), np.nan)
    divisible = np.isfinite(lags) & np.isfinite(delays) & (lags != 0)
    np.divide(delays, lags, out=ratios, where=divisible)
    valid = divisible & (ratios > 0)
    if valid.any():
        fallback = float(np.median(ratios[valid]))
        result[:] = fallback
        result[valid] = ratios[valid]
    return result


def first_positive_crossing_time(
    autocorrelation: np.ndarray,
    threshold: float,
    seconds_per_sample: float,
) -> float:
    """Linearly interpolate the first positive-lag downward crossing."""

    if (
        len(autocorrelation) % 2 != 1
        or not np.isfinite(threshold)
        or not np.isfinite(seconds_per_sample)
        or seconds_per_sample <= 0
    ):
        return float("nan")

    positive = autocorrelation[len(autocorrelation) // 2 :]
    if len(positive) < 2 or not np.isfinite(positive[0]):
        return float("nan")
    if positive[0] <= threshold:
        return 0.0

    for index in range(1, len(positive)):
        previous = positive[index - 1]
        current = positive[index]
        if not (np.isfinite(previous) and np.isfinite(current)):
            continue
        if previous > threshold and current <= threshold:
            if current == previous:
                lag_samples = float(index)
            else:
                fraction = (threshold - previous) / (current - previous)
                lag_samples = (index - 1) + float(fraction)
            return lag_samples * seconds_per_sample
    return float("nan")


def normalized_max_cross_correlation(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Return the maximum normalized cross-correlation of two functions."""

    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if (
        len(first) == 0
        or len(second) == 0
        or not np.all(np.isfinite(first))
        or not np.all(np.isfinite(second))
    ):
        return float("nan")

    first_centered = first - np.mean(first)
    second_centered = second - np.mean(second)
    denominator = (
        np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    )
    if not np.isfinite(denominator) or denominator == 0:
        return float("nan")

    correlation = signal.correlate(
        first_centered,
        second_centered,
        mode="full",
        method="fft",
    )
    return float(np.max(correlation / denominator))


def paired_autocorrelation_metrics(
    parquet_file: pq.ParquetFile,
    column_a: str,
    column_b: str,
    max_cross: np.ndarray,
    sample_seconds: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, ...]:
    """Calculate receiver metrics and correlation between both ACFs."""

    lag_zero_a: list[float] = []
    decorrelation_a: list[float] = []
    at_max_cross_a: list[float] = []
    lag_zero_b: list[float] = []
    decorrelation_b: list[float] = []
    at_max_cross_b: list[float] = []
    max_auto_cross: list[float] = []

    row_index = 0
    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=[column_a, column_b],
        use_threads=False,
    ):
        values_a = batch.column(0).to_pylist()
        values_b = batch.column(1).to_pylist()
        for value_a, value_b in zip(values_a, values_b, strict=True):
            if row_index >= len(max_cross):
                raise ValueError(
                    f"{column_a}/{column_b} contain more rows than scalars"
                )
            arrays: list[np.ndarray | None] = []
            for column, value in (
                (column_a, value_a),
                (column_b, value_b),
            ):
                if value is None:
                    arrays.append(None)
                    continue
                correlation = np.asarray(value, dtype=float)
                if len(correlation) % 2 != 1:
                    raise ValueError(
                        f"{column} must have odd-length full correlations; "
                        f"got {len(correlation)}"
                    )
                arrays.append(correlation)

            for correlation, lag_zero, decorrelation, at_max_cross in (
                (
                    arrays[0],
                    lag_zero_a,
                    decorrelation_a,
                    at_max_cross_a,
                ),
                (
                    arrays[1],
                    lag_zero_b,
                    decorrelation_b,
                    at_max_cross_b,
                ),
            ):
                if correlation is None:
                    lag_zero.append(np.nan)
                    decorrelation.append(np.nan)
                    at_max_cross.append(np.nan)
                    continue
                lag_zero.append(
                    float(correlation[len(correlation) // 2])
                )
                decorrelation.append(
                    first_positive_crossing_time(
                        correlation,
                        1 / np.e,
                        sample_seconds[row_index],
                    )
                )
                at_max_cross.append(
                    first_positive_crossing_time(
                        correlation,
                        max_cross[row_index],
                        sample_seconds[row_index],
                    )
                )

            if arrays[0] is None or arrays[1] is None:
                max_auto_cross.append(np.nan)
            else:
                max_auto_cross.append(
                    normalized_max_cross_correlation(arrays[0], arrays[1])
                )
            row_index += 1

    if row_index != len(max_cross):
        raise ValueError(
            f"{column_a}/{column_b} contain {row_index} rows; "
            f"expected {len(max_cross)}"
        )
    return tuple(
        np.asarray(values)
        for values in (
            lag_zero_a,
            decorrelation_a,
            at_max_cross_a,
            lag_zero_b,
            decorrelation_b,
            at_max_cross_b,
            max_auto_cross,
        )
    )


def autocorrelation_metrics(
    parquet_file: pq.ParquetFile,
    column: str,
    max_cross: np.ndarray,
    sample_seconds: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lag-zero value and the two requested crossing times."""

    lag_zero: list[float] = []
    decorrelation_times: list[float] = []
    max_cross_times: list[float] = []

    row_index = 0
    for value in _iter_list_values(parquet_file, column, batch_size):
        if row_index >= len(max_cross):
            raise ValueError(f"{column} contains more rows than scalar columns")
        if value is None:
            lag_zero.append(np.nan)
            decorrelation_times.append(np.nan)
            max_cross_times.append(np.nan)
            row_index += 1
            continue

        correlation = np.asarray(value, dtype=float)
        if len(correlation) % 2 != 1:
            raise ValueError(
                f"{column} must have odd-length full correlations; "
                f"got {len(correlation)}"
            )
        zero_value = correlation[len(correlation) // 2]
        lag_zero.append(float(zero_value))
        decorrelation_times.append(
            first_positive_crossing_time(
                correlation,
                1 / np.e,
                sample_seconds[row_index],
            )
        )
        max_cross_times.append(
            first_positive_crossing_time(
                correlation,
                max_cross[row_index],
                sample_seconds[row_index],
            )
        )
        row_index += 1

    if row_index != len(max_cross):
        raise ValueError(
            f"{column} contains {row_index} rows; expected {len(max_cross)}"
        )
    return (
        np.asarray(lag_zero),
        np.asarray(decorrelation_times),
        np.asarray(max_cross_times),
    )


def add_channel_metrics(
    frame: pd.DataFrame,
    parquet_file: pq.ParquetFile,
    channel: str,
    default_seconds_per_sample: float,
    batch_size: int,
) -> None:
    """Add or preserve all required scalar metrics for one SNR channel."""

    names = set(parquet_file.schema_arrow.names)
    cross_column = f"corr_norm_{channel}"
    max_column = f"max_corr_{channel}"
    lag_column = f"best_lag_{channel}"
    time_column = f"time_delay_{channel}"

    if max_column not in frame or lag_column not in frame:
        if cross_column not in names:
            raise ValueError(
                f"Channel {channel} lacks both stored cross metrics and "
                f"{cross_column}"
            )
        calculated_max, calculated_lag = cross_metrics_from_array(
            parquet_file,
            cross_column,
            batch_size,
        )
        if max_column not in frame:
            frame[max_column] = calculated_max
        if lag_column not in frame:
            frame[lag_column] = calculated_lag
        del calculated_max, calculated_lag
        gc.collect()

    sample_seconds = seconds_per_sample_values(
        frame,
        channel,
        default_seconds_per_sample,
    )
    if time_column not in frame:
        best_lags = pd.to_numeric(
            frame[lag_column],
            errors="coerce",
        ).to_numpy(float)
        frame[time_column] = best_lags * sample_seconds

    max_cross = pd.to_numeric(
        frame[max_column],
        errors="coerce",
    ).to_numpy(float)

    auto_a = f"auto_cor_A_{channel}"
    auto_b = f"auto_cor_B_{channel}"
    if auto_a in names and auto_b in names:
        (
            lag_zero_a,
            decorrelation_a,
            at_max_cross_a,
            lag_zero_b,
            decorrelation_b,
            at_max_cross_b,
            max_auto_cross,
        ) = paired_autocorrelation_metrics(
            parquet_file,
            auto_a,
            auto_b,
            max_cross,
            sample_seconds,
            batch_size,
        )
        frame[f"auto_cor_A_max_{channel}"] = lag_zero_a
        frame[f"decorrelation_time_A_{channel}"] = decorrelation_a
        frame[
            f"auto_cor_A_time_at_max_corr_{channel}"
        ] = at_max_cross_a
        frame[f"auto_cor_B_max_{channel}"] = lag_zero_b
        frame[f"decorrelation_time_B_{channel}"] = decorrelation_b
        frame[
            f"auto_cor_B_time_at_max_corr_{channel}"
        ] = at_max_cross_b
        frame[f"max_auto_cross_corr_{channel}"] = max_auto_cross
        del (
            lag_zero_a,
            decorrelation_a,
            at_max_cross_a,
            lag_zero_b,
            decorrelation_b,
            at_max_cross_b,
            max_auto_cross,
        )
        gc.collect()
        return

    for receiver in ("A", "B"):
        auto_column = f"auto_cor_{receiver}_{channel}"
        if auto_column not in names:
            continue
        lag_zero, decorrelation, at_max_cross = autocorrelation_metrics(
            parquet_file,
            auto_column,
            max_cross,
            sample_seconds,
            batch_size,
        )
        frame[f"auto_cor_{receiver}_max_{channel}"] = lag_zero
        frame[f"decorrelation_time_{receiver}_{channel}"] = decorrelation
        frame[
            f"auto_cor_{receiver}_time_at_max_corr_{channel}"
        ] = at_max_cross
        del lag_zero, decorrelation, at_max_cross
        gc.collect()


def required_output_columns(schema: pa.Schema) -> set[str]:
    """Return derived columns required for an output to be restart-safe."""

    names = set(schema.names)
    required: set[str] = set()
    for coordinate in COORDINATE_COLUMNS:
        if coordinate in names:
            required.update(
                {
                    f"{coordinate}_lat",
                    f"{coordinate}_lon",
                    f"{coordinate}_height",
                }
            )
    for channel in channel_suffixes(schema):
        for receiver in ("A", "B"):
            if f"auto_cor_{receiver}_{channel}" in names:
                required.update(
                    {
                        f"auto_cor_{receiver}_max_{channel}",
                        f"decorrelation_time_{receiver}_{channel}",
                        f"auto_cor_{receiver}_time_at_max_corr_{channel}",
                    }
                )
        if (
            f"auto_cor_A_{channel}" in names
            and f"auto_cor_B_{channel}" in names
        ):
            required.add(f"max_auto_cross_corr_{channel}")
    return required


def existing_output_is_current(source: Path, output: Path) -> bool:
    """Return whether an existing output contains the current data contract."""

    if not output.exists() or output.stat().st_mtime < source.stat().st_mtime:
        return False
    source_schema = pq.read_schema(source)
    output_schema = pq.read_schema(output)
    if not required_output_columns(source_schema).issubset(output_schema.names):
        return False
    if not set(S4_FILTER_COLUMNS).issubset(output_schema.names):
        return False

    s4_values = pq.read_table(
        output,
        columns=list(S4_FILTER_COLUMNS),
    ).to_pandas()
    return len(filter_mutual_s4(s4_values)) == len(s4_values)


def write_parquet_atomic(frame: pd.DataFrame, output: Path) -> None:
    """Write a compact scalar Parquet and atomically publish it."""

    table = pa.Table.from_pandas(frame, preserve_index=False)
    remaining_lists = [
        field.name for field in table.schema if is_list_type(field.type)
    ]
    if remaining_lists:
        raise ValueError(
            f"Level-1 output still contains list columns: {remaining_lists}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        pq.write_table(table, temporary, compression="zstd")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()


def reduce_file(
    source: Path,
    output: Path,
    default_seconds_per_sample: float = DEFAULT_SECONDS_PER_SAMPLE,
    batch_size: int = 256,
    overwrite: bool = False,
) -> ReductionResult:
    """Reduce one raw correlation file to a scalar-only level-1 file."""

    if not overwrite and existing_output_is_current(source, output):
        metadata = pq.ParquetFile(output).metadata
        return ReductionResult(
            source=source,
            output=output,
            row_count=metadata.num_rows,
            status="skipped",
        )

    parquet_file = pq.ParquetFile(source)
    frame = read_scalar_frame(parquet_file)
    for channel in channel_suffixes(parquet_file.schema_arrow):
        add_channel_metrics(
            frame,
            parquet_file,
            channel,
            default_seconds_per_sample,
            batch_size,
        )

    frame = filter_mutual_s4(frame)
    write_parquet_atomic(frame, output)
    stale_error = error_path_for(output)
    if stale_error.exists():
        stale_error.unlink()
    return ReductionResult(
        source=source,
        output=output,
        row_count=len(frame),
        status="written",
    )


def reduce_file_task(
    source: Path,
    output: Path,
    default_seconds_per_sample: float,
    batch_size: int,
    overwrite: bool,
) -> ReductionResult:
    """Worker wrapper that preserves a traceback beside a failed output."""

    try:
        return reduce_file(
            source,
            output,
            default_seconds_per_sample,
            batch_size,
            overwrite,
        )
    except Exception as error:
        error_path = error_path_for(output)
        error_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = error_path.with_name(f".{error_path.name}.tmp")
        diagnostic = (
            f"Source: {source}\n"
            f"Output: {output}\n"
            f"Exception: {type(error).__name__}: {error}\n\n"
            f"Traceback:\n{traceback.format_exc()}"
        )
        try:
            temporary.write_text(diagnostic, encoding="utf-8")
            temporary.replace(error_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return ReductionResult(
            source=source,
            output=output,
            row_count=0,
            status="failed",
            error=f"{type(error).__name__}: {error}",
        )


def unified_scalar_schema(files: list[Path]) -> pa.Schema:
    """Return a permissive union schema for level-1 files."""

    schemas = [pq.read_schema(path).remove_metadata() for path in files]
    return pa.unify_schemas(schemas, promote_options="permissive")


def align_table_to_schema(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """Add missing nullable columns and order/cast a table to one schema."""

    arrays = []
    for field in schema:
        if field.name not in table.column_names:
            arrays.append(pa.nulls(table.num_rows, type=field.type))
            continue
        column = table[field.name]
        if column.type != field.type:
            column = column.cast(field.type)
        arrays.append(column)
    return pa.Table.from_arrays(arrays, schema=schema)


def concatenate_level1_files(
    files: list[Path],
    output: Path,
    batch_size: int = 65_536,
) -> int:
    """Stream every individual level-1 file into one atomic Parquet."""

    if not files:
        raise ValueError("No level-1 files were supplied for concatenation")
    schema = unified_scalar_schema(files)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    total_rows = 0
    writer: pq.ParquetWriter | None = None
    try:
        writer = pq.ParquetWriter(temporary, schema, compression="zstd")
        for path in files:
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                table = align_table_to_schema(
                    pa.Table.from_batches([batch]).replace_schema_metadata(),
                    schema,
                )
                writer.write_table(table)
                total_rows += table.num_rows
        writer.close()
        writer = None
        temporary.replace(output)
    finally:
        if writer is not None:
            writer.close()
        if temporary.exists():
            temporary.unlink()
    return total_rows


def run(
    input_dir: Path,
    output_dir: Path | None = None,
    pattern: str = DEFAULT_PATTERN,
    workers: int = 4,
    default_seconds_per_sample: float = DEFAULT_SECONDS_PER_SAMPLE,
    batch_size: int = 256,
    overwrite: bool = False,
    combined_name: str = DEFAULT_COMBINED_NAME,
) -> Path:
    """Reduce all inputs with a file pool, then concatenate every result."""

    if workers < 1:
        raise ValueError("workers must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if Path(combined_name).name != combined_name:
        raise ValueError("combined_name must be a filename, not a path")

    input_dir = input_dir.resolve()
    output_dir = (
        output_dir.resolve()
        if output_dir is not None
        else input_dir / "lvl1"
    )
    sources = discover_input_files(input_dir, pattern)
    outputs = {
        source: output_path_for(source, output_dir) for source in sources
    }

    print(f"Source files: {len(sources)}")
    print(f"Workers: {workers}")
    print(f"Output directory: {output_dir}")

    results: list[ReductionResult] = []
    if workers == 1:
        for source in sources:
            result = reduce_file_task(
                source,
                outputs[source],
                default_seconds_per_sample,
                batch_size,
                overwrite,
            )
            results.append(result)
            print(
                f"[{result.status.upper()}] {source.name} -> "
                f"{result.output.name} ({result.row_count} rows)"
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    reduce_file_task,
                    source,
                    outputs[source],
                    default_seconds_per_sample,
                    batch_size,
                    overwrite,
                ): source
                for source in sources
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    f"[{result.status.upper()}] {result.source.name} -> "
                    f"{result.output.name} ({result.row_count} rows)"
                )

    failures = [result for result in results if result.status == "failed"]
    if failures:
        details = "\n".join(
            f"  {result.source.name}: {result.error}" for result in failures
        )
        raise RuntimeError(
            f"{len(failures)} file(s) failed. No combined Parquet was "
            f"written. Diagnostics are in {output_dir}.\n{details}"
        )

    combined_path = output_dir / combined_name
    ordered_outputs = [outputs[source] for source in sources]
    combined_rows = concatenate_level1_files(
        ordered_outputs,
        combined_path,
    )
    print(
        f"[COMBINED] {len(ordered_outputs)} files, {combined_rows} rows -> "
        f"{combined_path}"
    )
    return combined_path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: INPUT_DIR/lvl1",
    )
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--seconds-per-sample",
        type=float,
        default=DEFAULT_SECONDS_PER_SAMPLE,
        help=(
            "Fallback cadence when it cannot be inferred from best_lag and "
            "time_delay (default: 0.05 seconds)"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--combined-name", default=DEFAULT_COMBINED_NAME)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild individual level-1 files that already exist",
    )
    return parser


def main() -> None:
    """Run the command-line workflow."""

    arguments = build_parser().parse_args()
    run(
        input_dir=arguments.input_dir,
        output_dir=arguments.output_dir,
        pattern=arguments.pattern,
        workers=arguments.workers,
        default_seconds_per_sample=arguments.seconds_per_sample,
        batch_size=arguments.batch_size,
        overwrite=arguments.overwrite,
        combined_name=arguments.combined_name,
    )


if __name__ == "__main__":
    main()
