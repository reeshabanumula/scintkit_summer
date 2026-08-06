from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy import signal


MODULE_DIR = (
    Path(__file__).parents[1]
    / "src"
    / "scintkit"
    / "space_receiver_processing"
)
sys.path.insert(0, str(MODULE_DIR))

import reduce_cross_correlations_lvl1 as reducer


def write_raw_correlation(path: Path, include_channel_2: bool = False) -> None:
    frame = pd.DataFrame(
        {
            "source_a": ["receiver_a.pq"],
            "source_b": ["receiver_b.pq"],
            "minute": [pd.Timestamp("2023-05-01 00:00:00")],
            "svid": [5],
            "r_A": [[7.1, 35.1, 0.4]],
            "r_B": [[7.2, 35.2, 0.5]],
            "auto_cor_A_1": [[0.2, 0.5, 1.0, 0.5, 0.2]],
            "auto_cor_A_max_1": [1.0],
            "auto_cor_B_1": [[0.3, 0.6, 1.0, 0.6, 0.3]],
            "auto_cor_B_max_1": [1.0],
            "corr_norm_1": [[0.1, 0.3, 0.4, 0.7, 0.2]],
            "lag_norm_1": [[-2, -1, 0, 1, 2]],
            "max_corr_1": [0.7],
            "best_lag_1": [1],
            "time_delay_1": [0.05],
            "unneeded_list": [[1.0, 2.0]],
        }
    )
    if include_channel_2:
        frame["auto_cor_A_2"] = [[0.2, 0.5, 1.0, 0.5, 0.2]]
        frame["auto_cor_A_max_2"] = 1.0
        frame["auto_cor_B_2"] = [[0.3, 0.6, 1.0, 0.6, 0.3]]
        frame["auto_cor_B_max_2"] = 1.0
        frame["corr_norm_2"] = [[0.1, 0.3, 0.4, 0.7, 0.2]]
        frame["lag_norm_2"] = [[-2, -1, 0, 1, 2]]
        frame["max_corr_2"] = 0.7
        frame["best_lag_2"] = 1
        frame["time_delay_2"] = 0.05
    frame.to_parquet(path, index=False)


def test_reduce_file_flattens_coordinates_and_calculates_crossings(tmp_path):
    source = tmp_path / "sc003_corrs_20230501_7.212N_35.906E.pq"
    output = tmp_path / "lvl1" / (
        "sc003_corrs_20230501_7.212N_35.906E_lvl1.pq"
    )
    write_raw_correlation(source)

    result = reducer.reduce_file(source, output)

    assert result.status == "written"
    reduced = pd.read_parquet(output)
    assert reduced.loc[0, "r_A_lat"] == pytest.approx(7.1)
    assert reduced.loc[0, "r_A_lon"] == pytest.approx(35.1)
    assert reduced.loc[0, "r_A_height"] == pytest.approx(0.4)
    assert reduced.loc[0, "r_B_lat"] == pytest.approx(7.2)
    assert reduced.loc[0, "r_B_lon"] == pytest.approx(35.2)
    assert reduced.loc[0, "r_B_height"] == pytest.approx(0.5)

    expected_a_decorrelation = (
        1 + ((1 / np.e) - 0.5) / (0.2 - 0.5)
    ) * 0.05
    expected_b_decorrelation = (
        1 + ((1 / np.e) - 0.6) / (0.3 - 0.6)
    ) * 0.05
    assert reduced.loc[0, "auto_cor_A_max_1"] == pytest.approx(1.0)
    assert reduced.loc[0, "auto_cor_B_max_1"] == pytest.approx(1.0)
    assert reduced.loc[0, "decorrelation_time_A_1"] == pytest.approx(
        expected_a_decorrelation
    )
    assert reduced.loc[0, "decorrelation_time_B_1"] == pytest.approx(
        expected_b_decorrelation
    )
    assert reduced.loc[
        0, "auto_cor_A_time_at_max_corr_1"
    ] == pytest.approx(0.03)
    assert reduced.loc[
        0, "auto_cor_B_time_at_max_corr_1"
    ] == pytest.approx(0.0375)
    auto_a = np.asarray([0.2, 0.5, 1.0, 0.5, 0.2])
    auto_b = np.asarray([0.3, 0.6, 1.0, 0.6, 0.3])
    expected_auto_cross = np.max(
        signal.correlate(
            auto_a - auto_a.mean(),
            auto_b - auto_b.mean(),
            mode="full",
            method="direct",
        )
        / (
            np.linalg.norm(auto_a - auto_a.mean())
            * np.linalg.norm(auto_b - auto_b.mean())
        )
    )
    assert reduced.loc[0, "max_auto_cross_corr_1"] == pytest.approx(
        expected_auto_cross
    )

    output_schema = pq.read_schema(output)
    assert not any(
        reducer.is_list_type(field.type) for field in output_schema
    )
    for removed in (
        "r_A",
        "r_B",
        "auto_cor_A_1",
        "auto_cor_B_1",
        "corr_norm_1",
        "lag_norm_1",
        "unneeded_list",
    ):
        assert removed not in reduced


def test_run_reduces_all_sources_and_combines_union_schema(tmp_path):
    first = tmp_path / "sc003_corrs_20230501_7.212N_35.906E.pq"
    second = tmp_path / "sc003_corrs_20230502_7.212N_35.906E.pq"
    write_raw_correlation(first)
    write_raw_correlation(second, include_channel_2=True)

    combined_path = reducer.run(
        input_dir=tmp_path,
        pattern="sc003_corrs_*.pq",
        workers=1,
    )

    output_dir = tmp_path / "lvl1"
    assert (output_dir / f"{first.stem}_lvl1.pq").exists()
    assert (output_dir / f"{second.stem}_lvl1.pq").exists()
    assert combined_path == output_dir / "sc003_corrs_all_lvl1.pq"

    combined = pd.read_parquet(combined_path)
    assert len(combined) == 2
    assert combined["source_a"].tolist() == [
        "receiver_a.pq",
        "receiver_a.pq",
    ]
    assert pd.isna(combined.loc[0, "decorrelation_time_A_2"])
    assert np.isfinite(combined.loc[1, "decorrelation_time_A_2"])
    assert pd.isna(combined.loc[0, "max_auto_cross_corr_2"])
    assert np.isfinite(combined.loc[1, "max_auto_cross_corr_2"])
    assert not any(
        reducer.is_list_type(field.type)
        for field in pq.read_schema(combined_path)
    )


def test_cross_metrics_are_recovered_when_scalar_columns_are_missing(
    tmp_path,
):
    source = tmp_path / "sc003_corrs_missing_scalars.pq"
    output = tmp_path / "lvl1" / "sc003_corrs_missing_scalars_lvl1.pq"
    write_raw_correlation(source)
    frame = pd.read_parquet(source).drop(
        columns=["max_corr_1", "best_lag_1", "time_delay_1"]
    )
    frame.to_parquet(source, index=False)

    reducer.reduce_file(source, output)

    reduced = pd.read_parquet(output)
    assert reduced.loc[0, "max_corr_1"] == pytest.approx(0.7)
    assert reduced.loc[0, "best_lag_1"] == pytest.approx(1)
    assert reduced.loc[0, "time_delay_1"] == pytest.approx(0.05)


def test_combined_output_is_not_treated_as_an_input(tmp_path):
    source = tmp_path / "sc003_corrs_20230501.pq"
    source.touch()
    (tmp_path / "sc003_corrs_20230501_lvl1.pq").touch()

    discovered = reducer.discover_input_files(tmp_path, "*.pq")

    assert discovered == [source]


def test_old_output_is_rebuilt_when_new_metric_is_missing(tmp_path):
    source = tmp_path / "sc003_corrs_20230501.pq"
    output = tmp_path / "lvl1" / "sc003_corrs_20230501_lvl1.pq"
    write_raw_correlation(source)
    output.parent.mkdir()
    pd.DataFrame({"source_a": ["old"]}).to_parquet(output, index=False)

    result = reducer.reduce_file(source, output)

    assert result.status == "written"
    assert "max_auto_cross_corr_1" in pq.read_schema(output).names
