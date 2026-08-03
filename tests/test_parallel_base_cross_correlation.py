from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import pandas as pd
import pytest


MODULE_DIR = (
    Path(__file__).parents[1]
    / "src"
    / "scintkit"
    / "space_receiver_processing"
)
sys.path.insert(0, str(MODULE_DIR))

import parallel_base_cross_correlation as cross_correlation


def _return_pair_rows(pair):
    """Picklable worker used to exercise the real process pool."""

    file_a, file_b = pair
    return [{"source_a": file_a.name}], file_a, file_b


def test_discover_source_files_groups_filename_dates(tmp_path, monkeypatch):
    filenames = [
        "scintpi3_20221005_0000_359060.7812W_72122.4141S_v325.bin.zip",
        "scintpi3_20221004_2000_359072.7500W_72126.9375S_v325.bin.zip",
        "scintpi3_20221004_2000_359060.7812W_72122.4141S_v325.bin.zip",
    ]
    for filename in filenames:
        (tmp_path / filename).touch()

    monkeypatch.setattr(cross_correlation.cf, "storage_folder", str(tmp_path))
    monkeypatch.setattr(cross_correlation.cf, "input_pattern", "*.bin.zip")

    batches = cross_correlation.discover_source_files_by_day()

    assert list(batches) == [date(2022, 10, 4), date(2022, 10, 5)]
    assert [file.name for file in batches[date(2022, 10, 4)]] == sorted(
        filenames[1:]
    )
    assert [file.name for file in batches[date(2022, 10, 5)]] == [
        filenames[0]
    ]


def test_daily_conversion_uses_only_batch_files_and_all_workers(
    tmp_path,
    monkeypatch,
):
    storage_folder = tmp_path / "storage"
    scratch_folder = tmp_path / "scratch"
    storage_folder.mkdir()
    source_files = [
        storage_folder
        / "scintpi3_20221004_2000_359060.7812W_72122.4141S_v325.bin.zip",
        storage_folder
        / "scintpi3_20221004_2000_359072.7500W_72126.9375S_v325.bin.zip",
    ]
    for source_file in source_files:
        source_file.touch()

    captured: dict[str, object] = {}

    def fake_run_conversion(**kwargs):
        captured.update(kwargs)
        output_root = Path(kwargs["output_root"])
        outputs = []
        for input_file in kwargs["flist"]:
            output = output_root / f"{Path(input_file).stem}.pq"
            output.touch()
            outputs.append(str(output))
        return outputs

    monkeypatch.setattr(
        cross_correlation,
        "run_conversion",
        fake_run_conversion,
    )
    monkeypatch.setattr(
        cross_correlation.cf,
        "storage_folder",
        str(storage_folder),
    )
    monkeypatch.setattr(
        cross_correlation.cf,
        "scratch_folder",
        str(scratch_folder),
    )
    monkeypatch.setattr(cross_correlation.cf, "max_workers", 4)
    monkeypatch.setattr(cross_correlation.cf, "verbose", False)

    processing_folder = cross_correlation.create_cleanable_processing_scratch(
        date(2022, 10, 4),
        source_files,
    )

    assert captured["flist"] == [str(file) for file in source_files]
    assert captured["n_workers"] == 4
    assert Path(captured["temp_root"]).parent == processing_folder.parent
    assert "run_20221004_" in processing_folder.parent.name
    assert len(list(processing_folder.glob("*.pq"))) == 2

    cross_correlation.f.cleanup_processing_scratch(
        processing_folder,
        scratch_folder,
    )
    assert not processing_folder.parent.exists()


def test_parallel_worker_rows_are_combined_in_parent(monkeypatch):
    file_pairs = [
        (Path("receiver_a_1.pq"), Path("receiver_b_1.pq")),
        (Path("receiver_a_2.pq"), Path("receiver_b_2.pq")),
    ]
    observed: dict[str, object] = {}

    class FakeExecutor:
        def __init__(self, max_workers):
            observed["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def map(self, function, pairs, chunksize):
            observed["chunksize"] = chunksize
            return [
                ([{"worker": index}], file_a, file_b)
                for index, (file_a, file_b) in enumerate(pairs)
            ]

    monkeypatch.setattr(cross_correlation.cf, "max_workers", 4)
    monkeypatch.setattr(
        cross_correlation,
        "ProcessPoolExecutor",
        FakeExecutor,
    )

    rows, processed_pairs = cross_correlation.run_pairs(file_pairs)

    assert observed == {"max_workers": 4, "chunksize": 1}
    assert rows == [{"worker": 0}, {"worker": 1}]
    assert processed_pairs == file_pairs


def test_incomplete_receiver_date_honors_skip_configuration(
    monkeypatch,
    capsys,
):
    only_receiver_a = Path(
        "scintpi3_20221004_2000_359060.7812W_72122.4141S_v325_lvl0.pq"
    )
    monkeypatch.setattr(cross_correlation.cf, "r_latitude", 7.21224141)
    monkeypatch.setattr(cross_correlation.cf, "r_longitude", 35.90607812)
    monkeypatch.setattr(cross_correlation.cf, "lat_tol", 0.0005)
    monkeypatch.setattr(cross_correlation.cf, "lon_tol", 0.0005)
    monkeypatch.setattr(
        cross_correlation.cf,
        "unpaired_file_action",
        "skip",
    )

    result = cross_correlation.receiver_files_or_skip([only_receiver_a])

    assert result is None
    message = capsys.readouterr().out
    assert "Skipping incomplete date batch" in message
    assert "Receiver A files=1" in message
    assert "Receiver B candidates=0" in message


def test_real_process_pool_combines_results_from_four_workers(monkeypatch):
    file_pairs = [
        (Path(f"receiver_a_{index}.pq"), Path(f"receiver_b_{index}.pq"))
        for index in range(8)
    ]
    monkeypatch.setattr(cross_correlation.cf, "max_workers", 4)
    monkeypatch.setattr(
        cross_correlation,
        "process_file_pair",
        _return_pair_rows,
    )

    rows, processed_pairs = cross_correlation.run_pairs(file_pairs)

    assert [row["source_a"] for row in rows] == [
        file_a.name for file_a, _ in file_pairs
    ]
    assert processed_pairs == file_pairs


def test_failed_daily_conversion_removes_partial_scratch(tmp_path, monkeypatch):
    storage_folder = tmp_path / "storage"
    scratch_folder = tmp_path / "scratch"
    storage_folder.mkdir()
    source_file = (
        storage_folder
        / "scintpi3_20221004_2000_359060.7812W_72122.4141S_v325.bin.zip"
    )
    source_file.touch()

    monkeypatch.setattr(
        cross_correlation,
        "run_conversion",
        lambda **kwargs: [""],
    )
    monkeypatch.setattr(
        cross_correlation.cf,
        "storage_folder",
        str(storage_folder),
    )
    monkeypatch.setattr(
        cross_correlation.cf,
        "scratch_folder",
        str(scratch_folder),
    )
    monkeypatch.setattr(cross_correlation.cf, "max_workers", 4)
    monkeypatch.setattr(cross_correlation.cf, "verbose", False)

    with pytest.raises(RuntimeError, match="Conversion failed for 1 files"):
        cross_correlation.create_cleanable_processing_scratch(
            date(2022, 10, 4),
            [source_file],
        )

    assert list(scratch_folder.iterdir()) == []


def test_daily_save_combines_all_rows_into_filename_date(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cross_correlation.cf,
        "output_folder",
        str(tmp_path),
    )
    monkeypatch.setattr(
        cross_correlation.cf,
        "cross_correlation_file",
        "sc002_corrs.pq",
    )
    monkeypatch.setattr(cross_correlation.cf, "r_latitude", 7.21224141)
    monkeypatch.setattr(cross_correlation.cf, "r_longitude", 35.90607812)

    rows = [
        {"minute": pd.Timestamp("2022-10-04 23:59"), "worker": 0},
        {"minute": pd.Timestamp("2022-10-05 00:00"), "worker": 1},
    ]
    output_paths = cross_correlation.save_correlations(
        rows,
        filename_day=date(2022, 10, 4),
    )

    assert len(output_paths) == 1
    assert "20221004" in output_paths[0].name
    saved = pd.read_parquet(output_paths[0])
    assert saved["worker"].tolist() == [0, 1]


def test_process_one_day_does_not_write_a_skipped_date(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cross_correlation,
        "find_file_pairs",
        lambda processing_folder, files=None: [],
    )

    def unexpected_run_pairs(paired_files):
        raise AssertionError("run_pairs should not run for an incomplete date")

    monkeypatch.setattr(cross_correlation, "run_pairs", unexpected_run_pairs)

    outputs, pairs, row_count = cross_correlation.process_one_day(
        date(2022, 10, 4),
        tmp_path,
    )

    assert outputs == []
    assert pairs == []
    assert row_count == 0


def test_run_cleans_each_date_before_staging_the_next(tmp_path, monkeypatch):
    first_day = date(2022, 10, 4)
    second_day = date(2022, 10, 5)
    source_batches = {
        first_day: [Path("first.bin.zip")],
        second_day: [Path("second.bin.zip")],
    }
    events: list[tuple[str, date]] = []
    processing_days: dict[Path, date] = {}

    monkeypatch.setattr(
        cross_correlation,
        "discover_source_files_by_day",
        lambda: source_batches,
    )
    monkeypatch.setattr(
        cross_correlation,
        "receiver_files_or_skip",
        lambda files: (files, files),
    )

    def fake_create(filename_day, source_files):
        events.append(("stage", filename_day))
        processing_folder = tmp_path / filename_day.isoformat() / "pq"
        processing_days[processing_folder] = filename_day
        return processing_folder

    def fake_process(filename_day, processing_folder, files=None):
        events.append(("process", filename_day))
        return [Path(f"{filename_day}.pq")], [
            (Path(f"{filename_day}_a.pq"), Path(f"{filename_day}_b.pq"))
        ], 2

    def fake_cleanup(processing_folder, scratch_folder):
        events.append(("cleanup", processing_days[processing_folder]))

    monkeypatch.setattr(
        cross_correlation,
        "create_cleanable_processing_scratch",
        fake_create,
    )
    monkeypatch.setattr(cross_correlation, "process_one_day", fake_process)
    monkeypatch.setattr(
        cross_correlation.f,
        "cleanup_processing_scratch",
        fake_cleanup,
    )
    monkeypatch.setattr(
        cross_correlation,
        "write_processed_pairs_log",
        lambda pairs: tmp_path / "processed_pairs.txt",
    )
    monkeypatch.setattr(
        cross_correlation.cf,
        "scratch_folder",
        str(tmp_path),
    )

    output_paths, _ = cross_correlation.run()

    assert events == [
        ("stage", first_day),
        ("process", first_day),
        ("cleanup", first_day),
        ("stage", second_day),
        ("process", second_day),
        ("cleanup", second_day),
    ]
    assert output_paths == [Path(f"{first_day}.pq"), Path(f"{second_day}.pq")]
