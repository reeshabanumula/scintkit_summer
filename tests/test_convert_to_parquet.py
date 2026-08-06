from __future__ import annotations

from pathlib import Path
import zipfile

import pandas as pd

from scintkit.services import convert_to_parquet


def test_process_one_accepts_scintpi2_dat_zip_and_adds_nan_snr2(
    tmp_path,
):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    temp_root = tmp_path / "temp"
    input_root.mkdir()
    temp_root.mkdir()
    archive = input_root / (
        "scintpi2_20221004_2000_359060.7812W_"
        "72122.4141S_v324.dat.zip"
    )
    inner_name = archive.name.removesuffix(".zip")
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(
            inner_name,
            "2230 100.00 0 0 10 45 180 30 99 1.0 0 2.0 0 3.0 0 4.0\n"
            "2230 100.05 0 0 10 45 180 31 98 1.1 0 2.1 0 3.1 0 4.1\n",
        )

    result = convert_to_parquet.process_one(
        archive,
        input_root=input_root,
        output_root=output_root,
        temp_root=str(temp_root),
        verbose=False,
    )

    result_path = Path(result)
    assert result_path.name == archive.name.removesuffix(".dat.zip") + "_lvl0.pq"
    converted = pd.read_parquet(result_path)
    assert "snr2" in converted.columns
    assert converted["snr2"].isna().all()
