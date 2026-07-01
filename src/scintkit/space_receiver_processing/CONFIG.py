import pandas as pd

Data_folder1 = pd.read_parquet(r'C:\Users\irees\Downloads\Summer_learning\research26\Brazil_22_lvl0\scintpi3_20221004_2000_359060.7812W_72122.4141S_v325_lvl0.pq')
Data_folder2 = pd.read_parquet(r'C:\Users\irees\Downloads\Summer_learning\research26\Brazil_22_lvl0\scintpi3_20221004_2000_359072.7500W_72126.9375S_v325_lvl0.pq')
thresh = 0.2

#binzip conversion

from scintkit.pipelines.lvl0_convert_to_pq import run_conversion

run_conversion(
    mode="single",
    input_pattern=r"C:\Users\irees\Downloads\Summer_learning\research26\Brazil_22_data\*.bin.zip",
    input_root=r"C:\Users\irees\Downloads\Summer_learning\research26\Brazil_22_data",
    output_root=r"C:\Users\irees\Downloads\Summer_learning\research26\Brazil_22_lvl0",
    infer_missing=True,
    n_workers=1,
    temp_root = r"C:\Users\irees\Downloads\Summer_learning\research26\tmp",
    verbose=True,
)

