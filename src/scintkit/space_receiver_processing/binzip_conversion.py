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