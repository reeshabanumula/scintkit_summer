from pathlib import Path
from scintkit.pipelines.lvl0_convert_to_pq import run_conversion



# ============================================================
# DIRECT SCRATCH CONVERSION TEST
# ============================================================

scratch_input = Path(
    r"C:\Users\irees\Downloads\Summer_learning\research26"
    r"\scratch\run_20260727_182017\input"
)

scratch_output = Path(
    r"C:\Users\irees\Downloads\Summer_learning\research26"
    r"\scratch\run_20260727_182017\pq"
)


print("==============================================")
print("DIRECT CONVERSION TEST")
print("==============================================")

print("\nInput folder:")
print(scratch_input)

print("\nOutput folder:")
print(scratch_output)


# ============================================================
# CHECK INPUT FILES
# ============================================================

files = list(
    scratch_input.glob("*.bin.zip")
)

print(
    f"\nFound {len(files)} bin.zip files:"
)

for file in files:
    print(file)


if len(files) == 0:
    raise RuntimeError(
        "No bin.zip files found in scratch input."
    )


# ============================================================
# RUN CONVERSION
# ============================================================

print("\nStarting conversion...\n")


run_conversion(
    mode="single",
    input_pattern="*.bin.zip",
    input_root=str(scratch_input),
    output_root=str(scratch_output),
    infer_missing=False,
    n_workers=1,
    temp_root=None,
    verbose=True,
)


# ============================================================
# CHECK OUTPUT
# ============================================================

pq_files = list(
    scratch_output.glob("*.pq")
)

print("\n==============================================")
print("CONVERSION TEST COMPLETE")
print("==============================================")

print(
    f"\nFound {len(pq_files)} PQ files:"
)

for file in pq_files:
    print(file)