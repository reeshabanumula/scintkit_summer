import functions as f
import CONFIG as cf

import importlib
import shutil
from pathlib import Path


# ============================================================
# RELOAD MODULES
# ============================================================

importlib.reload(f)
importlib.reload(cf)


# ============================================================
# TEST SETTINGS
# ============================================================

print("\n")
print("================================================")
print("         SCRATCH WORKFLOW END-TO-END TEST")
print("================================================")


storage = Path(cf.storage_folder).resolve()
scratch = Path(cf.scratch_folder).resolve()
results = Path(cf.output_folder).resolve()


# ============================================================
# CHECK REQUIRED FOLDERS
# ============================================================

print("\nChecking folders...")

if not storage.exists():
    raise FileNotFoundError(
        f"\nStorage folder does not exist:\n{storage}"
    )

print(f"Storage folder: {storage}")
print(f"Scratch folder: {scratch}")
print(f"Results folder: {results}")


# ============================================================
# CREATE SCRATCH PARENT IF NEEDED
# ============================================================

scratch.mkdir(
    parents=True,
    exist_ok=True
)

results.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# RECORD ORIGINAL STORAGE FILES
# ============================================================

print("\n")
print("================================================")
print("        RECORDING ORIGINAL STORAGE")
print("================================================")


original_storage_files = sorted(
    file.name
    for file in storage.iterdir()
    if file.is_file()
)


print(
    f"Found {len(original_storage_files)} "
    f"files in storage."
)


for filename in original_storage_files:
    print(f"  {filename}")


if len(original_storage_files) == 0:

    raise RuntimeError(
        "Storage folder is empty.\n"
        "Put a few .bin.zip and/or .pq files "
        "in storage before running this test."
    )


# ============================================================
# PREPARE INPUT PATTERN
# ============================================================

# Your current configuration contains the full path:
#
# C:\...\something\*.bin.zip
#
# But the scratch function is going to search inside:
#
# scratch/run_XXXX/input/
#
# Therefore we extract only:
#
# *.bin.zip

input_pattern = Path(
    cf.input_pattern
).name


print("\n")
print("Input pattern from CONFIG:")
print(cf.input_pattern)

print("\nPattern used inside scratch:")
print(input_pattern)


# ============================================================
# TEST 1:
# CREATE SCRATCH WORKSPACE
# ============================================================

print("\n")
print("================================================")
print("       TEST 1: CREATE SCRATCH WORKSPACE")
print("================================================")


processing_folder = f.create_processing_scratch(
    storage=cf.storage_folder,
    scratch=cf.scratch_folder,
    input_pattern=input_pattern,
    temp_root=cf.temp_root,
    n_workers=1,
    verbose=cf.verbose
)


# ============================================================
# CHECK RETURNED PROCESSING FOLDER
# ============================================================

processing_folder = Path(
    processing_folder
).resolve()


print("\nReturned processing folder:")
print(processing_folder)


if not processing_folder.exists():

    raise RuntimeError(
        "create_processing_scratch() returned a "
        "folder that does not exist."
    )


print("\nProcessing folder exists.")


# ============================================================
# FIND SCRATCH RUN FOLDER
# ============================================================

scratch_run_folder = processing_folder.parent

scratch_input = (
    scratch_run_folder /
    "input"
)

scratch_pq = (
    scratch_run_folder /
    "pq"
)


print("\nScratch run structure:")
print(f"Run folder:  {scratch_run_folder}")
print(f"Input:       {scratch_input}")
print(f"PQ:          {scratch_pq}")


# ============================================================
# VERIFY INPUT FOLDER
# ============================================================

print("\n")
print("Checking scratch input files...")

if not scratch_input.exists():

    raise RuntimeError(
        "Scratch input folder does not exist."
    )


scratch_input_files = sorted(
    file.name
    for file in scratch_input.iterdir()
    if file.is_file()
)


print(
    f"Found {len(scratch_input_files)} "
    f"files in scratch input."
)


for filename in scratch_input_files:
    print(f"  {filename}")


# ============================================================
# VERIFY STORAGE WAS COPIED
# ============================================================

missing_from_scratch = (
    set(original_storage_files)
    -
    set(scratch_input_files)
)


if missing_from_scratch:

    raise RuntimeError(
        "Some storage files were not copied "
        "into the scratch input folder:\n"
        +
        "\n".join(
            str(x)
            for x in sorted(missing_from_scratch)
        )
    )


print(
    "\nAll original storage files were "
    "successfully copied to scratch."
)


# ============================================================
# VERIFY PQ FILES
# ============================================================

print("\n")
print("================================================")
print("          CHECKING PARQUET FILES")
print("================================================")


pq_files = sorted(
    scratch_pq.glob("*.pq")
)


print(
    f"Found {len(pq_files)} parquet files "
    f"in scratch."
)


for pq_file in pq_files:
    print(f"  {pq_file.name}")


if len(pq_files) == 0:

    raise RuntimeError(
        "No parquet files were found in "
        "the scratch processing folder."
    )


print(
    "\nParquet processing folder looks good."
)


# ============================================================
# TEST 2:
# SIMULATE PROCESSING OUTPUT
# ============================================================

print("\n")
print("================================================")
print("       SIMULATING SUCCESSFUL PROCESSING")
print("================================================")


# Your cleanup function assumes that the actual
# S4/correlation processing has already finished.
#
# For this test, we create a small marker file
# inside the results folder to represent output
# produced by the processing script.

test_output = (
    results /
    "SCRATCH_TEST_OUTPUT.txt"
)


with open(
    test_output,
    "w"
) as file:

    file.write(
        "This file represents output "
        "created during the scratch test.\n"
    )


print(
    f"Created test output:\n"
    f"{test_output}"
)


# ============================================================
# TEST 3:
# CLEAN UP SCRATCH
# ============================================================

print("\n")
print("================================================")
print("       TEST 3: CLEANUP SCRATCH WORKSPACE")
print("================================================")


f.cleanup_processing_scratch(
    processing_folder=processing_folder,
    scratch_folder=cf.scratch_folder
)


# ============================================================
# VERIFY SCRATCH WAS DELETED
# ============================================================

print("\n")
print("Checking that scratch run was deleted...")


if scratch_run_folder.exists():

    raise RuntimeError(
        "Scratch cleanup FAILED.\n"
        f"The following folder still exists:\n"
        f"{scratch_run_folder}"
    )


print(
    "Scratch run successfully deleted."
)


# ============================================================
# VERIFY STORAGE STILL EXISTS
# ============================================================

print("\n")
print("================================================")
print("       CHECKING ORIGINAL STORAGE")
print("================================================")


if not storage.exists():

    raise RuntimeError(
        "CRITICAL ERROR: Original storage "
        "folder no longer exists!"
    )


current_storage_files = sorted(
    file.name
    for file in storage.iterdir()
    if file.is_file()
)


print(
    f"Storage currently contains "
    f"{len(current_storage_files)} files."
)


for filename in current_storage_files:
    print(f"  {filename}")


# ============================================================
# VERIFY NO STORAGE FILES WERE DELETED
# ============================================================

if (
    current_storage_files
    != original_storage_files
):

    raise RuntimeError(
        "\nCRITICAL ERROR:\n"
        "The contents of storage changed "
        "during the scratch test.\n\n"
        f"Before:\n{original_storage_files}\n\n"
        f"After:\n{current_storage_files}"
    )


print(
    "\nOriginal storage files are "
    "completely unchanged."
)


# ============================================================
# VERIFY TEST OUTPUT EXISTS
# ============================================================

print("\nChecking results folder...")


if not test_output.exists():

    raise RuntimeError(
        "Test output disappeared from "
        "the results folder."
    )


print(
    f"Test output still exists:\n"
    f"{test_output}"
)


# ============================================================
# CLEAN UP TEST OUTPUT
# ============================================================

print("\nRemoving test output...")

test_output.unlink()

print("Test output removed.")


# ============================================================
# FINAL SUCCESS MESSAGE
# ============================================================

print("\n")
print("================================================")
print("              ALL TESTS PASSED")
print("================================================")

print(
    "\nThe scratch workflow successfully:"
)

print(
    "  [1] Created a temporary scratch run"
)

print(
    "  [2] Copied storage files into scratch"
)

print(
    "  [3] Converted/found parquet files in scratch"
)

print(
    "  [4] Simulated successful processing"
)

print(
    "  [5] Deleted the scratch run"
)

print(
    "  [6] Preserved the original storage files"
)

print(
    "  [7] Preserved the results output"
)

print("\n================================================")