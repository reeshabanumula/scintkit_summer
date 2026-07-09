from scintkit.pipelines.lvl0_convert_to_pq import run_conversion
import CONFIG as cf
import importlib

importlib.reload(cf)

print(cf.input_pattern)
print(cf.input_root)
print(cf.output_root)


import glob

print("Searching:")
print(cf.input_root + "\\" + cf.input_pattern)

files = glob.glob(cf.input_root + "\\" + cf.input_pattern)

print("Found:", len(files))
print(files[:3])




run_conversion(
    mode="single",
    input_pattern= cf.input_pattern,
    input_root= cf.input_root,
    output_root= cf.output_root,
    infer_missing=False,
    n_workers=1,
    temp_root = cf.temp_root,
    verbose= cf.verbose,
)

#should be independent

#run data through this script first to create pq
