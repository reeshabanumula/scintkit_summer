from scintkit.pipelines.lvl0_convert_to_pq import run_conversion
import CONFIG as cf


run_conversion(
    mode="single",
    input_pattern= cf.input_pattern,
    input_root= cf.input_root,
    output_root= cf.output_root,
    infer_missing=True,
    n_workers=1,
    temp_root = cf.temp_root,
    verbose= cf.verbose,
)

#should be independent