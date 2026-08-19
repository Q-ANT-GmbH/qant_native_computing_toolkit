use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bindings = bindgen::Builder::default()
        .header("external/dlpack/include/dlpack/dlpack.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate_comments(false) // Skip the comments as these will break the rust
        // doc tests
        .generate()
        .map_err(|e| {
            format!(
                "Failed to generate bindings from dlpack.h: {e} Are the submodules checked out?"
            )
        })?;

    let out_path = PathBuf::from(env::var("OUT_DIR")?);
    bindings.write_to_file(out_path.join("dlpack_bindings.rs"))?;
    Ok(())
}
