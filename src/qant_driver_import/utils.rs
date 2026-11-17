use std::ffi;

use crate::{
    errors::ToolkitError,
    qant_driver_import::dto::{C_STRING_DEFAULT_LENGTH, QantDriverError},
};

/// Size of the buffer the version string is written into by `get_version`.
pub const VERSION_BUFFER_LEN: usize = 512;

pub fn c_char_to_rust_str(c_chars: &[ffi::c_char; C_STRING_DEFAULT_LENGTH]) -> String {
    let c_str = unsafe { std::ffi::CStr::from_ptr(c_chars.as_ptr()) };
    c_str.to_string_lossy().to_string()
}

pub fn driver_err_to_result(err: QantDriverError) -> Result<(), ToolkitError> {
    if err.code == 0 {
        return Ok(());
    }
    Err(ToolkitError::from_str(&c_char_to_rust_str(&err.message)))
}

/// Calls a driver's `get_available_npus` and collects the (NPU id, serial number) pairs in driver order.
pub fn collect_available_npus(
    get_available_npus: impl FnOnce(
        *mut u32,
        *mut [ffi::c_char; C_STRING_DEFAULT_LENGTH],
        usize,
        *mut usize,
    ) -> QantDriverError,
) -> Result<Vec<(u32, String)>, ToolkitError> {
    const MAX_N_NPUS: usize = 20;
    let mut idx_arr = [0_u32; MAX_N_NPUS];
    let mut serial_arr = [[0; C_STRING_DEFAULT_LENGTH]; MAX_N_NPUS];
    let mut n_npus: usize = 0;
    driver_err_to_result(get_available_npus(
        idx_arr.as_mut_ptr(),
        serial_arr.as_mut_ptr(),
        MAX_N_NPUS,
        &mut n_npus,
    ))?;
    Ok(idx_arr
        .iter()
        .zip(&serial_arr)
        .take(n_npus)
        .map(|(&idx, serial)| (idx, c_char_to_rust_str(serial)))
        .collect())
}
