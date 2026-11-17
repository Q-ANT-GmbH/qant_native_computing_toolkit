use crate::{
    errors::ToolkitError,
    qant_driver_import::dto::{
        DeviceInfo, PerformanceCounterInfo, QantDataRole, QantDriverError, QantDriverHandle,
        QantVersionInfo, SensorInfo, VersionInfo,
    },
    qant_driver_import::{QantDeviceCapability, utils as driver_utils},
    toolkit_error, utils,
};
use half::bf16;
use std::ffi;

#[cfg_attr(feature = "cpu-backend", path = "backend_cpu.rs")]
#[cfg_attr(not(feature = "cpu-backend"), path = "backend_driver.rs")]
mod backend;

pub fn setup_logging(folder_name: &str, loglevel: tracing::Level) -> Result<(), ToolkitError> {
    let folder_cstr = match ffi::CString::new(folder_name) {
        Ok(cstr) => cstr,
        Err(_) => {
            return Err(ToolkitError::from_str(
                "cannot convert folder_name to CString",
            ));
        }
    };
    let loglevel_int = utils::log_level_to_int(loglevel);

    let driver_err =
        unsafe { backend::qant_driver_setup_logging(folder_cstr.as_ptr(), loglevel_int) };
    driver_utils::driver_err_to_result(driver_err)
}

/// Takes a c function and returns a rust function.
fn wrap_c_function_of_two_arrays(
    c_fn: unsafe extern "C" fn(
        id: u32,
        *const bf16,
        *const bf16,
        *mut bf16,
        usize,
    ) -> QantDriverError,
) -> impl Fn(u32, &[bf16], &[bf16]) -> Result<Vec<bf16>, ToolkitError> {
    move |id: u32, us: &[bf16], vs: &[bf16]| -> Result<Vec<bf16>, ToolkitError> {
        if us.len() != vs.len() {
            return Err(toolkit_error!(
                "input vectors must have same length, got {0} and {1}",
                us.len(),
                vs.len()
            ));
        }

        let len = us.len();
        let mut res = Vec::<bf16>::with_capacity(len);

        let driver_err = unsafe { c_fn(id, us.as_ptr(), vs.as_ptr(), res.as_mut_ptr(), len) };
        driver_utils::driver_err_to_result(driver_err)?;

        unsafe {
            res.set_len(len);
        }
        Ok(res)
    }
}

pub fn init_npu(npu_id: u32) -> Result<(), ToolkitError> {
    let driver_err = unsafe { backend::qant_driver_init_npu(npu_id) };
    driver_utils::driver_err_to_result(driver_err)
}

pub fn release_npu(npu_id: u32) -> Result<(), ToolkitError> {
    let driver_err = unsafe { backend::qant_driver_release_npu(npu_id) };
    driver_utils::driver_err_to_result(driver_err)
}

pub fn mul(npu_id: u32, us: &[bf16], vs: &[bf16]) -> Result<Vec<bf16>, ToolkitError> {
    let res = wrap_c_function_of_two_arrays(backend::qant_driver_mul_npu)(npu_id, us, vs)?;
    Ok(res)
}

pub fn matrix_vector_mul_batched(
    npu_id: u32,
    matrix: &[bf16],
    vector_batch: &[bf16],
    vector_len: usize,
) -> Result<Vec<bf16>, ToolkitError> {
    if !matrix.len().is_multiple_of(vector_len) || !vector_batch.len().is_multiple_of(vector_len) {
        return Err(toolkit_error!(
            "flattened matrix and vectors are not shape compatible, got matrix.len()={0}, vector_len={vector_len} and vector_batch.len()={1}",
            matrix.len(),
            vector_batch.len()
        ));
    }
    let n_cols = vector_len;
    let n_rows = matrix.len() / n_cols;
    let n_vecs = vector_batch.len() / vector_len;
    let mut res = Vec::<bf16>::with_capacity(n_rows * n_vecs);

    let driver_err = unsafe {
        backend::qant_driver_mul_matrix_vec_batched(
            npu_id,
            matrix.as_ptr(),
            vector_batch.as_ptr(),
            res.as_mut_ptr(),
            n_rows,
            n_cols,
            n_vecs,
        )
    };
    unsafe {
        res.set_len(n_rows * n_vecs);
    }

    driver_utils::driver_err_to_result(driver_err)?;
    Ok(res)
}

pub fn scaled_periodic_nl(
    npu_id: u32,
    us: &[bf16],
    vs: &[bf16],
) -> Result<Vec<bf16>, ToolkitError> {
    let res =
        wrap_c_function_of_two_arrays(backend::qant_driver_scaled_periodic_nl_npu)(npu_id, us, vs)?;
    Ok(res)
}

pub fn get_version(npu_id: u32) -> Result<String, ToolkitError> {
    let max_len = driver_utils::VERSION_BUFFER_LEN;
    let mut res = String::with_capacity(max_len);
    let driver_err = unsafe {
        backend::qant_driver_get_version(npu_id, res.as_mut_ptr() as *mut ffi::c_char, max_len)
    };
    driver_utils::driver_err_to_result(driver_err)?;
    let c_str = unsafe { std::ffi::CStr::from_ptr(res.as_ptr() as *const ffi::c_char) };
    let rust_str = c_str.to_string_lossy();

    Ok(rust_str.into())
}

/// Writes the version string directly into a caller-provided buffer (C/C++ interface).
pub fn get_version_ptr_based(
    npu_id: u32,
    output: *mut ffi::c_char,
    len: usize,
) -> Result<(), ToolkitError> {
    let driver_err = unsafe { backend::qant_driver_get_version(npu_id, output, len) };
    driver_utils::driver_err_to_result(driver_err)
}

pub fn get_device_info(npu_id: u32) -> Result<DeviceInfo, ToolkitError> {
    let device_info = unsafe { backend::qant_driver_get_device_info(npu_id) };
    driver_utils::driver_err_to_result(device_info.error)?;
    Ok(DeviceInfo {
        sensor: device_info.sensor_info,
        version_info: VersionInfo::from_c_struct(device_info.version_info),
    })
}

/// Returns the sensor info
/// just used for the direct C/C++ interface, all other functions should
/// use the get_device_info function
pub fn get_sensor_info(npu_id: u32) -> Result<SensorInfo, ToolkitError> {
    let device_info = unsafe { backend::qant_driver_get_device_info(npu_id) };
    driver_utils::driver_err_to_result(device_info.error)?;
    Ok(device_info.sensor_info)
}

/// Returns the version info as C struct
/// just used for the direct C/C++ interface, all other functions should
/// use the get_device_info function
pub fn get_version_info(npu_id: u32) -> Result<QantVersionInfo, ToolkitError> {
    let device_info = unsafe { backend::qant_driver_get_device_info(npu_id) };
    driver_utils::driver_err_to_result(device_info.error)?;
    Ok(device_info.version_info)
}

pub fn get_available_npus() -> Result<Vec<(u32, String)>, ToolkitError> {
    driver_utils::collect_available_npus(|idxs_out, serials_out, capacity, out_count| unsafe {
        backend::qant_driver_get_available_npus(idxs_out, serials_out, capacity, out_count)
    })
}

pub fn reset_perf_counter(npu_id: u32) -> Result<(), ToolkitError> {
    let driver_err = unsafe { backend::qant_driver_reset_perf_counter(npu_id) };
    driver_utils::driver_err_to_result(driver_err)?;
    Ok(())
}

pub fn get_perf_counter(npu_id: u32) -> Result<PerformanceCounterInfo, ToolkitError> {
    let mut counter = PerformanceCounterInfo::default();
    let err = unsafe {
        backend::qant_driver_get_perf_counter(npu_id, &mut counter as *mut PerformanceCounterInfo)
    };
    driver_utils::driver_err_to_result(err)?;
    Ok(counter)
}

// FIXME: refactor unsafe handling
#[allow(unused_unsafe)]
pub fn get_capabilities(npu_id: u32) -> Result<Vec<QantDeviceCapability>, ToolkitError> {
    unsafe { backend::qant_driver_get_device_capabilities(npu_id) }.into_result()
}

pub fn qant_driver_allocate(
    npu_id: u32,
    rows: usize,
    cols: usize,
    data_role: QantDataRole,
    out_handle: *mut QantDriverHandle,
) -> QantDriverError {
    unsafe { backend::qant_driver_allocate(npu_id, rows, cols, data_role, out_handle) }
}

pub fn qant_driver_free(handle: QantDriverHandle) -> QantDriverError {
    unsafe { backend::qant_driver_free(handle) }
}

pub fn qant_driver_upload(handle: QantDriverHandle, data_c: *const bf16) -> QantDriverError {
    unsafe { backend::qant_driver_upload(handle, data_c) }
}

pub fn qant_driver_download(handle: QantDriverHandle, output_c: *mut bf16) -> QantDriverError {
    unsafe { backend::qant_driver_download(handle, output_c) }
}

pub fn qant_driver_mul_matrix_vec_batched_mem(
    handle_matrix: QantDriverHandle,
    handle_vector: QantDriverHandle,
    handle_result: QantDriverHandle,
) -> QantDriverError {
    unsafe {
        backend::qant_driver_mul_matrix_vec_batched_mem(handle_matrix, handle_vector, handle_result)
    }
}
