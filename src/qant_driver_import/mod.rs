use crate::{errors::ToolkitError, toolkit_error, utils::log_level_to_int};
use half::bf16;
use libc::{self, c_char, c_int};
use std::{collections::HashMap, ffi::CString};

#[cfg_attr(feature = "cpu-backend", path = "backend_cpu.rs")]
#[cfg_attr(not(feature = "cpu-backend"), path = "backend_driver.rs")]
mod backend;

pub const C_STRING_DEFAULT_LENGTH: usize = 1024;

// Mirror the structs from qant_native_computing_driver.h
// If you encounter errors with passing structs, note that the compiler
// will not notify you if there are missing fields, it will just take whatever
// data comes after the valid fields (if owned by the program).
// So double check all fields to be sure.

#[derive(Debug)]
#[repr(C)]
pub struct QantDriverError {
    pub code: c_int,
    pub message: [c_char; C_STRING_DEFAULT_LENGTH],
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct SensorInfo {
    pub temp_pd1: f64,
    pub temp_pd2: f64,
    pub temp_pd3: f64,
    pub temp_dac_1: f64,
    pub temp_dac_2: f64,
    pub voltage_1v8: f64,
    pub voltage_3v3: f64,
    pub voltage_6v0_1: f64,
    pub voltage_6v0_2: f64,
    pub voltage_6v2: f64,
    pub voltage_6v5: f64,
    pub voltage_8v0_1: f64,
    pub voltage_8v0_2: f64,
    pub voltage_12v: f64,
    pub voltage_12v5: f64,
    pub voltage_n12v5: f64,
    pub current_12v: f64,
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct QantVersionInfo {
    pub fw_version: [c_char; C_STRING_DEFAULT_LENGTH],
    pub zephyr_version: [c_char; C_STRING_DEFAULT_LENGTH],
    pub board_serial_no: [c_char; C_STRING_DEFAULT_LENGTH],
}

#[repr(C)]
#[derive(Debug)]
pub struct QantDeviceInfo {
    pub sensor_info: SensorInfo,
    pub version_info: QantVersionInfo,
    pub error: QantDriverError,
}

#[derive(Debug, Clone)]
pub struct VersionInfo {
    pub fw_version: String,
    pub zephyr_version: String,
    pub board_serial_no: String,
}

impl VersionInfo {
    pub fn from_c_struct(info_c: QantVersionInfo) -> Self {
        Self {
            fw_version: c_char_to_rust_str(&info_c.fw_version),
            zephyr_version: c_char_to_rust_str(&info_c.zephyr_version),
            board_serial_no: c_char_to_rust_str(&info_c.board_serial_no),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub sensor: SensorInfo,
    pub version_info: VersionInfo,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct PerformanceCounterInfo {
    pub timebased_counter: f64, // time since last reset (in s)
    pub stalling_counter: f64,  // time spent in stalling mode (idle mode in s)
}

fn c_char_to_rust_str(c_chars: &[c_char; C_STRING_DEFAULT_LENGTH]) -> String {
    let c_str = unsafe { std::ffi::CStr::from_ptr(c_chars.as_ptr()) };
    c_str.to_string_lossy().to_string()
}

fn driver_err_to_result(err: QantDriverError) -> Result<(), ToolkitError> {
    if err.code == 0 {
        return Ok(());
    }
    Err(ToolkitError::from_str(&c_char_to_rust_str(&err.message)))
}

pub fn driver_setup_logging(
    folder_name: &str,
    loglevel: tracing::Level,
) -> Result<(), ToolkitError> {
    let folder_cstr = match CString::new(folder_name) {
        Ok(cstr) => cstr,
        Err(_) => {
            return Err(ToolkitError::from_str(
                "cannot convert folder_name to CString",
            ));
        }
    };
    let loglevel_int = log_level_to_int(loglevel);

    let driver_err =
        unsafe { backend::qant_driver_setup_logging(folder_cstr.as_ptr(), loglevel_int) };
    driver_err_to_result(driver_err)
}

/// Takes a c function and returns a rust function.
pub fn wrap_c_function_of_two_arrays(
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
        driver_err_to_result(driver_err)?;

        unsafe {
            res.set_len(len);
        }
        Ok(res)
    }
}

pub fn driver_init_npu(npu_id: u32) -> Result<(), ToolkitError> {
    let driver_err = unsafe { backend::qant_driver_init_npu(npu_id) };
    driver_err_to_result(driver_err)
}

pub fn driver_release_npu(npu_id: u32) -> Result<(), ToolkitError> {
    let driver_err = unsafe { backend::qant_driver_release_npu(npu_id) };
    driver_err_to_result(driver_err)
}

pub fn driver_init_npu_error_code(npu_id: u32) -> c_int {
    unsafe { backend::qant_driver_init_npu(npu_id).code }
}

pub fn driver_release_npu_error_code(npu_id: u32) -> c_int {
    unsafe { backend::qant_driver_release_npu(npu_id).code }
}

pub fn driver_mul(npu_id: u32, us: &[bf16], vs: &[bf16]) -> Result<Vec<bf16>, ToolkitError> {
    let res = wrap_c_function_of_two_arrays(backend::qant_driver_mul_npu)(npu_id, us, vs)?;
    Ok(res)
}

pub fn driver_matrix_vector_mul_batched(
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

    driver_err_to_result(driver_err)?;
    Ok(res)
}

pub fn driver_scaled_periodic_nl(
    npu_id: u32,
    us: &[bf16],
    vs: &[bf16],
) -> Result<Vec<bf16>, ToolkitError> {
    let res =
        wrap_c_function_of_two_arrays(backend::qant_driver_scaled_periodic_nl_npu)(npu_id, us, vs)?;
    Ok(res)
}

pub fn driver_get_version(npu_id: u32) -> Result<String, ToolkitError> {
    let max_len = 512;
    let mut res = String::with_capacity(max_len);
    let driver_err = unsafe {
        backend::qant_driver_get_version(npu_id, res.as_mut_ptr() as *mut c_char, max_len)
    };
    driver_err_to_result(driver_err)?;
    let c_str = unsafe { std::ffi::CStr::from_ptr(res.as_ptr() as *const c_char) };
    let rust_str = c_str.to_string_lossy();

    Ok(rust_str.into())
}

/// to pipe the function directly to the C/C++ interface
pub fn driver_get_version_ptr_based(
    npu_id: u32,
    output: *mut c_char,
    len: usize,
) -> QantDriverError {
    unsafe { backend::qant_driver_get_version(npu_id, output, len) }
}

pub fn driver_get_device_info(npu_id: u32) -> Result<DeviceInfo, ToolkitError> {
    let device_info = unsafe { backend::qant_driver_get_device_info(npu_id) };
    driver_err_to_result(device_info.error)?;
    Ok(DeviceInfo {
        sensor: device_info.sensor_info,
        version_info: VersionInfo::from_c_struct(device_info.version_info),
    })
}

/// Returns the sensor info
/// just used for the direct C/C++ interface, all other functions should
/// use the driver_get_device_info function
pub fn driver_get_sensor_info(npu_id: u32) -> SensorInfo {
    let device_info = unsafe { backend::qant_driver_get_device_info(npu_id) };
    device_info.sensor_info
}

/// Returns the version info
/// just used for the direct C/C++ interface, all other functions should
/// use the driver_get_device_info function
pub fn driver_get_version_info(npu_id: u32) -> QantVersionInfo {
    let device_info = unsafe { backend::qant_driver_get_device_info(npu_id) };
    device_info.version_info
}

pub fn driver_get_available_npus() -> Result<HashMap<u32, String>, ToolkitError> {
    const MAX_N_NPUS: usize = 10;
    let mut idx_arr = [0_u32; MAX_N_NPUS];
    let mut serial_arr = [[0; C_STRING_DEFAULT_LENGTH]; MAX_N_NPUS];
    let mut n_npus: usize = 0;
    let driver_err = unsafe {
        backend::qant_driver_get_available_npus(
            idx_arr.as_mut_ptr(),
            serial_arr.as_mut_ptr(),
            MAX_N_NPUS,
            &mut n_npus as *mut usize,
        )
    };
    driver_err_to_result(driver_err)?;
    let mut ret = HashMap::new();
    for i in 0..n_npus {
        ret.insert(idx_arr[i], c_char_to_rust_str(&serial_arr[i]));
    }
    Ok(ret)
}

/// to pipe the function directly to the C/C++ interface
pub fn driver_get_available_npus_ptr_based(
    idxs_out: *mut u32,
    serials_out: *mut [c_char; C_STRING_DEFAULT_LENGTH],
    capacity: usize,
    out_count: *mut usize,
) -> QantDriverError {
    unsafe { backend::qant_driver_get_available_npus(idxs_out, serials_out, capacity, out_count) }
}

pub fn driver_reset_perf_counter(npu_id: u32) -> Result<(), ToolkitError> {
    let driver_err = unsafe { backend::qant_driver_reset_perf_counter(npu_id) };
    driver_err_to_result(driver_err)?;
    Ok(())
}

pub fn driver_get_perf_counter(npu_id: u32) -> Result<PerformanceCounterInfo, ToolkitError> {
    let mut counter: PerformanceCounterInfo = PerformanceCounterInfo {
        timebased_counter: 0.0,
        stalling_counter: 0.0,
    };
    let err = unsafe {
        backend::qant_driver_get_perf_counter(npu_id, &mut counter as *mut PerformanceCounterInfo)
    };
    driver_err_to_result(err)?;
    Ok(counter)
}
