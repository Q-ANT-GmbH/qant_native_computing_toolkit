use std::ffi::{CStr, CString, c_char};

use half::bf16;

use crate::{
    errors::ToolkitError,
    qant_driver_import::dto::{
        DeviceInfo, PerformanceCounterInfo, QantVersionInfo, SensorInfo, VersionInfo,
    },
    qant_driver_import::{QantDeviceCapability, utils as driver_utils},
    toolkit_error, utils,
};

#[cfg_attr(feature = "cpu-backend", path = "backend_cpu.rs")]
#[cfg_attr(not(feature = "cpu-backend"), path = "backend_driver.rs")]
mod backend;

pub fn init_npu(npu_id: u32) -> Result<(), ToolkitError> {
    let driver_err = unsafe { backend::qant_gen2p_driver_init_npu(npu_id) };
    driver_utils::driver_err_to_result(driver_err)
}

pub fn release_npu(npu_id: u32) -> Result<(), ToolkitError> {
    let driver_err = unsafe { backend::qant_gen2p_driver_release_npu(npu_id) };
    driver_utils::driver_err_to_result(driver_err)
}

pub fn dft_batched(npu_id: u32, data: &[bf16], dft_size: usize) -> Result<Vec<bf16>, ToolkitError> {
    let len = data.len();

    if dft_size == 0 {
        return Err(toolkit_error!("dft_size must be > 0"));
    }

    if !len.is_multiple_of(2 * dft_size) {
        return Err(toolkit_error!(
            "input length {len} is not a multiple of 2 * dft_size ({dft_size})"
        ));
    }

    let n_dfts = len / (2 * dft_size);

    let mut res = Vec::<bf16>::with_capacity(len);
    let driver_err = unsafe {
        backend::qant_gen2p_driver_dft_batched(
            npu_id,
            data.as_ptr(),
            res.as_mut_ptr(),
            dft_size,
            n_dfts,
        )
    };

    driver_utils::driver_err_to_result(driver_err)?;
    unsafe {
        res.set_len(len);
    }

    Ok(res)
}

pub fn setup_logging(folder_name: &str, loglevel: tracing::Level) -> Result<(), ToolkitError> {
    let folder_cstr = match CString::new(folder_name) {
        Ok(cstr) => cstr,
        Err(_) => {
            return Err(ToolkitError::from_str(
                "cannot convert folder_name to CString",
            ));
        }
    };
    let loglevel_int = utils::log_level_to_int(loglevel);

    let driver_err =
        unsafe { backend::qant_gen2p_driver_setup_logging(folder_cstr.as_ptr(), loglevel_int) };
    driver_utils::driver_err_to_result(driver_err)
}

pub fn get_available_npus() -> Result<Vec<(u32, String)>, ToolkitError> {
    driver_utils::collect_available_npus(|idxs_out, serials_out, capacity, out_count| unsafe {
        backend::qant_gen2p_driver_get_available_npus(idxs_out, serials_out, capacity, out_count)
    })
}

pub fn get_version(npu_id: u32) -> Result<String, ToolkitError> {
    let mut buf = [0 as c_char; driver_utils::VERSION_BUFFER_LEN];
    get_version_ptr_based(npu_id, buf.as_mut_ptr(), buf.len())?;
    // SAFETY: the driver NUL-terminates the output within `buf.len()`.
    let c_str = unsafe { CStr::from_ptr(buf.as_ptr()) };
    Ok(c_str.to_string_lossy().into_owned())
}

/// Writes the version string directly into a caller-provided buffer (C/C++ interface).
pub fn get_version_ptr_based(
    npu_id: u32,
    output: *mut c_char,
    len: usize,
) -> Result<(), ToolkitError> {
    let driver_err = unsafe { backend::qant_gen2p_driver_get_version(npu_id, output, len) };
    driver_utils::driver_err_to_result(driver_err)
}

pub fn get_device_info(npu_id: u32) -> Result<DeviceInfo, ToolkitError> {
    let device_info = unsafe { backend::qant_gen2p_driver_get_device_info(npu_id) };
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
    let device_info = unsafe { backend::qant_gen2p_driver_get_device_info(npu_id) };
    driver_utils::driver_err_to_result(device_info.error)?;
    Ok(device_info.sensor_info)
}

/// Returns the version info as C struct
/// just used for the direct C/C++ interface, all other functions should
/// use the get_device_info function
pub fn get_version_info(npu_id: u32) -> Result<QantVersionInfo, ToolkitError> {
    let device_info = unsafe { backend::qant_gen2p_driver_get_device_info(npu_id) };
    driver_utils::driver_err_to_result(device_info.error)?;
    Ok(device_info.version_info)
}

pub fn reset_perf_counter(npu_id: u32) -> Result<(), ToolkitError> {
    let driver_err = unsafe { backend::qant_gen2p_driver_reset_perf_counter(npu_id) };
    driver_utils::driver_err_to_result(driver_err)
}

pub fn get_perf_counter(npu_id: u32) -> Result<PerformanceCounterInfo, ToolkitError> {
    let mut counter = PerformanceCounterInfo::default();
    let driver_err = unsafe { backend::qant_gen2p_driver_get_perf_counter(npu_id, &mut counter) };
    driver_utils::driver_err_to_result(driver_err)?;
    Ok(counter)
}

#[allow(unused_unsafe)] // the cpu-backend stub is a safe fn
pub fn get_capabilities(npu_id: u32) -> Result<Vec<QantDeviceCapability>, ToolkitError> {
    unsafe { backend::qant_gen2p_driver_get_device_capabilities(npu_id) }.into_result()
}
