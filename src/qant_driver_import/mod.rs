use crate::{errors::ToolkitError, toolkit_error};
use half::bf16;
use std::ffi::c_char;

pub mod dto;
pub use dto::{QantDataRole, QantDriverHandle};
mod device_manager;
pub(crate) use device_manager::check_npu_id;
pub use device_manager::{QantDeviceCapability, get_available_npus, get_capabilities};
mod gen2;
mod gen2p;
mod utils;

/// Sentinel `npu_id` for automatic NPU selection: the lowest-id NPU that
/// supports the requested operation. Passing an `npu_id` below this
/// value is invalid and results in an error.
pub const AUTO_NPU_ID: i32 = -1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NPUBackend {
    Gen2,
    Gen2p,
}

impl NPUBackend {
    fn get_available_npus(self) -> Result<Vec<(u32, String)>, ToolkitError> {
        match self {
            NPUBackend::Gen2 => gen2::get_available_npus(),
            NPUBackend::Gen2p => gen2p::get_available_npus(),
        }
    }
    fn get_capabilities(self, npu_id: u32) -> Result<Vec<QantDeviceCapability>, ToolkitError> {
        match self {
            NPUBackend::Gen2 => gen2::get_capabilities(npu_id),
            NPUBackend::Gen2p => gen2p::get_capabilities(npu_id),
        }
    }
    fn init_npu(self, npu_id: u32) -> Result<(), ToolkitError> {
        match self {
            NPUBackend::Gen2 => gen2::init_npu(npu_id),
            NPUBackend::Gen2p => gen2p::init_npu(npu_id),
        }
    }
    fn release_npu(self, npu_id: u32) -> Result<(), ToolkitError> {
        match self {
            NPUBackend::Gen2 => gen2::release_npu(npu_id),
            NPUBackend::Gen2p => gen2p::release_npu(npu_id),
        }
    }

    pub fn mul(self, npu_id: u32, us: &[bf16], vs: &[bf16]) -> Result<Vec<bf16>, ToolkitError> {
        match self {
            NPUBackend::Gen2 => gen2::mul(npu_id, us, vs),
            NPUBackend::Gen2p => Err(toolkit_error!(
                "NPU{npu_id} does not support MULTIPLICATION"
            )),
        }
    }

    pub fn matrix_vector_mul_batched(
        self,
        npu_id: u32,
        matrix: &[bf16],
        vector_batch: &[bf16],
        vector_len: usize,
    ) -> Result<Vec<bf16>, ToolkitError> {
        match self {
            NPUBackend::Gen2 => {
                gen2::matrix_vector_mul_batched(npu_id, matrix, vector_batch, vector_len)
            }
            NPUBackend::Gen2p => Err(toolkit_error!(
                "NPU{npu_id} does not support MULTIPLICATION"
            )),
        }
    }

    pub fn scaled_periodic_nl(
        self,
        npu_id: u32,
        us: &[bf16],
        vs: &[bf16],
    ) -> Result<Vec<bf16>, ToolkitError> {
        match self {
            NPUBackend::Gen2 => gen2::scaled_periodic_nl(npu_id, us, vs),
            NPUBackend::Gen2p => Err(toolkit_error!("NPU{npu_id} does not support NONLINEARITY")),
        }
    }

    pub fn dft_batched(
        self,
        npu_id: u32,
        data: &[bf16],
        dft_size: usize,
    ) -> Result<Vec<bf16>, ToolkitError> {
        match self {
            NPUBackend::Gen2p => gen2p::dft_batched(npu_id, data, dft_size),
            NPUBackend::Gen2 => Err(toolkit_error!("NPU{npu_id} does not support DFT")),
        }
    }

    pub fn reset_perf_counter(self, npu_id: u32) -> Result<(), ToolkitError> {
        match self {
            NPUBackend::Gen2 => gen2::reset_perf_counter(npu_id),
            NPUBackend::Gen2p => gen2p::reset_perf_counter(npu_id),
        }
    }

    pub fn get_perf_counter(
        self,
        npu_id: u32,
    ) -> Result<dto::PerformanceCounterInfo, ToolkitError> {
        match self {
            NPUBackend::Gen2 => gen2::get_perf_counter(npu_id),
            NPUBackend::Gen2p => gen2p::get_perf_counter(npu_id),
        }
    }

    pub fn get_version(self, npu_id: u32) -> Result<String, ToolkitError> {
        match self {
            NPUBackend::Gen2 => gen2::get_version(npu_id),
            NPUBackend::Gen2p => gen2p::get_version(npu_id),
        }
    }

    pub fn get_version_ptr_based(
        self,
        npu_id: u32,
        output: *mut c_char,
        len: usize,
    ) -> Result<(), ToolkitError> {
        match self {
            NPUBackend::Gen2 => gen2::get_version_ptr_based(npu_id, output, len),
            NPUBackend::Gen2p => gen2p::get_version_ptr_based(npu_id, output, len),
        }
    }

    pub fn get_device_info(self, npu_id: u32) -> Result<dto::DeviceInfo, ToolkitError> {
        match self {
            NPUBackend::Gen2 => gen2::get_device_info(npu_id),
            NPUBackend::Gen2p => gen2p::get_device_info(npu_id),
        }
    }

    pub fn get_sensor_info(self, npu_id: u32) -> Result<dto::SensorInfo, ToolkitError> {
        match self {
            NPUBackend::Gen2 => gen2::get_sensor_info(npu_id),
            NPUBackend::Gen2p => gen2p::get_sensor_info(npu_id),
        }
    }

    pub fn get_version_info(self, npu_id: u32) -> Result<dto::QantVersionInfo, ToolkitError> {
        match self {
            NPUBackend::Gen2 => gen2::get_version_info(npu_id),
            NPUBackend::Gen2p => gen2p::get_version_info(npu_id),
        }
    }
    pub fn allocate(
        self,
        npu_id: u32,
        rows: usize,
        cols: usize,
        data_role: QantDataRole,
    ) -> Result<QantDriverHandle, ToolkitError> {
        match self {
            NPUBackend::Gen2 => {
                let mut out_handle = QantDriverHandle { npu_id, key: 0 };
                utils::driver_err_to_result(gen2::qant_driver_allocate(
                    npu_id,
                    rows,
                    cols,
                    data_role,
                    &mut out_handle,
                ))?;
                Ok(out_handle)
            }
            NPUBackend::Gen2p => Err(toolkit_error!("NPU{npu_id} does not support MEM_HANDLING")),
        }
    }

    pub fn free(self, handle: QantDriverHandle) -> Result<(), ToolkitError> {
        let npu_id = handle.npu_id;
        match self {
            NPUBackend::Gen2 => {
                let driver_err = gen2::qant_driver_free(handle);
                utils::driver_err_to_result(driver_err)
            }
            NPUBackend::Gen2p => Err(toolkit_error!("NPU{npu_id} does not support MEM_HANDLING")),
        }
    }

    pub fn upload(self, handle: QantDriverHandle, data: &[bf16]) -> Result<(), ToolkitError> {
        let npu_id = handle.npu_id;
        match self {
            NPUBackend::Gen2 => {
                let driver_err = gen2::qant_driver_upload(handle, data.as_ptr());
                utils::driver_err_to_result(driver_err)
            }
            NPUBackend::Gen2p => Err(toolkit_error!("NPU{npu_id} does not support MEM_HANDLING")),
        }
    }

    pub fn download(self, handle: QantDriverHandle, len: usize) -> Result<Vec<bf16>, ToolkitError> {
        let npu_id = handle.npu_id;
        match self {
            NPUBackend::Gen2 => {
                let mut data = Vec::<bf16>::with_capacity(len);
                let driver_err = gen2::qant_driver_download(handle, data.as_mut_ptr());
                unsafe {
                    data.set_len(len);
                }
                utils::driver_err_to_result(driver_err)?;
                Ok(data)
            }
            NPUBackend::Gen2p => Err(toolkit_error!("NPU{npu_id} does not support MEM_HANDLING")),
        }
    }

    pub fn mul_matrix_vec_batched_mem(
        self,
        matrix_handle: QantDriverHandle,
        vector_handle: QantDriverHandle,
        output_handle: QantDriverHandle,
    ) -> Result<(), ToolkitError> {
        let npu_id = matrix_handle.npu_id;
        match self {
            NPUBackend::Gen2 => {
                let driver_err = gen2::qant_driver_mul_matrix_vec_batched_mem(
                    matrix_handle,
                    vector_handle,
                    output_handle,
                );
                utils::driver_err_to_result(driver_err)
            }
            NPUBackend::Gen2p => Err(toolkit_error!("NPU{npu_id} does not support MEM_HANDLING")),
        }
    }
}

pub fn setup_logging(folder_name: &str, loglevel: tracing::Level) -> Result<(), ToolkitError> {
    gen2::setup_logging(folder_name, loglevel)?;
    gen2p::setup_logging(folder_name, loglevel)?;
    Ok(())
}

/// `npu_id`: [`AUTO_NPU_ID`] (-1) selects the first NPU (any backend);
/// a non-negative id targets that NPU directly. Errors if `npu_id < -1`.
pub fn init_npu(npu_id: i32) -> Result<(), ToolkitError> {
    let npu = device_manager::resolve_any(npu_id)?;
    npu.backend.init_npu(npu.id)
}

/// See [`init_npu`] for `npu_id` semantics.
pub fn release_npu(npu_id: i32) -> Result<(), ToolkitError> {
    let npu = device_manager::resolve_any(npu_id)?;
    npu.backend.release_npu(npu.id)
}

/// `npu_id`: [`AUTO_NPU_ID`] (-1) selects the first NPU that supports
/// multiplication; a non-negative id targets that NPU directly. Errors if
/// `npu_id < -1`.
pub fn mul(npu_id: i32, us: &[bf16], vs: &[bf16]) -> Result<Vec<bf16>, ToolkitError> {
    let npu =
        device_manager::resolve_npu(npu_id, |npu| npu.supports(&[QantDeviceCapability::Mul]))?;
    npu.backend.mul(npu.id, us, vs)
}

/// See [`mul`] for `npu_id` semantics.
pub fn matrix_vector_mul_batched(
    npu_id: i32,
    matrix: &[bf16],
    vector_batch: &[bf16],
    vector_len: usize,
) -> Result<Vec<bf16>, ToolkitError> {
    let npu =
        device_manager::resolve_npu(npu_id, |npu| npu.supports(&[QantDeviceCapability::Mul]))?;
    npu.backend
        .matrix_vector_mul_batched(npu.id, matrix, vector_batch, vector_len)
}

/// `npu_id`: [`AUTO_NPU_ID`] (-1) selects the first NPU that supports
/// the nonlinearity operation; a non-negative id targets that NPU directly.
/// Errors if `npu_id < -1`.
pub fn scaled_periodic_nl(
    npu_id: i32,
    us: &[bf16],
    vs: &[bf16],
) -> Result<Vec<bf16>, ToolkitError> {
    let npu =
        device_manager::resolve_npu(npu_id, |npu| npu.supports(&[QantDeviceCapability::NonLin]))?;
    npu.backend.scaled_periodic_nl(npu.id, us, vs)
}

/// `npu_id`: [`AUTO_NPU_ID`] (-1) selects the first NPU that supports
/// memory handling; a non-negative id targets that NPU directly.
pub fn allocate(
    npu_id: i32,
    rows: usize,
    cols: usize,
    data_role: QantDataRole,
) -> Result<QantDriverHandle, ToolkitError> {
    let npu = device_manager::resolve_npu(npu_id, |npu| {
        npu.supports(&[QantDeviceCapability::MemHandling])
    })?;
    npu.backend.allocate(npu.id, rows, cols, data_role)
}

pub fn free(handle: QantDriverHandle) -> Result<(), ToolkitError> {
    let npu = device_manager::npu_by_id(handle.npu_id)?;
    npu.backend.free(handle)
}

pub fn upload(handle: QantDriverHandle, data: &[bf16]) -> Result<(), ToolkitError> {
    let npu = device_manager::npu_by_id(handle.npu_id)?;
    npu.backend.upload(handle, data)
}

pub fn download(handle: QantDriverHandle, len: usize) -> Result<Vec<bf16>, ToolkitError> {
    let npu = device_manager::npu_by_id(handle.npu_id)?;
    npu.backend.download(handle, len)
}

pub fn mul_matrix_vec_batched_mem(
    matrix_handle: QantDriverHandle,
    vector_handle: QantDriverHandle,
    output_handle: QantDriverHandle,
) -> Result<(), ToolkitError> {
    let npu = device_manager::npu_by_id(matrix_handle.npu_id)?;
    npu.backend
        .mul_matrix_vec_batched_mem(matrix_handle, vector_handle, output_handle)
}

/// `npu_id`: [`AUTO_NPU_ID`] (-1) selects the first NPU that supports
/// DFT; a non-negative id targets that NPU directly. Errors if `npu_id < -1`.
pub fn dft_batched(npu_id: i32, data: &[bf16], dft_size: usize) -> Result<Vec<bf16>, ToolkitError> {
    let npu =
        device_manager::resolve_npu(npu_id, |npu| npu.supports(&[QantDeviceCapability::Dft]))?;
    npu.backend.dft_batched(npu.id, data, dft_size)
}

/// `npu_id`: [`AUTO_NPU_ID`] (-1) selects the first NPU; a non-negative
/// id targets that NPU directly. Errors if `npu_id < -1`.
pub fn get_version(npu_id: i32) -> Result<String, ToolkitError> {
    let npu = device_manager::resolve_any(npu_id)?;
    npu.backend.get_version(npu.id)
}

/// See [`get_version`] for `npu_id` semantics.
pub fn get_version_ptr_based(
    npu_id: i32,
    output: *mut c_char,
    len: usize,
) -> Result<(), ToolkitError> {
    let npu = device_manager::resolve_any(npu_id)?;
    npu.backend.get_version_ptr_based(npu.id, output, len)
}

/// See [`get_version`] for `npu_id` semantics.
pub fn get_device_info(npu_id: i32) -> Result<dto::DeviceInfo, ToolkitError> {
    let npu = device_manager::resolve_any(npu_id)?;
    npu.backend.get_device_info(npu.id)
}

/// Sensor info as C struct, for the C/C++ interface.
/// See [`get_version`] for `npu_id` semantics.
pub fn get_sensor_info(npu_id: i32) -> Result<dto::SensorInfo, ToolkitError> {
    let npu = device_manager::resolve_any(npu_id)?;
    npu.backend.get_sensor_info(npu.id)
}

/// Version info as C struct, for the C/C++ interface.
/// See [`get_version`] for `npu_id` semantics.
pub fn get_version_info(npu_id: i32) -> Result<dto::QantVersionInfo, ToolkitError> {
    let npu = device_manager::resolve_any(npu_id)?;
    npu.backend.get_version_info(npu.id)
}

/// See [`get_version`] for `npu_id` semantics.
pub fn reset_perf_counter(npu_id: i32) -> Result<(), ToolkitError> {
    let npu = device_manager::resolve_any(npu_id)?;
    npu.backend.reset_perf_counter(npu.id)
}

/// See [`get_version`] for `npu_id` semantics.
pub fn get_perf_counter(npu_id: i32) -> Result<dto::PerformanceCounterInfo, ToolkitError> {
    let npu = device_manager::resolve_any(npu_id)?;
    npu.backend.get_perf_counter(npu.id)
}

#[cfg(test)]
mod tests {
    use super::device_manager::DEVICES;
    use super::*;

    #[test]
    fn npu_id_below_default_errors_across_the_public_api() {
        assert!(init_npu(-2).is_err());
        assert!(release_npu(-2).is_err());
        assert!(mul(-2, &[], &[]).is_err());
        assert!(matrix_vector_mul_batched(-2, &[], &[], 0).is_err());
        assert!(scaled_periodic_nl(-2, &[], &[]).is_err());
        assert!(dft_batched(-2, &[], 0).is_err());
        assert!(get_version(-2).is_err());
        assert!(get_device_info(-2).is_err());
        assert!(get_sensor_info(-2).is_err());
        assert!(get_version_info(-2).is_err());
        assert!(reset_perf_counter(-2).is_err());
        assert!(get_perf_counter(-2).is_err());
        assert!(get_capabilities(-2).is_err());
    }

    #[test]
    fn auto_npu_id_resolves_like_the_first_capable_npu() {
        // Whatever NPUs are enumerated in this environment (cpu-backend
        // simulation or real hardware), -1 must behave the same as
        // explicitly targeting the first NPU whose backend supports the
        // requested operation.
        let first_generic = DEVICES.first();
        let first_mul = DEVICES
            .iter()
            .find(|npu| npu.supports(&[QantDeviceCapability::Mul]));
        let first_dft = DEVICES
            .iter()
            .find(|npu| npu.supports(&[QantDeviceCapability::Dft]));

        if first_generic.is_some() {
            init_npu(AUTO_NPU_ID).unwrap();
            release_npu(AUTO_NPU_ID).unwrap();
        }

        match first_mul {
            Some(_) => assert!(mul(AUTO_NPU_ID, &[], &[]).is_ok()),
            None => assert!(mul(AUTO_NPU_ID, &[], &[]).is_err()),
        }

        match first_dft {
            Some(_) => {
                assert!(dft_batched(AUTO_NPU_ID, &[bf16::from_f32(0.0); 4], 2).is_ok())
            }
            None => assert!(dft_batched(AUTO_NPU_ID, &[], 0).is_err()),
        }
    }
}
