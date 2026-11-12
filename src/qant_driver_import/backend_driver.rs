use crate::qant_driver_import::{
    C_STRING_DEFAULT_LENGTH, PerformanceCounterInfo, QantDataRole, QantDeviceInfo, QantDriverError,
    QantDriverHandle,
};
use half::bf16;
use std::ffi::c_char;

#[link(name = "qant_native_computing_driver")]
unsafe extern "C" {
    pub fn qant_driver_init_npu(id: u32) -> QantDriverError;
}

#[link(name = "qant_native_computing_driver")]
unsafe extern "C" {
    pub fn qant_driver_release_npu(id: u32) -> QantDriverError;
}

#[link(name = "qant_native_computing_driver")]
unsafe extern "C" {
    pub fn qant_driver_setup_logging(folder_name: *const c_char, loglevel: i32) -> QantDriverError;
}

#[link(name = "qant_native_computing_driver")]
unsafe extern "C" {
    pub fn qant_driver_mul_npu(
        npu_id: u32,
        us: *const bf16,
        vs: *const bf16,
        output: *mut bf16,
        len: usize,
    ) -> QantDriverError;
}

#[link(name = "qant_native_computing_driver")]
unsafe extern "C" {
    pub fn qant_driver_mul_matrix_vec_batched(
        npu_id: u32,
        matrix: *const bf16,
        vector_batch: *const bf16,
        output: *mut bf16,
        n_rows: usize,
        n_cols: usize,
        n_vecs: usize,
    ) -> QantDriverError;
}

#[link(name = "qant_native_computing_driver")]
unsafe extern "C" {
    pub fn qant_driver_scaled_periodic_nl_npu(
        npu_id: u32,
        us: *const bf16,
        vs: *const bf16,
        output: *mut bf16,
        len: usize,
    ) -> QantDriverError;
}

#[link(name = "qant_native_computing_driver")]
unsafe extern "C" {
    pub fn qant_driver_allocate(
        npu_id: u32,
        rows: usize,
        cols: usize,
        data_role: QantDataRole,
        out_handle: *mut QantDriverHandle,
    ) -> QantDriverError;
}

#[link(name = "qant_native_computing_driver")]
unsafe extern "C" {
    pub fn qant_driver_free(handle: QantDriverHandle) -> QantDriverError;
}

#[link(name = "qant_native_computing_driver")]
unsafe extern "C" {
    pub fn qant_driver_upload(handle: QantDriverHandle, data_c: *const bf16) -> QantDriverError;
}

#[link(name = "qant_native_computing_driver")]
unsafe extern "C" {
    pub fn qant_driver_download(handle: QantDriverHandle, output_c: *mut bf16) -> QantDriverError;
}

#[link(name = "qant_native_computing_driver")]
unsafe extern "C" {
    pub fn qant_driver_mul_matrix_vec_batched_mem(
        handle_matrix: QantDriverHandle,
        handle_vector: QantDriverHandle,
        handle_result: QantDriverHandle,
    ) -> QantDriverError;
}

#[link(name = "qant_native_computing_driver")]
unsafe extern "C" {
    pub fn qant_driver_get_version(npu_id: u32, output: *mut c_char, len: usize)
    -> QantDriverError;
}

#[link(name = "qant_native_computing_driver")]
unsafe extern "C" {
    pub fn qant_driver_get_device_info(npu_id: u32) -> QantDeviceInfo;
}

#[link(name = "qant_native_computing_driver")]
unsafe extern "C" {
    pub fn qant_driver_get_available_npus(
        idxs_out: *mut u32,
        serials_out: *mut [c_char; C_STRING_DEFAULT_LENGTH],
        capacity: usize,
        out_count: *mut usize,
    ) -> QantDriverError;
}

#[link(name = "qant_native_computing_driver")]
unsafe extern "C" {
    pub fn qant_driver_reset_perf_counter(npu_id: u32) -> QantDriverError;
}

#[link(name = "qant_native_computing_driver")]
unsafe extern "C" {
    pub fn qant_driver_get_perf_counter(
        npu_id: u32,
        out: *mut PerformanceCounterInfo,
    ) -> QantDriverError;
}
