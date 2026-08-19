use crate::qant_driver_import::{
    C_STRING_DEFAULT_LENGTH, PerformanceCounterInfo, QantDeviceInfo, QantDriverError,
};
use half::bf16;
use libc::c_char;

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
