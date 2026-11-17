use std::ffi::c_char;

use half::bf16;

use crate::qant_driver_import::dto::{
    C_STRING_DEFAULT_LENGTH, PerformanceCounterInfo, QantDeviceCapabilities, QantDeviceInfo,
    QantDriverError,
};

#[link(name = "qant_native_computing_driver_gen2plus")]
unsafe extern "C" {
    pub fn qant_gen2p_driver_init_npu(id: u32) -> QantDriverError;
}

#[link(name = "qant_native_computing_driver_gen2plus")]
unsafe extern "C" {
    pub fn qant_gen2p_driver_release_npu(id: u32) -> QantDriverError;
}

// TODO: can we have a single const here?
#[link(name = "qant_native_computing_driver_gen2plus")]
unsafe extern "C" {
    pub fn qant_gen2p_driver_get_available_npus(
        idxs_out: *mut u32,
        serials_out: *mut [c_char; C_STRING_DEFAULT_LENGTH],
        capacity: usize,
        out_count: *mut usize,
    ) -> QantDriverError;
}

#[link(name = "qant_native_computing_driver_gen2plus")]
unsafe extern "C" {
    pub fn qant_gen2p_driver_dft_batched(
        npu_id: u32,
        input: *const bf16,
        output: *mut bf16,
        dft_size: usize,
        n_dfts: usize,
    ) -> QantDriverError;
}

#[link(name = "qant_native_computing_driver_gen2plus")]
unsafe extern "C" {
    pub fn qant_gen2p_driver_setup_logging(
        folder_name: *const c_char,
        loglevel: i32,
    ) -> QantDriverError;
}

#[link(name = "qant_native_computing_driver_gen2plus")]
unsafe extern "C" {
    pub fn qant_gen2p_driver_get_version(
        npu_id: u32,
        output: *mut c_char,
        len: usize,
    ) -> QantDriverError;
}

#[link(name = "qant_native_computing_driver_gen2plus")]
unsafe extern "C" {
    pub fn qant_gen2p_driver_get_device_info(npu_id: u32) -> QantDeviceInfo;
}

#[link(name = "qant_native_computing_driver_gen2plus")]
unsafe extern "C" {
    pub fn qant_gen2p_driver_reset_perf_counter(npu_id: u32) -> QantDriverError;
}

#[link(name = "qant_native_computing_driver_gen2plus")]
unsafe extern "C" {
    pub fn qant_gen2p_driver_get_perf_counter(
        npu_id: u32,
        out: *mut PerformanceCounterInfo,
    ) -> QantDriverError;
}

#[link(name = "qant_native_computing_driver_gen2plus")]
unsafe extern "C" {
    pub fn qant_gen2p_driver_get_device_capabilities(npu_id: u32) -> QantDeviceCapabilities;
}
