// Mirror the structs from qant_native_computing_driver.h
// If you encounter errors with passing structs, note that the compiler
// will not notify you if there are missing fields, it will just take whatever
// data comes after the valid fields (if owned by the program).
// So double check all fields to be sure.

use std::ffi::{c_char, c_int};

use crate::{
    errors::ToolkitError,
    qant_driver_import::{QantDeviceCapability, utils as driver_utils},
};

pub const C_STRING_DEFAULT_LENGTH: usize = 1024;

#[derive(Debug)]
#[repr(C)]
pub struct QantDriverError {
    pub code: c_int,
    pub message: [c_char; C_STRING_DEFAULT_LENGTH],
}

/// Constructors for the simulated drivers of the cpu-backend.
#[cfg(feature = "cpu-backend")]
impl QantDriverError {
    pub fn success() -> Self {
        Self {
            code: 0,
            message: [0; C_STRING_DEFAULT_LENGTH],
        }
    }

    /// Error with code -1 and `msg`, truncated to fit the NUL terminated message.
    pub fn error(msg: &str) -> Self {
        let mut message = [0; C_STRING_DEFAULT_LENGTH];
        let len = msg.len().min(C_STRING_DEFAULT_LENGTH - 1);
        for (dst, &src) in message.iter_mut().zip(&msg.as_bytes()[..len]) {
            *dst = src as c_char;
        }
        Self { code: -1, message }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct PerformanceCounterInfo {
    pub timebased_counter: f64, // time since last reset (in s)
    pub stalling_counter: f64,  // time spent in stalling mode (idle mode in s)
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

/// Fixed sensor readings and empty version strings, for the simulated drivers
/// of the cpu-backend.
#[cfg(feature = "cpu-backend")]
impl QantDeviceInfo {
    pub fn simulated() -> Self {
        Self {
            sensor_info: SensorInfo {
                temp_pd1: 35.0,
                temp_pd2: 35.0,
                temp_pd3: 35.0,
                temp_dac_1: 35.0,
                temp_dac_2: 35.0,
                voltage_1v8: 1.8,
                voltage_3v3: 3.3,
                voltage_6v2: 6.2,
                voltage_6v0_1: 6.0,
                voltage_6v0_2: 6.0,
                voltage_6v5: 6.5,
                voltage_8v0_1: 8.0,
                voltage_8v0_2: 8.0,
                voltage_12v: 12.0,
                voltage_12v5: 12.5,
                voltage_n12v5: -12.5,
                current_12v: 8.0,
            },
            version_info: QantVersionInfo {
                fw_version: [0; C_STRING_DEFAULT_LENGTH],
                zephyr_version: [0; C_STRING_DEFAULT_LENGTH],
                board_serial_no: [0; C_STRING_DEFAULT_LENGTH],
            },
            error: QantDriverError::success(),
        }
    }
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
            fw_version: driver_utils::c_char_to_rust_str(&info_c.fw_version),
            zephyr_version: driver_utils::c_char_to_rust_str(&info_c.zephyr_version),
            board_serial_no: driver_utils::c_char_to_rust_str(&info_c.board_serial_no),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub sensor: SensorInfo,
    pub version_info: VersionInfo,
}

#[repr(C)]
pub struct QantDeviceCapabilities {
    pub mul: bool,
    pub non_lin: bool,
    pub dft: bool,
    pub mem_handling: bool,
    pub error: QantDriverError,
}

impl QantDeviceCapabilities {
    /// The supported operations, or the driver error.
    pub fn into_result(self) -> Result<Vec<QantDeviceCapability>, ToolkitError> {
        driver_utils::driver_err_to_result(self.error)?;
        Ok([
            (self.mul, QantDeviceCapability::Mul),
            (self.non_lin, QantDeviceCapability::NonLin),
            (self.dft, QantDeviceCapability::Dft),
            (self.mem_handling, QantDeviceCapability::MemHandling),
        ]
        .into_iter()
        .filter_map(|(supported, capability)| supported.then_some(capability))
        .collect())
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QantDataRole {
    Weights = 0,
    Features = 1,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QantDriverHandle {
    pub npu_id: u32,
    pub key: u64,
}
