/*
 * This file provides a lightweight simulation backend for the toolkit,
 * designed to operate without any dependency on the driver.
 *
 * While the logic is intentionally minimal and not fully featured,
 * it is sufficient for validating toolkit integrations.
 *
 * This is particularly useful for scenarios such as compiler integration,
 * where the full driver stack is unnecessary.
 */

use crate::qant_driver_import::{
    C_STRING_DEFAULT_LENGTH, PerformanceCounterInfo, QantDeviceInfo, QantDriverError,
    QantVersionInfo, SensorInfo,
};
use half::bf16;
use libc::c_char;
use num_traits::Float;

fn return_success() -> QantDriverError {
    QantDriverError {
        code: 0,
        message: [0_i8; C_STRING_DEFAULT_LENGTH],
    }
}

pub unsafe fn qant_driver_init_npu(_: u32) -> QantDriverError {
    return_success()
}

pub unsafe fn qant_driver_release_npu(_: u32) -> QantDriverError {
    return_success()
}

pub unsafe fn qant_driver_setup_logging(_: *const c_char, _: i32) -> QantDriverError {
    const MSG: &[u8] = b"logging not available for cpu-backend";

    let copy_len = C_STRING_DEFAULT_LENGTH.min(MSG.len());
    let mut message = [0_i8; C_STRING_DEFAULT_LENGTH];

    message[..copy_len]
        .iter_mut()
        .zip(MSG.iter())
        .for_each(|(dst, &src)| *dst = src as i8);

    QantDriverError { code: -1, message }
}

pub unsafe extern "C" fn qant_driver_mul_npu(
    _: u32,
    us: *const bf16,
    vs: *const bf16,
    output: *mut bf16,
    len: usize,
) -> QantDriverError {
    let us = unsafe { std::slice::from_raw_parts(us, len) };
    let vs = unsafe { std::slice::from_raw_parts(vs, len) };
    let output = unsafe { std::slice::from_raw_parts_mut(output, len) };
    for i in 0..len {
        output[i] = us[i] * vs[i];
    }

    return_success()
}

pub unsafe extern "C" fn qant_driver_mul_matrix_vec_batched(
    _: u32,
    matrix: *const bf16,
    vector_batch: *const bf16,
    output: *mut bf16,
    n_rows: usize,
    n_cols: usize,
    n_vecs: usize,
) -> QantDriverError {
    let matrix = unsafe { std::slice::from_raw_parts(matrix, n_rows * n_cols) };
    let vector_batch = unsafe { std::slice::from_raw_parts(vector_batch, n_vecs * n_cols) };
    let output = unsafe { std::slice::from_raw_parts_mut(output, n_vecs * n_rows) };

    for vec_id in 0..n_vecs {
        for row in 0..n_rows {
            let mut sum = 0.0;

            for col in 0..n_cols {
                sum +=
                    bf16::to_f32(matrix[n_cols * row + col] * vector_batch[vec_id * n_cols + col]);
            }

            output[vec_id * n_rows + row] = bf16::from_f32(sum);
        }
    }

    return_success()
}

pub unsafe extern "C" fn qant_driver_scaled_periodic_nl_npu(
    _: u32,
    us: *const bf16,
    vs: *const bf16,
    output: *mut bf16,
    len: usize,
) -> QantDriverError {
    let us = unsafe { std::slice::from_raw_parts(us, len) };
    let vs = unsafe { std::slice::from_raw_parts(vs, len) };
    let output = unsafe { std::slice::from_raw_parts_mut(output, len) };

    for i in 0..len {
        let u = us[i];
        let v = vs[i];

        output[i] = u.cos() * v;
    }

    return_success()
}

pub unsafe fn qant_driver_get_version(_: u32, output: *mut c_char, len: usize) -> QantDriverError {
    unsafe {
        let output = std::slice::from_raw_parts_mut(output as *mut u8, len);
        let bytes = b"cpu-backend;commit:none;";
        let n = bytes.len().min(len.saturating_sub(1));
        output[..n].copy_from_slice(&bytes[..n]);
        output[n] = 0; // NUL terminator
    }

    return_success()
}

pub unsafe fn qant_driver_get_device_info(_: u32) -> QantDeviceInfo {
    QantDeviceInfo {
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
        error: QantDriverError {
            code: 0,
            message: [0; C_STRING_DEFAULT_LENGTH],
        },
    }
}

pub unsafe fn qant_driver_get_available_npus(
    idxs_out: *mut u32,
    serials_out: *mut [c_char; C_STRING_DEFAULT_LENGTH],
    capacity: usize,
    out_count: *mut usize,
) -> QantDriverError {
    unsafe {
        if capacity == 0 {
            return QantDriverError {
                code: 1,
                message: [0_i8; C_STRING_DEFAULT_LENGTH],
            };
        }

        let idxs_out = std::slice::from_raw_parts_mut(idxs_out, capacity);
        idxs_out[0] = 0;
        let serials_out = std::slice::from_raw_parts_mut(
            serials_out as *mut [u8; C_STRING_DEFAULT_LENGTH],
            capacity,
        );
        let bytes = b"cpu_backend";
        let max_len = C_STRING_DEFAULT_LENGTH.saturating_sub(1);
        let n = bytes.len().min(max_len);
        serials_out[0][..n].copy_from_slice(&bytes[..n]);
        serials_out[0][n] = 0; // NUL terminator
        *out_count = 1;
    }

    return_success()
}

pub unsafe fn qant_driver_reset_perf_counter(_: u32) -> QantDriverError {
    return_success()
}

pub unsafe fn qant_driver_get_perf_counter(
    _: u32,
    out: *mut PerformanceCounterInfo,
) -> QantDriverError {
    unsafe {
        *out = PerformanceCounterInfo {
            timebased_counter: 0.0,
            stalling_counter: 0.0,
        };
    }
    return_success()
}
