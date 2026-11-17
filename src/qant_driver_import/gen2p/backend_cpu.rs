use std::ffi::c_char;

use half::bf16;
use rustfft::num_complex::Complex;

use crate::qant_driver_import::dto::{
    C_STRING_DEFAULT_LENGTH, PerformanceCounterInfo, QantDeviceCapabilities, QantDeviceInfo,
    QantDriverError,
};

/// Ids of the simulated Gen2p NPUs.
pub const CPU_BACKEND_NPU_IDS: [u32; 3] = [10, 11, 12];

pub unsafe fn qant_gen2p_driver_init_npu(_: u32) -> QantDriverError {
    QantDriverError::success()
}

pub unsafe fn qant_gen2p_driver_release_npu(_: u32) -> QantDriverError {
    QantDriverError::success()
}

pub unsafe fn qant_gen2p_driver_get_available_npus(
    idxs_out: *mut u32,
    serials_out: *mut [c_char; C_STRING_DEFAULT_LENGTH],
    capacity: usize,
    out_count: *mut usize,
) -> QantDriverError {
    unsafe {
        if capacity == 0 {
            return QantDriverError::error("capacity must be > 0");
        }

        let count = CPU_BACKEND_NPU_IDS.len().min(capacity);
        let idxs_out = std::slice::from_raw_parts_mut(idxs_out, capacity);
        let serials_out = std::slice::from_raw_parts_mut(
            serials_out as *mut [u8; C_STRING_DEFAULT_LENGTH],
            capacity,
        );
        let max_len = C_STRING_DEFAULT_LENGTH.saturating_sub(1);
        for (i, &id) in CPU_BACKEND_NPU_IDS.iter().take(count).enumerate() {
            idxs_out[i] = id;
            let serial = format!("cpu_backend_gen2p_{id}");
            let bytes = serial.as_bytes();
            let n = bytes.len().min(max_len);
            serials_out[i][..n].copy_from_slice(&bytes[..n]);
            serials_out[i][n] = 0; // NUL terminator
        }
        *out_count = count;
    }

    QantDriverError::success()
}

pub unsafe extern "C" fn qant_gen2p_driver_dft_batched(
    _npu_id: u32,
    data: *const bf16,
    output: *mut bf16,
    dft_size: usize,
    n_dfts: usize,
) -> QantDriverError {
    // data format is a list of complex numbers consisting of 2 bf16 (real, imaginary)
    let num_elements = n_dfts * dft_size;
    let data = { unsafe { std::slice::from_raw_parts(data, num_elements * 2) } };
    let output = { unsafe { std::slice::from_raw_parts_mut(output, num_elements * 2) } };

    // cast res to complex format (interleaved real and imaginary parts)
    let mut res: Vec<Complex<bf16>> = data
        .chunks(2)
        .map(|co| Complex {
            re: co[0],
            im: co[1],
        })
        .collect();

    let fft = rustfft::FftPlanner::new().plan_fft_forward(dft_size);

    // perform n dfts in batch, each dft has num_elements complex numbers
    for i in 0..n_dfts {
        let start = i * dft_size;
        let end = start + dft_size;
        fft.process(&mut res[start..end]);
    }

    let res: Vec<bf16> = res.iter().flat_map(|c| [c.re, c.im]).collect();
    output.copy_from_slice(&res);

    QantDriverError::success()
}

pub unsafe fn qant_gen2p_driver_setup_logging(_: *const c_char, _: i32) -> QantDriverError {
    QantDriverError::error("logging not available for cpu-backend")
}

pub unsafe fn qant_gen2p_driver_get_version(
    _: u32,
    output: *mut c_char,
    len: usize,
) -> QantDriverError {
    if len == 0 {
        return QantDriverError::error("output buffer must not be empty");
    }
    let output = unsafe { std::slice::from_raw_parts_mut(output as *mut u8, len) };
    let bytes = b"cpu-backend-gen2p;commit:none;";
    let n = bytes.len().min(len - 1);
    output[..n].copy_from_slice(&bytes[..n]);
    output[n] = 0; // NUL terminator

    QantDriverError::success()
}

pub unsafe fn qant_gen2p_driver_get_device_info(_: u32) -> QantDeviceInfo {
    QantDeviceInfo::simulated()
}

pub unsafe fn qant_gen2p_driver_reset_perf_counter(_: u32) -> QantDriverError {
    QantDriverError::success()
}

pub unsafe fn qant_gen2p_driver_get_perf_counter(
    _: u32,
    out: *mut PerformanceCounterInfo,
) -> QantDriverError {
    unsafe {
        *out = PerformanceCounterInfo::default();
    }
    QantDriverError::success()
}

pub fn qant_gen2p_driver_get_device_capabilities(npu_id: u32) -> QantDeviceCapabilities {
    if CPU_BACKEND_NPU_IDS.contains(&npu_id) {
        QantDeviceCapabilities {
            mul: false,
            non_lin: false,
            dft: true,
            mem_handling: false,
            error: QantDriverError::success(),
        }
    } else {
        QantDeviceCapabilities {
            mul: false,
            non_lin: false,
            dft: false,
            mem_handling: false,
            error: QantDriverError::error("NPU not available"),
        }
    }
}
