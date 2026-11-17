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

use crate::qant_driver_import::dto::{
    C_STRING_DEFAULT_LENGTH, PerformanceCounterInfo, QantDataRole, QantDeviceCapabilities,
    QantDeviceInfo, QantDriverError, QantDriverHandle,
};
use half::bf16;
use num_traits::Float;
use std::collections::BTreeMap;
use std::ffi::c_char;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// Ids of the simulated Gen2 NPUs.
pub const CPU_BACKEND_NPU_IDS: [u32; 3] = [0, 1, 2];

/// A block of simulated device memory, identified by an opaque handle.
struct CpuMemoryBlock {
    data: Vec<bf16>,
    rows: usize,
    cols: usize,
    data_role: QantDataRole,
}

/// Global handle -> memory block store. `Mutex` is used to allow safe
/// concurrent access to the store.
///
/// `BTreeMap` is used instead of `HashMap` for two reasons: its `Box`-based
/// nodes have a traceable start pointer, so valgrind doesn't misreport this
/// never-dropped, process-lifetime store as "possibly lost" the way it would
/// `HashMap`'s `hashbrown` table (which only stores a pointer into the middle
/// of its allocation); and `BTreeMap::new()` is `const`, letting the store be
/// a plain `static` without `OnceLock`.
static STORE: Mutex<BTreeMap<u64, CpuMemoryBlock>> = Mutex::new(BTreeMap::new());

fn lock_store() -> std::sync::MutexGuard<'static, BTreeMap<u64, CpuMemoryBlock>> {
    STORE
        .lock()
        // unwrap_or_else needed because the lock might be poisoned if another thread panicked while holding it.
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn next_handle() -> u64 {
    static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);
    NEXT_HANDLE.fetch_add(1, Ordering::Relaxed)
}

pub unsafe fn qant_driver_init_npu(_: u32) -> QantDriverError {
    QantDriverError::success()
}

pub unsafe fn qant_driver_release_npu(_: u32) -> QantDriverError {
    QantDriverError::success()
}

pub unsafe fn qant_driver_setup_logging(_: *const c_char, _: i32) -> QantDriverError {
    QantDriverError::error("logging not available for cpu-backend")
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

    QantDriverError::success()
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

    QantDriverError::success()
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

    QantDriverError::success()
}

pub unsafe fn qant_driver_allocate(
    npu_id: u32,
    rows: usize,
    cols: usize,
    data_role: QantDataRole,
    out_handle: *mut QantDriverHandle,
) -> QantDriverError {
    let key = next_handle();
    let block = CpuMemoryBlock {
        data: vec![bf16::from_f32(0.0); rows * cols],
        rows,
        cols,
        data_role,
    };
    lock_store().insert(key, block);

    unsafe {
        *out_handle = QantDriverHandle { npu_id, key };
    }

    QantDriverError::success()
}

pub unsafe fn qant_driver_free(handle: QantDriverHandle) -> QantDriverError {
    match lock_store().remove(&handle.key) {
        Some(_) => QantDriverError::success(),
        None => QantDriverError::error("invalid memory handle"),
    }
}

pub unsafe fn qant_driver_upload(handle: QantDriverHandle, data_c: *const bf16) -> QantDriverError {
    let mut store = lock_store();
    let Some(block) = store.get_mut(&handle.key) else {
        return QantDriverError::error("invalid memory handle");
    };

    let src = unsafe { std::slice::from_raw_parts(data_c, block.rows * block.cols) };
    block.data.copy_from_slice(src);

    QantDriverError::success()
}

pub unsafe fn qant_driver_download(
    handle: QantDriverHandle,
    output_c: *mut bf16,
) -> QantDriverError {
    let store = lock_store();
    let Some(block) = store.get(&handle.key) else {
        return QantDriverError::error("invalid memory handle");
    };

    let dst = unsafe { std::slice::from_raw_parts_mut(output_c, block.rows * block.cols) };
    dst.copy_from_slice(&block.data);

    QantDriverError::success()
}

pub unsafe fn qant_driver_mul_matrix_vec_batched_mem(
    handle_matrix: QantDriverHandle,
    handle_vector: QantDriverHandle,
    handle_result: QantDriverHandle,
) -> QantDriverError {
    let mut store = lock_store();
    let (
        Some(matrix_cpu_memory_block),
        Some(vector_cpu_memory_block),
        Some(result_cpu_memory_block),
    ) = (
        store.get(&handle_matrix.key),
        store.get(&handle_vector.key),
        store.get(&handle_result.key),
    )
    else {
        return QantDriverError::error("invalid memory handle");
    };

    if matrix_cpu_memory_block.data_role != QantDataRole::Weights {
        return QantDriverError::error("data_role of matrix memory block is not Weights");
    }
    if vector_cpu_memory_block.data_role != QantDataRole::Features {
        return QantDriverError::error("data_role of vector memory block is not Features");
    }
    if result_cpu_memory_block.data_role != QantDataRole::Features {
        return QantDriverError::error("data_role of result memory block is not Features");
    }

    if matrix_cpu_memory_block.cols != vector_cpu_memory_block.cols {
        return QantDriverError::error(&format!(
            "matrix cols ({}) does not match vector cols ({})",
            matrix_cpu_memory_block.cols, vector_cpu_memory_block.cols
        ));
    }
    if result_cpu_memory_block.rows != vector_cpu_memory_block.rows {
        return QantDriverError::error(&format!(
            "result memory block rows ({}) does not match vector rows ({})",
            result_cpu_memory_block.rows, vector_cpu_memory_block.rows
        ));
    }
    if result_cpu_memory_block.cols != matrix_cpu_memory_block.rows {
        return QantDriverError::error(&format!(
            "result memory block cols ({}) does not match matrix rows ({})",
            result_cpu_memory_block.cols, matrix_cpu_memory_block.rows
        ));
    }

    let n_rows = matrix_cpu_memory_block.rows;
    let n_cols = matrix_cpu_memory_block.cols;
    let n_vecs = vector_cpu_memory_block.rows;

    let mut result = vec![bf16::from_f32(0.0); n_rows * n_vecs];
    for vec_id in 0..n_vecs {
        for row in 0..n_rows {
            let mut sum = 0.0;

            for col in 0..n_cols {
                sum += bf16::to_f32(
                    matrix_cpu_memory_block.data[n_cols * row + col]
                        * vector_cpu_memory_block.data[vec_id * n_cols + col],
                );
            }

            result[vec_id * n_rows + row] = bf16::from_f32(sum);
        }
    }
    let Some(result_cpu_memory_block) = store.get_mut(&handle_result.key) else {
        return QantDriverError::error("invalid memory handle");
    };

    result_cpu_memory_block.data = result;

    QantDriverError::success()
}

pub unsafe fn qant_driver_get_version(_: u32, output: *mut c_char, len: usize) -> QantDriverError {
    unsafe {
        let output = std::slice::from_raw_parts_mut(output as *mut u8, len);
        let bytes = b"cpu-backend;commit:none;";
        let n = bytes.len().min(len.saturating_sub(1));
        output[..n].copy_from_slice(&bytes[..n]);
        output[n] = 0; // NUL terminator
    }

    QantDriverError::success()
}

pub unsafe fn qant_driver_get_device_info(_: u32) -> QantDeviceInfo {
    QantDeviceInfo::simulated()
}

pub unsafe fn qant_driver_get_available_npus(
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
            let serial = format!("cpu_backend_{id}");
            let bytes = serial.as_bytes();
            let n = bytes.len().min(max_len);
            serials_out[i][..n].copy_from_slice(&bytes[..n]);
            serials_out[i][n] = 0; // NUL terminator
        }
        *out_count = count;
    }

    QantDriverError::success()
}

pub unsafe fn qant_driver_reset_perf_counter(_: u32) -> QantDriverError {
    QantDriverError::success()
}

pub unsafe fn qant_driver_get_perf_counter(
    _: u32,
    out: *mut PerformanceCounterInfo,
) -> QantDriverError {
    unsafe {
        *out = PerformanceCounterInfo::default();
    }
    QantDriverError::success()
}

pub fn qant_driver_get_device_capabilities(npu_id: u32) -> QantDeviceCapabilities {
    if CPU_BACKEND_NPU_IDS.contains(&npu_id) {
        QantDeviceCapabilities {
            mul: true,
            non_lin: true,
            dft: false,
            mem_handling: true,
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
