/*
How to forward a function from Rust to C:
1. Import the rust function
2. Write a wrapper that takes the C arguments as a tuple and returns a Result.
   This layer performs the conversion from C arguments to rust arguments.
   It is explicitly encouraged to return Err() when something goes wrong.
3. Write the extern C function that takes the C arguments as-is and returns a *mut DLManagedTensorVersioned.
   Use wrap_rust_fn to convert ToolkitErrors into nullpointers.
4. Add your function to qant_native_computing_toolkit.h
   Make sure to use C types
5. Add a test to /tests_C/test.cpp
*/
use std::{
    ffi::{CStr, c_char, c_int},
    ptr,
};

use crate::{
    bias,
    conv::{
        self, ConvAttributes, ConvTransposeAttributes, ConvTransposeWeightsHandle,
        ConvWeightsHandle,
    },
    data_structures::{QantTensor, QantTensor1, QantTensor2, QantTensor3, QantTensor4},
    errors::ToolkitError,
    global_settings::Value,
    linear,
    memory_management::{self, QantMemoryHandle},
    non_linear, norm,
    pooling::{self, PoolingAttributes},
    qant_driver_import::{self, QantDataRole, QantDeviceCapability, dto},
    toolkit_error, utils,
};

mod dlpack;
use dlpack::{DLDevice, DLDeviceType_kDLCPU, DLManagedTensorVersioned};
use half::bf16;

// Error convention of the C interface: functions return 0 on success and -1 on
// failure, or a valid pointer on success and NULL on failure. On failure, the
// error message is printed to stderr.

/// The single place where the C interface reports errors.
#[allow(clippy::disallowed_macros)]
fn print_error(e: &ToolkitError) {
    eprintln!("{e}");
}

/// Maps a result to the C error convention: 0 on success, -1 on failure.
fn to_error_code(result: Result<(), ToolkitError>) -> c_int {
    match result {
        Ok(()) => 0,
        Err(e) => {
            print_error(&e);
            -1
        }
    }
}

/// Maps a result to an owning pointer, or NULL on failure.
fn to_boxed_ptr<T>(result: Result<T, ToolkitError>) -> *mut T {
    match result {
        Ok(value) => Box::into_raw(Box::new(value)),
        Err(e) => {
            print_error(&e);
            ptr::null_mut()
        }
    }
}

fn setup_logging_fallible(folder_name: *const c_char, loglevel: c_int) -> Result<(), ToolkitError> {
    if folder_name.is_null() {
        return Err(ToolkitError::from_str("folder_name: got nullptr"));
    }
    let folder_name = unsafe { CStr::from_ptr(folder_name) }
        .to_str()
        .map_err(|_| ToolkitError::from_str("cannot read folder_name"))?;
    let loglevel = utils::int_to_loglevel(loglevel)?;
    qant_driver_import::setup_logging(folder_name, loglevel)
}

#[unsafe(no_mangle)]
pub extern "C" fn setup_logging(folder_name: *const c_char, loglevel: c_int) -> c_int {
    to_error_code(setup_logging_fallible(folder_name, loglevel))
}

/// Writes the value of `result` to `out`, see [`to_error_code`] for the return value.
fn write_out<T>(result: Result<T, ToolkitError>, out: *mut T) -> c_int {
    if out.is_null() {
        return to_error_code(Err(ToolkitError::from_str("out: got nullptr")));
    }
    to_error_code(result.map(|value| unsafe { out.write(value) }))
}

#[unsafe(no_mangle)]
pub extern "C" fn get_driver_info(npu_id: i32, output: *mut c_char, len: usize) -> c_int {
    if output.is_null() {
        return to_error_code(Err(ToolkitError::from_str("output: got nullptr")));
    }
    to_error_code(qant_driver_import::get_version_ptr_based(
        npu_id, output, len,
    ))
}

#[unsafe(no_mangle)]
pub extern "C" fn get_sensor_info(
    npu_id: i32,
    out: *mut qant_driver_import::dto::SensorInfo,
) -> c_int {
    write_out(qant_driver_import::get_sensor_info(npu_id), out)
}

#[unsafe(no_mangle)]
pub extern "C" fn get_version_info(
    npu_id: i32,
    out: *mut qant_driver_import::dto::QantVersionInfo,
) -> c_int {
    write_out(qant_driver_import::get_version_info(npu_id), out)
}

fn check_non_null<T>(ptr: *const T, name: &str) -> Result<(), ToolkitError> {
    if ptr.is_null() {
        return Err(toolkit_error!("{name}: got nullptr"));
    }
    Ok(())
}

/// Copies `s` into a NUL-terminated C char array, truncated if it does not fit.
fn rust_str_to_c_char(s: &str) -> [c_char; dto::C_STRING_DEFAULT_LENGTH] {
    let mut c_chars = [0 as c_char; dto::C_STRING_DEFAULT_LENGTH];
    for (dst, &src) in c_chars[..dto::C_STRING_DEFAULT_LENGTH - 1]
        .iter_mut()
        .zip(s.as_bytes())
    {
        *dst = src as c_char;
    }
    c_chars
}

/// Writes `len` to `out_count` and errors if it exceeds `capacity`.
fn check_capacity(len: usize, capacity: usize, out_count: *mut usize) -> Result<(), ToolkitError> {
    check_non_null(out_count, "out_count")?;
    unsafe { out_count.write(len) };
    if len > capacity {
        return Err(toolkit_error!(
            "{len} elements to write, but capacity is only {capacity}"
        ));
    }
    Ok(())
}

fn get_available_npus_fallible(
    operations: *const QantDeviceCapability,
    n_operations: usize,
    idxs_out: *mut u32,
    serials_out: *mut [c_char; qant_driver_import::dto::C_STRING_DEFAULT_LENGTH],
    capacity: usize,
    out_count: *mut usize,
) -> Result<(), ToolkitError> {
    check_non_null(idxs_out, "idxs_out")?;
    check_non_null(serials_out, "serials_out")?;
    // from_raw_parts requires a non-null pointer, even for an empty slice
    let ops: &[QantDeviceCapability] = match n_operations {
        0 => &[],
        n => {
            check_non_null(operations, "operations")?;
            // C may pass any int; check the raw values before reading them as the enum,
            // since an invalid discriminant is undefined behaviour
            let raw = unsafe { std::slice::from_raw_parts(operations.cast::<u32>(), n) };
            let is_valid = |v: u32| QantDeviceCapability::ALL.iter().any(|&c| c as u32 == v);
            if let Some(v) = raw.iter().find(|&&v| !is_valid(v)) {
                return Err(toolkit_error!("invalid QantDeviceCapability {v}"));
            }
            unsafe { std::slice::from_raw_parts(operations, n) }
        }
    };
    let npus = qant_driver_import::get_available_npus(ops)?;

    check_capacity(npus.len(), capacity, out_count)?;
    for (i, (id, serial)) in npus.iter().enumerate() {
        unsafe {
            idxs_out.add(i).write(*id);
            serials_out.add(i).write(rust_str_to_c_char(serial));
        }
    }
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "C" fn get_available_npus(
    operations: *const QantDeviceCapability,
    n_operations: usize,
    idxs_out: *mut u32,
    serials_out: *mut [c_char; qant_driver_import::dto::C_STRING_DEFAULT_LENGTH],
    capacity: usize,
    out_count: *mut usize,
) -> c_int {
    to_error_code(get_available_npus_fallible(
        operations,
        n_operations,
        idxs_out,
        serials_out,
        capacity,
        out_count,
    ))
}

fn get_capabilities_fallible(
    npu_id: i32,
    capabilities_out: *mut QantDeviceCapability,
    capacity: usize,
    out_count: *mut usize,
) -> Result<(), ToolkitError> {
    check_non_null(capabilities_out, "capabilities_out")?;
    let capabilities = qant_driver_import::get_capabilities(npu_id)?;

    check_capacity(capabilities.len(), capacity, out_count)?;
    unsafe {
        ptr::copy_nonoverlapping(capabilities.as_ptr(), capabilities_out, capabilities.len())
    };
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "C" fn get_capabilities(
    npu_id: i32,
    capabilities_out: *mut QantDeviceCapability,
    capacity: usize,
    out_count: *mut usize,
) -> c_int {
    to_error_code(get_capabilities_fallible(
        npu_id,
        capabilities_out,
        capacity,
        out_count,
    ))
}

#[unsafe(no_mangle)]
pub extern "C" fn reset_perf_counter(npu_id: i32) -> c_int {
    to_error_code(qant_driver_import::reset_perf_counter(npu_id))
}

#[unsafe(no_mangle)]
pub extern "C" fn get_perf_counter(
    npu_id: i32,
    out: *mut qant_driver_import::dto::PerformanceCounterInfo,
) -> c_int {
    write_out(qant_driver_import::get_perf_counter(npu_id), out)
}

// take a rust function that takes a **tuple of arguments** and returns a result,
// execute it,
// return the dltensor of the value if ok,
// print error and return null if not ok
// if it is with batch it returns a 4d tensor otherwise a 3d tensor
fn wrap_rust_fn<'a, F, Args, const D: usize>(
    func: F,
) -> impl Fn(Args) -> *mut DLManagedTensorVersioned
where
    F: Fn(Args) -> Result<(QantTensor<'a, D>, DLDevice), ToolkitError>,
{
    move |args: Args| {
        to_boxed_ptr(
            func(args).and_then(|(fdata, device)| dlpack::qant_tensor_to_dltensor(fdata, device)),
        )
    }
}

#[unsafe(no_mangle)]
// must avoid naming conflict with driver init_npu()
pub extern "C" fn init_npu(id: c_int) -> c_int {
    to_error_code(qant_driver_import::init_npu(id))
}

#[unsafe(no_mangle)]
pub extern "C" fn release_npu(id: c_int) -> c_int {
    to_error_code(qant_driver_import::release_npu(id))
}

#[unsafe(no_mangle)]
pub extern "C" fn npu_allocate(
    npu_id: i32,
    rows: usize,
    cols: usize,
    data_role: QantDataRole,
) -> *mut QantMemoryHandle {
    to_boxed_ptr(QantMemoryHandle::allocate(npu_id, rows, cols, data_role))
}

fn free_boxed<T>(handle: *mut T, name: &str) -> c_int {
    if handle.is_null() {
        return to_error_code(Err(toolkit_error!("got nullptr instead of *{name}")));
    }
    drop(unsafe { Box::from_raw(handle) });
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn npu_free(handle: *mut QantMemoryHandle) -> c_int {
    free_boxed(handle, "QantMemoryHandle")
}

fn npu_upload_fallible(
    npu_id: i32,
    data_dltensor_ptr: *const DLManagedTensorVersioned,
    data_role: QantDataRole,
) -> Result<QantMemoryHandle, ToolkitError> {
    dlpack::version_check(&[data_dltensor_ptr])?;
    let matrix: QantTensor2 = dlpack::dltensor_to_qant_tensor(data_dltensor_ptr, "data")?;
    QantMemoryHandle::from_matrix(npu_id, &matrix, data_role)
}

#[unsafe(no_mangle)]
pub extern "C" fn npu_upload(
    npu_id: i32,
    data_dltensor_ptr: *const DLManagedTensorVersioned,
    data_role: QantDataRole,
) -> *mut QantMemoryHandle {
    to_boxed_ptr(npu_upload_fallible(npu_id, data_dltensor_ptr, data_role))
}

fn npu_upload_into_fallible(
    handle: *const QantMemoryHandle,
    data_dltensor_ptr: *const DLManagedTensorVersioned,
) -> Result<(), ToolkitError> {
    if handle.is_null() {
        return Err(ToolkitError::from_str(
            "got nullptr instead of *QantMemoryHandle",
        ));
    }
    dlpack::version_check(&[data_dltensor_ptr])?;
    let matrix: QantTensor2 = dlpack::dltensor_to_qant_tensor(data_dltensor_ptr, "data")?;
    // borrow the handle to call upload, but don't take ownership of it
    unsafe { &*handle }.upload(&matrix)
}

#[unsafe(no_mangle)]
pub extern "C" fn npu_upload_into(
    handle: *const QantMemoryHandle,
    data_dltensor_ptr: *const DLManagedTensorVersioned,
) -> c_int {
    to_error_code(npu_upload_into_fallible(handle, data_dltensor_ptr))
}

fn npu_download_fallible<'a>(
    handle: *const QantMemoryHandle,
) -> Result<(QantTensor2<'a>, DLDevice), ToolkitError> {
    if handle.is_null() {
        return Err(ToolkitError::from_str(
            "got nullptr instead of *QantMemoryHandle",
        ));
    }
    // borrow the handle to call download, but don't take ownership of it
    let matrix = unsafe { &*handle }.download()?;
    let device = DLDevice {
        device_type: DLDeviceType_kDLCPU,
        device_id: 0,
    };
    Ok((matrix, device))
}

#[unsafe(no_mangle)]
pub extern "C" fn npu_download(handle: *const QantMemoryHandle) -> *mut DLManagedTensorVersioned {
    wrap_rust_fn(npu_download_fallible)(handle)
}

fn linear_fprop_fallible<'a>(
    (npu_id, features_dltensor_ptr, weights_dltensor_ptr): (
        c_int,
        *const DLManagedTensorVersioned,
        *const DLManagedTensorVersioned,
    ),
) -> Result<(QantTensor2<'a>, DLDevice), ToolkitError> {
    dlpack::version_check(&[features_dltensor_ptr, weights_dltensor_ptr])?;
    let device = dlpack::device_check(&[features_dltensor_ptr, weights_dltensor_ptr])?;
    let feature_data = dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;
    let filter_data = dlpack::dltensor_to_qant_tensor(weights_dltensor_ptr, "weights")?;
    let result = linear::linear_fprop(npu_id, &feature_data, &filter_data)?;
    Ok((result, device))
}

#[unsafe(no_mangle)]
pub extern "C" fn linear_fprop(
    npu_id: i32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    weights_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *mut DLManagedTensorVersioned {
    wrap_rust_fn(linear_fprop_fallible)((npu_id, features_dltensor_ptr, weights_dltensor_ptr))
}

fn linear_fprop_preloaded_weights_fallible<'a>(
    (features_dltensor_ptr, weights_handle): (
        *const DLManagedTensorVersioned,
        *const QantMemoryHandle,
    ),
) -> Result<(QantTensor2<'a>, DLDevice), ToolkitError> {
    if weights_handle.is_null() {
        return Err(ToolkitError::from_str(
            "got nullptr instead of *QantMemoryHandle",
        ));
    }
    dlpack::version_check(&[features_dltensor_ptr])?;
    let device = dlpack::device_check(&[features_dltensor_ptr])?;
    let features_data: QantTensor2 =
        dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;
    // borrow the handle, but don't take ownership of it
    let weights_handle = unsafe { &*weights_handle };

    let &[n_batches, _] = features_data.shape();
    let features_handle = QantMemoryHandle::from_matrix(
        weights_handle.npu_id() as i32,
        &features_data,
        QantDataRole::Features,
    )?;
    let result_handle = QantMemoryHandle::allocate(
        weights_handle.npu_id() as i32,
        n_batches,
        weights_handle.rows(),
        QantDataRole::Features,
    )?;
    memory_management::mul_matrix_vec_batched_mem(
        weights_handle,
        &features_handle,
        &result_handle,
    )?;
    Ok((result_handle.download()?, device))
}

#[unsafe(no_mangle)]
pub extern "C" fn linear_fprop_preloaded_weights(
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    weights_handle: *const QantMemoryHandle,
) -> *mut DLManagedTensorVersioned {
    wrap_rust_fn(linear_fprop_preloaded_weights_fallible)((features_dltensor_ptr, weights_handle))
}

#[unsafe(no_mangle)]
pub extern "C" fn linear_fprop_preloaded(
    features_handle: *const QantMemoryHandle,
    weights_handle: *const QantMemoryHandle,
    result_handle: *const QantMemoryHandle,
) -> c_int {
    if features_handle.is_null() || weights_handle.is_null() || result_handle.is_null() {
        return to_error_code(Err(ToolkitError::from_str(
            "got nullptr instead of *QantMemoryHandle",
        )));
    }
    // borrow the handles to call mul_matrix_vec_batched_mem, but don't take ownership of them
    let (features_handle, weights_handle, result_handle) =
        unsafe { (&*features_handle, &*weights_handle, &*result_handle) };
    to_error_code(memory_management::mul_matrix_vec_batched_mem(
        weights_handle,
        features_handle,
        result_handle,
    ))
}

fn conv_fprop_fallible<'a>(
    (npu_id, features_dltensor_ptr, weights_dltensor_ptr, padding, stride, dilation): (
        c_int,
        *const DLManagedTensorVersioned,
        *const DLManagedTensorVersioned,
        usize,
        usize,
        usize,
    ),
) -> Result<(QantTensor4<'a>, DLDevice), ToolkitError> {
    dlpack::version_check(&[features_dltensor_ptr, weights_dltensor_ptr])?;
    let device = dlpack::device_check(&[features_dltensor_ptr, weights_dltensor_ptr])?;
    let features_data = dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;
    let weights_data = dlpack::dltensor_to_qant_tensor(weights_dltensor_ptr, "weights")?;

    let conv_attrs = ConvAttributes::new(padding, stride, dilation)?;

    let result = conv::conv_fprop(npu_id, &features_data, &weights_data, conv_attrs)?;
    Ok((result, device))
}

#[unsafe(no_mangle)]
pub extern "C" fn conv_fprop(
    npu_id: i32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    weights_dltensor_ptr: *const DLManagedTensorVersioned,
    padding: usize,
    stride: usize,
    dilation: usize,
) -> *mut DLManagedTensorVersioned {
    wrap_rust_fn(conv_fprop_fallible)((
        npu_id,
        features_dltensor_ptr,
        weights_dltensor_ptr,
        padding,
        stride,
        dilation,
    ))
}

fn conv_transpose_fprop_fallible<'a>(
    (
        npu_id,
        features_dltensor_ptr,
        weights_dltensor_ptr,
        padding,
        stride,
        dilation,
        output_padding,
    ): (
        c_int,
        *const DLManagedTensorVersioned,
        *const DLManagedTensorVersioned,
        usize,
        usize,
        usize,
        usize,
    ),
) -> Result<(QantTensor4<'a>, DLDevice), ToolkitError> {
    dlpack::version_check(&[features_dltensor_ptr, weights_dltensor_ptr])?;
    let device = dlpack::device_check(&[features_dltensor_ptr, weights_dltensor_ptr])?;
    let features_data = dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;
    let weights_data = dlpack::dltensor_to_qant_tensor(weights_dltensor_ptr, "weights")?;
    let conv_attrs = ConvTransposeAttributes::new(padding, stride, dilation, output_padding)?;

    let result = conv::conv_transpose_fprop(npu_id, &features_data, &weights_data, conv_attrs)?;
    Ok((result, device))
}

#[unsafe(no_mangle)]
pub extern "C" fn conv_transpose_fprop(
    npu_id: i32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    weights_dltensor_ptr: *const DLManagedTensorVersioned,
    padding: usize,
    stride: usize,
    dilation: usize,
    output_padding: usize,
) -> *mut DLManagedTensorVersioned {
    wrap_rust_fn(conv_transpose_fprop_fallible)((
        npu_id,
        features_dltensor_ptr,
        weights_dltensor_ptr,
        padding,
        stride,
        dilation,
        output_padding,
    ))
}

fn upload_conv_weights_fallible(
    npu_id: i32,
    weights_dltensor_ptr: *const DLManagedTensorVersioned,
) -> Result<ConvWeightsHandle, ToolkitError> {
    dlpack::version_check(&[weights_dltensor_ptr])?;
    let weights_data: QantTensor4 =
        dlpack::dltensor_to_qant_tensor(weights_dltensor_ptr, "weights")?;
    conv::upload_conv_weights(npu_id, &weights_data)
}

#[unsafe(no_mangle)]
pub extern "C" fn upload_conv_weights(
    npu_id: i32,
    weights_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *mut ConvWeightsHandle {
    to_boxed_ptr(upload_conv_weights_fallible(npu_id, weights_dltensor_ptr))
}

#[unsafe(no_mangle)]
pub extern "C" fn conv_weights_free(handle: *mut ConvWeightsHandle) -> c_int {
    free_boxed(handle, "ConvWeightsHandle")
}

fn conv_fprop_preloaded_weights_fallible<'a>(
    (features_dltensor_ptr, weights_handle, padding, stride, dilation): (
        *const DLManagedTensorVersioned,
        *const ConvWeightsHandle,
        usize,
        usize,
        usize,
    ),
) -> Result<(QantTensor4<'a>, DLDevice), ToolkitError> {
    if weights_handle.is_null() {
        return Err(ToolkitError::from_str(
            "got nullptr instead of *ConvWeightsHandle",
        ));
    }
    dlpack::version_check(&[features_dltensor_ptr])?;
    let device = dlpack::device_check(&[features_dltensor_ptr])?;
    let features_data = dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;

    let conv_attrs = ConvAttributes::new(padding, stride, dilation)?;

    // borrow the handle, but don't take ownership of it
    let result = conv::conv_fprop_preloaded_weights(
        &features_data,
        unsafe { &*weights_handle },
        conv_attrs,
    )?;
    Ok((result, device))
}

#[unsafe(no_mangle)]
pub extern "C" fn conv_fprop_preloaded_weights(
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    weights_handle: *const ConvWeightsHandle,
    padding: usize,
    stride: usize,
    dilation: usize,
) -> *mut DLManagedTensorVersioned {
    wrap_rust_fn(conv_fprop_preloaded_weights_fallible)((
        features_dltensor_ptr,
        weights_handle,
        padding,
        stride,
        dilation,
    ))
}

fn upload_conv_transpose_weights_fallible(
    npu_id: i32,
    weights_dltensor_ptr: *const DLManagedTensorVersioned,
) -> Result<ConvTransposeWeightsHandle, ToolkitError> {
    dlpack::version_check(&[weights_dltensor_ptr])?;
    let weights_data: QantTensor4 =
        dlpack::dltensor_to_qant_tensor(weights_dltensor_ptr, "weights")?;
    conv::upload_conv_transpose_weights(npu_id, &weights_data)
}

#[unsafe(no_mangle)]
pub extern "C" fn upload_conv_transpose_weights(
    npu_id: i32,
    weights_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *mut ConvTransposeWeightsHandle {
    to_boxed_ptr(upload_conv_transpose_weights_fallible(
        npu_id,
        weights_dltensor_ptr,
    ))
}

#[unsafe(no_mangle)]
pub extern "C" fn conv_transpose_weights_free(handle: *mut ConvTransposeWeightsHandle) -> c_int {
    free_boxed(handle, "ConvTransposeWeightsHandle")
}

fn conv_transpose_fprop_preloaded_weights_fallible<'a>(
    (features_dltensor_ptr, weights_handle, padding, stride, dilation, output_padding): (
        *const DLManagedTensorVersioned,
        *const ConvTransposeWeightsHandle,
        usize,
        usize,
        usize,
        usize,
    ),
) -> Result<(QantTensor4<'a>, DLDevice), ToolkitError> {
    if weights_handle.is_null() {
        return Err(ToolkitError::from_str(
            "got nullptr instead of *ConvTransposeWeightsHandle",
        ));
    }
    dlpack::version_check(&[features_dltensor_ptr])?;
    let device = dlpack::device_check(&[features_dltensor_ptr])?;
    let features_data = dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;
    let conv_attrs = ConvTransposeAttributes::new(padding, stride, dilation, output_padding)?;

    // borrow the handle, but don't take ownership of it
    let result = conv::conv_transpose_fprop_preloaded_weights(
        &features_data,
        unsafe { &*weights_handle },
        conv_attrs,
    )?;
    Ok((result, device))
}

#[unsafe(no_mangle)]
pub extern "C" fn conv_transpose_fprop_preloaded_weights(
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    weights_handle: *const ConvTransposeWeightsHandle,
    padding: usize,
    stride: usize,
    dilation: usize,
    output_padding: usize,
) -> *mut DLManagedTensorVersioned {
    wrap_rust_fn(conv_transpose_fprop_preloaded_weights_fallible)((
        features_dltensor_ptr,
        weights_handle,
        padding,
        stride,
        dilation,
        output_padding,
    ))
}

fn add_bias_fprop_fallible<'a>(
    (npu_id, features_dltensor_ptr, bias_dltensor_ptr): (
        c_int,
        *const DLManagedTensorVersioned,
        *const DLManagedTensorVersioned,
    ),
) -> Result<(QantTensor2<'a>, DLDevice), ToolkitError> {
    dlpack::version_check(&[features_dltensor_ptr, bias_dltensor_ptr])?;
    let device = dlpack::device_check(&[features_dltensor_ptr, bias_dltensor_ptr])?;
    let feature_fdata = dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;
    let bias_fdata = dlpack::dltensor_to_qant_tensor(bias_dltensor_ptr, "bias")?;

    let ret = bias::add_bias_fprop(npu_id, &feature_fdata, &bias_fdata)?;

    Ok((ret, device))
}

#[unsafe(no_mangle)]
pub extern "C" fn add_bias_fprop(
    npu_id: i32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    bias_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *mut DLManagedTensorVersioned {
    wrap_rust_fn(add_bias_fprop_fallible)((npu_id, features_dltensor_ptr, bias_dltensor_ptr))
}

fn add_bias_for_conv2d_fprop_fallible<'a>(
    (npu_id, features_dltensor_ptr, bias_dltensor_ptr): (
        c_int,
        *const DLManagedTensorVersioned,
        *const DLManagedTensorVersioned,
    ),
) -> Result<(QantTensor4<'a>, DLDevice), ToolkitError> {
    dlpack::version_check(&[features_dltensor_ptr, bias_dltensor_ptr])?;
    let device = dlpack::device_check(&[features_dltensor_ptr, bias_dltensor_ptr])?;
    let feature_fdata = dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;
    let bias_fdata = dlpack::dltensor_to_qant_tensor(bias_dltensor_ptr, "bias")?;

    let ret = bias::add_bias_for_conv2d_fprop(npu_id, &feature_fdata, &bias_fdata)?;

    Ok((ret, device))
}

#[unsafe(no_mangle)]
pub extern "C" fn add_bias_for_conv2d_fprop(
    npu_id: i32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    bias_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *mut DLManagedTensorVersioned {
    wrap_rust_fn(add_bias_for_conv2d_fprop_fallible)((
        npu_id,
        features_dltensor_ptr,
        bias_dltensor_ptr,
    ))
}

fn mul_npu_fallible<'a>(
    (npu_id, us_dltensor_ptr, vs_dltensor_ptr): (
        c_int,
        *const DLManagedTensorVersioned,
        *const DLManagedTensorVersioned,
    ),
) -> Result<(QantTensor1<'a>, DLDevice), ToolkitError> {
    dlpack::version_check(&[us_dltensor_ptr, vs_dltensor_ptr])?;
    let device = dlpack::device_check(&[us_dltensor_ptr, vs_dltensor_ptr])?;
    let us_fdata = dlpack::dltensor_to_qant_tensor(us_dltensor_ptr, "us")?;
    let vs_fdata = dlpack::dltensor_to_qant_tensor(vs_dltensor_ptr, "vs")?;

    let result = crate::elementwise::mul(npu_id, &us_fdata, &vs_fdata)?;
    Ok((result, device))
}

#[unsafe(no_mangle)]
pub extern "C" fn mul_npu(
    npu_id: i32,
    us_dltensor_ptr: *const DLManagedTensorVersioned,
    vs_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *mut DLManagedTensorVersioned {
    wrap_rust_fn(mul_npu_fallible)((npu_id, us_dltensor_ptr, vs_dltensor_ptr))
}

fn relu_fprop_fallible<'a>(
    (npu_id, features_dltensor_ptr): (c_int, *const DLManagedTensorVersioned),
) -> Result<(QantTensor1<'a>, DLDevice), ToolkitError> {
    dlpack::version_check(&[features_dltensor_ptr])?;
    let device = dlpack::device_check(&[features_dltensor_ptr])?;
    let feature_fdata = dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;
    Ok((non_linear::relu_fprop(npu_id, &feature_fdata)?, device))
}

#[unsafe(no_mangle)]
pub extern "C" fn relu_fprop(
    npu_id: i32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *mut DLManagedTensorVersioned {
    wrap_rust_fn(relu_fprop_fallible)((npu_id, features_dltensor_ptr))
}

fn sigmoid_fprop_fallible<'a>(
    (npu_id, features_dltensor_ptr): (c_int, *const DLManagedTensorVersioned),
) -> Result<(QantTensor1<'a>, DLDevice), ToolkitError> {
    dlpack::version_check(&[features_dltensor_ptr])?;
    let device = dlpack::device_check(&[features_dltensor_ptr])?;
    let feature_fdata = dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;
    Ok((non_linear::sigmoid_fprop(npu_id, &feature_fdata)?, device))
}

#[unsafe(no_mangle)]
pub extern "C" fn sigmoid_fprop(
    npu_id: i32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *mut DLManagedTensorVersioned {
    wrap_rust_fn(sigmoid_fprop_fallible)((npu_id, features_dltensor_ptr))
}

fn softmax_fprop_fallible<'a>(
    (npu_id, features_dltensor_ptr): (c_int, *const DLManagedTensorVersioned),
) -> Result<(QantTensor2<'a>, DLDevice), ToolkitError> {
    dlpack::version_check(&[features_dltensor_ptr])?;
    let device = dlpack::device_check(&[features_dltensor_ptr])?;
    let feature_fdata = dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;
    let result = non_linear::softmax_fprop(npu_id, &feature_fdata)?;
    Ok((result, device))
}

#[unsafe(no_mangle)]
pub extern "C" fn softmax_fprop(
    npu_id: i32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *mut DLManagedTensorVersioned {
    wrap_rust_fn(softmax_fprop_fallible)((npu_id, features_dltensor_ptr))
}

fn batchnorm2d_fprop_fallible<'a>(
    (
        npu_id,
        features_dltensor_ptr,
        means_dltensor_ptr,
        variances_dltensor_ptr,
        weights_dltensor_ptr,
        bias_dltensor_ptr,
        eps,
    ): (
        c_int,
        *const DLManagedTensorVersioned,
        *const DLManagedTensorVersioned,
        *const DLManagedTensorVersioned,
        *const DLManagedTensorVersioned,
        *const DLManagedTensorVersioned,
        Value,
    ),
) -> Result<(QantTensor4<'a>, DLDevice), ToolkitError> {
    dlpack::version_check(&[
        features_dltensor_ptr,
        means_dltensor_ptr,
        variances_dltensor_ptr,
        weights_dltensor_ptr,
        bias_dltensor_ptr,
    ])?;
    let device = dlpack::device_check(&[
        features_dltensor_ptr,
        means_dltensor_ptr,
        variances_dltensor_ptr,
        weights_dltensor_ptr,
        bias_dltensor_ptr,
    ])?;
    let result = norm::batchnorm2d_fprop(
        npu_id,
        &dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?,
        &dlpack::dltensor_to_qant_tensor(means_dltensor_ptr, "means")?,
        &dlpack::dltensor_to_qant_tensor(variances_dltensor_ptr, "variances")?,
        &dlpack::dltensor_to_qant_tensor(weights_dltensor_ptr, "weights")?,
        &dlpack::dltensor_to_qant_tensor(bias_dltensor_ptr, "bias")?,
        eps,
    )?;
    Ok((result, device))
}

#[unsafe(no_mangle)]
pub extern "C" fn batchnorm2d_fprop(
    npu_id: i32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    means_dltensor_ptr: *const DLManagedTensorVersioned,
    variances_dltensor_ptr: *const DLManagedTensorVersioned,
    weights_dltensor_ptr: *const DLManagedTensorVersioned,
    bias_dltensor_ptr: *const DLManagedTensorVersioned,
    eps: f32,
) -> *mut DLManagedTensorVersioned {
    wrap_rust_fn(batchnorm2d_fprop_fallible)((
        npu_id,
        features_dltensor_ptr,
        means_dltensor_ptr,
        variances_dltensor_ptr,
        weights_dltensor_ptr,
        bias_dltensor_ptr,
        bf16::from_f32(eps),
    ))
}

fn maxpool2d_fprop_fallible<'a>(
    (npu_id, features_dltensor_ptr, kernel_height, kernel_width, padding, stride): (
        c_int,
        *const DLManagedTensorVersioned,
        usize,
        usize,
        usize,
        usize,
    ),
) -> Result<(QantTensor4<'a>, DLDevice), ToolkitError> {
    dlpack::version_check(&[features_dltensor_ptr])?;
    let device = dlpack::device_check(&[features_dltensor_ptr])?;
    let feature_data = dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;

    let pool_attrs = PoolingAttributes {
        kernel_height,
        kernel_width,
        padding,
        stride,
        count_include_pad: false,
    };

    let result = pooling::maxpool2d_fprop(npu_id, &feature_data, pool_attrs)?;
    Ok((result, device))
}

#[unsafe(no_mangle)]
pub extern "C" fn maxpool2d_fprop(
    npu_id: i32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    kernel_height: usize,
    kernel_width: usize,
    padding: usize,
    stride: usize,
) -> *mut DLManagedTensorVersioned {
    wrap_rust_fn(maxpool2d_fprop_fallible)((
        npu_id,
        features_dltensor_ptr,
        kernel_height,
        kernel_width,
        padding,
        stride,
    ))
}

fn avgpool2d_fprop_fallible<'a>(
    (
        npu_id,
        features_dltensor_ptr,
        kernel_height,
        kernel_width,
        padding,
        stride,
        count_include_pad,
    ): (
        c_int,
        *const DLManagedTensorVersioned,
        usize,
        usize,
        usize,
        usize,
        bool,
    ),
) -> Result<(QantTensor4<'a>, DLDevice), ToolkitError> {
    dlpack::version_check(&[features_dltensor_ptr])?;
    let device = dlpack::device_check(&[features_dltensor_ptr])?;
    let feature_data = dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;

    let pool_attrs = PoolingAttributes {
        kernel_height,
        kernel_width,
        padding,
        stride,
        count_include_pad,
    };

    let result = pooling::avgpool2d_fprop(npu_id, &feature_data, pool_attrs)?;
    Ok((result, device))
}

#[unsafe(no_mangle)]
pub extern "C" fn avgpool2d_fprop(
    npu_id: i32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    kernel_height: usize,
    kernel_width: usize,
    padding: usize,
    stride: usize,
    count_include_pad: bool,
) -> *mut DLManagedTensorVersioned {
    wrap_rust_fn(avgpool2d_fprop_fallible)((
        npu_id,
        features_dltensor_ptr,
        kernel_height,
        kernel_width,
        padding,
        stride,
        count_include_pad,
    ))
}

fn adaptive_maxpool2d_fprop_fallible<'a>(
    (npu_id, features_dltensor_ptr, output_height, output_width): (
        c_int,
        *const DLManagedTensorVersioned,
        usize,
        usize,
    ),
) -> Result<(QantTensor4<'a>, DLDevice), ToolkitError> {
    dlpack::version_check(&[features_dltensor_ptr])?;
    let device = dlpack::device_check(&[features_dltensor_ptr])?;
    let feature_data = dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;

    let result =
        pooling::adaptive_maxpool2d_fprop(npu_id, &feature_data, output_height, output_width)?;
    Ok((result, device))
}

#[unsafe(no_mangle)]
pub extern "C" fn adaptive_maxpool2d_fprop(
    npu_id: i32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    output_height: usize,
    output_width: usize,
) -> *mut DLManagedTensorVersioned {
    wrap_rust_fn(adaptive_maxpool2d_fprop_fallible)((
        npu_id,
        features_dltensor_ptr,
        output_height,
        output_width,
    ))
}

fn adaptive_avgpool2d_fprop_fallible<'a>(
    (npu_id, features_dltensor_ptr, output_height, output_width): (
        c_int,
        *const DLManagedTensorVersioned,
        usize,
        usize,
    ),
) -> Result<(QantTensor4<'a>, DLDevice), ToolkitError> {
    dlpack::version_check(&[features_dltensor_ptr])?;
    let device = dlpack::device_check(&[features_dltensor_ptr])?;
    let feature_data = dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;

    let result =
        pooling::adaptive_avgpool2d_fprop(npu_id, &feature_data, output_height, output_width)?;
    Ok((result, device))
}

#[unsafe(no_mangle)]
pub extern "C" fn adaptive_avgpool2d_fprop(
    npu_id: i32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    output_height: usize,
    output_width: usize,
) -> *mut DLManagedTensorVersioned {
    wrap_rust_fn(adaptive_avgpool2d_fprop_fallible)((
        npu_id,
        features_dltensor_ptr,
        output_height,
        output_width,
    ))
}

fn calc_scaled_periodic_nl_fallible_fprop<'a>(
    (npu_id, features_dltensor_ptr, weights_dltensor_ptr): (
        c_int,
        *const DLManagedTensorVersioned,
        *const DLManagedTensorVersioned,
    ),
) -> Result<(QantTensor1<'a>, DLDevice), ToolkitError> {
    dlpack::version_check(&[features_dltensor_ptr, weights_dltensor_ptr])?;
    let device = dlpack::device_check(&[features_dltensor_ptr, weights_dltensor_ptr])?;
    let feature_fdata = dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;
    let weights_fdata = dlpack::dltensor_to_qant_tensor(weights_dltensor_ptr, "weights")?;

    let result =
        crate::non_linear::calc_scaled_periodic_nl(npu_id, &feature_fdata, &weights_fdata)?;
    Ok((result, device))
}

#[unsafe(no_mangle)]
pub extern "C" fn calc_scaled_periodic_nl_fprop(
    npu_id: i32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    bias_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *mut DLManagedTensorVersioned {
    wrap_rust_fn(calc_scaled_periodic_nl_fallible_fprop)((
        npu_id,
        features_dltensor_ptr,
        bias_dltensor_ptr,
    ))
}

fn calc_kan_layer_fallible<'a>(
    (npu_id, features_dltensor_ptr, phis_mtensor_ptr, ampls_dltensor_ptr, ks_dltensor_ptr): (
        c_int,
        *const DLManagedTensorVersioned,
        *const DLManagedTensorVersioned,
        *const DLManagedTensorVersioned,
        *const DLManagedTensorVersioned,
    ),
) -> Result<(QantTensor2<'a>, DLDevice), ToolkitError> {
    dlpack::version_check(&[
        features_dltensor_ptr,
        phis_mtensor_ptr,
        ampls_dltensor_ptr,
        ks_dltensor_ptr,
    ])?;
    let device = dlpack::device_check(&[
        features_dltensor_ptr,
        phis_mtensor_ptr,
        ampls_dltensor_ptr,
        ks_dltensor_ptr,
    ])?;
    let feature_fdata = dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;
    let phis_fdata = dlpack::dltensor_to_qant_tensor(phis_mtensor_ptr, "phis")?;
    let ampls_fdata = dlpack::dltensor_to_qant_tensor(ampls_dltensor_ptr, "amplitudes")?;
    let ks_fdata = dlpack::dltensor_to_qant_tensor::<1>(ks_dltensor_ptr, "ks")?;

    let result = crate::non_linear::calc_kan_layer(
        npu_id,
        &feature_fdata,
        &phis_fdata,
        &ampls_fdata,
        ks_fdata.data(),
    )?;
    Ok((result, device))
}

#[unsafe(no_mangle)]
pub extern "C" fn calc_kan_layer_fprop(
    npu_id: i32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    phis_dltensor_ptr: *const DLManagedTensorVersioned,
    ampls_dltensor_ptr: *const DLManagedTensorVersioned,
    ks_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *mut DLManagedTensorVersioned {
    wrap_rust_fn(calc_kan_layer_fallible)((
        npu_id,
        features_dltensor_ptr,
        phis_dltensor_ptr,
        ampls_dltensor_ptr,
        ks_dltensor_ptr,
    ))
}

fn dft_fallible<'a>(
    (npu_id, data_dltensor_ptr): (c_int, *const DLManagedTensorVersioned),
) -> Result<(QantTensor3<'a>, DLDevice), ToolkitError> {
    dlpack::version_check(&[data_dltensor_ptr])?;
    let device = dlpack::device_check(&[data_dltensor_ptr])?;
    // (n_dfts, dft_size) bcomplex32 -> (n_dfts, dft_size, 2) bf16 with interleaved real and imaginary parts
    let data_fdata = dlpack::dltensor_bcomplex_to_qant_tensor3(data_dltensor_ptr, "data")?;

    let result = crate::dft::dft(npu_id, &data_fdata)?;
    Ok((result, device))
}

#[unsafe(no_mangle)]
pub extern "C" fn dft(
    npu_id: i32,
    data_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *mut DLManagedTensorVersioned {
    // not wrap_rust_fn: the result is returned as a bcomplex32 tensor instead of bf16
    to_boxed_ptr(
        dft_fallible((npu_id, data_dltensor_ptr))
            .and_then(|(result, device)| dlpack::qant_tensor3_to_bcomplex_dltensor(result, device)),
    )
}
