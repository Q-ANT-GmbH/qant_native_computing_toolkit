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
    ffi::{CStr, c_char, c_int, c_uint},
    ptr,
};

use crate::{
    bias,
    conv::{
        self, ConvAttributes, ConvTransposeAttributes, ConvTransposeWeightsHandle,
        ConvWeightsHandle,
    },
    data_structures::{QantTensor, QantTensor1, QantTensor2, QantTensor4},
    errors::ToolkitError,
    global_settings::Value,
    linear,
    memory_management::{self, QantMemoryHandle},
    non_linear, norm,
    pooling::{self, PoolingAttributes},
    qant_driver_import::{self, C_STRING_DEFAULT_LENGTH, PerformanceCounterInfo, QantDataRole},
    utils,
};

mod dlpack;
use dlpack::{DLDevice, DLDeviceType_kDLCPU, DLManagedTensorVersioned};
use half::bf16;

#[unsafe(no_mangle)]
#[allow(clippy::disallowed_macros)]
pub extern "C" fn setup_logging(folder_name: *const c_char, loglevel: c_int) -> c_int {
    if folder_name.is_null() {
        eprintln!("logging folder_name: got nullptr instead of *c_char");
        return -1;
    }
    let folder_name_cstr = unsafe { CStr::from_ptr(folder_name) };
    let folder_name_string = match folder_name_cstr.to_str() {
        Err(_) => {
            eprintln!("cannot read folder_name");
            return -1;
        }
        Ok(folder_name_str) => String::from(folder_name_str),
    };
    let loglvl = match utils::int_to_loglevel(loglevel) {
        Err(e) => {
            eprintln!("{e}");
            return -1;
        }
        Ok(l) => l,
    };
    match qant_driver_import::setup_logging(&folder_name_string, loglvl) {
        Err(e) => {
            eprintln!("{e}");
            -1
        }
        Ok(()) => 0,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_driver_info(npu_id: u32, output: *mut c_char, len: usize) -> c_int {
    qant_driver_import::get_version_ptr_based(npu_id, output, len).code
}

#[unsafe(no_mangle)]
pub extern "C" fn get_sensor_info(npu_id: u32) -> qant_driver_import::SensorInfo {
    qant_driver_import::get_sensor_info(npu_id)
}

#[unsafe(no_mangle)]
pub extern "C" fn get_version_info(npu_id: u32) -> qant_driver_import::QantVersionInfo {
    qant_driver_import::get_version_info(npu_id)
}

#[unsafe(no_mangle)]
pub extern "C" fn get_available_npus(
    idxs_out: *mut u32,
    serials_out: *mut [c_char; C_STRING_DEFAULT_LENGTH],
    capacity: usize,
    out_count: *mut usize,
) -> c_int {
    qant_driver_import::get_available_npus_ptr_based(idxs_out, serials_out, capacity, out_count)
        .code
}

#[unsafe(no_mangle)]
#[allow(clippy::disallowed_macros)]
pub extern "C" fn reset_perf_counter(npu_id: u32) -> c_int {
    match qant_driver_import::reset_perf_counter(npu_id) {
        Ok(_) => 0,
        Err(e) => {
            eprintln!("{e}");
            -1
        }
    }
}

#[unsafe(no_mangle)]
#[allow(clippy::disallowed_macros)]
pub extern "C" fn get_perf_counter(npu_id: u32, out: *mut PerformanceCounterInfo) -> c_int {
    match qant_driver_import::get_perf_counter(npu_id) {
        Ok(c) => {
            unsafe {
                *out = c;
            }
            0
        }
        Err(e) => {
            eprintln!("{e}");
            -1
        }
    }
}

// take a rust function that takes a **tuple of arguments** and returns a result,
// execute it,
// return the dltensor of the value if ok,
// print error and return null if not ok
// if it is with batch it returns a 4d tensor otherwise a 3d tensor
fn wrap_rust_fn<'a, F, Args, const D: usize>(
    func: F,
) -> impl Fn(Args) -> *const DLManagedTensorVersioned
where
    F: Fn(Args) -> Result<(QantTensor<'a, D>, DLDevice), ToolkitError>,
{
    move |args: Args| match func(args) {
        Ok((fdata, device)) => {
            let ret = match dlpack::qant_tensor_to_dltensor(fdata, device) {
                Ok(ret) => ret,
                Err(e) => {
                    #[allow(clippy::disallowed_macros)]
                    {
                        eprintln!("{e}");
                    }
                    return ptr::null();
                }
            };
            Box::into_raw(Box::new(ret))
        }
        Err(e) => {
            #[allow(clippy::disallowed_macros)]
            {
                eprintln!("{e}");
            }
            ptr::null()
        }
    }
}

#[unsafe(no_mangle)]
// must avoid naming conflict with driver init_npu()
pub extern "C" fn init_npu(id: c_uint) -> c_int {
    qant_driver_import::init_npu_error_code(id)
}

#[unsafe(no_mangle)]
pub extern "C" fn release_npu(id: c_uint) -> c_int {
    qant_driver_import::release_npu_error_code(id)
}

#[unsafe(no_mangle)]
#[allow(clippy::disallowed_macros)]
pub extern "C" fn npu_allocate(
    npu_id: u32,
    rows: usize,
    cols: usize,
    data_role: QantDataRole,
) -> *mut QantMemoryHandle {
    match QantMemoryHandle::allocate(npu_id, rows, cols, data_role) {
        Ok(handle) => Box::into_raw(Box::new(handle)),
        Err(e) => {
            eprintln!("{e}");
            std::ptr::null_mut()
        }
    }
}

#[allow(clippy::disallowed_macros)]
fn free_boxed<T>(handle: *mut T, name: &str) -> c_int {
    if handle.is_null() {
        eprintln!("got nullptr instead of *{name}");
        return -1;
    }
    drop(unsafe { Box::from_raw(handle) });
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn npu_free(handle: *mut QantMemoryHandle) -> c_int {
    free_boxed(handle, "QantMemoryHandle")
}

fn npu_upload_fallible(
    npu_id: u32,
    data_dltensor_ptr: *const DLManagedTensorVersioned,
    data_role: QantDataRole,
) -> Result<QantMemoryHandle, ToolkitError> {
    dlpack::version_check(&[data_dltensor_ptr])?;
    let matrix: QantTensor2 = dlpack::dltensor_to_qant_tensor(data_dltensor_ptr, "data")?;
    QantMemoryHandle::from_matrix(npu_id, &matrix, data_role)
}

#[unsafe(no_mangle)]
#[allow(clippy::disallowed_macros)]
pub extern "C" fn npu_upload(
    npu_id: u32,
    data_dltensor_ptr: *const DLManagedTensorVersioned,
    data_role: QantDataRole,
) -> *mut QantMemoryHandle {
    match npu_upload_fallible(npu_id, data_dltensor_ptr, data_role) {
        Ok(handle) => Box::into_raw(Box::new(handle)),
        Err(e) => {
            eprintln!("{e}");
            std::ptr::null_mut()
        }
    }
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
#[allow(clippy::disallowed_macros)]
pub extern "C" fn npu_upload_into(
    handle: *const QantMemoryHandle,
    data_dltensor_ptr: *const DLManagedTensorVersioned,
) -> c_int {
    match npu_upload_into_fallible(handle, data_dltensor_ptr) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("{e}");
            -1
        }
    }
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
pub extern "C" fn npu_download(handle: *const QantMemoryHandle) -> *const DLManagedTensorVersioned {
    wrap_rust_fn(npu_download_fallible)(handle)
}

fn linear_fprop_fallible<'a>(
    (npu_id, features_dltensor_ptr, weights_dltensor_ptr): (
        c_uint,
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
    npu_id: u32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    weights_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *const DLManagedTensorVersioned {
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
        weights_handle.npu_id(),
        &features_data,
        QantDataRole::Features,
    )?;
    let result_handle = QantMemoryHandle::allocate(
        weights_handle.npu_id(),
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
) -> *const DLManagedTensorVersioned {
    wrap_rust_fn(linear_fprop_preloaded_weights_fallible)((features_dltensor_ptr, weights_handle))
}

#[unsafe(no_mangle)]
#[allow(clippy::disallowed_macros)]
pub extern "C" fn linear_fprop_preloaded(
    features_handle: *const QantMemoryHandle,
    weights_handle: *const QantMemoryHandle,
    result_handle: *const QantMemoryHandle,
) -> c_int {
    if features_handle.is_null() || weights_handle.is_null() || result_handle.is_null() {
        eprintln!("got nullptr instead of *QantMemoryHandle");
        return -1;
    }
    // borrow the handles to call mul_matrix_vec_batched_mem, but don't take ownership of them
    let (features_handle, weights_handle, result_handle) =
        unsafe { (&*features_handle, &*weights_handle, &*result_handle) };
    match memory_management::mul_matrix_vec_batched_mem(
        weights_handle,
        features_handle,
        result_handle,
    ) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("{e}");
            -1
        }
    }
}

fn conv_fprop_fallible<'a>(
    (npu_id, features_dltensor_ptr, weights_dltensor_ptr, padding, stride, dilation): (
        c_uint,
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
    npu_id: u32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    weights_dltensor_ptr: *const DLManagedTensorVersioned,
    padding: usize,
    stride: usize,
    dilation: usize,
) -> *const DLManagedTensorVersioned {
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
        c_uint,
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
    npu_id: u32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    weights_dltensor_ptr: *const DLManagedTensorVersioned,
    padding: usize,
    stride: usize,
    dilation: usize,
    output_padding: usize,
) -> *const DLManagedTensorVersioned {
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
    npu_id: u32,
    weights_dltensor_ptr: *const DLManagedTensorVersioned,
) -> Result<ConvWeightsHandle, ToolkitError> {
    dlpack::version_check(&[weights_dltensor_ptr])?;
    let weights_data: QantTensor4 =
        dlpack::dltensor_to_qant_tensor(weights_dltensor_ptr, "weights")?;
    conv::upload_conv_weights(npu_id, &weights_data)
}

#[unsafe(no_mangle)]
#[allow(clippy::disallowed_macros)]
pub extern "C" fn upload_conv_weights(
    npu_id: u32,
    weights_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *mut ConvWeightsHandle {
    match upload_conv_weights_fallible(npu_id, weights_dltensor_ptr) {
        Ok(handle) => Box::into_raw(Box::new(handle)),
        Err(e) => {
            eprintln!("{e}");
            std::ptr::null_mut()
        }
    }
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
) -> *const DLManagedTensorVersioned {
    wrap_rust_fn(conv_fprop_preloaded_weights_fallible)((
        features_dltensor_ptr,
        weights_handle,
        padding,
        stride,
        dilation,
    ))
}

fn upload_conv_transpose_weights_fallible(
    npu_id: u32,
    weights_dltensor_ptr: *const DLManagedTensorVersioned,
) -> Result<ConvTransposeWeightsHandle, ToolkitError> {
    dlpack::version_check(&[weights_dltensor_ptr])?;
    let weights_data: QantTensor4 =
        dlpack::dltensor_to_qant_tensor(weights_dltensor_ptr, "weights")?;
    conv::upload_conv_transpose_weights(npu_id, &weights_data)
}

#[unsafe(no_mangle)]
#[allow(clippy::disallowed_macros)]
pub extern "C" fn upload_conv_transpose_weights(
    npu_id: u32,
    weights_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *mut ConvTransposeWeightsHandle {
    match upload_conv_transpose_weights_fallible(npu_id, weights_dltensor_ptr) {
        Ok(handle) => Box::into_raw(Box::new(handle)),
        Err(e) => {
            eprintln!("{e}");
            std::ptr::null_mut()
        }
    }
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
) -> *const DLManagedTensorVersioned {
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
        c_uint,
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
    npu_id: u32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    bias_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *const DLManagedTensorVersioned {
    wrap_rust_fn(add_bias_fprop_fallible)((npu_id, features_dltensor_ptr, bias_dltensor_ptr))
}

fn add_bias_for_conv2d_fprop_fallible<'a>(
    (npu_id, features_dltensor_ptr, bias_dltensor_ptr): (
        c_uint,
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
    npu_id: u32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    bias_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *const DLManagedTensorVersioned {
    wrap_rust_fn(add_bias_for_conv2d_fprop_fallible)((
        npu_id,
        features_dltensor_ptr,
        bias_dltensor_ptr,
    ))
}

fn mul_npu_fallible<'a>(
    (npu_id, us_dltensor_ptr, vs_dltensor_ptr): (
        c_uint,
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
    npu_id: u32,
    us_dltensor_ptr: *const DLManagedTensorVersioned,
    vs_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *const DLManagedTensorVersioned {
    wrap_rust_fn(mul_npu_fallible)((npu_id, us_dltensor_ptr, vs_dltensor_ptr))
}

fn relu_fprop_fallible<'a>(
    (npu_id, features_dltensor_ptr): (c_uint, *const DLManagedTensorVersioned),
) -> Result<(QantTensor1<'a>, DLDevice), ToolkitError> {
    dlpack::version_check(&[features_dltensor_ptr])?;
    let device = dlpack::device_check(&[features_dltensor_ptr])?;
    let feature_fdata = dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;
    Ok((non_linear::relu_fprop(npu_id, &feature_fdata), device))
}

#[unsafe(no_mangle)]
pub extern "C" fn relu_fprop(
    npu_id: u32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *const DLManagedTensorVersioned {
    wrap_rust_fn(relu_fprop_fallible)((npu_id, features_dltensor_ptr))
}

fn sigmoid_fprop_fallible<'a>(
    (npu_id, features_dltensor_ptr): (c_uint, *const DLManagedTensorVersioned),
) -> Result<(QantTensor1<'a>, DLDevice), ToolkitError> {
    dlpack::version_check(&[features_dltensor_ptr])?;
    let device = dlpack::device_check(&[features_dltensor_ptr])?;
    let feature_fdata = dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;
    Ok((non_linear::sigmoid_fprop(npu_id, &feature_fdata), device))
}

#[unsafe(no_mangle)]
pub extern "C" fn sigmoid_fprop(
    npu_id: u32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *const DLManagedTensorVersioned {
    wrap_rust_fn(sigmoid_fprop_fallible)((npu_id, features_dltensor_ptr))
}

fn softmax_fprop_fallible<'a>(
    (npu_id, features_dltensor_ptr): (c_uint, *const DLManagedTensorVersioned),
) -> Result<(QantTensor2<'a>, DLDevice), ToolkitError> {
    dlpack::version_check(&[features_dltensor_ptr])?;
    let device = dlpack::device_check(&[features_dltensor_ptr])?;
    let feature_fdata = dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;
    let result = non_linear::softmax_fprop(npu_id, &feature_fdata)?;
    Ok((result, device))
}

#[unsafe(no_mangle)]
pub extern "C" fn softmax_fprop(
    npu_id: u32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *const DLManagedTensorVersioned {
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
        c_uint,
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
    npu_id: u32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    means_dltensor_ptr: *const DLManagedTensorVersioned,
    variances_dltensor_ptr: *const DLManagedTensorVersioned,
    weights_dltensor_ptr: *const DLManagedTensorVersioned,
    bias_dltensor_ptr: *const DLManagedTensorVersioned,
    eps: f32,
) -> *const DLManagedTensorVersioned {
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
        c_uint,
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
    npu_id: u32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    kernel_height: usize,
    kernel_width: usize,
    padding: usize,
    stride: usize,
) -> *const DLManagedTensorVersioned {
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
        c_uint,
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
    npu_id: u32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    kernel_height: usize,
    kernel_width: usize,
    padding: usize,
    stride: usize,
    count_include_pad: bool,
) -> *const DLManagedTensorVersioned {
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
        c_uint,
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
    npu_id: u32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    output_height: usize,
    output_width: usize,
) -> *const DLManagedTensorVersioned {
    wrap_rust_fn(adaptive_maxpool2d_fprop_fallible)((
        npu_id,
        features_dltensor_ptr,
        output_height,
        output_width,
    ))
}

fn adaptive_avgpool2d_fprop_fallible<'a>(
    (npu_id, features_dltensor_ptr, output_height, output_width): (
        c_uint,
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
    npu_id: u32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    output_height: usize,
    output_width: usize,
) -> *const DLManagedTensorVersioned {
    wrap_rust_fn(adaptive_avgpool2d_fprop_fallible)((
        npu_id,
        features_dltensor_ptr,
        output_height,
        output_width,
    ))
}

fn calc_scaled_periodic_nl_fallible_fprop<'a>(
    (npu_id, features_dltensor_ptr, weights_dltensor_ptr): (
        c_uint,
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
    npu_id: u32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    bias_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *const DLManagedTensorVersioned {
    wrap_rust_fn(calc_scaled_periodic_nl_fallible_fprop)((
        npu_id,
        features_dltensor_ptr,
        bias_dltensor_ptr,
    ))
}

fn calc_kan_layer_fallible<'a>(
    (npu_id, features_dltensor_ptr, phis_mtensor_ptr, ampls_dltensor_ptr, ks_dltensor_ptr): (
        c_uint,
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
    npu_id: u32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    phis_dltensor_ptr: *const DLManagedTensorVersioned,
    ampls_dltensor_ptr: *const DLManagedTensorVersioned,
    ks_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *const DLManagedTensorVersioned {
    wrap_rust_fn(calc_kan_layer_fallible)((
        npu_id,
        features_dltensor_ptr,
        phis_dltensor_ptr,
        ampls_dltensor_ptr,
        ks_dltensor_ptr,
    ))
}
