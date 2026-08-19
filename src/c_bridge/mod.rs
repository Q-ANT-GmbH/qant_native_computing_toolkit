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
    ffi::{CStr, c_char, c_int, c_uint, c_void},
    mem,
    ptr::null,
};

use crate::{
    bias::{
        add_bias_for_conv2d_fprop as rust_add_bias_for_conv2d_fprop,
        add_bias_fprop as rust_add_bias_fprop,
    },
    c_bridge::dlpack::{
        DLDTYPE_BF16, DLDTYPE_F32, DLDTYPE_I16, DLDataType, DLPACK_MAJOR_VERSION,
        DLPACK_MINOR_VERSION, DLPackVersion, device_check, version_check,
    },
    conv::{
        ConvAttributes, ConvTransposeAttributes, conv_fprop as rust_conv_fprop,
        conv_transpose_fprop as rust_conv_transpose_fprop,
    },
    data_structures::{QantTensor, QantTensor1, QantTensor2, QantTensor4},
    errors::ToolkitError,
    global_settings::Value,
    linear::linear_fprop as rust_linear_fprop,
    mul_elementwise_f32 as rust_mul_f32,
    non_linear::{
        relu_fprop as rust_relu_fprop, sigmoid_fprop as rust_sigmoid_fprop,
        softmax_fprop as rust_softmax_fprop,
    },
    norm::batchnorm2d_fprop as rust_batchnorm2d_fprop,
    pooling::{
        PoolingAttributes, adaptive_avgpool2d_fprop as rust_adaptive_avgpool2d_fprop,
        adaptive_maxpool2d_fprop as rust_adaptive_maxpool2d_fprop,
        avgpool2d_fprop as rust_avgpool2d_fprop, maxpool2d_fprop as rust_maxpool2d_fprop,
    },
    qant_driver_import::{
        self, C_STRING_DEFAULT_LENGTH, PerformanceCounterInfo, driver_get_available_npus_ptr_based,
        driver_get_perf_counter, driver_reset_perf_counter,
    },
    setup_logging as rust_setup_logging,
    utils::int_to_loglevel,
};

mod dlpack;
use dlpack::{DLDevice, DLManagedTensorVersioned, DLTensor};
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
    let loglvl = match int_to_loglevel(loglevel) {
        Err(e) => {
            eprintln!("{e}");
            return -1;
        }
        Ok(l) => l,
    };
    match rust_setup_logging(&folder_name_string, loglvl) {
        Err(e) => {
            eprintln!("{e}");
            -1
        }
        Ok(()) => 0,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_driver_info(npu_id: u32, output: *mut c_char, len: usize) -> c_int {
    qant_driver_import::driver_get_version_ptr_based(npu_id, output, len).code
}

#[unsafe(no_mangle)]
pub extern "C" fn get_sensor_info(npu_id: u32) -> qant_driver_import::SensorInfo {
    qant_driver_import::driver_get_sensor_info(npu_id)
}

#[unsafe(no_mangle)]
pub extern "C" fn get_version_info(npu_id: u32) -> qant_driver_import::QantVersionInfo {
    qant_driver_import::driver_get_version_info(npu_id)
}

#[unsafe(no_mangle)]
pub extern "C" fn get_available_npus(
    idxs_out: *mut u32,
    serials_out: *mut [c_char; C_STRING_DEFAULT_LENGTH],
    capacity: usize,
    out_count: *mut usize,
) -> c_int {
    driver_get_available_npus_ptr_based(idxs_out, serials_out, capacity, out_count).code
}

#[unsafe(no_mangle)]
#[allow(clippy::disallowed_macros)]
pub extern "C" fn reset_perf_counter(npu_id: u32) -> c_int {
    match driver_reset_perf_counter(npu_id) {
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
    match driver_get_perf_counter(npu_id) {
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

#[unsafe(no_mangle)]
#[allow(clippy::disallowed_macros)]
pub extern "C" fn mul_npu_f32(
    npu_id: u32,
    us_dltensor_ptr: *const DLManagedTensorVersioned,
    vs_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *const DLManagedTensorVersioned {
    if us_dltensor_ptr.is_null() || vs_dltensor_ptr.is_null() {
        eprintln!("got nullptr instead of *DLManagedTensorVersioned");
        return null();
    }

    match version_check(&[us_dltensor_ptr, vs_dltensor_ptr]) {
        Ok(_) => (),
        Err(e) => {
            eprintln!("{e}");
            return null();
        }
    }

    let device = match device_check(&[us_dltensor_ptr, vs_dltensor_ptr]) {
        Ok(device) => device,
        Err(e) => {
            eprintln!("{e}");
            return null();
        }
    };
    let us_dltensor = unsafe { *us_dltensor_ptr };
    let vs_dltensor = unsafe { *vs_dltensor_ptr };
    let us_tensor = us_dltensor.dl_tensor;
    let vs_tensor = vs_dltensor.dl_tensor;

    if us_tensor.ndim != 1 || vs_tensor.ndim != 1 {
        eprintln!("us and vs must be 1d");
        return null();
    }
    if us_tensor.dtype != DLDTYPE_F32 || vs_tensor.dtype != DLDTYPE_F32 {
        eprintln!("us and vs must be {DLDTYPE_F32:?}");
        return null();
    }

    let len_u = unsafe { *us_tensor.shape.offset(0) } as usize;
    let len_v = unsafe { *vs_tensor.shape.offset(0) } as usize;

    let us = unsafe { std::slice::from_raw_parts(us_tensor.data as *const f32, len_u) };
    let vs = unsafe { std::slice::from_raw_parts(vs_tensor.data as *const f32, len_v) };

    let ws = match rust_mul_f32(npu_id, us, vs) {
        Ok(vals) => vals,
        Err(e) => {
            eprintln!("{e}");
            return null();
        }
    };

    let ws_shape = vec![ws.len() as dlpack::ShapeType];
    let ws_tensor = DLTensor {
        data: ws.as_ptr() as *mut c_void,
        device,
        ndim: 1,
        dtype: DLDTYPE_F32,
        shape: ws_shape.as_ptr() as *mut dlpack::ShapeType,
        strides: std::ptr::null::<dlpack::ShapeType>() as *mut dlpack::ShapeType,
        byte_offset: 0,
    };

    // ws_tensor is the wrapper that makes the data dlpack compliant
    // ws (the rust vector) is the actual data owner (i.e. "manager")
    let ret = DLManagedTensorVersioned {
        version: DLPackVersion {
            major: DLPACK_MAJOR_VERSION,
            minor: DLPACK_MINOR_VERSION,
        },
        flags: 0,
        dl_tensor: ws_tensor,
        manager_ctx: Box::into_raw(Box::new(ws)) as *mut c_void,
        deleter: Some(dlpack::delete_vector_managed_tensor_f32),
    };

    // we forget about the vectors here and trust that the caller
    // will invoke the deleter to manually destruct the resources
    // ws is automatically forgotten by Box::into_raw()
    #[allow(clippy::mem_forget)]
    mem::forget(ws_shape);

    Box::into_raw(Box::new(ret))
}

// take a rust function that takes a **tuple of arguments** and returns a result,
// execute it,
// return the dltensor of the value if ok,
// print error and return null if not ok
// if it is with batch it returns a 4d tensor otherwise a 3d tensor
fn wrap_rust_fn<'a, F, Args, const D: usize>(
    func: F,
    dtype: DLDataType,
) -> impl Fn(Args) -> *const DLManagedTensorVersioned
where
    F: Fn(Args) -> Result<(QantTensor<'a, D>, DLDevice), ToolkitError>,
{
    move |args: Args| match func(args) {
        Ok((fdata, device)) => {
            let ret = match dlpack::qant_tensor_to_dltensor(fdata, device, dtype) {
                Ok(ret) => ret,
                Err(e) => {
                    #[allow(clippy::disallowed_macros)]
                    {
                        eprintln!("{e}");
                    }
                    return null();
                }
            };
            Box::into_raw(Box::new(ret))
        }
        Err(e) => {
            #[allow(clippy::disallowed_macros)]
            {
                eprintln!("{e}");
            }
            null()
        }
    }
}

#[unsafe(no_mangle)]
// must avoid naming conflict with driver init_npu()
pub extern "C" fn init_npu(id: c_uint) -> c_int {
    qant_driver_import::driver_init_npu_error_code(id)
}

#[unsafe(no_mangle)]
pub extern "C" fn release_npu(id: c_uint) -> c_int {
    qant_driver_import::driver_release_npu_error_code(id)
}

fn linear_fprop_fallible<'a>(
    (npu_id, features_dltensor_ptr, weights_dltensor_ptr): (
        c_uint,
        *const DLManagedTensorVersioned,
        *const DLManagedTensorVersioned,
    ),
) -> Result<(QantTensor2<'a>, DLDevice), ToolkitError> {
    version_check(&[features_dltensor_ptr, weights_dltensor_ptr])?;
    let device = device_check(&[features_dltensor_ptr, weights_dltensor_ptr])?;
    let feature_data = dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;
    let filter_data = dlpack::dltensor_to_qant_tensor(weights_dltensor_ptr, "weights")?;
    let result = rust_linear_fprop(npu_id, &feature_data, &filter_data)?;
    Ok((result, device))
}

#[unsafe(no_mangle)]
pub extern "C" fn linear_fprop(
    npu_id: u32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    weights_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *const DLManagedTensorVersioned {
    wrap_rust_fn(linear_fprop_fallible, DLDTYPE_BF16)((
        npu_id,
        features_dltensor_ptr,
        weights_dltensor_ptr,
    ))
}

fn conv_fprop_fallible<'a>(
    (npu_id, features_dltensor_ptr, kernels_dltensor_ptr, padding, stride, dilation): (
        c_uint,
        *const DLManagedTensorVersioned,
        *const DLManagedTensorVersioned,
        usize,
        usize,
        usize,
    ),
) -> Result<(QantTensor4<'a>, DLDevice), ToolkitError> {
    version_check(&[features_dltensor_ptr, kernels_dltensor_ptr])?;
    let device = device_check(&[features_dltensor_ptr, kernels_dltensor_ptr])?;
    let feature_data = dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;
    let filter_data = dlpack::dltensor_to_qant_tensor(kernels_dltensor_ptr, "filters")?;

    let conv_attrs = ConvAttributes {
        padding,
        stride,
        dilation,
    };

    let result = rust_conv_fprop(npu_id, &feature_data, &filter_data, conv_attrs)?;
    Ok((result, device))
}

#[unsafe(no_mangle)]
pub extern "C" fn conv_fprop(
    npu_id: u32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    kernels_dltensor_ptr: *const DLManagedTensorVersioned,
    padding: usize,
    stride: usize,
    dilation: usize,
) -> *const DLManagedTensorVersioned {
    wrap_rust_fn(conv_fprop_fallible, DLDTYPE_BF16)((
        npu_id,
        features_dltensor_ptr,
        kernels_dltensor_ptr,
        padding,
        stride,
        dilation,
    ))
}

fn conv_transpose_fprop_fallible<'a>(
    (
        npu_id,
        features_dltensor_ptr,
        kernels_dltensor_ptr,
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
    version_check(&[features_dltensor_ptr, kernels_dltensor_ptr])?;
    let device = device_check(&[features_dltensor_ptr, kernels_dltensor_ptr])?;
    let feature_data = dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;
    let filter_data = dlpack::dltensor_to_qant_tensor(kernels_dltensor_ptr, "filters")?;
    let conv_attrs = ConvTransposeAttributes {
        padding,
        stride,
        dilation,
        output_padding,
    };

    let result = rust_conv_transpose_fprop(npu_id, &feature_data, &filter_data, conv_attrs)?;
    Ok((result, device))
}

#[unsafe(no_mangle)]
pub extern "C" fn conv_transpose_fprop(
    npu_id: u32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    kernels_dltensor_ptr: *const DLManagedTensorVersioned,
    padding: usize,
    stride: usize,
    dilation: usize,
    output_padding: usize,
) -> *const DLManagedTensorVersioned {
    wrap_rust_fn(conv_transpose_fprop_fallible, DLDTYPE_BF16)((
        npu_id,
        features_dltensor_ptr,
        kernels_dltensor_ptr,
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
    version_check(&[features_dltensor_ptr, bias_dltensor_ptr])?;
    let device = device_check(&[features_dltensor_ptr, bias_dltensor_ptr])?;
    let feature_fdata = dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;
    let bias_fdata = dlpack::dltensor_to_qant_tensor(bias_dltensor_ptr, "bias")?;

    let ret = rust_add_bias_fprop(npu_id, &feature_fdata, &bias_fdata)?;

    Ok((ret, device))
}

#[unsafe(no_mangle)]
pub extern "C" fn add_bias_fprop(
    npu_id: u32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    bias_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *const DLManagedTensorVersioned {
    wrap_rust_fn(add_bias_fprop_fallible, DLDTYPE_BF16)((
        npu_id,
        features_dltensor_ptr,
        bias_dltensor_ptr,
    ))
}

fn add_bias_for_conv2d_fprop_fallible<'a>(
    (npu_id, features_dltensor_ptr, bias_dltensor_ptr): (
        c_uint,
        *const DLManagedTensorVersioned,
        *const DLManagedTensorVersioned,
    ),
) -> Result<(QantTensor4<'a>, DLDevice), ToolkitError> {
    version_check(&[features_dltensor_ptr, bias_dltensor_ptr])?;
    let device = device_check(&[features_dltensor_ptr, bias_dltensor_ptr])?;
    let feature_fdata = dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;
    let bias_fdata = dlpack::dltensor_to_qant_tensor(bias_dltensor_ptr, "bias")?;

    let ret = rust_add_bias_for_conv2d_fprop(npu_id, &feature_fdata, &bias_fdata)?;

    Ok((ret, device))
}

#[unsafe(no_mangle)]
pub extern "C" fn add_bias_for_conv2d_fprop(
    npu_id: u32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    bias_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *const DLManagedTensorVersioned {
    wrap_rust_fn(add_bias_for_conv2d_fprop_fallible, DLDTYPE_BF16)((
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
    version_check(&[us_dltensor_ptr, vs_dltensor_ptr])?;
    let device = device_check(&[us_dltensor_ptr, vs_dltensor_ptr])?;
    let us_fdata = dlpack::dltensor_to_qant_tensor(us_dltensor_ptr, "us")?;
    let vs_fdata = dlpack::dltensor_to_qant_tensor(vs_dltensor_ptr, "vs")?;

    let result = crate::mul_elementwise_bf16(npu_id, &us_fdata, &vs_fdata)?;
    Ok((result, device))
}

#[unsafe(no_mangle)]
pub extern "C" fn mul_npu(
    npu_id: u32,
    us_dltensor_ptr: *const DLManagedTensorVersioned,
    vs_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *const DLManagedTensorVersioned {
    wrap_rust_fn(mul_npu_fallible, DLDTYPE_BF16)((npu_id, us_dltensor_ptr, vs_dltensor_ptr))
}

#[deprecated]
fn mul_npu_i16_fallible<'a>(
    (npu_id, us_dltensor_ptr, vs_dltensor_ptr): (
        c_uint,
        *const DLManagedTensorVersioned,
        *const DLManagedTensorVersioned,
    ),
) -> Result<(QantTensor1<'a>, DLDevice), ToolkitError> {
    version_check(&[us_dltensor_ptr, vs_dltensor_ptr])?;
    let device = device_check(&[us_dltensor_ptr, vs_dltensor_ptr])?;
    let us_fdata = dlpack::dltensor_to_qant_tensor(us_dltensor_ptr, "us")?;
    let vs_fdata = dlpack::dltensor_to_qant_tensor(vs_dltensor_ptr, "vs")?;
    let result = crate::mul_elementwise_bf16(npu_id, &us_fdata, &vs_fdata)?;
    Ok((result, device))
}

#[unsafe(no_mangle)]
#[deprecated]
pub extern "C" fn mul_npu_i16(
    npu_id: u32,
    us_dltensor_ptr: *const DLManagedTensorVersioned,
    vs_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *const DLManagedTensorVersioned {
    #[allow(deprecated)]
    wrap_rust_fn(mul_npu_i16_fallible, DLDTYPE_I16)((npu_id, us_dltensor_ptr, vs_dltensor_ptr))
}

fn relu_fprop_fallible<'a>(
    (npu_id, features_dltensor_ptr): (c_uint, *const DLManagedTensorVersioned),
) -> Result<(QantTensor1<'a>, DLDevice), ToolkitError> {
    version_check(&[features_dltensor_ptr])?;
    let device = device_check(&[features_dltensor_ptr])?;
    let feature_fdata = dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;
    Ok((rust_relu_fprop(npu_id, &feature_fdata), device))
}

#[unsafe(no_mangle)]
pub extern "C" fn relu_fprop(
    npu_id: u32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *const DLManagedTensorVersioned {
    wrap_rust_fn(relu_fprop_fallible, DLDTYPE_BF16)((npu_id, features_dltensor_ptr))
}

fn sigmoid_fprop_fallible<'a>(
    (npu_id, features_dltensor_ptr): (c_uint, *const DLManagedTensorVersioned),
) -> Result<(QantTensor1<'a>, DLDevice), ToolkitError> {
    version_check(&[features_dltensor_ptr])?;
    let device = device_check(&[features_dltensor_ptr])?;
    let feature_fdata = dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;
    Ok((rust_sigmoid_fprop(npu_id, &feature_fdata), device))
}

#[unsafe(no_mangle)]
pub extern "C" fn sigmoid_fprop(
    npu_id: u32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *const DLManagedTensorVersioned {
    wrap_rust_fn(sigmoid_fprop_fallible, DLDTYPE_BF16)((npu_id, features_dltensor_ptr))
}

fn softmax_fprop_fallible<'a>(
    (npu_id, features_dltensor_ptr): (c_uint, *const DLManagedTensorVersioned),
) -> Result<(QantTensor2<'a>, DLDevice), ToolkitError> {
    version_check(&[features_dltensor_ptr])?;
    let device = device_check(&[features_dltensor_ptr])?;
    let feature_fdata = dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;
    let result = rust_softmax_fprop(npu_id, &feature_fdata)?;
    Ok((result, device))
}

#[unsafe(no_mangle)]
pub extern "C" fn softmax_fprop(
    npu_id: u32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
) -> *const DLManagedTensorVersioned {
    wrap_rust_fn(softmax_fprop_fallible, DLDTYPE_BF16)((npu_id, features_dltensor_ptr))
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
    version_check(&[
        features_dltensor_ptr,
        means_dltensor_ptr,
        variances_dltensor_ptr,
        weights_dltensor_ptr,
        bias_dltensor_ptr,
    ])?;
    let device = device_check(&[
        features_dltensor_ptr,
        means_dltensor_ptr,
        variances_dltensor_ptr,
        weights_dltensor_ptr,
        bias_dltensor_ptr,
    ])?;
    let result = rust_batchnorm2d_fprop(
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
    wrap_rust_fn(batchnorm2d_fprop_fallible, DLDTYPE_BF16)((
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
    version_check(&[features_dltensor_ptr])?;
    let device = device_check(&[features_dltensor_ptr])?;
    let feature_data = dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;

    let pool_attrs = PoolingAttributes {
        kernel_height,
        kernel_width,
        padding,
        stride,
        count_include_pad: false,
    };

    let result = rust_maxpool2d_fprop(npu_id, &feature_data, pool_attrs)?;
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
    wrap_rust_fn(maxpool2d_fprop_fallible, DLDTYPE_BF16)((
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
    version_check(&[features_dltensor_ptr])?;
    let device = device_check(&[features_dltensor_ptr])?;
    let feature_data = dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;

    let pool_attrs = PoolingAttributes {
        kernel_height,
        kernel_width,
        padding,
        stride,
        count_include_pad,
    };

    let result = rust_avgpool2d_fprop(npu_id, &feature_data, pool_attrs)?;
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
    wrap_rust_fn(avgpool2d_fprop_fallible, DLDTYPE_BF16)((
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
    version_check(&[features_dltensor_ptr])?;
    let device = device_check(&[features_dltensor_ptr])?;
    let feature_data = dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;

    let result = rust_adaptive_maxpool2d_fprop(npu_id, &feature_data, output_height, output_width)?;
    Ok((result, device))
}

#[unsafe(no_mangle)]
pub extern "C" fn adaptive_maxpool2d_fprop(
    npu_id: u32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    output_height: usize,
    output_width: usize,
) -> *const DLManagedTensorVersioned {
    wrap_rust_fn(adaptive_maxpool2d_fprop_fallible, DLDTYPE_BF16)((
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
    version_check(&[features_dltensor_ptr])?;
    let device = device_check(&[features_dltensor_ptr])?;
    let feature_data = dlpack::dltensor_to_qant_tensor(features_dltensor_ptr, "features")?;

    let result = rust_adaptive_avgpool2d_fprop(npu_id, &feature_data, output_height, output_width)?;
    Ok((result, device))
}

#[unsafe(no_mangle)]
pub extern "C" fn adaptive_avgpool2d_fprop(
    npu_id: u32,
    features_dltensor_ptr: *const DLManagedTensorVersioned,
    output_height: usize,
    output_width: usize,
) -> *const DLManagedTensorVersioned {
    wrap_rust_fn(adaptive_avgpool2d_fprop_fallible, DLDTYPE_BF16)((
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
    version_check(&[features_dltensor_ptr, weights_dltensor_ptr])?;
    let device = device_check(&[features_dltensor_ptr, weights_dltensor_ptr])?;
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
    wrap_rust_fn(calc_scaled_periodic_nl_fallible_fprop, DLDTYPE_BF16)((
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
    version_check(&[
        features_dltensor_ptr,
        phis_mtensor_ptr,
        ampls_dltensor_ptr,
        ks_dltensor_ptr,
    ])?;
    let device = device_check(&[
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
    wrap_rust_fn(calc_kan_layer_fallible, DLDTYPE_BF16)((
        npu_id,
        features_dltensor_ptr,
        phis_dltensor_ptr,
        ampls_dltensor_ptr,
        ks_dltensor_ptr,
    ))
}
