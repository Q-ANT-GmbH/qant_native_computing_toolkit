#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

include!(concat!(env!("OUT_DIR"), "/dlpack_bindings.rs"));
use std::{ffi::c_void, slice};

use half::bf16;

use crate::{
    data_structures::{QantTensor, QantTensor3},
    errors::ToolkitError,
    toolkit_error,
};

pub type ShapeType = i64;

pub const DLDTYPE_BF16: DLDataType = DLDataType {
    code: DLDataTypeCode_kDLBfloat as u8,
    bits: 16,
    lanes: 1,
};

/// Complex number made of two bfloat16 values (real, imaginary), stored interleaved.
pub const DLDTYPE_BCOMPLEX32: DLDataType = DLDataType {
    code: DLDataTypeCode_kDLBcomplex as u8,
    bits: 32,
    lanes: 1,
};

impl PartialEq for DLDataType {
    fn eq(&self, other: &Self) -> bool {
        self.code == other.code && self.bits == other.bits && self.lanes == other.lanes
    }
}

impl PartialEq for DLDevice {
    fn eq(&self, other: &Self) -> bool {
        self.device_id == other.device_id && self.device_type == other.device_type
    }
}

fn delete_feature_managed_tensor<const D: usize>(self_: *mut DLManagedTensorVersioned) {
    // bringing all data managing objects in scope
    let dltensor = *unsafe { Box::from_raw(self_) };

    // manager is the underlying QantTensor
    let fdata = *unsafe { Box::from_raw(dltensor.manager_ctx as *mut QantTensor<D>) };

    // clean up the shape array
    let ndim = dltensor.dl_tensor.ndim as usize;
    let shape_vec = unsafe { Vec::from_raw_parts(dltensor.dl_tensor.shape, ndim, ndim) };

    // explicitly drop (optional in Rust, but good for clarity)
    std::mem::drop(fdata);
    std::mem::drop(shape_vec);
}

#[unsafe(no_mangle)]
pub extern "C" fn delete_feature_managed_tensor1(self_: *mut DLManagedTensorVersioned) {
    delete_feature_managed_tensor::<1>(self_);
}

#[unsafe(no_mangle)]
pub extern "C" fn delete_feature_managed_tensor2(self_: *mut DLManagedTensorVersioned) {
    delete_feature_managed_tensor::<2>(self_);
}

#[unsafe(no_mangle)]
pub extern "C" fn delete_feature_managed_tensor3(self_: *mut DLManagedTensorVersioned) {
    delete_feature_managed_tensor::<3>(self_);
}

#[unsafe(no_mangle)]
pub extern "C" fn delete_feature_managed_tensor4(self_: *mut DLManagedTensorVersioned) {
    delete_feature_managed_tensor::<4>(self_);
}

/// Checks that all tensors have the same major version as the toolkit library.
///
/// # Returns
///
/// * `Ok()` - If the major version of all tensors aligns with the major version of the toolkit library.
/// * `Err(ToolkitError)` - If any pointer is null or if any tensor has an incompatible major version.
pub fn version_check(tensors: &[*const DLManagedTensorVersioned]) -> Result<(), ToolkitError> {
    for &tensor_ptr in tensors {
        if tensor_ptr.is_null() {
            return Err(ToolkitError::from_str(
                "got nullptr instead of *DLManagedTensorVersioned",
            ));
        }

        let major_version = unsafe { (*tensor_ptr).version.major };
        if major_version != DLPACK_MAJOR_VERSION {
            return Err(toolkit_error!(
                "Major version of dlpack tensor does not align with toolkit library, got version {major_version}, expected major version {DLPACK_MAJOR_VERSION}"
            ));
        }
    }
    Ok(())
}

/// Checks that all tensors are on the same device.
///
/// Iterates through a slice of raw pointers to `DLManagedTensorVersioned` and verifies that
/// all tensors are on the same `DLDevice`. Returns the common device if successful.
///
/// # Returns
///
/// * `Ok(DLDevice)` - The device on which all tensors are located if they match.
/// * `Err(ToolkitError)` - If any pointer is null or if tensors are on different devices.
pub fn device_check(tensors: &[*const DLManagedTensorVersioned]) -> Result<DLDevice, ToolkitError> {
    let mut device: Option<DLDevice> = None;

    for &tensor_ptr in tensors {
        if tensor_ptr.is_null() {
            return Err(ToolkitError::from_str(
                "got nullptr instead of *DLManagedTensorVersioned",
            ));
        }

        unsafe {
            let current_device = (*tensor_ptr).dl_tensor.device;

            match device {
                None => device = Some(current_device),
                Some(prev_device) => {
                    if prev_device != current_device {
                        return Err(toolkit_error!(
                            "Device mismatch: Not all tensors are on the same device, got {prev_device:?} and {current_device:?}",
                        ));
                    }
                }
            }
        }
    }

    device.ok_or(ToolkitError::from_str(
        "Device mismatch: Not all tensors are on the same device",
    ))
}

fn to_usize_array<const D: usize>(shape: &[i64]) -> Result<[usize; D], ToolkitError> {
    if shape.len() != D {
        return Err(toolkit_error!(
            "features tensor must be {D}-dimensional, got {0}",
            shape.len()
        ));
    }

    let mut fixed = [0_usize; D];
    for (i, val) in shape.iter().enumerate() {
        fixed[i] = *val as usize;
    }

    Ok(fixed)
}

/// Validates a DLPack tensor (non-null, `D` dimensions, `dtype`) and returns its shape
/// together with its data reinterpreted as a flat bf16 slice of `bf16_per_element` values per element.
fn dltensor_to_bf16_slice<'a, const D: usize>(
    dltensor_ptr: *const DLManagedTensorVersioned,
    name: &str,
    dtype: DLDataType,
    bf16_per_element: usize,
) -> Result<(&'a [bf16], [usize; D]), ToolkitError> {
    if dltensor_ptr.is_null() {
        return Err(ToolkitError::from_str(
            "got nullpointer instead of *DLManagedTensorVersioned",
        ));
    }
    let dltensor = unsafe { *dltensor_ptr };
    let tensor = dltensor.dl_tensor;
    let ndim = tensor.ndim as usize;

    if ndim != D {
        return Err(toolkit_error!(
            "{name} must be {D}-dimensional, got {ndim}-dimensional"
        ));
    }

    let shape = unsafe { slice::from_raw_parts(tensor.shape as *const ShapeType, ndim) };
    let shape_fixed: [usize; D] = to_usize_array::<D>(shape)?;

    let len_data: ShapeType = shape.iter().copied().product();
    if tensor.dtype != dtype {
        return Err(toolkit_error!(
            "Invalid dtype, {name} has dtype {0:?}, only {dtype:?} supported",
            tensor.dtype
        ));
    }
    let data: &[bf16] = unsafe {
        slice::from_raw_parts(
            tensor.data as *const bf16,
            len_data as usize * bf16_per_element,
        )
    };
    Ok((data, shape_fixed))
}

pub fn dltensor_to_qant_tensor<'a, const D: usize>(
    dltensor_ptr: *const DLManagedTensorVersioned,
    name: &str,
) -> Result<QantTensor<'a, D>, ToolkitError> {
    let (data, shape) = dltensor_to_bf16_slice::<D>(dltensor_ptr, name, DLDTYPE_BF16, 1)?;
    QantTensor::new(data, shape)
}

/// Converts a 2D bcomplex32 DLPack tensor of shape `(n, m)` into a bf16 `QantTensor3` of shape
/// `(n, m, 2)` holding the interleaved real and imaginary parts (zero-copy).
pub fn dltensor_bcomplex_to_qant_tensor3<'a>(
    dltensor_ptr: *const DLManagedTensorVersioned,
    name: &str,
) -> Result<QantTensor3<'a>, ToolkitError> {
    let (data, [n, m]) = dltensor_to_bf16_slice::<2>(dltensor_ptr, name, DLDTYPE_BCOMPLEX32, 2)?;
    QantTensor::new(data, [n, m, 2])
}

pub fn qant_tensor_to_dltensor<const D: usize>(
    qant_tensor: QantTensor<D>,
    device: DLDevice,
) -> Result<DLManagedTensorVersioned, ToolkitError> {
    let shape: Vec<ShapeType> = qant_tensor
        .shape()
        .iter()
        .map(|&x| x as ShapeType)
        .collect();
    qant_tensor_to_dltensor_with(qant_tensor, device, DLDTYPE_BF16, shape)
}

/// Converts a bf16 `QantTensor3` of shape `(n, m, 2)` (interleaved real and imaginary parts)
/// into a 2D bcomplex32 DLPack tensor of shape `(n, m)` (zero-copy).
pub fn qant_tensor3_to_bcomplex_dltensor(
    qant_tensor: QantTensor3,
    device: DLDevice,
) -> Result<DLManagedTensorVersioned, ToolkitError> {
    let [n, m, re_im] = *qant_tensor.shape();
    if re_im != 2 {
        return Err(toolkit_error!(
            "last dimension must hold real and imaginary part (size 2), got {re_im}"
        ));
    }
    let shape: Vec<ShapeType> = vec![n as ShapeType, m as ShapeType];
    qant_tensor_to_dltensor_with(qant_tensor, device, DLDTYPE_BCOMPLEX32, shape)
}

/// Wraps `qant_tensor` into a DLPack tensor with the given `dtype` and `shape`.
/// `shape` must describe the same memory as the tensor; it may differ from the tensor's own shape
/// (e.g. when exposing bf16 pairs as complex elements). The deleter frees the shape using `ndim`.
fn qant_tensor_to_dltensor_with<const D: usize>(
    qant_tensor: QantTensor<D>,
    device: DLDevice,
    dtype: DLDataType,
    shape: Vec<ShapeType>,
) -> Result<DLManagedTensorVersioned, ToolkitError> {
    // the deleter rebuilds the shape vec with capacity == ndim
    let shape = shape.into_boxed_slice().into_vec();

    let arr_ptr = qant_tensor.data().as_ptr() as *mut c_void;
    let manager_ctx = Box::into_raw(Box::new(qant_tensor)) as *mut c_void;

    let tensor = DLTensor {
        data: arr_ptr,
        device,
        ndim: shape.len() as i32,
        dtype,
        shape: shape.as_ptr() as *mut ShapeType,
        strides: std::ptr::null::<ShapeType>() as *mut ShapeType,
        byte_offset: 0,
    };

    #[allow(clippy::mem_forget)]
    std::mem::forget(shape);
    // the Box::into_raw forgets qant_array automatically

    let deleter = match D {
        1 => delete_feature_managed_tensor1,
        2 => delete_feature_managed_tensor2,
        3 => delete_feature_managed_tensor3,
        4 => delete_feature_managed_tensor4,
        _ => return Err(toolkit_error!("tensor must have 1-4 dimensions, got {D}")),
    };

    Ok(DLManagedTensorVersioned {
        version: DLPackVersion {
            major: DLPACK_MAJOR_VERSION,
            minor: DLPACK_MINOR_VERSION,
        },
        flags: 0,
        dl_tensor: tensor,
        manager_ctx,
        deleter: Some(deleter),
    })
}
