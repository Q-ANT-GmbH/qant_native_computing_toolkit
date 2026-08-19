#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

include!(concat!(env!("OUT_DIR"), "/dlpack_bindings.rs"));
use std::{ffi::c_void, slice};

use half::bf16;

use crate::{
    data_format_conversion::{fixeds_to_floats, floats_to_fixeds},
    data_structures::QantTensor,
    errors::ToolkitError,
    toolkit_error,
};

pub type ShapeType = i64;

pub const DLDTYPE_I16: DLDataType = DLDataType {
    code: DLDataTypeCode_kDLInt as u8,
    bits: 16,
    lanes: 1,
};
pub const DLDTYPE_BF16: DLDataType = DLDataType {
    code: DLDataTypeCode_kDLBfloat as u8,
    bits: 16,
    lanes: 1,
};
pub const DLDTYPE_F32: DLDataType = DLDataType {
    code: DLDataTypeCode_kDLFloat as u8,
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

#[unsafe(no_mangle)]
pub extern "C" fn delete_vector_managed_tensor_f32(self_: *mut DLManagedTensorVersioned) {
    // bringing all data managing objects in scope
    // so rust will call their destructors automatically at the end of this function
    let dltensor_box = unsafe { Box::from_raw(self_) };
    let dltensor = *dltensor_box;

    // for a vector-backed tensor, the manager is the underlying vector
    // the use is to go out of scope
    #[allow(unused_variables)]
    let ctx = unsafe { Box::from_raw(dltensor.manager_ctx as *mut Vec<f32>) };

    // we also have to clean up the shape array
    let ndim = dltensor.dl_tensor.ndim as usize;
    #[allow(unused_variables)]
    let shape_vec = unsafe { Vec::from_raw_parts(dltensor.dl_tensor.shape, ndim, ndim) };
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

pub fn dltensor_to_qant_tensor<'a, const D: usize>(
    dltensor_ptr: *const DLManagedTensorVersioned,
    name: &str,
) -> Result<QantTensor<'a, D>, ToolkitError> {
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
    let arr: QantTensor<'a, D> = match tensor.dtype.code {
        code if code == DLDataTypeCode_kDLBfloat as u8 => {
            let data: &[bf16] =
                unsafe { slice::from_raw_parts(tensor.data as *const bf16, len_data as usize) };
            QantTensor::new(data, shape_fixed)?
        }
        code if code == DLDataTypeCode_kDLInt as u8 => {
            let data: &[i16] =
                unsafe { slice::from_raw_parts(tensor.data as *const i16, len_data as usize) };
            QantTensor::new(fixeds_to_floats(data), shape_fixed)?
        }
        code => {
            return Err(toolkit_error!(
                "Invalid dtype, DLTensor has dtype.code {0:?}, only {DLDataTypeCode_kDLBfloat} and {DLDataTypeCode_kDLInt} supported",
                code
            ));
        }
    };

    Ok(arr)
}

pub fn qant_tensor_to_dltensor<const D: usize>(
    qant_tensor: QantTensor<D>,
    device: DLDevice,
    dtype: DLDataType,
) -> Result<DLManagedTensorVersioned, ToolkitError> {
    let shape: Vec<ShapeType> = qant_tensor
        .shape()
        .iter()
        .map(|&x| x as ShapeType)
        .collect();

    let (arr_ptr, manager_ctx) = match dtype.code {
        code if code == DLDataTypeCode_kDLBfloat as u8 => (
            qant_tensor.data().as_ptr() as *mut c_void,
            Box::into_raw(Box::new(qant_tensor)) as *mut c_void,
        ),
        code if code == DLDataTypeCode_kDLInt as u8 => {
            if D != 1 {
                return Err(toolkit_error!(
                    "only 1D tensors can be converted to int16 DLTensors, got {D}D tensor"
                ));
            }
            let qant_tensor_buffer = QantTensor::new(
                bytemuck::try_cast_vec(floats_to_fixeds(qant_tensor.data()))
                    .map_err(|_| toolkit_error!("Failed to cast data buffer"))?,
                [shape.iter().product::<ShapeType>() as usize],
            )?;
            (
                qant_tensor_buffer.data().as_ptr() as *mut c_void,
                Box::into_raw(Box::new(qant_tensor_buffer)) as *mut c_void,
            )
        }
        code => {
            return Err(toolkit_error!(
                "Invalid dtype, DLTensor has dtype.code {0:?}, only {DLDataTypeCode_kDLBfloat} and {DLDataTypeCode_kDLInt} supported",
                code
            ));
        }
    };

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
