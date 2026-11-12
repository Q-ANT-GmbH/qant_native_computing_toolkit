use std::borrow::Cow;

use crate::{errors::ToolkitError, global_settings::Value, toolkit_error};

pub struct QantTensor<'a, const D: usize> {
    data: Cow<'a, [Value]>,
    dims: [usize; D],
}

impl<'a, const D: usize> QantTensor<'a, D> {
    // Constructor for both borrowed and owned data
    pub fn new<ContainerType: Into<Cow<'a, [Value]>>>(
        data: ContainerType,
        dims: [usize; D],
    ) -> Result<Self, ToolkitError> {
        let data_cow = data.into();

        if data_cow.len() != dims.iter().product::<usize>() {
            return Err(toolkit_error!(
                "shape and length of data are inconsistent, got length={0} and shape={1:?}",
                data_cow.len(),
                dims,
            ));
        }

        Ok(Self {
            data: data_cow,
            dims,
        })
    }

    pub fn data(&self) -> &[Value] {
        &self.data
    }

    pub fn shape(&self) -> &[usize; D] {
        &self.dims
    }

    pub fn len(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

pub type QantTensor1<'a> = QantTensor<'a, 1>;
pub type QantTensor2<'a> = QantTensor<'a, 2>;
pub type QantTensor3<'a> = QantTensor<'a, 3>;
pub type QantTensor4<'a> = QantTensor<'a, 4>;
