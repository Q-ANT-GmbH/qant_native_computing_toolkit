use std::num::NonZeroUsize;

use crate::data_structures::QantTensor2;
use crate::global_settings::Value;
use crate::memory_management::{self, QantMemoryHandle};
use crate::qant_driver_import::QantDataRole;
use crate::toolkit_error;
use crate::{
    data_structures::{QantTensor3, QantTensor4},
    errors::ToolkitError,
};

#[derive(Clone, Copy)]
pub struct ConvAttributes {
    pub(crate) padding: usize,
    pub(crate) stride: NonZeroUsize,
    pub(crate) dilation: usize,
}

impl ConvAttributes {
    /// Returns an error if `stride` is zero.
    pub fn new(padding: usize, stride: usize, dilation: usize) -> Result<Self, ToolkitError> {
        Ok(Self {
            padding,
            stride: non_zero_stride(stride)?,
            dilation,
        })
    }
}

#[derive(Clone, Copy)]
pub struct ConvTransposeAttributes {
    pub(crate) padding: usize,
    pub(crate) stride: NonZeroUsize,
    pub(crate) dilation: usize,
    pub(crate) output_padding: usize,
}

impl ConvTransposeAttributes {
    /// Returns an error if `stride` is zero.
    pub fn new(
        padding: usize,
        stride: usize,
        dilation: usize,
        output_padding: usize,
    ) -> Result<Self, ToolkitError> {
        Ok(Self {
            padding,
            stride: non_zero_stride(stride)?,
            dilation,
            output_padding,
        })
    }
}

pub struct ConvWeightsHandle {
    handle: QantMemoryHandle,
    shape: [usize; 4],
}

impl ConvWeightsHandle {
    pub fn npu_id(&self) -> u32 {
        self.handle.npu_id()
    }

    pub fn shape(&self) -> [usize; 4] {
        self.shape
    }
}

pub fn upload_conv_weights(
    npu_id: u32,
    weights: &QantTensor4,
) -> Result<ConvWeightsHandle, ToolkitError> {
    let &[out_channels, in_channels, kernel_height, kernel_width] = weights.shape();
    let n_cols = in_channels * kernel_height * kernel_width;

    let weights_as_matrix = QantTensor2::new(weights.data(), [out_channels, n_cols])?;
    let handle = QantMemoryHandle::from_matrix(npu_id, &weights_as_matrix, QantDataRole::Weights)?;

    Ok(ConvWeightsHandle {
        handle,
        shape: *weights.shape(),
    })
}

/// This function performs a forward pass of a 2d convolution with the provided weights on the
/// provided features.
///
/// First, the required multiplications are extracted of the convolution operation.
/// Second, the multiplications are performed via the Q.ANT Native Computing Driver.
/// Third, reassemble the multiplications result and return the convolution result.
///
/// # Returns
/// - `Ok(QantTensor4)`: The resulting feature data after applying convolution.
/// - `Err(ToolkitError)`: An error if the operation fails, e.g., due to invalid dimensions or attributes.
pub fn conv_fprop<'b>(
    npu_id: u32,
    features: &QantTensor4,
    weights: &QantTensor4,
    attributes: ConvAttributes,
) -> Result<QantTensor4<'b>, ToolkitError> {
    let weights_handle = upload_conv_weights(npu_id, weights)?;
    conv_fprop_preloaded_weights(features, &weights_handle, attributes)
}

pub fn conv_fprop_preloaded_weights<'b>(
    features: &QantTensor4,
    weights: &ConvWeightsHandle,
    attributes: ConvAttributes,
) -> Result<QantTensor4<'b>, ToolkitError> {
    if attributes.dilation != 1 {
        return Err(ToolkitError::from_str("dilation != 1 not implemented yet"));
    };

    let &[
        n_batches,
        features_channels,
        features_height,
        features_width,
    ] = features.shape();
    let [out_channels, in_channels, kernel_height, kernel_width] = weights.shape();

    if features_channels != in_channels {
        return Err(toolkit_error!(
            "in_channels in features ({features_channels}) != in_channels in weights ({in_channels})",
        ));
    };

    let stride = attributes.stride.get();
    // calc output size
    let output_height = (features_height + 2 * attributes.padding - kernel_height) / stride + 1;
    let output_width = (features_width + 2 * attributes.padding - kernel_width) / stride + 1;

    // im2row supports asymmetric kernels, padding and stride
    let kernel_size = Size {
        height: kernel_height,
        width: kernel_width,
    };
    let padding = Size {
        height: attributes.padding,
        width: attributes.padding,
    };
    let stride = Size {
        height: stride,
        width: stride,
    };
    let patches = im2row(features, kernel_size, padding, stride)?;

    mul_weights_with_patches(
        &weights.handle,
        &patches,
        [n_batches, out_channels, output_height, output_width],
    )
}

pub struct ConvTransposeWeightsHandle {
    handle: QantMemoryHandle,
    shape: [usize; 4],
}

impl ConvTransposeWeightsHandle {
    pub fn npu_id(&self) -> u32 {
        self.handle.npu_id()
    }

    pub fn shape(&self) -> [usize; 4] {
        self.shape
    }
}

pub fn upload_conv_transpose_weights(
    npu_id: u32,
    weights: &QantTensor4,
) -> Result<ConvTransposeWeightsHandle, ToolkitError> {
    let &[in_channels, out_channels, kernel_height, kernel_width] = weights.shape();
    let n_cols = in_channels * kernel_height * kernel_width;

    // Reorder weights from (C_in, C_out, H, W) — the public API convention — to
    // (C_out, C_in, H, W) that mul_matrix_vec_batched_mem expects (one contiguous vector per output channel).
    // The H*W spatial block is innermost in both layouts, so each iteration is a contiguous memcpy.
    let weights_reordered: Vec<Value> = {
        let src = weights.data();
        let hw = kernel_height * kernel_width;
        let mut out = vec![Value::default(); src.len()];
        for c_in in 0..in_channels {
            for c_out in 0..out_channels {
                let src_start = c_in * (out_channels * hw) + c_out * hw;
                let dst_start = c_out * (in_channels * hw) + c_in * hw;
                out[dst_start..dst_start + hw].copy_from_slice(&src[src_start..src_start + hw]);
            }
        }
        out
    };

    let weights_as_matrix = QantTensor2::new(&weights_reordered, [out_channels, n_cols])?;
    let handle = QantMemoryHandle::from_matrix(npu_id, &weights_as_matrix, QantDataRole::Weights)?;

    Ok(ConvTransposeWeightsHandle {
        handle,
        shape: *weights.shape(),
    })
}

pub fn conv_transpose_fprop<'b>(
    npu_id: u32,
    features: &QantTensor4,
    weights: &QantTensor4,
    attributes: ConvTransposeAttributes,
) -> Result<QantTensor4<'b>, ToolkitError> {
    let weights_handle = upload_conv_transpose_weights(npu_id, weights)?;
    conv_transpose_fprop_preloaded_weights(features, &weights_handle, attributes)
}

pub fn conv_transpose_fprop_preloaded_weights<'b>(
    features: &QantTensor4,
    weights: &ConvTransposeWeightsHandle,
    attributes: ConvTransposeAttributes,
) -> Result<QantTensor4<'b>, ToolkitError> {
    if attributes.dilation != 1 {
        return Err(ToolkitError::from_str("dilation != 1 not implemented yet"));
    }
    if attributes.output_padding != 0 {
        return Err(ToolkitError::from_str(
            "output_padding != 0 not implemented yet",
        ));
    }

    let &[
        n_batches,
        features_channels,
        features_height,
        features_width,
    ] = features.shape();
    let [in_channels, out_channels, kernel_height, kernel_width] = weights.shape();

    if features_channels != in_channels {
        return Err(toolkit_error!(
            "in_channels in features ({features_channels}) != in_channels in weights ({in_channels})",
        ));
    }
    let stride = attributes.stride.get();
    // Output size for conv-transpose
    let output_height = (features_height - 1) * stride - 2 * attributes.padding
        + attributes.dilation * (kernel_height - 1)
        + attributes.output_padding
        + 1;
    let output_width = (features_width - 1) * stride - 2 * attributes.padding
        + attributes.dilation * (kernel_width - 1)
        + attributes.output_padding
        + 1;

    // Build the patch matrix U (M x K) with M = H_out * W_out and K = C_in * K_h * K_w.
    // Each row corresponds to one output position (oy, ox), each column to a (c_in, k_y, k_x)
    // position. im2row_transpose supports asymmetric kernels, padding and stride.
    let kernel_size = Size {
        height: kernel_height,
        width: kernel_width,
    };
    let padding = Size {
        height: attributes.padding,
        width: attributes.padding,
    };
    let stride = Size {
        height: stride,
        width: stride,
    };
    let patches = im2row_transpose(features, kernel_size, padding, stride)?;

    // the weights (C_out x K) are the matrix, each row of U is a vector in the batch
    mul_weights_with_patches(
        &weights.handle,
        &patches,
        [n_batches, out_channels, output_height, output_width],
    )
}

fn non_zero_stride(stride: usize) -> Result<NonZeroUsize, ToolkitError> {
    NonZeroUsize::new(stride).ok_or_else(|| toolkit_error!("stride must be > 0, got {stride}"))
}

struct Size {
    height: usize,
    width: usize,
}

// multiplies the weights with every patch (one row of `patches`) on the NPU holding the weights
// and returns the result in the shape `output_shape` (n_batches x out_channels x height x width)
fn mul_weights_with_patches<'b>(
    weights: &QantMemoryHandle,
    patches: &QantTensor3,
    output_shape: [usize; 4],
) -> Result<QantTensor4<'b>, ToolkitError> {
    let [n_batches, out_channels, output_height, output_width] = output_shape;
    let n_rows = patches.shape()[0] * patches.shape()[1];
    let patches = QantTensor2::new(patches.data(), [n_rows, patches.shape()[2]])?;

    let patches_handle =
        QantMemoryHandle::from_matrix(weights.npu_id(), &patches, QantDataRole::Features)?;
    let result_handle = QantMemoryHandle::allocate(
        weights.npu_id(),
        n_rows,
        out_channels,
        QantDataRole::Features,
    )?;
    memory_management::mul_matrix_vec_batched_mem(weights, &patches_handle, &result_handle)?;
    let result_data = result_handle.download()?;

    // the patches are used as a vector batch, so the result has the structure:
    // n_batches x output_height x output_width x out_channels
    // -> n_batches x out_channels x output_height x output_width
    let result_data = nhwc_to_nchw(
        result_data.data(),
        n_batches,
        out_channels,
        output_height,
        output_width,
    );
    QantTensor4::new(result_data, output_shape)
}

// translates a 4D features tensor into a matrix for matrix-vec-based convolution
// each kernel patch is in a separate row (im2row, an alternative to im2col)
fn im2row<'b>(
    features: &QantTensor4,
    kernel_size: Size,
    padding: Size,
    stride: Size,
) -> Result<QantTensor3<'b>, ToolkitError> {
    let &[
        n_batches,
        features_channels,
        features_height,
        features_width,
    ] = features.shape();

    // The final matrix will have the size [kernel_position x (kernel_pixel_position * channel_in)]
    // Each row will be summed up in the end, therefore all kernel pixels over all input
    // channels should be in a row
    let kernel_pixel_num = kernel_size.width * kernel_size.height;
    let width_col = kernel_pixel_num * features_channels;
    // calc output size
    let kernels_per_height =
        (features_height + 2 * padding.height - kernel_size.height) / stride.height + 1;
    let kernels_per_width =
        (features_width + 2 * padding.width - kernel_size.width) / stride.width + 1;
    let height_col = kernels_per_width * kernels_per_height;

    let per_batch_size = width_col * height_col;
    let mat_data_size = per_batch_size * n_batches;

    // With this pattern, we do not write zeros to every position first.
    // As we are overwriting every value, this is not required
    let mut mat_data = vec![Value::ZERO; mat_data_size];

    for i in 0..mat_data.len() {
        let batch_id = i / per_batch_size;
        let i = i % per_batch_size;
        let kernel_col = i / width_col;
        let kernel_pixel = i % (kernel_pixel_num);
        let chl_in = i % width_col / kernel_pixel_num;
        let chl_pos_in = features_width * features_height * (chl_in + batch_id * features_channels);

        let w_out = kernel_col % kernels_per_width;
        let h_out = kernel_col / kernels_per_width;

        let k_y = kernel_pixel / kernel_size.width;
        let k_x = kernel_pixel % kernel_size.width;

        let h_in = (h_out * stride.height) as i64 - padding.height as i64;
        let w_in = (w_out * stride.width) as i64 - padding.width as i64;

        let h = h_in + k_y as i64;
        let w = w_in + k_x as i64;

        // is the pixel inside the picture
        let valid_pos = h >= 0 && w >= 0 && h < features_height as i64 && w < features_width as i64;
        let val = if valid_pos {
            features.data()[chl_pos_in + h as usize * features_width + w as usize]
        } else {
            Value::ZERO
        };

        // row major
        let col = chl_in * kernel_pixel_num + kernel_size.width * k_y + k_x;

        mat_data[(batch_id * per_batch_size) + kernel_col * width_col + col] = val;
    }

    let mat_feature_data = QantTensor3::new(mat_data, [n_batches, height_col, width_col])?;
    Ok(mat_feature_data)
}

// translates a 4D features tensor into a matrix for matrix-vec-based transpose convolution
// each kernel patch is in a separate row (im2row, an alternative to im2col)
fn im2row_transpose<'b>(
    features: &QantTensor4,
    kernel_size: Size,
    padding: Size,
    stride: Size,
) -> Result<QantTensor3<'b>, ToolkitError> {
    let &[
        n_batches,
        features_channels,
        features_height,
        features_width,
    ] = features.shape();
    let features_size = features_width * features_height;
    let kernel_pixel_num = kernel_size.width * kernel_size.height;
    let output_padding = 0;
    let dilation = 1;
    // Output size for conv-transpose
    let output_height = (features_height - 1) * stride.height - 2 * padding.height
        + dilation * (kernel_size.height - 1)
        + output_padding
        + 1;
    let output_width = (features_width - 1) * stride.width - 2 * padding.width
        + dilation * (kernel_size.width - 1)
        + output_padding
        + 1;
    let kernels_per_in_chl = output_width * output_height;
    let k_dim = features_channels * kernel_pixel_num;
    let m_dim = output_width * output_height;
    let per_batch_size = m_dim * k_dim;
    let mut mat_data = vec![Value::ZERO; per_batch_size * n_batches];

    for i in 0..mat_data.len() {
        let batch_id = i / per_batch_size;
        let cin = (i / kernel_pixel_num) % features_channels;
        let kx = i % kernel_size.width;
        let ky = (i / kernel_size.width) % kernel_size.height;
        let kernel_iteration = (i / (features_channels * kernel_pixel_num)) % kernels_per_in_chl;

        // Calculate output position coordinates
        let ox = kernel_iteration % output_width;
        let oy = kernel_iteration / output_width;

        // Apply transpose convolution coordinate transformation
        let feat_x = (ox as i32 + padding.width as i32 - kx as i32) / stride.width as i32;
        let feat_y = (oy as i32 + padding.height as i32 - ky as i32) / stride.height as i32;

        // Check for valid indices
        let val = if feat_x >= 0 || feat_y >= 0 {
            let feat_x = feat_x as usize;
            let feat_y = feat_y as usize;

            // Verify exact strided positioning
            if (ox + padding.width) >= kx
                && (oy + padding.height) >= ky
                && (ox + padding.width - kx).is_multiple_of(stride.width)
                && (oy + padding.height - ky).is_multiple_of(stride.height)
                && feat_x < features_width
                && feat_y < features_height
            {
                // only use the real features data, when we are within the features bounds
                // otherwise the default padding is zero
                features.data()[batch_id * features_channels * features_size
                    + cin * features_size
                    + feat_y * features_width
                    + feat_x]
            } else {
                // Zero padding
                Value::ZERO
            }
        } else {
            // Zero padding
            Value::ZERO
        };

        let row = oy * output_width + ox;
        let col_offset = cin * kernel_pixel_num + ky * kernel_size.width + kx;
        mat_data[batch_id * per_batch_size + row * k_dim + col_offset] = val;
    }

    let mat_tensor = QantTensor3::new(mat_data, [n_batches, m_dim, k_dim])?;
    Ok(mat_tensor)
}

fn nhwc_to_nchw(
    data: &[Value],
    batches: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Vec<Value> {
    let num_positions = height * width;
    if channels == 1 || num_positions == 1 {
        return data.to_vec();
    }

    let mut reordered = vec![data[0]; data.len()];
    for b in 0..batches {
        for pos in 0..num_positions {
            for c in 0..channels {
                // Original index (B × position × C)
                let src_idx = (b * num_positions + pos) * channels + c;

                // Reordered index (B × C × position)
                let dst_idx = (b * channels + c) * num_positions + pos;

                reordered[dst_idx] = data[src_idx];
            }
        }
    }
    reordered
}

#[cfg(test)]
mod tests {
    use crate::utils::test_utils;
    use half::bf16;

    use super::*;

    fn get_features_1_1_4_4<'a>() -> QantTensor4<'a> {
        // 0.1, 0.2, 0.3, 0.4
        // 0.5, 0.6, 0.7, 0.8
        // 0.9, 1., 0.11, 0.21
        // 0.31, 0.41, 0.51, 0.61
        let features_data = test_utils::to_bf16s(&vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 0.11, 0.21, 0.31, 0.41, 0.51, 0.61,
        ]);
        QantTensor4::new(features_data, [1, 1, 4, 4]).unwrap()
    }
    fn get_features_1_3_4_4<'a>() -> QantTensor4<'a> {
        let features_data = test_utils::to_bf16s(&vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 0.11, 0.21, 0.31, 0.41, 0.51,
            0.61, //
            0.101, 0.201, 0.301, 0.401, 0.501, 0.601, 0.701, 0.801, 0.901, 1.001, 0.111, 0.211,
            0.311, 0.411, 0.511, 0.611, //
            0.102, 0.202, 0.302, 0.402, 0.502, 0.602, 0.702, 0.802, 0.902, 1.002, 0.112, 0.212,
            0.312, 0.412, 0.512, 0.612,
        ]);
        QantTensor4::new(features_data, [1, 3, 4, 4]).unwrap()
    }

    fn get_features_1_3_5_4<'a>() -> QantTensor4<'a> {
        let features_data = test_utils::to_bf16s(&vec![
            0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14,
            0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.220, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28,
            0.29, 0.3, 0.310, 0.320, 0.330, 0.340, 0.350, 0.360, 0.370, 0.380, 0.390, 0.4, 0.410,
            0.420, 0.430, 0.440, 0.450, 0.460, 0.470, 0.480, 0.490, 0.5, 0.510, 0.520, 0.530,
            0.540, 0.550, 0.560, 0.570, 0.580, 0.590,
        ]);
        QantTensor4::new(features_data, [1, 3, 5, 4]).unwrap()
    }

    #[test]

    fn test_conv_fprop_simple_weights() {
        let features = get_features_1_1_4_4();

        // Create weights
        let weights_data = vec![1, 0, 0, 0];
        let weights = QantTensor4::new(test_utils::to_bf16s(&weights_data), [1, 1, 2, 2]).unwrap();

        // Set convolution attributes
        let attributes = ConvAttributes::new(0, 1, 1).unwrap();

        // Perform convolution
        let result = conv_fprop(0, &features, &weights, attributes).unwrap();

        // Expected output
        let expected_data = vec![0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.9, 1., 0.110];
        let expected_output =
            QantTensor4::new(test_utils::to_bf16s(&expected_data), [1, 1, 3, 3]).unwrap();

        // Assert the results
        assert_eq!(result.shape(), expected_output.shape());
        test_utils::assert_almost_equal_vec(result.data(), expected_output.data(), 0.02);
    }

    #[test]
    fn test_conv_fprop() {
        let features = get_features_1_1_4_4();

        // Create weights
        let weights_data = vec![1., 0., 0., 1.];
        let weights = QantTensor4::new(test_utils::to_bf16s(&weights_data), [1, 1, 2, 2]).unwrap();

        // Set convolution attributes
        let attributes = ConvAttributes::new(0, 1, 1).unwrap();

        // Perform convolution
        let result = conv_fprop(0, &features, &weights, attributes).unwrap();

        // Expected output
        let expected_data = vec![0.7, 0.9, 1.1, 1.5, 0.71, 0.91, 1.31, 1.51, 0.72];
        let expected_output =
            QantTensor4::new(test_utils::to_bf16s(&expected_data), [1, 1, 3, 3]).unwrap();

        // Assert the results
        assert_eq!(result.shape(), expected_output.shape());
        test_utils::assert_almost_equal_vec(result.data(), expected_output.data(), 0.02);
    }

    #[test]
    fn test_conv_fprop_padding_1() {
        let features = get_features_1_1_4_4();

        // Create weights
        let weights_data = test_utils::to_bf16s(&vec![1., 0., 0., 1.]);
        let weights = QantTensor4::new(&weights_data, [1, 1, 2, 2]).unwrap();

        // Set convolution attributes
        let attributes = ConvAttributes::new(1, 1, 1).unwrap();

        // Perform convolution
        let result = conv_fprop(0, &features, &weights, attributes).unwrap();

        // Expected output
        let expected_data = test_utils::to_bf16s(&vec![
            0.1, 0.2, 0.3, 0.4, 0., 0.5, 0.7, 0.9, 1.1, 0.4, 0.9, 1.5, 0.71, 0.91, 0.8, 0.31, 1.31,
            1.51, 0.72, 0.21, 0., 0.31, 0.41, 0.51, 0.61,
        ]);
        let expected_output = QantTensor4::new(expected_data, [1, 1, 5, 5]).unwrap();

        // Assert the results
        assert_eq!(result.shape(), expected_output.shape());
        test_utils::assert_almost_equal_vec(result.data(), expected_output.data(), 0.02);
    }

    #[test]

    fn test_conv_fprop_multichannel() {
        let features = get_features_1_3_4_4();

        // Create weights
        let weights_data = test_utils::to_bf16s(&vec![
            1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., // -
            0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., // -
            0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., // -
            1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., // -
        ]);
        let weights = QantTensor4::new(&weights_data, [4, 3, 2, 2]).unwrap();

        // Set convolution attributes
        let attributes = ConvAttributes::new(0, 1, 1).unwrap();

        // Perform convolution
        let result = conv_fprop(0, &features, &weights, attributes).unwrap();

        // Expected output
        let expected_data = test_utils::to_bf16s(&vec![
            0.7, 0.9, 1.100, 1.500, 0.710, 0.910, 1.310, 1.510, 0.720, // -
            0.702, 0.902, 1.102, 1.502, 0.712, 0.912, 1.312, 1.512, 0.722, // -
            0.704, 0.904, 1.104, 1.504, 0.714, 0.914, 1.314, 1.514, 0.724, // -
            0.803, 1.103, 1.403, 2.003, 2.303, 1.613, 2.213, 1.523, 0.833, // -
        ]);
        let expected_output = QantTensor4::new(expected_data, [1, 4, 3, 3]).unwrap();

        // Assert the results
        assert_eq!(result.shape(), expected_output.shape());
        test_utils::assert_almost_equal_vec(result.data(), expected_output.data(), 0.02);
    }

    #[test]
    fn test_conv_fprop_stride2_simple_weights() {
        let features = get_features_1_1_4_4();

        // Create weights
        let weights_data = test_utils::to_bf16s(&vec![1., 0., 0., 1.]);
        let weights = QantTensor4::new(&weights_data, [1, 1, 2, 2]).unwrap();

        // Set convolution attributes
        let attributes = ConvAttributes::new(0, 2, 1).unwrap();

        // Perform convolution
        let result = conv_fprop(0, &features, &weights, attributes).unwrap();

        // Expected output
        let expected_data = test_utils::to_bf16s(&vec![0.7, 1.100, 1.310, 0.720]);
        let expected_output = QantTensor4::new(expected_data, [1, 1, 2, 2]).unwrap();

        // Assert the results
        assert_eq!(result.shape(), expected_output.shape());
        test_utils::assert_almost_equal_vec(result.data(), expected_output.data(), 0.02);
    }

    #[test]
    fn test_conv_fprop_3x4x5_features_2x2x3_weights_simple() {
        let features = get_features_1_3_5_4();

        // Create weights data (2x3x2)
        let weights_data = test_utils::to_bf16s(&vec![
            0.1, 0., 0., 0., 0., 0., // --
            0., 0., 0., 0., 0., 0., // --
            0., 0., 0., 0., 0., 0., // --
            0., 0., 0., 0., 0., 0., // --
            0., 0., 0., 0., 0., 0., // --
            0., 0., 0., 0., 0., 0.,
        ]);
        let weights = QantTensor4::new(&weights_data, [2, 3, 2, 3]).unwrap();

        // Set convolution attributes
        let attributes = ConvAttributes::new(0, 1, 1).unwrap();

        // Perform convolution
        let result = conv_fprop(0, &features, &weights, attributes).unwrap();

        // Expected output dimensions
        assert_eq!(result.shape().to_vec(), [1, 2, 4, 2]);

        // Expected output data
        // Note: You'll need to calculate the expected values based on your convolution implementation
        let expected_output = test_utils::to_bf16s(&vec![
            0., 0.001, 0.004, 0.005, 0.008, 0.009, 0.012, 0.013, 0., 0., 0., 0., 0., 0., 0., 0.,
        ]);
        assert_eq!(result.len(), expected_output.len());

        test_utils::assert_almost_equal_vec(result.data(), &expected_output, 0.001);
    }

    #[test]

    fn test_conv_fprop_3x4x5_features_2x2x3_weights() {
        let features = get_features_1_3_5_4();

        // Create weights data (2x3x2)
        let weights_data = test_utils::to_bf16s(&vec![
            0.1, 0.125, 0.150, 0.175, 0.2, 0.225, 0.250, 0.275, 0.3, 0.325, 0.350, 0.375, 0.4,
            0.425, 0.450, 0.475, 0.5, 0.525, //
            0.550, 0.575, 0.600, 0.625, 0.650, 0.675, 0.7, 0.725, 0.750, 0.775, 0.800, 0.825,
            0.850, 0.875, 0.9, 0.925, 0.950, 0.975,
        ]);
        let weights = QantTensor4::new(&weights_data, [2, 3, 2, 3]).unwrap();

        // Set convolution attributes
        let attributes = ConvAttributes::new(0, 1, 1).unwrap();

        // Perform convolution
        let result = conv_fprop(0, &features, &weights, attributes).unwrap();

        // Expected output dimensions
        assert_eq!(result.shape().to_vec(), [1, 2, 4, 2]);

        // Expected output data
        // Note: You'll need to calculate the expected values based on your convolution implementation
        let expected_output = test_utils::to_bf16s(&vec![
            1.670, 1.726, 1.895, 1.951, 2.120, 2.176, 2.345, 2.401, 3.533, 3.670, 4.082, 4.219,
            4.631, 4.768, 5.180, 5.317,
        ]);
        assert_eq!(result.len(), expected_output.len());
        test_utils::assert_almost_equal_vec(result.data(), &expected_output, 0.03);
    }

    #[test]

    fn test_conv_transpose_fprop_multichannel() {
        let features = get_features_1_3_4_4();

        // Create weights in (C_in, C_out, H, W) convention
        let weights_data = test_utils::to_bf16s(&vec![
            1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
            0., // c_in=0: c_out=0,1,2,3 each (h,w)
            0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.,
            0., // c_in=1: c_out=0,1,2,3
            0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1.,
            0., // c_in=2: c_out=0,1,2,3
        ]);
        let weights = QantTensor4::new(&weights_data, [3, 4, 2, 2]).unwrap();

        // Set convolution attributes
        let attributes = ConvTransposeAttributes::new(0, 1, 1, 0).unwrap();

        // Perform convolution
        let result = conv_transpose_fprop(0, &features, &weights, attributes).unwrap();

        // Expected output
        let expected_data = test_utils::to_bf16s(&vec![
            0.1, 0.2, 0.3, 0.4, 0., //--
            0.5, 0.7, 0.9, 1.100, 0.4, //--
            0.9, 1.500, 0.710, 0.910, 0.800, //--
            0.310, 1.310, 1.510, 0.720, 0.210, //--
            0., 0.310, 0.410, 0.510, 0.610, //--
            //--
            0.101, 0.201, 0.301, 0.401, 0., //--
            0.501, 0.702, 0.902, 1.102, 0.401, //--
            0.901, 1.502, 0.712, 0.912, 0.801, //--
            0.311, 1.312, 1.512, 0.722, 0.211, //--
            0., 0.311, 0.411, 0.511, 0.611, //--
            //--
            0.102, 0.202, 0.302, 0.402, 0., //--
            0.502, 0.704, 0.904, 1.104, 0.402, //--
            0.902, 1.504, 0.714, 0.914, 0.802, //--
            0.312, 1.314, 1.514, 0.724, 0.212, //--
            0., 0.312, 0.412, 0.512, 0.612, //--
            //--
            0.1, 0.301, 0.501, 0.701, 0.401, //--
            0.602, 1.303, 1.603, 1.903, 0.801, //--
            1.402, 2.503, 1.813, 1.123, 0.211, //--
            1.212, 1.723, 1.033, 1.333, 0.611, //--
            0.312, 0.412, 0.512, 0.612, 0.,
        ]);
        let expected_output = QantTensor4::new(expected_data, [1, 4, 5, 5]).unwrap();

        // Assert the results
        assert_eq!(result.shape(), expected_output.shape());
        test_utils::assert_almost_equal_vec(result.data(), expected_output.data(), 0.02);
    }

    #[test]

    fn test_conv_transpose_fprop_simple() {
        let features = get_features_1_3_4_4();
        // Create weights
        let weights_data = test_utils::to_bf16s(&vec![
            1., 0., 0., 1., // c_in=0: c_out=0,1,2,3
            0., 1., 0., 1., // c_in=1: c_out=0,1,2,3
            0., 0., 1., 1., // c_in=2: c_out=0,1,2,3
        ]);
        let weights = QantTensor4::new(&weights_data, [3, 4, 1, 1]).unwrap();

        // Set convolution attributes
        let attributes = ConvTransposeAttributes::new(0, 1, 1, 0).unwrap();

        // Perform convolution
        let result = conv_transpose_fprop(0, &features, &weights, attributes).unwrap();

        // Expected output
        let expected_data = test_utils::to_bf16s(&vec![
            0.1, 0.2, 0.3, 0.4, // --
            0.5, 0.600, 0.7, 0.800, // --
            0.9, 1., 0.110, 0.210, // --
            0.310, 0.410, 0.510, 0.610, // --
            // --
            0.101, 0.201, 0.301, 0.401, // --
            0.501, 0.601, 0.701, 0.801, // --
            0.901, 1.001, 0.111, 0.211, // --
            0.311, 0.411, 0.511, 0.611, // --
            // --
            0.102, 0.202, 0.302, 0.402, // --
            0.502, 0.602, 0.702, 0.802, // --
            0.902, 1.002, 0.112, 0.212, // --
            0.312, 0.412, 0.512, 0.612, // --
            // --
            0.303, 0.603, 0.903, 1.203, // --
            1.503, 1.803, 2.103, 2.403, // --
            2.703, 3.003, 0.333, 0.633, // --
            0.933, 1.233, 1.533, 1.833,
        ]);

        let expected_output = QantTensor4::new(expected_data, [1, 4, 4, 4]).unwrap();

        // Assert the results
        assert_eq!(result.shape(), expected_output.shape());
        test_utils::assert_almost_equal_vec(result.data(), expected_output.data(), 0.02);
    }

    #[test]
    fn test_im2row_0() {
        let features_data = test_utils::to_bf16s(&vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let features = QantTensor4::new(features_data, [1, 1, 3, 3]).unwrap();
        let kernel_size = Size {
            height: 2,
            width: 2,
        };
        let padding = Size {
            height: 0,
            width: 0,
        };
        let stride = Size {
            height: 1,
            width: 1,
        };

        let col_data = im2row(&features, kernel_size, padding, stride).unwrap();
        assert_eq!(col_data.shape().to_vec(), [1, 4, 4]);

        let expected = test_utils::to_bf16s(&vec![1, 2, 4, 5, 2, 3, 5, 6, 4, 5, 7, 8, 5, 6, 8, 9]);
        assert_eq!(col_data.data(), expected);
    }

    #[test]
    fn test_im2row_1() {
        let features_data = test_utils::to_bf16s(&vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 15, 25, 35, 45, 55, 65,
            75, 85, 95,
        ]);
        let features = QantTensor4::new(features_data, [1, 3, 3, 3]).unwrap();
        let kernel_size = Size {
            height: 2,
            width: 2,
        };
        let padding = Size {
            height: 0,
            width: 0,
        };
        let stride = Size {
            height: 1,
            width: 1,
        };

        let col_data = im2row(&features, kernel_size, padding, stride).unwrap();
        assert_eq!(col_data.shape().to_vec(), [1, 4, 4 * 3]);

        let expected = test_utils::to_bf16s(&vec![
            1, 2, 4, 5, 10, 20, 40, 50, 15, 25, 45, 55, 2, 3, 5, 6, 20, 30, 50, 60, 25, 35, 55, 65,
            4, 5, 7, 8, 40, 50, 70, 80, 45, 55, 75, 85, 5, 6, 8, 9, 50, 60, 80, 90, 55, 65, 85, 95,
        ]);
        assert_eq!(col_data.data(), expected);
    }

    #[test]
    fn test_im2row_2() {
        let features = get_features_1_3_4_4();
        let kernel_size = Size {
            height: 2,
            width: 2,
        };
        let padding = Size {
            height: 0,
            width: 0,
        };
        let stride = Size {
            height: 1,
            width: 1,
        };

        let col_data = im2row(&features, kernel_size, padding, stride).unwrap();
        assert_eq!(col_data.shape().to_vec(), [1, 9, 4 * 3]);

        let expected = test_utils::to_bf16s(&vec![
            0.1, 0.2, 0.5, 0.6, 0.101, 0.201, 0.501, 0.601, 0.102, 0.202, 0.502, 0.602, 0.2, 0.3,
            0.6, 0.7, 0.201, 0.301, 0.601, 0.701, 0.202, 0.302, 0.602, 0.702, 0.3, 0.4, 0.7, 0.8,
            0.301, 0.401, 0.701, 0.801, 0.302, 0.402, 0.702, 0.802, 0.5, 0.6, 0.9, 1., 0.501,
            0.601, 0.901, 1.001, 0.502, 0.602, 0.902, 1.002, 0.6, 0.7, 1., 0.11, 0.601, 0.701,
            1.001, 0.111, 0.602, 0.702, 1.002, 0.112, 0.7, 0.8, 0.11, 0.21, 0.701, 0.801, 0.111,
            0.211, 0.702, 0.802, 0.112, 0.212, 0.9, 1., 0.31, 0.41, 0.901, 1.001, 0.311, 0.411,
            0.902, 1.002, 0.312, 0.412, 1., 0.11, 0.41, 0.51, 1.001, 0.111, 0.411, 0.511, 1.002,
            0.112, 0.412, 0.512, 0.11, 0.21, 0.51, 0.61, 0.111, 0.211, 0.511, 0.611, 0.112, 0.212,
            0.512, 0.612,
        ]);
        assert_eq!(col_data.data(), expected);
    }
    #[test]
    fn test_im2row_batched() {
        let features_data = test_utils::to_bf16s(&vec![
            100, 200, 300, 101, 201, 301, 102, 202, 302, 103, 203, 303,
        ]);
        let features = QantTensor4::new(features_data, [4, 3, 1, 1]).unwrap();
        let kernel_size = Size {
            height: 1,
            width: 1,
        };
        let padding = Size {
            height: 0,
            width: 0,
        };
        let stride = Size {
            height: 1,
            width: 1,
        };

        let col_data = im2row(&features, kernel_size, padding, stride).unwrap();
        assert_eq!(col_data.shape().to_vec(), [4, 1, 3]);

        let expected = test_utils::to_bf16s(&vec![
            100, 200, 300, 101, 201, 301, 102, 202, 302, 103, 203, 303,
        ]);
        assert_eq!(col_data.data(), expected);
    }

    #[test]
    fn test_im2row_batched_2() {
        let features_data = test_utils::to_bf16s(&vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 0.11, 0.21, 0.31, 0.41, 0.51,
            0.61, // --
            0.101, 0.201, 0.301, 0.401, 0.501, 0.601, 0.701, 0.801, 0.901, 1.001, 0.111, 0.211,
            0.311, 0.411, 0.511, 0.611, // --
            0.102, 0.202, 0.302, 0.402, 0.502, 0.602, 0.702, 0.802, 0.902, 1.002, 0.112, 0.212,
            0.312, 0.412, 0.512, 0.612,
        ]);
        let features = QantTensor4::new(features_data, [3, 1, 4, 4]).unwrap();
        let kernel_size = Size {
            width: 2,
            height: 2,
        };
        let padding = Size {
            width: 0,
            height: 0,
        };
        let stride = Size {
            width: 1,
            height: 1,
        };

        let col_data = im2row(&features, kernel_size, padding, stride).unwrap();
        assert_eq!(col_data.shape().to_vec(), [3, 9, 4]);
        let expected = test_utils::to_bf16s(&vec![
            0.1, 0.2, 0.5, 0.6, 0.2, 0.3, 0.6, 0.7, 0.3, 0.4, 0.7, 0.8, 0.5, 0.6, 0.9, 1., 0.6,
            0.7, 1., 0.11, 0.7, 0.8, 0.11, 0.21, 0.9, 1., 0.31, 0.41, 1., 0.11, 0.41, 0.51, 0.11,
            0.21, 0.51, 0.61, 0.101, 0.201, 0.501, 0.601, 0.201, 0.301, 0.601, 0.701, 0.301, 0.401,
            0.701, 0.801, 0.501, 0.601, 0.901, 1.001, 0.601, 0.701, 1.001, 0.111, 0.701, 0.801,
            0.111, 0.211, 0.901, 1.001, 0.311, 0.411, 1.001, 0.111, 0.411, 0.511, 0.111, 0.211,
            0.511, 0.611, 0.102, 0.202, 0.502, 0.602, 0.202, 0.302, 0.602, 0.702, 0.302, 0.402,
            0.702, 0.802, 0.502, 0.602, 0.902, 1.002, 0.602, 0.702, 1.002, 0.112, 0.702, 0.802,
            0.112, 0.212, 0.902, 1.002, 0.312, 0.412, 1.002, 0.112, 0.412, 0.512, 0.112, 0.212,
            0.512, 0.612,
        ]);
        assert_eq!(col_data.data(), expected);
    }

    #[test]
    fn test_im2row_transpose_batched() {
        let single = get_features_1_3_4_4();
        let features_data: Vec<Value> = single
            .data()
            .iter()
            .copied()
            .chain(single.data().iter().map(|&v| v * bf16::from_f32(2.)))
            .collect();
        let features = QantTensor4::new(&features_data, [2, 3, 4, 4]).unwrap();
        let kernel_size = || Size {
            height: 2,
            width: 2,
        };
        let padding = || Size {
            height: 1,
            width: 1,
        };
        let stride = || Size {
            height: 2,
            width: 2,
        };

        let batched = im2row_transpose(&features, kernel_size(), padding(), stride()).unwrap();
        let &[n_batches, m_dim, k_dim] = batched.shape();
        assert_eq!(n_batches, 2);
        let per_batch = m_dim * k_dim;

        // every batch must equal the unbatched patches of that batch
        for b in 0..2 {
            let single =
                QantTensor4::new(&features_data[b * 48..(b + 1) * 48], [1, 3, 4, 4]).unwrap();
            let expected = im2row_transpose(&single, kernel_size(), padding(), stride()).unwrap();
            assert_eq!(expected.shape(), &[1, m_dim, k_dim]);
            assert_eq!(
                &batched.data()[b * per_batch..(b + 1) * per_batch],
                expected.data()
            );
        }
    }

    #[test]
    fn test_conv_transpose_fprop_batched() {
        let single = get_features_1_3_4_4();
        let features_data: Vec<Value> = single
            .data()
            .iter()
            .copied()
            .chain(single.data().iter().map(|&v| v * bf16::from_f32(0.5)))
            .collect();
        let features = QantTensor4::new(&features_data, [2, 3, 4, 4]).unwrap();

        let weights_data = test_utils::to_bf16s(&vec![
            1., 0., 0., 1., 0.5, 0., 0., 0., 0., 0.25, 0., 0., 1., 0., 0., 0., //
            0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., //
            0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0.,
        ]);
        let weights = QantTensor4::new(&weights_data, [3, 4, 2, 2]).unwrap();
        let attributes = ConvTransposeAttributes::new(0, 1, 1, 0).unwrap();

        let result = conv_transpose_fprop(0, &features, &weights, attributes).unwrap();
        assert_eq!(result.shape(), &[2, 4, 5, 5]);

        // every batch must equal the unbatched result of that batch
        let per_batch = 4 * 5 * 5;
        for b in 0..2 {
            let single =
                QantTensor4::new(&features_data[b * 48..(b + 1) * 48], [1, 3, 4, 4]).unwrap();
            let expected = conv_transpose_fprop(0, &single, &weights, attributes).unwrap();
            test_utils::assert_almost_equal_vec(
                &result.data()[b * per_batch..(b + 1) * per_batch],
                expected.data(),
                0.02,
            );
        }
    }

    #[test]
    fn test_attributes_reject_zero_stride() {
        assert!(ConvAttributes::new(0, 0, 1).is_err());
        assert!(ConvTransposeAttributes::new(0, 0, 1, 0).is_err());
    }

    #[test]
    fn test_upload_conv_weights_metadata() {
        let weights = QantTensor4::new(
            test_utils::to_bf16s(&vec![0.5; 1 * 2 * 3 * 4]),
            [1, 2, 3, 4],
        )
        .unwrap();
        let weights_handle = upload_conv_weights(0, &weights).unwrap();
        assert_eq!(weights_handle.npu_id(), 0);
        assert_eq!(weights_handle.shape(), [1, 2, 3, 4]);
    }

    #[test]
    fn test_upload_conv_transpose_weights_metadata() {
        let weights = QantTensor4::new(
            test_utils::to_bf16s(&vec![0.5; 1 * 2 * 3 * 4]),
            [1, 2, 3, 4],
        )
        .unwrap();
        let weights_handle = upload_conv_transpose_weights(0, &weights).unwrap();
        assert_eq!(weights_handle.npu_id(), 0);
        assert_eq!(weights_handle.shape(), [1, 2, 3, 4]);
    }
}
