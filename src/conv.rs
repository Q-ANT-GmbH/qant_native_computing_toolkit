use crate::data_format_conversion::to_bf16;
use crate::global_settings::Value;
use crate::linear::mul_mat_vec_batched;
use crate::toolkit_error;
use crate::{
    data_structures::{QantTensor3, QantTensor4},
    errors::ToolkitError,
};

#[derive(Clone, Copy)]
pub struct ConvAttributes {
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
}

#[derive(Clone, Copy)]
pub struct ConvTransposeAttributes {
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub output_padding: usize,
}

struct Size {
    height: usize,
    width: usize,
}

/// This function performs a forward pass of a 2d convolution with the provided filter on the
/// provided image
///
/// First, the required multiplications are extracted of the convolution operation.
/// Second, the multiplications are performed via the Q.ANT Native Computing Driver.
/// Third, reassemble the multiplications result and return the convolution result.
///
/// # Returns
/// - `Ok(QantTensor4)`: The resulting feature data after applying convolution.
/// - `Err(ToolkitError)`: An error if the operation fails, e.g., due to invalid dimensions or attributes.
pub fn conv_fprop<'a, 'b>(
    npu_id: u32,
    image: &QantTensor4<'a>,
    filter: &QantTensor4<'a>,
    attributes: ConvAttributes,
) -> Result<QantTensor4<'b>, ToolkitError> {
    if attributes.dilation != 1 {
        return Err(ToolkitError::from_str("dilation != 1 not implemented yet"));
    };

    let &[n_batches, image_channels, image_height, image_width] = image.shape();
    let &[
        filter_out_channels,
        filter_in_channels,
        filter_height,
        filter_width,
    ] = filter.shape();

    if image_channels != filter_in_channels {
        return Err(toolkit_error!(
            "image channels does not match filter in_channels, got {image_channels} and {filter_in_channels}",
        ));
    };

    // calc output size
    let output_height =
        (image_height + 2 * attributes.padding - filter_height) / attributes.stride + 1;
    let output_width =
        (image_width + 2 * attributes.padding - filter_width) / attributes.stride + 1;
    let output_channels = filter_out_channels;

    // im2matrix supports asymmetric kernels, padding and stride
    let kernel_size = Size {
        height: filter_height,
        width: filter_width,
    };
    let padding = Size {
        height: attributes.padding,
        width: attributes.padding,
    };
    let stride = Size {
        height: attributes.stride,
        width: attributes.stride,
    };
    let im_as_col = im2row(image, kernel_size, padding, stride)?;
    let result_data = mul_mat_vec_batched(
        npu_id,
        im_as_col.data(),
        im_as_col.shape()[1] * im_as_col.shape()[0],
        im_as_col.shape()[2],
        filter.data(),
    )?;

    // if we have batched input images, we have to sort the data after the matrix vector
    // multiplication, as the kernels are used as vector batch and therefore the result
    // has the structure: output_channel x image_batch x output_height x output_width
    // -> image_batch x output_channel x output_height x output_width
    let result_data = swap_channels_batches(
        result_data,
        output_channels,
        image.shape()[0],
        output_height,
        output_width,
    );

    QantTensor4::new(
        result_data,
        [n_batches, output_channels, output_height, output_width],
    )
}

// translates a multidimensional image into a matrix for matrix-vec-based convolution
// each kernel patch is in a separate row (im2row, an alternative to im2col)
fn im2row<'a>(
    image: &QantTensor4<'a>,
    kernel_size: Size,
    padding: Size,
    stride: Size,
) -> Result<QantTensor3<'a>, ToolkitError> {
    let &[n_batches, image_channels, image_height, image_width] = image.shape();

    // The final matrix will have the size [kernel_position x (kernel_pixel_position * channel_in)]
    // Each row will be summed up in the end, therefore all kernel pixels over all input
    // channels should be in a row
    let kernel_pixel_num = kernel_size.width * kernel_size.height;
    let width_col = kernel_pixel_num * image_channels;
    // calc output size
    let kernels_per_height =
        (image_height + 2 * padding.height - kernel_size.height) / stride.height + 1;
    let kernels_per_width =
        (image_width + 2 * padding.width - kernel_size.width) / stride.width + 1;
    let height_col = kernels_per_width * kernels_per_height;

    let per_batch_size = width_col * height_col;
    let mat_data_size = per_batch_size * n_batches;

    // With this pattern, we do not write zeros to every position first.
    // As we are overwriting every value, this is not required
    let mut mat_data = vec![to_bf16(0.0); mat_data_size];

    for i in 0..mat_data.len() {
        let batch_id = i / per_batch_size;
        let i = i % per_batch_size;
        let kernel_col = i / width_col;
        let kernel_pixel = i % (kernel_pixel_num);
        let chl_in = i % width_col / kernel_pixel_num;
        let chl_pos_in = image_width * image_height * (chl_in + batch_id * image_channels);

        let w_out = kernel_col % kernels_per_width;
        let h_out = kernel_col / kernels_per_width;

        let k_y = kernel_pixel / kernel_size.width;
        let k_x = kernel_pixel % kernel_size.width;

        let h_in = (h_out * stride.height) as i64 - padding.height as i64;
        let w_in = (w_out * stride.width) as i64 - padding.width as i64;

        let h = h_in + k_y as i64;
        let w = w_in + k_x as i64;

        // is the pixel inside the picture
        let valid_pos = h >= 0 && w >= 0 && h < image_height as i64 && w < image_width as i64;
        let val = if valid_pos {
            image.data()[chl_pos_in + h as usize * image_width + w as usize]
        } else {
            to_bf16(0.0)
        };

        // row major
        let col = chl_in * kernel_pixel_num + kernel_size.width * k_y + k_x;

        mat_data[(batch_id * per_batch_size) + kernel_col * width_col + col] = val;
    }

    let mat_feature_data = QantTensor3::new(mat_data, [n_batches, height_col, width_col])?;
    Ok(mat_feature_data)
}

fn swap_channels_batches(
    data: Vec<Value>,
    channels: usize,
    batches: usize,
    height: usize,
    width: usize,
) -> Vec<Value> {
    if batches == 1 || channels == 1 {
        return data;
    }

    let mut swapped = vec![data[0]; data.len()];
    for c in 0..channels {
        for b in 0..batches {
            for h in 0..height {
                for w in 0..width {
                    // Original index (C × B × H × W)
                    let src_idx = ((c * batches + b) * height + h) * width + w;

                    // Swapped index (B × C × H × W)
                    let dst_idx = ((b * channels + c) * height + h) * width + w;

                    swapped[dst_idx] = data[src_idx];
                }
            }
        }
    }
    swapped
}

pub fn conv_transpose_fprop<'a, 'b>(
    npu_id: u32,
    image: &QantTensor4<'a>,
    filter: &QantTensor4<'a>,
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

    let &[n_batches, image_channels, image_height, image_width] = image.shape();
    let &[
        filter_in_channels,
        filter_out_channels,
        filter_height,
        filter_width,
    ] = filter.shape();

    if image_channels != filter_in_channels {
        return Err(toolkit_error!(
            "image channels does not match filter in_channels, got {image_channels} and {filter_in_channels}",
        ));
    }
    if n_batches > 1 {
        return Err(ToolkitError::from_str("does not support batched input."));
    }

    // Output size for conv-transpose
    let output_height = (image_height - 1) * attributes.stride - 2 * attributes.padding
        + attributes.dilation * (filter_height - 1)
        + attributes.output_padding
        + 1;
    let output_width = (image_width - 1) * attributes.stride - 2 * attributes.padding
        + attributes.dilation * (filter_width - 1)
        + attributes.output_padding
        + 1;
    let output_channels = filter_out_channels;

    // -----------------------------
    // 1. Build U: (M x K) "image-row" for TConv
    //    M = H_out * W_out, K = Cin*Kh*Kw.
    //    Each row corresponds to one (oy, ox); each column is a (cin, ky, kx) position.
    // -----------------------------
    // im2matrix_transposed supports asymmetric kernels, padding and stride
    let kernel_size = Size {
        height: filter_height,
        width: filter_width,
    };
    let padding = Size {
        height: attributes.padding,
        width: attributes.padding,
    };
    let stride = Size {
        height: attributes.stride,
        width: attributes.stride,
    };
    let mat_data = im2row_transpose(image, kernel_size, padding, stride)?;

    // Reorder filter from (C_in, C_out, H, W) — the public API convention — to
    // (C_out, C_in, H, W) that mul_mat_vec_batched expects (one contiguous vector per output channel).
    // The H*W spatial block is innermost in both layouts, so each iteration is a contiguous memcpy.
    let filter_reordered: Vec<Value> = {
        let src = filter.data();
        let hw = filter_height * filter_width;
        let mut out = vec![Value::default(); src.len()];
        for c_in in 0..filter_in_channels {
            for c_out in 0..filter_out_channels {
                let src_start = c_in * (filter_out_channels * hw) + c_out * hw;
                let dst_start = c_out * (filter_in_channels * hw) + c_in * hw;
                out[dst_start..dst_start + hw].copy_from_slice(&src[src_start..src_start + hw]);
            }
        }
        out
    };

    // -----------------------------
    // 3. Batched Matrix Vector Multiplication: C = [U (M x K) * Vs (K)] N-times => (N x M)
    // -----------------------------
    let result_mat = mul_mat_vec_batched(
        npu_id,
        mat_data.data(),
        mat_data.shape()[1] * mat_data.shape()[0],
        mat_data.shape()[2],
        &filter_reordered,
    )?;

    // if we have batched input images, we have to sort the data after the matrix vector
    // multiplication, as the kernels are used as vector batch and therefore the result
    // has the structure: output_channel x image_batch x output_height x output_width
    // -> image_batch x output_channel x output_height x output_width
    let result_data = swap_channels_batches(
        result_mat,
        output_channels,
        image.shape()[0],
        output_height,
        output_width,
    );

    QantTensor4::new(
        result_data,
        [n_batches, output_channels, output_height, output_width],
    )
}

// translates a multidimensional image into a matrix for matrix-vec-based transpose convolution
// each kernel patch is in a separate row (im2row, an alternative to im2col)
fn im2row_transpose<'a>(
    image: &QantTensor4<'a>,
    kernel_size: Size,
    padding: Size,
    stride: Size,
) -> Result<QantTensor3<'a>, ToolkitError> {
    let &[n_batches, image_channels, image_height, image_width] = image.shape();
    let image_size = image_width * image_height;
    let kernel_pixel_num = kernel_size.width * kernel_size.height;
    let output_padding = 0;
    let dilation = 1;
    // Output size for conv-transpose
    let output_height = (image_height - 1) * stride.height - 2 * padding.height
        + dilation * (kernel_size.height - 1)
        + output_padding
        + 1;
    let output_width = (image_width - 1) * stride.width - 2 * padding.width
        + dilation * (kernel_size.width - 1)
        + output_padding
        + 1;
    let filters_per_in_chl = output_width * output_height;
    let k_dim = image_channels * kernel_pixel_num;
    let per_batch_size = output_width * output_height;
    let m_dim = per_batch_size * n_batches;
    let mut mat_data = vec![to_bf16(0.0); k_dim * m_dim];

    for i in 0..mat_data.len() {
        let batch_id = i / (per_batch_size * k_dim);
        let cin = (i / kernel_pixel_num) % image_channels;
        let kx = i % kernel_size.width;
        let ky = (i / kernel_size.width) % kernel_size.height;
        let filter_iteration = (i / (image_channels * kernel_pixel_num)) % filters_per_in_chl;

        // Calculate output position coordinates
        let ox = filter_iteration % output_width;
        let oy = filter_iteration / output_width;

        // Apply transpose convolution coordinate transformation
        let img_x = (ox as i32 + padding.width as i32 - kx as i32) / stride.width as i32;
        let img_y = (oy as i32 + padding.height as i32 - ky as i32) / stride.height as i32;

        // Check for valid indices
        let val = if img_x >= 0 || img_y >= 0 {
            let img_x = img_x as usize;
            let img_y = img_y as usize;

            // Verify exact strided positioning
            if (ox + padding.width) >= kx
                && (oy + padding.height) >= ky
                && (ox + padding.width - kx).is_multiple_of(stride.width)
                && (oy + padding.height - ky).is_multiple_of(stride.height)
                && img_x < image_width
                && img_y < image_height
            {
                // only use the real image data, when we are within the image
                // otherwise the default padding is zero
                image.data()[batch_id * image_channels * image_size
                    + cin * image_size
                    + img_y * image_width
                    + img_x]
            } else {
                // Zero padding
                to_bf16(0.0)
            }
        } else {
            // Zero padding
            to_bf16(0.0)
        };

        let row = oy * output_width + ox;
        let col_offset = cin * kernel_pixel_num + ky * kernel_size.width + kx;
        mat_data[row * k_dim + col_offset] = val;
    }

    let mat_tensor = QantTensor3::new(mat_data, [n_batches, m_dim, k_dim])?;
    Ok(mat_tensor)
}

#[cfg(test)]
mod tests {
    use crate::{data_format_conversion::to_bf16s, utils::test_utils::assert_almost_equal_vec};

    use super::*;

    fn get_image_1_1_4_4<'a>() -> QantTensor4<'a> {
        // 0.1, 0.2, 0.3, 0.4
        // 0.5, 0.6, 0.7, 0.8
        // 0.9, 1., 0.11, 0.21
        // 0.31, 0.41, 0.51, 0.61
        let image_data = to_bf16s(&vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 0.11, 0.21, 0.31, 0.41, 0.51, 0.61,
        ]);
        QantTensor4::new(image_data, [1, 1, 4, 4]).unwrap()
    }
    fn get_image_1_3_4_4<'a>() -> QantTensor4<'a> {
        let image_data = to_bf16s(&vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 0.11, 0.21, 0.31, 0.41, 0.51,
            0.61, //
            0.101, 0.201, 0.301, 0.401, 0.501, 0.601, 0.701, 0.801, 0.901, 1.001, 0.111, 0.211,
            0.311, 0.411, 0.511, 0.611, //
            0.102, 0.202, 0.302, 0.402, 0.502, 0.602, 0.702, 0.802, 0.902, 1.002, 0.112, 0.212,
            0.312, 0.412, 0.512, 0.612,
        ]);
        QantTensor4::new(image_data, [1, 3, 4, 4]).unwrap()
    }

    fn get_image_1_3_5_4<'a>() -> QantTensor4<'a> {
        let image_data = to_bf16s(&vec![
            0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14,
            0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.220, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28,
            0.29, 0.3, 0.310, 0.320, 0.330, 0.340, 0.350, 0.360, 0.370, 0.380, 0.390, 0.4, 0.410,
            0.420, 0.430, 0.440, 0.450, 0.460, 0.470, 0.480, 0.490, 0.5, 0.510, 0.520, 0.530,
            0.540, 0.550, 0.560, 0.570, 0.580, 0.590,
        ]);
        QantTensor4::new(image_data, [1, 3, 5, 4]).unwrap()
    }

    #[test]

    fn test_conv_fprop_simple_kernel() {
        let image = get_image_1_1_4_4();

        // Create filter
        let filter_data = vec![1, 0, 0, 0];
        let filter = QantTensor4::new(to_bf16s(&filter_data), [1, 1, 2, 2]).unwrap();

        // Set convolution attributes
        let attributes = ConvAttributes {
            padding: 0,
            stride: 1,
            dilation: 1,
        };

        // Perform convolution
        let result = conv_fprop(0, &image, &filter, attributes).unwrap();

        // Expected output
        let expected_data = vec![0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.9, 1., 0.110];
        let expected_output = QantTensor4::new(to_bf16s(&expected_data), [1, 1, 3, 3]).unwrap();

        // Assert the results
        assert_eq!(result.shape(), expected_output.shape());
        assert_almost_equal_vec(result.data(), expected_output.data(), 0.02);
    }

    #[test]
    fn test_conv_fprop() {
        let image = get_image_1_1_4_4();

        // Create filter
        let filter_data = vec![1., 0., 0., 1.];
        let filter = QantTensor4::new(to_bf16s(&filter_data), [1, 1, 2, 2]).unwrap();

        // Set convolution attributes
        let attributes = ConvAttributes {
            padding: 0,
            stride: 1,
            dilation: 1,
        };

        // Perform convolution
        let result = conv_fprop(0, &image, &filter, attributes).unwrap();

        // Expected output
        let expected_data = vec![0.7, 0.9, 1.1, 1.5, 0.71, 0.91, 1.31, 1.51, 0.72];
        let expected_output = QantTensor4::new(to_bf16s(&expected_data), [1, 1, 3, 3]).unwrap();

        // Assert the results
        assert_eq!(result.shape(), expected_output.shape());
        assert_almost_equal_vec(result.data(), expected_output.data(), 0.02);
    }

    #[test]
    fn test_conv_fprop_padding_1() {
        let image = get_image_1_1_4_4();

        // Create filter
        let filter_data = to_bf16s(&vec![1., 0., 0., 1.]);
        let filter = QantTensor4::new(&filter_data, [1, 1, 2, 2]).unwrap();

        // Set convolution attributes
        let attributes = ConvAttributes {
            padding: 1,
            stride: 1,
            dilation: 1,
        };

        // Perform convolution
        let result = conv_fprop(0, &image, &filter, attributes).unwrap();

        // Expected output
        let expected_data = to_bf16s(&vec![
            0.1, 0.2, 0.3, 0.4, 0., 0.5, 0.7, 0.9, 1.1, 0.4, 0.9, 1.5, 0.71, 0.91, 0.8, 0.31, 1.31,
            1.51, 0.72, 0.21, 0., 0.31, 0.41, 0.51, 0.61,
        ]);
        let expected_output = QantTensor4::new(expected_data, [1, 1, 5, 5]).unwrap();

        // Assert the results
        assert_eq!(result.shape(), expected_output.shape());
        assert_almost_equal_vec(result.data(), expected_output.data(), 0.02);
    }

    #[test]

    fn test_conv_fprop_multichannel() {
        let image = get_image_1_3_4_4();

        // Create filter
        let filter_data = to_bf16s(&vec![
            1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., // -
            0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., // -
            0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., // -
            1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., // -
        ]);
        let filter = QantTensor4::new(&filter_data, [4, 3, 2, 2]).unwrap();

        // Set convolution attributes
        let attributes = ConvAttributes {
            padding: 0,
            stride: 1,
            dilation: 1,
        };

        // Perform convolution
        let result = conv_fprop(0, &image, &filter, attributes).unwrap();

        // Expected output
        let expected_data = to_bf16s(&vec![
            0.7, 0.9, 1.100, 1.500, 0.710, 0.910, 1.310, 1.510, 0.720, // -
            0.702, 0.902, 1.102, 1.502, 0.712, 0.912, 1.312, 1.512, 0.722, // -
            0.704, 0.904, 1.104, 1.504, 0.714, 0.914, 1.314, 1.514, 0.724, // -
            0.803, 1.103, 1.403, 2.003, 2.303, 1.613, 2.213, 1.523, 0.833, // -
        ]);
        let expected_output = QantTensor4::new(expected_data, [1, 4, 3, 3]).unwrap();

        // Assert the results
        assert_eq!(result.shape(), expected_output.shape());
        assert_almost_equal_vec(result.data(), expected_output.data(), 0.02);
    }

    #[test]
    fn test_conv_fprop_stride2_simple_kernel() {
        let image = get_image_1_1_4_4();

        // Create filter
        let filter_data = to_bf16s(&vec![1., 0., 0., 1.]);
        let filter = QantTensor4::new(&filter_data, [1, 1, 2, 2]).unwrap();

        // Set convolution attributes
        let attributes = ConvAttributes {
            padding: 0,
            stride: 2,
            dilation: 1,
        };

        // Perform convolution
        let result = conv_fprop(0, &image, &filter, attributes).unwrap();

        // Expected output
        let expected_data = to_bf16s(&vec![0.7, 1.100, 1.310, 0.720]);
        let expected_output = QantTensor4::new(expected_data, [1, 1, 2, 2]).unwrap();

        // Assert the results
        assert_eq!(result.shape(), expected_output.shape());
        assert_almost_equal_vec(result.data(), expected_output.data(), 0.02);
    }

    #[test]
    fn test_conv_fprop_3x4x5_input_2x2x3_filter_simple() {
        let input = get_image_1_3_5_4();

        // Create filter data (2x3x2)
        let filter_data = to_bf16s(&vec![
            0.1, 0., 0., 0., 0., 0., // --
            0., 0., 0., 0., 0., 0., // --
            0., 0., 0., 0., 0., 0., // --
            0., 0., 0., 0., 0., 0., // --
            0., 0., 0., 0., 0., 0., // --
            0., 0., 0., 0., 0., 0.,
        ]);
        let filter = QantTensor4::new(&filter_data, [2, 3, 2, 3]).unwrap();

        // Set convolution attributes
        let attributes = ConvAttributes {
            padding: 0,
            stride: 1,
            dilation: 1,
        };

        // Perform convolution
        let result = conv_fprop(0, &input, &filter, attributes).unwrap();

        // Expected output dimensions
        assert_eq!(result.shape().to_vec(), [1, 2, 4, 2]);

        // Expected output data
        // Note: You'll need to calculate the expected values based on your convolution implementation
        let expected_output = to_bf16s(&vec![
            0., 0.001, 0.004, 0.005, 0.008, 0.009, 0.012, 0.013, 0., 0., 0., 0., 0., 0., 0., 0.,
        ]);
        assert_eq!(result.len(), expected_output.len());

        assert_almost_equal_vec(result.data(), &expected_output, 0.001);
    }

    #[test]

    fn test_conv_fprop_3x4x5_input_2x2x3_filter() {
        let input = get_image_1_3_5_4();

        // Create filter data (2x3x2)
        let filter_data = to_bf16s(&vec![
            0.1, 0.125, 0.150, 0.175, 0.2, 0.225, 0.250, 0.275, 0.3, 0.325, 0.350, 0.375, 0.4,
            0.425, 0.450, 0.475, 0.5, 0.525, //
            0.550, 0.575, 0.600, 0.625, 0.650, 0.675, 0.7, 0.725, 0.750, 0.775, 0.800, 0.825,
            0.850, 0.875, 0.9, 0.925, 0.950, 0.975,
        ]);
        let filter = QantTensor4::new(&filter_data, [2, 3, 2, 3]).unwrap();

        // Set convolution attributes
        let attributes = ConvAttributes {
            padding: 0,
            stride: 1,
            dilation: 1,
        };

        // Perform convolution
        let result = conv_fprop(0, &input, &filter, attributes).unwrap();

        // Expected output dimensions
        assert_eq!(result.shape().to_vec(), [1, 2, 4, 2]);

        // Expected output data
        // Note: You'll need to calculate the expected values based on your convolution implementation
        let expected_output = to_bf16s(&vec![
            1.670, 1.726, 1.895, 1.951, 2.120, 2.176, 2.345, 2.401, 3.533, 3.670, 4.082, 4.219,
            4.631, 4.768, 5.180, 5.317,
        ]);
        assert_eq!(result.len(), expected_output.len());
        assert_almost_equal_vec(result.data(), &expected_output, 0.03);
    }

    #[test]

    fn test_conv_transpose_fprop_multichannel() {
        let image = get_image_1_3_4_4();

        // Create filter in (C_in, C_out, H, W) convention
        let filter_data = to_bf16s(&vec![
            1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
            0., // c_in=0: c_out=0,1,2,3 each (h,w)
            0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.,
            0., // c_in=1: c_out=0,1,2,3
            0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1.,
            0., // c_in=2: c_out=0,1,2,3
        ]);
        let filter = QantTensor4::new(&filter_data, [3, 4, 2, 2]).unwrap();

        // Set convolution attributes
        let attributes = ConvTransposeAttributes {
            padding: 0,
            stride: 1,
            dilation: 1,
            output_padding: 0,
        };

        // Perform convolution
        let result = conv_transpose_fprop(0, &image, &filter, attributes).unwrap();

        // Expected output
        let expected_data = to_bf16s(&vec![
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
        assert_almost_equal_vec(result.data(), expected_output.data(), 0.02);
    }

    #[test]

    fn test_conv_transpose_fprop_simple() {
        let image = get_image_1_3_4_4();
        // Create filter
        let filter_data = to_bf16s(&vec![
            1., 0., 0., 1., // c_in=0: c_out=0,1,2,3
            0., 1., 0., 1., // c_in=1: c_out=0,1,2,3
            0., 0., 1., 1., // c_in=2: c_out=0,1,2,3
        ]);
        let filter = QantTensor4::new(&filter_data, [3, 4, 1, 1]).unwrap();

        // Set convolution attributes
        let attributes = ConvTransposeAttributes {
            padding: 0,
            stride: 1,
            dilation: 1,
            output_padding: 0,
        };

        // Perform convolution
        let result = conv_transpose_fprop(0, &image, &filter, attributes).unwrap();

        // Expected output
        let expected_data = to_bf16s(&vec![
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
        assert_almost_equal_vec(result.data(), expected_output.data(), 0.02);
    }

    #[test]
    fn test_im2row_0() {
        let image_data = to_bf16s(&vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let image = QantTensor4::new(image_data, [1, 1, 3, 3]).unwrap();
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

        let col_data = im2row(&image, kernel_size, padding, stride).unwrap();
        assert_eq!(col_data.shape().to_vec(), [1, 4, 4]);

        let expected = to_bf16s(&vec![1, 2, 4, 5, 2, 3, 5, 6, 4, 5, 7, 8, 5, 6, 8, 9]);
        assert_eq!(col_data.data(), expected);
    }

    #[test]
    fn test_im2row_1() {
        let image_data = to_bf16s(&vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 15, 25, 35, 45, 55, 65,
            75, 85, 95,
        ]);
        let image = QantTensor4::new(image_data, [1, 3, 3, 3]).unwrap();
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

        let col_data = im2row(&image, kernel_size, padding, stride).unwrap();
        assert_eq!(col_data.shape().to_vec(), [1, 4, 4 * 3]);

        let expected = to_bf16s(&vec![
            1, 2, 4, 5, 10, 20, 40, 50, 15, 25, 45, 55, 2, 3, 5, 6, 20, 30, 50, 60, 25, 35, 55, 65,
            4, 5, 7, 8, 40, 50, 70, 80, 45, 55, 75, 85, 5, 6, 8, 9, 50, 60, 80, 90, 55, 65, 85, 95,
        ]);
        assert_eq!(col_data.data(), expected);
    }

    #[test]
    fn test_im2row_2() {
        let image = get_image_1_3_4_4();
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

        let col_data = im2row(&image, kernel_size, padding, stride).unwrap();
        assert_eq!(col_data.shape().to_vec(), [1, 9, 4 * 3]);

        let expected = to_bf16s(&vec![
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
        let image_data = to_bf16s(&vec![
            100, 200, 300, 101, 201, 301, 102, 202, 302, 103, 203, 303,
        ]);
        let image = QantTensor4::new(image_data, [4, 3, 1, 1]).unwrap();
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

        let col_data = im2row(&image, kernel_size, padding, stride).unwrap();
        assert_eq!(col_data.shape().to_vec(), [4, 1, 3]);

        let expected = to_bf16s(&vec![
            100, 200, 300, 101, 201, 301, 102, 202, 302, 103, 203, 303,
        ]);
        assert_eq!(col_data.data(), expected);
    }

    #[test]
    fn test_im2row_batched_2() {
        let image_data = to_bf16s(&vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 0.11, 0.21, 0.31, 0.41, 0.51,
            0.61, // --
            0.101, 0.201, 0.301, 0.401, 0.501, 0.601, 0.701, 0.801, 0.901, 1.001, 0.111, 0.211,
            0.311, 0.411, 0.511, 0.611, // --
            0.102, 0.202, 0.302, 0.402, 0.502, 0.602, 0.702, 0.802, 0.902, 1.002, 0.112, 0.212,
            0.312, 0.412, 0.512, 0.612,
        ]);
        let image = QantTensor4::new(image_data, [3, 1, 4, 4]).unwrap();
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

        let col_data = im2row(&image, kernel_size, padding, stride).unwrap();
        assert_eq!(col_data.shape().to_vec(), [3, 9, 4]);
        let expected = to_bf16s(&vec![
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
}
