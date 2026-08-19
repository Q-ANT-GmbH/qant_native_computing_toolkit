use half::bf16;

use crate::{data_structures::QantTensor4, errors::ToolkitError, toolkit_error};
use rayon::prelude::*;

#[derive(Clone, Copy)]
pub struct PoolingAttributes {
    pub kernel_height: usize,
    pub kernel_width: usize,
    pub stride: usize,
    pub padding: usize,
    pub count_include_pad: bool,
}

#[derive(Clone, Copy)]
enum PoolingMode {
    Average,
    Maximum,
}

impl std::fmt::Display for PoolingMode {
    fn fmt(&self, format: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            PoolingMode::Average => "avg",
            PoolingMode::Maximum => "max",
        };
        write!(format, "{text}")
    }
}

/// Performs 2D pooling on the input feature data.
///
/// # Parameters
/// - `image`: A reference to the input feature data (`QantTensor4`) to be processed.
///   This represents the image or feature map on which pooling will be applied.
/// - `attributes`: A `PoolingAttributes` struct containing the parameters for the pooling operation:
///   - `kernel_height`: The height of the pooling kernel.
///   - `kernel_width`: The width of the pooling kernel.
///   - `stride`: The stride of the pooling operation (how far the kernel moves after each step).
///   - `padding`: The number of pixels added around the edges of the input feature map.
///   - `count_include_pad`: When True, will include the zero-padding in the averaging calculation, if mode is `Average`.
/// - `mode`: Specifies whether to apply average or max pooling to the input data.
///
/// # Returns
/// - `Ok(QantTensor4)`: The resulting feature data after applying pooling.
/// - `Err(ToolkitError)`: An error if the operation fails, such as due to invalid dimensions or attributes.
fn pool2d_fprop<'b>(
    _npu_id: u32,
    image: &QantTensor4<'_>,
    attributes: PoolingAttributes,
    mode: PoolingMode,
) -> Result<QantTensor4<'b>, ToolkitError> {
    let &[n_batches, data_channels, data_height, data_width] = image.shape();

    if n_batches > 1 {
        return Err(ToolkitError::from_str("does not support batched input."));
    }

    let output_height =
        (data_height + 2 * attributes.padding - attributes.kernel_height) / attributes.stride + 1;
    let output_width =
        (data_width + 2 * attributes.padding - attributes.kernel_width) / attributes.stride + 1;
    let output_channels = data_channels;

    let image_size = data_width * data_height;
    let image_data = image.data();
    let padding = attributes.padding;
    let stride = attributes.stride;
    let kh = attributes.kernel_height;
    let kw = attributes.kernel_width;
    let count_include_pad = attributes.count_include_pad;
    let output_spatial = output_height * output_width;

    let result_data = (0..output_channels * output_spatial)
        .into_par_iter()
        .map(|p| -> Result<bf16, ToolkitError> {
            let ch = p / output_spatial;
            let oy = (p % output_spatial) / output_width;
            let ox = (p % output_spatial) % output_width;

            let mut sum = 0.0f32;
            let mut count = 0usize;
            let mut max_val: Option<bf16> = None;

            for ky in 0..kh {
                for kx in 0..kw {
                    let img_x = ox * stride + kx;
                    let img_y = oy * stride + ky;

                    if padding > 0
                        && (img_x < padding
                            || img_y < padding
                            || img_x >= data_width + padding
                            || img_y >= data_height + padding)
                    {
                        if matches!(mode, PoolingMode::Average) && count_include_pad {
                            count += 1;
                        }
                        continue;
                    }

                    let real_x = img_x - padding;
                    let real_y = img_y - padding;

                    let val = image_data[ch * image_size + real_y * data_width + real_x];
                    sum += f32::from(val);
                    count += 1;
                    max_val = Some(match max_val {
                        None => val,
                        Some(m) => {
                            if val.total_cmp(&m).is_gt() {
                                val
                            } else {
                                m
                            }
                        }
                    });
                }
            }

            if count == 0 {
                return Err(toolkit_error!(
                    "pool window at output position {ch},{oy},{ox} contains no real pixels \
                     (window falls entirely in padding)"
                ));
            }
            match mode {
                PoolingMode::Average => Ok(bf16::from_f32(sum / count as f32)),
                PoolingMode::Maximum => max_val.ok_or_else(|| {
                    toolkit_error!(
                        "unable to find maximum in pool window at output position {ch},{oy},{ox}"
                    )
                }),
            }
        })
        .collect::<Result<Vec<bf16>, ToolkitError>>()?;

    QantTensor4::new(
        result_data,
        [
            image.shape()[0],
            output_channels,
            output_height,
            output_width,
        ],
    )
}

/// Performs 2D adaptive pooling on the input feature data. Only works for symmetric input and output sizes.
/// Input size has to be integer multiple of output size.
///
/// # Parameters
/// - `image`: A reference to the input feature data (`QantTensor4`) to be processed.
///   This represents the image or feature map on which pooling will be applied.
/// - `output_height`: The target output height, has to be equal to `output_width`.
/// - `output_width`: The target output width, has to be equal to `output_height`.
/// - `mode`: Specifies whether to apply average or max pooling to the input data.
///
/// # Returns
/// - `Ok(QantTensor4)`: The resulting feature data after applying pooling.
/// - `Err(ToolkitError)`: An error if the operation fails, such as due to invalid dimensions or attributes.
fn adaptivepool2d_fprop<'b>(
    _npu_id: u32,
    image: &QantTensor4<'_>,
    output_height: usize,
    output_width: usize,
    mode: PoolingMode,
) -> Result<QantTensor4<'b>, ToolkitError> {
    let [_, _, data_height, data_width] = image.shape();

    if data_height != data_width {
        return Err(toolkit_error!(
            "input must be symmetric, got ({data_height}, {data_width})"
        ));
    }

    if output_height != output_width {
        return Err(toolkit_error!(
            "output must be symmetric, got ({output_height}, {output_width})"
        ));
    }

    if data_height % output_height != 0 {
        return Err(toolkit_error!(
            "input size must be multiple integer of output size, got {data_height} and {output_height}"
        ));
    }

    let stride = data_height / output_height;
    let kernel_size = data_height - (output_height - 1) * stride;
    let attributes = PoolingAttributes {
        kernel_height: kernel_size,
        kernel_width: kernel_size,
        stride,
        padding: 0,
        count_include_pad: false,
    };
    pool2d_fprop(_npu_id, image, attributes, mode)
}

/// Performs 2D max pooling on the input feature data.
///
/// # Parameters
/// - `image`: A reference to the input feature data (`QantTensor4`) to be processed.
///   This represents the image or feature map on which max pooling will be applied.
/// - `attributes`: A `PoolingAttributes` struct containing the parameters for the pooling operation:
///   - `kernel_height`: The height of the pooling kernel.
///   - `kernel_width`: The width of the pooling kernel.
///   - `stride`: The stride of the pooling operation (how far the kernel moves after each step).
///   - `padding`: The number of pixels added around the edges of the input feature map.
///   - `count_include_pad`: Ignored here, just for compatibility with avgpool.
///
/// # Returns
/// - `Ok(QantTensor4)`: The resulting feature data after applying max pooling.
/// - `Err(ToolkitError)`: An error if the operation fails, such as due to invalid dimensions or attributes.
pub fn maxpool2d_fprop<'b>(
    _npu_id: u32,
    image: &QantTensor4<'_>,
    attributes: PoolingAttributes,
) -> Result<QantTensor4<'b>, ToolkitError> {
    pool2d_fprop(_npu_id, image, attributes, PoolingMode::Maximum)
}

/// Performs 2D avg pooling on the input feature data.
///
/// # Parameters
/// - `image`: A reference to the input feature data (`QantTensor4`) to be processed.
///   This represents the image or feature map on which avg pooling will be applied.
/// - `attributes`: A `PoolingAttributes` struct containing the parameters for the pooling operation:
///   - `kernel_height`: The height of the pooling kernel.
///   - `kernel_width`: The width of the pooling kernel.
///   - `stride`: The stride of the pooling operation (how far the kernel moves after each step).
///   - `padding`: The number of pixels added around the edges of the input feature map.
///   - `count_include_pad`: When True, will include the zero-padding in the averaging calculation
///
/// # Returns
/// - `Ok(QantTensor4)`: The resulting feature data after applying avg pooling.
/// - `Err(ToolkitError)`: An error if the operation fails, such as due to invalid dimensions or attributes.
pub fn avgpool2d_fprop<'b>(
    _npu_id: u32,
    image: &QantTensor4<'_>,
    attributes: PoolingAttributes,
) -> Result<QantTensor4<'b>, ToolkitError> {
    pool2d_fprop(_npu_id, image, attributes, PoolingMode::Average)
}

/// Performs 2D max adaptive pooling on the input feature data. Only works for symmetric input and output sizes.
/// Input size has to be integer multiple of output size.
///
/// # Parameters
/// - `image`: A reference to the input feature data (`QantTensor4`) to be processed.
///   This represents the image or feature map on which pooling will be applied.
/// - `output_height`: The target output height, has to be equal to `output_width`.
/// - `output_width`: The target output width, has to be equal to `output_height`.
///
/// # Returns
/// - `Ok(QantTensor4)`: The resulting feature data after applying pooling.
/// - `Err(ToolkitError)`: An error if the operation fails, such as due to invalid dimensions or attributes.
pub fn adaptive_maxpool2d_fprop<'b>(
    _npu_id: u32,
    image: &QantTensor4<'_>,
    output_height: usize,
    output_width: usize,
) -> Result<QantTensor4<'b>, ToolkitError> {
    adaptivepool2d_fprop(
        _npu_id,
        image,
        output_height,
        output_width,
        PoolingMode::Maximum,
    )
}

/// Performs 2D adaptive avg pooling on the input feature data. Only works for symmetric input and output sizes.
/// Input size has to be integer multiple of output size.
///
/// # Parameters
/// - `image`: A reference to the input feature data (`QantTensor4`) to be processed.
///   This represents the image or feature map on which pooling will be applied.
/// - `output_height`: The target output height, has to be equal to `output_width`.
/// - `output_width`: The target output width, has to be equal to `output_height`.
///
/// # Returns
/// - `Ok(QantTensor4)`: The resulting feature data after applying pooling.
/// - `Err(ToolkitError)`: An error if the operation fails, such as due to invalid dimensions or attributes.
pub fn adaptive_avgpool2d_fprop<'b>(
    _npu_id: u32,
    image: &QantTensor4<'_>,
    output_height: usize,
    output_width: usize,
) -> Result<QantTensor4<'b>, ToolkitError> {
    adaptivepool2d_fprop(
        _npu_id,
        image,
        output_height,
        output_width,
        PoolingMode::Average,
    )
}

#[cfg(test)]
mod tests {
    use crate::{
        data_format_conversion::to_bf16s, global_settings::Value,
        utils::test_utils::assert_almost_equal_vec,
    };

    use super::*;

    fn get_default_img_data() -> Vec<Value> {
        to_bf16s(&vec![
            0.1, 0.2, 0.3, 0.4, //
            0.5, 0.6, 0.7, 0.8, //
            0.9, 1., 0.11, 0.21, //
            0.31, 0.41, 0.51, 0.61,
        ])
    }

    #[test]
    fn test_pool2d_fprop_0() {
        // Create input image
        let image_data = get_default_img_data();
        let image = QantTensor4::new(&image_data, [1, 1, 4, 4]).unwrap();

        // Set convolution attributes
        let attributes = PoolingAttributes {
            kernel_height: 2,
            kernel_width: 2,
            padding: 0,
            stride: 1,
            count_include_pad: false,
        };

        // Perform convolution
        let result_max = maxpool2d_fprop(0, &image, attributes).unwrap();
        let result_avg = avgpool2d_fprop(0, &image, attributes).unwrap();

        // Expected output
        let expected_data_max = to_bf16s(&vec![0.6, 0.7, 0.8, 1., 1., 0.8, 1., 1., 0.61]);
        let expected_data_avg = to_bf16s(&vec![
            0.35, 0.45, 0.55, 0.75, 0.602, 0.455, 0.655, 0.507, 0.36,
        ]);
        let expected_output_max = QantTensor4::new(expected_data_max, [1, 1, 3, 3]).unwrap();
        let expected_output_avg = QantTensor4::new(expected_data_avg, [1, 1, 3, 3]).unwrap();

        // Assert the results
        assert_eq!(result_max.shape(), expected_output_max.shape());
        assert_almost_equal_vec(result_max.data(), expected_output_max.data(), 0.01);

        assert_eq!(result_avg.shape(), expected_output_avg.shape());
        assert_almost_equal_vec(result_avg.data(), expected_output_avg.data(), 0.01);
    }

    #[test]
    fn test_pool2d_fprop_1() {
        // Create input image
        let image_data = get_default_img_data();
        let image = QantTensor4::new(&image_data, [1, 1, 4, 4]).unwrap();

        // Set convolution attributes
        let attributes = PoolingAttributes {
            kernel_height: 2,
            kernel_width: 2,
            padding: 0,
            stride: 2,
            count_include_pad: false,
        };

        // Perform convolution
        let result_max = maxpool2d_fprop(0, &image, attributes).unwrap();
        let result_avg = avgpool2d_fprop(0, &image, attributes).unwrap();

        // Expected output (max value in each pooling kernel)
        // 4x4 input with 2x2 kernels
        let expected_data_max = to_bf16s(&vec![0.6, 0.8, 1., 0.61]);
        let expected_data_avg = to_bf16s(&vec![0.35, 0.55, 0.655, 0.36]);
        let expected_output_max = QantTensor4::new(expected_data_max, [1, 1, 2, 2]).unwrap();
        let expected_output_avg = QantTensor4::new(expected_data_avg, [1, 1, 2, 2]).unwrap();

        // Assert the results
        assert_eq!(result_max.shape(), expected_output_max.shape());
        assert_almost_equal_vec(result_max.data(), expected_output_max.data(), 0);

        assert_eq!(result_avg.shape(), expected_output_avg.shape());
        assert_almost_equal_vec(result_avg.data(), expected_output_avg.data(), 0);
    }

    #[test]
    fn test_pool2d_fprop_2() {
        // Create input image
        let image_data = get_default_img_data();
        let image = QantTensor4::new(&image_data, [1, 4, 2, 2]).unwrap();

        // Set convolution attributes
        let attributes = PoolingAttributes {
            kernel_height: 2,
            kernel_width: 2,
            padding: 0,
            stride: 1,
            count_include_pad: false,
        };

        // Perform convolution
        let result_max = maxpool2d_fprop(0, &image, attributes).unwrap();
        let result_avg = avgpool2d_fprop(0, &image, attributes).unwrap();

        // Expected output (max value in each pooling kernel)
        // four channels with 2x2 input with 2x2 kernels
        let expected_data_max = to_bf16s(&vec![0.4, 0.8, 1., 0.61]);
        let expected_data_avg = to_bf16s(&vec![0.25, 0.650, 0.555, 0.46]);
        let expected_output_max = QantTensor4::new(expected_data_max, [1, 4, 1, 1]).unwrap();
        let expected_output_avg = QantTensor4::new(expected_data_avg, [1, 4, 1, 1]).unwrap();

        // Assert the results
        assert_eq!(result_max.shape(), expected_output_max.shape());
        assert_almost_equal_vec(result_max.data(), expected_output_max.data(), 0);

        assert_eq!(result_avg.shape(), expected_output_avg.shape());
        assert_almost_equal_vec(result_avg.data(), expected_output_avg.data(), 0);
    }

    // Helper: simple 2×2 image used by the padding tests below.
    fn get_simple_2x2_img_data() -> Vec<Value> {
        to_bf16s(&[1.0, 2.0, 3.0, 4.0])
    }

    #[test]
    fn test_pool2d_fprop_padding_avg_excl() {
        // 2×2 input, 2×2 kernel, padding=1, stride=1 → 3×3 output.
        // Every output position has at least one kernel pixel in the padding region.
        // count_include_pad=false: denominator counts only real pixels.
        let image_data = get_simple_2x2_img_data();
        let image = QantTensor4::new(&image_data, [1, 1, 2, 2]).unwrap();

        let attributes = PoolingAttributes {
            kernel_height: 2,
            kernel_width: 2,
            padding: 1,
            stride: 1,
            count_include_pad: false,
        };

        let result = avgpool2d_fprop(0, &image, attributes).unwrap();

        // (0,0) corner: 3 pad pixels, 1 real (1.0) → 1.0/1
        // (0,1) top edge: 2 pad, 2 real (1,2) → 3.0/2
        // (0,2) corner: 3 pad, 1 real (2.0) → 2.0/1
        // (1,0) left edge: 2 pad, 2 real (1,3) → 4.0/2
        // (1,1) center: 0 pad, 4 real (1,2,3,4) → 10.0/4
        // (1,2) right edge: 2 pad, 2 real (2,4) → 6.0/2
        // (2,0) corner: 3 pad, 1 real (3.0) → 3.0/1
        // (2,1) bottom edge: 2 pad, 2 real (3,4) → 7.0/2
        // (2,2) corner: 3 pad, 1 real (4.0) → 4.0/1
        let expected = to_bf16s(&[1.0, 1.5, 2.0, 2.0, 2.5, 3.0, 3.0, 3.5, 4.0]);
        let expected_out = QantTensor4::new(expected, [1, 1, 3, 3]).unwrap();

        assert_eq!(result.shape(), expected_out.shape());
        assert_almost_equal_vec(result.data(), expected_out.data(), 0.01);
    }

    #[test]
    fn test_pool2d_fprop_padding_avg_incl() {
        // Same geometry as test above; count_include_pad=true: pad pixels count toward divisor
        // (contributing 0 to the sum), so the denominator is always the full kernel area (4).
        let image_data = get_simple_2x2_img_data();
        let image = QantTensor4::new(&image_data, [1, 1, 2, 2]).unwrap();

        let attributes = PoolingAttributes {
            kernel_height: 2,
            kernel_width: 2,
            padding: 1,
            stride: 1,
            count_include_pad: true,
        };

        let result = avgpool2d_fprop(0, &image, attributes).unwrap();

        // Divisor is always 4 (full kernel); pad slots contribute 0 to the sum.
        // (0,0): 1.0/4   (0,1): 3.0/4   (0,2): 2.0/4
        // (1,0): 4.0/4   (1,1): 10.0/4  (1,2): 6.0/4
        // (2,0): 3.0/4   (2,1): 7.0/4   (2,2): 4.0/4
        let expected = to_bf16s(&[0.25, 0.75, 0.5, 1.0, 2.5, 1.5, 0.75, 1.75, 1.0]);
        let expected_out = QantTensor4::new(expected, [1, 1, 3, 3]).unwrap();

        assert_eq!(result.shape(), expected_out.shape());
        assert_almost_equal_vec(result.data(), expected_out.data(), 0.01);
    }

    #[test]
    fn test_pool2d_fprop_padding_max() {
        // Max pooling ignores pad pixels entirely; even a window touching three padding sides
        // returns the max of the single real pixel it covers.
        let image_data = get_simple_2x2_img_data();
        let image = QantTensor4::new(&image_data, [1, 1, 2, 2]).unwrap();

        let attributes = PoolingAttributes {
            kernel_height: 2,
            kernel_width: 2,
            padding: 1,
            stride: 1,
            count_include_pad: false,
        };

        let result = maxpool2d_fprop(0, &image, attributes).unwrap();

        // (0,0): max[1]=1   (0,1): max[1,2]=2   (0,2): max[2]=2
        // (1,0): max[1,3]=3 (1,1): max[1,2,3,4]=4 (1,2): max[2,4]=4
        // (2,0): max[3]=3   (2,1): max[3,4]=4   (2,2): max[4]=4
        let expected = to_bf16s(&[1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 3.0, 4.0, 4.0]);
        let expected_out = QantTensor4::new(expected, [1, 1, 3, 3]).unwrap();

        assert_eq!(result.shape(), expected_out.shape());
        assert_almost_equal_vec(result.data(), expected_out.data(), 0);
    }

    #[test]
    fn test_pool2d_fprop_padding_with_stride() {
        // 4×4 input, 2×2 kernel, padding=1, stride=2 → 3×3 output.
        // With stride=2, corner output cells see only 1 real pixel (3 in padding);
        // interior cells see the full kernel with no padding.  Exercises both excl and incl.
        let image_data = get_default_img_data();
        let image = QantTensor4::new(&image_data, [1, 1, 4, 4]).unwrap();

        let attributes = PoolingAttributes {
            kernel_height: 2,
            kernel_width: 2,
            padding: 1,
            stride: 2,
            count_include_pad: false,
        };

        let result_max = maxpool2d_fprop(0, &image, attributes).unwrap();
        let result_avg_excl = avgpool2d_fprop(0, &image, attributes).unwrap();
        let result_avg_incl = avgpool2d_fprop(
            0,
            &image,
            PoolingAttributes {
                count_include_pad: true,
                ..attributes
            },
        )
        .unwrap();

        let expected_shape = [1, 1, 3, 3];

        // Input layout (row-major):
        //   0.1  0.2  0.3  0.4
        //   0.5  0.6  0.7  0.8
        //   0.9  1.0  0.11 0.21
        //   0.31 0.41 0.51 0.61
        //
        // Output positions (oy,ox) with stride=2, padding=1:
        //   (0,0): 3 pad + real[0,0]=0.1   → max 0.1 | excl 0.1/1  | incl 0.1/4
        //   (0,1): 2 pad + real[0,1]=0.2, [0,2]=0.3 → max 0.3 | excl 0.25 | incl 0.125
        //   (0,2): 3 pad + real[0,3]=0.4   → max 0.4 | excl 0.4/1  | incl 0.4/4
        //   (1,0): 2 pad + real[1,0]=0.5, [2,0]=0.9 → max 0.9 | excl 0.7  | incl 0.35
        //   (1,1): 0 pad + real[1,1..2,1..2]        → max 1.0 | excl=incl 2.41/4
        //   (1,2): 2 pad + real[1,3]=0.8, [2,3]=0.21 → max 0.8 | excl 0.505 | incl 0.2525
        //   (2,0): 3 pad + real[3,0]=0.31  → max 0.31 | excl 0.31  | incl 0.31/4
        //   (2,1): 2 pad + real[3,1]=0.41, [3,2]=0.51 → max 0.51 | excl 0.46 | incl 0.23
        //   (2,2): 3 pad + real[3,3]=0.61  → max 0.61 | excl 0.61  | incl 0.61/4
        let expected_max = QantTensor4::new(
            to_bf16s(&[0.1, 0.3, 0.4, 0.9, 1.0, 0.8, 0.31, 0.51, 0.61]),
            expected_shape,
        )
        .unwrap();
        let expected_avg_excl = QantTensor4::new(
            to_bf16s(&[0.1, 0.25, 0.4, 0.7, 0.6025, 0.505, 0.31, 0.46, 0.61]),
            expected_shape,
        )
        .unwrap();
        let expected_avg_incl = QantTensor4::new(
            to_bf16s(&[
                0.025, 0.125, 0.1, 0.35, 0.6025, 0.2525, 0.0775, 0.23, 0.1525,
            ]),
            expected_shape,
        )
        .unwrap();

        assert_eq!(result_max.shape(), expected_max.shape());
        assert_almost_equal_vec(result_max.data(), expected_max.data(), 0.01);

        assert_eq!(result_avg_excl.shape(), expected_avg_excl.shape());
        assert_almost_equal_vec(result_avg_excl.data(), expected_avg_excl.data(), 0.01);

        assert_eq!(result_avg_incl.shape(), expected_avg_incl.shape());
        assert_almost_equal_vec(result_avg_incl.data(), expected_avg_incl.data(), 0.01);
    }

    #[test]
    fn test_pool2d_fprop_fully_padded_window_is_err() {
        // padding=2 > kernel=2: the top-left output position (0,0) maps to image coords
        // (0..1, 0..1) which are all < padding=2, so count==0.  Both modes must return Err
        // rather than NaN (avg) or a misleading Ok (max).
        let image_data = get_simple_2x2_img_data();
        let image = QantTensor4::new(&image_data, [1, 1, 2, 2]).unwrap();

        let attributes = PoolingAttributes {
            kernel_height: 2,
            kernel_width: 2,
            padding: 2,
            stride: 1,
            count_include_pad: false,
        };

        assert!(maxpool2d_fprop(0, &image, attributes).is_err());
        assert!(avgpool2d_fprop(0, &image, attributes).is_err());
    }

    #[test]
    fn test_pool2d_fprop_3() {
        // Create input image
        let image_data = get_default_img_data();
        let image = QantTensor4::new(&image_data, [1, 4, 2, 2]).unwrap();

        // Set convolution attributes
        let attributes = PoolingAttributes {
            kernel_height: 1,
            kernel_width: 1,
            padding: 0,
            stride: 1,
            count_include_pad: false,
        };

        // Perform convolution
        let result_max = maxpool2d_fprop(0, &image, attributes).unwrap();
        let result_avg = avgpool2d_fprop(0, &image, attributes).unwrap();

        // Expected output (max value in each pooling kernel)
        // four channels with 2x2 input with 1x1 kernels
        let expected_data = to_bf16s(&vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 0.11, 0.21, 0.31, 0.41, 0.51, 0.61,
        ]);
        let expected_output = QantTensor4::new(expected_data, [1, 4, 2, 2]).unwrap();

        // Assert the results
        assert_eq!(result_max.shape(), expected_output.shape());
        assert_almost_equal_vec(result_max.data(), expected_output.data(), 0);

        assert_eq!(result_avg.shape(), expected_output.shape());
        assert_almost_equal_vec(result_avg.data(), expected_output.data(), 0);
    }
}
