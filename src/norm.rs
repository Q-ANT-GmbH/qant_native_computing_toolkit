use half::bf16;
use num_traits::Float;

use crate::{
    data_structures::{QantTensor1, QantTensor4},
    errors::ToolkitError,
    global_settings::Value,
    qant_driver_import::driver_mul,
    toolkit_error,
};

///
/// # Arguments
///
/// * `features` - A reference to the input feature data.
/// * `means` - A slice containing the mean values for each feature.
/// * `variances` - A slice containing the variance values for each feature.
/// * `weights` - A slice containing the learned scale parameters.
/// * `bias` - A slice containing the learned bias parameters.
/// * `eps` - A small value added to the variance for numerical stability.
///
/// # Returns
///
/// Returns the normalized feature data.
pub fn batchnorm2d_fprop<'b>(
    npu_id: u32,
    features: &QantTensor4,
    means: &QantTensor1,
    variances: &QantTensor1,
    weights: &QantTensor1,
    bias: &QantTensor1,
    eps: Value,
) -> Result<QantTensor4<'b>, ToolkitError> {
    if features.shape()[0] > 1 {
        return Err(ToolkitError::from_str("does not support batched input."));
    }

    if features.shape()[1] != means.len() {
        return Err(toolkit_error!(
            "means does not match data channels, got {0} and {1}",
            features.shape()[1],
            means.len()
        ));
    }
    if features.shape()[1] != variances.len() {
        return Err(toolkit_error!(
            "variances does not match data channels, got {0} and {1}",
            features.shape()[1],
            variances.len()
        ));
    }
    if features.shape()[1] != weights.len() {
        return Err(toolkit_error!(
            "weights does not match data channels, got {0} and {1}",
            features.shape()[1],
            weights.len()
        ));
    }
    if features.shape()[1] != bias.len() {
        return Err(toolkit_error!(
            "bias does not match data channels, got {0} and {1}",
            features.shape()[1],
            bias.len()
        ));
    }

    let channel_size = features.shape()[2] * features.shape()[3];
    let dividers: Vec<Value> = variances.data().iter().map(|v| (*v + eps).sqrt()).collect();

    // we split the operation into three parts to perform the multiplication on the NPU
    // 1. normalize input
    let mut normalized = vec![bf16::from_f32(0.); features.len()];
    for (i, feature) in features.data().iter().enumerate() {
        let channel = i / channel_size;

        normalized[i] = (feature - means.data()[channel]) / dividers[channel];
    }

    // 2. weigh the normalized input (on NPU)
    let weights_flat: Vec<Value> = weights
        .data()
        .iter()
        .flat_map(|&w| std::iter::repeat_n(w, channel_size))
        .collect();
    let mut result = driver_mul(npu_id, &normalized, &weights_flat)?;

    // 3. add bias
    for (i, res) in result.iter_mut().enumerate() {
        let channel = i / channel_size;
        *res += bias.data()[channel];
    }

    QantTensor4::new(result, *features.shape())
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use crate::{
        data_format_conversion::to_bf16s,
        data_structures::{QantTensor1, QantTensor4},
        norm::batchnorm2d_fprop,
        utils::test_utils::assert_almost_equal_vec,
    };

    #[test]
    fn batchnorm2d_test() {
        let feature = QantTensor4::new(to_bf16s(&vec![0.5; 10]), [1, 10, 1, 1]).unwrap();

        let means = QantTensor1::new(to_bf16s(&vec![0.5; 10]), [10]).unwrap();
        let variances = QantTensor1::new(to_bf16s(&vec![1.; 10]), [10]).unwrap();
        let weights = QantTensor1::new(to_bf16s(&vec![1.; 10]), [10]).unwrap();
        let bias = QantTensor1::new(to_bf16s(&vec![0.; 10]), [10]).unwrap();
        let eps = bf16::from_f32(0.);

        let expected = to_bf16s(&vec![0.]);

        let res = batchnorm2d_fprop(0, &feature, &means, &variances, &weights, &bias, eps).unwrap();

        assert_eq!(res.shape(), feature.shape());
        assert_almost_equal_vec(res.data(), &expected, 0.05);

        let feature = QantTensor4::new(
            to_bf16s(&vec![0.5, 0., 1., 0.25, 2., 0.1, 0.3, 0.4, 0.9, 0.333]),
            [1, 10, 1, 1],
        )
        .unwrap();

        let expected = to_bf16s(&vec![
            0., -0.5, 0.5, -0.25, 1.5, -0.4, -0.2, -0.1, 0.4, -0.167,
        ]);

        let res = batchnorm2d_fprop(0, &feature, &means, &variances, &weights, &bias, eps).unwrap();

        assert_eq!(res.shape(), feature.shape());
        assert_almost_equal_vec(res.data(), &expected, 0.05);
    }
}
