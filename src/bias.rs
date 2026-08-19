use crate::{
    data_structures::{QantTensor1, QantTensor2, QantTensor4},
    errors::ToolkitError,
    global_settings::Value,
    toolkit_error,
};

/// Add a bias to the feature_data
pub fn add_bias_fprop<'a, 'b>(
    _npu_id: u32,
    feature_data: &QantTensor2<'a>, // shape (batches, flattened_features)
    bias_data: &QantTensor1<'a>,    // shape (flattened_features)
) -> Result<QantTensor2<'b>, ToolkitError> {
    if feature_data.shape()[1] != bias_data.len() {
        return Err(toolkit_error!(
            "feature_data and bias_data must have compatible shapes, got {0:?} and {1:?}",
            feature_data.shape(),
            bias_data.shape()
        ));
    }

    let result_data: Vec<Value> = feature_data
        .data()
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let j = i % bias_data.len();
            f + bias_data.data()[j]
        })
        .collect();
    QantTensor2::new(result_data, *feature_data.shape())
}

/// Add a bias to the image
pub fn add_bias_for_conv2d_fprop<'a, 'b>(
    _npu_id: u32,
    image: &QantTensor4<'a>,
    bias_data: &QantTensor1<'a>,
) -> Result<QantTensor4<'b>, ToolkitError> {
    let &[_, image_channels, image_height, image_width] = image.shape();

    if image_channels != bias_data.len() {
        return Err(toolkit_error!(
            "image and bias_data must have compatible shapes, got {0:?} and {1:?}",
            image.shape(),
            bias_data.shape()
        ));
    }

    let image_size = image_height * image_width;
    let n_values_per_batch = image_channels * image_size;
    let result_data: Vec<Value> = image
        .data()
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let j = (i % n_values_per_batch) / image_size;
            f + bias_data.data()[j]
        })
        .collect();
    QantTensor4::new(result_data, *image.shape())
}

/// Add a bias to the feature_data
/// inplace: overwrite feature_data
pub fn add_bias_fprop_inplace(
    feature_data: &mut QantTensor2,
    bias_data: &QantTensor1,
) -> Result<(), ToolkitError> {
    if feature_data.shape()[1] != bias_data.len() {
        return Err(toolkit_error!(
            "feature_data and bias_data must have compatible shapes, got {0:?} and {1:?}",
            feature_data.shape(),
            bias_data.shape()
        ));
    }

    // Perform element-wise addition of bias to feature data
    feature_data
        .data_mut()
        .to_mut()
        .iter_mut()
        .zip(bias_data.data().iter())
        .for_each(|(feature, &bias)| {
            *feature += bias;
        });
    Ok(())
}

/// Add a bias to the image
pub fn add_bias_for_conv2d_fprop_inplace(
    _npu_id: u32,
    image: &mut QantTensor4,
    bias_data: &QantTensor1,
) -> Result<(), ToolkitError> {
    let &[_, image_channels, image_height, image_width] = image.shape();

    if image_channels != bias_data.len() {
        return Err(toolkit_error!(
            "image and bias_data must have compatible shapes, got {0:?} and {1:?}",
            image.shape(),
            bias_data.shape()
        ));
    }

    let image_size = image_height * image_width;
    let n_values_per_batch = image_channels * image_size;
    image
        .data_mut()
        .to_mut()
        .iter_mut()
        .enumerate()
        .for_each(|(i, f)| {
            let j = (i % n_values_per_batch) / image_size;
            *f += bias_data.data()[j]
        });
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::data_format_conversion::to_bf16s;

    use super::*;

    #[test]
    fn add_bias_test() {
        let feature_data =
            QantTensor2::new(to_bf16s(&[1, 2, 3, 4, 5, 6, 7, 8, 9]), [1, 9]).unwrap();
        let bias_data = QantTensor1::new(to_bf16s(&[9, 8, 7, 6, 5, 4, 3, 2, 1]), [9]).unwrap();

        let result = add_bias_fprop(0, &feature_data, &bias_data).unwrap();
        let expected = to_bf16s(&vec![10; 9]);

        assert_eq!(result.shape(), feature_data.shape());
        assert_eq!(result.data().to_vec(), expected);
    }

    #[test]
    fn add_bias_for_conv2d_test() {
        let features = to_bf16s(&vec![1; 24]);
        let bias = to_bf16s(&vec![1, 2, 3]);
        let feature_data = QantTensor4::new(features, [2, 3, 2, 2]).unwrap();
        let bias_data = QantTensor1::new(bias, [3]).unwrap();

        let result = add_bias_for_conv2d_fprop(0, &feature_data, &bias_data).unwrap();
        let expected = to_bf16s(&vec![
            2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
        ]);

        assert_eq!(result.shape(), feature_data.shape());
        assert_eq!(result.data().to_vec(), expected);
    }

    #[test]
    fn add_bias_test_inplace() {
        let mut feature_data =
            QantTensor2::new(to_bf16s(&vec![1, 2, 3, 4, 5, 6, 7, 8, 9]), [1, 9]).unwrap();
        let bias_data = QantTensor1::new(to_bf16s(&vec![9, 8, 7, 6, 5, 4, 3, 2, 1]), [9]).unwrap();

        add_bias_fprop_inplace(&mut feature_data, &bias_data).unwrap();
        let expected = to_bf16s(&vec![10; 9]);

        assert_eq!(feature_data.shape().to_vec(), [1, 9]);
        assert_eq!(feature_data.data().to_vec(), expected);
    }

    #[test]
    fn add_bias_for_conv2d_test_inplace() {
        let features = vec![1; 24];
        let bias = vec![1, 2, 3];
        let mut feature_data = QantTensor4::new(to_bf16s(&features), [2, 3, 2, 2]).unwrap();
        let bias_data = QantTensor1::new(to_bf16s(&bias), [3]).unwrap();

        add_bias_for_conv2d_fprop_inplace(0, &mut feature_data, &bias_data).unwrap();
        let expected = to_bf16s(&vec![
            2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
        ]);

        assert_eq!(feature_data.shape().to_vec(), [2, 3, 2, 2]);
        assert_eq!(feature_data.data().to_vec(), expected);
    }
}
