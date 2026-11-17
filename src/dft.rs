use crate::{data_structures::QantTensor3, errors::ToolkitError, qant_driver_import};

/// Performs a Discrete Fourier Transform (DFT) on the batched input complex tensor.
/// The input tensor must have a shape of [batch, dft_size, 2], where the last dimension represents the real and imaginary parts of the complex numbers.
///
/// # Arguments
/// * `npu_id` - The identifier of the NPU on which to perform the operation. `-1`
///   (`AUTO_NPU_ID`) selects the first NPU that supports DFT. Errors if `npu_id < -1`.
/// * `input` - A reference to a `QantTensor3` containing the complex input data for the batched DFT.
///
/// # Returns
/// - `Ok(QantTensor3)`: The resulting feature data after applying dft.
/// - `Err(ToolkitError)`: An error if the operation fails, such as due to invalid dimensions.
pub fn dft<'a, 'b>(npu_id: i32, input: &QantTensor3<'a>) -> Result<QantTensor3<'b>, ToolkitError> {
    if input.shape()[2] != 2 {
        return Err(ToolkitError::from_str(
            "input has to be a complex tensor with a shape [*, 2].",
        ));
    }

    let dft_size = input.shape()[1];
    let data = qant_driver_import::dft_batched(npu_id, input.data(), dft_size)?;

    QantTensor3::new(data, *input.shape())
}

#[cfg(test)]
mod tests {
    use crate::utils::test_utils;
    use half::bf16;
    use rustfft::num_complex::Complex;

    use super::*;

    /// Returns the lowest id among the available NPUs that supports DFT.
    fn dft_npu_id() -> i32 {
        qant_driver_import::get_available_npus(&[qant_driver_import::QantDeviceCapability::Dft])
            .unwrap()
            .first()
            .expect("no available NPU supports DFT")
            .0 as i32
    }

    const RTOL: f64 = 2e-2;

    /// Asserts that the relative L2 error of each DFT in the batch is below `rtol`.
    #[track_caller]
    fn assert_relative_l2_error(left: &QantTensor3, right: &QantTensor3, rtol: f64) {
        assert_eq!(left.shape(), right.shape());
        let dft_len = 2 * left.shape()[1];
        for (l, r) in std::iter::zip(left.data().chunks(dft_len), right.data().chunks(dft_len)) {
            let l2_error = std::iter::zip(l, r)
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                .sum::<f64>()
                .sqrt();
            let l2_norm = r.iter().map(|&b| f64::from(b).powi(2)).sum::<f64>().sqrt();
            let rel_error = l2_error / l2_norm;
            if rel_error >= rtol {
                panic!("relative L2 error {rel_error} >= {rtol}")
            }
        }
    }

    fn to_complex(data: &QantTensor3) -> Result<Vec<Complex<bf16>>, ToolkitError> {
        if data.shape()[2] != 2 {
            return Err(ToolkitError::from_str(
                "input has to be a complex tensor with a shape [*, 2].",
            ));
        }

        Ok(data
            .data()
            .chunks(2)
            .map(|co| Complex {
                re: co[0],
                im: co[1],
            })
            .collect())
    }

    fn to_tensor<'a>(
        data: Vec<Complex<bf16>>,
        n_dfts: usize,
    ) -> Result<QantTensor3<'a>, ToolkitError> {
        let shape = [n_dfts, data.len() / n_dfts, 2];

        let mut tensor_data = Vec::with_capacity(data.len() * 2);
        for c in data {
            tensor_data.push(c.re);
            tensor_data.push(c.im);
        }
        QantTensor3::new(tensor_data, shape)
    }

    #[test]
    fn test_dft() {
        let dft_size = 4;
        let input_data = test_utils::to_bf16s(&vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        let input = QantTensor3::new(input_data, [1, dft_size, 2]).unwrap();

        let result = dft(dft_npu_id(), &input).unwrap();

        let fft = rustfft::FftPlanner::new().plan_fft_forward(dft_size);
        let mut buffer = to_complex(&input).unwrap();
        fft.process(&mut buffer);
        let expected = to_tensor(buffer, 1).unwrap();

        assert_relative_l2_error(&result, &expected, RTOL);
    }

    #[test]
    fn test_dft_batched() {
        let input_data = test_utils::to_bf16s(&vec![
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        ]);
        let n_dfts = 3;
        let dft_size = 2;
        let input = QantTensor3::new(input_data, [n_dfts, dft_size, 2]).unwrap();

        let result = dft(dft_npu_id(), &input);

        let fft = rustfft::FftPlanner::new().plan_fft_forward(dft_size);
        let mut buffer = to_complex(&input).unwrap();
        for i in 0..n_dfts {
            let start = i * dft_size;
            let end = start + dft_size;
            fft.process(&mut buffer[start..end]);
        }
        let expected = to_tensor(buffer, input.shape()[0]).unwrap();

        assert!(result.is_ok());
        assert_relative_l2_error(&result.unwrap(), &expected, RTOL)
    }

    #[test]
    fn test_dft_error() {
        let input_data = test_utils::to_bf16s(&vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        let input = QantTensor3::new(input_data, [1, 8, 1]).unwrap();

        let result = dft(0, &input);

        assert!(result.is_err());
    }

    #[test]
    fn test_complex_conversion() {
        let input_data = test_utils::to_bf16s(&vec![1.0, 0.1, 1.0, 0.2, 1.0, 0.3, 1.0, 0.3]);
        let input = QantTensor3::new(input_data, [1, 4, 2]).unwrap();

        let result = to_tensor(to_complex(&input).unwrap(), 1).unwrap();

        assert_eq!(result.data(), input.data());
        assert_eq!(result.shape(), input.shape());
    }
}
