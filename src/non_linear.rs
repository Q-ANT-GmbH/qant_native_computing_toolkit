use std::iter::repeat_n;

use crate::data_structures::{QantTensor1, QantTensor2, QantTensor3};
use crate::errors::ToolkitError;
use crate::global_settings::Value;
use crate::qant_driver_import::{driver_mul, driver_scaled_periodic_nl};
use crate::toolkit_error;
use half::bf16;
use num_traits::Float;

#[inline]
fn relu(v: Value) -> Value {
    v.max(Value::from_f32(0.0))
}

/// Applies a ReLu function to all values
/// ReLu is defined as: max(x, 0)
pub fn relu_fprop<'b>(_npu_id: u32, features: &QantTensor1<'_>) -> QantTensor1<'b> {
    // TODO: implement multi-threading
    let data: Vec<Value> = features.data().iter().map(|e| relu(*e)).collect();

    // the shape check here can't fail
    #[allow(clippy::unwrap_used)]
    QantTensor1::new(data, *features.shape()).unwrap()
}

/// Applies a ReLu function to all values inplace
/// ReLu is defined as: max(x, 0)
pub fn relu_fprop_inplace(features: &mut QantTensor1) {
    features
        .data_mut()
        .to_mut()
        .iter_mut()
        .for_each(|e| *e = relu(*e));
}

#[inline]
fn sigmoid(v: Value) -> Value {
    bf16::ONE / (bf16::ONE + (-v).exp())
}

/// Applies a Sigmoid function to all values
/// Sigmoid is defined as: 1 / (1 + exp(-x))
pub fn sigmoid_fprop<'b>(_npu_id: u32, features: &QantTensor1<'_>) -> QantTensor1<'b> {
    // TODO: implement multi-threading
    let data: Vec<Value> = features.data().iter().map(|e| sigmoid(*e)).collect();

    // the shape check here can't fail
    #[allow(clippy::unwrap_used)]
    QantTensor1::new(data, *features.shape()).unwrap()
}

/// Applies a softmax function to the feature_data
/// Softmax is defined as: exp(x_i) / sum_over_i(exp(x_i))
pub fn softmax_fprop<'b>(
    _npu_id: u32,
    features: &QantTensor2<'_>,
) -> Result<QantTensor2<'b>, ToolkitError> {
    if features.shape()[0] > 1 {
        return Err(ToolkitError::from_str("does not support batched input."));
    }

    let sum: Value = features.data().iter().map(|&e| e.exp()).sum();
    let data: Vec<Value> = features.data().iter().map(|&e| e.exp() / sum).collect();

    QantTensor2::new(data, *features.shape())
}

/// Calculates a scaled periodic nonlinear function for all pairs of values in feature_data and weights_data
pub fn calc_scaled_periodic_nl<'a, 'b>(
    npu_id: u32,
    feature_data: &QantTensor1<'a>,
    weights_data: &QantTensor1<'a>,
) -> Result<QantTensor1<'b>, ToolkitError> {
    if feature_data.shape()[0] != weights_data.shape()[0] {
        return Err(toolkit_error!(
            "feature_data and weights_data must have same length, got {0} and {1}",
            feature_data.shape()[0],
            weights_data.shape()[0]
        ));
    }
    let result_data = crate::qant_driver_import::driver_scaled_periodic_nl(
        npu_id,
        feature_data.data(),
        weights_data.data(),
    )?;
    QantTensor1::new(result_data, *feature_data.shape())
}

// take a flattened matrix of shape (rows_before, cols_before) and transpose it
// to a new shape (rows_after=cols_before, cols_after=rows_before)
fn transpose_2d_flat<T: Clone>(
    data: &[T],
    rows_before: usize,
    cols_before: usize,
) -> Result<Vec<T>, ToolkitError> {
    if data.len() != rows_before * cols_before {
        return Err(toolkit_error!(
            "transpose_2d_flat: invalid matrix description, got {0:?} and {1}",
            data.len(),
            rows_before * cols_before
        ));
    };

    let mut transposed = Vec::with_capacity(data.len());

    for col in 0..cols_before {
        for row in 0..rows_before {
            transposed.push(data[row * cols_before + col].clone());
        }
    }
    Ok(transposed)
}

/// Calculates a Q.ANT version of a KAN layer (https://arxiv.org/abs/2404.19756) based on scaled_periodic_nl.
/// The mathematical function is y_bj = sum_{i, k} tcos(ks_k * x_bi + phis_{jik}) * ampls_{jik},
/// where x_bi is a batch of input vectors, ks the frequency components,
/// phis the phase offsets (tensor), ampls the amplitude (tensor), and y_bj the batch of outputs.
/// tcos() denotes the cosine-related shape of the periodic optical nonlinearity.
pub fn calc_kan_layer<'a, 'b>(
    npu_id: u32,
    xs: &QantTensor2<'a>,    // shape (n_batches, n_channels_in)
    phis: &QantTensor3<'a>,  // shape (n_channels_out, n_channels_in, n_ks)
    ampls: &QantTensor3<'a>, // shape (n_channels_out, n_channels_in, n_ks)
    ks: &[Value],            // shape (n_ks)
) -> Result<QantTensor2<'b>, ToolkitError> {
    let n_batch = xs.shape()[0];
    let n_in = xs.shape()[1];
    let n_out = phis.shape()[0];
    let n_ks = ks.len();

    if n_in != phis.shape()[1] {
        return Err(toolkit_error!(
            "phis length along n_channels_in dimension does not match input data length, got {n_in} and {0}",
            phis.shape()[1]
        ));
    }
    if n_ks != phis.shape()[2] {
        return Err(toolkit_error!(
            "phis length along k dimension does not match length of ks, got {n_ks} and {0}",
            phis.shape()[2]
        ));
    }
    if phis.shape() != ampls.shape() {
        return Err(toolkit_error!(
            "amps and phis must have the same shape, got {0:?} and {1:?}",
            phis.shape(),
            ampls.shape()
        ));
    }

    // On NPU, nonlinearity is an elementwise operation.
    // The order of elements in the list we send to the NPU must be according to
    // out_idx, in_idx, k_idx, batch_idx

    // nomenclature: X_jikb is X with out-index j, in-index i, k-index k and batch-index b,
    // element-wise operations can only be performed on tensors with the same indices.
    // indices (=dimensions) are added/inserted by repeating elements of all dimension to the right
    // of the insertion position (row major ordering)

    // data comes in as:
    // xs.data = x_bi
    // ampls.data/phis.data = ampls_jik/phis_jik
    // ks.data = ks_k

    // goal: send arrays in shape X_jikb to the NPU, then sum over i and k, and return Y_bj

    // xs: swap dimensions to move batch to the end
    let xs_ib = transpose_2d_flat(xs.data(), n_batch, n_in)?;
    // xs: add k dimension in the middle by repeating the last dimension
    let xs_ikb: Vec<_> = xs_ib
        .chunks(n_batch)
        .flat_map(|chunk| repeat_n(chunk, n_ks).flatten())
        .cloned()
        .collect();

    // ks: add batch in the end
    let ks_kb: Vec<_> = ks.iter().flat_map(|&k| repeat_n(k, n_batch)).collect();
    // ks: add in dimension in front
    let ks_ikb: Vec<_> = ks_kb.repeat(n_in);

    // x * k product on NPU
    let xtimesk_ikb = driver_mul(npu_id, &xs_ikb, &ks_ikb)?;

    // add out dimension in front
    let xtimesk_jikb = xtimesk_ikb.repeat(n_out);
    // phis and ampls: insert batch dimension in the end
    let phis_jikb = phis.data().iter().flat_map(|&elem| repeat_n(elem, n_batch));
    let amps_jikb: Vec<_> = ampls
        .data()
        .iter()
        .flat_map(|&elem| repeat_n(elem, n_batch))
        .collect();

    // xtimesk + phi on cpu
    let cos_arg_jikb: Vec<Value> = xtimesk_jikb
        .iter()
        .zip(phis_jikb)
        .map(|(&kx, phi)| kx + phi)
        .collect();

    // nonlinearity on NPU
    let res_jikb = driver_scaled_periodic_nl(npu_id, &cos_arg_jikb, &amps_jikb)?;

    // sum over i and k
    let res_jb: Vec<_> = res_jikb
        // create chunk for last 3 indices
        .chunks(n_in * n_ks * n_batch)
        .flat_map(|chunk_ikb| {
            (0..n_batch).map(|batch_idx| {
                // sum over the first two indices of the chunk, for each batch idx
                let mut sum = Value::from_f32(0.);
                for in_idx in 0..n_in {
                    for k_idx in 0..n_ks {
                        let flat_idx = ((in_idx * n_ks + k_idx) * n_batch) + batch_idx;
                        sum += chunk_ikb[flat_idx];
                    }
                }
                sum
            })
        })
        .collect();
    let res_bj = transpose_2d_flat(&res_jb, n_out, n_batch)?;
    QantTensor2::new(res_bj, [n_batch, n_out])
}

#[cfg(test)]
mod tests {
    use crate::{data_format_conversion::to_bf16s, utils::test_utils::assert_almost_equal_vec};

    use super::*;

    #[test]
    fn relu_test() {
        let v = to_bf16s(&vec![-1, 0, 10, 5]);
        let expected = to_bf16s(&vec![0, 0, 10, 5]);

        let mut features = QantTensor1::new(&v, [4]).unwrap();
        let res_out_place = relu_fprop(0, &features);
        assert_eq!(res_out_place.data(), expected);

        relu_fprop_inplace(&mut features);
        assert_eq!(features.data(), expected);
    }

    #[test]
    fn sigmoid_test() {
        let v = to_bf16s(&vec![-0.001, 0., 0.01, 0.5, 1., 10., -1.]);
        let expected = to_bf16s(&vec![0.5, 0.5, 0.503, 0.621, 0.730, 1.0, 0.269]);

        let features = QantTensor1::new(&v, [7]).unwrap();
        let res_out_place = sigmoid_fprop(0, &features);
        assert_eq!(res_out_place.data(), expected);
    }

    #[test]
    fn softmax_test() {
        let v = to_bf16s(&vec![-5., 3.1, 1.3, 0.4, 2.1, 4.11, 5.4, 6.22]);
        let features = QantTensor2::new(v, [1, 8]).unwrap();
        let res = softmax_fprop(0, &features).unwrap();

        let expected = to_bf16s(&vec![
            8.165836e-6,
            0.026733398,
            0.004425049,
            0.0018081665,
            0.009887695,
            0.07470703,
            0.27148438,
            0.609375,
        ]);

        assert_eq!(res.data(), expected);
    }

    #[test]
    fn nonlinearity() {
        let vs = to_bf16s(&vec![-1., -0.5, 0., 0.5, 1.]);
        let us = to_bf16s(&vec![-20., 1., 0., 0.5, -1.]);
        let ws_npu = driver_scaled_periodic_nl(0, &us, &vs).unwrap();
        let ws_cpu: Vec<_> = us
            .iter()
            .zip(vs.iter())
            .map(|(&u, &v)| u.cos() * v)
            .collect();
        println!("{ws_npu:?} {ws_cpu:?}");
        assert_almost_equal_vec(&ws_npu, &ws_cpu, 0.1);
    }

    #[test]
    fn kan_shape() {
        let n_in = 5;
        let n_out = 10;
        let n_ks = 3;
        let n_batches = 4;

        let xs_in =
            QantTensor2::new(to_bf16s(&vec![0.8; n_batches * n_in]), [n_batches, n_in]).unwrap();
        let ampls = QantTensor3::new(
            to_bf16s(&vec![0.7; n_out * n_in * n_ks]),
            [n_out, n_in, n_ks],
        )
        .unwrap();
        let phis = QantTensor3::new(
            to_bf16s(&vec![0.6; n_out * n_in * n_ks]),
            [n_out, n_in, n_ks],
        )
        .unwrap();
        let ks = to_bf16s(&vec![0.5; n_ks]);

        let res = calc_kan_layer(0, &xs_in, &phis, &ampls, &ks).unwrap();

        assert_eq!(res.shape().clone(), [n_batches, n_out]);
    }
}
