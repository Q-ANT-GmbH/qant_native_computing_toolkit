use crate::{
    data_structures::QantTensor2, errors::ToolkitError, global_settings::Value,
    qant_driver_import::driver_matrix_vector_mul_batched, toolkit_error,
};

pub fn linear_fprop<'a, 'b>(
    npu_id: u32,
    flatten_features: &QantTensor2<'a>,
    filter: &QantTensor2<'a>,
) -> Result<QantTensor2<'b>, ToolkitError> {
    if flatten_features.shape()[1] != filter.shape()[1] {
        return Err(toolkit_error!(
            "feature channels does not match filter in channels, got {0} and {1}",
            flatten_features.shape()[1],
            filter.shape()[1]
        ));
    };

    let result = mul_mat_vec_batched(
        npu_id,
        filter.data(),
        filter.shape()[0],
        filter.shape()[1],
        flatten_features.data(),
    )?;

    QantTensor2::new(result, [flatten_features.shape()[0], filter.shape()[0]])
}

/// Multiplies a matrix m[row, column] with a vector v with batch support
pub fn mul_mat_vec_batched(
    npu_id: u32,
    matrix: &[Value],
    n_rows: usize,
    n_columns: usize,
    vectors: &[Value],
) -> Result<Vec<Value>, ToolkitError> {
    if matrix.len() != n_rows * n_columns {
        return Err(toolkit_error!(
            "matrix data len does not match specified row and column number, got {0} and {1}",
            n_rows * n_columns,
            matrix.len(),
        ));
    }
    if !vectors.len().is_multiple_of(n_columns) {
        return Err(toolkit_error!(
            "there is no batch count which matches the vector length and the matrix columns, got {0} and {1}",
            vectors.len(),
            n_columns,
        ));
    };

    let res_batched = driver_matrix_vector_mul_batched(npu_id, matrix, vectors, n_columns)?;

    Ok(res_batched)
}

#[cfg(test)]
mod tests {
    use crate::{
        data_format_conversion::to_bf16s, data_structures::QantTensor2,
        utils::test_utils::assert_almost_equal_vec,
    };

    use super::*;

    #[test]
    fn mat_vec_test() {
        // i16 scheme contains 3 decimal positions
        // 0.1, 0.2, 0.3, ...
        let matrix = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1., 1., 1.],
        ]
        .as_flattened();
        let matrix_columns = 3;
        let matrix_rows = 4;
        let vector = [0., 1., 0.5];
        let res = mul_mat_vec_batched(
            0,
            &to_bf16s(matrix),
            matrix_rows,
            matrix_columns,
            &to_bf16s(&vector),
        )
        .unwrap();
        println!("{res:?}");
        let res_shouldbe = to_bf16s(&vec![0.35, 0.8, 1.25, 1.5]);

        assert_almost_equal_vec(&res, &res_shouldbe, 0.05);
    }

    #[test]
    fn mat_vec_batched_test() {
        // i16 scheme contains 3 decimal positions
        // 0.1, 0.2, 0.3, ...
        let matrix = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1., 1., 1.],
        ]
        .as_flattened();
        let matrix_columns = 3;
        let matrix_rows = 4;
        let vectors = [0., 1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5];
        let res = mul_mat_vec_batched(
            0,
            &to_bf16s(&matrix),
            matrix_rows,
            matrix_columns,
            &to_bf16s(&vectors),
        )
        .unwrap();
        println!("{res:?}");
        let res_shouldbe = to_bf16s(&vec![
            0.35, 0.8, 1.25, 1.5, 0.35, 0.8, 1.25, 1.5, 0.3, 0.75, 1.2, 1.5,
        ]);

        assert_almost_equal_vec(&res, &res_shouldbe, 0.05);
    }

    #[test]
    fn linear_fprop_test() {
        let feature = QantTensor2::new(
            to_bf16s(&vec![
                0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, //
                0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
            ]),
            [2, 10],
        )
        .unwrap();

        let filter = QantTensor2::new(
            to_bf16s(&vec![
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                0.7, 0.8, 0.9, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.,
            ]),
            [5, 10],
        )
        .unwrap();

        let expected = to_bf16s(&vec![4.5, 0., 2.25, 2.85, 1.2, 4.5, 0., 2.25, 2.85, 1.2]);

        let res = linear_fprop(0, &feature, &filter).unwrap();

        assert_eq!(res.shape()[1], filter.shape()[0]);
        assert_almost_equal_vec(res.data(), &expected, 0.05);
    }
}
