use crate::{
    data_structures::QantTensor2,
    errors::ToolkitError,
    qant_driver_import::{self, QantDataRole, QantDriverHandle},
    toolkit_error,
};

pub struct QantMemoryHandle {
    rows: usize,
    cols: usize,
    data_role: QantDataRole,
    handle: QantDriverHandle,
}

impl QantMemoryHandle {
    pub fn allocate(
        npu_id: u32,
        rows: usize,
        cols: usize,
        data_role: QantDataRole,
    ) -> Result<Self, ToolkitError> {
        let handle = qant_driver_import::allocate(npu_id, rows, cols, data_role)?;
        Ok(Self {
            rows,
            cols,
            data_role,
            handle,
        })
    }

    /// Allocates device memory sized for `matrix` and uploads it.
    pub fn from_matrix(
        npu_id: u32,
        matrix: &QantTensor2,
        data_role: QantDataRole,
    ) -> Result<Self, ToolkitError> {
        let &[rows, cols] = matrix.shape();
        let handle = Self::allocate(npu_id, rows, cols, data_role)?;
        handle.upload(matrix)?;
        Ok(handle)
    }

    pub fn npu_id(&self) -> u32 {
        self.handle.npu_id
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn upload(&self, matrix: &QantTensor2) -> Result<(), ToolkitError> {
        if matrix.shape()[0] != self.rows || matrix.shape()[1] != self.cols {
            return Err(toolkit_error!(
                "Matrix dimensions do not match memory handle dimensions. Got matrix: {}x{}, expected: {}x{}",
                matrix.shape()[0],
                matrix.shape()[1],
                self.rows,
                self.cols
            ));
        }

        qant_driver_import::upload(self.handle, matrix.data())
    }

    pub fn download<'b>(&self) -> Result<QantTensor2<'b>, ToolkitError> {
        if self.data_role != QantDataRole::Features {
            return Err(toolkit_error!("Download of weights is not supported."));
        }

        let output = qant_driver_import::download(self.handle, self.rows * self.cols)?;
        QantTensor2::new(output, [self.rows, self.cols])
    }
}

impl Drop for QantMemoryHandle {
    fn drop(&mut self) {
        if let Err(err) = qant_driver_import::free(self.handle) {
            toolkit_error!(
                "Failed to free device memory handle {:?} on NPU {}: {:?}",
                self.handle,
                self.handle.npu_id,
                err
            );
        }
    }
}

pub fn mul_matrix_vec_batched_mem(
    handle_matrix: &QantMemoryHandle,
    handle_vector: &QantMemoryHandle,
    handle_result: &QantMemoryHandle,
) -> Result<(), ToolkitError> {
    if handle_matrix.data_role != QantDataRole::Weights
        || handle_vector.data_role != QantDataRole::Features
        || handle_result.data_role != QantDataRole::Features
    {
        return Err(toolkit_error!(
            "Invalid data roles for matrix-vector multiplication. Expected (Weights, Features, Features), got ({:?}, {:?}, {:?})",
            handle_matrix.data_role,
            handle_vector.data_role,
            handle_result.data_role
        ));
    }

    if handle_matrix.cols != handle_vector.cols {
        return Err(toolkit_error!(
            "matrix_handle cols ({}) != vector_handle cols ({})",
            handle_matrix.cols,
            handle_vector.cols
        ));
    }
    if handle_result.rows != handle_vector.rows {
        return Err(toolkit_error!(
            "Result memory block rows ({}) != vector_handle rows ({})",
            handle_result.rows,
            handle_vector.rows
        ));
    }
    if handle_result.cols != handle_matrix.rows {
        return Err(toolkit_error!(
            "Result memory block cols ({}) != matrix_handle rows ({})",
            handle_result.cols,
            handle_matrix.rows
        ));
    }

    qant_driver_import::mul_matrix_vec_batched_mem(
        handle_matrix.handle,
        handle_vector.handle,
        handle_result.handle,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_structures::QantTensor2;
    use crate::utils::test_utils;

    #[test]
    fn test_allocate_and_drop_matrix() {
        let npu_id = 0;
        let rows = 2;
        let cols = 3;

        QantMemoryHandle::allocate(npu_id, rows, cols, QantDataRole::Weights).unwrap();
    }

    #[test]
    fn test_upload_and_download_matrix() {
        let npu_id = 0;
        let rows = 2;
        let cols = 3;

        let data = test_utils::to_bf16s(&vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let matrix = QantTensor2::new(data.clone(), [rows, cols]).unwrap();
        let handle =
            QantMemoryHandle::allocate(npu_id, rows, cols, QantDataRole::Features).unwrap();
        handle.upload(&matrix).unwrap();
        let downloaded_matrix = handle.download().unwrap();
        assert_eq!(data, downloaded_matrix.data().to_vec());
    }

    #[test]
    fn test_from_matrix() {
        let data = test_utils::to_bf16s(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let matrix = QantTensor2::new(data.clone(), [2, 3]).unwrap();
        let handle = QantMemoryHandle::from_matrix(0, &matrix, QantDataRole::Features).unwrap();
        assert_eq!((handle.rows(), handle.cols()), (2, 3));
        assert_eq!(data, handle.download().unwrap().data().to_vec());
    }

    #[test]
    fn test_mul_matrix_vec_batched_mem() {
        let npu_id = 0;
        let rows = 4;
        let cols = 3;
        let vector_rows = 3;
        let handle_matrix =
            QantMemoryHandle::allocate(npu_id, rows, cols, QantDataRole::Weights).unwrap();
        let handle_vector =
            QantMemoryHandle::allocate(npu_id, vector_rows, cols, QantDataRole::Features).unwrap();
        let handle_result =
            QantMemoryHandle::allocate(npu_id, vector_rows, rows, QantDataRole::Features).unwrap();

        let matrix = QantTensor2::new(
            test_utils::to_bf16s(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]),
            [rows, cols],
        )
        .unwrap();
        let vectors = QantTensor2::new(
            test_utils::to_bf16s(&[0., 1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5]),
            [vector_rows, cols],
        )
        .unwrap();
        handle_matrix.upload(&matrix).unwrap();
        handle_vector.upload(&vectors).unwrap();

        mul_matrix_vec_batched_mem(&handle_matrix, &handle_vector, &handle_result).unwrap();

        let result = handle_result.download().unwrap();
        let res_shouldbe = test_utils::to_bf16s(&vec![
            0.35, 0.8, 1.25, 1.5, 0.35, 0.8, 1.25, 1.5, 0.3, 0.75, 1.2, 1.5,
        ]);

        test_utils::assert_almost_equal_vec(&res_shouldbe, result.data(), 0.05);
    }

    #[test]
    fn download_rejects_weights_handle() {
        // download only supports features; this is backend-independent
        let npu_id = 0;
        let handle = QantMemoryHandle::allocate(npu_id, 2, 3, QantDataRole::Weights).unwrap();

        assert!(handle.download().is_err());
    }

    fn check_for_error(
        mat_shape: (usize, usize),
        vec_shape: (usize, usize),
        res_shape: (usize, usize),
        layouts: (QantDataRole, QantDataRole, QantDataRole),
    ) {
        let npu_id = 0;
        let matrix =
            QantMemoryHandle::allocate(npu_id, mat_shape.0, mat_shape.1, layouts.0).unwrap();
        let vector =
            QantMemoryHandle::allocate(npu_id, vec_shape.0, vec_shape.1, layouts.1).unwrap();
        let result =
            QantMemoryHandle::allocate(npu_id, res_shape.0, res_shape.1, layouts.2).unwrap();

        assert!(mul_matrix_vec_batched_mem(&matrix, &vector, &result).is_err());
    }

    #[test]
    fn mul_matrix_vec_batched_mem_rejects_wrong_matrix_role() {
        check_for_error(
            (4, 3),
            (2, 3),
            (2, 4),
            (
                QantDataRole::Features,
                QantDataRole::Features,
                QantDataRole::Features,
            ),
        );
    }

    #[test]
    fn mul_matrix_vec_batched_mem_rejects_wrong_vector_role() {
        check_for_error(
            (4, 3),
            (2, 3),
            (2, 4),
            (
                QantDataRole::Weights,
                QantDataRole::Weights,
                QantDataRole::Features,
            ),
        );
    }

    #[test]
    fn mul_matrix_vec_batched_mem_rejects_wrong_result_role() {
        check_for_error(
            (4, 3),
            (2, 3),
            (2, 4),
            (
                QantDataRole::Weights,
                QantDataRole::Features,
                QantDataRole::Weights,
            ),
        );
    }

    #[test]
    fn mul_matrix_vec_batched_mem_rejects_vector_length_mismatch() {
        check_for_error(
            (4, 3),
            (2, 5),
            (2, 4),
            (
                QantDataRole::Weights,
                QantDataRole::Features,
                QantDataRole::Features,
            ),
        );
    }

    #[test]
    fn mul_matrix_vec_batched_mem_rejects_result_rows_mismatch() {
        check_for_error(
            (4, 3),
            (2, 3),
            (9, 4),
            (
                QantDataRole::Weights,
                QantDataRole::Features,
                QantDataRole::Features,
            ),
        );
    }

    #[test]
    fn mul_matrix_vec_batched_mem_rejects_result_columns_mismatch() {
        check_for_error(
            (4, 3),
            (2, 3),
            (2, 9),
            (
                QantDataRole::Weights,
                QantDataRole::Features,
                QantDataRole::Features,
            ),
        );
    }

    #[test]
    fn double_free_returns_error() {
        let handle = qant_driver_import::allocate(0, 2, 3, QantDataRole::Features).unwrap();
        qant_driver_import::free(handle).unwrap();

        // freeing an already-freed handle must return a clean error, not panic
        // or silently affect a handle that has since reused the same slot.
        assert!(qant_driver_import::free(handle).is_err());
    }

    #[test]
    fn use_after_free_returns_error() {
        let handle = qant_driver_import::allocate(0, 2, 3, QantDataRole::Features).unwrap();
        qant_driver_import::free(handle).unwrap();

        let data = test_utils::to_bf16s(&[0.0; 6]);

        // every operation on a freed handle must return an error, not panic.
        assert!(qant_driver_import::upload(handle, &data).is_err());
        assert!(qant_driver_import::download(handle, 6).is_err());

        let other_matrix = qant_driver_import::allocate(0, 1, 3, QantDataRole::Weights).unwrap();
        let other_result = qant_driver_import::allocate(0, 2, 1, QantDataRole::Features).unwrap();
        assert!(
            qant_driver_import::mul_matrix_vec_batched_mem(other_matrix, handle, other_result)
                .is_err()
        );

        qant_driver_import::free(other_matrix).unwrap();
        qant_driver_import::free(other_result).unwrap();
    }
}
