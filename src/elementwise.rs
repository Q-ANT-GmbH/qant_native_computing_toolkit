use crate::{
    data_structures::QantTensor1, errors::ToolkitError, qant_driver_import, toolkit_error,
};

/// Multiplies elementwise us and vs, for any shape
pub fn mul<'a, 'b>(
    npu_id: u32,
    us: &QantTensor1<'a>,
    vs: &QantTensor1<'a>,
) -> Result<QantTensor1<'b>, ToolkitError> {
    if us.shape() != vs.shape() {
        return Err(toolkit_error!(
            "us and vs must have the same length, got {0} and {1}",
            us.shape()[0],
            vs.shape()[0]
        ));
    }

    let ws = qant_driver_import::mul(npu_id, us.data(), vs.data())?;

    QantTensor1::new(ws, *us.shape())
}

#[cfg(test)]
mod tests {
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    use crate::utils::test_utils;

    use super::*;

    #[test]
    fn multithreading() {
        let mut handles = vec![];

        for _ in 0..100 {
            let handle = std::thread::spawn(move || {
                // all threads want to work on npu 0
                let n_numbers = 1e4 as usize;
                let mut rng = StdRng::seed_from_u64(12345);
                let us = test_utils::to_bf16s(
                    &(0..n_numbers)
                        .map(|_| rng.random_range(-1.0f32..1.0))
                        .collect::<Vec<_>>(),
                );
                let vs = test_utils::to_bf16s(
                    &(0..n_numbers)
                        .map(|_| rng.random_range(-1.0f32..1.0))
                        .collect::<Vec<_>>(),
                );
                let tol = 0.01;

                let us_tensor = QantTensor1::new(&us, [n_numbers]).unwrap();
                let vs_tensor = QantTensor1::new(&vs, [n_numbers]).unwrap();
                let res = mul(0, &us_tensor, &vs_tensor).unwrap();
                let expected: Vec<_> = us.iter().zip(&vs).map(|(&u, &v)| u * v).collect();
                test_utils::assert_almost_equal_vec(res.data(), &expected, tol);
            });

            handles.push(handle);
        }

        // Wait for all threads to finish
        for handle in handles {
            handle.join().unwrap();
        }
    }
}
