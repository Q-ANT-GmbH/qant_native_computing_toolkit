use crate::errors::ToolkitError;
use num_traits::{FromPrimitive, ToPrimitive};
use tracing::Level;

/// Returns the index of the maximum element in the slice.
pub fn argmax<T: Ord>(v: &[T]) -> Result<usize, ToolkitError> {
    match v
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.cmp(b.1))
        .map(|(index, _)| index)
    {
        None => Err(ToolkitError::from_str("container has no max value")),
        Some(idx) => Ok(idx),
    }
}

/// Returns the maximum element in the slice.
pub fn max_with_errhand<T: Ord + Copy>(v: &[T]) -> Result<T, ToolkitError> {
    match v.iter().max() {
        None => Err(ToolkitError::from_str("container has no max value")),
        Some(v) => Ok(*v),
    }
}

/// Returns the average element of the slice.
pub fn avg_with_errhand<T>(v: &[T]) -> Result<T, ToolkitError>
where
    T: ToPrimitive + FromPrimitive + Copy,
{
    if v.is_empty() {
        return Err(ToolkitError::from_str("container is empty"));
    }

    let sum: f64 = v
        .iter()
        .map(|&x| {
            x.to_f64()
                .ok_or_else(|| ToolkitError::from_str("conversion to f64 failed"))
        })
        .collect::<Result<Vec<f64>, ToolkitError>>()?
        .iter()
        .sum();
    let mean = sum / (v.len() as f64);

    T::from_f64(mean).ok_or(ToolkitError::from_str("conversion error"))
}

pub fn log_level_to_int(level: Level) -> i32 {
    match level {
        Level::TRACE => 0,
        Level::DEBUG => 1,
        Level::INFO => 2,
        Level::WARN => 3,
        Level::ERROR => 4,
    }
}

pub fn int_to_loglevel(level_int: i32) -> Result<Level, ToolkitError> {
    match level_int {
        0 => Ok(Level::TRACE),
        1 => Ok(Level::DEBUG),
        2 => Ok(Level::INFO),
        3 => Ok(Level::WARN),
        4 => Ok(Level::ERROR),
        _ => Err(ToolkitError::from_str(&format!(
            "invalid log level: {level_int}"
        ))),
    }
}

#[cfg(test)]
pub mod test_utils {
    pub trait NumberTrait: Into<f64> + Copy {}
    impl<T> NumberTrait for T where T: Into<f64> + Copy {}

    #[track_caller]
    pub fn assert_almost_equal<T: NumberTrait, T2: NumberTrait>(left: T, right: T, atol: T2) {
        // function is used for testing only, can be slow from casting
        let left_f64: f64 = left.into();
        let right_f64: f64 = right.into();
        let atol_f64: f64 = atol.into();
        if (left_f64 - right_f64).abs() > atol_f64 {
            panic!("{left_f64} != {right_f64} up to absolute tolerance {atol_f64}")
        }
    }

    #[track_caller]
    pub fn assert_almost_equal_vec<T: NumberTrait, T2: NumberTrait>(
        left: &[T],
        right: &[T],
        atol: T2,
    ) {
        for (&a, &b) in std::iter::zip(left, right) {
            assert_almost_equal(a, b, atol);
        }
    }
}
