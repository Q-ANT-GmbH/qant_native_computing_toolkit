use crate::errors::ToolkitError;
use tracing::Level;

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
    use half::bf16;

    pub trait NumberTrait: Into<f64> + Copy {}
    impl<T> NumberTrait for T where T: Into<f64> + Copy {}

    pub fn to_bf16s<T: Into<f64> + Copy>(x: &[T]) -> Vec<bf16> {
        x.iter().map(|&x| bf16::from_f64(x.into())).collect()
    }

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
