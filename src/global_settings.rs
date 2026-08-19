use half::bf16;
use once_cell::sync::Lazy;

/// This factor is used to support the deprecated int16 multiplication, will be removed with 2.4
pub static FLOAT_TO_FIXED_FAC_F32: f32 = 1000.0;
pub static FLOAT_TO_FIXED_FAC_BF16: Lazy<bf16> =
    Lazy::new(|| bf16::from_f32(FLOAT_TO_FIXED_FAC_F32));

pub type Value = bf16;
