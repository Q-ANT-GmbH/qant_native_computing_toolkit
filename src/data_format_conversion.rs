use half::bf16;

use crate::global_settings::FLOAT_TO_FIXED_FAC_BF16;

#[inline]
pub fn float_to_fixed(v: bf16) -> i16 {
    bf16::to_f32(v * *FLOAT_TO_FIXED_FAC_BF16) as i16
}

#[inline]
pub fn fixed_to_float(v: i16) -> bf16 {
    bf16::from_f32(v as f32) / *FLOAT_TO_FIXED_FAC_BF16
}

pub fn floats_to_fixeds(vs: &[bf16]) -> Vec<i16> {
    vs.iter().map(|&e| float_to_fixed(e)).collect()
}

pub fn fixeds_to_floats(vs: &[i16]) -> Vec<bf16> {
    vs.iter().map(|&e| fixed_to_float(e)).collect()
}

pub fn to_bf16<T: Into<f64>>(x: T) -> bf16 {
    bf16::from_f64(x.into())
}

pub fn to_bf16s<T: Into<f64> + Copy>(x: &[T]) -> Vec<bf16> {
    x.iter().map(|&x| to_bf16(x)).collect()
}

pub fn to_f32(x: bf16) -> f32 {
    x.into()
}
pub fn to_f32s(x: &[bf16]) -> Vec<f32> {
    x.iter().map(|&x| to_f32(x)).collect()
}
