//! Module to expose some selected functions to python.
use half::bf16;
use numpy::ndarray::{Array, Array1};
use numpy::{
    IntoPyArray, PyArray, PyArray1, PyArray2, PyArray4, PyArrayMethods, PyReadonlyArray,
    PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4, PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;
use pyo3::{Bound, PyResult, Python, exceptions, pymodule, types::PyModule};

// needs to be matched exactly by the python numpy dtype
type Feature = crate::global_settings::Value;

use crate::data_structures::QantTensor;
use crate::errors::ToolkitError;
use crate::utils;

impl From<ToolkitError> for PyErr {
    fn from(err: ToolkitError) -> PyErr {
        exceptions::PyRuntimeError::new_err(format!("Rust core error: {err}"))
    }
}

#[pyfunction]
fn init_npu(id: u32) -> PyResult<()> {
    Ok(crate::init_npu(id)?)
}

#[pyfunction]
fn release_npu(id: u32) -> PyResult<()> {
    Ok(crate::release_npu(id)?)
}

#[pyfunction]
fn setup_logging(folder_name: String, loglevel: i32) -> PyResult<()> {
    let loglvl = utils::int_to_loglevel(loglevel)?;
    Ok(crate::setup_logging(&folder_name, loglvl)?)
}

#[pyfunction]
fn get_driver_info(npu_id: u32) -> PyResult<String> {
    Ok(crate::get_driver_info(npu_id)?)
}

fn pyarray_to_slice<'py, Type: numpy::Element, Dim: ndarray::Dimension>(
    x: &'py PyReadonlyArray<'py, Type, Dim>,
) -> Result<&'py [Type], pyo3::PyErr> {
    if !x.is_c_contiguous() {
        return Err(ToolkitError::from_str(
            "Cannot convert python array to slice. Data must be C-contiguous in memory",
        )
        .into());
    };
    Ok(x.as_slice()?)
}

fn pyarray_to_qant_tensor<'py, const D: usize, DNumpy: ndarray::Dimension>(
    pyarr: &'py PyReadonlyArray<'py, Feature, DNumpy>,
) -> Result<QantTensor<'py, D>, pyo3::PyErr> {
    if pyarr.ndim() != D {
        return Err(exceptions::PyRuntimeError::new_err(format!(
            "wrong dimensions, got {0}, expected {D}",
            pyarr.ndim()
        )));
    }

    let shape = pyarr.shape();
    let data = pyarray_to_slice(pyarr)?;
    let dims = shape.try_into()?;
    let arr: QantTensor<D> = QantTensor::new(data, dims)?;

    Ok(arr)
}

fn qant_tensor_to_pyarray<'py, const D: usize, DNumpy>(
    py: Python<'py>,
    fdata: QantTensor<D>,
) -> PyResult<Bound<'py, PyArray<Feature, DNumpy>>>
where
    DNumpy: ndarray::Dimension,
{
    let dims: DNumpy = DNumpy::from_dimension(&numpy::IxDyn(fdata.shape())).ok_or_else(|| {
        exceptions::PyValueError::new_err(format!(
            "Invalid tensor dimension, cannot convert from shape {0:?}",
            &fdata.shape()
        ))
    })?;

    match Array::from_shape_vec(dims, fdata.data().to_vec()) {
        Err(e) => Err(exceptions::PyRuntimeError::new_err(format!(
            "cannot create rust array: {e}"
        ))),
        Ok(res_array) => Ok(res_array.into_pyarray(py)),
    }
}

#[pyfunction]
fn linear_fprop<'py>(
    py: Python<'py>,
    npu_id: u32,
    feature_arr: PyReadonlyArray2<'py, Feature>,
    filter_arr: PyReadonlyArray2<'py, Feature>,
) -> PyResult<Bound<'py, PyArray2<Feature>>> {
    let feature_data = pyarray_to_qant_tensor(&feature_arr)?;
    let filter_data = pyarray_to_qant_tensor(&filter_arr)?;

    let res = crate::linear::linear_fprop(npu_id, &feature_data, &filter_data)?;

    qant_tensor_to_pyarray(py, res)
}

#[pyfunction]
fn conv_fprop<'py>(
    py: Python<'py>,
    npu_id: u32,
    feature_arr: PyReadonlyArray4<'py, Feature>,
    filter_arr: PyReadonlyArray4<'py, Feature>,
    padding: usize,
    stride: usize,
    dilation: usize,
) -> PyResult<Bound<'py, PyArray4<Feature>>> {
    let feature_data = pyarray_to_qant_tensor(&feature_arr)?;
    let filter_data = pyarray_to_qant_tensor(&filter_arr)?;

    let conv_attrs = crate::conv::ConvAttributes {
        padding,
        stride,
        dilation,
    };

    let res = crate::conv::conv_fprop(npu_id, &feature_data, &filter_data, conv_attrs)?;

    qant_tensor_to_pyarray(py, res)
}

#[pyfunction]
fn conv_transpose_fprop<'py>(
    py: Python<'py>,
    npu_id: u32,
    feature_arr: PyReadonlyArray4<'py, Feature>,
    filter_arr: PyReadonlyArray4<'py, Feature>,
    padding: usize,
    stride: usize,
    dilation: usize,
    output_padding: usize,
) -> PyResult<Bound<'py, PyArray4<Feature>>> {
    let feature_data = pyarray_to_qant_tensor(&feature_arr)?;
    let filter_data = pyarray_to_qant_tensor(&filter_arr)?;

    let conv_attrs = crate::conv::ConvTransposeAttributes {
        padding,
        stride,
        dilation,
        output_padding,
    };

    let res = crate::conv::conv_transpose_fprop(npu_id, &feature_data, &filter_data, conv_attrs)?;

    qant_tensor_to_pyarray(py, res)
}

#[pyfunction]
fn add_bias_fprop<'py>(
    py: Python<'py>,
    npu_id: u32,
    feature_arr: PyReadonlyArray2<'py, Feature>,
    bias_arr: PyReadonlyArray1<'py, Feature>,
) -> PyResult<Bound<'py, PyArray2<Feature>>> {
    let feature_data = pyarray_to_qant_tensor(&feature_arr)?;
    let bias_data = pyarray_to_qant_tensor(&bias_arr)?;

    if feature_data.shape()[1] != bias_data.shape()[0] {
        return Err(exceptions::PyRuntimeError::new_err(format!(
            "features and bias shapes do not match {} elements vs {} elements",
            feature_data.shape()[1],
            bias_data.shape()[0]
        )));
    }

    let res = crate::bias::add_bias_fprop(npu_id, &feature_data, &bias_data)?;

    qant_tensor_to_pyarray(py, res)
}

#[pyfunction]
fn add_bias_for_conv2d_fprop<'py>(
    py: Python<'py>,
    npu_id: u32,
    feature_arr: PyReadonlyArray4<'py, Feature>,
    bias_arr: PyReadonlyArray1<'py, Feature>,
) -> PyResult<Bound<'py, PyArray4<Feature>>> {
    let feature_data = pyarray_to_qant_tensor(&feature_arr)?;
    let bias_data = pyarray_to_qant_tensor(&bias_arr)?;

    if feature_data.shape()[1] != bias_data.shape()[0] {
        return Err(exceptions::PyRuntimeError::new_err(format!(
            "features and bias shapes do not match {} elements vs {} elements",
            feature_data.shape()[1],
            bias_data.shape()[0]
        )));
    }

    let res = crate::bias::add_bias_for_conv2d_fprop(npu_id, &feature_data, &bias_data)?;

    qant_tensor_to_pyarray(py, res)
}

#[pyfunction]
fn calc_scaled_periodic_nl_fprop<'py>(
    py: Python<'py>,
    npu_id: u32,
    feature_arr: PyReadonlyArray1<'py, Feature>,
    weights_arr: PyReadonlyArray1<'py, Feature>,
) -> PyResult<Bound<'py, PyArray1<Feature>>> {
    let feature_data = pyarray_to_qant_tensor(&feature_arr)?;
    let weights_data = pyarray_to_qant_tensor(&weights_arr)?;

    let res = crate::non_linear::calc_scaled_periodic_nl(npu_id, &feature_data, &weights_data)?;

    qant_tensor_to_pyarray(py, res)
}

#[pyfunction]
fn calc_kan_layer_fprop<'py>(
    py: Python<'py>,
    npu_id: u32,
    feature_arr: PyReadonlyArray2<'py, Feature>,
    phis_arr: PyReadonlyArray3<'py, Feature>,
    ampls_arr: PyReadonlyArray3<'py, Feature>,
    ks_arr: PyReadonlyArray1<'py, Feature>,
) -> PyResult<Bound<'py, PyArray2<Feature>>> {
    let feature_data = pyarray_to_qant_tensor(&feature_arr)?;
    let phis_data = pyarray_to_qant_tensor(&phis_arr)?;
    let ampls_data = pyarray_to_qant_tensor(&ampls_arr)?;
    let ks = ks_arr.to_owned_array().to_vec();

    let res =
        crate::non_linear::calc_kan_layer(npu_id, &feature_data, &phis_data, &ampls_data, &ks)?;

    qant_tensor_to_pyarray(py, res)
}

#[pyfunction]
fn mul_f32<'py>(
    py: Python<'py>,
    npu_id: u32,
    us: PyReadonlyArray1<'py, f32>,
    vs: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let us_slice = pyarray_to_slice(&us)?;
    let vs_slice = pyarray_to_slice(&vs)?;

    let res = crate::mul_elementwise_f32(npu_id, us_slice, vs_slice)?;
    let res_array = Array1::from_vec(res);
    Ok(res_array.into_pyarray(py))
}

#[pyfunction]
fn mul<'py>(
    py: Python<'py>,
    npu_id: u32,
    us: PyReadonlyArray1<'py, Feature>,
    vs: PyReadonlyArray1<'py, Feature>,
) -> PyResult<Bound<'py, PyArray1<Feature>>> {
    let us_fdata = pyarray_to_qant_tensor(&us)?;
    let vs_fdata = pyarray_to_qant_tensor(&vs)?;

    let res = crate::mul_elementwise_bf16(npu_id, &us_fdata, &vs_fdata)?;
    qant_tensor_to_pyarray(py, res)
}

#[pyfunction]
fn relu_fprop<'py>(
    py: Python<'py>,
    npu_id: u32,
    feature_arr: PyReadonlyArray1<'py, Feature>,
) -> PyResult<Bound<'py, PyArray1<Feature>>> {
    let feature_data = pyarray_to_qant_tensor(&feature_arr)?;
    let res = crate::non_linear::relu_fprop(npu_id, &feature_data);
    qant_tensor_to_pyarray(py, res)
}

#[pyfunction]
fn sigmoid_fprop<'py>(
    py: Python<'py>,
    npu_id: u32,
    feature_arr: PyReadonlyArray1<'py, Feature>,
) -> PyResult<Bound<'py, PyArray1<Feature>>> {
    let feature_data = pyarray_to_qant_tensor(&feature_arr)?;
    let res = crate::non_linear::sigmoid_fprop(npu_id, &feature_data);
    qant_tensor_to_pyarray(py, res)
}

#[pyfunction]
fn softmax_fprop<'py>(
    py: Python<'py>,
    npu_id: u32,
    feature_arr: PyReadonlyArray2<'py, Feature>,
) -> PyResult<Bound<'py, PyArray2<Feature>>> {
    let feature_data = pyarray_to_qant_tensor(&feature_arr)?;
    let res = crate::non_linear::softmax_fprop(npu_id, &feature_data)?;
    qant_tensor_to_pyarray(py, res)
}

#[pyfunction]
fn batchnorm2d_fprop<'py>(
    py: Python<'py>,
    npu_id: u32,
    feature_arr: PyReadonlyArray4<'py, Feature>,
    means: PyReadonlyArray1<'py, Feature>,
    variances: PyReadonlyArray1<'py, Feature>,
    weights: PyReadonlyArray1<'py, Feature>,
    bias: PyReadonlyArray1<'py, Feature>,
    eps: f32,
) -> PyResult<Bound<'py, PyArray4<Feature>>> {
    let feature_data = pyarray_to_qant_tensor(&feature_arr)?;
    let means_fdata = pyarray_to_qant_tensor(&means)?;
    let variances_fdata = pyarray_to_qant_tensor(&variances)?;
    let weights_fdata = pyarray_to_qant_tensor(&weights)?;
    let bias_fdata = pyarray_to_qant_tensor(&bias)?;

    let res = crate::norm::batchnorm2d_fprop(
        npu_id,
        &feature_data,
        &means_fdata,
        &variances_fdata,
        &weights_fdata,
        &bias_fdata,
        bf16::from_f32(eps),
    )?;

    qant_tensor_to_pyarray(py, res)
}

#[pyfunction]
fn maxpool2d_fprop<'py>(
    py: Python<'py>,
    npu_id: u32,
    feature_arr: PyReadonlyArray4<'py, Feature>,
    kernel_height: usize,
    kernel_width: usize,
    stride: usize,
    padding: usize,
) -> PyResult<Bound<'py, PyArray4<Feature>>> {
    let feature_data = pyarray_to_qant_tensor(&feature_arr)?;

    let pool_attrs = crate::pooling::PoolingAttributes {
        kernel_height,
        kernel_width,
        stride,
        padding,
        count_include_pad: false,
    };

    let res = crate::pooling::maxpool2d_fprop(npu_id, &feature_data, pool_attrs)?;

    qant_tensor_to_pyarray(py, res)
}

#[pyfunction]
fn avgpool2d_fprop<'py>(
    py: Python<'py>,
    npu_id: u32,
    feature_arr: PyReadonlyArray4<'py, Feature>,
    kernel_height: usize,
    kernel_width: usize,
    stride: usize,
    padding: usize,
    count_include_pad: bool,
) -> PyResult<Bound<'py, PyArray4<Feature>>> {
    let feature_data = pyarray_to_qant_tensor(&feature_arr)?;

    let pool_attrs = crate::pooling::PoolingAttributes {
        kernel_height,
        kernel_width,
        stride,
        padding,
        count_include_pad,
    };

    let res = crate::pooling::avgpool2d_fprop(npu_id, &feature_data, pool_attrs)?;

    qant_tensor_to_pyarray(py, res)
}

#[pyfunction]
fn adaptive_maxpool2d_fprop<'py>(
    py: Python<'py>,
    npu_id: u32,
    feature_arr: PyReadonlyArray4<'py, Feature>,
    output_height: usize,
    output_width: usize,
) -> PyResult<Bound<'py, PyArray4<Feature>>> {
    let feature_data = pyarray_to_qant_tensor(&feature_arr)?;

    let res = crate::pooling::adaptive_maxpool2d_fprop(
        npu_id,
        &feature_data,
        output_height,
        output_width,
    )?;

    qant_tensor_to_pyarray(py, res)
}

#[pyfunction]
fn adaptive_avgpool2d_fprop<'py>(
    py: Python<'py>,
    npu_id: u32,
    feature_arr: PyReadonlyArray4<'py, Feature>,
    output_height: usize,
    output_width: usize,
) -> PyResult<Bound<'py, PyArray4<Feature>>> {
    let feature_data = pyarray_to_qant_tensor(&feature_arr)?;

    let res = crate::pooling::adaptive_avgpool2d_fprop(
        npu_id,
        &feature_data,
        output_height,
        output_width,
    )?;

    qant_tensor_to_pyarray(py, res)
}

#[pyfunction]
fn get_sensor_info(py: Python, npu_id: u32) -> PyResult<Bound<PyDict>> {
    let info = crate::get_device_info(npu_id)?;
    let ret_dict = PyDict::new(py);

    let sens_info = info.sensor;
    ret_dict.set_item("temp_pd1", sens_info.temp_pd1)?;
    ret_dict.set_item("temp_pd2", sens_info.temp_pd2)?;
    ret_dict.set_item("temp_pd3", sens_info.temp_pd3)?;
    ret_dict.set_item("temp_dac_1", sens_info.temp_dac_1)?;
    ret_dict.set_item("temp_dac_2", sens_info.temp_dac_2)?;
    ret_dict.set_item("voltage_1v8", sens_info.voltage_1v8)?;
    ret_dict.set_item("voltage_3v3", sens_info.voltage_3v3)?;
    ret_dict.set_item("voltage_6v0_1", sens_info.voltage_6v0_1)?;
    ret_dict.set_item("voltage_6v0_2", sens_info.voltage_6v0_2)?;
    ret_dict.set_item("voltage_6v2", sens_info.voltage_6v2)?;
    ret_dict.set_item("voltage_6v5", sens_info.voltage_6v5)?;
    ret_dict.set_item("voltage_8v0_1", sens_info.voltage_8v0_1)?;
    ret_dict.set_item("voltage_8v0_2", sens_info.voltage_8v0_2)?;
    ret_dict.set_item("voltage_12v", sens_info.voltage_12v)?;
    ret_dict.set_item("voltage_12v5", sens_info.voltage_12v5)?;
    ret_dict.set_item("voltage_n12v5", sens_info.voltage_n12v5)?;
    ret_dict.set_item("current_12v", sens_info.current_12v)?;
    Ok(ret_dict)
}

#[pyfunction]
fn get_version_info(py: Python, npu_id: u32) -> PyResult<Bound<PyDict>> {
    let info = crate::get_device_info(npu_id)?;
    let ret_dict = PyDict::new(py);

    ret_dict.set_item("fw_version", info.version_info.fw_version)?;
    ret_dict.set_item("zephyr_version", info.version_info.zephyr_version)?;
    ret_dict.set_item("board_serial_no", info.version_info.board_serial_no)?;

    Ok(ret_dict)
}

#[pyfunction]
fn get_available_npus(py: Python) -> PyResult<Bound<PyDict>> {
    let npus = crate::get_available_npus()?;
    let ret_dict = PyDict::new(py);

    for (npu_id, npu_serial) in npus.into_iter() {
        ret_dict.set_item(npu_id, npu_serial)?;
    }
    Ok(ret_dict)
}

#[pyfunction]
fn reset_perf_counter(npu_id: u32) -> PyResult<()> {
    Ok(crate::qant_driver_import::driver_reset_perf_counter(
        npu_id,
    )?)
}

#[pyfunction]
fn get_perf_counter(py: Python, npu_id: u32) -> PyResult<Bound<PyDict>> {
    let info = crate::qant_driver_import::driver_get_perf_counter(npu_id)?;

    let ret_dict = PyDict::new(py);
    ret_dict.set_item("timebased_counter", info.timebased_counter)?;
    ret_dict.set_item("stalling_counter", info.stalling_counter)?;
    Ok(ret_dict)
}

// This function is mandated by pyo3 and tells it the name of the python module,
// as well as the functions that are available in it.
#[pymodule]
#[pyo3(name = "_backend")]
fn qant_native_computing_toolkit(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init_npu, m)?)?;
    m.add_function(wrap_pyfunction!(mul_f32, m)?)?;
    m.add_function(wrap_pyfunction!(mul, m)?)?;
    m.add_function(wrap_pyfunction!(release_npu, m)?)?;
    m.add_function(wrap_pyfunction!(setup_logging, m)?)?;
    m.add_function(wrap_pyfunction!(get_driver_info, m)?)?;
    m.add_function(wrap_pyfunction!(linear_fprop, m)?)?;
    m.add_function(wrap_pyfunction!(conv_fprop, m)?)?;
    m.add_function(wrap_pyfunction!(add_bias_fprop, m)?)?;
    m.add_function(wrap_pyfunction!(add_bias_for_conv2d_fprop, m)?)?;
    m.add_function(wrap_pyfunction!(calc_scaled_periodic_nl_fprop, m)?)?;
    m.add_function(wrap_pyfunction!(calc_kan_layer_fprop, m)?)?;
    m.add_function(wrap_pyfunction!(relu_fprop, m)?)?;
    m.add_function(wrap_pyfunction!(softmax_fprop, m)?)?;
    m.add_function(wrap_pyfunction!(batchnorm2d_fprop, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid_fprop, m)?)?;
    m.add_function(wrap_pyfunction!(conv_transpose_fprop, m)?)?;
    m.add_function(wrap_pyfunction!(maxpool2d_fprop, m)?)?;
    m.add_function(wrap_pyfunction!(avgpool2d_fprop, m)?)?;
    m.add_function(wrap_pyfunction!(adaptive_maxpool2d_fprop, m)?)?;
    m.add_function(wrap_pyfunction!(adaptive_avgpool2d_fprop, m)?)?;
    m.add_function(wrap_pyfunction!(get_sensor_info, m)?)?;
    m.add_function(wrap_pyfunction!(get_version_info, m)?)?;
    m.add_function(wrap_pyfunction!(get_available_npus, m)?)?;
    m.add_function(wrap_pyfunction!(reset_perf_counter, m)?)?;
    m.add_function(wrap_pyfunction!(get_perf_counter, m)?)?;
    Ok(())
}
