#define CATCH_CONFIG_MAIN
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <iostream>
#include <random>
#include <string>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xtensor.hpp>

#include "dlpack_utils.h"
#include "qant_native_computing_toolkit.h"

typedef xt::xarray<std::bfloat16_t> Tensor;

namespace q_generic = ::qant_native_computing_toolkit::generic;
namespace q_native = ::qant_native_computing_toolkit::native;
namespace q_ai = ::qant_native_computing_toolkit::ai;
namespace q_info = ::qant_native_computing_toolkit::info;

TEST_CASE("version check")
{
    REQUIRE(QANT_NATIVE_COMPUTING_TOOLKIT_MAJOR_VERSION == 2);
    REQUIRE(QANT_NATIVE_COMPUTING_TOOLKIT_MINOR_VERSION >= 0);
}

TEST_CASE("init release")
{
    REQUIRE(q_generic::init_npu(0) == 0);
    REQUIRE(q_generic::release_npu(0) == 0);
}

TEST_CASE("mul_f32")
{

    std::vector<float> us = {0.5, 0.2, 0.3, 0.4};
    std::vector<float> vs = {0.5, 0.3, 0.2, 0.1};
    auto us_tensor = vector_to_dltensor<float>(us);
    auto vs_tensor = vector_to_dltensor<float>(vs);

    auto ws_dltensor = q_native::mul_npu_f32(0, &us_tensor, &vs_tensor);
    REQUIRE(q_generic::release_npu(0) == 0);
    REQUIRE(ws_dltensor != nullptr);
    auto ws = tensor_to_vector_view<float>(ws_dltensor);

    auto atol = 0.002;
    for (int i = 0; i < 3; ++i)
    {
        CHECK_THAT(ws[i], Catch::Matchers::WithinAbs(us[i] * vs[i], atol));
    }

    us_tensor.deleter(&us_tensor);
    vs_tensor.deleter(&vs_tensor);
    ws_dltensor->deleter(ws_dltensor);
}

TEST_CASE("mul_i16")
{
    std::vector<int16_t> us = {500, 200, 300, 400};
    std::vector<int16_t> vs = {500, 300, 200, 100};
    auto us_tensor = vector_to_dltensor<std::int16_t>(us);
    auto vs_tensor = vector_to_dltensor<std::int16_t>(vs);

    auto ws_dltensor = q_native::mul_npu_i16(0, &us_tensor, &vs_tensor);
    REQUIRE(q_generic::release_npu(0) == 0);
    REQUIRE(ws_dltensor != nullptr);
    auto ws = tensor_to_vector_view<std::int16_t>(ws_dltensor);

    auto atol = 0.002;
    for (int i = 0; i < 3; ++i)
    {
        CHECK_THAT(ws[i], Catch::Matchers::WithinAbs((us[i] * vs[i]) / 1000, atol));
    }

    us_tensor.deleter(&us_tensor);
    vs_tensor.deleter(&vs_tensor);
    ws_dltensor->deleter(ws_dltensor);
}

TEST_CASE("mul_bf16")
{
    Tensor us = xt::random::rand({100}, -1.0, 1.0);
    Tensor vs = xt::random::rand({100}, -1.0, 1.0);

    auto us_tensor = xarray_to_dltensor(us);
    auto vs_tensor = xarray_to_dltensor(vs);

    auto res_qant_dltensor = q_native::mul_npu(0, &us_tensor, &vs_tensor);
    REQUIRE(q_generic::release_npu(0) == 0);
    REQUIRE(res_qant_dltensor != nullptr);

    auto res_qant = dltensor_to_tensor_view(res_qant_dltensor);

    auto res_shouldbe = us * vs;
    REQUIRE(xt::allclose(res_qant, res_shouldbe, 0.0, 0.01));

    us_tensor.deleter(&us_tensor);
    vs_tensor.deleter(&vs_tensor);
    res_qant_dltensor->deleter(res_qant_dltensor);
}

TEST_CASE("linear_fprop")
{
    auto n_columns = 10;
    auto n_rows = 20;

    Tensor vec = xt::random::rand({1, n_columns}, -1.0, 1.0);
    Tensor mat = xt::random::rand({n_rows, n_columns}, -1.0, 1.0);

    std::cout << "Random Vector:\n"
              << vec << std::endl;
    std::cout << "Random Matrix:\n"
              << mat << std::endl;

    auto vec_dltensor = xarray_to_dltensor(vec);
    auto mat_dltensor = xarray_to_dltensor(mat);

    auto result_qant_dltensor = q_native::linear_fprop(0, &vec_dltensor, &mat_dltensor);
    REQUIRE(q_generic::release_npu(0) == 0);

    REQUIRE(result_qant_dltensor != nullptr);
    auto result_qant = dltensor_to_tensor_view(result_qant_dltensor);

    auto result_shouldbe = xt::linalg::tensordot(mat, vec, {1}, {1});

    std::cout << "result should be:\n"
              << result_shouldbe << std::endl;
    std::cout << "result qant:\n"
              << result_qant << std::endl;
    REQUIRE(xt::allclose(result_qant.reshape({-1}), result_shouldbe.reshape({-1}), 0.0, 0.025));

    mat_dltensor.deleter(&mat_dltensor);
    vec_dltensor.deleter(&vec_dltensor);
    result_qant_dltensor->deleter(result_qant_dltensor);
}

TEST_CASE("conv_fprop")
{
    Tensor features = xt::random::rand({1, 12, 3, 4}, -1.0, 1.0);
    Tensor filter = xt::random::rand({3, 12, 5, 6}, -1.0, 1.0);

    auto features_dltensor = xarray_to_dltensor(features);
    auto filter_dltensor = xarray_to_dltensor(filter);

    auto qant_res_dltensor = q_ai::conv_fprop(0, &features_dltensor, &filter_dltensor, 2, 3, 1);
    REQUIRE(q_generic::release_npu(0) == 0);
    REQUIRE(qant_res_dltensor != nullptr);

    features_dltensor.deleter(&features_dltensor);
    filter_dltensor.deleter(&filter_dltensor);
    qant_res_dltensor->deleter(qant_res_dltensor);
}

TEST_CASE("conv_transpose_fprop")
{
    Tensor features = xt::random::rand({1, 12, 3, 4}, -1.0, 1.0);
    Tensor filter = xt::random::rand({12, 3, 5, 6}, -1.0, 1.0);

    auto features_dltensor = xarray_to_dltensor(features);
    auto filter_dltensor = xarray_to_dltensor(filter);

    auto qant_res_dltensor = q_ai::conv_transpose_fprop(0, &features_dltensor, &filter_dltensor, 2, 3, 1, 0);
    REQUIRE(q_generic::release_npu(0) == 0);
    REQUIRE(qant_res_dltensor != nullptr);

    features_dltensor.deleter(&features_dltensor);
    filter_dltensor.deleter(&filter_dltensor);
    qant_res_dltensor->deleter(qant_res_dltensor);
}

TEST_CASE("add_bias_fprop")
{
    Tensor features = xt::random::rand({1, 14}, -1.0, 1.0);
    Tensor bias = xt::random::rand({14}, -1.0, 1.0);

    auto features_dltensor = xarray_to_dltensor(features);
    auto bias_dltensor = xarray_to_dltensor(bias);

    auto qant_res_dltensor = q_ai::add_bias_fprop(0, &features_dltensor, &bias_dltensor);

    REQUIRE(qant_res_dltensor != nullptr);
    auto qant_res = dltensor_to_tensor_view(qant_res_dltensor);
    REQUIRE(qant_res.shape() == (features + bias).shape());
    REQUIRE(qant_res == features + bias);

    // batch input test
    Tensor batched_features = xt::random::rand({5, 14}, -1.0, 1.0);

    auto batched_features_dltensor = xarray_to_dltensor(batched_features);

    auto qant_batched_res_dltensor = q_ai::add_bias_fprop(0, &batched_features_dltensor, &bias_dltensor);
    REQUIRE(q_generic::release_npu(0) == 0);

    REQUIRE(qant_res_dltensor != nullptr);
    auto qant_batched_res = dltensor_to_tensor_view(qant_batched_res_dltensor);
    bias = xt::repeat(bias.reshape({1, 14}), 5, 0);
    REQUIRE(qant_batched_res.shape() == (batched_features + bias).shape());
    REQUIRE(qant_batched_res == batched_features + bias);

    features_dltensor.deleter(&features_dltensor);
    batched_features_dltensor.deleter(&batched_features_dltensor);
    bias_dltensor.deleter(&bias_dltensor);
    qant_res_dltensor->deleter(qant_res_dltensor);
    qant_batched_res_dltensor->deleter(qant_batched_res_dltensor);
}

TEST_CASE("add_bias_for_conv2d_fprop")
{
    // batch input test, note: this test does not test the result
    Tensor batched_features = xt::random::rand({2, 3, 2, 2}, -1.0, 1.0);
    Tensor bias = xt::random::rand({3}, -1.0, 1.0);

    auto batched_features_dltensor = xarray_to_dltensor(batched_features);
    auto bias_dltensor = xarray_to_dltensor(bias);

    auto qant_batched_res_dltensor = q_ai::add_bias_for_conv2d_fprop(0, &batched_features_dltensor, &bias_dltensor);
    REQUIRE(q_generic::release_npu(0) == 0);

    REQUIRE(qant_batched_res_dltensor != nullptr);

    batched_features_dltensor.deleter(&batched_features_dltensor);
    bias_dltensor.deleter(&bias_dltensor);
    qant_batched_res_dltensor->deleter(qant_batched_res_dltensor);
}

TEST_CASE("relu_fprop")
{
    Tensor features = xt::random::rand({144}, -1.0, 1.0);
    auto features_dltensor = xarray_to_dltensor(features);

    auto qant_res_dltensor = q_ai::relu_fprop(0, &features_dltensor);
    REQUIRE(q_generic::release_npu(0) == 0);
    REQUIRE(qant_res_dltensor != nullptr);
    auto qant_res = dltensor_to_tensor_view(qant_res_dltensor);
    REQUIRE(qant_res.shape() == features.shape());
    REQUIRE(qant_res == xt::maximum(features, 0.));

    features_dltensor.deleter(&features_dltensor);
    qant_res_dltensor->deleter(qant_res_dltensor);
}

TEST_CASE("sigmoid")
{
    Tensor features = xt::random::rand({144}, -1.0, 1.0);
    auto features_dltensor = xarray_to_dltensor(features);

    auto qant_res_dltensor = q_ai::sigmoid_fprop(0, &features_dltensor);
    REQUIRE(q_generic::release_npu(0) == 0);
    REQUIRE(qant_res_dltensor != nullptr);

    features_dltensor.deleter(&features_dltensor);
    qant_res_dltensor->deleter(qant_res_dltensor);
}

TEST_CASE("softmax")
{
    Tensor features = xt::random::rand({1, 144}, -1.0, 1.0);
    auto features_dltensor = xarray_to_dltensor(features);

    auto qant_res_dltensor = q_ai::softmax_fprop(0, &features_dltensor);
    REQUIRE(q_generic::release_npu(0) == 0);
    REQUIRE(qant_res_dltensor != nullptr);

    features_dltensor.deleter(&features_dltensor);
    qant_res_dltensor->deleter(qant_res_dltensor);
}

TEST_CASE("batchnorm2d")
{
    std::float32_t eps = 0.01f;
    auto n_chnls = 12;
    Tensor features = xt::cast<std::bfloat16_t>(xt::random::rand<float>({1, n_chnls, 3, 4}, -1.0, 1.0));
    Tensor means = xt::cast<std::bfloat16_t>(xt::random::rand<float>({n_chnls}, -1.0, 1.0));
    Tensor variances = xt::cast<std::bfloat16_t>(xt::random::rand<float>({n_chnls}, eps, 1.0));
    Tensor weights = xt::cast<std::bfloat16_t>(xt::random::rand<float>({n_chnls}, -1.0, 1.0));
    Tensor bias = xt::cast<std::bfloat16_t>(xt::random::rand<float>({n_chnls}, -1.0, 1.0));

    auto features_dltensor = xarray_to_dltensor(features);
    auto means_dltensor = xarray_to_dltensor(means);
    auto variances_dltensor = xarray_to_dltensor(variances);
    auto weights_dltensor = xarray_to_dltensor(weights);
    auto bias_dltensor = xarray_to_dltensor(bias);

    auto qant_res_dltensor = q_ai::batchnorm2d_fprop(0, &features_dltensor, &means_dltensor, &variances_dltensor, &weights_dltensor, &bias_dltensor, eps);
    REQUIRE(q_generic::release_npu(0) == 0);
    REQUIRE(qant_res_dltensor != nullptr);

    features_dltensor.deleter(&features_dltensor);
    means_dltensor.deleter(&means_dltensor);
    variances_dltensor.deleter(&variances_dltensor);
    weights_dltensor.deleter(&weights_dltensor);
    bias_dltensor.deleter(&bias_dltensor);
    qant_res_dltensor->deleter(qant_res_dltensor);
}

TEST_CASE("maxpool2d_fprop")
{
    Tensor features = xt::random::rand({1, 12, 3, 4}, -1.0, 1.0);

    auto features_dltensor = xarray_to_dltensor(features);

    auto qant_res_dltensor = q_ai::maxpool2d_fprop(0, &features_dltensor, 2, 2, 1, 1);
    REQUIRE(q_generic::release_npu(0) == 0);
    REQUIRE(qant_res_dltensor != nullptr);

    features_dltensor.deleter(&features_dltensor);
    qant_res_dltensor->deleter(qant_res_dltensor);
}

Tensor scaled_periodic_nl_cpu(Tensor const &us, Tensor const &vs)
{
    auto us_f32 = xt::cast<float>(us);
    auto vs_f32 = xt::cast<float>(vs);
    auto res = xt::cos(us_f32) * vs_f32;
    return xt::cast<std::bfloat16_t>(res);
}

TEST_CASE("scaled_periodic_nl")
{
    Tensor const features = xt::random::rand({4}, -7.0, 7.0);
    Tensor const weights = xt::random::rand({4}, -2.0, 2.0);

    auto features_dltensor = xarray_to_dltensor(features);
    auto weights_dltensor = xarray_to_dltensor(weights);

    auto qant_res_dltensor = q_native::calc_scaled_periodic_nl_fprop(0, &features_dltensor, &weights_dltensor);
    REQUIRE(q_generic::release_npu(0) == 0);
    REQUIRE(qant_res_dltensor != nullptr);

    auto res_npu = dltensor_to_tensor_view(qant_res_dltensor);

    auto res_shouldbe = scaled_periodic_nl_cpu(features, weights);
    std::cout << res_npu << res_shouldbe;
    REQUIRE(xt::allclose(res_npu.reshape({-1}), res_shouldbe.reshape({-1}), 0.0, 0.15));

    features_dltensor.deleter(&features_dltensor);
    weights_dltensor.deleter(&weights_dltensor);
    qant_res_dltensor->deleter(qant_res_dltensor);
}

TEST_CASE("kan_layer")
{
    Tensor const features = xt::random::rand({1, 10}, -1.0, 1.0);
    Tensor const phis = xt::random::rand({20, 10, 3}, -1.0, 1.0);
    Tensor const ampls = xt::random::rand({20, 10, 3}, -1.0, 1.0);
    Tensor const ks = xt::random::rand({3}, -1.0, 1.0);

    auto features_dltensor = xarray_to_dltensor(features);
    auto phis_dltensor = xarray_to_dltensor(phis);
    auto ampls_dltensor = xarray_to_dltensor(ampls);
    auto ks_dltensor = xarray_to_dltensor(ks);

    auto qant_res_dltensor = q_ai::calc_kan_layer_fprop(0, &features_dltensor, &phis_dltensor, &ampls_dltensor, &ks_dltensor);
    REQUIRE(q_generic::release_npu(0) == 0);
    REQUIRE(qant_res_dltensor != nullptr);

    features_dltensor.deleter(&features_dltensor);
    phis_dltensor.deleter(&phis_dltensor);
    ampls_dltensor.deleter(&ampls_dltensor);
    ks_dltensor.deleter(&ks_dltensor);
    qant_res_dltensor->deleter(qant_res_dltensor);
}

TEST_CASE("logging")
{
    REQUIRE(q_info::setup_logging("/tmp/", 2) == 0);
}

TEST_CASE("driver_info")
{
    // the function has a C API, but let's use it as Cpp-like as possible

    // allocate large space to be sure to get all info
    std::string buffer(QANT_NATIVE_COMPUTING_TOOLKIT_DEFAULT_CHAR_ARR_LENGTH, '\0');

    int result = q_info::get_driver_info(0, buffer.data(), buffer.size());
    REQUIRE(q_generic::release_npu(0) == 0);
    REQUIRE(result == 0);
    // shrink back to the actual required size
    buffer.resize(std::strlen(buffer.c_str()));
    std::cout << buffer << std::endl;
}

TEST_CASE("sensor_info")
{
    q_info::QantSensorInfo result = q_info::get_sensor_info(0);
    REQUIRE(q_generic::release_npu(0) == 0);

    std::cout << result.voltage_3v3 << std::endl;
    REQUIRE(result.voltage_3v3 > 3.);
    REQUIRE(result.current_12v >= 0.0);
}

TEST_CASE("version_info")
{
    q_info::QantVersionInfo result = q_info::get_version_info(0);
    REQUIRE(q_generic::release_npu(0) == 0);

    std::string zephyr_version(result.zephyr_version);
    std::cout << zephyr_version << std::endl;
    REQUIRE(zephyr_version.find("error") == std::string::npos);
}

TEST_CASE("available_npus")
{
    size_t const n_npus_max = 10;
    uint32_t idxs[n_npus_max];
    char serials[n_npus_max][QANT_NATIVE_COMPUTING_TOOLKIT_DEFAULT_CHAR_ARR_LENGTH];
    size_t n_npus = 0;

    int const err = q_info::get_available_npus(idxs, serials, n_npus_max, &n_npus);
    REQUIRE(err == 0);
    REQUIRE(n_npus > 0);
}

TEST_CASE("reset_performance_counter")
{
    REQUIRE(q_info::reset_perf_counter(0) == 0);
    REQUIRE(q_generic::release_npu(0) == 0);
}

TEST_CASE("get_performance_counter")
{
    q_info::QantPerformanceCounterInfo counter;

    int error = q_info::get_perf_counter(0, &counter);
    REQUIRE(q_generic::release_npu(0) == 0);
    REQUIRE(error == 0);
    REQUIRE(counter.stalling_counter >= 0);
    REQUIRE(counter.timebased_counter >= 0);
}
