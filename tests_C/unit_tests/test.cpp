#define CATCH_CONFIG_MAIN
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <complex>
#include <iostream>
#include <numbers>
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

// Returns the lowest id among the available NPUs that supports DFT.
int32_t dft_npu_id()
{
    size_t const n_npus_max = 20;
    uint32_t idxs[n_npus_max];
    char serials[n_npus_max][QANT_NATIVE_COMPUTING_TOOLKIT_DEFAULT_CHAR_ARR_LENGTH];
    size_t n_npus = 0;
    q_generic::QantDeviceCapability const ops[] = {q_generic::Dft};
    REQUIRE(q_generic::get_available_npus(ops, 1, idxs, serials, n_npus_max, &n_npus) == 0);
    REQUIRE(n_npus > 0);
    return static_cast<int32_t>(idxs[0]);
}

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

TEST_CASE("init release auto npu")
{
    REQUIRE(q_generic::init_npu(QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID) == 0);
    REQUIRE(q_generic::release_npu(QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID) == 0);
}

TEST_CASE("npu_id below default sentinel is rejected")
{
    REQUIRE(q_generic::init_npu(-2) == -1);
    REQUIRE(q_generic::release_npu(-2) == -1);
}

TEST_CASE("memory_allocate_and_free")
{
    auto handle = q_generic::npu_allocate(0, 2, 3, q_generic::QantDataRole::Weights);
    REQUIRE(handle != nullptr);

    REQUIRE(q_generic::npu_free(handle) == 0);
    REQUIRE(q_generic::release_npu(0) == 0);
}

TEST_CASE("memory_allocate auto npu")
{
    auto handle = q_generic::npu_allocate(QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID, 2, 3, q_generic::QantDataRole::Weights);
    REQUIRE(handle != nullptr);

    REQUIRE(q_generic::npu_free(handle) == 0);
    REQUIRE(q_generic::release_npu(0) == 0);
}

TEST_CASE("memory_upload_and_download")
{
    auto rows = 2;
    auto cols = 3;
    Tensor mat = xt::random::rand({rows, cols}, -1.0, 1.0);
    auto mat_dltensor = xarray_to_dltensor(mat);

    auto handle = q_generic::npu_allocate(0, rows, cols, q_generic::QantDataRole::Features);
    REQUIRE(handle != nullptr);

    REQUIRE(q_generic::npu_upload_into(handle, &mat_dltensor) == 0);

    auto downloaded_dltensor = q_generic::npu_download(handle);
    REQUIRE(downloaded_dltensor != nullptr);
    auto downloaded = dltensor_to_tensor_view(downloaded_dltensor);

    REQUIRE(xt::allclose(downloaded, mat, 0.0, 0.0));

    mat_dltensor.deleter(&mat_dltensor);
    downloaded_dltensor->deleter(downloaded_dltensor);
    REQUIRE(q_generic::npu_free(handle) == 0);
    REQUIRE(q_generic::release_npu(0) == 0);
}

TEST_CASE("linear_fprop_preloaded")
{
    auto n_channels_out = 4;
    auto n_channels_in = 3;
    auto n_batches = 3;

    Tensor weights = xt::random::rand({n_channels_out, n_channels_in}, -1.0, 1.0);
    Tensor features = xt::random::rand({n_batches, n_channels_in}, -1.0, 1.0);
    auto weights_dltensor = xarray_to_dltensor(weights);
    auto features_dltensor = xarray_to_dltensor(features);

    auto weights_handle = q_generic::npu_allocate(0, n_channels_out, n_channels_in, q_generic::QantDataRole::Weights);
    auto features_handle = q_generic::npu_upload(0, &features_dltensor, q_generic::QantDataRole::Features);
    auto result_handle = q_generic::npu_allocate(0, n_batches, n_channels_out, q_generic::QantDataRole::Features);
    REQUIRE(weights_handle != nullptr);
    REQUIRE(features_handle != nullptr);
    REQUIRE(result_handle != nullptr);

    REQUIRE(q_generic::npu_upload_into(weights_handle, &weights_dltensor) == 0);

    REQUIRE(q_native::linear_fprop_preloaded(features_handle, weights_handle, result_handle) == 0);

    weights_dltensor.deleter(&weights_dltensor);
    features_dltensor.deleter(&features_dltensor);
    REQUIRE(q_generic::npu_free(weights_handle) == 0);
    REQUIRE(q_generic::npu_free(features_handle) == 0);
    REQUIRE(q_generic::npu_free(result_handle) == 0);
    REQUIRE(q_generic::release_npu(0) == 0);
}

TEST_CASE("mul_elementwise")
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

TEST_CASE("linear_fprop_preloaded_weights")
{
    auto n_channels_in = 10;
    auto n_channels_out = 20;
    auto n_batches = 3;

    Tensor weights = xt::random::rand({n_channels_out, n_channels_in}, -1.0, 1.0);
    auto weights_dltensor = xarray_to_dltensor(weights);

    auto weights_handle = q_generic::npu_upload(0, &weights_dltensor, q_generic::QantDataRole::Weights);
    REQUIRE(weights_handle != nullptr);

    // upload the weights once and reuse them for two consecutive forward passes
    for (int i = 0; i < 2; i++)
    {
        Tensor features = xt::random::rand({n_batches, n_channels_in}, -1.0, 1.0);
        auto features_dltensor = xarray_to_dltensor(features);

        auto result_qant_dltensor = q_native::linear_fprop_preloaded_weights(&features_dltensor, weights_handle);
        REQUIRE(result_qant_dltensor != nullptr);
        auto result_qant = dltensor_to_tensor_view(result_qant_dltensor);

        auto result_shouldbe = xt::linalg::tensordot(features, weights, {1}, {1});
        REQUIRE(result_qant.shape() == result_shouldbe.shape());
        REQUIRE(xt::allclose(result_qant, result_shouldbe, 0.0, 0.025));

        features_dltensor.deleter(&features_dltensor);
        result_qant_dltensor->deleter(result_qant_dltensor);
    }

    REQUIRE(q_generic::npu_free(weights_handle) == 0);
    REQUIRE(q_generic::release_npu(0) == 0);
    weights_dltensor.deleter(&weights_dltensor);
}

TEST_CASE("conv_fprop")
{
    Tensor features = xt::random::rand({1, 12, 3, 4}, -1.0, 1.0);
    Tensor weights = xt::random::rand({3, 12, 5, 6}, -1.0, 1.0);

    auto features_dltensor = xarray_to_dltensor(features);
    auto weights_dltensor = xarray_to_dltensor(weights);

    auto qant_res_dltensor = q_ai::conv_fprop(0, &features_dltensor, &weights_dltensor, 2, 3, 1);
    REQUIRE(q_generic::release_npu(0) == 0);
    REQUIRE(qant_res_dltensor != nullptr);

    features_dltensor.deleter(&features_dltensor);
    weights_dltensor.deleter(&weights_dltensor);
    qant_res_dltensor->deleter(qant_res_dltensor);
}

TEST_CASE("conv_transpose_fprop")
{
    Tensor features = xt::random::rand({1, 12, 3, 4}, -1.0, 1.0);
    Tensor weights = xt::random::rand({12, 3, 5, 6}, -1.0, 1.0);

    auto features_dltensor = xarray_to_dltensor(features);
    auto weights_dltensor = xarray_to_dltensor(weights);

    auto qant_res_dltensor = q_ai::conv_transpose_fprop(0, &features_dltensor, &weights_dltensor, 2, 3, 1, 0);
    REQUIRE(q_generic::release_npu(0) == 0);
    REQUIRE(qant_res_dltensor != nullptr);

    features_dltensor.deleter(&features_dltensor);
    weights_dltensor.deleter(&weights_dltensor);
    qant_res_dltensor->deleter(qant_res_dltensor);
}

TEST_CASE("conv_fprop_preloaded_weights")
{
    Tensor features = xt::random::rand({1, 12, 3, 4}, -1.0, 1.0);
    Tensor weights = xt::random::rand({3, 12, 5, 6}, -1.0, 1.0);

    auto features_dltensor = xarray_to_dltensor(features);
    auto weights_dltensor = xarray_to_dltensor(weights);

    auto weights_handle = q_ai::upload_conv_weights(0, &weights_dltensor);
    REQUIRE(weights_handle != nullptr);

    auto qant_res_dltensor = q_ai::conv_fprop_preloaded_weights(&features_dltensor, weights_handle, 2, 3, 1);
    REQUIRE(qant_res_dltensor != nullptr);

    REQUIRE(q_ai::conv_weights_free(weights_handle) == 0);
    REQUIRE(q_generic::release_npu(0) == 0);

    features_dltensor.deleter(&features_dltensor);
    weights_dltensor.deleter(&weights_dltensor);
    qant_res_dltensor->deleter(qant_res_dltensor);
}

TEST_CASE("conv_transpose_fprop_preloaded_weights")
{
    Tensor features = xt::random::rand({1, 12, 3, 4}, -1.0, 1.0);
    Tensor weights = xt::random::rand({12, 3, 5, 6}, -1.0, 1.0);

    auto features_dltensor = xarray_to_dltensor(features);
    auto weights_dltensor = xarray_to_dltensor(weights);

    auto weights_handle = q_ai::upload_conv_transpose_weights(0, &weights_dltensor);
    REQUIRE(weights_handle != nullptr);

    auto qant_res_dltensor = q_ai::conv_transpose_fprop_preloaded_weights(&features_dltensor, weights_handle, 2, 3, 1, 0);
    REQUIRE(qant_res_dltensor != nullptr);

    REQUIRE(q_ai::conv_transpose_weights_free(weights_handle) == 0);
    REQUIRE(q_generic::release_npu(0) == 0);

    features_dltensor.deleter(&features_dltensor);
    weights_dltensor.deleter(&weights_dltensor);
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

// Relative error ||res - ref|| / ||ref|| of res against a naive O(N^2) DFT
// X[k] = sum_n x[n] * exp(-2*pi*i*k*n/N) of data.
// data has shape (n_dfts, dft_size, 2), res holds n_dfts * dft_size interleaved
// (real, imaginary) values.
double dft_relative_error(Tensor const &data, std::bfloat16_t const *res)
{
    size_t const n_dfts = data.shape(0);
    size_t const n = data.shape(1);
    double err_sq = 0.0;
    double ref_sq = 0.0;
    for (size_t b = 0; b < n_dfts; ++b)
    {
        for (size_t k = 0; k < n; ++k)
        {
            std::complex<double> ref = 0.0;
            for (size_t j = 0; j < n; ++j)
            {
                std::complex<double> const x(data(b, j, 0), data(b, j, 1));
                ref += x * std::polar(1.0, -2.0 * std::numbers::pi * double(k * j % n) / double(n));
            }
            size_t const idx = 2 * (b * n + k);
            std::complex<double> const npu(res[idx], res[idx + 1]);
            err_sq += std::norm(npu - ref);
            ref_sq += std::norm(ref);
        }
    }
    return std::sqrt(err_sq / ref_sq);
}

void check_dft(int32_t const npu_id)
{
    // (n_dfts, dft_size, 2) with interleaved real and imaginary parts,
    // passed as a (n_dfts, dft_size) bcomplex32 tensor
    Tensor const data = xt::random::rand({2, 1024, 2}, -1.0, 1.0);

    auto data_dltensor = xarray_to_bcomplex_dltensor(data);

    auto qant_res_dltensor = q_native::dft(npu_id, &data_dltensor);
    REQUIRE(qant_res_dltensor != nullptr);

    DLTensor const &res = qant_res_dltensor->dl_tensor;
    REQUIRE(res.ndim == 2);
    REQUIRE(res.shape[0] == 2);
    REQUIRE(res.shape[1] == 1024);
    REQUIRE(res.dtype.code == kDLBcomplex);
    REQUIRE(res.dtype.bits == 32);
    REQUIRE(res.dtype.lanes == 1);

    double const rel_err = dft_relative_error(data, static_cast<std::bfloat16_t const *>(res.data));
    std::cout << "dft relative error: " << rel_err << std::endl;
    REQUIRE(rel_err < 0.05);

    data_dltensor.deleter(&data_dltensor);
    qant_res_dltensor->deleter(qant_res_dltensor);
}

TEST_CASE("dft")
{
    auto const npu_id = dft_npu_id();
    check_dft(npu_id);
    REQUIRE(q_generic::release_npu(npu_id) == 0);
}

TEST_CASE("dft auto npu")
{
    // DFT is only supported by Gen2p NPUs, so the default sentinel must
    // resolve to a Gen2p NPU rather than e.g. the first enumerated NPU.
    check_dft(QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID);
}

TEST_CASE("dft rejects bfloat16 input")
{
    // the former (n_dfts, dft_size, 2) bfloat16 interface is no longer accepted
    Tensor const data = xt::random::rand({2, 16, 2}, -1.0, 1.0);

    auto data_dltensor = xarray_to_dltensor(data);

    auto qant_res_dltensor = q_native::dft(QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID, &data_dltensor);
    REQUIRE(qant_res_dltensor == nullptr);

    data_dltensor.deleter(&data_dltensor);
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
    q_info::QantSensorInfo result;
    REQUIRE(q_info::get_sensor_info(0, &result) == 0);
    REQUIRE(q_generic::release_npu(0) == 0);

    std::cout << result.voltage_3v3 << std::endl;
    REQUIRE(result.voltage_3v3 > 3.);
    REQUIRE(result.current_12v >= 0.0);
}

TEST_CASE("version_info")
{
    q_info::QantVersionInfo result;
    REQUIRE(q_info::get_version_info(0, &result) == 0);
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

    int const err = q_generic::get_available_npus(nullptr, 0, idxs, serials, n_npus_max, &n_npus);
    REQUIRE(err == 0);
    REQUIRE(n_npus > 0);
    for (size_t i = 1; i < n_npus; ++i)
    {
        REQUIRE(idxs[i - 1] < idxs[i]);
    }

    // every listed NPU supports all requested operations
    q_generic::QantDeviceCapability const ops[] = {q_generic::Mul, q_generic::MemHandling};
    uint32_t filtered_idxs[n_npus_max];
    size_t n_filtered = 0;
    REQUIRE(q_generic::get_available_npus(ops, 2, filtered_idxs, serials, n_npus_max, &n_filtered) == 0);
    REQUIRE(n_filtered <= n_npus);
    for (size_t i = 0; i < n_filtered; ++i)
    {
        q_generic::QantDeviceCapability capabilities[4];
        size_t n_capabilities = 0;
        REQUIRE(q_info::get_capabilities(static_cast<int32_t>(filtered_idxs[i]), capabilities, 4, &n_capabilities) == 0);
        auto const supports = [&](q_generic::QantDeviceCapability op)
        { return std::find(capabilities, capabilities + n_capabilities, op) != capabilities + n_capabilities; };
        REQUIRE(supports(q_generic::Mul));
        REQUIRE(supports(q_generic::MemHandling));
    }

    // capacity too small: fails, but reports the required count
    size_t required = 0;
    REQUIRE(q_generic::get_available_npus(nullptr, 0, idxs, serials, 0, &required) != 0);
    REQUIRE(required == n_npus);
}

TEST_CASE("capabilities")
{
    q_generic::QantDeviceCapability capabilities[4];
    size_t n_capabilities = 0;
    REQUIRE(q_info::get_capabilities(0, capabilities, 4, &n_capabilities) == 0);
    REQUIRE(n_capabilities <= 4);
    REQUIRE(q_info::get_capabilities(-2, capabilities, 4, &n_capabilities) != 0);

    // capacity too small: fails, but reports the required count
    size_t required = 0;
    if (n_capabilities > 0)
    {
        REQUIRE(q_info::get_capabilities(0, capabilities, 0, &required) != 0);
        REQUIRE(required == n_capabilities);
    }
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
