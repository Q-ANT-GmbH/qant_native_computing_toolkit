/**
 * This file is a standalone executable, to be used for the memory leak check
 * with valgrind.
 */

#include <xtensor/xrandom.hpp>
#include <xtensor/xtensor.hpp>

#include "dlpack_utils.h"
#include "qant_native_computing_toolkit.h"

typedef xt::xarray<std::bfloat16_t> Tensor;
namespace q_generic = ::qant_native_computing_toolkit::generic;
namespace q_ai = ::qant_native_computing_toolkit::ai;

int main()
{
    // Batchnorm takes the most tensors so we use it for memory safety testing

    auto n_chnls = 1000;
    std::float32_t eps = 0.01f;
    Tensor features = xt::cast<std::bfloat16_t>(xt::random::rand<float>({1, n_chnls, 3, 5}, -1.0, 1.0));
    Tensor means = xt::cast<std::bfloat16_t>(xt::random::rand<float>({n_chnls}, -1.0, 1.0));
    Tensor variances = xt::cast<std::bfloat16_t>(xt::random::rand<float>({n_chnls}, eps, 1.0));
    Tensor weights = xt::cast<std::bfloat16_t>(xt::random::rand<float>({n_chnls}, -1.0, 1.0));
    Tensor bias = xt::cast<std::bfloat16_t>(xt::random::rand<float>({n_chnls}, -1.0, 1.0));

    auto features_dltensor = xarray_to_dltensor(features);
    auto means_dltensor = xarray_to_dltensor(means);
    auto variances_dltensor = xarray_to_dltensor(variances);
    auto weights_dltensor = xarray_to_dltensor(weights);
    auto bias_dltensor = xarray_to_dltensor(bias);

    auto qant_res_dltensor = q_ai::batchnorm2d_fprop(0,
                                                     &features_dltensor,
                                                     &means_dltensor,
                                                     &variances_dltensor,
                                                     &weights_dltensor,
                                                     &bias_dltensor,
                                                     eps);
    q_generic::release_npu(0);

    // We could implement this as the destructor of DLTensorManaged,
    // but we want to keep the struct as it comes from the official header.
    features_dltensor.deleter(&features_dltensor);
    means_dltensor.deleter(&means_dltensor);
    variances_dltensor.deleter(&variances_dltensor);
    weights_dltensor.deleter(&weights_dltensor);
    bias_dltensor.deleter(&bias_dltensor);

    // This calls the deleter of the tensor that was created by the rust library
    qant_res_dltensor->deleter(qant_res_dltensor);

    return 0;
}
