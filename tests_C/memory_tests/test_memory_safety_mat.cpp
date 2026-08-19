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
namespace q_native = ::qant_native_computing_toolkit::native;

int main()
{
    auto npu_id = 0;
    auto batches = 2;
    auto in_channels = 8;
    auto out_channels = 4;
    Tensor features = xt::cast<std::bfloat16_t>(xt::random::rand<float>({batches, in_channels}, -1.0, 1.0));
    Tensor weights = xt::cast<std::bfloat16_t>(xt::random::rand<float>({out_channels, in_channels}, -1.0, 1.0));

    auto features_dltensor = xarray_to_dltensor(features);
    auto weights_dltensor = xarray_to_dltensor(weights);

    auto qant_res_dltensor = q_native::linear_fprop(npu_id, &features_dltensor, &weights_dltensor);
    q_generic::release_npu(npu_id);

    // We could implement this as the destructor of DLTensorManaged,
    // but we want to keep the struct as it comes from the official header.
    features_dltensor.deleter(&features_dltensor);
    weights_dltensor.deleter(&weights_dltensor);

    // This calls the deleter of the tensor that was created by the rust library
    qant_res_dltensor->deleter(qant_res_dltensor);

    return 0;
}
