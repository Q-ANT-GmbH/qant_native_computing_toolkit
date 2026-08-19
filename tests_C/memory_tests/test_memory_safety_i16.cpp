/**
 * This file is a standalone executable, to be used for the memory leak check
 * with valgrind. This tests the memory safety of the deprecated i16 function,
 * which is the only function that uses a buffer internally due to type conversions.
 */

#include <xtensor/xrandom.hpp>
#include <xtensor/xtensor.hpp>

#include "dlpack_utils.h"
#include "qant_native_computing_toolkit.h"

typedef xt::xarray<int16_t> Tensor;
namespace q_generic = ::qant_native_computing_toolkit::generic;
namespace q_native = ::qant_native_computing_toolkit::native;

int main()
{
    auto npu_id = 0;
    std::vector<int16_t> us = {500, 200, 300, 400};
    std::vector<int16_t> vs = {500, 300, 200, 100};

    auto us_tensor = vector_to_dltensor<std::int16_t>(us);
    auto vs_tensor = vector_to_dltensor<std::int16_t>(vs);

    auto res_qant_dltensor = q_native::mul_npu_i16(npu_id, &us_tensor, &vs_tensor);
    q_generic::release_npu(npu_id);

    us_tensor.deleter(&us_tensor);
    vs_tensor.deleter(&vs_tensor);
    res_qant_dltensor->deleter(res_qant_dltensor);
}
