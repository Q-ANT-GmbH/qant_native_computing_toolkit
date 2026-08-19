#ifndef DLPACK_UTILS_H
#define DLPACK_UTILS_H

#include <dlpack/dlpack.h>
#include <stdfloat>
#include <vector>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>

void delete_dltensor(struct DLManagedTensorVersioned *self)
{
    delete[] self->dl_tensor.shape;
    // the rest should be cleaned up when the corresponding data managers
    // go out of scope
}

template <typename T>
constexpr uint8_t dlpack_code()
{
    if constexpr (std::same_as<T, int16_t>)
        return kDLInt;
    else if constexpr (std::same_as<T, float>)
        return kDLFloat;
    else
        static_assert([]
                      { return false; }(),
                      "Type not implemented.");
}

template <typename T>
constexpr uint8_t dlpack_bits()
{
    if constexpr (std::same_as<T, int16_t>)
        return 16;
    else if constexpr (std::same_as<T, float>)
        return 32;
    else
        static_assert([]
                      { return false; }(),
                      "Type not implemented.");
}

template <typename T>
    requires(
        std::same_as<T, int16_t> ||
        std::same_as<T, float>)
DLManagedTensorVersioned
vector_to_dltensor(const std::vector<T> &vec)
{
    // creates a managed DL tensor as a view on the data owned by vec

    int64_t const len = vec.size();

    DLTensor tensor = DLTensor();
    tensor.ndim = 1;
    tensor.shape = new int64_t[tensor.ndim];
    tensor.shape[0] = len;
    tensor.strides = NULL;

    tensor.dtype.code = dlpack_code<T>();
    tensor.dtype.bits = dlpack_bits<T>();
    tensor.dtype.lanes = 1;

    tensor.data = const_cast<void *>(static_cast<const void *>(vec.data()));
    tensor.byte_offset = 0;
    tensor.device.device_type = kDLCPU;
    tensor.device.device_id = 0;

    DLManagedTensorVersioned dltensor = DLManagedTensorVersioned();
    dltensor.dl_tensor = tensor;
    dltensor.version = DLPackVersion{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    dltensor.manager_ctx = const_cast<void *>(static_cast<const void *>(&vec));
    dltensor.deleter = delete_dltensor;

    return dltensor;
}

template <typename T>
    requires(
        std::same_as<T, int16_t> ||
        std::same_as<T, float>)
std::vector<T> tensor_to_vector_view(DLManagedTensorVersioned const *tensor)
{
    T *data = static_cast<T *>(tensor->dl_tensor.data);
    std::vector<T> vec(data, data + tensor->dl_tensor.shape[0]);
    return vec;
}

DLManagedTensorVersioned
xarray_to_dltensor(xt::xarray<std::bfloat16_t> const &xarr)
{
    // creates a managed DL tensor as a view on the data owned by the xtensor

    DLTensor tensor = DLTensor();

    int32_t ndim = xarr.dimension();
    auto shape = xarr.shape();
    tensor.ndim = ndim;
    tensor.shape = new int64_t[ndim];
    for (int i = 0; i < ndim; ++i)
    {
        tensor.shape[i] = shape[i];
    };
    tensor.strides = NULL;

    tensor.dtype.code = kDLBfloat;
    tensor.dtype.bits = 16;
    tensor.dtype.lanes = 1;

    tensor.data = const_cast<void *>(static_cast<const void *>(xarr.data()));
    tensor.byte_offset = 0;
    tensor.device.device_type = kDLCPU;
    tensor.device.device_id = 0;

    DLManagedTensorVersioned dltensor = DLManagedTensorVersioned();
    dltensor.dl_tensor = tensor;
    dltensor.version = DLPackVersion{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    dltensor.manager_ctx = const_cast<void *>(static_cast<const void *>(&xarr));
    dltensor.deleter = delete_dltensor;

    return dltensor;
}

xt::xarray<std::bfloat16_t>
dltensor_to_tensor_view(DLManagedTensorVersioned const *dltensor)
{
    // create a view on the dltensor managed data
    DLTensor tensor = dltensor->dl_tensor;

    std::vector<int64_t> shape(tensor.shape, tensor.shape + tensor.ndim);
    int64_t len_data =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());

    std::bfloat16_t *data_ptr = static_cast<std::bfloat16_t *>(tensor.data);

    return xt::adapt(data_ptr, len_data, xt::no_ownership(), shape);
}

#endif