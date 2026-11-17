#ifndef DLPACK_UTILS_H
#define DLPACK_UTILS_H

#include <dlpack/dlpack.h>
#include <stdexcept>
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

DLManagedTensorVersioned
xarray_to_dltensor(xt::xarray<std::bfloat16_t> const &xarr, std::vector<int64_t> const &shape,
                   DLDataType dtype)
{
    // creates a managed DL tensor as a view on the data owned by the xtensor,
    // with the given shape and dtype

    DLTensor tensor = DLTensor();

    int32_t ndim = shape.size();
    tensor.ndim = ndim;
    tensor.shape = new int64_t[ndim];
    for (int i = 0; i < ndim; ++i)
    {
        tensor.shape[i] = shape[i];
    };
    tensor.strides = NULL;

    tensor.dtype = dtype;

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

DLManagedTensorVersioned
xarray_to_dltensor(xt::xarray<std::bfloat16_t> const &xarr)
{
    // bfloat16 tensor with the same shape as the xtensor
    std::vector<int64_t> shape(xarr.shape().begin(), xarr.shape().end());
    return xarray_to_dltensor(xarr, shape, DLDataType{kDLBfloat, 16, 1});
}

DLManagedTensorVersioned
xarray_to_bcomplex_dltensor(xt::xarray<std::bfloat16_t> const &xarr)
{
    // the xtensor of shape (..., 2) holds interleaved real and imaginary parts;
    // the DL tensor views it as bcomplex32 (2x bfloat16) with shape (...)
    auto const &xshape = xarr.shape();
    if (xshape.empty() || xshape.back() != 2)
    {
        throw std::invalid_argument("last dimension must have size 2 (real, imaginary)");
    }
    std::vector<int64_t> shape(xshape.begin(), xshape.end() - 1);
    return xarray_to_dltensor(xarr, shape, DLDataType{kDLBcomplex, 32, 1});
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