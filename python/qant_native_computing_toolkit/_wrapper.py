from __future__ import annotations

import pathlib
import warnings

import numpy as np
from ml_dtypes import bfloat16

import qant_native_computing_toolkit._backend as backend

from ._utils import (
    assert_nonnegative_int,
    assert_np_type_dim,
    assert_same_shape,
    repeat_or_noop,
)


def init_npu(id: int) -> None:
    """
    Initialises the Native Processing Unit. Optional, not necessary for computation.

    Args:
        id: The identifier of the NPU that is to be initialised.
    """
    assert_nonnegative_int(id)
    backend.init_npu(id)


def release_npu(id: int) -> None:
    """
    Releases an NPU and frees associated memory.
    Only relevant if multiple users want to access the same device.

    Args:
        id: The identifier of the NPU that is to be released.
    """
    assert_nonnegative_int(id)
    backend.release_npu(id)


def mul_f32(us: np.ndarray, vs: np.ndarray, device_id: int = 0) -> np.ndarray:
    """
    Multiply two one-dimensional arrays element-wise.

    Args:
        us (np.ndarray): First input array, must be of type `np.float32`.
        vs (np.ndarray): Second input array, must be of type `np.float32`.
        device_id (int, optional): The identifier of the NPU on which the operation is performed.

    Returns:
        np.ndarray: A new array of the same shape as `us` and `vs`, containing the
        element-wise product of the two input arrays, with type `np.float32`.
    """

    for input, input_name in zip([us, vs], ["us", "vs"]):
        assert_np_type_dim(input, np.float32, 1, name=input_name)
    assert_same_shape(us, vs, name_x="us", name_y="vs")
    assert_nonnegative_int(device_id, name="device_id")

    return backend.mul_f32(device_id, us, vs)


def mul_i16(us: np.ndarray, vs: np.ndarray, device_id: int = 0) -> np.ndarray:
    """
    Multiply two arrays element-wise.

    Args:
        us (np.ndarray): First input array, must be of type `np.int16` and have same shape as vs.
        vs (np.ndarray): Second input array, must be of type `np.int16` and have same shape as us.
        device_id (int, optional): The identifier of the NPU on which the operation is performed.

    Returns:
        np.ndarray: A new array of the same shape as `us` and `vs`, containing the
        element-wise product of the two input arrays, with type `np.int16`.
    """
    warnings.warn(
        "`mul_i16` is deprecated, use `mul_elementwise` instead",
        DeprecationWarning,
        stacklevel=2,
    )
    for input, input_name in zip([us, vs], ["us", "vs"]):
        assert_np_type_dim(input, np.int16, None, name=input_name)
    assert_same_shape(us, vs, name_x="us", name_y="vs")
    assert_nonnegative_int(device_id, name="device_id")

    us_bf16, vs_bf16 = (us / 1_000).astype(bfloat16), (vs / 1_000).astype(bfloat16)

    shape = us.shape

    return (
        backend.mul(device_id, us_bf16.reshape(-1), vs_bf16.reshape(-1)).reshape(shape)
        * 1_000
    ).astype(np.int16)


def mul_elementwise(us: np.ndarray, vs: np.ndarray, device_id: int = 0) -> np.ndarray:
    """
    Multiply two arrays element-wise.

    Args:
        us (np.ndarray): First input array, must be of type `bfloat16` and have same shape as vs.
        vs (np.ndarray): Second input array, must be of type `bfloat16` and have same shape as us.
        device_id (int, optional): The identifier of the NPU on which the operation is performed.

    Returns:
        np.ndarray: A new array of the same shape as `us` and `vs`, containing the
        element-wise product of the two input arrays, with type `bfloat16`.
    """
    for input, input_name in zip([us, vs], ["us", "vs"]):
        assert_np_type_dim(input, bfloat16, None, name=input_name)
    assert_same_shape(us, vs, name_x="us", name_y="vs")
    assert_nonnegative_int(device_id, name="device_id")

    shape = us.shape

    return backend.mul(device_id, us.reshape(-1), vs.reshape(-1)).reshape(shape)


def linear_fprop(
    features: np.ndarray, weights: np.ndarray, device_id: int = 0
) -> np.ndarray:
    """
    Performs a linear forward propagation between input features and a weight matrix.
    Operation is performed along the last dimension of features. Additional dimensions are treated as batch dimensions.

    The operation computes ``output = features @ weights.T``, i.e.:

    .. math::

       \\text{output}[i, j] = \\sum_k \\text{features}[i, k] \\cdot \\text{weights}[j, k]

    Note that this is **not** a plain ``features @ weights`` — the weight matrix is
    implicitly transposed, matching the convention where ``weights`` has shape
    ``(n_channels_out, n_channels_in)``.

    Args:
        features (np.ndarray): An ND array of input features with shape (<batch dimensions>, n_channels_in). Must be dtype=bfloat16.
        weights (np.ndarray): A 2D array representing the weights with shape (n_channels_out, n_channels_in). Must be dtype=bfloat16.
        device_id (int, optional): The identifier of the NPU on which the operation is performed.

    Returns:
        np.ndarray: An ND array containing the result of the forward propagation with shape (<batch dimensions>, n_channels_out).
    """
    original_shape = features.shape
    features = features.reshape(-1, original_shape[-1])
    assert_np_type_dim(features, bfloat16, 2, name="features")
    assert_np_type_dim(weights, bfloat16, 2, name="weights")
    assert_nonnegative_int(device_id, name="device_id")

    if not features.shape[-1] == weights.shape[1]:
        raise ValueError(
            f"number of features ({features.shape[-1]}) != number of weights input channels ({weights.shape[1]})"
        )

    return backend.linear_fprop(device_id, features, weights).reshape(
        *original_shape[:-1], weights.shape[0]
    )


def conv_fprop(
    features: np.ndarray,
    kernels: np.ndarray,
    padding: int,
    stride: int,
    dilation: int,
    device_id: int = 0,
) -> np.ndarray:
    """
    Performs a forward pass through a convolution layer.

    Args:
        features (np.ndarray): A 3D array of input features with shape (n_channels_in, height, width) or a 4D array of input features with the shape (batches, n_channels_in, height, width). Must be dtype=bfloat16.
        kernels (np.ndarray): A 4D array representing the filter with shape (n_channels_out, n_channels_in, height, width). Must be dtype=bfloat16.
        padding (int): Amount of padding added to the input features.
        stride (int): Step size for moving the filter window over the input features.
        dilation (int): Dilation ("zoom out") of the filter window.
        device_id (int, optional): The identifier of the NPU on which the operation is performed.

    Returns:
        np.ndarray: An ND array containing the result of the convolution with shape ([batches], n_channels_out, height, width).
    """
    assert_np_type_dim(features, bfloat16, [3, 4], name="features")
    assert_np_type_dim(kernels, bfloat16, 4, name="kernels")

    if not features.shape[-3] == kernels.shape[1]:
        raise ValueError(
            f"number of features ({features.shape[0]}) != number of kernels input channels ({kernels.shape[1]})"
        )

    assert_nonnegative_int(padding, name="padding")
    assert_nonnegative_int(stride, name="stride")
    assert_nonnegative_int(dilation, name="dilation")
    assert_nonnegative_int(device_id, name="device_id")

    features_batched = features[np.newaxis, ...] if features.ndim == 3 else features
    result = backend.conv_fprop(
        device_id, features_batched, kernels, padding, stride, dilation
    )
    return result[0] if features.ndim == 3 else result


def conv_transpose_fprop(
    features: np.ndarray,
    kernels: np.ndarray,
    padding: int,
    stride: int,
    dilation: int,
    output_padding: int,
    device_id: int = 0,
) -> np.ndarray:
    """
    Performs a forward pass through a transposed convolution layer.

    Args:
        features (np.ndarray): A 3D array of input features with shape (n_channels_in, height, width) or a 4D array of input features with the shape (batches, n_channels_in, height, width). Must be dtype=bfloat16.
        kernels (np.ndarray): A 4D array representing the filter with shape (n_channels_in, n_channels_out, height, width). Must be dtype=bfloat16.
        padding (int): Amount of padding added to the input features.
        stride (int): Step size for moving the filter window over the input features.
        dilation (int): Dilation ("zoom out") of the filter window.
        output_padding (int): Padding for the returned features.
        device_id (int, optional): The identifier of the NPU on which the operation is performed.

    Returns:
        np.ndarray: An ND array containing the result of the transpose convolution with shape ([batches], n_channels_out, height, width).
    """
    assert_np_type_dim(features, bfloat16, [3, 4], name="features")
    assert_np_type_dim(kernels, bfloat16, 4, name="kernels")

    if not features.shape[-3] == kernels.shape[0]:
        raise ValueError(
            f"number of features ({features.shape[-3]}) != number of kernels input channels ({kernels.shape[0]})"
        )

    assert_nonnegative_int(padding, name="padding")
    assert_nonnegative_int(stride, name="stride")
    assert_nonnegative_int(dilation, name="dilation")
    assert_nonnegative_int(output_padding, name="output_padding")
    assert_nonnegative_int(device_id, name="device_id")

    features_batched = features[np.newaxis, ...] if features.ndim == 3 else features
    result = backend.conv_transpose_fprop(
        device_id, features_batched, kernels, padding, stride, dilation, output_padding
    )
    return result[0] if features.ndim == 3 else result


def add_bias_fprop(
    features: np.ndarray, bias: np.ndarray, device_id: int = 0
) -> np.ndarray:
    """
    Performs a forward pass through a bias adder. Aka element-wise addition.
    If 3D features are used, this function assumes the layout (n_channels, height, width)

    Args:
        features (np.ndarray): An array of 1 to 4 dimensions of input features with shape ([batches], n_channels, [height, width]). Must be dtype=bfloat16.
        bias (np.ndarray): A 1D array of bias values with shape (n_channels). Must be dtype=bfloat16.
        device_id (int, optional): The identifier of the NPU on which the operation is performed.

    Returns:
        np.ndarray: A 1-4D array of input features with shape ([batches], n_channels, [height, width]) containing the result of the addition.
    """
    assert_np_type_dim(features, bfloat16, [1, 2, 3, 4], name="features")
    assert_np_type_dim(bias, bfloat16, 1, name="bias")
    assert_nonnegative_int(device_id, name="device_id")

    if features.ndim == 1:
        features = features[np.newaxis, ...]
        return backend.add_bias_fprop(device_id, features, bias)[0]
    elif features.ndim == 2:
        return backend.add_bias_fprop(device_id, features, bias)
    elif features.ndim == 3:
        features = features[np.newaxis, ...]
        return backend.add_bias_for_conv2d_fprop(device_id, features, bias)[0]
    elif features.ndim == 4:
        return backend.add_bias_for_conv2d_fprop(device_id, features, bias)
    else:
        raise ValueError(
            f"features must have one of the dimensions in [1, 2, 3, 4], but has dimension {features.ndim}"
        )


def calc_scaled_periodic_nl_fprop(
    features: np.ndarray, weights: np.ndarray, device_id: int = 0
) -> np.ndarray:
    """
    Calculates a scaled periodic nonlinearity pairwise for all elements of features u and weights v.
    The output for each input pair is w(u,v) = f(u) * v, where f() is a 2pi - periodic function
    with values between -1 and 1, similar to a cosine.

    Args:
        features (np.ndarray): A 1D array of input features. Must be dtype=bfloat16.
        weights (np.ndarray): A 1D array of weights. Must be dtype=bfloat16.
        device_id (int, optional): The identifier of the NPU on which the operation is performed.

    Returns:
        np.ndarray: A 1D array containing the result, same shape as input features.
    """
    assert_np_type_dim(features, bfloat16, 1, name="features")
    assert_np_type_dim(weights, bfloat16, 1, name="weights")
    assert_same_shape(features, weights, name_x="features", name_y="weights")
    assert_nonnegative_int(device_id, name="device_id")

    return backend.calc_scaled_periodic_nl_fprop(device_id, features, weights)


def calc_kan_layer_fprop(
    features: np.ndarray,
    phis: np.ndarray,
    ampls: np.ndarray,
    ks: np.ndarray,
    device_id: int = 0,
) -> np.ndarray:
    """
    Calculates a Q.ANT version of a KAN layer (https://arxiv.org/abs/2404.19756) based on scaled_periodic_nl.
    The mathematical function is y_j = sum_{i, l} tcos(ks_l * x_i + phis_{jil}) * ampls_{jil},
    where x_i is the input (vector), ks the frequency components,
    phis the phase offsets (tensor) and ampls the amplitude (tensor) and y_j the output (vector).
    tcos() denotes the cosine-related shape of the periodic optical nonlinearity.

    Args:
        features (np.ndarray): An ND input array with shape (<batch dimensions>, n_channels_in), dtype=bfloat16.
                            The operation is performed along the last axis, additional dimensions
                            are treated as batch dimensions.
        phis (np.ndarray): A 3D array of phase offsets, shape (n_channels_out, n_channels_in, len(ks)), dtype=bfloat16.
        ampls (np.ndarray): A 3D array of amplitudes, same shape as phis, dtype=bfloat16.
        ks (np.ndarray): A 1D array of frequency components, dtype=bfloat16.
        device_id (int, optional): The identifier of the NPU on which the operation is performed.

    Returns:
        np.ndarray: The result of the KAN layer with shape (<batch dimensions>, n_channels_out), dtype=bfloat16.
    """
    # if there are multiple batch dimensions, flatten them into one
    shape_features = np.array(np.shape(features))
    features = features.reshape([-1, shape_features[-1]])
    assert_np_type_dim(features, bfloat16, 2, name="features")
    assert_np_type_dim(phis, bfloat16, 3, name="phis")
    assert_np_type_dim(ampls, bfloat16, 3, name="ampls")
    assert_np_type_dim(ks, bfloat16, 1, name="ks")
    assert_nonnegative_int(device_id, name="device_id")
    assert_same_shape(phis, ampls, name_x="phis", name_y="ampls")
    # recreate the batch dimensions
    shape_res = shape_features
    shape_res[-1] = phis.shape[0]  # = n_channels_out
    return backend.calc_kan_layer_fprop(device_id, features, phis, ampls, ks).reshape(
        shape_res
    )


def relu_fprop(features: np.ndarray, device_id: int = 0) -> np.ndarray:
    """
    Performs a forward pass through a `ReLU <https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`_ layer.

    Args:
        features (np.ndarray): An ND array of input features. Must be dtype=bfloat16.
        device_id (int, optional): The identifier of the NPU on which the operation is performed.

    Returns:
        np.ndarray: An ND array containing the result of the ReLU operation, same shape as input features.
    """

    assert_np_type_dim(features, bfloat16, None, name="features")
    assert_nonnegative_int(device_id, name="device_id")
    shape = features.shape
    return backend.relu_fprop(device_id, features.flatten()).reshape(shape)


def sigmoid_fprop(features: np.ndarray, device_id: int = 0) -> np.ndarray:
    """
    Performs a forward pass through a `Sigmoid <https://en.wikipedia.org/wiki/Sigmoid_function>`_ layer.

    Args:
        features (np.ndarray): An ND array of input features. Must be dtype=bfloat16.
        device_id (int, optional): The identifier of the NPU on which the operation is performed.

    Returns:
        np.ndarray: An ND array containing the result of the Sigmoid operation with the same shape as the input.
    """
    assert_np_type_dim(features, bfloat16, None, name="features")
    assert_nonnegative_int(device_id, name="device_id")
    shape = features.shape
    return backend.sigmoid_fprop(device_id, features.flatten()).reshape(shape)


def softmax_fprop(features: np.ndarray, device_id: int = 0) -> np.ndarray:
    """
    Performs a forward pass through a `Softmax <https://en.wikipedia.org/wiki/Softmax_function>`_ layer.

    Args:
        features (np.ndarray): An ND array of input features. Must be dtype=bfloat16. If multidimensional, the first dimension must be 1.
        device_id (int, optional): The identifier of the NPU on which the operation is performed.

    Returns:
        np.ndarray: An ND array, containing the result of the Softmax operation with the same shape as the input.
    """
    original_shape = features.shape
    features = features.reshape(-1, original_shape[-1])
    assert_np_type_dim(features, bfloat16, 2, name="features")
    assert_nonnegative_int(device_id, name="device_id")
    res = backend.softmax_fprop(device_id, features).reshape(original_shape)
    return res


def batchnorm2d_fprop(
    features: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    eps: int,
    device_id: int = 0,
) -> np.ndarray:
    """
    Performs a forward pass through a `batchnorm2d <https://en.wikipedia.org/wiki/Normalization_(machine_learning)#Batch_normalization>`_ layer

    Args:
        features (np.ndarray): A 3D array of input features with shape (n_channels, height, width) or a 4D array of input features with the shape (batches, n_channels, height, width). Must be dtype=bfloat16.
        means (np.ndarray): A 1D array containing the mean values for each input feature.
        variances (np.ndarray): A 1D array containing the variance values for each input feature.
        weights (np.ndarray): A 1D array containing the scale parameters for each input feature.
        bias (np.ndarray): A 1D array containing the bias parameters for each input feature.
        eps (float): A small value added to the variance for numerical stability.
        device_id (int, optional): The identifier of the NPU on which the operation is performed.


    Returns:
        np.ndarray: An ND array containing the result of the batchnorm operation with shape ([batches], n_channels, height, width).
    """
    assert_np_type_dim(features, bfloat16, [3, 4], name="features")
    assert_np_type_dim(means, bfloat16, 1, name="means")
    assert_np_type_dim(variances, bfloat16, 1, name="variances")
    assert_np_type_dim(weights, bfloat16, 1, name="weights")
    assert_np_type_dim(bias, bfloat16, 1, name="bias")
    assert_nonnegative_int(device_id, name="device_id")

    features_batched = features[np.newaxis, ...] if features.ndim == 3 else features
    result = backend.batchnorm2d_fprop(
        device_id, features_batched, means, variances, weights, bias, eps
    )
    return result[0] if features.ndim == 3 else result


def maxpool2d_fprop(
    features: np.ndarray,
    kernel_size: int | tuple[int, int],
    stride: int,
    padding: int,
    device_id: int = 0,
) -> np.ndarray:
    """
    Performs a forward pass through a `maxpooling2d <https://en.wikipedia.org/wiki/Pooling_layer>`_ layer

    Args:
        features (np.ndarray): A 3D array of input features with shape (n_channels, height, width) or a 4D array of input features with the shape (batches, n_channels, height, width). Must be dtype=bfloat16.
        kernel_size (int | (int, int)): Pooling window dimensions. If int, same size is used for both dimensions.
        stride (int): Step size for sliding pooling window (must be non-negative).
        padding (int): Zero-padding size to apply to input boundaries (must be non-negative).
        device_id (int, optional): The identifier of the device on which the operation is performed.

    Returns:
        np.ndarray: An ND array containing the result of the maxpool operation with shape ([batches], n_channels, output_height, output_width).
    """

    assert_np_type_dim(features, bfloat16, [3, 4], name="features")

    kernel_height, kernel_width = repeat_or_noop(kernel_size, 2, "kernel_size")

    assert_nonnegative_int(padding, name="padding")
    assert_nonnegative_int(stride, name="stride")
    assert_nonnegative_int(kernel_height, name="kernel_height")
    assert_nonnegative_int(kernel_width, name="kernel_width")
    assert_nonnegative_int(device_id, name="device_id")

    features_batched = features[np.newaxis, ...] if features.ndim == 3 else features
    result = backend.maxpool2d_fprop(
        device_id, features_batched, kernel_height, kernel_width, stride, padding
    )
    return result[0] if features.ndim == 3 else result


def avgpool2d_fprop(
    features: np.ndarray,
    kernel_size: int | tuple[int, int],
    stride: int,
    padding: int,
    count_include_pad: bool = True,
    device_id: int = 0,
) -> np.ndarray:
    """
    Apply 2D `avg pooling <https://en.wikipedia.org/wiki/Pooling_layer>`_ operation to input features using specified parameters.

    Args:
        features (np.ndarray): A 3D array of input features with shape (n_channels, height, width) or a 4D array of input features with the shape (batches, n_channels, height, width). Must be dtype=bfloat16.
        kernel_size (int | (int, int)): Pooling window dimensions. If int, same size is used for both dimensions.
        stride (int): Step size for sliding pooling window (must be non-negative).
        padding (int): Zero-padding size to apply to input boundaries (must be non-negative).
        count_include_pad (bool, optional): When True, will include the zero-padding in the averaging calculation.
        device_id (int, optional): The identifier of the device on which the operation is performed.

    Returns:
        np.ndarray: An ND array containing the result of the avgpool operation with shape ([batches], n_channels, output_height, output_width).
    """

    assert_np_type_dim(features, bfloat16, [3, 4], name="features")

    kernel_height, kernel_width = repeat_or_noop(kernel_size, 2, "kernel_size")

    assert_nonnegative_int(padding, name="padding")
    assert_nonnegative_int(stride, name="stride")
    assert_nonnegative_int(kernel_height, name="kernel_height")
    assert_nonnegative_int(kernel_width, name="kernel_width")
    assert_nonnegative_int(device_id, name="device_id")

    features_batched = features[np.newaxis, ...] if features.ndim == 3 else features
    result = backend.avgpool2d_fprop(
        device_id,
        features_batched,
        kernel_height,
        kernel_width,
        stride,
        padding,
        count_include_pad,
    )
    return result[0] if features.ndim == 3 else result


def adaptive_maxpool2d_fprop(
    features: np.ndarray,
    output_size: int | tuple[int, int],
    device_id: int = 0,
) -> np.ndarray:
    """
    Performs a forward pass through an adaptive maxpooling2d layer.
    Only works for symmetric input and output sizes. Input size has to be integer multiple of output size.

    Args:
        features (np.ndarray): A 3D array of input features with shape (n_channels, height, width) or a 4D array of input features with the shape (batches, n_channels, height, width). Must be dtype=bfloat16.
        output_size (int | (int, int)): Pooling window dimensions. If int, same size is used for both dimensions.
        device_id (int, optional): The identifier of the device on which the operation is performed.

    Returns:
        np.ndarray: An ND array containing the result of the adaptive maxpool operation with shape ([batches], n_channels, output_height, output_width).
    """

    assert_np_type_dim(features, bfloat16, [3, 4], name="features")

    output_height, output_width = repeat_or_noop(output_size, 2, "output_size")

    assert_nonnegative_int(output_height, name="output_height")
    assert_nonnegative_int(output_width, name="output_width")
    assert_nonnegative_int(device_id, name="device_id")

    features_batched = features[np.newaxis, ...] if features.ndim == 3 else features
    result = backend.adaptive_maxpool2d_fprop(
        device_id, features_batched, output_height, output_width
    )
    return result[0] if features.ndim == 3 else result


def adaptive_avgpool2d_fprop(
    features: np.ndarray,
    output_size: int | tuple[int, int],
    device_id: int = 0,
) -> np.ndarray:
    """
    Performs a forward pass through an adaptive avgpooling2d layer.
    Only works for symmetric input and output sizes. Input size has to be integer multiple of output size.

    Args:
        features (np.ndarray): A 3D array of input features with shape (n_channels, height, width) or a 4D array of input features with the shape (batches, n_channels, height, width). Must be dtype=bfloat16.
        output_size (int | (int, int)): Pooling window dimensions. If int, same size is used for both dimensions.
        device_id (int, optional): The identifier of the device on which the operation is performed.

    Returns:
        np.ndarray: An ND array containing the result of the adaptive avgpool operation with shape ([batches], n_channels, output_height, output_width).
    """

    assert_np_type_dim(features, bfloat16, [3, 4], name="features")

    output_height, output_width = repeat_or_noop(output_size, 2, "output_size")

    assert_nonnegative_int(output_height, name="output_height")
    assert_nonnegative_int(output_width, name="output_width")
    assert_nonnegative_int(device_id, name="device_id")

    features_batched = features[np.newaxis, ...] if features.ndim == 3 else features
    result = backend.adaptive_avgpool2d_fprop(
        device_id, features_batched, output_height, output_width
    )
    return result[0] if features.ndim == 3 else result


def get_driver_info(device_id: int = 0) -> str:
    """
    Gets detailed information about the Q.ANT native computing driver and its direct dependencies.

    Args:
        device_id (int, optional): The identifier of the device on which the operation is performed.

    Returns:
        The information as a string

    """
    assert_nonnegative_int(device_id, name="device_id")
    return backend.get_driver_info(device_id)


def get_sensor_info(device_id: int) -> dict:
    """
    Get detailed information about the NPU sensors, e.g. temperature.

    Args:
        device_id (int): The identifier of the device on which the operation is performed.

    Returns:
        The information as a dictionary.
    """
    assert_nonnegative_int(device_id, name="device_id")
    return backend.get_sensor_info(device_id)


def get_available_npus() -> dict:
    """
    Lists all available NPUs on the system.

    Args:

    Returns:
        npus (dict[int, str]): The mapping from device_id to serial number
    """
    return backend.get_available_npus()


def get_version_info(device_id: int) -> dict:
    """
    Gets information about the firmware versions.

    Args:
        device_id (int): The identifier of the device on which the operation is performed.

    Returns:
        The version information as a dictionary.
    """
    assert_nonnegative_int(device_id, name="device_id")
    return backend.get_version_info(device_id)


def get_perf_counter(device_id: int) -> dict:
    """
    Gets the performance counters.

    Args:
        device_id (int): The identifier of the device on which the operation is performed.

    Returns:
        The information as a dictionary:
            'timebased_counter': Seconds since the last reset call.
            'stalling_counter': Seconds since the last reset call spent in idle (stalling) mode.
    """
    assert_nonnegative_int(device_id, name="device_id")
    return backend.get_perf_counter(device_id)


def reset_perf_counter(device_id: int):
    """
    Reset the performance counter.

    Args:
        device_id (int): The identifier of the device on which the operation is performed.

    Returns:

    """
    assert_nonnegative_int(device_id, name="device_id")
    return backend.reset_perf_counter(device_id)


def setup_logging(folder_name: str | pathlib.Path, loglevel: int):
    """
    Enable logging. Log messages are written to stdout and to a file

    Args:
        folder_name (str or pathlib.Path): The name of the folder in which the logfiles are stored. Individual log files are created in this folder and identified by timestamp.
        loglevel (int): The log level from 1 = debug to 4 = error.

    Returns:

    """
    folder_name_path = pathlib.Path(folder_name).resolve()
    folder_name_path.mkdir(exist_ok=True, parents=True)
    return backend.setup_logging(folder_name_path.as_posix(), loglevel)
