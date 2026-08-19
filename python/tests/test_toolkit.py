import pathlib
import tempfile
import time

import numpy as np
import pytest
import qant_native_computing_toolkit.ai as q_ai
import qant_native_computing_toolkit.generic as q_generic
import qant_native_computing_toolkit.info as q_info
import qant_native_computing_toolkit.native as q_native
import torch
import utils
from ml_dtypes import bfloat16
from numpy.testing import assert_allclose, assert_array_less, assert_equal


# use the same (seeded) random number generator in all tests
@pytest.fixture(scope="session")
def rng():
    rng = np.random.default_rng(2)
    return rng


def test_init():
    q_generic.init_npu(0)
    q_generic.release_npu(0)


def check_mul32(rng):
    # factored out of the test to be used in different tests
    us = rng.random(1000).astype(np.float32)
    vs = rng.random(1000).astype(np.float32)
    ws = q_native.mul_f32(us, vs)
    assert_allclose(ws, us * vs, atol=0.03)
    mean_err = np.mean(np.abs(ws - us * vs))
    assert_array_less(mean_err, 0.004)


def test_mulf32(rng):
    check_mul32(rng)


@pytest.mark.parametrize(
    "features_shape", [(10), ((10, 2)), ((10, 2, 3)), ((10, 2, 3, 4))]
)
def test_muli16(features_shape, rng):
    us = rng.uniform(-1_000, 1_000, size=features_shape).astype(np.int16)
    vs = rng.uniform(-1_000, 1_000, size=features_shape).astype(np.int16)
    ws = q_native.mul_i16(us, vs)

    assert_allclose(
        ws.astype(bfloat16) / 1_000,
        us.astype(bfloat16) / 1_000 * vs.astype(bfloat16) / 1_000,
        atol=0.01,
    )


@pytest.mark.parametrize(
    "features_shape", [(10), ((10, 2)), ((10, 2, 3)), ((10, 2, 3, 4))]
)
def test_mul(features_shape, rng):
    us = rng.uniform(-1.0, 1.0, size=features_shape).astype(bfloat16)
    vs = rng.uniform(-1.0, 1.0, size=features_shape).astype(bfloat16)
    ws = q_native.mul_elementwise(us, vs)

    assert_allclose(
        ws,
        us * vs,
        atol=0.01,
    )


@pytest.mark.parametrize(
    "us, exception",
    [
        ([0.1, 0.2, 0.3], TypeError),
        (np.array([0.1, 0.2, 0.2]).astype(np.float64), TypeError),
        (np.array([[0.1, 0.2, 0.2], [0.3, 0.2, 0.1]]).astype(np.float32), ValueError),
        (np.array([0.1, 0.2, 0.3, 0.4]).astype(np.float32), ValueError),
    ],
)
def test_mul_f32_errors(us, exception):
    vs = np.array([0.1, 0.2, 0.2]).astype(np.float32)
    with pytest.raises(exception):
        q_native.mul_f32(us, vs)


@pytest.mark.parametrize(
    "features_shape, filter_shape, mean_err_tol, atol",
    [
        ((500,), (200, 500), 0.1, 0.2),
        ((1, 500), (200, 500), 0.1, 0.2),
        ((1, 2, 500), (200, 500), 0.1, 0.2),
        ((1, 2, 3, 500), (200, 500), 0.1, 0.2),
        ((10, 500), (200, 500), 0.1, 0.2),
    ],
)
def test_linear_fprop(features_shape, filter_shape, mean_err_tol, atol, rng):
    features = rng.uniform(-1.0, 1.0, size=features_shape).astype(bfloat16)
    filter = rng.uniform(-1.0, 1.0, size=filter_shape).astype(bfloat16)
    res_qant = q_native.linear_fprop(features, filter)

    assert_equal(res_qant.shape, (*features_shape[:-1], filter_shape[0]))

    features_torch = utils.numpy2torchbfloat(features)
    filter_torch = utils.numpy2torchbfloat(filter)
    res_torch = utils.torch2numpybfloat(
        torch.nn.functional.linear(features_torch, filter_torch)
    )

    mean_err = np.mean(np.abs(res_qant - res_torch))

    # matrix-vec adds errors of individual multiplications:
    # => mean error grows with sqrt(N), worst case with N
    assert_array_less(mean_err, mean_err_tol)
    assert_allclose(res_qant, res_torch, atol=atol)


def test_linear_fprop_preserves_1x1_output():
    # linear_fprop shouldn't squeeze result to a single value -> always return a 1d list
    features = np.ones((1, 1), dtype=bfloat16)
    filter = np.ones((1, 1), dtype=bfloat16)

    res_qant = q_native.linear_fprop(features, filter)

    assert_equal(res_qant.ndim, 2)
    assert_equal(res_qant.shape, (1, 1))


def test_linear_fprop_shape_error(rng):
    features = rng.uniform(-1.0, 1.0, size=(100, 1)).astype(bfloat16)
    filter_wrong_shape = rng.uniform(-1.0, 1.0, size=(200, 101)).astype(bfloat16)
    with pytest.raises(ValueError):
        q_native.linear_fprop(features, filter_wrong_shape)


@pytest.mark.parametrize(
    "features_shape, weights_shape",
    [
        ((10, 500), (200, 500)),  # plain 2-D: (n_batches, n_in)
        ((3, 10, 500), (200, 500)),  # extra batch dim: (..., n_in)
    ],
)
def test_linear_fprop_einsum(features_shape, weights_shape, rng):
    """Verifies the formula  output[...,j] = Σ_k features[...,k] * weights[j,k]
    using np.einsum('...k,jk->...j', features, weights) as the reference.
    """
    features = rng.uniform(-1.0, 1.0, size=features_shape).astype(bfloat16)
    weights = rng.uniform(-1.0, 1.0, size=weights_shape).astype(bfloat16)

    res_qant = q_native.linear_fprop(features, weights)

    # einsum directly encodes the documented formula; compute in float32 for precision
    ref = np.einsum(
        "...k,jk->...j", features.astype(np.float32), weights.astype(np.float32)
    ).astype(bfloat16)

    assert_equal(res_qant.shape, ref.shape)
    mean_err = np.mean(np.abs(res_qant.astype(np.float32) - ref.astype(np.float32)))
    assert_array_less(mean_err, 0.1)
    assert_allclose(res_qant, ref, atol=0.2)


@pytest.mark.parametrize(
    "features_shape, filter_shape, low, high, padding, stride, dilation, tol_kwargs",
    [
        ((100, 1, 1), (200, 100, 1, 1), 0, 1.0, 0, 1, 1, {"rtol": 0.05}),
        ((1, 100, 1, 1), (200, 100, 1, 1), 0, 1.0, 0, 1, 1, {"rtol": 0.05}),
        ((1, 100, 1, 1), (200, 100, 1, 1), 0, 1.0, 0, 3, 1, {"rtol": 0.05}),
        ((1, 1, 3, 3), (1, 1, 3, 3), 0, 1.0, 0, 1, 1, {"rtol": 0.05}),
        ((1, 2, 3, 3), (1, 2, 3, 3), 0, 1.0, 0, 1, 1, {"atol": 0.05}),
        ((1, 3, 4, 4), (10, 3, 2, 2), -0.1, 0.1, 0, 1, 1, {"atol": 0.01}),
        ((1, 3, 4, 4), (10, 3, 3, 2), -0.1, 0.1, 0, 1, 1, {"atol": 0.01}),
        ((1, 3, 4, 4), (10, 3, 2, 3), -0.1, 0.1, 0, 1, 1, {"atol": 0.01}),
    ],
)
def test_conv_fprop(
    features_shape,
    filter_shape,
    low,
    high,
    padding,
    stride,
    dilation,
    tol_kwargs,
    rng,
):
    # TODO: dilation != 1
    features = rng.uniform(low, high, size=features_shape).astype(bfloat16)
    filter = rng.uniform(low, high, size=filter_shape).astype(bfloat16)
    res_qant = q_ai.conv_fprop(
        features, filter, padding=padding, stride=stride, dilation=dilation
    )

    features_torch = utils.numpy2torchbfloat(features)
    filter_torch = utils.numpy2torchbfloat(filter)
    if len(features_shape) == 3:
        res_torch = utils.torch2numpybfloat(
            torch.nn.functional.conv2d(
                features_torch[torch.newaxis, ...],
                filter_torch,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )[0]
    else:
        res_torch = utils.torch2numpybfloat(
            torch.nn.functional.conv2d(
                features_torch,
                filter_torch,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )

    assert_equal(res_qant.shape, res_torch.shape)
    assert_allclose(res_qant, res_torch, **tol_kwargs)


@pytest.mark.parametrize(
    "features_shape, filter_shape, low, high, padding, stride, dilation, tol_kwargs",
    [
        ((10, 1, 1), (10, 20, 1, 1), 0, 1.0, 0, 1, 1, {"rtol": 0.05}),
        ((1, 10, 1, 1), (10, 20, 1, 1), 0, 1.0, 0, 1, 1, {"rtol": 0.05}),
        ((1, 10, 3, 3), (10, 20, 2, 2), 0, 1.0, 0, 1, 1, {"rtol": 0.05}),
        ((1, 10, 3, 3), (10, 20, 2, 2), 0, 1.0, 0, 2, 1, {"rtol": 0.05}),
        ((1, 10, 3, 3), (10, 20, 2, 2), 0, 1.0, 0, 3, 1, {"rtol": 0.05}),
        ((1, 10, 3, 3), (10, 20, 2, 1), 0, 1.0, 0, 3, 1, {"rtol": 0.05}),
        ((1, 10, 3, 3), (10, 20, 1, 2), 0, 1.0, 0, 3, 1, {"rtol": 0.05}),
    ],
)
def test_conv_transpose_fprop(
    features_shape,
    filter_shape,
    low,
    high,
    padding,
    stride,
    dilation,
    tol_kwargs,
    rng,
):
    features = rng.uniform(low, high, size=features_shape).astype(bfloat16)
    filter = rng.uniform(low, high, size=filter_shape).astype(bfloat16)

    res_qant = q_ai.conv_transpose_fprop(
        features,
        filter,
        padding=padding,
        stride=stride,
        dilation=dilation,
        output_padding=0,
    )

    features_torch = utils.numpy2torchbfloat(features)
    filter_torch = utils.numpy2torchbfloat(filter)
    if len(features_shape) == 3:
        res_torch = torch.nn.functional.conv_transpose2d(
            features_torch[torch.newaxis, ...],
            filter_torch,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output_padding=0,
        )[0]
    else:
        res_torch = torch.nn.functional.conv_transpose2d(
            features_torch,
            filter_torch,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output_padding=0,
        )

    assert_equal(res_qant.shape, res_torch.shape)
    assert_allclose(res_qant, utils.torch2numpybfloat(res_torch), **tol_kwargs)


@pytest.mark.parametrize(
    "features_shape, bias_shape, low, high",
    [
        ((1,), (1), 0, 1.0),
        ((2,), (2), 0, 1.0),
        ((3, 2), (2), 0, 1.0),
        ((2, 4, 4), (2), 0, 1.0),
        ((3, 2, 4, 4), (2), 0, 1.0),
        ((100, 1), (1), 0, 1.0),
        ((100, 2), (2), 0, 1.0),
        ((5, 100), (100), 0, 1.0),
    ],
)
def test_add_bias_fprop(features_shape, bias_shape, low, high, rng):
    features = rng.uniform(low, high, size=features_shape).astype(bfloat16)
    bias = rng.uniform(low, high, size=bias_shape).astype(bfloat16)
    res_qant = q_ai.add_bias_fprop(features, bias)
    if len(features_shape) == 3 or features.ndim == 4:
        bias = np.broadcast_to(bias[:, None, None], features_shape)
    assert_equal(res_qant, features + bias)


def test_noncontiguous(rng):
    features = rng.uniform(-1.0, 1.0, size=(100, 200)).astype(bfloat16)
    bias = rng.uniform(-1.0, 1.0, size=(200)).astype(bfloat16)
    features_T = features.T

    with pytest.raises(RuntimeError):
        q_ai.add_bias_fprop(features_T, bias)


def check_calc_scaled_periodic_nl_fprop(rng):
    n_elems = 500
    features = rng.uniform(0, 10.0, size=(n_elems)).astype(bfloat16)
    weights = rng.uniform(-1.0, 1.0, size=(n_elems)).astype(bfloat16)
    res_npu = q_native.calc_scaled_periodic_nl_fprop(features, weights)
    res_cpu = np.cos(features) * weights
    # large tolerance: the nonlinear function isn't exactly a cosine
    assert_allclose(res_npu, res_cpu, atol=0.2)


def test_calc_scaled_periodic_nl_fprop(rng):
    check_calc_scaled_periodic_nl_fprop(rng)


def test_mul_nonlin_swap(rng):
    check_mul32(rng)
    check_mul32(rng)
    check_calc_scaled_periodic_nl_fprop(rng)
    check_calc_scaled_periodic_nl_fprop(rng)
    check_mul32(rng)
    check_calc_scaled_periodic_nl_fprop(rng)
    check_mul32(rng)
    check_calc_scaled_periodic_nl_fprop(rng)


def test_kan_layer(rng):
    n_in_features = 10
    n_out_features = 20

    xs = rng.uniform(0, 2 * np.pi, size=(n_in_features,)).astype(bfloat16)
    ks = np.array([1.0, 2.0, 2.5]).astype(bfloat16)
    phis_layer0 = rng.uniform(
        -1, 1, size=(n_out_features, n_in_features, len(ks))
    ).astype(bfloat16)
    ampls_layer0 = rng.uniform(-1, 1, size=phis_layer0.shape).astype(bfloat16)

    res_npu = q_ai.calc_kan_layer_fprop(xs, phis_layer0, ampls_layer0, ks)

    res_cpu = np.zeros(n_out_features)
    for k_idx, k in enumerate(ks):
        res_cpu += np.sum(
            np.cos(k * xs[None, :] + phis_layer0[:, :, k_idx])
            * ampls_layer0[:, :, k_idx],
            axis=1,
        )
    assert_allclose(res_npu, res_cpu, atol=0.26)
    assert_array_less(np.mean(np.abs(res_npu - res_cpu)), 0.08)


def test_kan_layer_batched(rng):
    n_batches = 2
    n_in_features = 10
    n_out_features = 20
    xs = rng.uniform(0, 2 * np.pi, size=(n_batches, n_in_features)).astype(bfloat16)
    ks = np.array([1.0, 2.0, 2.5]).astype(bfloat16)
    phis_layer0 = rng.uniform(
        -1, 1, size=(n_out_features, n_in_features, len(ks))
    ).astype(bfloat16)
    ampls_layer0 = rng.uniform(-1, 1, size=phis_layer0.shape).astype(bfloat16)

    res_npu = q_ai.calc_kan_layer_fprop(xs, phis_layer0, ampls_layer0, ks)

    res_cpu = []
    for batch_idx in range(n_batches):
        res_batch = np.zeros(n_out_features)
        for k_idx, k in enumerate(ks):
            res_batch += np.sum(
                np.cos(k * xs[None, batch_idx, :] + phis_layer0[:, :, k_idx])
                * ampls_layer0[:, :, k_idx],
                axis=1,
            )
        res_cpu.append(res_batch)
    res_cpu = np.array(res_cpu)
    assert_allclose(res_npu, res_cpu, atol=0.27)
    assert_array_less(np.mean(np.abs(res_npu - res_cpu)), 0.08)


@pytest.mark.parametrize(
    "features_shape", [(100), ((100, 2)), ((100, 2, 3)), ((100, 2, 3, 4))]
)
def test_relu_fprop(features_shape, rng):
    features = rng.uniform(-1.0, 1.0, size=features_shape).astype(bfloat16)
    relu_np = features.copy()
    relu_np[relu_np < 0] = 0

    assert_equal(q_ai.relu_fprop(features), relu_np)


@pytest.mark.parametrize(
    "features_shape", [(100), ((100, 2)), ((100, 2, 3)), ((100, 2, 3, 4))]
)
def test_sigmoid_fprop(features_shape, rng):
    # arbitrary dimensions of the input should be supported
    features = rng.uniform(0, 1.0, size=features_shape).astype(bfloat16)
    torch_features = utils.numpy2torchbfloat(features)
    assert_allclose(
        q_ai.sigmoid_fprop(features),
        utils.torch2numpybfloat(torch.sigmoid(torch_features)),
        atol=0.005,
    )


@pytest.mark.parametrize(
    "features_shape", [(100), ((1, 100)), ((1, 1, 100)), ((1, 1, 1, 100))]
)
def test_softmax_fprop(features_shape, rng):
    features = rng.uniform(0, 1.0, size=features_shape).astype(bfloat16)
    softmax = q_ai.softmax_fprop(features)
    # TODO: tolerance should be smaller
    assert_allclose(softmax.sum(-1), 1, rtol=0.06)
    assert np.all(softmax > 0)


@pytest.mark.parametrize("features_shape", [((100, 1, 3)), ((1, 100, 1, 3))])
def test_batchnorm2d_fprop(features_shape, rng):
    features = rng.uniform(0, 1.0, size=(features_shape)).astype(bfloat16)
    means = rng.uniform(0, 1.0, size=(features_shape[-3],)).astype(bfloat16)
    variances = rng.uniform(1.0, 2.0, size=(features_shape[-3],)).astype(bfloat16)
    weights = rng.uniform(0, 1.0, size=(features_shape[-3],)).astype(bfloat16)
    bias = rng.uniform(0, 1.0, size=(features_shape[-3],)).astype(bfloat16)
    eps = 0  # default value is 1e-5
    res_qant = q_ai.batchnorm2d_fprop(features, means, variances, weights, bias, eps)

    features_torch = utils.numpy2torchbfloat(features)
    means_torch = utils.numpy2torchbfloat(means)
    variances_torch = utils.numpy2torchbfloat(variances)
    weights_torch = utils.numpy2torchbfloat(weights)
    bias_torch = utils.numpy2torchbfloat(bias)
    eps_torch = eps
    if len(features_shape) == 3:
        res_torch = torch.nn.functional.batch_norm(
            features_torch[torch.newaxis, ...],
            means_torch,
            variances_torch,
            weights_torch,
            bias_torch,
            training=False,
            momentum=0.0,
            eps=eps_torch,
        )[0]
    else:
        res_torch = torch.nn.functional.batch_norm(
            features_torch,
            means_torch,
            variances_torch,
            weights_torch,
            bias_torch,
            training=False,
            momentum=0.0,
            eps=eps_torch,
        )

    assert_equal(res_qant.shape, res_torch.shape)
    assert_allclose(res_qant, utils.torch2numpybfloat(res_torch), atol=0.015)


@pytest.mark.parametrize(
    "features_shape, kernel_size, stride, padding",
    [
        ((2, 2, 2), 1, 1, 0),
        ((1, 2, 2, 2), 1, 1, 0),
        ((1, 100, 1, 1), 1, 1, 0),
        ((1, 100, 10, 10), (3, 3), 1, 0),
        ((1, 100, 32, 55), 5, 1, 0),
        ((1, 100, 10, 10), (3, 3), 1, 1),
        ((1, 100, 10, 10), (3, 3), 2, 0),
        ((1, 100, 10, 10), (3, 3), 2, 1),
        ((1, 100, 10, 10), (3, 2), 1, 0),
        ((1, 100, 10, 10), (2, 3), 1, 0),
    ],
)
def test_maxpool2d_fprop(features_shape, kernel_size, stride, padding, rng):
    features = rng.uniform(0, 1.0, size=features_shape).astype(bfloat16)

    res_qant = q_ai.maxpool2d_fprop(
        features, kernel_size=kernel_size, stride=stride, padding=padding
    )

    features_torch = utils.numpy2torchbfloat(features)
    if len(features_shape) == 3:
        res_torch = torch.nn.functional.max_pool2d(
            features_torch[torch.newaxis, ...],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )[0]
    else:
        res_torch = torch.nn.functional.max_pool2d(
            features_torch, kernel_size=kernel_size, stride=stride, padding=padding
        )

    assert_equal(res_qant.shape, res_torch.shape)
    assert_allclose(res_qant, utils.torch2numpybfloat(res_torch), rtol=0.001)


@pytest.mark.parametrize(
    "features_shape, kernel_size, stride, padding, count_include_pad",
    [
        ((2, 2, 2), 1, 1, 0, False),
        ((1, 2, 2, 2), 1, 1, 0, False),
        ((1, 100, 1, 1), 1, 1, 0, False),
        ((1, 100, 10, 10), (3, 3), 1, 0, False),
        ((1, 100, 32, 55), 5, 1, 0, False),
        ((1, 100, 10, 10), (3, 3), 1, 1, False),
        ((1, 100, 10, 10), (3, 3), 1, 1, True),
        ((1, 100, 10, 10), (3, 3), 2, 0, False),
        ((1, 100, 10, 10), (3, 3), 2, 1, False),
        ((1, 100, 10, 10), (3, 3), 2, 1, True),
        ((1, 100, 10, 10), (3, 2), 1, 0, False),
        ((1, 100, 10, 10), (2, 3), 1, 0, False),
    ],
)
def test_avgpool2d_fprop(
    features_shape, kernel_size, stride, padding, count_include_pad, rng
):
    features = rng.uniform(0, 1.0, size=features_shape).astype(bfloat16)

    res_qant = q_ai.avgpool2d_fprop(
        features,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        count_include_pad=count_include_pad,
    )

    features_torch = utils.numpy2torchbfloat(features)
    if len(features_shape) == 3:
        res_torch = torch.nn.functional.avg_pool2d(
            features_torch[torch.newaxis, ...],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            count_include_pad=count_include_pad,
        )[0]
    else:
        res_torch = torch.nn.functional.avg_pool2d(
            features_torch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            count_include_pad=count_include_pad,
        )

    assert_equal(res_qant.shape, res_torch.shape)
    assert_allclose(res_qant, utils.torch2numpybfloat(res_torch), atol=0.01)


@pytest.mark.parametrize(
    "features_shape, output_size",
    [
        ((2, 2, 2), 1),
        ((1, 2, 2, 2), 1),
        ((1, 100, 1, 1), 1),
        ((1, 100, 9, 9), (3, 3)),
        ((1, 100, 9, 9), (9, 9)),
    ],
)
def test_adaptive_maxpool2d_fprop(features_shape, output_size, rng):
    features = rng.uniform(0, 1.0, size=features_shape).astype(bfloat16)

    res_qant = q_ai.adaptive_maxpool2d_fprop(features, output_size=output_size)

    features_torch = utils.numpy2torchbfloat(features)
    if len(features_shape) == 3:
        res_torch = torch.nn.functional.adaptive_max_pool2d(
            features_torch[torch.newaxis, ...],
            output_size=output_size,
        )[0]
    else:
        res_torch = torch.nn.functional.adaptive_max_pool2d(
            features_torch,
            output_size=output_size,
        )

    assert_equal(res_qant.shape, res_torch.shape)
    assert_allclose(res_qant, utils.torch2numpybfloat(res_torch), atol=0.001)


@pytest.mark.parametrize(
    "features_shape, output_size",
    [
        ((2, 2, 2), 1),
        ((1, 2, 2, 2), 1),
        ((1, 100, 1, 1), 1),
        ((1, 100, 9, 9), (3, 3)),
        ((1, 100, 9, 9), (9, 9)),
    ],
)
def test_adaptive_avgpool2d_fprop(features_shape, output_size, rng):
    features = rng.uniform(0, 1.0, size=features_shape).astype(bfloat16)

    res_qant = q_ai.adaptive_avgpool2d_fprop(features, output_size=output_size)

    features_torch = utils.numpy2torchbfloat(features)
    if len(features_shape) == 3:
        res_torch = torch.nn.functional.adaptive_avg_pool2d(
            features_torch[torch.newaxis, ...],
            output_size=output_size,
        )[0]
    else:
        res_torch = torch.nn.functional.adaptive_avg_pool2d(
            features_torch,
            output_size=output_size,
        )

    assert_equal(res_qant.shape, res_torch.shape)
    assert_allclose(res_qant, utils.torch2numpybfloat(res_torch), atol=0.01)


def test_service_info():
    sensor_info = q_info.get_sensor_info(0)
    version_info = q_info.get_version_info(0)
    driver_info = q_info.get_driver_info(0)
    print(f"{sensor_info=}")
    print(f"{version_info=}")
    print(f"{driver_info=}")
    assert "temp_pd1" in sensor_info
    assert "fw_version" in version_info
    assert "commit" in driver_info


def test_get_available_npus():
    npus = q_generic.get_available_npus()
    assert len(npus.keys()) > 0


def test_performance_measurement():
    q_info.reset_perf_counter(0)
    time.sleep(0.1)
    counters = q_info.get_perf_counter(0)
    assert "stalling_counter" in counters


@pytest.mark.logging
def test_logging():
    with tempfile.TemporaryDirectory() as tmpdir:
        q_info.setup_logging(tmpdir, 1)
        tmpdir_path = pathlib.Path(tmpdir)
        log_files = list(tmpdir_path.glob("qant_driver_log*.txt"))
        assert len(log_files) > 0

        with pytest.raises(RuntimeError):
            # illegal level
            q_info.setup_logging(tmpdir, 1234)
        with pytest.raises(RuntimeError):
            # setup twice
            q_info.setup_logging(tmpdir, 1)
