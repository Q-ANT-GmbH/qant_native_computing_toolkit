from __future__ import annotations

import numpy as np

"""
This file contains util functions for the python wrapper
"""


def assert_is_numpy(x, name: str = "input"):
    if not isinstance(x, np.ndarray):
        raise TypeError(f"{name} must be numpy array but is {type(x)}")


def assert_nonnegative_int(x, name: str = "input"):
    if not isinstance(x, int):
        raise TypeError(f"{name} must be int but is {type(x)}")
    if not x >= 0:
        raise ValueError(f"{name} must be nonnegative, but is {x}")


# Sentinel npu_id/device_id for automatic NPU selection: the first NPU that
# supports the requested operation. Kept in sync with AUTO_NPU_ID in
# src/qant_driver_import/mod.rs and QANT_NATIVE_COMPUTING_TOOLKIT_AUTO_NPU_ID
# in src/c_bridge/qant_native_computing_toolkit_generic.h.
AUTO_NPU_ID = -1


def assert_valid_npu_id(x, name: str = "device_id"):
    if not isinstance(x, int):
        raise TypeError(f"{name} must be int but is {type(x)}")
    if x < AUTO_NPU_ID:
        raise ValueError(
            f"{name} must be {AUTO_NPU_ID} (auto-select) or a non-negative NPU id, but is {x}"
        )


def assert_is_dtype(x: np.ndarray, dtype: type, name="input"):
    dtype_is = x.dtype
    if dtype_is != dtype:
        raise TypeError(f"{name} must be of dtype {dtype} but is {dtype_is}")


def assert_has_ndim(x: np.ndarray, ndim: int | list, name="input"):
    if isinstance(ndim, int) and not x.ndim == ndim:
        raise ValueError(
            f"{name} must have dimension {ndim}, but has dimension {x.ndim}"
        )
    if isinstance(ndim, list) and x.ndim not in ndim:
        raise ValueError(
            f"{name} must have one of the dimensions in {ndim}, but has dimension {x.ndim}"
        )


def assert_np_type_dim(x, dtype: type, ndim: int | list | None, name="input"):
    assert_is_numpy(x, name=name)
    assert_is_dtype(x, dtype, name=name)
    if ndim is not None:
        assert_has_ndim(x, ndim, name=name)


def assert_same_shape(x: np.ndarray, y: np.ndarray, name_x="input_0", name_y="input_1"):
    if not np.all(x.shape == y.shape):
        raise ValueError(
            f"{name_x} and {name_y} must have same shape, but are {x.shape}, {y.shape}"
        )


def repeat_or_noop(x: int | tuple, num_of_elements: int, name: str):
    if isinstance(x, int):
        return (x for _ in range(num_of_elements))
    else:
        if len(x) != num_of_elements:
            raise ValueError(
                f"{name} must be a single value or a tuple of {num_of_elements} values"
            )
        return x
