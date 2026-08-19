import numpy as np
import pytest
import qant_native_computing_toolkit.utils as qutils
from numpy.testing import assert_equal


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((1), np.int16),
        ((100), np.int16),
        ((1, 1, 1), np.int16),
        ((1, 2, 3, 4, 5, 6), np.int16),
        ((1), np.float16),
        ((100), np.float16),
        ((1, 1, 1), np.float16),
        ((1, 2, 3, 4, 5, 6), np.float16),
    ],
)
def test_align_ndarray_page_boundary(shape, dtype):
    x = np.random.random(shape).astype(dtype)
    y = qutils.align_ndarray_page_boundary(x.copy())

    # check for data
    assert_equal(y.shape, x.shape)
    assert_equal(y, x)
    assert_equal(y.dtype, x.dtype)
    # check for alignment
    start_addr = y.__array_interface__["data"][0]
    assert_equal(start_addr % 4096, 0)


def test_align_ndarray_page_boundary_noop_for_aligned():
    x = np.random.random((1, 2, 3, 4, 5)).astype(np.int16)
    x = qutils.align_ndarray_page_boundary(x)
    start_addr_first = x.__array_interface__["data"][0]
    x = qutils.align_ndarray_page_boundary(x)
    start_addr_second = x.__array_interface__["data"][0]

    assert_equal(start_addr_second, start_addr_first)
