import numpy as np

PAGE_ALIGNMENT = 4096


def align_ndarray_page_boundary(data: np.ndarray):
    """
    Return a copy of a NumPy array whose underlying data buffer is aligned to a
    page-sized memory boundary.

    Args:
        data (np.ndarray): NumPy array which has to be aligned.


    Returns:
         A new NumPy array with the same shape and dtype as `data` aligned to PAGE_ALIGNMENT bytes.

    """

    if np.isscalar(data) or data.__array_interface__["data"][0] % PAGE_ALIGNMENT == 0:
        return data

    nbytes = int(np.prod(data.shape) * np.dtype(data.dtype).itemsize)
    buf = np.empty(nbytes + PAGE_ALIGNMENT, dtype=np.uint8)

    start_addr = buf.__array_interface__["data"][0]
    offset = (-start_addr) % PAGE_ALIGNMENT

    aligned_buf = buf[offset : offset + nbytes]
    aligned_buf = aligned_buf.view(data.dtype).reshape(data.shape)
    aligned_buf[:] = data[:]
    return aligned_buf
