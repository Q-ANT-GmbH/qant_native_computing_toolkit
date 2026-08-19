import numpy as np
import torch
from ml_dtypes import bfloat16

"""
Util functions only meant for testing purposes.
"""


def fold_xs(xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fold input into the range (0, pi/2)

    Returns:
        (xs_folded, signs): The folded values and the sign that a cosine at the original position would have.
    """

    xs_abs = np.fmod(np.abs(xs), 2 * np.pi)

    first_quarter = xs_abs <= np.pi / 2
    second_quarter = np.logical_and(np.pi / 2 < xs_abs, xs_abs <= np.pi)
    third_quarter = np.logical_and(np.pi < xs_abs, xs_abs <= 3 / 2 * np.pi)
    fourth_quarter = 3 / 2 * np.pi < xs_abs

    xs_folded = np.empty_like(xs)
    signs = np.empty_like(xs)

    xs_folded[first_quarter] = xs_abs[first_quarter]
    signs[first_quarter] = 1
    xs_folded[second_quarter] = np.pi - xs_abs[second_quarter]
    signs[second_quarter] = -1
    xs_folded[third_quarter] = xs_abs[third_quarter] - np.pi
    signs[third_quarter] = -1
    xs_folded[fourth_quarter] = 2 * np.pi - xs_abs[fourth_quarter]
    signs[fourth_quarter] = 1

    return xs_folded, signs


def numpy2torchbfloat(x: np.ndarray):
    """
    Converts a numpy.ndarray of type ml_dtypes.bfloat16 to a torch.tensor of type torch.bfloat16.
    Helper function needed because numpy/ml_dtypes bfloat16 is not compatible with torch.
    """
    return torch.tensor(x.astype(np.float32), dtype=torch.bfloat16)


def torch2numpybfloat(x: torch.Tensor):
    """
    Converts a torch.tensor of type torch.bfloat16 to a numpy.ndarray of type ml_dtypes.bfloat16.
    Helper function needed because numpy/ml_dtypes bfloat16 is not compatible with torch.
    """
    return x.to(torch.float32).numpy().astype(bfloat16)
