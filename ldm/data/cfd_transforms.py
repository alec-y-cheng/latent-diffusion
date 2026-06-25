"""Physical transforms for CFD conditioning tensors."""

import numpy as np


UK_ROOF_X = 0
UK_ROOF_Y = 1
UK_ROOF_DIRECTION_SIN = 6
UK_ROOF_DIRECTION_COS = 7


def transform_uk_roof_sample(x, y, transpose=False, hflip=False, vflip=False):
    """Apply a physical D4 transform to UK roof conditioning and targets."""
    if x.ndim != 3 or x.shape[0] != 8:
        raise ValueError(f"Expected UK roof input with shape (8,H,W), got {x.shape}")
    if y.ndim != 3 or y.shape[1:] != x.shape[1:]:
        raise ValueError(f"Input and target spatial shapes differ: {x.shape}, {y.shape}")

    if transpose:
        x = np.swapaxes(x, 1, 2)
        y = np.swapaxes(y, 1, 2)
        previous = x.copy()
        x[UK_ROOF_X] = previous[UK_ROOF_Y]
        x[UK_ROOF_Y] = previous[UK_ROOF_X]
        x[UK_ROOF_DIRECTION_SIN] = previous[UK_ROOF_DIRECTION_COS]
        x[UK_ROOF_DIRECTION_COS] = previous[UK_ROOF_DIRECTION_SIN]

    if hflip:
        x = x[:, :, ::-1]
        y = y[:, :, ::-1]
        x[UK_ROOF_X] *= -1
        x[UK_ROOF_DIRECTION_SIN] *= -1

    if vflip:
        x = x[:, ::-1, :]
        y = y[:, ::-1, :]
        x[UK_ROOF_Y] *= -1
        x[UK_ROOF_DIRECTION_COS] *= -1

    return np.ascontiguousarray(x), np.ascontiguousarray(y)
