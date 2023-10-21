# -----------------------------------------------------------------------------
# Copyright 2021 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Analysis of differential resistance and IV curves of superconducting devices."""
from typing import Literal, Optional

import numpy as np


def extract_switching_current(
    bias: np.ndarray,
    dvdi: np.ndarray,
    *,
    side: Literal["positive", "negative", "both"] = "positive",
    threshold: Optional[float] = None,
    interp: bool = False,
    offset: float = 0,
) -> np.ndarray:
    """Extract the switching currents from a set of differential resistance curves.

    This function will also work for V(I) curves if `offset` and an explicit `threshold`
    are given.

    Parameters
    ----------
    bias
        N-dimensional array of bias current, assumed to be swept along the last axis.
    dvdi
        N-dimensional array of differential resistance at the corresponding `bias`
        values; same shape as `bias`.
    side : optional
        Which branch of the switching current to extract (positive, negative, or both).
        If "both", a tuple of (negative, positive) switching currents is returned.
    threshold : optional
        The switching current is determined as the first `bias` value for which `dvdi`
        rises above `threshold`.
        If None, the threshold is inferred as half the rise from the minimum to the
        maximum `dvdi` value in the direction of `side`.
    interp : optional
        If true, linearly interpolate `dvdi` vs `bias` to more accurately detect the
        switching current.
    offset : optional
        A constant value to subtract from `dvdi`.

    Returns
    -------
    ic: ndarray
        The positive or negative branch of the switching current.  If `side` is "both",
        ic[0] = ic- and ic[1] = ic+.  The returned arrays have the same shape as the
        input arrays without the last axis.
    """
    if side not in ("positive", "negative", "both"):
        raise ValueError("`side` should be one of: 'positive', 'negative', 'both'")

    dvdi -= offset

    if side != "negative":
        ic_p = find_rising_edge(
            bias, np.where(bias >= 0, dvdi, np.nan), threshold=threshold, interp=interp
        )
        ic_p = np.nan_to_num(ic_p)
    if side != "positive":
        ic_n = find_rising_edge(
            bias[..., ::-1],
            # np.abs is used to support V(I) curves as well as dV/dI
            np.where(bias <= 0, np.abs(dvdi), np.nan)[..., ::-1],
            threshold=threshold,
            interp=interp,
        )
        ic_n = np.nan_to_num(ic_n)

    return (
        ic_p
        if side == "positive"
        else ic_n
        if side == "negative"
        else np.array((ic_n, ic_p))
    )


def find_rising_edge(x, y, *, threshold=None, interp=False):
    """Find the first `x` where `y` exceeds `threshold` (along the last axis).

    `x` and `y` must have the same shape.

    If `threshold` is None, it will be inferred as half the rise from min(y) to max(y).

    If `interp` is True, linearly interpolate between the two points below and above
    `threshold` to get a more accurate value of `x` at the crossing.
    """
    if threshold is None:
        threshold = ((np.nanmin(y, axis=-1) + np.nanmax(y, axis=-1)) / 2)[
            ..., np.newaxis
        ]
    index = np.argmax(y > threshold, axis=-1)
    x1 = np.take_along_axis(x, index[..., np.newaxis], axis=-1).squeeze(axis=-1)
    if interp:
        x0 = np.take_along_axis(x, index[..., np.newaxis] - 1, axis=-1).squeeze(axis=-1)
        y0 = np.take_along_axis(y, index[..., np.newaxis] - 1, axis=-1).squeeze(axis=-1)
        y1 = np.take_along_axis(y, index[..., np.newaxis], axis=-1).squeeze(axis=-1)
        dydx = (y1 - y0) / (x1 - x0)
        return x0 + (np.squeeze(threshold) - y0) / dydx
    else:
        return x1
