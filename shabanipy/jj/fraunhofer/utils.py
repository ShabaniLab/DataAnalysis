# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Routines to analyse data taken on JJ.

"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import GaussianModel
from scipy.signal import peak_widths


def f2k_from_periodicity_and_width(periodicty: float, width: float) -> float:
    """Field to k estimate from the periodicity of a pattern and the assumed width."""
    return 2 * np.pi / (periodicty * width)


def find_fraunhofer_center(
    field: np.ndarray,
    ic: np.ndarray,
    *,
    field_lim: Optional[Tuple[float, float]] = None,
    return_fit: bool = False,
    debug: bool = False,
) -> Union[float, lmfit.model.ModelResult]:
    """Extract the field at which the Fraunhofer is centered.

    The center is found by fitting the largest peak.

    Parameters
    ----------
    field : np.ndarray
        1D array of the magnetic field applied of the JJ.
    ic : np.ndarray
        1D array of the JJ critical current.
    field_lim : optional (float, float)
        Limit search to within field_lim (min, max).
    return_fit: bool
        Return the ModelResult of the fitting.

    Returns
    -------
    center: float
        Field at which the center of the pattern is located.

    OR

    result: lmfit.model.ModelResult
        If return_fit=True, the entire ModelResult from the fitting procedure is
        returned.  The `field` values used in the fit can be accessed with
        `result.xdata`.
    """
    if field_lim is not None:
        max_loc = np.argmax(
            np.where((field_lim[0] < field) & (field < field_lim[1]), ic, -np.inf)
        )
    else:
        max_loc = np.argmax(ic)
    width, *_ = peak_widths(ic, [max_loc], rel_height=0.5)
    width_index = int(round(width[0] * 0.65))
    start_index = max(0, max_loc - width_index)
    subset_field = field[start_index : max_loc + width_index + 1]
    subset_ic = ic[start_index : max_loc + width_index + 1]
    model = GaussianModel()

    params = model.guess(subset_ic, subset_field)
    out = model.fit(subset_ic, params, x=subset_field)
    out.xdata = subset_field

    if debug:
        plt.figure()
        plt.plot(field, ic)
        plt.plot(subset_field, out.best_fit)
        plt.show()

    if return_fit:
        return out
    else:
        return out.best_values["center"]


def recenter_fraunhofer(
    field: np.ndarray, ic: np.ndarray, debug: bool = False
) -> np.ndarray:
    """Correct the offset in field of a Fraunhofer pattern.

    Parameters
    ----------
    field : np.ndarray
        ND array of the magnetic field applied of the JJ, the last dimension is
        expected to be swept.
    ic : np.ndarray
        ND array of the JJ critical current.

    Returns
    -------
    np.ndarray
        Field array from which the offset has been removed.

    """
    it = np.nditer(field[..., 0], ["multi_index"])
    res = np.copy(field)
    for b in it:
        index = it.multi_index
        center = find_fraunhofer_center(field[index], ic[index], debug=debug)
        res[index] -= center

    return res


def symmetrize_fraunhofer(
    field: np.ndarray, ic: np.ndarray, debug: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetrize a Fraunhofer pattern.

    We conserve the side on which more lobes are visible and perform a mirror.

    Parameters
    ----------
    field : np.ndarray
        1D array of the magnetic field applied of the JJ, the last dimension is
        expected to be swept. The field should be offset free.
    ic : np.ndarray
        1D array of the JJ critical current.

    Returns
    -------
    np.ndarray
        New field array which has been symmetrizes.
    np.ndarray
        Critical current symmetrized with respect to 0 field.

    """
    # Ensure we get increasing value of field
    if field[0] > field[1]:
        field = field[::-1]
        ic = ic[::-1]

    index = np.argmin(np.abs(field))
    if index == 0:
        side = "positive"
        f = field
        i = ic
    elif index == len(field) - 1:
        side = "negative"
        f = field
        i = ic
    else:
        if len(field[:index]) > len(field[index + 1 :]):
            side = "negative"
            f = field[: index + 1]
            i = ic[: index + 1]
        else:
            side = "positive"
            f = field[index:]
            i = ic[index:]

    if side == "positive":
        out = (
            np.concatenate((-f[1:][::-1], f), axis=None),
            np.concatenate((i[1:][::-1], i), axis=None),
        )
    else:
        out = (
            np.concatenate((f, -f[:-1][::-1]), axis=None),
            np.concatenate((i, i[:-1][::-1]), axis=None),
        )

    if debug:
        plt.figure()
        plt.plot(*out)
        plt.show()

    return out
