"""Reduction of arrays onto coarser grids along one axis.

Two operations, one convention. A time-frequency array is often needed on
a grid coarser than the one it was produced on: a display has a few
hundred pixels per axis, and a flagger that works on blocks of ``M`` time
samples emits one decision per block rather than one per sample. Both
cases pool a long axis into `n_bins` cells, and both need the *same*
partition of that axis, or a pooled quantity and a pooled label stop
describing the same cell.

The partition is `numpy.linspace(0, length, n_bins + 1)` truncated to
integers: bins are as equal as integer division allows, they cover every
cell of the axis, and none is empty as long as ``n_bins <= length``. The
axis length need not divide by `n_bins` -- the remainder is spread over
the bins rather than dropped off the end, so no data is silently lost.
Asking for at least as many bins as there are cells is a no-op and
returns the input unchanged.

The two rules differ in what "pooling" means, and the difference is
deliberate:

* `bin_mean` averages, which is what a power spectrogram wants.
* `bin_any` takes the logical OR, which is the only honest way to shrink
  a ground-truth mask: a pooled cell is contaminated if *any* cell inside
  it was, so pooling can never claim a cell is clean when part of it was
  not. It also means pooled truth is optimistic about how much of a cell
  the interference actually filled, which is the price of comparing masks
  across grids at all.

Masks are boolean with True = contaminated, as everywhere in this package.
"""

from __future__ import annotations

import numpy as np

__all__ = ["bin_any", "bin_edges", "bin_mean"]


def bin_edges(length: int, n_bins: int) -> np.ndarray:
    """Start index of every bin when `length` cells are pooled into `n_bins`.

    Parameters
    ----------
    length : int
        Number of cells along the axis being pooled.
    n_bins : int
        Number of bins wanted, ``1 <= n_bins <= length``.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_bins + 1,)`` integer array of edges, starting at 0 and
        ending at `length`. Bin ``k`` covers ``edges[k]:edges[k + 1]``.
    """
    return np.linspace(0, length, n_bins + 1).astype(np.intp)


def bin_mean(values: np.ndarray, axis: int, n_bins: int) -> np.ndarray:
    """Average `values` down to `n_bins` along `axis`.

    Parameters
    ----------
    values : numpy.ndarray
        Array to pool. Any numeric dtype; the units of the result are the
        units of the input, since this is a mean and not a sum.
    axis : int
        Axis to pool along.
    n_bins : int
        Number of bins wanted.

    Returns
    -------
    numpy.ndarray
        Same shape as `values` except that `axis` has length `n_bins`. If
        ``n_bins >= values.shape[axis]`` the input is returned unchanged
        (not a copy), since there is nothing to pool.

    Notes
    -----
    Bins are as equal as integer division allows and cover every cell, so
    no data is dropped off the end of the axis.
    """
    length = values.shape[axis]
    if n_bins >= length:
        return values
    edges = bin_edges(length, n_bins)
    counts = np.diff(edges)
    totals = np.add.reduceat(values, edges[:-1], axis=axis)
    shape = [1] * values.ndim
    shape[axis] = n_bins
    return totals / counts.reshape(shape)


def bin_any(mask: np.ndarray, axis: int, n_bins: int) -> np.ndarray:
    """Pool a boolean mask down to `n_bins` along `axis` with an ANY rule.

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean array, True = contaminated.
    axis : int
        Axis to pool along.
    n_bins : int
        Number of bins wanted.

    Returns
    -------
    numpy.ndarray
        Boolean array, same shape as `mask` except that `axis` has length
        `n_bins`. If ``n_bins >= mask.shape[axis]`` the input is returned
        unchanged (not a copy).

    Notes
    -----
    A pooled cell is True if *any* cell inside it was, which is the only
    honest way to shrink ground truth: it never claims a clean cell where
    the library flagged interference.
    """
    length = mask.shape[axis]
    if n_bins >= length:
        return mask
    edges = bin_edges(length, n_bins)
    return np.maximum.reduceat(mask.astype(np.uint8), edges[:-1], axis=axis) > 0
