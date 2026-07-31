"""Reduction of arrays onto coarser grids along one axis.

A time-frequency array is often needed on a grid coarser than the one it
was produced on: a display has a few hundred pixels per axis, and a
flagger that accumulates ``M`` time samples emits one decision per
accumulation rather than one per sample. Whenever a quantity and a label
are pooled for comparison they must land on the *same* partition of the
axis, or the pooled pair stops describing the same cell -- which is the
whole reason this module exists rather than a `reshape` at each call
site.

Two partitions, and picking the wrong one is a silent error
--------------------------------------------------------------
**Fit-to-count** (`bin_edges`, `bin_mean`, `bin_any`): the axis is cut
into exactly `n_bins` pieces at ``numpy.linspace(0, length, n_bins + 1)``
truncated to integers. Bins are as equal as integer division allows, they
cover every cell, and none is empty as long as ``n_bins <= length``. When
the length does not divide by `n_bins` the remainder is **spread over the
bins**, so no data is dropped. This is what a display wants: fill the
pixels you have, lose nothing.

**Fixed-block** (`block_any`): the axis is cut into consecutive blocks of
exactly `block_size` cells and the final partial block is **discarded**.
This is what an accumulating flagger does -- `spectral_kurtosis_mask`
reads samples ``[k*M, (k+1)*M)`` and reaches no decision at all about the
tail -- so it is what ground truth must use to be compared against one.

The two agree only when `block_size` divides the axis length exactly.
Otherwise they interleave differently and a pooled label can sit one bin
away from the pooled decision it is scored against, which looks like a
flagger that half-works rather than like a bug. Pool truth for a
spectral-kurtosis mask with `rfi_simulator.metrics.pool_truth_accumulations`
(which uses `block_any`), never with the fit-to-count rule.

Pooling rules
-------------
* `bin_mean` averages, which is what a power spectrogram wants.
* `bin_any` and `block_any` take the logical OR, which is the only honest
  way to shrink a ground-truth mask: a pooled cell is contaminated if
  *any* cell inside it was, so pooling can never claim a cell is clean
  when part of it was not. It also means pooled truth is optimistic about
  how much of a cell the interference actually filled, which is the price
  of comparing masks across grids at all.

Masks are boolean with True = contaminated, as everywhere in this package.
"""

from __future__ import annotations

import numpy as np

__all__ = ["bin_any", "bin_edges", "bin_mean", "block_any"]


def _check_n_bins(n_bins: int) -> int:
    """Validate a bin count.

    Parameters
    ----------
    n_bins : int
        Candidate number of bins.

    Returns
    -------
    int
        `n_bins` as an int.

    Raises
    ------
    ValueError
        If `n_bins` is below 1. Zero or negative counts otherwise reach
        `numpy.linspace` and come back as an empty or reversed partition,
        which silently produces an array of the wrong shape instead of an
        error.
    """
    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    return n_bins


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

    Raises
    ------
    ValueError
        If `n_bins` is below 1.
    """
    return np.linspace(0, length, _check_n_bins(n_bins) + 1).astype(np.intp)


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
        Number of bins wanted, ``>= 1``.

    Returns
    -------
    numpy.ndarray
        Same shape as `values` except that `axis` has length `n_bins`. If
        ``n_bins >= values.shape[axis]`` the input is returned unchanged
        (not a copy), since there is nothing to pool.

    Raises
    ------
    ValueError
        If `n_bins` is below 1.

    Notes
    -----
    Bins are as equal as integer division allows and cover every cell, so
    no data is dropped off the end of the axis. See the module docstring
    for when that is the wrong rule.
    """
    n_bins = _check_n_bins(n_bins)
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
        Number of bins wanted, ``>= 1``.

    Returns
    -------
    numpy.ndarray
        Boolean array, same shape as `mask` except that `axis` has length
        `n_bins`. If ``n_bins >= mask.shape[axis]`` the input is returned
        unchanged (not a copy).

    Raises
    ------
    ValueError
        If `n_bins` is below 1.

    Notes
    -----
    A pooled cell is True if *any* cell inside it was, which is the only
    honest way to shrink ground truth: it never claims a clean cell where
    the library flagged interference.

    This is the *fit-to-count* partition. To pool truth against a flagger
    that accumulates fixed blocks of samples, use `block_any` instead --
    see the module docstring.
    """
    n_bins = _check_n_bins(n_bins)
    length = mask.shape[axis]
    if n_bins >= length:
        return mask
    edges = bin_edges(length, n_bins)
    return np.maximum.reduceat(mask.astype(np.uint8), edges[:-1], axis=axis) > 0


def block_any(mask: np.ndarray, block_size: int, axis: int = -1) -> np.ndarray:
    """Pool a boolean mask into fixed blocks along `axis`, dropping the tail.

    The partition an accumulating detector actually uses: block ``k``
    covers cells ``[k * block_size, (k + 1) * block_size)``, and the final
    ``length % block_size`` cells belong to no block and are discarded.

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean array, True = contaminated.
    block_size : int
        Cells per block, ``>= 1``. Must not exceed ``mask.shape[axis]``.
    axis : int, optional
        Axis to pool along. Default -1, the time axis by this package's
        convention.

    Returns
    -------
    numpy.ndarray
        Boolean array, same shape as `mask` except that `axis` has length
        ``mask.shape[axis] // block_size``. Always a new array.

    Raises
    ------
    ValueError
        If `mask` is not boolean, or if `block_size` is below 1 or longer
        than the axis.

    Notes
    -----
    Dropping the tail is the point, not a limitation: the samples in it
    took part in no accumulation, so the detector reached no decision
    about them and they must not be scored either way. Pooling them into
    a neighbouring block would charge the detector a false negative for
    data it never saw.

    Examples
    --------
    >>> import numpy as np
    >>> mask = np.zeros((1, 10), dtype=bool)
    >>> mask[0, 5] = True  # inside block 1
    >>> mask[0, 9] = True  # inside the dropped tail
    >>> block_any(mask, 4)
    array([[False,  True]])
    """
    values = np.asarray(mask)
    if values.dtype != np.bool_:
        raise ValueError(f"block_any needs a boolean mask, got dtype {values.dtype}")
    block_size = int(block_size)
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    length = values.shape[axis]
    if block_size > length:
        raise ValueError(
            f"block_size must be <= the length of axis {axis} ({length}), got {block_size}"
        )
    n_blocks = length // block_size
    work = np.moveaxis(values, axis, -1)[..., : n_blocks * block_size]
    work = work.reshape(*work.shape[:-1], n_blocks, block_size)
    return np.moveaxis(work.any(axis=-1), -1, axis)
