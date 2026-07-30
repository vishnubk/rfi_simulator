"""Tests for rfi_simulator.binning.

`bin_mean` and `bin_any` used to live inside the web front end, where the
only consumer was the waterfall display. They are now library code, so
these tests pin the behaviour that move had to preserve exactly: the
ragged case (an axis length that does not divide by the bin count) is the
one where two plausible implementations differ, since one of them drops
the remainder off the end.
"""

import numpy as np
import pytest

from rfi_simulator import bin_any, bin_mean
from rfi_simulator.binning import bin_edges
from rfi_simulator.webui import simulate as webui_simulate


def reference_bin_mean(values, axis, n_bins):
    """The pooling the web front end shipped before the move.

    Reproduced here verbatim so the equality test compares against the
    old code rather than against the new code restated.
    """
    length = values.shape[axis]
    if n_bins >= length:
        return values
    edges = np.linspace(0, length, n_bins + 1).astype(np.intp)
    counts = np.diff(edges)
    totals = np.add.reduceat(values, edges[:-1], axis=axis)
    shape = [1] * values.ndim
    shape[axis] = n_bins
    return totals / counts.reshape(shape)


def reference_bin_any(mask, axis, n_bins):
    """The ANY-pooling the web front end shipped before the move."""
    length = mask.shape[axis]
    if n_bins >= length:
        return mask
    edges = np.linspace(0, length, n_bins + 1).astype(np.intp)
    return np.maximum.reduceat(mask.astype(np.uint8), edges[:-1], axis=axis) > 0


# Lengths and bin counts chosen so that most pairs are ragged: 100 // 7 is
# 14 with a remainder of 2, 63 // 8 is 7 with a remainder of 7.
RAGGED_CASES = [(100, 7), (63, 8), (17, 5), (10, 3), (9, 2), (5, 4), (4, 4), (3, 8)]


@pytest.mark.parametrize(("length", "n_bins"), RAGGED_CASES)
def test_bin_mean_matches_the_previous_webui_implementation(length, n_bins):
    """Averaging is unchanged by the move, including when it is ragged."""
    rng = np.random.default_rng(20260730)
    values = rng.standard_normal((3, length, 2))
    for axis in (1,):
        expected = reference_bin_mean(values, axis=axis, n_bins=n_bins)
        np.testing.assert_array_equal(bin_mean(values, axis=axis, n_bins=n_bins), expected)


@pytest.mark.parametrize(("length", "n_bins"), RAGGED_CASES)
def test_bin_any_matches_the_previous_webui_implementation(length, n_bins):
    """ANY-pooling is unchanged by the move, including when it is ragged."""
    rng = np.random.default_rng(20260731)
    mask = rng.random((3, length, 2)) < 0.2
    expected = reference_bin_any(mask, axis=1, n_bins=n_bins)
    np.testing.assert_array_equal(bin_any(mask, axis=1, n_bins=n_bins), expected)


def test_webui_uses_the_library_implementation():
    """The front end re-imports rather than keeping its own copy."""
    assert webui_simulate.bin_mean is bin_mean
    assert webui_simulate.bin_any is bin_any


def test_bin_edges_cover_the_whole_axis_without_empty_bins():
    """Every cell lands in exactly one bin, and no bin is empty.

    This is the property that makes the ragged case well defined: the
    remainder is spread over the bins instead of being dropped.
    """
    for length in range(1, 40):
        for n_bins in range(1, length + 1):
            edges = bin_edges(length, n_bins)
            assert edges[0] == 0
            assert edges[-1] == length
            widths = np.diff(edges)
            assert widths.size == n_bins
            assert widths.min() >= 1
            # As equal as integer division allows.
            assert widths.max() - widths.min() <= 1


def test_bin_mean_of_a_ragged_axis_averages_unequal_bins_correctly():
    """Hand-computed: 5 cells into 2 bins is a 2-cell bin then a 3-cell bin.

    The edges are ``linspace(0, 5, 3) = [0, 2.5, 5]`` truncated to
    ``[0, 2, 5]``, so the remainder goes to the *last* bin, and each bin
    is averaged by its own width rather than by a nominal one.
    """
    values = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
    np.testing.assert_allclose(bin_mean(values, axis=0, n_bins=2), [1.5, 11.0])


def test_bin_any_is_true_when_any_cell_in_the_bin_is_true():
    """One contaminated cell contaminates its whole bin, and only its bin."""
    mask = np.zeros(9, dtype=bool)
    mask[4] = True
    np.testing.assert_array_equal(bin_any(mask, axis=0, n_bins=3), [False, True, False])


def test_binning_is_a_no_op_when_the_axis_is_already_short_enough():
    """Asking for at least as many bins as cells returns the input."""
    values = np.arange(4.0)
    mask = np.array([True, False, True, False])
    assert bin_mean(values, axis=0, n_bins=4) is values
    assert bin_mean(values, axis=0, n_bins=9) is values
    assert bin_any(mask, axis=0, n_bins=4) is mask
