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

from rfi_simulator import bin_any, bin_mean, block_any
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


@pytest.mark.parametrize("n_bins", [0, -1])
def test_binning_rejects_a_non_positive_bin_count(n_bins):
    """A zero or negative count is an error, not an empty partition.

    Left unchecked it reaches `numpy.linspace` and comes back as an empty
    or reversed set of edges, so the caller gets an array of the wrong
    shape instead of a complaint.
    """
    for call in (
        lambda: bin_mean(np.zeros((2, 8)), axis=1, n_bins=n_bins),
        lambda: bin_any(np.zeros((2, 8), dtype=bool), axis=1, n_bins=n_bins),
        lambda: bin_edges(8, n_bins),
    ):
        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            call()


# ----------------------------------------------------------------------
# block_any: the fixed-block partition
# ----------------------------------------------------------------------
def test_block_any_uses_fixed_blocks_and_drops_the_tail():
    """Hand-computed: 10 cells in blocks of 4 is two blocks, two cells dropped.

    Cell 5 is inside block 1 and must land there; cell 9 is inside the
    two-cell tail that belongs to no block and must vanish entirely.
    """
    mask = np.zeros((1, 10), dtype=bool)
    mask[0, 5] = True
    mask[0, 9] = True
    np.testing.assert_array_equal(block_any(mask, 4), [[False, True]])


def test_block_any_differs_from_bin_any_on_a_ragged_axis():
    """The two partitions are genuinely different, and silently so.

    This is the whole reason `block_any` exists: with 10 cells and a
    block size of 4, `bin_any` into 2 bins cuts at cell 5 while
    `block_any` cuts at cell 4. A cell-4 flag therefore lands in bin 0
    under one rule and block 1 under the other, and a cell-8 flag lands
    in bin 1 under one rule and in the discarded tail under the other.
    Neither raises.
    """
    inside = np.zeros((1, 10), dtype=bool)
    inside[0, 4] = True
    np.testing.assert_array_equal(bin_any(inside, axis=1, n_bins=2), [[True, False]])
    np.testing.assert_array_equal(block_any(inside, 4), [[False, True]])

    tail = np.zeros((1, 10), dtype=bool)
    tail[0, 8] = True
    np.testing.assert_array_equal(bin_any(tail, axis=1, n_bins=2), [[False, True]])
    np.testing.assert_array_equal(block_any(tail, 4), [[False, False]])


def test_block_any_agrees_with_bin_any_when_the_block_size_divides():
    """The two rules coincide exactly on a divisible axis."""
    rng = np.random.default_rng(41)
    mask = rng.random((3, 64)) < 0.1
    for block_size in (1, 2, 4, 8, 16, 32, 64):
        np.testing.assert_array_equal(
            block_any(mask, block_size), bin_any(mask, axis=1, n_bins=64 // block_size)
        )


def test_block_any_pools_the_chosen_axis_only():
    """Leading axes pass through untouched, and `axis` is honoured."""
    rng = np.random.default_rng(42)
    mask = rng.random((2, 5, 12)) < 0.2
    assert block_any(mask, 5).shape == (2, 5, 2)
    assert block_any(mask, 2, axis=1).shape == (2, 2, 12)
    for antenna in range(2):
        np.testing.assert_array_equal(block_any(mask, 5)[antenna], block_any(mask[antenna], 5))


def test_block_any_returns_a_new_array():
    """A block size of 1 is the identity in value but not in identity."""
    mask = np.array([[True, False]])
    pooled = block_any(mask, 1)
    np.testing.assert_array_equal(pooled, mask)
    pooled[0, 1] = True
    assert not mask[0, 1]


def test_block_any_validates_its_input():
    """A non-boolean mask, a zero block, or an oversized block all raise."""
    with pytest.raises(ValueError, match="boolean mask"):
        block_any(np.zeros((2, 8)), 4)
    with pytest.raises(ValueError, match="block_size must be >= 1"):
        block_any(np.zeros((2, 8), dtype=bool), 0)
    with pytest.raises(ValueError, match="block_size must be <="):
        block_any(np.zeros((2, 8), dtype=bool), 9)
