"""Tests for rfi_simulator.metrics.

Every score here is hand-computed on a mask small enough to count by eye,
because the value of a metrics module is entirely in its edge cases: the
interesting inputs are the ones where a denominator vanishes, and those
are exactly the ones a "looks about right" test never reaches. The
degenerate cases are enumerated deliberately -- clean truth, fully
contaminated truth, empty prediction, saturated prediction, and perfect
anti-correlation.
"""

import math

import numpy as np
import pytest

from rfi_simulator import confusion_counts, flag_scores, pool_truth
from rfi_simulator.binning import bin_any

# A 2x3 toy whose counts are tp=1, fp=1, fn=1, tn=3. Small enough to
# check by eye, but not symmetric, so a transposed or swapped-argument
# implementation cannot pass by accident.
TOY_TRUTH = np.array([[True, True, False], [False, False, False]])
TOY_PREDICTED = np.array([[True, False, False], [False, False, True]])


def test_confusion_counts_match_a_hand_count():
    """The four outcomes of the 2x3 toy, counted by hand."""
    assert confusion_counts(TOY_PREDICTED, TOY_TRUTH) == (1, 1, 1, 3)


def test_confusion_counts_are_not_symmetric_in_their_arguments():
    """Swapping predicted and truth swaps false positives with false negatives.

    A regression guard: the two arguments are easy to transpose, and the
    counts are the only place the mix-up is visible, since precision and
    recall then swap too.
    """
    tp, fp, fn, tn = confusion_counts(TOY_PREDICTED, TOY_TRUTH)
    assert confusion_counts(TOY_TRUTH, TOY_PREDICTED) == (tp, fn, fp, tn)


def test_confusion_counts_cover_every_cell():
    """The counts partition the array: nothing is counted twice or missed."""
    rng = np.random.default_rng(3)
    truth = rng.random((7, 11)) < 0.3
    predicted = rng.random((7, 11)) < 0.5
    assert sum(confusion_counts(predicted, truth)) == truth.size


def test_confusion_counts_reject_mismatched_shapes():
    """Broadcastable shapes are a bug, not a convenience."""
    with pytest.raises(ValueError, match="same shape"):
        confusion_counts(np.zeros((4, 1), dtype=bool), np.zeros((4, 3), dtype=bool))


@pytest.mark.parametrize("array", [np.zeros((2, 2), dtype=np.int8), np.zeros((2, 2))])
def test_confusion_counts_reject_non_boolean_masks(array):
    """A statistic array must be thresholded into a decision before scoring."""
    with pytest.raises(ValueError, match="boolean mask"):
        confusion_counts(array, np.zeros((2, 2), dtype=bool))
    with pytest.raises(ValueError, match="boolean mask"):
        confusion_counts(np.zeros((2, 2), dtype=bool), array)


def test_flag_scores_match_hand_computed_values():
    """Every score of the 2x3 toy, worked out from tp=1, fp=1, fn=1, tn=3.

    precision = 1/2, recall = 1/2, f1 = 2/(2+1+1) = 1/2,
    false-positive rate = 1/(1+3) = 1/4, truth occupancy = 2/6,
    predicted occupancy = 2/6, and
    mcc = (1*3 - 1*1) / sqrt(2 * 2 * 4 * 4) = 2/8 = 1/4.
    """
    scores = flag_scores(TOY_PREDICTED, TOY_TRUTH)
    assert scores["tp"] == 1.0
    assert scores["fp"] == 1.0
    assert scores["fn"] == 1.0
    assert scores["tn"] == 3.0
    assert scores["precision"] == pytest.approx(0.5)
    assert scores["recall"] == pytest.approx(0.5)
    assert scores["f1"] == pytest.approx(0.5)
    assert scores["mcc"] == pytest.approx(0.25)
    assert scores["false_positive_rate"] == pytest.approx(0.25)
    assert scores["truth_occupancy"] == pytest.approx(2.0 / 6.0)
    assert scores["predicted_occupancy"] == pytest.approx(2.0 / 6.0)


def test_flag_scores_of_a_perfect_prediction():
    """A mask equal to the truth scores 1 everywhere it is defined."""
    scores = flag_scores(TOY_TRUTH.copy(), TOY_TRUTH)
    assert scores["precision"] == 1.0
    assert scores["recall"] == 1.0
    assert scores["f1"] == 1.0
    assert scores["mcc"] == pytest.approx(1.0)
    assert scores["false_positive_rate"] == 0.0


def test_flag_scores_of_a_perfectly_inverted_prediction():
    """The complement of the truth has mcc -1 and recall 0."""
    scores = flag_scores(~TOY_TRUTH, TOY_TRUTH)
    assert scores["mcc"] == pytest.approx(-1.0)
    assert scores["recall"] == 0.0
    assert scores["precision"] == 0.0
    assert scores["false_positive_rate"] == 1.0


def test_flag_scores_of_clean_truth_and_empty_prediction():
    """Nothing to find and nothing found: both ratios undefined, mcc 0.

    Returning 0.0 for precision here would report a failure where there
    was no test, so the convention is NaN. The MCC denominator vanishes
    (both masks are constant) and the convention there is 0.
    """
    clean = np.zeros((2, 3), dtype=bool)
    scores = flag_scores(clean, clean)
    assert math.isnan(scores["precision"])
    assert math.isnan(scores["recall"])
    assert math.isnan(scores["f1"])
    assert scores["mcc"] == 0.0
    assert scores["false_positive_rate"] == 0.0
    assert scores["truth_occupancy"] == 0.0
    assert scores["predicted_occupancy"] == 0.0


def test_flag_scores_of_clean_truth_with_false_positives():
    """Flagging clean data: precision 0, recall undefined, mcc 0."""
    clean = np.zeros((2, 3), dtype=bool)
    predicted = np.zeros((2, 3), dtype=bool)
    predicted[0, 0] = True
    scores = flag_scores(predicted, clean)
    assert scores["precision"] == 0.0
    assert math.isnan(scores["recall"])
    assert scores["f1"] == 0.0
    assert scores["mcc"] == 0.0
    assert scores["false_positive_rate"] == pytest.approx(1.0 / 6.0)


def test_flag_scores_of_fully_contaminated_truth():
    """No clean cells: the false-positive rate is undefined."""
    dirty = np.ones((2, 3), dtype=bool)
    scores = flag_scores(dirty.copy(), dirty)
    assert math.isnan(scores["false_positive_rate"])
    assert scores["recall"] == 1.0
    assert scores["precision"] == 1.0
    assert scores["mcc"] == 0.0  # truth is constant: denominator vanishes
    assert scores["truth_occupancy"] == 1.0


def test_flag_scores_of_flag_everything_has_zero_mcc():
    """The degenerate strategy that beats accuracy scores nothing on mcc.

    This is the reason MCC is in the dictionary at all: on sparse
    interference, "flag everything" and "flag nothing" both look
    respectable on some scores and both score exactly 0 here.
    """
    truth = np.zeros((4, 4), dtype=bool)
    truth[0, 0] = True
    assert flag_scores(np.ones((4, 4), dtype=bool), truth)["mcc"] == 0.0
    assert flag_scores(np.zeros((4, 4), dtype=bool), truth)["mcc"] == 0.0


def test_flag_scores_of_disjoint_non_empty_masks():
    """Predicting the wrong cells scores f1 0, not NaN."""
    truth = np.array([[True, False]])
    predicted = np.array([[False, True]])
    scores = flag_scores(predicted, truth)
    assert scores["f1"] == 0.0
    assert scores["precision"] == 0.0
    assert scores["recall"] == 0.0


def test_flag_scores_reject_empty_masks():
    """There is nothing to score in a zero-sized array."""
    with pytest.raises(ValueError, match="empty masks"):
        flag_scores(np.zeros((0, 3), dtype=bool), np.zeros((0, 3), dtype=bool))


def test_flag_scores_is_within_range_for_random_masks():
    """The bounded scores stay bounded over many random cases."""
    rng = np.random.default_rng(17)
    for _ in range(50):
        truth = rng.random((8, 8)) < rng.uniform(0.05, 0.95)
        predicted = rng.random((8, 8)) < rng.uniform(0.05, 0.95)
        scores = flag_scores(predicted, truth)
        assert -1.0 <= scores["mcc"] <= 1.0
        for key in ("precision", "recall", "f1", "false_positive_rate"):
            value = scores[key]
            assert math.isnan(value) or 0.0 <= value <= 1.0


# ----------------------------------------------------------------------
# pool_truth
# ----------------------------------------------------------------------
def test_pool_truth_uses_the_any_rule():
    """One contaminated fine cell contaminates the coarse cell it lands in."""
    truth = np.zeros((2, 8), dtype=bool)
    truth[0, 5] = True
    pooled = pool_truth(truth, (2, 4))
    np.testing.assert_array_equal(pooled, [[False, False, True, False], [False] * 4])


def test_pool_truth_matches_the_webui_pooling():
    """Identical to the axis-by-axis `bin_any` the front end applies.

    The web front end pools the ground truth onto its display grid one
    axis at a time; `pool_truth` is the same operation expressed as a
    target shape, and the two must agree cell for cell or a mask drawn on
    screen and a mask scored in a test would mean different things.
    """
    rng = np.random.default_rng(5)
    truth = rng.random((3, 100, 63)) < 0.05
    expected = bin_any(truth, axis=2, n_bins=8)
    expected = bin_any(expected, axis=1, n_bins=7)
    np.testing.assert_array_equal(pool_truth(truth, (3, 7, 8)), expected)


def test_pool_truth_is_the_identity_when_the_shape_already_matches():
    """A matching target shape changes nothing -- and does not alias the input."""
    truth = np.zeros((2, 4), dtype=bool)
    truth[1, 2] = True
    pooled = pool_truth(truth, (2, 4))
    np.testing.assert_array_equal(pooled, truth)
    pooled[0, 0] = True
    assert not truth[0, 0]


def test_pool_truth_never_loses_a_contaminated_cell():
    """Pooling can only ever add contamination, never remove it.

    Checked by pooling a second time, from the coarse grid straight to a
    coarser one: ANY-pooling is transitive, so pooling in two steps and
    pooling in one must agree, and both must still see the single
    contaminated fine cell.
    """
    rng = np.random.default_rng(9)
    truth = rng.random((16, 90)) < 0.02
    assert truth.any()
    coarse = pool_truth(truth, (4, 9))
    assert coarse.any()
    assert coarse.sum() <= truth.sum()
    np.testing.assert_array_equal(pool_truth(coarse, (2, 3)), pool_truth(truth, (2, 3)))


def test_pool_truth_rejects_upsampling():
    """Labels can be coarsened, never invented."""
    truth = np.zeros((2, 4), dtype=bool)
    with pytest.raises(ValueError, match=r"shape\[1\] must be in"):
        pool_truth(truth, (2, 8))
    with pytest.raises(ValueError, match=r"shape\[0\] must be in"):
        pool_truth(truth, (0, 4))


def test_pool_truth_rejects_a_shape_of_the_wrong_rank():
    """A target shape must name every axis."""
    with pytest.raises(ValueError, match="2 axes"):
        pool_truth(np.zeros((2, 4), dtype=bool), (4,))


def test_pool_truth_rejects_a_non_boolean_mask():
    """Ground truth is a decision, so it is boolean."""
    with pytest.raises(ValueError, match="boolean mask"):
        pool_truth(np.zeros((2, 4)), (2, 2))
