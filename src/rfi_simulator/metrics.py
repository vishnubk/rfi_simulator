"""Scoring a predicted flag mask against ground truth.

This module answers one question: given a boolean mask a flagger
produced and the boolean mask the simulator knows to be true, how good
was the flagger? Everything here is cell-counting -- no physics, no
weighting by how much power a cell actually held. A cell is right or it
is wrong.

Conventions
-----------
* **Masks are boolean, True = contaminated.** "Positive" therefore means
  "flagged as contaminated", so a false positive is a clean cell that was
  flagged away, and a false negative is interference that survived into
  the science data. The two are not equally expensive and no single score
  here pretends otherwise: report `flag_scores` as a whole.
* **Undefined ratios are NaN, not zero.** Every score here is a ratio,
  and each has an input for which its denominator vanishes -- no cells
  were flagged, or the truth is entirely clean. Returning 0.0 in those
  cases would say "the flagger scored badly"; the honest answer is that
  the quantity is not defined, and NaN says so and refuses to be averaged
  into a summary by accident. The one exception is the Matthews
  correlation coefficient, which is 0 by convention when its denominator
  vanishes, matching the usual definition: a degenerate predictor has no
  correlation with the truth.
* **Grids must match.** `confusion_counts` compares cell for cell and
  refuses mismatched shapes rather than broadcasting, since a
  broadcastable mismatch (say ``(n_chan, 1)`` against
  ``(n_chan, n_time)``) is always a bug. When a flagger ran on a coarser
  grid than the labels, bring the truth to the flagger's grid first --
  with `pool_truth_accumulations` for `spectral_kurtosis_mask`, and with
  `pool_truth` for a display or any other fit-to-count grid. The two
  partition an axis differently and are not interchangeable; picking the
  wrong one costs real score rather than raising.

Why the Matthews correlation coefficient
----------------------------------------
Interference occupancy is usually a small fraction of a data set, so
accuracy is useless (flag nothing, score 99 %) and even F1 ignores the
true negatives entirely. MCC uses all four cells of the confusion matrix
and stays near zero for any degenerate strategy -- flag everything, flag
nothing, flag at random -- which makes it the one number to look at first
when comparing flaggers on data whose occupancy varies.
"""

from __future__ import annotations

import numpy as np

from rfi_simulator.binning import bin_any, block_any

__all__ = ["confusion_counts", "flag_scores", "pool_truth", "pool_truth_accumulations"]


def _as_mask(array: np.ndarray, name: str) -> np.ndarray:
    """Validate and normalize one mask argument.

    Parameters
    ----------
    array : numpy.ndarray
        Candidate mask.
    name : str
        Argument name, for the error message.

    Returns
    -------
    numpy.ndarray
        The mask as a boolean array.

    Raises
    ------
    ValueError
        If the array is not of boolean dtype. Integer or float masks are
        rejected on purpose: silently treating "any nonzero value" as
        contaminated would quietly accept a probability map or a
        statistic array, and score it as if it were a decision.
    """
    values = np.asarray(array)
    if values.dtype != np.bool_:
        raise ValueError(
            f"{name} must be a boolean mask (True = contaminated), got dtype "
            f"{values.dtype}. Threshold it into a decision before scoring."
        )
    return values


def confusion_counts(predicted: np.ndarray, truth: np.ndarray) -> tuple[int, int, int, int]:
    """Count the four outcomes of a predicted mask against ground truth.

    Parameters
    ----------
    predicted : numpy.ndarray
        Boolean mask a flagger produced, True = flagged as contaminated.
    truth : numpy.ndarray
        Boolean ground-truth mask of the same shape, True = actually
        contaminated.

    Returns
    -------
    tp : int
        Contaminated cells that were flagged.
    fp : int
        Clean cells that were flagged (data thrown away for nothing).
    fn : int
        Contaminated cells that were missed (interference left in).
    tn : int
        Clean cells that were left alone.

    Raises
    ------
    ValueError
        If either argument is not boolean, or if the shapes differ. Use
        `pool_truth` when the flagger ran on a coarser grid than the
        labels.

    Examples
    --------
    >>> import numpy as np
    >>> truth = np.array([[True, False], [True, False]])
    >>> predicted = np.array([[True, True], [False, False]])
    >>> confusion_counts(predicted, truth)
    (1, 1, 1, 1)
    """
    predicted = _as_mask(predicted, "predicted")
    truth = _as_mask(truth, "truth")
    if predicted.shape != truth.shape:
        raise ValueError(
            f"predicted and truth must have the same shape, got {predicted.shape} "
            f"and {truth.shape}. Pool the truth onto the flagger's grid with pool_truth."
        )
    tp = int(np.count_nonzero(predicted & truth))
    fp = int(np.count_nonzero(predicted & ~truth))
    fn = int(np.count_nonzero(~predicted & truth))
    tn = int(np.count_nonzero(~predicted & ~truth))
    return tp, fp, fn, tn


def _ratio(numerator: float, denominator: float) -> float:
    """A ratio, or NaN when it is undefined.

    Parameters
    ----------
    numerator, denominator : float
        Counts.

    Returns
    -------
    float
        ``numerator / denominator``, or NaN if `denominator` is zero.
    """
    if denominator == 0.0:
        return float("nan")
    return float(numerator) / float(denominator)


def flag_scores(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Standard classification scores for a predicted flag mask.

    Parameters
    ----------
    predicted : numpy.ndarray
        Boolean mask a flagger produced, True = flagged as contaminated.
    truth : numpy.ndarray
        Boolean ground-truth mask of the same shape.

    Returns
    -------
    dict of str to float
        Eleven entries, all floats:

        ``tp``, ``fp``, ``fn``, ``tn``
            The raw counts from `confusion_counts`, so a caller can
            re-derive any score not listed here.
        ``precision``
            ``tp / (tp + fp)``: the fraction of flagged cells that were
            really contaminated. NaN when nothing was flagged.
        ``recall``
            ``tp / (tp + fn)``: the fraction of contaminated cells that
            were caught. NaN when the truth holds no contamination.
        ``f1``
            ``2 tp / (2 tp + fp + fn)``, the harmonic mean of precision
            and recall. NaN only when both are undefined, i.e. clean
            truth and empty prediction. 0.0 when the two masks are
            disjoint and both non-empty.
        ``mcc``
            Matthews correlation coefficient in ``[-1, 1]``: 1 for a
            perfect mask, 0 for chance, negative for anti-correlation.
            Exactly 0.0 when its denominator vanishes, which happens for
            any constant prediction (flag everything, flag nothing) or
            any constant truth.
        ``false_positive_rate``
            ``fp / (fp + tn)``: the fraction of *clean* cells thrown
            away. This is the cost side of the trade-off and the quantity
            two flaggers must be equalized on before their recalls are
            compared. NaN when the truth is entirely contaminated.
        ``truth_occupancy``
            ``(tp + fn) / n_cells``: the fraction of cells that really
            were contaminated. Without it precision is uninterpretable --
            a precision of 0.5 is excellent at 1 % occupancy and terrible
            at 90 %.
        ``predicted_occupancy``
            ``(tp + fp) / n_cells``: the fraction of cells flagged, i.e.
            how much data the flagger cost.

    Raises
    ------
    ValueError
        If either argument is not boolean, if the shapes differ, or if
        the masks are empty (no cells to score).

    Notes
    -----
    The MCC is computed as

    ``(tp * tn - fp * fn) / sqrt((tp+fp)(tp+fn)(tn+fp)(tn+fn))``

    in float64. The products are formed from Python ints before the cast,
    so the numerator is exact for any array that fits in memory.

    Examples
    --------
    >>> import numpy as np
    >>> truth = np.array([[True, False], [True, False]])
    >>> scores = flag_scores(truth.copy(), truth)
    >>> scores["precision"], scores["recall"], scores["mcc"]
    (1.0, 1.0, 1.0)
    >>> clean = np.zeros((2, 2), dtype=bool)
    >>> scores = flag_scores(clean, clean)
    >>> scores["precision"], scores["recall"], scores["mcc"]
    (nan, nan, 0.0)
    """
    tp, fp, fn, tn = confusion_counts(predicted, truth)
    n_cells = tp + fp + fn + tn
    if n_cells == 0:
        raise ValueError("cannot score empty masks: there are no cells to compare")

    numerator = tp * tn - fp * fn
    denominator = float((tp + fp)) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = 0.0 if denominator == 0.0 else float(numerator) / float(np.sqrt(denominator))

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "mcc": mcc,
        "false_positive_rate": _ratio(fp, fp + tn),
        "truth_occupancy": _ratio(tp + fn, n_cells),
        "predicted_occupancy": _ratio(tp + fp, n_cells),
    }


def pool_truth(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Pool a fine-grained truth mask onto a coarser fit-to-count grid.

    Ground truth is labelled at the resolution the data was simulated at,
    one cell per channel per time sample. A display, or any consumer that
    has a fixed number of cells to fill, is coarser. This brings the
    labels onto that grid so the two can be compared cell for cell.

    .. warning::

       This is **not** the right pooling for `spectral_kurtosis_mask`.
       The target shape here is a bin *count*, and the remainder of a
       ragged axis is spread across all the bins; a spectral-kurtosis mask
       is decided on fixed blocks of ``M`` samples with the tail dropped.
       The two partitions coincide only when ``M`` divides the time axis
       exactly, and otherwise labels land in the wrong bin -- which shows
       up as a mediocre score, not as an error. Use
       `pool_truth_accumulations` for that pairing.

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean ground-truth mask, e.g. one plane of
        `rfi_simulator.voltages.VoltageBlock.rfi_mask`.
    shape : tuple of int
        Target shape, with the same number of axes as `mask`. Each entry
        must be at least 1 and at most the corresponding entry of
        ``mask.shape``; an axis that already matches is left alone.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape `shape`, pooled with
        `rfi_simulator.binning.bin_any`: a coarse cell is contaminated if
        *any* fine cell inside it was.

    Raises
    ------
    ValueError
        If `mask` is not boolean, if `shape` has a different number of
        axes, or if any target length is not in ``[1, mask.shape[axis]]``
        -- upsampling labels would invent structure that was never
        simulated.

    Notes
    -----
    ANY-pooling is deliberately generous to the flagger's *recall* and
    harsh on its *precision*: a coarse cell containing one contaminated
    fine cell out of a hundred is labelled contaminated outright, so
    catching it counts as a full hit while leaving it counts as a full
    miss. That is the conservative choice for excision -- a partly
    contaminated cell is not clean -- but it means scores from different
    grids are not comparable with each other, only within a grid.

    Examples
    --------
    >>> import numpy as np
    >>> truth = np.zeros((2, 8), dtype=bool)
    >>> truth[0, 5] = True
    >>> pool_truth(truth, (2, 4))
    array([[False, False,  True, False],
           [False, False, False, False]])
    """
    values = _as_mask(mask, "mask")
    shape = tuple(int(length) for length in shape)
    if len(shape) != values.ndim:
        raise ValueError(
            f"shape must have {values.ndim} axes to match mask.shape {values.shape}, got {shape}"
        )
    for axis, length in enumerate(shape):
        if not 1 <= length <= values.shape[axis]:
            raise ValueError(
                f"shape[{axis}] must be in [1, {values.shape[axis]}] (pooling only "
                f"coarsens), got {length} for mask.shape {values.shape}"
            )
    for axis, length in enumerate(shape):
        values = bin_any(values, axis=axis, n_bins=length)
    return np.array(values, dtype=bool)


def pool_truth_accumulations(mask: np.ndarray, m: int) -> np.ndarray:
    """Pool a truth mask onto an accumulating flagger's time grid.

    The counterpart of `rfi_simulator.flaggers.spectral_kurtosis_mask`,
    which reads time samples in fixed blocks ``[k*m, (k+1)*m)`` and
    discards the final ``n_time % m`` samples. This applies exactly the
    same partition to the labels, so the two arrays line up cell for cell
    whatever ``m`` and ``n_time`` are.

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean ground-truth mask of shape ``(..., n_chan, n_time)``, e.g.
        ``block.rfi_mask.any(axis=0)``. Only the last axis is pooled;
        every other axis is passed through, since spectral kurtosis
        coarsens time and nothing else.
    m : int
        Accumulation length in time samples -- the same ``m`` the flagger
        was given.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(..., n_chan, n_time // m)``: an
        accumulation is contaminated if *any* of its ``m`` samples was.

    Raises
    ------
    ValueError
        If `mask` is not boolean, or if `m` is below 1 or longer than the
        time axis.

    Notes
    -----
    Two failure modes this exists to prevent, both of which score as a
    half-working flagger rather than raising:

    * Pooling with `pool_truth` onto ``n_time // m`` bins spreads the
      remainder across every bin, so on a ragged axis a burst inside
      accumulation ``k`` can be labelled into bins ``k`` and ``k - 1``.
      A flagger that got every accumulation right then scores a recall
      of about 0.5.
    * Keeping the truncated tail charges the flagger a false negative for
      samples that took part in no accumulation, i.e. for a decision it
      was never asked to make.

    Examples
    --------
    >>> import numpy as np
    >>> from rfi_simulator import spectral_kurtosis_mask
    >>> truth = np.zeros((1, 10), dtype=bool)
    >>> truth[0, 5] = True   # inside accumulation 1
    >>> truth[0, 9] = True   # inside the dropped tail
    >>> pool_truth_accumulations(truth, m=4)
    array([[False,  True]])
    """
    values = _as_mask(mask, "mask")
    if values.ndim < 1:
        raise ValueError("mask must have at least one axis (time is the last)")
    return np.array(block_any(values, int(m), axis=-1), dtype=bool)
