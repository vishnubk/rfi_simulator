r"""Classical interference flaggers: reference detectors to beat.

Three published, non-learned detectors, implemented from first principles
with numpy alone. They are the yardstick any excision algorithm should be
measured against, and they are deliberately the *simple* forms of the
published methods -- an honest baseline is one whose behaviour can be
derived on paper, not one tuned until it wins.

Conventions shared by every flagger here
----------------------------------------
* **Masks are boolean, True = contaminated**, as everywhere in this
  package.
* **Flaggers see data, never labels.** They take plain arrays, not
  simulator objects, so there is no path by which a flagger could read
  `rfi_simulator.voltages.VoltageBlock.rfi_mask`. Scoring is somebody
  else's job (`rfi_simulator.metrics`).
* **Deterministic.** No generator is drawn from, so a mask is a pure
  function of the input array and the parameters.
* **Non-finite cells are always flagged.** NaN and ``+/-inf`` alike are
  read as missing data or as a broken upstream step; either way they are
  not clean, and leaving them unflagged would let them poison the
  statistics of whatever runs next. They are also excluded from each
  detector's own statistics rather than merely tolerated -- an infinity
  carried into a median or a running sum silently blinds a detector to
  the real interference sitting beside it.
* **The last axis is time, the second-to-last is frequency.** Leading
  axes (antenna, polarization, ...) are looped over implicitly by
  broadcasting: each is flagged independently.
* **Statistics are available.** Every flagger accepts
  ``return_statistic=True`` and then returns ``(mask, statistic)``, with
  the statistic on the same grid as the mask. Masks are thresholded
  decisions and throw away the ranking information a
  threshold-independent score (ROC, AUC) needs; the raw statistic keeps
  it.

Grids
-----
`mad_clip_mask` and `sumthreshold_mask` return a mask on the grid of
their input. `spectral_kurtosis_mask` accumulates over blocks of ``M``
time samples and therefore returns a mask of shape
``(..., n_chan, n_time // M)``: one decision per accumulation, not one
per sample, with the final ``n_time % M`` samples taking part in no
accumulation at all. Comparing that with per-sample ground truth needs
`rfi_simulator.metrics.pool_truth_accumulations`, which reproduces both
the fixed-block partition and the dropped tail --
`rfi_simulator.metrics.pool_truth` does neither and misaligns the labels
whenever ``M`` does not divide ``n_time``.

What each detector is sensitive to
----------------------------------
The three are complementary, which is the reason to ship all of them:

* Spectral kurtosis looks at the *shape* of the intensity distribution
  within an accumulation and is blind to its mean. It sees a signal that
  is too steady (a carrier) or too spiky (a burst) even when the extra
  power is small, but it cannot see interference that is itself
  Gaussian and constant.
* MAD clipping looks only at the mean power of a cell against that
  channel's own history. It sees any excess power regardless of its
  statistics, and is blind to a transmitter that is on for the whole
  block (the median absorbs it).
* SumThreshold looks at *runs* of mildly elevated cells and is the only
  one of the three that can pull out interference well below the
  single-cell detection threshold, provided it is contiguous in time or
  frequency.

Deliberate simplifications
--------------------------
Spectral-kurtosis thresholds use the asymptotic Gaussian approximation
to the estimator's distribution rather than the exact Pearson type IV
form, and the SumThreshold thresholds must be supplied by the caller in
units of the residual it is handed rather than derived from a noise
model. Both are documented per function.
"""

from __future__ import annotations

import warnings
from statistics import NormalDist

import numpy as np

__all__ = [
    "MAD_TO_SIGMA",
    "SUMTHRESHOLD_RHO",
    "mad_clip_mask",
    "spectral_kurtosis_mask",
    "sumthreshold_mask",
]

SUMTHRESHOLD_RHO = 1.5
"""float: Default threshold-decay base of `sumthreshold_mask`. The
threshold for a window of ``M`` cells is ``chi_1 / rho**log2(M)``; 1.5 is
the value used in the method's original description."""

MAD_TO_SIGMA = 1.4826
"""float: Scaling that turns a median absolute deviation into a standard
deviation for Gaussian data, i.e. ``1 / Phi^-1(3/4)``."""


def _two_sided_z(pfa: float) -> float:
    """Gaussian deviate whose two-sided tail probability is `pfa`.

    Parameters
    ----------
    pfa : float
        Total probability of false alarm, in ``(0, 1)``, split equally
        between the two tails.

    Returns
    -------
    float
        ``z`` such that ``P(|N(0, 1)| > z) == pfa``.
    """
    if not 0.0 < pfa < 1.0:
        raise ValueError(f"pfa must be in (0, 1), got {pfa}")
    return float(NormalDist().inv_cdf(1.0 - 0.5 * pfa))


def spectral_kurtosis_mask(
    voltages: np.ndarray,
    m: int,
    pfa: float = 0.0027,
    *,
    return_statistic: bool = False,
):
    r"""Generalized spectral kurtosis of channelized complex voltages.

    The estimator of Nita & Gary (2010) for complex channelized
    voltages: within each channel, over an accumulation of ``M``
    consecutive time samples,

    .. math::

        S_1 = \sum_{k=1}^{M} |x_k|^2, \qquad
        S_2 = \sum_{k=1}^{M} |x_k|^4, \qquad
        \widehat{SK} = \frac{M + 1}{M - 1}
                       \left(\frac{M S_2}{S_1^2} - 1\right).

    Note that :math:`S_1` and :math:`S_2` are built from the *squared
    modulus* of the complex sample, not from its real and imaginary parts
    separately: the statistic describes the distribution of the
    instantaneous power.

    For circular complex Gaussian noise the instantaneous power is
    exponentially distributed and the estimator is unbiased with
    :math:`E[\widehat{SK}] = 1`. A coherent carrier has constant power
    and drives the statistic towards 0; an impulsive burst that occupies
    a small fraction of the accumulation drives it above 1. Both
    directions are interference, so the test is two-sided -- that is the
    whole point of the statistic, and a one-sided version would miss
    every carrier.

    Parameters
    ----------
    voltages : numpy.ndarray
        Complex channelized voltages of shape ``(n_chan, n_time)`` or
        ``(n_ant, n_chan, n_time)``, in any amplitude unit (the statistic
        is scale-free). This is the *pre-detection* product: the
        information the statistic uses is destroyed by averaging power,
        so it cannot be recovered from a spectrogram.
    m : int
        Accumulation length ``M`` in time samples, ``>= 2``. The time
        axis is split into ``n_time // m`` consecutive accumulations.
    pfa : float, optional
        Nominal total probability of false alarm per accumulation for
        Gaussian noise, split equally between the low and high tails.
        Default 0.0027, the two-sided 3-sigma value. Nominal, not
        realized -- see Notes.
    return_statistic : bool, optional
        If True, also return the estimator values. Default False.

    Returns
    -------
    mask : numpy.ndarray
        Boolean array of shape ``(..., n_chan, n_time // m)``, True where
        the accumulation is flagged.
    statistic : numpy.ndarray, optional
        Returned only if `return_statistic`: float64 array of the same
        shape holding :math:`\widehat{SK}`, NaN where it is undefined
        (see Notes).

    Raises
    ------
    ValueError
        If `voltages` is not complex, is not 2- or 3-dimensional, if `m`
        is below 2 or exceeds the number of time samples, or if `pfa` is
        outside ``(0, 1)``.

    Notes
    -----
    **Thresholds.** The detection limits are
    ``1 +/- z * sqrt(4 / M)``, from the asymptotic variance
    :math:`\mathrm{Var}(\widehat{SK}) \approx 4/M` and a Gaussian
    approximation to the estimator's distribution. That approximation is
    the deliberate simplification here: the exact distribution is a
    Pearson type IV, right-skewed, approaching Gaussian only as
    :math:`1/\sqrt{M}`. The skew makes the true upper tail heavier than
    Gaussian and the true lower tail lighter, so the realized false-alarm
    rate **exceeds** `pfa`, and almost all of the excess sits in the
    upper tail. At the default `pfa` the realized rate on Gaussian noise
    is about 3.7 times nominal at ``M = 64`` and still about 1.4 times
    nominal at ``M = 1024``; the ratio is worse for tighter `pfa` and
    better for looser. At ``M = 1024`` and ``pfa = 0.1`` it is within a
    few per cent. So: treat the default as a *sensitivity* setting, and
    calibrate the threshold on noise-only data whenever the false-alarm
    rate itself matters.

    **Truncation, and pooling truth to match.** If ``m`` does not divide
    ``n_time``, the last ``n_time % m`` samples take part in no
    accumulation and are dropped. They are not flagged and not reported;
    a partial accumulation would have a different variance and a
    different threshold, which is a worse trade than losing under ``m``
    samples. Because of that, ground truth must be brought onto this grid
    with `rfi_simulator.metrics.pool_truth_accumulations`, which applies
    the identical fixed-block partition and drops the identical tail.
    Pooling with `rfi_simulator.metrics.pool_truth` onto ``n_time // m``
    bins is **wrong whenever ``m`` does not divide ``n_time``**: it
    spreads the remainder across every bin, so labels land beside the
    decisions they are scored against and a perfect mask can score a
    recall near 0.5.

    **Undefined cells.** An accumulation containing a non-finite sample,
    or one whose total power is exactly zero, has no defined statistic.
    Its `statistic` entry is NaN and it is flagged.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> noise = (rng.standard_normal((4, 2048)) + 1j * rng.standard_normal((4, 2048))) / np.sqrt(2)
    >>> mask, sk = spectral_kurtosis_mask(noise, m=256, return_statistic=True)
    >>> mask.shape
    (4, 8)
    >>> bool(abs(sk.mean() - 1.0) < 0.1)
    True
    """
    values = np.asarray(voltages)
    if not np.iscomplexobj(values):
        raise ValueError(
            "spectral_kurtosis_mask needs complex channelized voltages; got dtype "
            f"{values.dtype}. Pass the pre-detection voltages, not a power spectrogram."
        )
    if values.ndim not in (2, 3):
        raise ValueError(
            "voltages must have shape (n_chan, n_time) or (n_ant, n_chan, n_time), "
            f"got shape {values.shape}"
        )
    if int(m) != m:
        raise ValueError(
            f"m must be a whole number of time samples, got {m!r}. Truncating it "
            "silently would change the accumulation grid the mask is defined on."
        )
    m = int(m)
    if m < 2:
        raise ValueError(f"m must be >= 2 (the estimator divides by M - 1), got {m}")
    n_time = values.shape[-1]
    if m > n_time:
        raise ValueError(f"m must be <= the number of time samples ({n_time}), got {m}")
    z = _two_sided_z(float(pfa))

    n_accum = n_time // m
    trimmed = values[..., : n_accum * m]
    # The squared modulus, in float64: S1 and S2 are sums of |x|**2 and
    # |x|**4, never of the real and imaginary parts taken separately. The
    # separate-parts version is the classic implementation bug -- it
    # halves the effective sample count and shifts the noise mean of the
    # estimator from 1 to 2.
    real = trimmed.real.astype(np.float64)
    imag = trimmed.imag.astype(np.float64)
    power = real * real + imag * imag
    power = power.reshape(*values.shape[:-1], n_accum, m)

    s1 = power.sum(axis=-1)
    s2 = np.square(power).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        statistic = ((m + 1.0) / (m - 1.0)) * (m * s2 / np.square(s1) - 1.0)
    statistic = np.where(s1 > 0.0, statistic, np.nan)

    half_width = z * np.sqrt(4.0 / m)
    with np.errstate(invalid="ignore"):
        mask = (statistic < 1.0 - half_width) | (statistic > 1.0 + half_width)
    mask |= ~np.isfinite(statistic)

    if return_statistic:
        return mask, statistic
    return mask


def mad_clip_mask(
    spectrogram: np.ndarray,
    n_sigma: float = 5.0,
    *,
    return_statistic: bool = False,
):
    """Per-channel robust sigma clipping of a spectrogram.

    For each channel independently, the median and the median absolute
    deviation (MAD) are taken over time, the MAD is scaled by
    `MAD_TO_SIGMA` to a Gaussian-equivalent standard deviation, and cells
    further than `n_sigma` from the median in either direction are
    flagged.

    Statistics are per channel, never global: a receiver's bandpass and
    the sky's own spectral structure make the mean level a strong
    function of frequency, and a global threshold would flag the loud end
    of the band wholesale. Median and MAD rather than mean and standard
    deviation, because the contamination this is meant to find would
    otherwise inflate the very scale it is being compared against.

    Parameters
    ----------
    spectrogram : numpy.ndarray
        Real array of shape ``(..., n_chan, n_time)``, typically the
        detected power ``|v|**2`` in Jy. Any real-valued per-cell
        quantity works; the statistic is dimensionless.
    n_sigma : float, optional
        Threshold in Gaussian-equivalent standard deviations, ``> 0``.
        Default 5.0.
    return_statistic : bool, optional
        If True, also return the signed deviations. Default False.

    Returns
    -------
    mask : numpy.ndarray
        Boolean array of the same shape as `spectrogram`.
    statistic : numpy.ndarray, optional
        Returned only if `return_statistic`: float64 array of the same
        shape holding the signed deviation in Gaussian-equivalent sigmas,
        ``(x - median) / (1.4826 * MAD)``. NaN where the input was not
        finite, and ``+/-inf`` in a channel with no scale (see Notes).
        This array is a natural input to `sumthreshold_mask`, which wants
        a background-subtracted, noise-normalized residual and which
        treats both of those as missing data.

    Raises
    ------
    ValueError
        If `spectrogram` is complex, has fewer than two dimensions, or if
        `n_sigma` is not positive.

    Notes
    -----
    **The n-sigma is nominal, not calibrated.** Detected power is
    :math:`\\chi^2` distributed -- exponential for a single complex
    sample, and only slowly Gaussian as samples are averaged -- so it is
    right-skewed, and the realized false-alarm rate of an ``n``-sigma cut
    is not the Gaussian ``2 Q(n)``. On single-sample exponential power
    the upper tail dominates completely and the realized rate at 5 sigma
    is of order a percent, several orders of magnitude above the Gaussian
    value. Treat `n_sigma` as a knob to be calibrated on clean data for
    the specific product being flagged, not as a false-alarm rate.

    **Degenerate channels.** A channel whose MAD is zero (constant, or
    more than half its samples identical) has no usable scale. Its
    deviation is reported as 0 where the cell equals the median and
    ``+/-inf`` elsewhere, so an otherwise-constant channel with a single
    outlier still flags the outlier and nothing else.

    **Missing data.** Every non-finite input cell -- NaN *and* ``+/-inf``
    -- is treated as missing: excluded from the channel's median and MAD,
    given a NaN deviation, and flagged. Infinities have to be excluded
    rather than carried through, because a channel with half its cells at
    ``+inf`` would otherwise take an infinite median, produce an all-NaN
    deviation, and flag nothing at all -- hiding any genuine outlier
    sharing the channel. A channel that is entirely missing is flagged
    wholesale.

    Examples
    --------
    >>> import numpy as np
    >>> power = np.ones((2, 8))
    >>> power[0, 3] = 50.0
    >>> mad_clip_mask(power)[0]
    array([False, False, False,  True, False, False, False, False])
    """
    values = np.asarray(spectrogram)
    if np.iscomplexobj(values):
        raise ValueError(
            "mad_clip_mask needs a real spectrogram; got dtype "
            f"{values.dtype}. Pass detected power, e.g. numpy.abs(v) ** 2."
        )
    if values.ndim < 2:
        raise ValueError(
            f"spectrogram must have shape (..., n_chan, n_time), got shape {values.shape}"
        )
    n_sigma = float(n_sigma)
    if not n_sigma > 0.0:
        raise ValueError(f"n_sigma must be > 0, got {n_sigma}")

    values = values.astype(np.float64)
    # Every non-finite cell is missing data, infinities included. Leaving
    # an infinity in place would put it into the median: a channel with
    # half its cells at +inf gets an infinite median, an all-NaN
    # deviation, and nothing flagged at all -- not even a co-channel
    # outlier of 1e9. Excluding them from the statistics and flagging
    # them outright is the only reading that cannot hide a neighbour.
    is_missing = ~np.isfinite(values)
    finite = np.where(is_missing, np.nan, values)

    # An all-missing channel makes nanmedian warn about an empty slice.
    # That is a defined, intended outcome here (the channel is flagged
    # wholesale), so the warning is suppressed narrowly rather than left
    # to crash a warnings-as-errors pipeline.
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        median = np.nanmedian(finite, axis=-1, keepdims=True)
        mad = np.nanmedian(np.abs(finite - median), axis=-1, keepdims=True)
        scale = MAD_TO_SIGMA * mad

        residual = finite - median
        # A channel with zero MAD has no scale: a cell that sits on the
        # median is zero deviations away, anything else is infinitely far.
        degenerate = np.where(residual > 0.0, np.inf, np.where(residual < 0.0, -np.inf, 0.0))
        with np.errstate(divide="ignore"):
            statistic = np.where(scale > 0.0, residual / scale, degenerate)
        statistic = np.where(is_missing, np.nan, statistic)
        mask = np.abs(statistic) > n_sigma
    mask |= is_missing

    if return_statistic:
        return mask, statistic
    return mask


def _sumthreshold_axis(
    values: np.ndarray, mask: np.ndarray, window: int, threshold: float, axis: int
) -> np.ndarray:
    """One SumThreshold pass of a single window size along one axis.

    Parameters
    ----------
    values : numpy.ndarray
        Float64 residual array.
    mask : numpy.ndarray
        Boolean mask accumulated so far, same shape as `values`.
    window : int
        Window length in cells, ``>= 1``.
    threshold : float
        Per-cell threshold ``chi_M``; a window is flagged when its sum
        exceeds ``window * threshold``.
    axis : int
        Axis to slide the window along.

    Returns
    -------
    numpy.ndarray
        Boolean mask, `mask` unioned with the cells this pass flagged.

    Notes
    -----
    Cells already flagged enter the sum **at exactly the threshold
    value**, rather than being excluded from it. That is the detail that
    makes the method work: a flagged cell then contributes neither
    evidence for nor against its neighbours, so one very bright sample
    cannot drag a whole wide window over the line, while a run of
    genuinely elevated cells still accumulates. Excluding them instead
    (and comparing the mean of what is left) is a different, more
    permissive algorithm.

    New flags are accumulated against the mask as it stood at the start
    of the pass, so the result does not depend on the order the windows
    are visited.
    """
    work = np.moveaxis(values, axis, -1)
    flags = np.moveaxis(mask, axis, -1)
    length = work.shape[-1]
    if window > length:
        return mask

    substituted = np.where(flags, threshold, work)
    zeros = np.zeros(substituted.shape[:-1] + (1,), dtype=np.float64)
    cumulative = np.concatenate([zeros, np.cumsum(substituted, axis=-1)], axis=-1)
    sums = cumulative[..., window:] - cumulative[..., :-window]
    exceeds = sums > window * threshold

    # A cell is flagged if any window *starting* in [j - window + 1, j]
    # exceeded, so count starts inside that trailing range.
    starts = np.zeros(work.shape, dtype=np.int64)
    starts[..., : length - window + 1] = exceeds
    counts = np.concatenate(
        [np.zeros(starts.shape[:-1] + (1,), dtype=np.int64), np.cumsum(starts, axis=-1)],
        axis=-1,
    )
    index = np.arange(length)
    covered = counts[..., index + 1] - counts[..., np.maximum(index - window + 1, 0)] > 0
    return np.moveaxis(flags | covered, -1, axis)


def sumthreshold_mask(
    residual: np.ndarray,
    chi_1: float = 6.0,
    iterations: int = 5,
    *,
    rho: float = SUMTHRESHOLD_RHO,
    return_statistic: bool = False,
):
    r"""SumThreshold flagging of a time-frequency residual.

    The combinatorial thresholding of Offringa et al. (2010), written
    from the published description. Successive passes use window sizes
    ``M = 1, 2, 4, ..., 2**(iterations - 1)``. A run of ``M`` consecutive
    cells is flagged when its sum exceeds :math:`M \chi_M`, with

    .. math::

        \chi_M = \frac{\chi_1}{\rho^{\log_2 M}}, \qquad \rho = 1.5.

    The per-cell threshold therefore falls as the window widens, which is
    what lets interference too faint to trip a single-cell cut be found
    once it is seen as a contiguous run. Each window size is applied
    first along time and then along frequency, so a run in either
    direction is caught.

    Parameters
    ----------
    residual : numpy.ndarray
        Real array of shape ``(..., n_chan, n_time)`` holding a
        **background-subtracted, noise-normalized** residual: the method
        thresholds the values it is given directly, so a raw spectrogram
        with a bandpass in it would flag the loud half of the band at
        window 1. The signed deviation array from
        ``mad_clip_mask(..., return_statistic=True)`` is the intended
        input, in which case `chi_1` is in units of sigma.
    chi_1 : float, optional
        Single-cell threshold, in the units of `residual`. Default 6.0.
    iterations : int, optional
        Number of window sizes, ``>= 1``. Default 5, i.e. windows up to
        16 cells.
    rho : float, optional
        Base of the threshold decay. Default `SUMTHRESHOLD_RHO`.
    return_statistic : bool, optional
        If True, also return the residual actually thresholded (the input
        as float64, non-finite cells preserved). Default False. For
        API symmetry with the other flaggers: unlike them, this method
        emits no per-cell score of its own, only a decision.

    Returns
    -------
    mask : numpy.ndarray
        Boolean array of the same shape as `residual`.
    statistic : numpy.ndarray, optional
        Returned only if `return_statistic`.

    Raises
    ------
    ValueError
        If `residual` is complex, has fewer than two dimensions, if
        `chi_1` is not positive, if `iterations` is below 1, or if `rho`
        is not greater than 1 (``rho <= 1`` would make wide windows no
        more sensitive than narrow ones, or infinitely sensitive).

    Notes
    -----
    **One-sided.** Only positive excursions are flagged, since the method
    is aimed at *added* power. Pass ``numpy.abs(residual)`` for a
    two-sided test.

    **False-alarm rate.** Only the first pass has an analytic rate: with
    ``iterations=1`` and a standard-normal residual, exactly the cells
    above `chi_1` are flagged, a fraction ``Q(chi_1)``. Every further
    window size lowers the per-cell threshold by another factor of `rho`
    while asking for a longer run, and the two do not cancel: the
    realized rate rises **steeply** with `iterations`. On standard-normal
    noise at ``chi_1 = 4`` it goes from ``3e-5`` at one window to roughly
    ``1e-2`` at five, a factor of a few hundred. There is no closed form
    -- the passes are not independent, since a flagged cell enters later
    sums at the threshold value -- so `chi_1` and `iterations` must be
    calibrated together on clean data. Raising `chi_1` is the cheaper
    knob: at ``chi_1 = 6`` and five windows the rate is back near
    ``3e-5``.

    **Missing data.** Every non-finite cell -- NaN and ``+/-inf`` alike --
    is flagged before the first pass and then enters every sum at the
    threshold value, so it neither hides nor manufactures flags in its
    neighbours. Infinities must be handled and not merely tolerated: one
    ``-inf`` left in the values would drive the running sum of every
    window containing it to ``-inf``, and since the test is one-sided
    that window then flags nothing, blinding the method to real
    interference next door. `mad_clip_mask` emits ``-inf`` for
    below-median cells of a channel with no scale, so the recommended
    input can contain them.

    Examples
    --------
    >>> import numpy as np
    >>> residual = np.zeros((4, 16))
    >>> residual[1, 4:12] = 2.5  # a faint run: no single cell reaches chi_1
    >>> mask = sumthreshold_mask(residual, chi_1=6.0, iterations=5)
    >>> bool(mask[1, 4:12].all()), bool(mask[0].any())
    (True, False)
    """
    values = np.asarray(residual)
    if np.iscomplexobj(values):
        raise ValueError(f"sumthreshold_mask needs a real residual; got dtype {values.dtype}.")
    if values.ndim < 2:
        raise ValueError(
            f"residual must have shape (..., n_chan, n_time), got shape {values.shape}"
        )
    iterations = int(iterations)
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    rho = float(rho)
    if not rho > 1.0:
        raise ValueError(f"rho must be > 1, got {rho}")
    chi_1 = float(chi_1)
    if not chi_1 > 0.0:
        raise ValueError(
            f"chi_1 must be > 0, got {chi_1}. A non-positive threshold flags every "
            "cell whose run sums to zero, i.e. essentially everything."
        )

    values = values.astype(np.float64)
    # Non-finite cells are flagged up front and neutralized in the working
    # copy. Both halves matter. A single -inf left in the values poisons
    # the running sums of every window that contains it -- they go to
    # -inf or NaN, compare False, and suppress detection of genuine
    # interference beside it -- and the one-sided test would never flag
    # the -inf cell itself. Infinities are not hypothetical here: the
    # recommended input, the deviation array of `mad_clip_mask`, emits
    # them for a channel with no scale.
    mask = ~np.isfinite(values)
    clean = np.where(mask, 0.0, values)

    for step in range(iterations):
        window = 2**step
        chi_m = chi_1 / rho**step
        mask = _sumthreshold_axis(clean, mask, window, chi_m, axis=-1)
        mask = _sumthreshold_axis(clean, mask, window, chi_m, axis=-2)

    if return_statistic:
        return mask, values
    return mask
