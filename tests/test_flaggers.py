"""Tests for rfi_simulator.flaggers.

The tests that carry the most weight here are:

* `test_spectral_kurtosis_is_unity_for_gaussian_noise` -- the estimator's
  noise mean is 1 only if ``S1`` and ``S2`` are built from the squared
  *modulus* of the complex sample. Building them from the real and
  imaginary parts separately is the classic implementation error and
  moves the noise mean to 2, so this test is the one that catches it.
* `test_sumthreshold_does_not_spread_a_single_bright_sample` -- the other
  classic error is letting an already-flagged cell keep its own value in
  later window sums. One very bright sample would then drag every window
  containing it over the line and smear the flag across the row. Flagged
  cells must enter later sums at exactly the threshold value.
* `test_spectral_kurtosis_beats_mad_clipping_on_low_duty_bursts` -- the
  reason all three detectors are here: at a matched false-positive
  budget, the pre-detection statistic recovers bursts that per-channel
  clipping of the accumulated power cannot see.

Statistical tests are seeded and their tolerances are stated in units of
the binomial standard error of the count being measured, so a failure
means the behaviour changed rather than that the dice came up badly.
"""

import math
import warnings
from statistics import NormalDist

import numpy as np
import pytest
from conftest import START_TIME, random_flat_array, zenith_phase_center

from rfi_simulator import (
    NarrowbandTransmitter,
    VoltageSimulator,
    bin_mean,
    flag_scores,
    mad_clip_mask,
    pool_truth_accumulations,
    spectral_kurtosis_mask,
    sumthreshold_mask,
)
from rfi_simulator.flaggers import MAD_TO_SIGMA


def gaussian_voltages(rng, shape):
    """Unit-power circular complex Gaussian voltages, complex64."""
    parts = rng.standard_normal(size=(*shape, 2), dtype=np.float32)
    parts *= np.float32(1.0 / np.sqrt(2.0))
    return parts.view(np.complex64)[..., 0]


def binomial_sigma(rate, n_trials):
    """Standard error of an observed fraction of `n_trials` at true `rate`."""
    return math.sqrt(rate * (1.0 - rate) / n_trials)


def upper_tail(z):
    """``P(N(0, 1) > z)``."""
    return 1.0 - NormalDist().cdf(z)


# ----------------------------------------------------------------------
# Spectral kurtosis: the statistic itself
# ----------------------------------------------------------------------
def test_spectral_kurtosis_is_unity_for_gaussian_noise():
    """The estimator is unbiased at 1 with variance 4/M on pure noise.

    Both numbers are the documented basis of the thresholds. The mean is
    also the guard against the squared-modulus bug: with ``S1`` and
    ``S2`` accumulated from the real and imaginary parts separately, the
    per-sample power is chi-squared with one degree of freedom instead of
    two, ``E[p^2] / E[p]^2`` becomes 3 rather than 2, and the estimator
    would sit at 2.
    """
    rng = np.random.default_rng(101)
    m = 512
    voltages = gaussian_voltages(rng, (64, 512 * m))
    _, sk = spectral_kurtosis_mask(voltages, m=m, return_statistic=True)
    assert sk.size == 64 * 512
    # Standard error of the mean is sqrt(4 / M / N).
    assert abs(sk.mean() - 1.0) < 4.0 * math.sqrt(4.0 / m / sk.size)
    assert sk.var() == pytest.approx(4.0 / m, rel=0.1)


def test_spectral_kurtosis_is_invariant_to_the_voltage_scale():
    """The statistic is scale-free, so a bandpass cannot bias it.

    This is the structural advantage over `mad_clip_mask`: no per-channel
    normalization is needed because multiplying a channel by any constant
    leaves ``M S2 / S1^2`` unchanged.
    """
    rng = np.random.default_rng(102)
    voltages = gaussian_voltages(rng, (4, 4096)).astype(np.complex128)
    gains = np.array([1.0, 10.0, 100.0, 0.01])[:, np.newaxis]
    _, plain = spectral_kurtosis_mask(voltages, m=256, return_statistic=True)
    _, scaled = spectral_kurtosis_mask(voltages * gains, m=256, return_statistic=True)
    np.testing.assert_allclose(scaled, plain, rtol=1e-9)


@pytest.mark.parametrize("cycles_per_sample", [0.0, 0.1, 0.37])
def test_spectral_kurtosis_collapses_for_a_coherent_tone(cycles_per_sample):
    """A constant-modulus carrier drives the statistic to 0, not to 1.

    A tone has no power fluctuation at all, so ``M S2 = S1^2`` exactly and
    the estimator is zero regardless of the tone's frequency or
    amplitude. This is interference that raises no eyebrows in the mean
    power of a long accumulation but is unmistakable here -- and it is
    why the test has to be two-sided.
    """
    m = 256
    phase = 2j * np.pi * cycles_per_sample * np.arange(4 * m)
    tone = np.exp(phase).astype(np.complex64)[np.newaxis, :]
    mask, sk = spectral_kurtosis_mask(tone, m=m, return_statistic=True)
    np.testing.assert_allclose(sk, 0.0, atol=1e-6)
    assert mask.all()


def test_spectral_kurtosis_rises_for_impulsive_bursts():
    """A burst filling a small fraction of the accumulation pushes SK above 1.

    Hand-computable: with one sample of power ``P`` among ``M - 1``
    samples of power 1, ``M S2 / S1^2`` tends to ``M P^2 / (P + M - 1)^2``,
    which is far above 2 as soon as ``P`` is a sizeable fraction of ``M``.
    """
    m = 64
    power = np.ones(m)
    power[7] = 400.0
    voltages = np.sqrt(power).astype(np.complex64)[np.newaxis, :]
    _, sk = spectral_kurtosis_mask(voltages, m=m, return_statistic=True)
    s1 = power.sum()
    s2 = np.square(power).sum()
    expected = ((m + 1) / (m - 1)) * (m * s2 / s1**2 - 1.0)
    assert sk[0, 0] == pytest.approx(expected)
    assert sk[0, 0] > 10.0


def test_spectral_kurtosis_truncates_a_ragged_time_axis():
    """Samples that do not fill a whole accumulation are dropped, not padded.

    A partial accumulation would have a different variance and therefore
    a different threshold, so it is discarded and the result is exactly
    what the truncated array would have given.
    """
    rng = np.random.default_rng(103)
    voltages = gaussian_voltages(rng, (3, 1000))
    mask, sk = spectral_kurtosis_mask(voltages, m=64, return_statistic=True)
    assert mask.shape == (3, 15)  # 1000 // 64 == 15, the last 40 samples go
    _, trimmed = spectral_kurtosis_mask(voltages[:, : 15 * 64], m=64, return_statistic=True)
    np.testing.assert_array_equal(sk, trimmed)


def test_spectral_kurtosis_accepts_a_per_antenna_cube():
    """A 3-D input is flagged antenna by antenna, independently."""
    rng = np.random.default_rng(104)
    voltages = gaussian_voltages(rng, (5, 8, 1024))
    mask, sk = spectral_kurtosis_mask(voltages, m=256, return_statistic=True)
    assert mask.shape == (5, 8, 4)
    for antenna in range(5):
        _, single = spectral_kurtosis_mask(voltages[antenna], m=256, return_statistic=True)
        np.testing.assert_array_equal(sk[antenna], single)


def test_spectral_kurtosis_flags_undefined_accumulations():
    """Non-finite samples and zero-power accumulations are flagged, statistic NaN."""
    rng = np.random.default_rng(105)
    voltages = gaussian_voltages(rng, (4, 128)).astype(np.complex64)
    voltages[0, 5] = np.nan
    voltages[1, :] = 0.0
    voltages[3, 70] = np.inf
    mask, sk = spectral_kurtosis_mask(voltages, m=64, return_statistic=True)
    assert mask[0, 0] and math.isnan(sk[0, 0])
    assert mask[1].all() and np.isnan(sk[1]).all()
    assert mask[3, 1] and math.isnan(sk[3, 1])
    assert not mask[0, 1]  # the second accumulation of row 0 is untouched
    assert not mask[3, 0]


def test_spectral_kurtosis_does_not_leak_runtime_warnings():
    """Undefined accumulations are a defined outcome, so nothing warns."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spectral_kurtosis_mask(np.full((2, 8), np.nan, dtype=np.complex128), m=4)
        spectral_kurtosis_mask(np.zeros((2, 8), dtype=np.complex64), m=4)
        spectral_kurtosis_mask(
            np.full((2, 8), np.inf, dtype=np.complex128), m=4, return_statistic=True
        )


def test_spectral_kurtosis_is_blind_to_steady_gaussian_interference():
    """The documented limitation: noise-like, always-on interference is invisible.

    A transmitter whose modulation is itself circular complex Gaussian
    and whose power does not vary within the accumulation has exactly the
    statistics of louder noise, so the estimator stays at 1 however
    bright it is. Detecting that case needs a detector sensitive to the
    mean, which is what `mad_clip_mask` is for -- the two are
    complementary, not ranked.
    """
    rng = np.random.default_rng(107)
    contaminated = 10.0 * gaussian_voltages(rng, (16, 65536))
    mask, sk = spectral_kurtosis_mask(contaminated, m=1024, return_statistic=True)
    assert abs(sk.mean() - 1.0) < 0.01
    assert mask.mean() < 0.02


def test_spectral_kurtosis_is_deterministic():
    """Two identical calls give bit-identical masks: no generator is drawn."""
    rng = np.random.default_rng(106)
    voltages = gaussian_voltages(rng, (4, 512))
    first = spectral_kurtosis_mask(voltages, m=128)
    second = spectral_kurtosis_mask(voltages, m=128)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"m": 1}, "m must be >= 2"),
        ({"m": 4096}, "m must be <="),
        ({"m": 2.9}, "m must be a whole number"),
        ({"m": 64.5}, "m must be a whole number"),
        ({"m": 64, "pfa": 0.0}, r"pfa must be in \(0, 1\)"),
        ({"m": 64, "pfa": 1.0}, r"pfa must be in \(0, 1\)"),
    ],
)
def test_spectral_kurtosis_validates_its_parameters(kwargs, message):
    """Bad accumulation lengths and probabilities raise, they do not clamp.

    A fractional `m` is rejected rather than truncated: ``m=2.9`` would
    silently become 2, changing the grid the mask is defined on and the
    grid the truth has to be pooled onto with it.
    """
    voltages = np.zeros((2, 256), dtype=np.complex64)
    with pytest.raises(ValueError, match=message):
        spectral_kurtosis_mask(voltages, **kwargs)


def test_spectral_kurtosis_accepts_integral_floats_and_numpy_integers():
    """Whole-number floats and numpy integers are fine; only fractions are not."""
    rng = np.random.default_rng(108)
    voltages = gaussian_voltages(rng, (2, 256))
    reference = spectral_kurtosis_mask(voltages, m=64)
    for value in (64.0, np.int64(64), np.int32(64)):
        np.testing.assert_array_equal(spectral_kurtosis_mask(voltages, m=value), reference)


def test_spectral_kurtosis_rejects_real_input():
    """A power spectrogram is not a substitute for the voltages."""
    with pytest.raises(ValueError, match="complex channelized voltages"):
        spectral_kurtosis_mask(np.ones((4, 256)), m=64)


def test_spectral_kurtosis_rejects_the_wrong_rank():
    """Only (n_chan, n_time) and (n_ant, n_chan, n_time) are meaningful."""
    with pytest.raises(ValueError, match="n_chan, n_time"):
        spectral_kurtosis_mask(np.ones(256, dtype=np.complex64), m=64)
    with pytest.raises(ValueError, match="n_chan, n_time"):
        spectral_kurtosis_mask(np.ones((2, 2, 2, 256), dtype=np.complex64), m=64)


# ----------------------------------------------------------------------
# Spectral kurtosis: false-alarm behaviour
# ----------------------------------------------------------------------
def sk_false_positive_rate(m, pfa, n_accumulations, seed, n_chan=16):
    """Fraction of pure-noise accumulations flagged, generated in chunks.

    Chunked so the test never holds more than a few million samples at
    once, which keeps the suite laptop-friendly.
    """
    rng = np.random.default_rng(seed)
    per_chunk = max(1, 1_000_000 // (m * n_chan))
    flagged = 0
    total = 0
    remaining = n_accumulations
    while remaining > 0:
        n_now = min(per_chunk, remaining)
        voltages = gaussian_voltages(rng, (n_chan, n_now * m))
        mask = spectral_kurtosis_mask(voltages, m=m, pfa=pfa)
        flagged += int(mask.sum())
        total += mask.size
        remaining -= n_now
    return flagged / total, total


def test_spectral_kurtosis_false_positive_rate_on_pure_noise():
    """At long accumulations and a loose pfa, the realized rate is the nominal one.

    This is the regime where the Gaussian approximation to the
    estimator's distribution holds: ``M = 1024`` and a 10 % nominal rate.
    The tolerance is the larger of three binomial standard errors and 8 %
    of the nominal rate; the second term is the residual Pearson-IV skew,
    which the next test measures directly.
    """
    pfa = 0.1
    rate, n_cells = sk_false_positive_rate(m=1024, pfa=pfa, n_accumulations=512, seed=201)
    tolerance = max(3.0 * binomial_sigma(pfa, n_cells), 0.08 * pfa)
    assert abs(rate - pfa) < tolerance


def test_spectral_kurtosis_thresholds_are_optimistic_at_small_m():
    """The documented small-M caveat, measured.

    The exact distribution of the estimator is right-skewed, so the
    Gaussian thresholds under-cover: the realized false-alarm rate at the
    default 3-sigma setting exceeds the nominal one, the excess lives
    almost entirely in the upper tail, and it shrinks as the accumulation
    lengthens. Pinned so that shipping exact Pearson-IV thresholds later
    shows up here as a deliberate change.
    """
    pfa = 0.0027
    ratios = {}
    for m, n_accumulations in ((64, 8192), (1024, 4096)):
        rate, _ = sk_false_positive_rate(m=m, pfa=pfa, n_accumulations=n_accumulations, seed=202)
        ratios[m] = rate / pfa
    # Measured at these seeds and sample counts: about 3.7 and about 1.4,
    # each with a binomial spread of a few per cent.
    assert 3.0 < ratios[64] < 4.5
    assert 1.1 < ratios[1024] < 2.0
    assert ratios[1024] < ratios[64]

    # The excess is one-sided: the lower tail under-fires, the upper tail
    # over-fires, which is exactly what a right-skewed statistic does.
    rng = np.random.default_rng(203)
    voltages = gaussian_voltages(rng, (16, 64 * 4096))
    _, sk = spectral_kurtosis_mask(voltages, m=64, pfa=pfa, return_statistic=True)
    half_width = NormalDist().inv_cdf(1.0 - 0.5 * pfa) * math.sqrt(4.0 / 64)
    low = float((sk < 1.0 - half_width).mean())
    high = float((sk > 1.0 + half_width).mean())
    assert low < 0.5 * pfa
    assert high > 2.0 * pfa


# ----------------------------------------------------------------------
# MAD clipping
# ----------------------------------------------------------------------
def test_mad_clip_false_positive_rate_matches_the_gaussian_nominal_rate():
    """On genuinely Gaussian data the n-sigma cut delivers the Gaussian rate.

    The 1.4826 scaling is what makes this true; without it the realized
    rate would be wrong by orders of magnitude at 3 sigma.
    """
    rng = np.random.default_rng(301)
    values = rng.standard_normal((256, 4096))
    for n_sigma in (2.0, 3.0):
        nominal = 2.0 * upper_tail(n_sigma)
        rate = float(mad_clip_mask(values, n_sigma).mean())
        assert abs(rate - nominal) < 4.0 * binomial_sigma(nominal, values.size)


def test_mad_scaling_recovers_the_standard_deviation():
    """1.4826 * MAD estimates the Gaussian sigma, which is what n_sigma means."""
    rng = np.random.default_rng(302)
    values = 7.0 * rng.standard_normal(200_000)
    mad = np.median(np.abs(values - np.median(values)))
    assert MAD_TO_SIGMA * mad == pytest.approx(7.0, rel=0.02)


def test_mad_clip_false_positive_rate_on_chi_squared_power_is_pinned_empirically():
    """On detected power the Gaussian n-sigma is nominal only, so pin the truth.

    Single-sample power is exponential and strongly right-skewed. A
    5-sigma cut, which would flag 6e-7 of Gaussian data, flags of order a
    percent here, and essentially every false alarm is on the high side.
    The number is pinned rather than derived because the point of the
    test is that the Gaussian value does *not* apply.
    """
    rng = np.random.default_rng(303)
    power = rng.standard_exponential((256, 4096))
    rate = float(mad_clip_mask(power, 5.0).mean())
    assert 0.010 < rate < 0.020
    assert rate > 1000.0 * 2.0 * upper_tail(5.0)
    _, deviation = mad_clip_mask(power, 5.0, return_statistic=True)
    assert float((deviation < -5.0).mean()) == 0.0


def test_mad_clip_statistics_are_per_channel():
    """A loud channel does not raise the threshold of a quiet one.

    With a global scale, the quiet channel's outlier would be invisible
    and the loud channel would be flagged wholesale.
    """
    rng = np.random.default_rng(305)
    values = np.empty((2, 400))
    values[0] = 1000.0 + rng.standard_normal(400)  # loud channel, unit scatter
    values[1] = 0.001 * rng.standard_normal(400)  # quiet channel, tiny scatter
    values[1, 50] = 0.05  # a 50-sigma outlier by the quiet channel's own scale
    mask = mad_clip_mask(values, 5.0)
    assert mask[1, 50]
    assert not mask[0].any()
    # A single global median and scale would be set by the loud channel and
    # would miss the outlier completely.
    global_scale = MAD_TO_SIGMA * np.median(np.abs(values - np.median(values)))
    assert abs(values[1, 50] - np.median(values)) / global_scale < 5.0


def test_mad_clip_scale_is_not_inflated_by_the_contamination_it_looks_for():
    """Median and MAD survive contamination that would blind mean and sigma.

    A fifth of the samples carry a bright transmitter. The standard
    deviation triples, so a 5-sigma cut on mean/std finds nothing; the
    MAD is untouched and every contaminated cell is flagged.
    """
    rng = np.random.default_rng(304)
    values = rng.standard_normal((1, 500))
    values[0, ::5] += 30.0
    mask = mad_clip_mask(values, 5.0)
    assert mask[0, ::5].all()
    assert not mask[0, 1::5].any()
    naive = np.abs(values - values.mean()) / values.std()
    assert not (naive > 5.0).any()


def test_mad_clip_is_blind_to_a_permanently_contaminated_channel():
    """The documented limitation: the median absorbs interference that never stops.

    Per-channel statistics are what make the detector robust to a
    bandpass, and they are also what make a transmitter that is on for
    the whole record indistinguishable from a hot channel. This is the
    mirror image of the kurtosis's blind spot.
    """
    rng = np.random.default_rng(306)
    power = rng.standard_exponential((2, 4096))
    power[1] *= 100.0  # a transmitter on for the entire record
    mask, deviation = mad_clip_mask(power, 5.0, return_statistic=True)
    assert mask[1].mean() == pytest.approx(mask[0].mean(), abs=0.01)
    np.testing.assert_allclose(np.median(deviation[1]), 0.0, atol=1e-12)


def test_mad_clip_handles_a_channel_with_zero_mad():
    """A constant channel has no scale: only cells off the median are flagged."""
    values = np.ones((2, 8))
    values[0, 3] = 50.0
    mask, deviation = mad_clip_mask(values, 5.0, return_statistic=True)
    assert mask[0, 3]
    assert not mask[0, [0, 1, 2, 4, 5, 6, 7]].any()
    assert not mask[1].any()
    assert math.isinf(deviation[0, 3])
    np.testing.assert_array_equal(deviation[1], np.zeros(8))


def test_mad_clip_flags_nan_cells():
    """NaN is never clean, and it does not poison its channel's statistics."""
    values = np.zeros((2, 9))
    values[0, 4] = np.nan
    mask, deviation = mad_clip_mask(values, 5.0, return_statistic=True)
    assert mask[0, 4]
    assert math.isnan(deviation[0, 4])
    assert int(mask.sum()) == 1


def test_mad_clip_infinities_do_not_hide_a_co_channel_outlier():
    """Infinities are missing data, not extreme data.

    Five ``+inf`` cells out of eight give the channel an infinite median
    if they are carried into the statistics; every deviation is then NaN
    and *nothing* is flagged -- not the infinities, and not the 1e9
    outlier sharing the channel. Excluding them from the median and MAD
    and flagging them outright is the only reading in which the outlier
    survives.
    """
    values = np.array([[np.inf] * 5 + [1.0, 1e9, 1.0]])
    mask, deviation = mad_clip_mask(values, 5.0, return_statistic=True)
    np.testing.assert_array_equal(mask[0], [True] * 5 + [False, True, False])
    assert np.isnan(deviation[0, :5]).all()


def test_mad_clip_flags_a_wholly_missing_channel():
    """A channel with no finite cell is flagged wholesale, quietly."""
    values = np.zeros((2, 8))
    values[0] = np.nan
    values[1, 3] = -np.inf
    mask = mad_clip_mask(values, 5.0)
    assert mask[0].all()
    assert int(mask[1].sum()) == 1


def test_mad_clip_does_not_leak_runtime_warnings():
    """Degenerate channels are a defined outcome, so nothing warns.

    `numpy.nanmedian` warns on an all-NaN slice, which would crash any
    pipeline running with warnings as errors. The suppression is narrow:
    only the two messages this function can legitimately provoke.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mad_clip_mask(np.full((2, 8), np.nan))
        mad_clip_mask(np.array([[np.inf] * 8, [1.0] * 8]))
        mad_clip_mask(np.ones((2, 8)))  # zero MAD everywhere
        mad_clip_mask(np.zeros((2, 8)), 5.0, return_statistic=True)


def test_mad_clip_deviations_are_signed():
    """The statistic keeps the sign, so a caller can make a one-sided cut."""
    values = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 100.0, -100.0]])
    _, deviation = mad_clip_mask(values, 5.0, return_statistic=True)
    assert deviation[0, 5] > 0.0
    assert deviation[0, 6] < 0.0
    assert deviation[0, 2] == pytest.approx(0.0)


def test_mad_clip_validates_its_input():
    """Complex input, a 1-D input and a non-positive threshold all raise."""
    with pytest.raises(ValueError, match="real spectrogram"):
        mad_clip_mask(np.ones((4, 8), dtype=np.complex64))
    with pytest.raises(ValueError, match="n_chan, n_time"):
        mad_clip_mask(np.ones(8))
    with pytest.raises(ValueError, match="n_sigma must be > 0"):
        mad_clip_mask(np.ones((4, 8)), 0.0)


# ----------------------------------------------------------------------
# SumThreshold
# ----------------------------------------------------------------------
def test_sumthreshold_with_one_window_is_a_plain_threshold():
    """The first pass has an analytic meaning: flag every cell above chi_1.

    With a single window size the method degenerates to a per-cell cut,
    so on a standard-normal residual the realized false-positive rate is
    exactly the upper tail ``Q(chi_1)``.
    """
    rng = np.random.default_rng(401)
    residual = rng.standard_normal((256, 4096))
    mask = sumthreshold_mask(residual, chi_1=3.0, iterations=1)
    np.testing.assert_array_equal(mask, residual > 3.0)
    nominal = upper_tail(3.0)
    rate = float(mask.mean())
    assert abs(rate - nominal) < 4.0 * binomial_sigma(nominal, residual.size)


def test_sumthreshold_is_one_sided():
    """Only added power is flagged; a deep negative excursion is not."""
    residual = np.zeros((4, 16))
    residual[0, 8] = -50.0
    assert not sumthreshold_mask(residual, chi_1=6.0).any()
    assert sumthreshold_mask(np.abs(residual), chi_1=6.0)[0, 8]


@pytest.mark.parametrize("axis", [0, 1])
def test_sumthreshold_finds_a_faint_run_no_single_cell_reveals(axis):
    """The whole point of the method, in both the time and frequency passes.

    Eight contiguous cells at 2.5 sigma, at rows/columns 4 to 11: no cell
    comes near ``chi_1 = 6``, and no window narrower than 8 reaches its
    own threshold either (``4 * chi_4 = 4 * 6/1.5^2 = 10.7`` against a sum
    of 10). The window-8 threshold is ``8 * chi_8 = 8 * 6/1.5^3 = 14.2``
    against a sum of 20, so that pass -- and only that pass -- catches it.

    The flag comes out four cells wider than the run, at 2 to 13, and
    that is correct rather than sloppy: a window overhanging the run by
    one or two cells still sums 17.5 or 15.0 and still exceeds 14.2, so
    it flags everything it covers. Widening a detection by less than the
    window that made it is the price of the method's sensitivity.
    """
    residual = np.zeros((16, 16))
    stripe = (slice(4, 12), 9) if axis == 0 else (9, slice(4, 12))
    residual[stripe] = 2.5

    assert not sumthreshold_mask(residual, chi_1=6.0, iterations=3).any()
    mask = sumthreshold_mask(residual, chi_1=6.0, iterations=4)
    assert mask[stripe].all()
    expected = np.zeros((16, 16), dtype=bool)
    expected[(slice(2, 14), 9) if axis == 0 else (9, slice(2, 14))] = True
    np.testing.assert_array_equal(mask, expected)


def test_sumthreshold_does_not_spread_a_single_bright_sample():
    """Flagged cells enter later sums at the threshold, not at their own value.

    One sample a thousand sigma high. Window 1 flags it. If it kept its
    own value in the window-2, -4, -8 and -16 sums, every window
    containing it would blow past its threshold and the flag would smear
    across 31 cells. Substituting the threshold value makes a flagged
    cell contribute exactly the neutral amount, so nothing else is
    touched.
    """
    residual = np.zeros((8, 64))
    residual[3, 20] = 1000.0
    mask = sumthreshold_mask(residual, chi_1=6.0, iterations=5)
    assert mask[3, 20]
    assert int(mask.sum()) == 1


def test_sumthreshold_thresholds_decay_as_powers_of_rho():
    """chi_M = chi_1 / rho^log2(M), checked at the boundary of each window.

    For each window size the run is set just above and just below its own
    threshold, which fails if the decay uses the wrong base or the wrong
    exponent.
    """
    chi_1, rho = 6.0, 1.5
    for step in range(5):
        window = 2**step
        chi_m = chi_1 / rho**step
        for offset, expected in ((1e-6, True), (-1e-6, False)):
            residual = np.zeros((4, 32))
            residual[1, 8 : 8 + window] = chi_m + offset
            # Only this window size may fire: give it exactly `step + 1`
            # passes so no wider window is available.
            mask = sumthreshold_mask(residual, chi_1=chi_1, iterations=step + 1)
            assert bool(mask[1, 8 : 8 + window].all()) is expected


def test_sumthreshold_false_positive_rate_grows_with_iterations():
    """The documented cost of extra windows, measured.

    Only the single-window rate is analytic. Each further window size
    lowers the per-cell threshold by another factor of rho, and the rate
    climbs steeply: at ``chi_1 = 4`` it goes from about 3e-5 to about 1e-2
    over five windows. Pinned because `iterations` looks free and is not.
    """
    rng = np.random.default_rng(402)
    residual = rng.standard_normal((256, 4096))
    rates = [
        float(sumthreshold_mask(residual, chi_1=4.0, iterations=it).mean()) for it in range(1, 6)
    ]
    assert rates == sorted(rates)
    assert rates[0] == pytest.approx(upper_tail(4.0), rel=0.3)
    assert 5e-3 < rates[4] < 2e-2
    # Raising chi_1 is the cheaper knob than shortening the window ladder.
    assert float(sumthreshold_mask(residual, chi_1=6.0, iterations=5).mean()) < 1e-4


def test_sumthreshold_flags_nan_cells_without_spreading_them():
    """NaN is flagged up front and then behaves like any other flagged cell."""
    residual = np.zeros((4, 32))
    residual[2, 10] = np.nan
    mask = sumthreshold_mask(residual, chi_1=6.0, iterations=5)
    assert mask[2, 10]
    assert int(mask.sum()) == 1


def test_sumthreshold_survives_an_infinite_cell_beside_real_interference():
    """A single -inf must not blind the method to its neighbours.

    Left in the values, one ``-inf`` drives the running sum of every
    window containing it to ``-inf``; the one-sided test then flags
    nothing in any of those windows, so the faint 8-cell run right next
    to it goes undetected -- and the ``-inf`` cell itself is never
    flagged either, since it is a *negative* excursion. This is not a
    contrived input: `mad_clip_mask` emits ``-inf`` for below-median
    cells of a channel with no scale, and its deviation array is the
    documented input to this function.
    """
    residual = np.zeros((4, 16))
    residual[1, 4:12] = 2.5
    residual[1, 3] = -np.inf
    mask = sumthreshold_mask(residual, chi_1=6.0, iterations=5)
    assert mask[1, 4:12].all()
    assert mask[1, 3]
    assert not mask[0].any()

    # +inf is missing data too, and does not spread into its neighbours.
    positive = np.zeros((4, 16))
    positive[2, 7] = np.inf
    spread = sumthreshold_mask(positive, chi_1=6.0, iterations=5)
    assert spread[2, 7]
    assert int(spread.sum()) == 1


def test_sumthreshold_chain_from_mad_clip_deviations_is_safe():
    """The documented chain end to end, on a channel with no scale.

    The zero-MAD channel makes `mad_clip_mask` emit ``+/-inf``, which is
    exactly the input that used to poison the sums.
    """
    power = np.ones((3, 32))
    power[0, 10] = 100.0  # zero-MAD channel with one outlier
    power[1] = np.arange(32) * 0.01
    power[1, 20:28] += 0.5  # a faint run in a channel that does have a scale
    _, deviation = mad_clip_mask(power, return_statistic=True)
    mask = sumthreshold_mask(deviation, chi_1=6.0, iterations=5)
    assert mask[0, 10]
    assert mask[1, 20:28].any()


def test_sumthreshold_does_not_leak_runtime_warnings():
    """Non-finite input is neutralized before any arithmetic touches it."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sumthreshold_mask(np.full((2, 8), np.nan))
        sumthreshold_mask(np.full((2, 8), -np.inf))
        sumthreshold_mask(np.full((2, 8), np.inf), return_statistic=True)


def test_sumthreshold_is_deterministic():
    """Two identical calls give bit-identical masks."""
    rng = np.random.default_rng(403)
    residual = rng.standard_normal((16, 256))
    first = sumthreshold_mask(residual, chi_1=4.0)
    second = sumthreshold_mask(residual, chi_1=4.0)
    np.testing.assert_array_equal(first, second)


def test_sumthreshold_validates_its_input():
    """Complex input, a 1-D input, and degenerate ladder parameters raise."""
    with pytest.raises(ValueError, match="real residual"):
        sumthreshold_mask(np.ones((4, 8), dtype=np.complex64))
    with pytest.raises(ValueError, match="n_chan, n_time"):
        sumthreshold_mask(np.ones(8))
    with pytest.raises(ValueError, match="iterations must be >= 1"):
        sumthreshold_mask(np.ones((4, 8)), iterations=0)
    with pytest.raises(ValueError, match="rho must be > 1"):
        sumthreshold_mask(np.ones((4, 8)), rho=1.0)
    for chi_1 in (0.0, -1.0):
        with pytest.raises(ValueError, match="chi_1 must be > 0"):
            sumthreshold_mask(np.ones((4, 8)), chi_1)


def test_sumthreshold_windows_wider_than_the_axis_are_skipped():
    """A short axis simply stops contributing passes rather than raising."""
    residual = np.zeros((2, 3))
    residual[0, 1] = 100.0
    mask = sumthreshold_mask(residual, chi_1=6.0, iterations=8)
    assert mask[0, 1]
    assert int(mask.sum()) == 1


# ----------------------------------------------------------------------
# End to end, against the simulator's own ground truth
# ----------------------------------------------------------------------
N_CHAN = 32
N_TIME = 4096
N_BLOCKS = 8
ACCUMULATION = 256
CHAN_WIDTH_HZ = 30517.578125
SAMPLE_PERIOD_S = 1.0 / CHAN_WIDTH_HZ
CENTER_FREQ_HZ = 1.405e9
TOWER_ENU_M = np.array([2000.0, 0.0, 30.0])


def make_simulator(tower, seed=4):
    """A small noise-only observation, optionally with one transmitter.

    Three antennas, 32 channels, 8 blocks of 4096 samples: enough cells
    for a stable recall (128 accumulations per channel) and small enough
    to stay laptop-fast. No sky sources, so the only structure in the
    voltages is receiver noise plus whatever the transmitter adds.
    """
    array = random_flat_array(3, seed=2)
    duration_s = N_BLOCKS * N_TIME * SAMPLE_PERIOD_S
    phase_center = zenith_phase_center(array, START_TIME, duration_s)
    return VoltageSimulator(
        array,
        phase_center,
        START_TIME,
        (),
        rfi_sources=() if tower is None else (tower,),
        center_freq_hz=CENTER_FREQ_HZ,
        n_chan=N_CHAN,
        chan_width_hz=CHAN_WIDTH_HZ,
        n_time_per_block=N_TIME,
        n_blocks=N_BLOCKS,
        noise_std=1.0,
        rng=np.random.default_rng(seed),
    )


def make_tower(received_power_jy, duty_cycle, frame_samples):
    """A transmitter three channels wide at the band center."""
    return NarrowbandTransmitter(
        position_enu_m=TOWER_ENU_M,
        center_freq_hz=CENTER_FREQ_HZ,
        bandwidth_hz=3 * CHAN_WIDTH_HZ,
        received_power_jy=received_power_jy,
        duty_cycle=duty_cycle,
        frame_duration_s=frame_samples * SAMPLE_PERIOD_S,
        name="tower",
    )


def collect_statistics(simulator, antenna=0):
    """Both statistics and the pooled truth on one common accumulation grid.

    Spectral kurtosis reads the complex voltages of one antenna; MAD
    clipping reads the same antenna's power averaged over the *same*
    accumulations. `pool_truth_accumulations` puts the labels on that
    grid too -- the same fixed blocks the kurtosis decided on -- so all
    three arrays are directly comparable and neither detector is handed a
    finer view than the other.
    """
    n_accumulations = N_TIME // ACCUMULATION
    kurtosis, deviation, truth = [], [], []
    for block in simulator.blocks():
        voltages = block.data[antenna]
        _, sk = spectral_kurtosis_mask(voltages, m=ACCUMULATION, return_statistic=True)
        power = bin_mean(np.abs(voltages) ** 2, axis=1, n_bins=n_accumulations)
        _, dev = mad_clip_mask(power, return_statistic=True)
        kurtosis.append(sk)
        deviation.append(dev)
        if block.n_rfi_sources:
            truth.append(pool_truth_accumulations(block.rfi_mask[0], ACCUMULATION))
    return (
        np.concatenate(kurtosis, axis=1),
        np.concatenate(deviation, axis=1),
        np.concatenate(truth, axis=1) if truth else None,
    )


def test_all_three_flaggers_recover_a_high_snr_transmitter():
    """Acceptance: recall above 0.9 for all three at their default settings.

    The transmitter is three channels wide, 200 Jy against a noise power
    of 1 Jy, and switches in frames a quarter of an accumulation long
    with a 10 % duty cycle. That is deliberately the *easy* case, and it
    is the one every detector has to pass: the interference is bright,
    it modulates within an accumulation (so the kurtosis sees it), it
    lifts the accumulated power far above the channel median (so the
    clipping sees it), and it occupies contiguous runs in both time and
    frequency (so SumThreshold sees it).
    """
    tower = make_tower(received_power_jy=200.0, duty_cycle=0.1, frame_samples=ACCUMULATION // 4)
    kurtosis, deviation, truth = collect_statistics(make_simulator(tower))
    assert 0.01 < truth.mean() < 0.2  # sparse, as real interference is

    half_width = NormalDist().inv_cdf(1.0 - 0.5 * 0.0027) * math.sqrt(4.0 / ACCUMULATION)
    masks = {
        "spectral_kurtosis": (kurtosis < 1.0 - half_width) | (kurtosis > 1.0 + half_width),
        "mad_clip": np.abs(deviation) > 5.0,
        "sumthreshold": sumthreshold_mask(deviation, chi_1=6.0),
    }
    for name, mask in masks.items():
        scores = flag_scores(mask, truth)
        assert scores["recall"] > 0.9, f"{name} recall {scores['recall']:.3f}"
        assert scores["mcc"] > 0.5, f"{name} mcc {scores['mcc']:.3f}"


def test_spectral_kurtosis_beats_mad_clipping_on_low_duty_bursts():
    """Acceptance: what the pre-detection statistic buys, at a matched budget.

    The transmitter now fires single-sample bursts with a duty cycle of
    0.1 %, so a contaminated accumulation typically holds one bright
    sample among 256. Averaged into the accumulated power that burst is
    diluted by a factor of 256 and barely clears the channel's own
    scatter; in the kurtosis it is undiluted, because the statistic
    measures the *shape* of the intensity distribution rather than its
    mean.

    Both detectors are calibrated on the interference-free run with the
    same seed -- their thresholds are the 99th percentile of their own
    clean-data statistic -- so they spend the same 1 % false-positive
    budget and only their recalls are being compared.
    """
    budget = 0.01
    tower = make_tower(received_power_jy=90.0, duty_cycle=0.001, frame_samples=1)
    clean_kurtosis, clean_deviation, _ = collect_statistics(make_simulator(None))
    kurtosis, deviation, truth = collect_statistics(make_simulator(tower))

    kurtosis_mask = kurtosis > np.quantile(clean_kurtosis, 1.0 - budget)
    deviation_mask = deviation > np.quantile(clean_deviation, 1.0 - budget)
    sk_scores = flag_scores(kurtosis_mask, truth)
    mad_scores = flag_scores(deviation_mask, truth)

    # The budgets really are matched before the recalls are compared.
    assert abs(sk_scores["false_positive_rate"] - mad_scores["false_positive_rate"]) < 0.005
    assert sk_scores["false_positive_rate"] < 2.0 * budget

    assert sk_scores["recall"] > 0.6
    assert mad_scores["recall"] < 0.3
    assert sk_scores["recall"] - mad_scores["recall"] > 0.4
    assert sk_scores["mcc"] > 3.0 * mad_scores["mcc"]


def test_flaggers_never_see_the_ground_truth():
    """Blind by construction: every flagger takes an array, not a block.

    There is no argument through which a mask could reach a flagger, so a
    benchmark cannot accidentally score a detector that peeked.
    """
    import inspect

    for flagger in (spectral_kurtosis_mask, mad_clip_mask, sumthreshold_mask):
        parameters = set(inspect.signature(flagger).parameters)
        assert not parameters & {"block", "truth", "mask", "rfi_mask", "simulator"}
