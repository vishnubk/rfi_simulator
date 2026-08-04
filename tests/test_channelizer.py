"""Tests for rfi_simulator.channelizer -- the polyphase-filterbank response.

The physics claims under test are the three statistics that separate a real
filterbank from the perfect-channelizer assumption:

* channelized noise has *temporal memory*, lag-1 autocorrelation ``rho[1]``;
* neighbouring channels are *correlated*, coherence ``gamma``;
* a carrier between two channel centers appears in *both*, in the ratio
  ``H(delta - 1) / H(delta)``.

Both correlations are also asserted in the *power* domain -- the squares
``rho[1]**2`` and ``|gamma|**2`` -- because that is what a measurement of
a real backend's dynamic spectrum returns and what the defaults are
tuned against, and because a filterbank started from zeros gets the power
statistics wrong (see the warm-start section) while getting the voltage
ones right.

Every one of them is predicted in closed form from the prototype filter, so
the tests assert the *measured* statistic against the *predicted* one
rather than against a hard-coded number; the predictions themselves are
printed in the assertion messages and pinned in
`test_default_predictions_match_the_documented_values` so that a change of
default cannot slip past unnoticed.

The other half of the file guards the invariant that matters more than any
of that: with no channelizer attached, the simulator produces exactly the
bytes it always did.
"""

import hashlib

import numpy as np
import pytest
from conftest import zenith_phase_center

from rfi_simulator import (
    OCCUPANCY_THRESHOLD,
    NarrowbandTransmitter,
    PFBChannelizer,
    PointSource,
    SpectralLineForeground,
    VoltageSimulator,
    correlate,
    dirty_image,
)
from rfi_simulator.channelizer import (
    DEFAULT_N_TAPS,
    DEFAULT_SINC_BANDWIDTH,
    DEFAULT_WINDOW,
    WINDOWS,
    ideal_channel_weights,
)
from rfi_simulator.rfi import enu_from_horizontal

# A tone this far off a channel center is the awkward case: neither channel
# owns it, so the leakage into the neighbour is a large fraction of the
# main channel rather than a small correction.
OFFSET_CHANNELS = 0.403


def noise_simulator(array, start_time, **kwargs):
    """A noise-only simulator: nothing in the data but receiver noise."""
    options = dict(n_chan=32, n_time_per_block=2000, n_blocks=4, noise_std=1.0)
    options.update(kwargs)
    phase_center = zenith_phase_center(array, start_time, duration_s=1.0)
    return VoltageSimulator(
        array,
        phase_center,
        start_time,
        [],
        rng=np.random.default_rng(7),
        **options,
    )


def stack(sim):
    """All of a simulator's blocks, concatenated along the time axis."""
    return np.concatenate([block.data for block in sim.blocks()], axis=2)


def lag_one_autocorrelation(data):
    """Normalized lag-1 temporal autocorrelation of ``(n_ant, n_chan, n_time)``."""
    return complex(np.mean(data[:, :, 1:] * np.conj(data[:, :, :-1])) / np.mean(np.abs(data) ** 2))


def adjacent_coherence(data):
    """Normalized zero-lag coherence between neighbouring channels."""
    cross = np.mean(data[:, :-1] * np.conj(data[:, 1:]))
    power = np.sqrt(np.mean(np.abs(data[:, :-1]) ** 2) * np.mean(np.abs(data[:, 1:]) ** 2))
    return complex(cross / power)


# ----------------------------------------------------------------------
# Default off: the data must not move
# ----------------------------------------------------------------------
#: sha256 of ``VoltageBlock.data`` for blocks 0-2 of the reference
#: configuration built in `reference_simulator`, recorded from the
#: simulator as it stood before the filterbank model existed. These are a
#: contract: the perfect-channelizer path is the package's default output
#: and must stay bit-for-bit stable.
REFERENCE_DIGESTS = (
    "8ff749c1f6deeafca9b22df6168273b2e36b0088d6d9f9a9b24f99c0e8bd5902",
    "778c7316dbad8afba9d7594868ceeab6156d9f2b8851f983288e6842abc4dc15",
    "3e9ee5aec84b11724871b58caaafdf106def2ea3a2cf6d8f031bc86581c7db82",
)


def reference_simulator(array, start_time, **kwargs):
    """A little of everything -- sky, line, interference -- at a fixed seed."""
    phase_center = zenith_phase_center(array, start_time, duration_s=1.0)
    source = PointSource.from_lm(phase_center, (0.005, -0.003), flux_jy=3.0)
    line = SpectralLineForeground(
        center_freq_hz=1.40505e9, fwhm_hz=6.0e4, line_flux_jy=2.0, name="line"
    )
    transmitter = NarrowbandTransmitter(
        enu_from_horizontal(70.0, 1.0, 4000.0), 1.40512e9, 1.5e5, 400.0, name="tx"
    )
    return VoltageSimulator(
        array,
        phase_center,
        start_time,
        [source],
        rfi_sources=[transmitter],
        spectral_lines=[line],
        n_chan=32,
        n_time_per_block=64,
        n_blocks=3,
        rng=np.random.default_rng(20260803),
        **kwargs,
    )


def test_no_channelizer_reproduces_the_recorded_bytes(default_array, start_time):
    """The default path is bit-for-bit what it was before the model existed."""
    sim = reference_simulator(default_array, start_time)
    assert sim.channelizer is None
    for index, expected in enumerate(REFERENCE_DIGESTS):
        data = np.ascontiguousarray(sim.block(index).data)
        assert hashlib.sha256(data.tobytes()).hexdigest() == expected, (
            f"block {index} changed; the perfect-channelizer output is a contract"
        )


def test_no_channelizer_leaves_the_block_untouched(default_array, start_time):
    """With no filterbank, `block` returns the ideal voltages unmodified."""
    sim = reference_simulator(default_array, start_time)
    ideal, rfi_mask, celestial_mask = sim._ideal_block(1)
    block = sim.block(1)
    assert np.array_equal(block.data, ideal)
    assert np.array_equal(block.rfi_mask, rfi_mask)
    assert np.array_equal(block.celestial_mask, celestial_mask)
    assert block.channelizer is None


def test_channelizer_is_recorded_on_the_block(default_array, start_time):
    """The filterbank is ground truth, carried like the gains are."""
    pfb = PFBChannelizer()
    sim = reference_simulator(default_array, start_time, channelizer=pfb)
    assert sim.block(0).channelizer is pfb


# ----------------------------------------------------------------------
# Seed-tree isolation
# ----------------------------------------------------------------------
def test_attaching_a_channelizer_does_not_disturb_any_realization(default_array, start_time):
    """Same seed, same draws: the operator is deterministic, not stochastic.

    Compared at the *ideal* stage, before the filterbank runs, which is
    exactly the statement that the channelizer consumes no randomness and
    shifts nobody else's stream.
    """
    plain = reference_simulator(default_array, start_time)
    filtered = reference_simulator(default_array, start_time, channelizer=PFBChannelizer())
    for index in range(3):
        ideal_plain, mask_plain, line_plain = plain._ideal_block(index)
        ideal_filtered, mask_filtered, line_filtered = filtered._ideal_block(index)
        assert np.array_equal(ideal_plain, ideal_filtered)
        assert np.array_equal(mask_plain, mask_filtered)
        assert np.array_equal(line_plain, line_filtered)


def test_a_sub_channel_carrier_draws_the_same_stream_either_way(default_array, start_time):
    """Sub-channel placement changes where the power goes, not the draws.

    The carrier's own realization moves -- it is placed differently -- but
    it must still consume exactly one channel's worth of waveform, so the
    sky-and-noise branch and every other source are untouched.
    """
    phase_center = zenith_phase_center(default_array, start_time, duration_s=1.0)
    carrier = NarrowbandTransmitter(
        enu_from_horizontal(10.0, 2.0, 3000.0),
        1.4051e9,
        0.0,
        50.0,
        waveform="constant_envelope",
        name="carrier",
    )
    other = NarrowbandTransmitter(
        enu_from_horizontal(200.0, 3.0, 5000.0), 1.4049e9, 2.0e5, 30.0, name="other"
    )

    def build(**kwargs):
        return VoltageSimulator(
            default_array,
            phase_center,
            start_time,
            [],
            rfi_sources=[carrier, other],
            n_chan=32,
            n_time_per_block=64,
            n_blocks=2,
            rng=np.random.default_rng(11),
            **kwargs,
        )

    plain = build()
    filtered = build(channelizer=PFBChannelizer())
    for index in range(2):
        # The second source is drawn from its own seed-tree branch, so it
        # is only identical if the first source consumed the same numbers.
        rngs_plain = plain.rfi_block_rngs(index)
        rngs_filtered = filtered.rfi_block_rngs(index)
        contribution_plain, _ = other.contribution(plain.block_context(index, rngs_plain[1]))
        contribution_filtered, _ = other.contribution(
            filtered.block_context(index, rngs_filtered[1])
        )
        assert np.array_equal(contribution_plain, contribution_filtered)


# ----------------------------------------------------------------------
# Derived quantities of the default filter
# ----------------------------------------------------------------------
def test_default_predictions_match_the_documented_values():
    """The shipped defaults predict the statistics the docs quote.

    Hard numbers on purpose: they are the reason the three defaults are
    what they are, and changing any of them should force a deliberate
    update here. The prediction is all but independent of the channel
    count, which is why one channelizer can serve simulators of any width.
    """
    pfb = PFBChannelizer()
    assert (pfb.n_taps, pfb.window, pfb.sinc_bandwidth) == (
        DEFAULT_N_TAPS,
        DEFAULT_WINDOW,
        DEFAULT_SINC_BANDWIDTH,
    )
    for n_chan in (32, 64, 384):
        assert pfb.temporal_autocorrelation(n_chan)[1] == pytest.approx(0.149, abs=0.002)
        assert abs(pfb.adjacent_channel_coherence(n_chan)) == pytest.approx(0.126, abs=0.002)
        main = pfb.channel_response(OFFSET_CHANNELS, n_chan)
        neighbour = pfb.channel_response(OFFSET_CHANNELS - 1.0, n_chan)
        assert abs(neighbour / main) == pytest.approx(0.435, abs=0.005)
        # The power-domain pair, which is what a dynamic-spectrum
        # measurement of a real backend returns -- these are the numbers
        # the defaults are actually fitted to.
        assert pfb.temporal_power_autocorrelation(n_chan)[1] == pytest.approx(0.0221, abs=0.0005)
        assert pfb.adjacent_channel_power_correlation(n_chan) == pytest.approx(0.0159, abs=0.0005)


def test_prototype_is_normalized_so_that_channel_power_is_preserved():
    """``sum(h**2) == n_chan`` is what makes `apply` power-neutral."""
    for n_taps in (1, 2, 4, 8):
        for window in WINDOWS:
            pfb = PFBChannelizer(n_taps=n_taps, window=window)
            h = pfb.prototype_filter(16)
            assert h.size == n_taps * 16
            assert float(np.sum(h**2)) == pytest.approx(16.0, rel=1e-12)


def test_longer_filters_approach_the_perfect_channelizer():
    """Both correlations shrink monotonically as the prototype sharpens."""
    lags, coherences = [], []
    for n_taps in (2, 4, 8, 16):
        pfb = PFBChannelizer(n_taps=n_taps)
        lags.append(abs(pfb.temporal_autocorrelation(32)[1]))
        coherences.append(abs(pfb.adjacent_channel_coherence(32)))
    assert coherences == sorted(coherences, reverse=True)
    assert lags[-1] < lags[1] < 0.2


def test_single_tap_filterbank_has_no_temporal_memory():
    """One tap is a plain windowed transform: leakage, but no memory."""
    pfb = PFBChannelizer(n_taps=1)
    assert pfb.temporal_autocorrelation(32).tolist() == [1.0]
    assert abs(pfb.adjacent_channel_coherence(32)) > 0.5
    assert pfb.apply(np.zeros((32, 8), dtype=np.complex64))[1] is None


def test_ideal_channel_weights_are_a_unit_partition_of_the_carrier():
    """The perfect-channelizer spread conserves power and snaps at integers."""
    weights = ideal_channel_weights(np.arange(64) - 20.25, 64)
    assert float(np.sum(np.abs(weights) ** 2)) == pytest.approx(1.0, rel=1e-12)
    centered = ideal_channel_weights(np.arange(64) - 20.0, 64)
    assert np.abs(centered).argmax() == 20
    assert float(np.abs(centered[20])) == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(np.abs(np.delete(centered, 20)), 0.0, atol=1e-10)


# ----------------------------------------------------------------------
# Noise statistics through the simulator
# ----------------------------------------------------------------------
def test_channelized_noise_has_the_predicted_temporal_memory(default_array, start_time):
    """Lag-1 autocorrelation matches ``rho[1]``; without the filterbank it is zero."""
    pfb = PFBChannelizer()
    filtered = stack(noise_simulator(default_array, start_time, channelizer=pfb))
    plain = stack(noise_simulator(default_array, start_time))

    predicted = pfb.temporal_autocorrelation(32)[1]
    measured = lag_one_autocorrelation(filtered)
    # ~640k independent-ish sample pairs, so the estimator's spread is well
    # under a per cent; 0.01 is a generous several sigma.
    assert measured.real == pytest.approx(predicted, abs=0.01), (
        f"predicted rho[1] = {predicted:.4f}"
    )
    assert abs(lag_one_autocorrelation(plain)) < 0.01


def test_neighbouring_channels_have_the_predicted_coherence(default_array, start_time):
    """Adjacent-channel coherence matches ``gamma``; without it, nothing."""
    pfb = PFBChannelizer()
    filtered = stack(noise_simulator(default_array, start_time, channelizer=pfb))
    plain = stack(noise_simulator(default_array, start_time))

    predicted = pfb.adjacent_channel_coherence(32)
    measured = adjacent_coherence(filtered)
    assert abs(measured) == pytest.approx(abs(predicted), abs=0.01), (
        f"predicted |gamma| = {abs(predicted):.4f}"
    )
    assert measured.real == pytest.approx(predicted.real, abs=0.01)
    assert abs(adjacent_coherence(plain)) < 0.01


def test_default_statistics_sit_in_the_expected_band(default_array, start_time):
    """The measured coloring is in the range a wideband backend shows.

    The point of the exercise: a lag-1 autocorrelation of order 0.1 and an
    adjacent-channel coherence of order 0.1, where the perfect channelizer
    gives 0.0005 and 0.007.
    """
    data = stack(noise_simulator(default_array, start_time, channelizer=PFBChannelizer()))
    assert 0.10 <= lag_one_autocorrelation(data).real <= 0.17
    assert 0.08 <= abs(adjacent_coherence(data)) <= 0.15


def test_channel_power_is_preserved(default_array, start_time):
    """The filterbank redistributes power between channels; it never adds any."""
    filtered = stack(noise_simulator(default_array, start_time, channelizer=PFBChannelizer()))
    plain = stack(noise_simulator(default_array, start_time))

    assert np.mean(np.abs(filtered) ** 2) == pytest.approx(np.mean(np.abs(plain) ** 2), rel=0.005)
    per_channel = np.mean(np.abs(filtered) ** 2, axis=(0, 2)) / np.mean(
        np.abs(plain) ** 2, axis=(0, 2)
    )
    # Band edges included: the filterbank treats the band as cyclic, so
    # they are no worse behaved than the middle.
    assert np.all(np.abs(per_channel - 1.0) < 0.03)


def test_apply_preserves_power_exactly_for_white_input():
    """The operator alone, without the simulator's statistical scatter."""
    pfb = PFBChannelizer()
    rng = np.random.default_rng(3)
    n_chan, n_time = 64, 8000
    data = (
        rng.standard_normal((n_chan, n_time)) + 1j * rng.standard_normal((n_chan, n_time))
    ).astype(np.complex64) / np.sqrt(2.0)
    out, _ = pfb.apply(data)
    assert np.mean(np.abs(out) ** 2) == pytest.approx(np.mean(np.abs(data) ** 2), rel=0.01)


# ----------------------------------------------------------------------
# Block seams
# ----------------------------------------------------------------------
def test_filter_state_carries_across_block_boundaries(default_array, start_time):
    """The correlation at a seam is the same as anywhere else in a block.

    A filterbank restarted at every block would give an uncorrelated seam
    -- which is what the ``cold`` comparison measures -- and would leave a
    periodic artifact at the block rate for a flagger to key on.
    """
    pfb = PFBChannelizer()
    sim = noise_simulator(default_array, start_time, channelizer=pfb, n_blocks=6)
    blocks = [block.data for block in sim.blocks()]
    power = np.mean([np.mean(np.abs(b) ** 2) for b in blocks])

    seam = (
        np.mean([np.mean(b[:, :, 0] * np.conj(a[:, :, -1])) for a, b in zip(blocks, blocks[1:])])
        / power
    )
    predicted = pfb.temporal_autocorrelation(32)[1]
    # Only n_ant * n_chan pairs per seam, so the tolerance is set by the
    # sample size (~1/sqrt(5 * 10 * 32) = 0.025), not by the model.
    assert seam.real == pytest.approx(predicted, abs=0.08), f"predicted rho[1] = {predicted:.4f}"

    cold = [pfb.apply(sim._ideal_block(i)[0])[0] for i in range(6)]
    cold_seam = (
        np.mean([np.mean(b[:, :, 0] * np.conj(a[:, :, -1])) for a, b in zip(cold, cold[1:])])
        / power
    )
    assert abs(cold_seam) < 0.05


def test_blocks_are_pure_functions_of_their_index(default_array, start_time):
    """Out-of-order generation reproduces the sequential result exactly."""
    pfb = PFBChannelizer()
    sequential = [
        block.data
        for block in noise_simulator(
            default_array, start_time, channelizer=pfb, n_time_per_block=64, n_blocks=4
        ).blocks()
    ]
    shuffled = noise_simulator(
        default_array, start_time, channelizer=pfb, n_time_per_block=64, n_blocks=4
    )
    for index in (2, 0, 3, 1, 2):
        assert np.array_equal(shuffled.block(index).data, sequential[index])


# ----------------------------------------------------------------------
# Warm start: the observation does not begin with the backend switching on
# ----------------------------------------------------------------------
def warm_start_simulator(array, start_time, **kwargs):
    """Noise-only, short blocks, a filterbank attached."""
    options = dict(n_chan=32, n_time_per_block=200, n_blocks=1, channelizer=PFBChannelizer())
    options.update(kwargs)
    return noise_simulator(array, start_time, **options)


def test_block_zero_starts_with_a_full_filter_by_default(default_array, start_time):
    """No zero, no ramp: sample 0 is as loud as any other sample.

    A cold filter makes ``P(t=0)`` strongly attenuated (identically zero
    for windows with zero endpoints) and the next ``n_taps - 1`` samples
    ramp up, in every channel at once -- the single most conspicuous
    artifact a simulated recording can carry, since an actual backend's
    output always comes from a system that has been running for hours.
    """
    sim = warm_start_simulator(default_array, start_time)
    assert sim.warm_start
    data = sim.block(0).data
    power = np.mean(np.abs(data) ** 2, axis=(0, 1))
    reference = power[20:].mean()

    assert np.all(power[:4] > 0.0)
    # The whole ramp, sample by sample, sits within the sampling scatter
    # of the middle of the block (~sqrt(2 / (n_ant * n_chan)) = 8 %).
    assert np.allclose(power[:4] / reference, 1.0, atol=0.15)

    cold = warm_start_simulator(default_array, start_time, warm_start=False).block(0).data
    cold_power = np.mean(np.abs(cold) ** 2, axis=(0, 1))
    # Switched on at t = 0: the first sample sees only the last polyphase
    # branch, which carries a quarter of a per cent of the filter's energy,
    # and the ramp is not complete until sample n_taps - 1.
    assert cold_power[0] / reference < 0.01
    assert cold_power[1] / reference < 0.6
    assert cold_power[4] / reference == pytest.approx(1.0, abs=0.15)


def test_the_first_samples_are_distributed_like_the_middle_of_the_block(default_array, start_time):
    """Across seeds, ``P(t=0..3)`` and mid-block power share a distribution.

    Stronger than the single-block check above: a residual transient
    would show up as a systematic offset of the *mean* over many
    independent realizations, well below the per-block scatter.
    """
    edge, middle = [], []
    for seed in range(24):
        sim = VoltageSimulator(
            default_array,
            zenith_phase_center(default_array, start_time, duration_s=1.0),
            start_time,
            [],
            n_chan=32,
            n_time_per_block=64,
            n_blocks=1,
            channelizer=PFBChannelizer(),
            rng=np.random.default_rng(seed),
        )
        power = np.abs(sim.block(0).data) ** 2
        edge.append(power[:, :, :4].mean())
        middle.append(power[:, :, 20:].mean())
    edge, middle = np.asarray(edge), np.asarray(middle)
    ratio = edge.mean() / middle.mean()
    # ~24 * 10 * 32 * 4 independent-ish edge samples, so the mean ratio's
    # standard error is about 1.5 %.
    assert ratio == pytest.approx(1.0, abs=0.06), f"edge/middle power ratio = {ratio:.4f}"


def test_warm_start_is_deterministic_and_index_pure(default_array, start_time):
    """Block 0 is the same however -- and however often -- it is asked for."""
    first = warm_start_simulator(default_array, start_time, n_blocks=3).block(0).data
    again = warm_start_simulator(default_array, start_time, n_blocks=3).block(0).data
    assert np.array_equal(again, first)

    sim = warm_start_simulator(default_array, start_time, n_blocks=3)
    # Out of order, after the seam cache has been filled by later blocks.
    sim.block(2)
    sim.block(1)
    assert np.array_equal(sim.block(0).data, first)
    assert np.array_equal(sim.block(0).data, first)
    assert np.array_equal(next(iter(sim.blocks())).data, first)


def test_warm_start_does_not_touch_the_other_blocks_or_the_ideal_stream(default_array, start_time):
    """It fills a filter state and nothing else: no draw moves."""
    warm = warm_start_simulator(default_array, start_time, n_blocks=3)
    cold = warm_start_simulator(default_array, start_time, n_blocks=3, warm_start=False)
    for index in range(3):
        assert np.array_equal(warm._ideal_block(index)[0], cold._ideal_block(index)[0])
    for index in (1, 2):
        assert np.array_equal(warm.block(index).data, cold.block(index).data)
    assert not np.array_equal(warm.block(0).data, cold.block(0).data)


def test_the_first_sample_of_block_zero_carries_the_filters_memory(default_array, start_time):
    """``corr(t=0, t=1)`` is ``rho[1]``, as it is everywhere else.

    The complement of the power test above: a warm start has to give the
    leading edge the filter's *correlation*, not merely the right
    variance. A cold start cannot -- sample 0 is built from a single
    polyphase branch, so it barely shares any input with sample 1.
    """
    pfb = PFBChannelizer()
    predicted = pfb.temporal_autocorrelation(32)[1]

    def edge_correlation(warm_start):
        pairs, powers = [], []
        for seed in range(24):
            sim = VoltageSimulator(
                default_array,
                zenith_phase_center(default_array, start_time, duration_s=1.0),
                start_time,
                [],
                n_chan=32,
                n_time_per_block=64,
                n_blocks=1,
                channelizer=pfb,
                warm_start=warm_start,
                rng=np.random.default_rng(seed),
            )
            data = sim.block(0).data
            pairs.append(np.mean(data[:, :, 1] * np.conj(data[:, :, 0])))
            powers.append(np.mean(np.abs(data) ** 2))
        return complex(np.mean(pairs) / np.mean(powers))

    # 24 * 10 * 32 pairs, so the estimator's spread is ~0.011.
    warm = edge_correlation(True)
    assert warm.real == pytest.approx(predicted, abs=0.04), f"predicted rho[1] = {predicted:.4f}"
    assert edge_correlation(False).real < 0.5 * predicted


def test_warm_start_is_off_for_a_memoryless_or_absent_filterbank(default_array, start_time):
    """Nothing to warm up: the state is `None` and costs nothing."""
    single = warm_start_simulator(default_array, start_time, channelizer=PFBChannelizer(n_taps=1))
    assert single._seam_state(0) is None
    plain = noise_simulator(default_array, start_time, n_chan=32, n_time_per_block=64, n_blocks=1)
    assert plain._seam_state(0) is None


# ----------------------------------------------------------------------
# Power-domain statistics
# ----------------------------------------------------------------------
def test_power_correlations_are_the_squares_of_the_voltage_ones():
    """The Gaussian fourth-moment identity, as an API guarantee."""
    for pfb in (PFBChannelizer(), PFBChannelizer(n_taps=6, window="blackman")):
        for delta in (1, 2, 3):
            assert pfb.channel_power_correlation(32, delta) == pytest.approx(
                abs(pfb.channel_coherence(32, delta)) ** 2, rel=1e-12
            )
        assert pfb.adjacent_channel_power_correlation(32) == pfb.channel_power_correlation(32, 1)
        assert np.allclose(
            pfb.temporal_power_autocorrelation(32), pfb.temporal_autocorrelation(32) ** 2
        )


def test_measured_dynamic_spectrum_correlations_match_the_prediction(default_array, start_time):
    """What a dynamic-spectrum statistic measures, against what the filter predicts.

    The quantity of interest is the correlation of *detected power*
    between neighbouring channels of the array-mean dynamic spectrum --
    ``|gamma|**2``, not ``|gamma|``. Simulation and prediction have to
    agree here or the defaults are being tuned against the wrong number:
    the mismatch that this pins used to be a factor of two, and it came
    from the cold-start ramp, not from the filter.
    """
    pfb = PFBChannelizer()
    data = stack(noise_simulator(default_array, start_time, channelizer=pfb))
    power = (np.abs(data.astype(np.complex128)) ** 2).mean(axis=0)  # (n_chan, n_time)
    normalized = power / power.mean(axis=1, keepdims=True)

    chan_corr = np.mean((normalized[:-1] - 1) * (normalized[1:] - 1)) / np.mean(
        (normalized - 1) ** 2
    )
    lag1 = np.mean(
        np.mean((normalized[:, :-1] - 1) * (normalized[:, 1:] - 1), axis=1)
        / np.var(normalized, axis=1)
    )
    assert chan_corr == pytest.approx(pfb.adjacent_channel_power_correlation(32), abs=0.004), (
        f"predicted |gamma|^2 = {pfb.adjacent_channel_power_correlation(32):.4f}"
    )
    assert lag1 == pytest.approx(pfb.temporal_power_autocorrelation(32)[1], abs=0.004), (
        f"predicted rho[1]^2 = {pfb.temporal_power_autocorrelation(32)[1]:.4f}"
    )
    # Non-adjacent channels are uncorrelated: the filter reaches exactly
    # one channel, so a positive floor at larger separations would mean a
    # common-mode artifact (a transient, a gain wobble) and not leakage.
    far = np.mean((normalized[:-5] - 1) * (normalized[5:] - 1)) / np.mean((normalized - 1) ** 2)
    assert abs(far) < 0.004


def test_a_cold_start_inflates_the_measured_channel_correlation(default_array, start_time):
    """Why the warm start is the default, as a number.

    The zero-and-ramp at ``t = 0`` is common to every channel, so it
    contributes a positive term to *every* channel pair's power
    correlation -- which would let a classifier separate simulated data
    from a real system's on this feature alone.
    """
    pfb = PFBChannelizer()

    def chan_corr(sim):
        block = sim.block(0).data
        power = (np.abs(block.astype(np.complex128)) ** 2).mean(axis=0)
        normalized = power / power.mean(axis=1, keepdims=True)
        return float(
            np.mean((normalized[:-1] - 1) * (normalized[1:] - 1)) / np.mean((normalized - 1) ** 2)
        )

    options = dict(n_chan=32, n_time_per_block=256, n_blocks=1, channelizer=pfb)
    warm = chan_corr(noise_simulator(default_array, start_time, **options))
    cold = chan_corr(noise_simulator(default_array, start_time, warm_start=False, **options))
    predicted = pfb.adjacent_channel_power_correlation(32)
    assert warm == pytest.approx(predicted, abs=0.01)
    assert cold > warm + 0.03


def test_trailing_state_matches_the_state_apply_returns():
    """The seam can be rebuilt from a block without filtering it."""
    pfb = PFBChannelizer(n_taps=5)
    rng = np.random.default_rng(1)
    data = (rng.standard_normal((3, 16, 40)) + 1j * rng.standard_normal((3, 16, 40))).astype(
        np.complex64
    )
    _, state = pfb.apply(data)
    assert np.allclose(pfb.trailing_state(data), state, atol=1e-5)


# ----------------------------------------------------------------------
# Carriers at arbitrary frequencies
# ----------------------------------------------------------------------
def carrier_simulator(array, start_time, offset_channels, **kwargs):
    """One unmodulated carrier, offset from a channel center, no noise."""
    options = dict(n_chan=32, n_time_per_block=512, n_blocks=1)
    options.update(kwargs)
    phase_center = zenith_phase_center(array, start_time, duration_s=1.0)
    probe = VoltageSimulator(
        array,
        phase_center,
        start_time,
        [],
        noise_std=0.0,
        rng=np.random.default_rng(5),
        **options,
    )
    target = probe.n_chan // 2
    freq_hz = probe.freq_hz[target] + offset_channels * probe.chan_width_hz
    carrier = NarrowbandTransmitter(
        enu_from_horizontal(45.0, 5.0, 3000.0),
        freq_hz,
        0.0,
        100.0,
        waveform="constant_envelope",
        name="carrier",
    )
    sim = VoltageSimulator(
        array,
        phase_center,
        start_time,
        [],
        rfi_sources=[carrier],
        noise_std=0.0,
        rng=np.random.default_rng(5),
        **options,
    )
    return sim, target


def test_off_center_carrier_leaks_by_the_predicted_ratio(default_array, start_time):
    """A tone between two channels appears in both, in the ratio ``H`` predicts."""
    pfb = PFBChannelizer()
    sim, target = carrier_simulator(default_array, start_time, OFFSET_CHANNELS, channelizer=pfb)
    block = sim.block(0)
    power = np.mean(np.abs(block.data) ** 2, axis=(0, 2))

    offsets = OFFSET_CHANNELS - (np.arange(sim.n_chan) - target)
    predicted = np.abs(pfb.channel_response(offsets, sim.n_chan)) ** 2
    predicted /= predicted[target]
    measured = power / power[target]

    ratio = np.sqrt(measured[target + 1])
    assert ratio == pytest.approx(np.sqrt(predicted[target + 1]), rel=0.02), (
        f"predicted leakage amplitude ratio = {np.sqrt(predicted[target + 1]):.4f}"
    )
    # And it is a large fraction, not a rounding error: a tone 0.4 channels
    # off center is genuinely shared between two channels.
    assert 0.3 < ratio < 0.8
    for channel in (target - 1, target + 2):
        # A few parts in 1e4, not exactly zero: the carrier's symbol
        # sequence is drawn per block, so the block boundary is a small
        # phase discontinuity that splatters across the band over the
        # filter's span. That happens at every seam; with `warm_start` it
        # happens at the leading edge of block 0 too, instead of block 0
        # alone being the one block with no seam behind it.
        assert measured[channel] == pytest.approx(predicted[channel], abs=1e-3)


def test_off_center_carrier_labels_every_channel_it_reaches(default_array, start_time):
    """Ground truth follows the power: the leaked channel is labelled too."""
    pfb = PFBChannelizer()
    sim, target = carrier_simulator(default_array, start_time, OFFSET_CHANNELS, channelizer=pfb)
    block = sim.block(0)
    occupied = block.rfi_mask[0].any(axis=1)
    assert occupied[target] and occupied[target + 1]

    power = np.mean(np.abs(block.data) ** 2, axis=(0, 2))
    above_threshold = power > OCCUPANCY_THRESHOLD * power.max()
    assert np.array_equal(occupied, above_threshold)


def test_off_center_carrier_beats_against_the_sample_rate(default_array, start_time):
    """The sub-channel offset shows up as a phase ramp in time."""
    sim, target = carrier_simulator(
        default_array, start_time, OFFSET_CHANNELS, channelizer=PFBChannelizer()
    )
    series = sim.block(0).data[0, target]
    step = np.angle(np.mean(series[1:] * np.conj(series[:-1]))) / (2.0 * np.pi)
    assert step == pytest.approx(OFFSET_CHANNELS, abs=1e-3)


def test_a_channel_centered_carrier_still_lands_in_one_channel(default_array, start_time):
    """Zero offset reduces to the single-channel convention, filterbank or not."""
    for channelizer in (None, PFBChannelizer()):
        sim, target = carrier_simulator(default_array, start_time, 0.0, channelizer=channelizer)
        power = np.mean(np.abs(sim.block(0).data) ** 2, axis=(0, 2))
        assert power.argmax() == target
        assert (power.sum() - power[target]) / power.sum() < 5e-4


def test_without_a_channelizer_a_carrier_snaps_to_a_channel(default_array, start_time):
    """The old behavior is untouched: no filterbank, no sub-channel resolution."""
    sim, target = carrier_simulator(default_array, start_time, OFFSET_CHANNELS)
    block = sim.block(0)
    power = np.mean(np.abs(block.data) ** 2, axis=(0, 2))
    assert power.argmax() == target
    assert (power.sum() - power[target]) / power.sum() < 1e-6
    assert block.rfi_mask[0].any(axis=1).sum() == 1


def test_leaky_filterbank_widens_the_ground_truth_labels(default_array, start_time):
    """A prototype that spreads a centered carrier must widen its labels too."""
    leaky = PFBChannelizer(n_taps=1)
    assert leaky.leakage_radius(32, OCCUPANCY_THRESHOLD) >= 1
    assert PFBChannelizer().leakage_radius(32, OCCUPANCY_THRESHOLD) == 0

    sim, target = carrier_simulator(default_array, start_time, 0.0, channelizer=leaky)
    block = sim.block(0)
    occupied = block.rfi_mask[0].any(axis=1)
    assert occupied[target - 1] and occupied[target] and occupied[target + 1]


# ----------------------------------------------------------------------
# The sky still images
# ----------------------------------------------------------------------
def test_a_source_still_images_at_the_right_place_through_the_filterbank(default_array, start_time):
    """Antenna-to-antenna coherence survives: same position, same flux.

    The filterbank is applied per antenna with the same filter, so the only
    effect on a visibility is that each channel now averages its
    neighbours' fringe phases -- a sub-per-cent correction at these
    baselines, and no shift at all.
    """
    pixel_rad = 2e-4
    grid = np.arange(-40, 41) * pixel_rad
    lm = (float(np.sin(np.deg2rad(0.3))), float(np.sin(np.deg2rad(-0.2))))
    phase_center = zenith_phase_center(default_array, start_time, duration_s=1.0)
    source = PointSource.from_lm(phase_center, lm, flux_jy=1.0)

    peaks = {}
    for label, channelizer in (("plain", None), ("pfb", PFBChannelizer())):
        sim = VoltageSimulator(
            default_array,
            phase_center,
            start_time,
            [source],
            noise_std=0.0,
            n_chan=16,
            n_blocks=4,
            n_time_per_block=250,
            channelizer=channelizer,
            rng=np.random.default_rng(99),
        )
        image, l_grid, m_grid = dirty_image(correlate(sim.blocks()), grid, grid)
        i_m, i_l = np.unravel_index(np.argmax(image), image.shape)
        peaks[label] = (l_grid[i_l], m_grid[i_m], image.max())

    assert peaks["pfb"][0] == pytest.approx(lm[0], abs=0.5 * pixel_rad)
    assert peaks["pfb"][1] == pytest.approx(lm[1], abs=0.5 * pixel_rad)
    assert peaks["pfb"][2] == pytest.approx(1.0, rel=0.03)
    assert peaks["pfb"][2] == pytest.approx(peaks["plain"][2], rel=0.03)


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(n_taps=0), "n_taps must be >= 1"),
        (dict(n_taps=-3), "n_taps must be >= 1"),
        (dict(n_taps=2.5), "n_taps must be an integer"),
        (dict(window="kaiser"), "window must be one of"),
        (dict(window=4), "window must be one of"),
        (dict(sinc_bandwidth=0.0), "sinc_bandwidth must be finite and > 0"),
        (dict(sinc_bandwidth=-1.0), "sinc_bandwidth must be finite and > 0"),
        (dict(sinc_bandwidth=np.nan), "sinc_bandwidth must be finite and > 0"),
        (dict(sinc_bandwidth=np.inf), "sinc_bandwidth must be finite and > 0"),
    ],
)
def test_construction_rejects_nonsense(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PFBChannelizer(**kwargs)


def test_prototype_rejects_a_nonsense_channel_count():
    with pytest.raises(ValueError, match="n_chan must be >= 1"):
        PFBChannelizer().prototype_filter(0)
    with pytest.raises(ValueError, match="n_chan must be >= 1"):
        ideal_channel_weights([0.0], 0)


def test_leakage_radius_rejects_a_nonsense_threshold():
    with pytest.raises(ValueError, match="threshold must be finite and > 0"):
        PFBChannelizer().leakage_radius(32, 0.0)
    with pytest.raises(ValueError, match="threshold must be finite and > 0"):
        PFBChannelizer().leakage_radius(32, np.nan)


def test_apply_rejects_the_wrong_shapes():
    pfb = PFBChannelizer()
    with pytest.raises(ValueError, match=r"shape \(\.\.\., n_chan, n_time\)"):
        pfb.apply(np.zeros(8, dtype=np.complex64))
    with pytest.raises(ValueError, match=r"shape \(\.\.\., n_chan, n_time\)"):
        pfb.trailing_state(np.zeros(8, dtype=np.complex64))
    with pytest.raises(ValueError, match="state must have shape"):
        pfb.apply(
            np.zeros((16, 8), dtype=np.complex64),
            np.zeros((2, 16), dtype=np.complex64),
        )


def test_simulator_rejects_a_non_channelizer(default_array, start_time):
    phase_center = zenith_phase_center(default_array, start_time, duration_s=1.0)
    with pytest.raises(ValueError, match="channelizer must be a PFBChannelizer or None"):
        VoltageSimulator(
            default_array,
            phase_center,
            start_time,
            [],
            n_chan=8,
            n_blocks=1,
            n_time_per_block=8,
            channelizer="hann",
            rng=np.random.default_rng(0),
        )


def test_repr_round_trips_the_parameters():
    pfb = PFBChannelizer(n_taps=6, window="blackman", sinc_bandwidth=0.9)
    assert repr(pfb) == "PFBChannelizer(n_taps=6, window='blackman', sinc_bandwidth=0.9)"


def test_instance_is_immutable_after_construction():
    """Mutating a shared instance in place would silently stale its cache.

    A channelizer can be attached to several simulators and its prototype
    filter is cached per channel count (see `prototype_filter`), so
    rebinding an attribute after construction would leave any already-cached
    ``n_chan`` serving the old filter while a new ``n_chan`` gets the new
    one -- one instance, two inconsistent filterbanks. Immutability rules
    that out entirely.
    """
    pfb = PFBChannelizer()
    for name, value in (("n_taps", 8), ("window", "blackman"), ("sinc_bandwidth", 0.5)):
        with pytest.raises(AttributeError, match="immutable"):
            setattr(pfb, name, value)
    with pytest.raises(AttributeError, match="immutable"):
        pfb.new_attribute = 1
