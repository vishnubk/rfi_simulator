"""Tests for the realism options of the interference sources.

Four features are covered here, and each of them exists because the
default model is wrong in a way that matters for a flagging benchmark:

* **Per-antenna coupling.** Real interference power is distributed over an
  array by paths this package does not model, so it is wildly uneven; the
  default geometry alone makes every antenna nearly identical.
* **Constant-envelope waveforms.** The canonical pre-detection detector,
  spectral kurtosis, is *blind* to a Gaussian-modulated carrier at any
  power. `test_spectral_kurtosis_sees_a_constant_envelope_carrier` is the
  test that justifies the whole feature: same power, same channel, one
  waveform detected and the other invisible.
* **Clocked on/off patterns**, which are periodic rather than i.i.d.
* **Harmonic combs**, where many lines belong to one device.
"""

import numpy as np
import pytest
from conftest import zenith_phase_center

from rfi_simulator import (
    ArrayConfig,
    CombTransmitter,
    ImpulsiveBroadband,
    NarrowbandTransmitter,
    VoltageSimulator,
    constant_envelope,
    correlate,
    path_delays_s,
    resolve_coupling,
    spectral_kurtosis_mask,
)
from rfi_simulator.rfi import spreading_amplitudes

TOWER_ENU_M = np.array([2000.0, 0.0, 0.0])
TOWER_CENTER_FREQ_HZ = 1.4053e9
TOWER_BANDWIDTH_HZ = 1.5e5


def make_tower(**kwargs):
    """A narrowband transmitter 2 km east, on by default."""
    options = dict(
        position_enu_m=TOWER_ENU_M,
        center_freq_hz=TOWER_CENTER_FREQ_HZ,
        bandwidth_hz=TOWER_BANDWIDTH_HZ,
        received_power_jy=200.0,
        name="tower",
    )
    options.update(kwargs)
    return NarrowbandTransmitter(**options)


def make_simulator(array, start_time, sources=(), rfi_sources=(), **kwargs):
    """Small-but-real simulator, mirroring the one in test_rfi."""
    options = dict(
        n_chan=32,
        n_blocks=3,
        n_time_per_block=200,
        noise_std=0.0,
        rng=np.random.default_rng(20261001),
    )
    options.update(kwargs)
    phase_center = zenith_phase_center(array, start_time, duration_s=0.1)
    return VoltageSimulator(
        array, phase_center, start_time, sources, rfi_sources=rfi_sources, **options
    )


def three_antenna_array() -> ArrayConfig:
    """Antennas at the origin, 100 m east and 100 m north."""
    return ArrayConfig(
        antenna_positions_enu_m=np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 100.0, 0.0]]),
        latitude_deg=37.234,
        longitude_deg=-118.282,
        height_m=1222.0,
        name="hand_computed",
    )


def band_power_per_antenna(data: np.ndarray) -> np.ndarray:
    """Mean power per antenna over the occupied part of a block."""
    occupied = np.abs(data).max(axis=(0, 2)) > 0.0
    return (np.abs(data[:, occupied, :]) ** 2).mean(axis=(1, 2))


# ----------------------------------------------------------------------
# Per-antenna coupling
# ----------------------------------------------------------------------
def test_uniform_coupling_is_a_no_op(default_array, start_time):
    """The default and an explicit all-ones vector are bit-identical.

    Two things at once: leaving `coupling` alone must not change any
    existing dataset, and the coupling code path itself must be exact, so
    that a coupling of 1.0 cannot perturb an antenna in the last bit.
    """
    n_antennas = default_array.n_antennas

    def run(coupling):
        sim = make_simulator(
            default_array, start_time, rfi_sources=[make_tower(coupling=coupling)], n_blocks=1
        )
        return sim.block(0)

    default_block = run(None)
    ones_block = run(np.ones(n_antennas))
    np.testing.assert_array_equal(ones_block.data, default_block.data)

    # A run with no coupling configured still publishes the truth, as ones.
    np.testing.assert_array_equal(default_block.rfi_coupling, np.ones((1, n_antennas)))
    assert default_block.rfi_coupling.shape == (1, n_antennas)


def test_per_antenna_power_follows_coupling_squared(default_array, start_time):
    """Coupling multiplies amplitude, so received power scales as its square.

    The pattern used here is the shape real measurements show: one antenna
    an order of magnitude above the rest -- 4.6 in amplitude is a factor 21
    in power -- with the remaining antennas at unity, i.e. an array whose
    *median* antenna sees nothing unusual while one of them is swamped.
    """
    n_antennas = default_array.n_antennas
    coupling = np.ones(n_antennas)
    coupling[3] = np.sqrt(21.0)
    coupling[7] = 0.5

    tower = make_tower(coupling=coupling)
    sim = make_simulator(
        default_array, start_time, rfi_sources=[tower], n_blocks=1, n_time_per_block=4000
    )
    power = band_power_per_antenna(sim.block(0).data)

    # The geometry contributes its own (small) per-antenna amplitude, so the
    # prediction is the product of the two, squared.
    geometry = spreading_amplitudes(TOWER_ENU_M, default_array.antenna_positions_enu_m)
    expected = (coupling * geometry) ** 2
    np.testing.assert_allclose(power / power.mean(), expected / expected.mean(), rtol=0.06)

    # The headline numbers: one antenna at 21x the median, in power.
    ratios = power / np.median(power)
    assert ratios[3] == pytest.approx(21.0, rel=0.06)
    assert ratios[7] == pytest.approx(0.25, rel=0.06)
    assert np.median(ratios) == pytest.approx(1.0, rel=0.02)


def test_coupling_ground_truth_is_recoverable(default_array, start_time):
    """The resolved vector is published on the source and on every block."""
    coupling = np.linspace(0.5, 3.0, default_array.n_antennas)
    tower = make_tower(coupling=coupling)
    sparks = ImpulsiveBroadband(
        rate_hz=100.0, coupling={"type": "lognormal", "sigma_db": 4.0, "seed": 5}
    )
    sim = make_simulator(default_array, start_time, rfi_sources=[tower, sparks], n_blocks=2)

    truth = sim.rfi_coupling()
    assert truth.shape == (2, default_array.n_antennas)
    np.testing.assert_allclose(truth[0], coupling)
    np.testing.assert_allclose(truth[1], sparks.coupling_amplitudes(default_array.n_antennas))

    # Coupling is a property of the installation: the same on every block.
    for index in range(2):
        np.testing.assert_array_equal(sim.block(index).rfi_coupling, truth)

    # And it is read-only, so a caller cannot edit the ground truth.
    with pytest.raises(ValueError):
        tower.coupling_amplitudes(default_array.n_antennas)[0] = 2.0


def test_lognormal_coupling_is_seeded_and_has_the_configured_width():
    """A lognormal spec is repeatable, unit-median, and sigma_db wide."""
    spec = {"type": "lognormal", "sigma_db": 6.0, "seed": 11}
    first = resolve_coupling(spec, 8)
    np.testing.assert_array_equal(first, resolve_coupling(dict(spec), 8))
    assert not np.allclose(first, resolve_coupling({**spec, "seed": 12}, 8))

    # sigma_db is the rms of the per-antenna *power* in dB, as elsewhere in
    # the package, and the median amplitude is 1.
    many = resolve_coupling({"type": "lognormal", "sigma_db": 6.0, "seed": 3}, 20000)
    power_db = 20.0 * np.log10(many)
    assert power_db.std() == pytest.approx(6.0, rel=0.03)
    assert np.median(many) == pytest.approx(1.0, abs=0.02)

    np.testing.assert_array_equal(resolve_coupling(None, 4), np.ones(4))


def test_coupling_leaves_the_ground_truth_mask_alone(default_array, start_time):
    """Labels describe the emitter, not what each antenna heard.

    A deliberate convention (documented in `rfi_simulator.rfi`): the mask
    has no antenna axis, so an antenna with *zero* coupling still carries
    the source's labels. The per-antenna information lives in the coupling
    vector instead.
    """
    n_antennas = default_array.n_antennas
    coupling = np.ones(n_antennas)
    coupling[0] = 0.0

    uniform = make_simulator(
        default_array, start_time, rfi_sources=[make_tower()], n_blocks=1
    ).block(0)
    coupled = make_simulator(
        default_array, start_time, rfi_sources=[make_tower(coupling=coupling)], n_blocks=1
    ).block(0)

    np.testing.assert_array_equal(coupled.rfi_mask, uniform.rfi_mask)
    assert coupled.rfi_mask.any()

    # The silent antenna really is silent, and the others are untouched.
    np.testing.assert_array_equal(coupled.data[0], np.zeros_like(coupled.data[0]))
    np.testing.assert_array_equal(coupled.data[1:], uniform.data[1:])


@pytest.mark.parametrize(
    ("coupling", "match"),
    [
        ({"type": "spatial", "sigma_db": 1.0, "seed": 1}, "coupling type"),
        ({"type": "lognormal", "sigma_db": -1.0, "seed": 1}, "sigma_db"),
        ({"type": "lognormal", "sigma_db": float("nan"), "seed": 1}, "sigma_db"),
        ({"type": "lognormal", "sigma_db": float("inf"), "seed": 1}, "sigma_db"),
        ({"type": "lognormal", "sigma_db": 1.0}, "seed"),
        ({"type": "lognormal", "sigma_db": 1.0, "seed": 1, "extra": 2}, "unexpected keys"),
        ([[1.0, 2.0]], r"shape \(n_antennas,\)"),
        ([1.0, -2.0], "linear amplitudes"),
        ([1.0, np.nan], "non-finite"),
        ([1.0, np.inf], "non-finite"),
    ],
)
def test_coupling_validation(coupling, match):
    with pytest.raises(ValueError, match=match):
        make_tower(coupling=coupling)


def test_coupling_length_must_match_the_array(default_array, start_time):
    """A vector of the wrong length is caught when the array is known."""
    tower = make_tower(coupling=np.ones(default_array.n_antennas + 1))
    sim = make_simulator(default_array, start_time, rfi_sources=[tower], n_blocks=1)
    with pytest.raises(ValueError, match="antennas"):
        sim.block(0)


def test_coupling_does_not_alias_the_caller_s_array(default_array, start_time):
    """`_normalize_coupling` freezes its own copy, not the caller's array.

    Freezing the caller's own float64 array in place would make it
    read-only outside the source's control -- surprising for code that
    passed in a plain numpy array expecting to keep using it.
    """
    coupling = np.ones(default_array.n_antennas)
    make_tower(coupling=coupling)
    coupling[0] = 2.0  # still writable
    assert coupling.flags.writeable


# ----------------------------------------------------------------------
# Constant-envelope waveform
# ----------------------------------------------------------------------
def test_spectral_kurtosis_sees_a_constant_envelope_carrier(default_array, start_time):
    """The point of the feature: a Gaussian carrier is invisible to SK.

    Same channel, same received power, same noise -- only the modulation
    differs. Spectral kurtosis is 1 for Gaussian interference *by
    construction*, so the canonical pre-detection detector cannot flag it at
    any power; a constant-envelope carrier drives the statistic to nearly
    zero and is flagged immediately.
    """
    m = 1024
    n_time = 4 * m

    def statistic(waveform):
        carrier = NarrowbandTransmitter(
            position_enu_m=TOWER_ENU_M,
            center_freq_hz=1.405e9,
            bandwidth_hz=0.0,
            received_power_jy=1.0e4,
            waveform=waveform,
            name="carrier",
        )
        sim = make_simulator(
            default_array,
            start_time,
            rfi_sources=[carrier],
            n_chan=16,
            n_blocks=1,
            n_time_per_block=n_time,
            noise_std=1.0,
        )
        block = sim.block(0)
        channel = int(np.argmax(np.abs(block.data[0]).mean(axis=1)))
        mask, sk = spectral_kurtosis_mask(block.data[0], m=m, return_statistic=True)
        return float(np.mean(sk[channel])), bool(mask[channel].all()), sk

    sk_constant, flagged_constant, _ = statistic("constant_envelope")
    sk_gaussian, flagged_gaussian, sk_all = statistic("gaussian")

    # The whole feature, in two numbers.
    assert sk_constant < 0.1, f"constant-envelope SK should collapse, got {sk_constant}"
    assert flagged_constant, "a constant-envelope carrier must be flagged"
    assert sk_gaussian == pytest.approx(1.0, abs=0.15), (
        f"a Gaussian carrier must look like noise to SK, got {sk_gaussian}"
    )
    assert not flagged_gaussian, "the Gaussian carrier is invisible to SK -- that is the point"

    # The Gaussian carrier is not merely unflagged in its own channel: it is
    # statistically indistinguishable from the noise-only channels.
    assert np.nanmean(sk_all) == pytest.approx(1.0, abs=0.1)


def test_constant_envelope_has_no_envelope_variance(default_array, start_time):
    """Pre-noise, every sample of a constant-envelope carrier has |v| equal."""
    carrier = make_tower(bandwidth_hz=0.0, waveform="constant_envelope")
    sim = make_simulator(
        default_array, start_time, rfi_sources=[carrier], n_blocks=1, n_time_per_block=512
    )
    data = sim.block(0).data
    channel = int(np.argmax(np.abs(data[0]).mean(axis=1)))

    modulus = np.abs(data[:, channel, :])
    assert modulus.std(axis=1).max() / modulus.mean() < 1e-5

    # The Gaussian default fluctuates by order unity, as an exponential
    # power distribution must.
    gaussian = make_tower(bandwidth_hz=0.0)
    gaussian_sim = make_simulator(
        default_array, start_time, rfi_sources=[gaussian], n_blocks=1, n_time_per_block=512
    )
    gaussian_data = gaussian_sim.block(0).data
    gaussian_modulus = np.abs(gaussian_data[:, channel, :])
    assert gaussian_modulus.std(axis=1).mean() / gaussian_modulus.mean() > 0.3


def test_constant_envelope_keeps_the_near_field_geometry(start_time):
    """The delay phase is applied identically for both waveforms.

    A carrier is only interference-like if it correlates between antennas
    with the right phase, so the visibility phase slope must still measure
    the exact path-delay difference -- including the 8.3 ns one that a
    plane-wave approximation says is zero.
    """
    array = three_antenna_array()
    tower = make_tower(
        center_freq_hz=1.405e9,
        bandwidth_hz=1.0e7,
        received_power_jy=100.0,
        waveform="constant_envelope",
    )
    sim = make_simulator(array, start_time, rfi_sources=[tower], n_chan=64, n_blocks=1)
    vis = correlate(sim.blocks(), fringe_stop=False)

    tau_s = path_delays_s(TOWER_ENU_M, array.antenna_positions_enu_m)
    for ant_1, ant_2 in ((0, 1), (0, 2)):
        row = vis.data[0, vis.baseline_index(ant_1, ant_2)]
        phase = np.unwrap(np.angle(row))
        slope = np.polyfit(vis.freq_hz - vis.freq_hz.mean(), phase, 1)[0]
        measured_delay_s = -slope / (2.0 * np.pi)
        assert measured_delay_s == pytest.approx(tau_s[ant_1] - tau_s[ant_2], rel=1e-4)

        # Fully coherent: the cross-power modulus is the geometric mean of
        # the two autopowers, not a fraction of it.
        auto_1 = np.abs(vis.data[0, vis.baseline_index(ant_1, ant_1)]).mean()
        auto_2 = np.abs(vis.data[0, vis.baseline_index(ant_2, ant_2)]).mean()
        assert np.abs(row).mean() == pytest.approx(np.sqrt(auto_1 * auto_2), rel=1e-5)


def test_constant_envelope_symbols_are_unit_modulus_and_held():
    """The waveform helper: unit modulus, and one symbol per hold."""
    samples = constant_envelope(np.random.default_rng(0), (3, 10), 4)
    assert samples.dtype == np.complex64
    np.testing.assert_allclose(np.abs(samples), 1.0, atol=1e-6)
    np.testing.assert_array_equal(samples[:, 0:4], samples[:, 0:1] * np.ones((1, 4)))
    assert samples.shape == (3, 10)
    # Different symbols do occur, i.e. the carrier is modulated.
    assert np.unique(samples).size > 1

    with pytest.raises(ValueError, match="samples_per_symbol"):
        constant_envelope(np.random.default_rng(0), (2, 4), 0)


def test_symbol_rate_tracks_the_emission_bandwidth(default_array, start_time):
    """One symbol per sample when the emission fills a channel, more below."""
    sim = make_simulator(default_array, start_time, n_blocks=1, n_time_per_block=64)
    ctx = sim.block_context(0, np.random.default_rng(0))
    tower = make_tower(waveform="constant_envelope")

    assert tower.symbol_length_samples(ctx, sim.chan_width_hz) == 1
    assert tower.symbol_length_samples(ctx, 10.0 * sim.chan_width_hz) == 1
    assert tower.symbol_length_samples(ctx, sim.chan_width_hz / 4.0) == 4
    # A pure carrier is unmodulated: one symbol for the whole block.
    assert tower.symbol_length_samples(ctx, 0.0) == ctx.n_time


def test_waveform_validation():
    with pytest.raises(ValueError, match="waveform must be one of"):
        make_tower(waveform="bpsk")


# ----------------------------------------------------------------------
# Clocked on/off patterns
# ----------------------------------------------------------------------
def test_periodic_envelope_mask_follows_the_frame_pattern(default_array, start_time):
    """A clocked envelope is deterministic, continuous and the right period."""
    period_s = 2.4e-3
    duty = 0.3
    tower = make_tower(envelope={"type": "periodic", "period_s": period_s, "duty": duty})
    sim = make_simulator(
        default_array, start_time, rfi_sources=[tower], n_blocks=4, n_time_per_block=1000
    )

    masks = [block.rfi_mask[0] for block in sim.blocks()]
    on = np.concatenate([mask.any(axis=0) for mask in masks])

    # The pattern is exactly the analytic one, across block boundaries: the
    # frames belong to the observation, not to a block.
    elapsed_s = np.arange(on.size) * sim.sample_period_s
    cycles = elapsed_s / period_s
    expected = (cycles - np.floor(cycles)) < duty
    np.testing.assert_array_equal(on, expected)
    assert on.mean() == pytest.approx(duty, abs=0.02)

    # The period is recoverable from the envelope alone.
    centered = on.astype(np.float64) - on.mean()
    correlation = np.correlate(centered, centered, mode="full")[on.size - 1 :]
    lag_samples = int(round(period_s / sim.sample_period_s))
    peak_lag = int(np.argmax(correlation[lag_samples // 2 :])) + lag_samples // 2
    assert peak_lag == pytest.approx(lag_samples, abs=2)

    # Deterministic: a different seed gives the identical pattern.
    other = make_simulator(
        default_array,
        start_time,
        rfi_sources=[make_tower(envelope={"type": "periodic", "period_s": period_s, "duty": duty})],
        n_blocks=4,
        n_time_per_block=1000,
        rng=np.random.default_rng(7),
    )
    np.testing.assert_array_equal(other.block(2).rfi_mask[0], masks[2])


def test_periodic_envelope_phase_shifts_the_frames(default_array, start_time):
    """`phase` slides the pattern by a fraction of a period."""

    def on_pattern(phase):
        tower = make_tower(
            envelope={"type": "periodic", "period_s": 1.0e-3, "duty": 0.5, "phase": phase}
        )
        sim = make_simulator(
            default_array, start_time, rfi_sources=[tower], n_blocks=1, n_time_per_block=1000
        )
        return sim.block(0).rfi_mask[0].any(axis=0)

    unshifted = on_pattern(0.0)
    inverted = on_pattern(0.5)
    # A half-period shift of a 50 % duty cycle inverts the pattern.
    assert np.mean(unshifted == ~inverted) > 0.99


def test_duty_cycle_and_envelope_are_mutually_exclusive():
    with pytest.raises(ValueError, match="duty_cycle and envelope"):
        make_tower(duty_cycle=0.5, envelope={"type": "periodic", "period_s": 1e-3, "duty": 0.5})


@pytest.mark.parametrize(
    ("envelope", "match"),
    [
        ({"type": "iid", "period_s": 1e-3}, "envelope type"),
        ({"type": "periodic", "period_s": 0.0}, "period_s"),
        ({"type": "periodic", "period_s": 1e-3, "duty": 1.5}, "duty"),
        ({"type": "periodic", "period_s": 1e-3, "rate_hz": 60.0}, "unexpected keys"),
        ("periodic", "envelope must be None or a mapping"),
    ],
)
def test_envelope_validation(envelope, match):
    with pytest.raises(ValueError, match=match):
        make_tower(envelope=envelope)


def test_periodic_impulses_land_at_the_configured_spacing(default_array, start_time):
    """A clocked pulse train fires at a fixed rate, continuous across blocks."""
    rate_hz = 600.0
    sparks = ImpulsiveBroadband(
        arrival={"type": "periodic", "rate_hz": rate_hz},
        received_power_jy=1.0e3,
        max_power_ratio=1.0,
        name="mains",
    )
    sim = make_simulator(
        default_array, start_time, rfi_sources=[sparks], n_blocks=4, n_time_per_block=1000
    )

    times_s = []
    for index, block in enumerate(sim.blocks()):
        hits = np.flatnonzero(block.rfi_mask[0].any(axis=0))
        times_s.append(index * block.duration_s + hits * sim.sample_period_s)
    times_s = np.concatenate(times_s)

    expected_spacing_s = 1.0 / rate_hz
    assert times_s.size == pytest.approx(rate_hz * 4 * sim.block_duration_s, abs=1)
    spacings_s = np.diff(times_s)
    np.testing.assert_allclose(spacings_s, expected_spacing_s, atol=2.0 * sim.sample_period_s)

    # Jitter spreads the spacings without changing the mean rate.
    jitter_s = 20.0 * sim.sample_period_s
    jittered = ImpulsiveBroadband(
        arrival={"type": "periodic", "rate_hz": rate_hz, "jitter_s": jitter_s},
        received_power_jy=1.0e3,
        max_power_ratio=1.0,
        name="mains",
    )
    jitter_sim = make_simulator(
        default_array, start_time, rfi_sources=[jittered], n_blocks=4, n_time_per_block=1000
    )
    jitter_times_s = []
    for index, block in enumerate(jitter_sim.blocks()):
        hits = np.flatnonzero(block.rfi_mask[0].any(axis=0))
        jitter_times_s.append(index * block.duration_s + hits * sim.sample_period_s)
    jitter_times_s = np.sort(np.concatenate(jitter_times_s))

    jitter_spacings_s = np.diff(jitter_times_s)
    assert jitter_spacings_s.std() > 2.0 * sim.sample_period_s
    assert jitter_spacings_s.max() <= expected_spacing_s + 2.0 * jitter_s
    assert jitter_spacings_s.mean() == pytest.approx(expected_spacing_s, rel=0.05)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"arrival": "regular"}, "arrival must be"),
        ({"arrival": {"type": "poisson"}}, "arrival type"),
        ({"arrival": {"type": "periodic", "rate_hz": 0.0}}, "rate_hz must be > 0"),
        ({"arrival": {"type": "periodic", "rate_hz": 60.0, "jitter_s": -1.0}}, "jitter_s"),
        ({"arrival": {"type": "periodic", "rate_hz": 60.0, "duty": 0.5}}, "unexpected keys"),
    ],
)
def test_impulsive_arrival_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ImpulsiveBroadband(**kwargs)


def test_impulsive_rate_is_given_exactly_once():
    """Either the Poisson rate or the periodic one -- never both, never neither."""
    with pytest.raises(ValueError, match="Poisson arrivals need a rate_hz"):
        ImpulsiveBroadband()
    with pytest.raises(ValueError, match="both set the event rate"):
        ImpulsiveBroadband(60.0, arrival={"type": "periodic", "rate_hz": 60.0})

    periodic = ImpulsiveBroadband(arrival={"type": "periodic", "rate_hz": 120.0})
    assert periodic.rate_hz == 120.0


# ----------------------------------------------------------------------
# Harmonic combs
# ----------------------------------------------------------------------
COMB_FUNDAMENTAL_HZ = 4.5e7
COMB_HARMONICS = (28, 29, 30, 31, 32, 33, 34, 35)
COMB_POSITION_ENU_M = np.array([600.0, -400.0, 10.0])


def make_comb(**kwargs):
    """An eight-harmonic comb, six of whose lines reach the wide test band."""
    options = dict(
        position_enu_m=COMB_POSITION_ENU_M,
        fundamental_hz=COMB_FUNDAMENTAL_HZ,
        harmonic_numbers=COMB_HARMONICS,
        received_powers_jy=400.0,
        bandwidth_hz=0.0,
        name="comb",
    )
    options.update(kwargs)
    return CombTransmitter(**options)


def make_wide_simulator(array, start_time, comb, **kwargs):
    """A 256 MHz band, wide enough to hold several harmonics of one comb."""
    options = dict(
        n_chan=256,
        chan_width_hz=1.0e6,
        n_blocks=1,
        n_time_per_block=64,
        noise_std=0.0,
        rng=np.random.default_rng(99),
    )
    options.update(kwargs)
    return make_simulator(array, start_time, rfi_sources=[comb], **options)


def test_comb_lines_land_at_exact_multiples_of_the_fundamental(default_array, start_time):
    """Every synthesized line sits at n times the fundamental."""
    comb = make_comb()
    sim = make_wide_simulator(default_array, start_time, comb)
    block = sim.block(0)

    in_band = comb.in_band_harmonics(sim.freq_hz, sim.chan_width_hz)
    skipped = comb.skipped_harmonics(sim.freq_hz, sim.chan_width_hz)
    assert in_band.tolist() == [29, 30, 31, 32, 33, 34]
    assert skipped.tolist() == [28, 35]

    occupied = np.flatnonzero(np.abs(block.data).max(axis=(0, 2)) > 0.0)
    expected = [int(np.argmin(np.abs(sim.freq_hz - n * COMB_FUNDAMENTAL_HZ))) for n in in_band]
    assert occupied.tolist() == sorted(expected)

    # Each occupied channel really is the nearest one to n * f0, to within
    # half a channel.
    for n, channel in zip(in_band, sorted(expected)):
        assert abs(sim.freq_hz[channel] - n * COMB_FUNDAMENTAL_HZ) <= 0.5 * sim.chan_width_hz

    # One source, one mask entry -- the union over the harmonics.
    assert block.rfi_source_names == ("comb",)
    assert int(block.rfi_mask[0].any(axis=1).sum()) == in_band.size


def test_comb_harmonics_share_position_and_coupling(default_array, start_time):
    """One device: every line carries the identical per-antenna pattern.

    With a constant-envelope waveform the received power of a line is
    deterministic, so "identical" can be asserted exactly rather than
    statistically.
    """
    n_antennas = default_array.n_antennas
    coupling = np.ones(n_antennas)
    coupling[2] = 4.0
    coupling[5] = 0.25

    comb = make_comb(coupling=coupling, waveform="constant_envelope")
    sim = make_wide_simulator(default_array, start_time, comb)
    block = sim.block(0)

    lines = np.flatnonzero(np.abs(block.data).max(axis=(0, 2)) > 0.0)
    assert lines.size == 6
    power = (np.abs(block.data[:, lines, :]) ** 2).mean(axis=2)  # (n_ant, n_lines)

    # The per-antenna pattern of every line is the same, and it is the one
    # the coupling and the shared geometry predict.
    patterns = power / power[0][np.newaxis, :]
    for i_line in range(1, lines.size):
        np.testing.assert_allclose(patterns[:, i_line], patterns[:, 0], rtol=1e-4)

    geometry = spreading_amplitudes(COMB_POSITION_ENU_M, default_array.antenna_positions_enu_m)
    expected = (coupling * geometry) ** 2
    np.testing.assert_allclose(patterns[:, 0], expected / expected[0], rtol=1e-4)

    np.testing.assert_allclose(block.rfi_coupling[0], coupling)


def test_comb_reports_per_harmonic_ground_truth(default_array, start_time):
    """Per-harmonic labels, tagged ``name[n]``, consistent with the union mask."""
    powers = dict(zip(COMB_HARMONICS, [1.0] * len(COMB_HARMONICS)))
    powers[32] = 400.0
    comb = make_comb(received_powers_jy=[powers[n] for n in COMB_HARMONICS])
    sim = make_wide_simulator(default_array, start_time, comb)
    block = sim.block(0)

    assert comb.harmonic_names == tuple(f"comb[{n}]" for n in COMB_HARMONICS)
    ctx = sim.block_context(0, sim.rfi_block_rngs(0)[0])
    masks = comb.harmonic_masks(ctx)
    assert masks.shape == (comb.n_harmonics, sim.n_chan, sim.n_time_per_block)

    labelled = {n: int(mask.any(axis=1).sum()) for n, mask in zip(COMB_HARMONICS, masks)}
    # Out-of-band harmonics are labelled silent; in-band ones occupy one
    # channel each -- including the weak lines, because each harmonic is
    # thresholded against its own peak.
    assert [n for n, count in labelled.items() if count] == [29, 30, 31, 32, 33, 34]
    assert all(count == 1 for count in labelled.values() if count)

    # The block's single mask uses the device-wide peak, so the lines 400x
    # below the strong one fall under the labelling threshold there. The
    # per-harmonic masks are how a weak line is recovered.
    assert int(block.rfi_mask[0].any(axis=1).sum()) == 1
    assert masks.any(axis=0).sum() > block.rfi_mask[0].sum()


def test_comb_harmonics_switch_together(default_array, start_time):
    """A shared envelope: the whole device keys on and off as one."""
    comb = make_comb(envelope={"type": "periodic", "period_s": 5.0e-4, "duty": 0.5})
    sim = make_wide_simulator(default_array, start_time, comb, n_time_per_block=512)
    ctx = sim.block_context(0, sim.rfi_block_rngs(0)[0])
    masks = comb.harmonic_masks(ctx)

    on_patterns = [mask.any(axis=0) for mask in masks if mask.any()]
    assert len(on_patterns) == 6
    for pattern in on_patterns[1:]:
        np.testing.assert_array_equal(pattern, on_patterns[0])
    assert on_patterns[0].mean() == pytest.approx(0.5, abs=0.05)


def test_comb_entirely_outside_the_band_raises(default_array, start_time):
    """Skipping single lines is normal; a comb that misses the band is not."""
    comb = make_comb(fundamental_hz=1.0e6, harmonic_numbers=(2, 3))
    sim = make_wide_simulator(default_array, start_time, comb)
    with pytest.raises(ValueError, match="none of which reach the simulated band"):
        sim.block(0)


def test_comb_power_per_line_is_measured_at_the_array_origin(default_array, start_time):
    """Each line delivers its own received_power_jy to the origin antenna."""
    array = ArrayConfig(
        antenna_positions_enu_m=np.array([[0.0, 0.0, 0.0], [40.0, -25.0, 0.0]]),
        latitude_deg=37.234,
        longitude_deg=-118.282,
        height_m=1222.0,
    )
    comb = make_comb(
        harmonic_numbers=(31, 32),
        received_powers_jy=(300.0, 100.0),
        waveform="constant_envelope",
    )
    sim = make_wide_simulator(array, start_time, comb, n_time_per_block=128)
    data = sim.block(0).data

    lines = np.flatnonzero(np.abs(data).max(axis=(0, 2)) > 0.0)
    powers = (np.abs(data[0, lines, :]) ** 2).mean(axis=1)
    np.testing.assert_allclose(powers, [300.0, 100.0], rtol=1e-4)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"fundamental_hz": 0.0}, "fundamental_hz"),
        ({"harmonic_numbers": ()}, "at least one harmonic"),
        ({"harmonic_numbers": (0, 2)}, ">= 1"),
        ({"harmonic_numbers": (2, 2)}, "unique"),
        ({"received_powers_jy": (1.0, 2.0)}, "one value or one per harmonic"),
        ({"received_powers_jy": -1.0}, "received_powers_jy"),
        ({"bandwidth_hz": -1.0}, "bandwidth_hz"),
        ({"waveform": "am"}, "waveform"),
    ],
)
def test_comb_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_comb(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"fundamental_hz": float("nan")}, "fundamental_hz"),
        ({"fundamental_hz": float("inf")}, "fundamental_hz"),
        ({"bandwidth_hz": float("nan")}, "bandwidth_hz"),
        ({"bandwidth_hz": float("inf")}, "bandwidth_hz"),
        ({"received_powers_jy": float("nan")}, "received_powers_jy"),
        ({"received_powers_jy": float("inf")}, "received_powers_jy"),
        ({"received_powers_jy": (1.0, float("nan"), 1.0)}, "received_powers_jy"),
    ],
)
def test_comb_rejects_non_finite_parameters(kwargs, match):
    """As with the narrowband guards, NaN and Inf must not pass a ``< 0`` check."""
    if "received_powers_jy" in kwargs and isinstance(kwargs["received_powers_jy"], tuple):
        kwargs = dict(kwargs, harmonic_numbers=(28, 29, 30))
    with pytest.raises(ValueError, match=match):
        make_comb(**kwargs)


def test_comb_powers_follow_the_sorted_harmonic_order():
    """Per-harmonic powers are matched up after sorting, not before."""
    comb = make_comb(harmonic_numbers=(32, 30), received_powers_jy=(5.0, 9.0))
    assert comb.harmonic_numbers.tolist() == [30, 32]
    np.testing.assert_allclose(comb.received_powers_jy, [9.0, 5.0])
    assert comb.harmonic_freqs_hz.tolist() == [30 * COMB_FUNDAMENTAL_HZ, 32 * COMB_FUNDAMENTAL_HZ]
