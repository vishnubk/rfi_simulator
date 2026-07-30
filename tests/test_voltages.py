"""Tests for rfi_simulator.voltages.

The two tests that matter most here are:

* `test_delay_phase_uses_rf_frequency` -- the per-antenna phase must be
  ``exp(-2 pi i f_rf tau)`` with the *RF* frequency of the channel. Using
  the baseband offset alone still makes fringes, so only a direct check
  against the RF phase catches it.
* `test_source_realization_is_shared_and_noise_is_independent` -- one sky
  signal for all antennas, independent receiver noise per antenna.
"""

import numpy as np
import pytest
from astropy import units as u
from conftest import SOURCE_L, SOURCE_M, zenith_phase_center

from rfi_simulator import ArrayConfig, PointSource, VoltageSimulator
from rfi_simulator.delays import (
    earth_location,
    geometric_delays_s,
    source_unit_vectors_enu,
)


def make_simulator(array, start_time, sources=(), **kwargs):
    """Small-but-real simulator: 16 channels, 3 blocks of 64 samples."""
    options = dict(
        n_chan=16,
        n_blocks=3,
        n_time_per_block=64,
        noise_std=0.0,
        rng=np.random.default_rng(20261001),
    )
    options.update(kwargs)
    phase_center = zenith_phase_center(array, start_time, duration_s=0.1)
    return VoltageSimulator(array, phase_center, start_time, sources, **options)


def test_block_shapes_and_dtypes(default_array, start_time):
    """Blocks are [n_ant, n_chan, n_time] complex64, one per integration."""
    sim = make_simulator(default_array, start_time, noise_std=1.0)
    blocks = list(sim.blocks())

    assert len(blocks) == 3
    for block in blocks:
        assert block.data.shape == (10, 16, 64)
        assert block.data.dtype == np.complex64
        assert block.n_antennas == 10
        assert block.n_chan == 16
        assert block.n_time == 64


def test_blocks_is_a_generator(default_array, start_time):
    """`blocks()` streams; it does not materialize the observation."""
    sim = make_simulator(default_array, start_time, noise_std=1.0)
    stream = sim.blocks()
    assert not isinstance(stream, list)
    first = next(stream)
    assert first.data.shape[0] == 10


def test_frequency_axis_is_ascending_and_centered(default_array, start_time):
    """RF channel centers are ascending and symmetric about the band center."""
    sim = make_simulator(default_array, start_time)
    freq_hz = sim.freq_hz

    assert freq_hz.shape == (16,)
    assert np.all(np.diff(freq_hz) > 0)
    assert freq_hz.mean() == pytest.approx(sim.center_freq_hz, rel=1e-12)
    np.testing.assert_allclose(np.diff(freq_hz), sim.chan_width_hz, rtol=1e-12)
    assert sim.bandwidth_hz == pytest.approx(16 * sim.chan_width_hz)


def test_default_band_matches_the_design(default_array, start_time):
    """The shipped defaults are the DSA-110-shaped band from the design doc."""
    sim = VoltageSimulator(
        default_array,
        zenith_phase_center(default_array, start_time, 2.0),
        start_time,
        rng=np.random.default_rng(0),
    )
    assert sim.n_chan == 384
    assert sim.center_freq_hz == pytest.approx(1.405e9)
    assert sim.chan_width_hz == pytest.approx(30517.578125)
    assert sim.bandwidth_hz == pytest.approx(11.71875e6)
    assert sim.sample_period_s == pytest.approx(32.768e-6)
    assert sim.block_duration_s == pytest.approx(32.768e-3)
    assert sim.duration_s == pytest.approx(61 * 32.768e-3)


def test_delay_phase_uses_rf_frequency(default_array, start_time):
    """Antenna phases follow exp(-2 pi i f_RF tau), not the baseband offset.

    With a single source and no noise, every antenna carries the same
    sky spectrum times a pure per-(antenna, channel) phase, so dividing
    one antenna by another isolates that phase exactly.
    """
    array = default_array
    phase_center = zenith_phase_center(array, start_time, duration_s=0.1)
    source = PointSource.from_lm(phase_center, (SOURCE_L, SOURCE_M), flux_jy=1.0)
    sim = make_simulator(array, start_time, [source])

    block = sim.block(1)
    location = earth_location(array)
    s_hat = source_unit_vectors_enu(source.coord, block.center_time, location)
    tau_s = geometric_delays_s(array.antenna_positions_enu_m, s_hat)

    ratio = block.data[:, :, 0] / block.data[0, :, 0][np.newaxis, :]
    expected = np.exp(-2j * np.pi * sim.freq_hz[np.newaxis, :] * (tau_s - tau_s[0])[:, np.newaxis])
    np.testing.assert_allclose(ratio, expected, atol=2e-4)

    # The wrong-but-plausible choice -- baseband offsets only -- is grossly
    # different, which is exactly why this test exists.
    baseband = sim.freq_hz - sim.center_freq_hz
    wrong = np.exp(-2j * np.pi * baseband[np.newaxis, :] * (tau_s - tau_s[0])[:, np.newaxis])
    assert np.abs(ratio - wrong).max() > 0.5


def test_delays_change_between_blocks(default_array, start_time):
    """Earth rotation is in the code path: blocks do not share one delay."""
    sim = make_simulator(default_array, start_time, n_blocks=2, n_time_per_block=1000)
    first, second = sim.block(0), sim.block(1)
    assert not np.array_equal(first.phase_center_delays_s, second.phase_center_delays_s)
    assert not np.array_equal(first.e_l_enu, second.e_l_enu)


def test_source_realization_is_shared_and_noise_is_independent(default_array, start_time):
    """One sky signal for all antennas; receiver noise independent per antenna."""
    array = default_array
    phase_center = zenith_phase_center(array, start_time, duration_s=0.1)
    source = PointSource.from_lm(phase_center, (SOURCE_L, SOURCE_M), flux_jy=1.0)

    # Source only: de-rotating each antenna's delay phase must recover one
    # identical spectrum on every antenna.
    sim = make_simulator(array, start_time, [source], n_chan=8, n_time_per_block=32)
    block = sim.block(0)
    location = earth_location(array)
    s_hat = source_unit_vectors_enu(source.coord, block.center_time, location)
    tau_s = geometric_delays_s(array.antenna_positions_enu_m, s_hat)
    derotated = block.data * np.exp(
        2j * np.pi * sim.freq_hz[np.newaxis, :, np.newaxis] * tau_s[:, np.newaxis, np.newaxis]
    )
    for i_ant in range(1, array.n_antennas):
        np.testing.assert_allclose(derotated[i_ant], derotated[0], atol=2e-4)

    # Noise only: antennas must be uncorrelated.
    noise_sim = make_simulator(
        array, start_time, [], noise_std=1.0, n_chan=8, n_time_per_block=4000
    )
    noise = noise_sim.block(0).data
    n_samples = noise.shape[1] * noise.shape[2]
    cross = np.mean(noise[0] * np.conjugate(noise[1]))
    auto = np.mean(np.abs(noise[0]) ** 2)
    assert auto == pytest.approx(1.0, rel=0.1)
    assert abs(cross) < 6.0 / np.sqrt(n_samples)


def test_noise_power_matches_noise_std(default_array, start_time):
    """E|n|^2 equals noise_std**2, so noise_std**2 acts as an SEFD in Jy."""
    sim = make_simulator(
        default_array, start_time, [], noise_std=2.0, n_chan=8, n_time_per_block=4000
    )
    power = np.mean(np.abs(sim.block(0).data) ** 2)
    assert power == pytest.approx(4.0, rel=0.02)


def test_source_power_matches_flux(default_array, start_time):
    """E|v|^2 for a noiseless single-source run equals the source flux."""
    array = default_array
    phase_center = zenith_phase_center(array, start_time, duration_s=0.1)
    source = PointSource.from_lm(phase_center, (SOURCE_L, SOURCE_M), flux_jy=3.0)
    sim = make_simulator(array, start_time, [source], n_chan=8, n_time_per_block=4000)
    power = np.mean(np.abs(sim.block(0).data) ** 2)
    assert power == pytest.approx(3.0, rel=0.05)


def test_reproducible_with_the_same_seed(default_array, start_time):
    """The same seed gives bit-identical voltages; a different seed does not."""
    array = default_array
    phase_center = zenith_phase_center(array, start_time, duration_s=0.1)
    source = PointSource.from_lm(phase_center, (SOURCE_L, SOURCE_M), flux_jy=1.0)

    def run(seed):
        sim = make_simulator(
            array, start_time, [source], noise_std=1.0, rng=np.random.default_rng(seed)
        )
        return np.stack([block.data for block in sim.blocks()])

    np.testing.assert_array_equal(run(7), run(7))
    assert not np.array_equal(run(7), run(8))


def test_zero_flux_source_is_a_no_op(default_array, start_time):
    """A zero-flux source contributes nothing and consumes no randomness."""
    array = default_array
    phase_center = zenith_phase_center(array, start_time, duration_s=0.1)
    faint = PointSource.from_lm(phase_center, (SOURCE_L, SOURCE_M), flux_jy=0.0)

    with_source = make_simulator(
        array, start_time, [faint], noise_std=1.0, rng=np.random.default_rng(3)
    ).block(0)
    without = make_simulator(
        array, start_time, [], noise_std=1.0, rng=np.random.default_rng(3)
    ).block(0)
    np.testing.assert_array_equal(with_source.data, without.data)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_chan": 0}, "n_chan"),
        ({"n_blocks": 0}, "n_blocks"),
        ({"n_time_per_block": 0}, "n_time_per_block"),
        ({"chan_width_hz": 0.0}, "chan_width_hz"),
        ({"noise_std": -1.0}, "noise_std"),
    ],
)
def test_invalid_parameters_raise(default_array, start_time, kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_simulator(default_array, start_time, **kwargs)


def test_block_index_is_range_checked(default_array, start_time):
    sim = make_simulator(default_array, start_time)
    with pytest.raises(ValueError, match="out of range"):
        sim.block(sim.n_blocks)


def test_block_times_advance_by_the_block_duration(default_array, start_time):
    """Block start times tile the observation with no gaps or overlaps."""
    sim = make_simulator(default_array, start_time)
    starts = sim.block_start_times()
    offsets_s = (starts - start_time).to_value(u.s)
    np.testing.assert_allclose(offsets_s, np.arange(sim.n_blocks) * sim.block_duration_s, atol=1e-9)
    centers = sim.block_center_times()
    np.testing.assert_allclose(
        (centers - starts).to_value(u.s), 0.5 * sim.block_duration_s, atol=1e-9
    )


def test_two_sources_add_in_power(default_array, start_time):
    """Independent sources add incoherently: total power is the flux sum."""
    array = default_array
    phase_center = zenith_phase_center(array, start_time, duration_s=0.1)
    sources = [
        PointSource.from_lm(phase_center, (0.005, 0.0), flux_jy=1.0),
        PointSource.from_lm(phase_center, (-0.004, 0.006), flux_jy=2.0),
    ]
    sim = make_simulator(array, start_time, sources, n_chan=8, n_time_per_block=4000)
    power = np.mean(np.abs(sim.block(0).data) ** 2)
    assert power == pytest.approx(3.0, rel=0.05)


def test_non_scalar_phase_center_is_rejected(default_array, start_time):
    from astropy.coordinates import SkyCoord

    coords = SkyCoord(ra=[10.0, 20.0] * u.deg, dec=[0.0, 1.0] * u.deg, frame="icrs")
    with pytest.raises(ValueError, match="scalar SkyCoord"):
        VoltageSimulator(default_array, coords, start_time, rng=np.random.default_rng(0))


def test_simulator_accepts_a_custom_array_layout(start_time):
    """Antenna positions are an input, never baked in."""
    array = ArrayConfig(
        antenna_positions_enu_m=np.array([[0.0, 0.0, 0.0], [37.0, -11.0, 0.0]]),
        latitude_deg=37.234,
        longitude_deg=-118.282,
        height_m=1222.0,
    )
    sim = make_simulator(array, start_time, noise_std=1.0)
    assert sim.block(0).data.shape[0] == 2
