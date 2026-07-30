"""Tests for rfi_simulator.correlator.

Covers acceptance criteria 4 (autocorrelations real and positive) and 5
(zero-baseline) from ``docs/design_stage2.md``, plus the conjugation and
fringe-stopping conventions.
"""

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from conftest import SOURCE_L, SOURCE_M, zenith_phase_center

from rfi_simulator import (
    ArrayConfig,
    PointSource,
    VoltageSimulator,
    baseline_index_pairs,
    correlate,
)


def run(array, start_time, sources=(), *, include_autos=True, **kwargs):
    """Simulate and correlate a short observation."""
    options = dict(
        n_chan=32,
        n_blocks=4,
        n_time_per_block=250,
        noise_std=1.0,
        rng=np.random.default_rng(20261002),
    )
    options.update(kwargs)
    phase_center = zenith_phase_center(array, start_time, duration_s=0.5)
    sim = VoltageSimulator(array, phase_center, start_time, sources, **options)
    return sim, correlate(sim.blocks(), include_autos=include_autos)


def test_baseline_index_pairs_counts():
    """Autos included by default; excluding them drops n_ant rows."""
    with_autos = baseline_index_pairs(10)
    without = baseline_index_pairs(10, include_autos=False)
    assert with_autos.shape == (55, 2)
    assert without.shape == (45, 2)
    assert np.all(with_autos[:, 0] <= with_autos[:, 1])
    assert np.all(without[:, 0] < without[:, 1])


def test_visibility_shape_and_metadata(default_array, start_time):
    """Visibilities are [n_int, n_base, n_chan] with autos included."""
    sim, vis = run(default_array, start_time)

    assert vis.data.shape == (4, 55, 32)
    assert vis.data.dtype == np.complex64
    assert vis.n_int == 4
    assert vis.n_baselines == 55
    assert vis.n_chan == 32
    assert vis.auto_mask.sum() == 10
    assert vis.cross_mask.sum() == 45
    assert vis.n_samples == 250
    assert vis.integration_time_s == pytest.approx(250 * sim.sample_period_s)
    assert vis.time_mjd.shape == (4,)
    assert np.all(np.diff(vis.time_mjd) > 0)
    np.testing.assert_allclose(vis.freq_hz, sim.freq_hz)
    assert vis.baseline_vectors_enu_m.shape == (55, 3)
    assert vis.e_l_enu.shape == (4, 3)


def test_baseline_vectors_follow_the_conjugation_convention(default_array, start_time):
    """The stored baseline vector is r_i - r_j for the pair V_ij = <v_i v_j*>."""
    _, vis = run(default_array, start_time, n_chan=4, n_time_per_block=8, n_blocks=1)
    positions = default_array.antenna_positions_enu_m
    expected = positions[vis.ant_1] - positions[vis.ant_2]
    np.testing.assert_allclose(vis.baseline_vectors_enu_m, expected, atol=1e-12)


def test_autocorrelations_are_real_and_positive(default_array, start_time):
    """Acceptance criterion 4."""
    array = default_array
    phase_center = zenith_phase_center(array, start_time, duration_s=0.5)
    source = PointSource.from_lm(phase_center, (SOURCE_L, SOURCE_M), flux_jy=2.0)
    _, vis = run(array, start_time, [source], noise_std=1.5)

    autos = vis.data[:, vis.auto_mask, :]
    assert np.all(autos.real > 0.0)
    # "Real to numerical precision": the residual imaginary part is pure
    # complex64 round-off from the accumulation order, ~1e-9 relative.
    assert np.all(np.abs(autos.imag) < 1e-6 * autos.real)

    # Autos measure total system power: source flux + noise power.
    assert autos.real.mean() == pytest.approx(2.0 + 1.5**2, rel=0.02)


def test_zero_baseline_visibility_is_real_and_equals_the_flux(start_time):
    """Acceptance criterion 5: co-located antennas see the same sky."""
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],  # co-located with antenna 0
            [40.0, -25.0, 0.0],
        ]
    )
    with pytest.warns(UserWarning, match="duplicate"):
        array = ArrayConfig(
            antenna_positions_enu_m=positions,
            latitude_deg=37.234,
            longitude_deg=-118.282,
            height_m=1222.0,
        )

    phase_center = zenith_phase_center(array, start_time, duration_s=0.5)
    source = PointSource.from_lm(phase_center, (SOURCE_L, SOURCE_M), flux_jy=1.0)

    # Noiseless: the two co-located antennas record identical voltages.
    sim_clean, vis_clean = run(
        array, start_time, [source], noise_std=0.0, n_chan=8, n_time_per_block=64, n_blocks=1
    )
    block = sim_clean.block(0)
    np.testing.assert_array_equal(block.data[0], block.data[1])

    zero_baseline = vis_clean.baseline_index(0, 1)
    np.testing.assert_allclose(vis_clean.data[:, zero_baseline, :].imag, 0.0, atol=1e-6)

    # With noise: still real to within the noise, and equal to the flux.
    _, vis = run(array, start_time, [source], noise_std=1.0, n_chan=64, n_time_per_block=500)
    zero_baseline = vis.baseline_index(0, 1)
    values = vis.data[:, zero_baseline, :]
    assert values.real.mean() == pytest.approx(1.0, abs=0.05)
    assert abs(values.imag.mean()) < 0.05

    # The auto of antenna 0 and the zero-length cross differ only by the
    # receiver noise power.
    auto = vis.data[:, vis.baseline_index(0, 0), :].real.mean()
    assert auto - values.real.mean() == pytest.approx(1.0, abs=0.05)


def test_source_at_phase_center_gives_a_real_visibility(default_array, start_time):
    """Fringe stopping works: a source at (0, 0) has zero fringe phase."""
    array = default_array
    phase_center = zenith_phase_center(array, start_time, duration_s=0.5)
    source = PointSource.from_lm(phase_center, (0.0, 0.0), flux_jy=1.0)
    _, vis = run(array, start_time, [source], noise_std=0.0, n_chan=32, n_time_per_block=250)

    cross = vis.data[:, vis.cross_mask, :]
    phases_rad = np.angle(cross)
    assert np.abs(phases_rad).max() < 1e-3
    assert cross.real.mean() == pytest.approx(1.0, rel=0.02)


def test_offset_source_gives_a_fringe(default_array, start_time):
    """A source away from the phase center leaves a residual fringe phase."""
    array = default_array
    phase_center = zenith_phase_center(array, start_time, duration_s=0.5)
    source = PointSource.from_lm(phase_center, (SOURCE_L, SOURCE_M), flux_jy=1.0)
    _, vis = run(array, start_time, [source], noise_std=0.0)

    cross = vis.data[:, vis.cross_mask, :]
    assert np.abs(np.angle(cross)).max() > 0.5
    # ...but the amplitude is still the source flux on every baseline.
    np.testing.assert_allclose(np.abs(cross).mean(), 1.0, rtol=0.05)


def test_fringe_stop_can_be_disabled(default_array, start_time):
    """Without fringe stopping, a phase-center source still carries its delay phase.

    The phase center here is deliberately 40 degrees off the zenith: over a
    flat array a zenith phase center has zero geometric delay, so fringe
    stopping would be a no-op and the test would prove nothing.
    """
    array = default_array
    zenith = zenith_phase_center(array, start_time, duration_s=0.5)
    phase_center = SkyCoord(ra=zenith.ra, dec=zenith.dec - 40.0 * u.deg, frame="icrs")
    source = PointSource.from_lm(phase_center, (0.0, 0.0), flux_jy=1.0)
    options = dict(
        n_chan=32,
        n_blocks=2,
        n_time_per_block=100,
        noise_std=0.0,
    )

    stopped = correlate(
        VoltageSimulator(
            array, phase_center, start_time, [source], rng=np.random.default_rng(5), **options
        ).blocks()
    )
    unstopped = correlate(
        VoltageSimulator(
            array, phase_center, start_time, [source], rng=np.random.default_rng(5), **options
        ).blocks(),
        fringe_stop=False,
    )

    assert np.abs(np.angle(stopped.data[:, stopped.cross_mask, :])).max() < 1e-3
    assert np.abs(np.angle(unstopped.data[:, unstopped.cross_mask, :])).max() > 0.5


def test_correlate_can_drop_autos(default_array, start_time):
    _, vis = run(default_array, start_time, include_autos=False, n_chan=4, n_time_per_block=8)
    assert vis.n_baselines == 45
    assert not np.any(vis.auto_mask)


def test_correlate_rejects_an_empty_stream():
    with pytest.raises(ValueError, match="no voltage blocks"):
        correlate(iter([]))


def test_baseline_index_lookup_is_order_insensitive(default_array, start_time):
    _, vis = run(default_array, start_time, n_chan=4, n_time_per_block=8, n_blocks=1)
    assert vis.baseline_index(3, 7) == vis.baseline_index(7, 3)
    with pytest.raises(KeyError):
        vis.baseline_index(0, 99)


def test_conjugate_symmetry_of_the_correlation(default_array, start_time):
    """<v_i v_j*> and <v_j v_i*> are complex conjugates by construction."""
    array = default_array
    phase_center = zenith_phase_center(array, start_time, duration_s=0.5)
    source = PointSource.from_lm(phase_center, (SOURCE_L, SOURCE_M), flux_jy=1.0)
    sim = VoltageSimulator(
        array,
        phase_center,
        start_time,
        [source],
        n_chan=8,
        n_blocks=1,
        n_time_per_block=64,
        noise_std=0.0,
        rng=np.random.default_rng(11),
    )
    block = sim.block(0)
    v_ij = np.mean(block.data[2] * np.conjugate(block.data[5]), axis=-1)
    v_ji = np.mean(block.data[5] * np.conjugate(block.data[2]), axis=-1)
    np.testing.assert_allclose(v_ij, np.conjugate(v_ji), atol=1e-6)

    vis = correlate([block])
    row = vis.data[0, vis.baseline_index(2, 5), :]
    tau = block.phase_center_delays_s
    stop = np.exp(2j * np.pi * (tau[2] - tau[5]) * block.freq_hz)
    np.testing.assert_allclose(row, (v_ij * stop).astype(np.complex64), rtol=1e-3, atol=1e-5)
