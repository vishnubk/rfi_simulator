"""Tests for rfi_simulator.calibration and its integration with correlate().

The load-bearing tests here are:

* `test_defaults_are_off` / `test_calibration_errors_default_off_is_bit_identical`
  -- the feature must be a pure no-op unless switched on, and switching it
  on with all parameters at zero must reproduce the data `correlate`
  produced before the feature existed.
* `test_phase_errors_produce_the_expected_coherence_loss` -- ties the
  configured phase-error rms to the analytically expected dirty-image
  peak loss, ``exp(-sigma_phi**2)``.
* `test_delay_errors_produce_a_linear_phase_ramp` -- a residual delay
  error must show up as a phase slope of exactly ``2 pi tau_ij f``.
* `test_amplitude_errors_do_not_touch_the_voltages` -- the error is
  applied purely at the visibility level: `VoltageBlock.data` is
  bit-identical whether or not `calibration_errors` is later applied in
  `correlate`.
* `test_calibration_error_ground_truth_matches_the_applied_factors` -- the
  factors recorded on `Visibilities.calibration_error_gains` must be
  exactly what was multiplied onto the data, not merely correlated with
  it.
"""

import numpy as np
import pytest
from conftest import SOURCE_L, SOURCE_M, random_flat_array, zenith_phase_center

from rfi_simulator import (
    CalibrationErrors,
    PointSource,
    VoltageSimulator,
    correlate,
    dirty_image,
)

FREQ_HZ = np.linspace(1.40e9, 1.42e9, 96)


def make_simulator(array, start_time, sources=(), **kwargs):
    """Small-but-real simulator, matching the other test modules' conventions."""
    options = dict(
        n_chan=16,
        n_blocks=2,
        n_time_per_block=256,
        noise_std=1.0,
        rng=np.random.default_rng(20261001),
    )
    options.update(kwargs)
    phase_center = zenith_phase_center(array, start_time, duration_s=0.1)
    return VoltageSimulator(array, phase_center, start_time, sources, **options)


# ----------------------------------------------------------------------
# Construction, repeatability, validation
# ----------------------------------------------------------------------
def test_identity_model_is_exactly_unit_factor():
    model = CalibrationErrors.identity(5)
    factors = model.factors(FREQ_HZ)
    assert factors.shape == (5, FREQ_HZ.size)
    np.testing.assert_array_equal(factors, np.ones_like(factors))


def test_defaults_are_off():
    """`from_params` with no options is the identity, and needs no seed."""
    model = CalibrationErrors.from_params(4)
    np.testing.assert_array_equal(model.factors(FREQ_HZ), np.ones((4, FREQ_HZ.size)))


def test_same_seed_gives_identical_factors_and_different_seeds_do_not():
    kwargs = dict(phase_error_deg_rms=8.0, delay_error_ns_rms=0.3, amplitude_error_db_rms=0.4)
    first = CalibrationErrors.from_params(12, seed=11, **kwargs).factors(FREQ_HZ)
    again = CalibrationErrors.from_params(12, seed=11, **kwargs).factors(FREQ_HZ)
    other = CalibrationErrors.from_params(12, seed=12, **kwargs).factors(FREQ_HZ)
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


def test_effects_are_independently_seeded():
    """Adding a delay error leaves the phase-error draw untouched."""
    plain = CalibrationErrors.from_params(8, seed=99, phase_error_deg_rms=6.0)
    with_delay = CalibrationErrors.from_params(
        8, seed=99, phase_error_deg_rms=6.0, delay_error_ns_rms=0.5
    )
    np.testing.assert_array_equal(plain.phase_error_rad, with_delay.phase_error_rad)
    assert not np.all(with_delay.delay_error_s == 0.0)


def test_model_is_frozen_and_arrays_are_read_only():
    model = CalibrationErrors.from_params(4, seed=1, phase_error_deg_rms=5.0)
    with pytest.raises(Exception):
        model.phase_error_rad = np.ones(4)
    with pytest.raises(ValueError):
        model.phase_error_rad[0] = 2.0
    factors = model.factors(FREQ_HZ)
    factors[0, 0] = 123.0
    assert model.factors(FREQ_HZ)[0, 0] != 123.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_antennas": 0}, "n_antennas"),
        ({"phase_error_deg_rms": -1.0}, "phase_error_deg_rms"),
        ({"delay_error_ns_rms": -1.0}, "delay_error_ns_rms"),
        ({"amplitude_error_db_rms": -1.0}, "amplitude_error_db_rms"),
        ({"phase_error_deg_rms": 5.0, "seed": None}, "needs an rng or a seed"),
        ({"seed": 3, "rng": np.random.default_rng(0)}, "either rng or seed"),
    ],
)
def test_from_params_validates(kwargs, match):
    options = dict(n_antennas=6, seed=4)
    options.update(kwargs)
    with pytest.raises(ValueError, match=match):
        CalibrationErrors.from_params(**options)


@pytest.mark.parametrize(
    ("param", "match"),
    [
        ("phase_error_deg_rms", "phase_error_deg_rms"),
        ("delay_error_ns_rms", "delay_error_ns_rms"),
        ("amplitude_error_db_rms", "amplitude_error_db_rms"),
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_from_params_rejects_non_finite_parameters(param, match, value):
    options = dict(n_antennas=6, seed=4)
    options[param] = value
    with pytest.raises(ValueError, match=match):
        CalibrationErrors.from_params(**options)


def test_factors_reject_non_finite_freq_hz():
    model = CalibrationErrors.from_params(4, seed=4, phase_error_deg_rms=5.0)
    bad_freq = FREQ_HZ.copy()
    bad_freq[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        model.factors(bad_freq)


def test_constructor_validates_shapes_and_finiteness():
    with pytest.raises(ValueError, match="same shape"):
        CalibrationErrors(
            phase_error_rad=np.zeros(4), delay_error_s=np.zeros(3), amplitude_error_db=np.zeros(4)
        )
    with pytest.raises(ValueError, match="non-finite"):
        CalibrationErrors(
            phase_error_rad=np.array([0.0, np.nan]),
            delay_error_s=np.zeros(2),
            amplitude_error_db=np.zeros(2),
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_constructor_rejects_non_finite_reference_freq_hz(value):
    with pytest.raises(ValueError, match="reference_freq_hz"):
        CalibrationErrors(
            phase_error_rad=np.zeros(2),
            delay_error_s=np.zeros(2),
            amplitude_error_db=np.zeros(2),
            reference_freq_hz=value,
        )


# ----------------------------------------------------------------------
# Statistics: the parameters must be measurable back out
# ----------------------------------------------------------------------
def test_phase_error_rms_matches_the_configured_degrees():
    rms_deg = 12.0
    model = CalibrationErrors.from_params(4000, seed=2026, phase_error_deg_rms=rms_deg)
    measured_deg = np.rad2deg(np.std(model.phase_error_rad))
    assert measured_deg == pytest.approx(rms_deg, rel=0.05)


def test_delay_error_rms_matches_the_configured_ns():
    rms_ns = 0.4
    model = CalibrationErrors.from_params(4000, seed=17, delay_error_ns_rms=rms_ns)
    measured_ns = np.std(model.delay_error_s) * 1e9
    assert measured_ns == pytest.approx(rms_ns, rel=0.05)


def test_amplitude_error_rms_matches_the_configured_db():
    rms_db = 0.6
    model = CalibrationErrors.from_params(4000, seed=9, amplitude_error_db_rms=rms_db)
    measured_db = np.std(model.amplitude_error_db)
    assert measured_db == pytest.approx(rms_db, rel=0.05)


# ----------------------------------------------------------------------
# Application point: correlate(calibration_errors=...)
# ----------------------------------------------------------------------
def test_calibration_errors_default_off_is_bit_identical(default_array, start_time):
    """None, and an all-zero model, both reproduce the uncalibrated-error data."""
    source = PointSource.from_lm(
        zenith_phase_center(default_array, start_time, 0.1), (SOURCE_L, SOURCE_M), flux_jy=2.0
    )
    sim = make_simulator(default_array, start_time, [source])
    blocks = list(sim.blocks())
    plain = correlate(blocks)
    identity = correlate(blocks, calibration_errors=CalibrationErrors.identity(sim.n_antennas))
    np.testing.assert_array_equal(identity.data, plain.data)
    assert plain.calibration_error_gains is None
    assert identity.calibration_error_gains is not None
    np.testing.assert_array_equal(
        identity.calibration_error_gains, np.ones_like(identity.calibration_error_gains)
    )


def test_calibration_errors_do_not_touch_the_voltages(default_array, start_time):
    """The error lives purely in correlate(): VoltageBlock.data never changes."""
    sim = make_simulator(default_array, start_time)
    block = sim.block(0)
    other_block = make_simulator(default_array, start_time).block(0)
    np.testing.assert_array_equal(block.data, other_block.data)

    # Applying calibration_errors in correlate() cannot have touched the
    # block objects that were already built.
    errors = CalibrationErrors.from_params(
        sim.n_antennas, seed=3, amplitude_error_db_rms=1.0, phase_error_deg_rms=10.0
    )
    correlate([block], calibration_errors=errors)
    np.testing.assert_array_equal(block.data, other_block.data)


def test_visibilities_scale_as_ci_cj_conjugate(default_array, start_time):
    """correlate() applies c_i(f) c_j(f)*, the baseline structure a
    calibration exercise has to solve for."""
    source = PointSource.from_lm(
        zenith_phase_center(default_array, start_time, 0.1), (SOURCE_L, SOURCE_M), flux_jy=5.0
    )
    sim = make_simulator(default_array, start_time, [source])
    blocks = list(sim.blocks())
    plain = correlate(blocks)
    errors = CalibrationErrors.from_params(
        sim.n_antennas,
        seed=44,
        phase_error_deg_rms=15.0,
        delay_error_ns_rms=0.2,
        amplitude_error_db_rms=0.5,
    )
    miscalibrated = correlate(blocks, calibration_errors=errors)

    c = errors.factors(plain.freq_hz)
    factor = c[plain.ant_1] * np.conjugate(c[plain.ant_2])  # (n_baselines, n_chan)
    np.testing.assert_allclose(
        miscalibrated.data, plain.data * factor.astype(np.complex64), rtol=2e-5, atol=2e-5
    )


def test_calibration_error_ground_truth_matches_the_applied_factors(default_array, start_time):
    sim = make_simulator(default_array, start_time)
    blocks = list(sim.blocks())
    errors = CalibrationErrors.from_params(
        sim.n_antennas, seed=8, phase_error_deg_rms=9.0, delay_error_ns_rms=0.1
    )
    vis = correlate(blocks, calibration_errors=errors)
    expected = errors.factors(vis.freq_hz).astype(np.complex64)
    np.testing.assert_array_equal(vis.calibration_error_gains, expected)


def test_calibration_errors_antenna_count_must_match(default_array, start_time):
    sim = make_simulator(default_array, start_time)
    with pytest.raises(ValueError, match="describes 3 antennas"):
        correlate(sim.blocks(), calibration_errors=CalibrationErrors.identity(3))


def test_calibration_errors_must_be_a_calibration_errors_instance(default_array, start_time):
    sim = make_simulator(default_array, start_time)
    with pytest.raises(ValueError, match="CalibrationErrors"):
        correlate(sim.blocks(), calibration_errors=np.ones(sim.n_antennas))


# ----------------------------------------------------------------------
# Expected observables
# ----------------------------------------------------------------------
@pytest.mark.parametrize("sigma_deg", [10.0, 30.0])
def test_phase_errors_produce_the_expected_coherence_loss(start_time, sigma_deg):
    """Peak(errors) / Peak(clean) ~ exp(-sigma_phi**2), sigma_phi in radians.

    Autocorrelations are blind to a phase-only error (c_i c_i* = 1), so
    they are dropped -- exactly the same reasoning
    test_instrument.test_random_phases_degrade_the_image_and_calibration_restores_it
    uses for the true-gain analogue of this test.
    """
    array = random_flat_array(n_antennas=40, seed=12)
    phase_center = zenith_phase_center(array, start_time, 0.1)
    source = PointSource.from_lm(phase_center, (SOURCE_L, SOURCE_M), flux_jy=20.0)
    sim = make_simulator(array, start_time, [source], noise_std=0.0)
    blocks = list(sim.blocks())

    clean = correlate(blocks, include_autos=False)
    clean_peak = dirty_image(clean)[0].max()

    errors = CalibrationErrors.from_params(sim.n_antennas, seed=77, phase_error_deg_rms=sigma_deg)
    miscalibrated = correlate(blocks, include_autos=False, calibration_errors=errors)
    peak = dirty_image(miscalibrated)[0].max()

    sigma_rad = np.deg2rad(sigma_deg)
    expected_ratio = np.exp(-(sigma_rad**2))
    measured_ratio = peak / clean_peak
    assert measured_ratio == pytest.approx(expected_ratio, rel=0.2)


def test_delay_errors_produce_a_linear_phase_ramp(default_array, start_time):
    """A residual delay error tau_ij gives a per-channel phase slope 2 pi tau_ij f."""
    source = PointSource.from_lm(
        zenith_phase_center(default_array, start_time, 0.1), (SOURCE_L, SOURCE_M), flux_jy=5.0
    )
    sim = make_simulator(default_array, start_time, [source], n_chan=64, noise_std=0.0)
    blocks = list(sim.blocks())
    plain = correlate(blocks)

    errors = CalibrationErrors.from_params(sim.n_antennas, seed=6, delay_error_ns_rms=2.0)
    miscalibrated = correlate(blocks, calibration_errors=errors)

    # Autocorrelations are noiseless here (no receiver noise, no sky
    # source), so the ratio isolates exactly the applied phase ramp.
    i, j = 1, 4
    row = (
        miscalibrated.data[:, plain.baseline_index(i, j), :]
        / plain.data[:, plain.baseline_index(i, j), :]
    )
    tau_ij = errors.delay_error_s[i] - errors.delay_error_s[j]
    f_ref = plain.freq_hz.mean()
    expected_phase = np.unwrap(2.0 * np.pi * tau_ij * (plain.freq_hz - f_ref))
    measured_phase = np.unwrap(np.angle(row[0]))
    # Compare slopes (a constant offset from `phase_error_deg_rms` is
    # zero here, so slope and level should both match, but the slope is
    # the load-bearing claim).
    expected_slope = np.polyfit(plain.freq_hz, expected_phase, 1)[0]
    measured_slope = np.polyfit(plain.freq_hz, measured_phase, 1)[0]
    assert measured_slope == pytest.approx(expected_slope, rel=1e-3)
    # The slope is referenced to the band center, so the phase there --
    # not at the band edge or at zero RF frequency -- is the intercept.
    measured_at_ref = np.interp(f_ref, plain.freq_hz, measured_phase)
    assert measured_at_ref == pytest.approx(0.0, abs=1e-3)


def test_delay_error_phase_defaults_to_zero_at_the_band_mean():
    """`reference_freq_hz` defaults to the mean of whatever grid is passed."""
    errors = CalibrationErrors.from_params(3, seed=7, delay_error_ns_rms=5.0)
    assert errors.reference_freq_hz is None
    factors = errors.factors(FREQ_HZ)
    f_ref = FREQ_HZ.mean()
    at_ref = np.interp(f_ref, FREQ_HZ, np.unwrap(np.angle(factors[1])))
    assert at_ref == pytest.approx(0.0, abs=1e-6)


def test_reference_freq_hz_can_be_pinned_explicitly():
    """An explicit reference makes the zero-phase point independent of the grid."""
    pinned = CalibrationErrors.from_params(
        3, seed=7, delay_error_ns_rms=5.0, reference_freq_hz=1.41e9
    )
    assert pinned.reference_freq_hz == pytest.approx(1.41e9)
    factors_a = pinned.factors(FREQ_HZ)
    factors_b = pinned.factors(np.linspace(1.40e9, 1.50e9, 40))
    phase_a = np.interp(1.41e9, FREQ_HZ, np.unwrap(np.angle(factors_a[1])))
    grid_b = np.linspace(1.40e9, 1.50e9, 40)
    phase_b = np.interp(1.41e9, grid_b, np.unwrap(np.angle(factors_b[1])))
    assert phase_a == pytest.approx(0.0, abs=1e-6)
    assert phase_b == pytest.approx(0.0, abs=1e-6)


def test_small_residual_delay_barely_perturbs_the_dirty_image(default_array, start_time):
    """The observable that motivated referencing the slope to band center.

    At absolute RF frequency a 0.3 ns residual delay contributes ~150 deg
    of near-constant phase at L band and visibly corrupts the image; once
    the slope is referenced to the band center that same residual is a
    gentle phase ramp across the band and the dirty-image peak barely
    moves.
    """
    source = PointSource.from_lm(
        zenith_phase_center(default_array, start_time, 0.1), (SOURCE_L, SOURCE_M), flux_jy=5.0
    )
    sim = make_simulator(default_array, start_time, [source], n_chan=64, noise_std=0.0)
    blocks = list(sim.blocks())
    plain = correlate(blocks, include_autos=False)
    plain_peak = dirty_image(plain)[0].max()

    errors = CalibrationErrors.from_params(sim.n_antennas, seed=42, delay_error_ns_rms=0.3)
    miscalibrated = correlate(blocks, include_autos=False, calibration_errors=errors)
    miscalibrated_peak = dirty_image(miscalibrated)[0].max()

    assert miscalibrated_peak == pytest.approx(plain_peak, rel=0.05)


def test_amplitude_errors_do_not_touch_the_voltages(default_array, start_time):
    """Amplitude errors change visibility amplitudes, never VoltageBlock.data."""
    source = PointSource.from_lm(
        zenith_phase_center(default_array, start_time, 0.1), (SOURCE_L, SOURCE_M), flux_jy=3.0
    )
    sim = make_simulator(default_array, start_time, [source])
    blocks = list(sim.blocks())
    reference_data = [b.data.copy() for b in blocks]

    plain = correlate(blocks)
    errors = CalibrationErrors.from_params(sim.n_antennas, seed=21, amplitude_error_db_rms=1.5)
    miscalibrated = correlate(blocks, calibration_errors=errors)

    for block, reference in zip(blocks, reference_data):
        np.testing.assert_array_equal(block.data, reference)

    amp = np.abs(errors.factors(plain.freq_hz))
    amplitude_factor = amp[plain.ant_1] * amp[plain.ant_2]
    np.testing.assert_allclose(
        np.abs(miscalibrated.data),
        np.abs(plain.data) * amplitude_factor,
        rtol=2e-4,
        atol=2e-4,
    )
    assert not np.allclose(np.abs(miscalibrated.data), np.abs(plain.data))


# ----------------------------------------------------------------------
# n_pol=2
# ----------------------------------------------------------------------
def test_single_model_broadcasts_to_both_polarizations(default_array, start_time):
    sim = make_simulator(default_array, start_time, n_pol=2)
    blocks = list(sim.blocks())
    errors = CalibrationErrors.from_params(sim.n_antennas, seed=5, phase_error_deg_rms=8.0)
    vis = correlate(blocks, calibration_errors=errors)

    assert vis.calibration_error_gains.shape == (sim.n_antennas, 2, vis.n_chan)
    np.testing.assert_array_equal(
        vis.calibration_error_gains[:, 0, :], vis.calibration_error_gains[:, 1, :]
    )


def test_sequence_of_two_models_gives_independent_per_pol_errors(default_array, start_time):
    sim = make_simulator(default_array, start_time, n_pol=2)
    blocks = list(sim.blocks())
    errors_x = CalibrationErrors.from_params(sim.n_antennas, seed=1, phase_error_deg_rms=8.0)
    errors_y = CalibrationErrors.from_params(sim.n_antennas, seed=2, phase_error_deg_rms=8.0)
    vis = correlate(blocks, calibration_errors=[errors_x, errors_y])

    assert vis.calibration_error_gains.shape == (sim.n_antennas, 2, vis.n_chan)
    assert not np.array_equal(
        vis.calibration_error_gains[:, 0, :], vis.calibration_error_gains[:, 1, :]
    )
    np.testing.assert_allclose(
        vis.calibration_error_gains[:, 0, :], errors_x.factors(vis.freq_hz), rtol=1e-6
    )
    np.testing.assert_allclose(
        vis.calibration_error_gains[:, 1, :], errors_y.factors(vis.freq_hz), rtol=1e-6
    )


def test_calibration_errors_sequence_must_match_n_pol(default_array, start_time):
    sim = make_simulator(default_array, start_time, n_pol=2)
    errors = CalibrationErrors.from_params(sim.n_antennas, seed=1, phase_error_deg_rms=5.0)
    with pytest.raises(ValueError, match="n_pol=2"):
        correlate(sim.blocks(), calibration_errors=[errors])
