"""Tests for rfi_simulator.instrument and the in-simulation quantizer.

The load-bearing tests here are:

* `test_instrument_none_matches_identity_bit_for_bit` -- switching the
  feature off, and switching it on with unit gains, must both reproduce the
  data the simulator produced before per-antenna gains existed.
* `test_visibilities_scale_as_gi_gj_conjugate` -- gains applied to voltages
  must reach the correlator as the ``g_i g_j*`` structure calibration
  solves for; anything else (e.g. applying a gain per baseline, or
  forgetting the conjugate) fails here.
* the statistical tests, which tie the *parameters* (dB of power scatter,
  dB of bandpass ripple) to what a measurement of the simulated data
  recovers -- a parameter nobody can measure back out is not a parameter.
"""

import numpy as np
import pytest
from conftest import SOURCE_L, SOURCE_M, random_flat_array, zenith_phase_center

from rfi_simulator import (
    ArrayConfig,
    InstrumentModel,
    PointSource,
    VoltageSimulator,
    correlate,
    dirty_image,
)
from rfi_simulator.io.packed_voltage import quantize_roundtrip
from rfi_simulator.voltages import DEFAULT_QUANT_TARGET_COUNTS

FREQ_HZ = np.linspace(1.40e9, 1.42e9, 96)


def make_simulator(array, start_time, sources=(), **kwargs):
    """Small-but-real simulator, matching test_voltages' conventions."""
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
# InstrumentModel: construction, repeatability, immutability
# ----------------------------------------------------------------------
def test_identity_model_is_exactly_unit_gain():
    model = InstrumentModel.identity(5)
    gains = model.gains(FREQ_HZ)
    assert gains.shape == (5, FREQ_HZ.size)
    np.testing.assert_array_equal(gains, np.ones_like(gains))
    np.testing.assert_array_equal(model.amplitude_db, np.zeros(5))
    np.testing.assert_array_equal(model.phase_rad, np.zeros(5))


def test_defaults_are_off():
    """`from_params` with no options is the identity, and needs no seed."""
    model = InstrumentModel.from_params(4)
    np.testing.assert_array_equal(model.gains(FREQ_HZ), np.ones((4, FREQ_HZ.size)))
    assert model.n_bandpass_modes == 0


def test_same_seed_gives_identical_gains_and_different_seeds_do_not():
    kwargs = dict(gain_scatter_db=0.4, phase_offsets="uniform", bandpass_ripple_db=0.05)
    first = InstrumentModel.from_params(12, seed=11, **kwargs).gains(FREQ_HZ)
    again = InstrumentModel.from_params(12, seed=11, **kwargs).gains(FREQ_HZ)
    other = InstrumentModel.from_params(12, seed=12, **kwargs).gains(FREQ_HZ)
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


def test_bandpass_is_repeatable_across_evaluations():
    """The bandpass is a property of the antenna, not a per-call draw."""
    model = InstrumentModel.from_params(6, seed=5, bandpass_ripple_db=0.05, freq_hz=FREQ_HZ)
    np.testing.assert_array_equal(model.gains(FREQ_HZ), model.gains(FREQ_HZ))


def test_effects_are_independently_seeded():
    """Adding a bandpass leaves the amplitude and phase draws untouched."""
    plain = InstrumentModel.from_params(8, seed=99, gain_scatter_db=0.5, phase_offsets="uniform")
    rippled = InstrumentModel.from_params(
        8, seed=99, gain_scatter_db=0.5, phase_offsets="uniform", bandpass_ripple_db=0.05
    )
    np.testing.assert_array_equal(plain.scalar_gains, rippled.scalar_gains)
    assert rippled.n_bandpass_modes > 0


def test_model_is_frozen_and_arrays_are_read_only():
    model = InstrumentModel.from_params(4, seed=1, gain_scatter_db=0.3)
    with pytest.raises(Exception):
        model.scalar_gains = np.ones(4)
    with pytest.raises(ValueError):
        model.scalar_gains[0] = 2.0
    # The returned gains are a copy: editing them cannot corrupt the truth.
    gains = model.gains(FREQ_HZ)
    gains[0, 0] = 123.0
    assert model.gains(FREQ_HZ)[0, 0] != 123.0


def test_explicit_scalar_gains_are_used_verbatim():
    values = np.array([1.0, 0.5, 2.0 + 1.0j])
    model = InstrumentModel.from_gains(values)
    gains = model.gains(FREQ_HZ)
    assert gains.shape == (3, FREQ_HZ.size)
    for i_ant, value in enumerate(values):
        np.testing.assert_allclose(gains[i_ant], value)


def test_explicit_gain_table_needs_its_own_grid():
    table = np.exp(1j * np.linspace(0.0, 1.0, 3 * FREQ_HZ.size)).reshape(3, FREQ_HZ.size)
    model = InstrumentModel.from_gains(table, FREQ_HZ)
    np.testing.assert_allclose(model.gains(FREQ_HZ), table)
    with pytest.raises(ValueError, match="does not interpolate"):
        model.gains(FREQ_HZ[:10])
    with pytest.raises(ValueError, match="does not interpolate"):
        model.gains(FREQ_HZ + 1e6)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_antennas": 0}, "n_antennas"),
        ({"gain_scatter_db": -1.0}, "gain_scatter_db"),
        ({"bandpass_ripple_db": -0.1}, "bandpass_ripple_db"),
        ({"bandpass_n_modes": 0}, "bandpass_n_modes"),
        ({"phase_offsets": "gaussian"}, "phase_offsets"),
        ({"gain_scatter_db": 0.4, "seed": None}, "needs an rng or a seed"),
        ({"seed": 3, "rng": np.random.default_rng(0)}, "either rng or seed"),
    ],
)
def test_from_params_validates(kwargs, match):
    options = dict(n_antennas=6, seed=4)
    options.update(kwargs)
    with pytest.raises(ValueError, match=match):
        InstrumentModel.from_params(**options)


@pytest.mark.parametrize(
    ("param", "match"),
    [
        ("gain_scatter_db", "gain_scatter_db"),
        ("bandpass_ripple_db", "bandpass_ripple_db"),
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_from_params_rejects_non_finite_scatter_parameters(param, match, value):
    """A NaN or Inf scatter parameter must not slip past the ``< 0`` guard.

    NaN in particular fails every comparison, so `gain_scatter_db > 0.0`
    would be False and the feature would silently look switched off while
    actually holding a non-finite value.
    """
    options = dict(n_antennas=6, seed=4)
    options[param] = value
    with pytest.raises(ValueError, match=match):
        InstrumentModel.from_params(**options)
        InstrumentModel.from_params(options.pop("n_antennas"), **options)


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((np.ones((2, 3)),), "requires the freq_hz grid"),
        ((np.ones(3), FREQ_HZ), "meaningless"),
        ((np.ones((2, 2, 2)),), "must have shape"),
        ((np.array([1.0, np.nan]),), "non-finite"),
    ],
)
def test_from_gains_validates(args, match):
    with pytest.raises(ValueError, match=match):
        InstrumentModel.from_gains(*args)


# ----------------------------------------------------------------------
# Statistics: the parameters must be measurable back out
# ----------------------------------------------------------------------
def test_amplitude_scatter_matches_the_configured_db():
    """The rms of the per-antenna power in dB equals `gain_scatter_db`."""
    scatter_db = 0.6
    model = InstrumentModel.from_params(4000, seed=2026, gain_scatter_db=scatter_db)
    measured = np.std(model.amplitude_db)
    assert measured == pytest.approx(scatter_db, rel=0.05)
    # And the mean level is unbiased in dB.
    assert abs(np.mean(model.amplitude_db)) < 0.1 * scatter_db


def test_bandpass_ripple_rms_matches_the_configured_db():
    ripple_db = 0.05
    model = InstrumentModel.from_params(
        400, seed=7, bandpass_ripple_db=ripple_db, bandpass_n_modes=3, freq_hz=FREQ_HZ
    )
    ripple = model.bandpass_db(FREQ_HZ)
    assert ripple.shape == (400, FREQ_HZ.size)
    assert np.std(ripple) == pytest.approx(ripple_db, rel=0.1)
    # Each antenna's ripple averages to zero across the band, so the
    # bandpass adds no net power on top of the amplitude scatter.
    assert np.abs(ripple.mean(axis=1)).max() < 0.2 * ripple_db
    # ... and it is smooth: a few modes, not channel-to-channel noise.
    steps = np.abs(np.diff(ripple, axis=1))
    assert steps.max() < np.abs(ripple).max()


def test_bandpass_is_smoother_with_fewer_modes():
    common = dict(seed=3, bandpass_ripple_db=0.1, freq_hz=FREQ_HZ)
    few = InstrumentModel.from_params(200, bandpass_n_modes=2, **common).bandpass_db(FREQ_HZ)
    many = InstrumentModel.from_params(200, bandpass_n_modes=12, **common).bandpass_db(FREQ_HZ)
    assert np.std(np.diff(few, axis=1)) < np.std(np.diff(many, axis=1))


def test_uniform_phase_offsets_cover_the_circle():
    model = InstrumentModel.from_params(4000, seed=8, phase_offsets="uniform")
    phase = model.phase_rad
    # Uniform on the circle: the mean phasor is ~0 and every quadrant fills.
    assert abs(np.mean(np.exp(1j * phase))) < 0.05
    counts, _ = np.histogram(phase, bins=4, range=(-np.pi, np.pi))
    assert counts.sum() == phase.size
    assert counts.min() > 0.2 * phase.size  # each quadrant holds roughly a quarter
    # Amplitudes stay exactly unity when only phases are switched on.
    np.testing.assert_allclose(np.abs(model.scalar_gains), 1.0)


# ----------------------------------------------------------------------
# Integration with the simulator
# ----------------------------------------------------------------------
def test_instrument_none_matches_identity_bit_for_bit(default_array, start_time):
    """The default is a no-op, and unit gains are a no-op too."""
    source = PointSource.from_lm(
        zenith_phase_center(default_array, start_time, 0.1), (SOURCE_L, SOURCE_M), flux_jy=2.0
    )
    reference = make_simulator(default_array, start_time, [source]).block(0)
    identity = make_simulator(
        default_array,
        start_time,
        [source],
        instrument=InstrumentModel.identity(default_array.n_antennas),
    ).block(0)
    np.testing.assert_array_equal(identity.data, reference.data)
    assert reference.gains is None
    assert identity.gains is not None
    np.testing.assert_array_equal(identity.gains, np.ones_like(identity.gains))
    assert reference.clip_fraction is None and reference.quant_scale is None


def test_gains_do_not_disturb_the_sky_and_noise_realization(default_array, start_time):
    """Gains are a multiplication, not a reseeding: dividing them out returns
    exactly the ungained data."""
    model = InstrumentModel.from_params(
        default_array.n_antennas, seed=17, gain_scatter_db=0.7, phase_offsets="uniform"
    )
    plain = make_simulator(default_array, start_time).block(0)
    gained = make_simulator(default_array, start_time, instrument=model).block(0)
    recovered = gained.data / gained.gains[:, :, np.newaxis]
    np.testing.assert_allclose(recovered, plain.data, rtol=2e-5, atol=2e-6)


def test_per_antenna_power_scatter_appears_in_the_data(start_time):
    """Autocorrelated power follows |g_i|**2 -- the flat-array sim does not."""
    array = random_flat_array(n_antennas=40, seed=4)
    model = InstrumentModel.from_params(array.n_antennas, seed=21, gain_scatter_db=0.8)
    sim = make_simulator(array, start_time, instrument=model, n_chan=8, n_time_per_block=2000)
    block = sim.block(0)
    power = np.mean(np.abs(block.data) ** 2, axis=(1, 2))  # (n_ant,)
    expected = np.abs(model.scalar_gains) ** 2 * sim.noise_std**2
    np.testing.assert_allclose(power, expected, rtol=0.05)

    # The measured power scatter in dB matches the configured scatter, and
    # a run without an instrument model is essentially perfectly flat.
    measured_db = 10.0 * np.log10(power / power.mean())
    assert np.std(measured_db) == pytest.approx(0.8, rel=0.25)
    flat = make_simulator(array, start_time, n_chan=8, n_time_per_block=2000).block(0)
    flat_power = np.mean(np.abs(flat.data) ** 2, axis=(1, 2))
    assert np.std(10.0 * np.log10(flat_power / flat_power.mean())) < 0.1


def test_bandpass_appears_in_the_data(start_time):
    """Per-channel power follows the model's dB ripple, repeatably per block."""
    array = random_flat_array(n_antennas=30, seed=6)
    model = InstrumentModel.from_params(
        array.n_antennas, seed=31, bandpass_ripple_db=0.4, bandpass_n_modes=2
    )
    sim = make_simulator(array, start_time, instrument=model, n_chan=24, n_time_per_block=4000)
    truth_db = model.bandpass_db(sim.freq_hz)
    # Average two blocks: the ripple is repeatable, the noise is not.
    power = np.mean(
        [np.mean(np.abs(sim.block(i).data) ** 2, axis=2) for i in range(sim.n_blocks)], axis=0
    )
    measured_db = 10.0 * np.log10(power / power.mean(axis=1, keepdims=True))
    assert np.std(measured_db - truth_db) < 0.3 * np.std(truth_db)


def test_visibilities_scale_as_gi_gj_conjugate(default_array, start_time):
    """correlate() sees g_i g_j*: the structure calibration has to solve for."""
    source = PointSource.from_lm(
        zenith_phase_center(default_array, start_time, 0.1), (SOURCE_L, SOURCE_M), flux_jy=5.0
    )
    model = InstrumentModel.from_params(
        default_array.n_antennas, seed=44, gain_scatter_db=1.0, phase_offsets="uniform"
    )
    plain = correlate(make_simulator(default_array, start_time, [source]).blocks())
    gained = correlate(
        make_simulator(default_array, start_time, [source], instrument=model).blocks()
    )
    g = model.scalar_gains
    factor = g[plain.ant_1] * np.conjugate(g[plain.ant_2])  # (n_baselines,)
    np.testing.assert_allclose(
        gained.data, plain.data * factor[np.newaxis, :, np.newaxis], rtol=2e-4, atol=2e-4
    )
    # Amplitude-only statement, stated separately because it is the one an
    # amplitude calibration checks: |V_ij| scales by |g_i||g_j|.
    amplitude_factor = np.abs(g[plain.ant_1]) * np.abs(g[plain.ant_2])
    np.testing.assert_allclose(
        np.abs(gained.data),
        np.abs(plain.data) * amplitude_factor[np.newaxis, :, np.newaxis],
        rtol=2e-4,
        atol=2e-4,
    )


def test_random_phases_degrade_the_image_and_calibration_restores_it(default_array, start_time):
    """An uncalibrated array loses its point source; dividing out the truth
    brings the peak back."""
    array = random_flat_array(n_antennas=24, seed=9)
    phase_center = zenith_phase_center(array, start_time, 0.1)
    source = PointSource.from_lm(phase_center, (SOURCE_L, SOURCE_M), flux_jy=20.0)
    model = InstrumentModel.from_params(array.n_antennas, seed=55, phase_offsets="uniform")

    def image_peak(instrument):
        # Autocorrelations are blind to a phase-only gain (g_i g_i* = 1), so
        # they would dilute the effect being measured here.
        vis = correlate(
            make_simulator(
                array, start_time, [source], noise_std=0.0, instrument=instrument
            ).blocks(),
            include_autos=False,
        )
        return vis, dirty_image(vis)[0].max()

    clean_vis, clean_peak = image_peak(None)
    gained_vis, gained_peak = image_peak(model)
    assert gained_peak < 0.4 * clean_peak

    # Dividing the ground-truth gains back out restores the source exactly.
    g = model.scalar_gains
    factor = g[gained_vis.ant_1] * np.conjugate(g[gained_vis.ant_2])
    gained_vis.data = (gained_vis.data / factor[np.newaxis, :, np.newaxis]).astype(np.complex64)
    assert dirty_image(gained_vis)[0].max() == pytest.approx(clean_peak, rel=0.02)


def test_instrument_antenna_count_must_match_the_array(default_array, start_time):
    with pytest.raises(ValueError, match="describes 3 antennas"):
        make_simulator(default_array, start_time, instrument=InstrumentModel.identity(3))


def test_instrument_must_be_a_model(default_array, start_time):
    with pytest.raises(ValueError, match="InstrumentModel"):
        make_simulator(default_array, start_time, instrument=np.ones(10))


# ----------------------------------------------------------------------
# In-simulation quantization
# ----------------------------------------------------------------------
def test_quantization_defaults_to_off(default_array, start_time):
    block = make_simulator(default_array, start_time).block(0)
    assert block.clip_fraction is None
    assert block.quant_scale is None


@pytest.mark.parametrize("mode", ["int8", "float16", "int4 "])
def test_unknown_quantization_mode_is_rejected(default_array, start_time, mode):
    with pytest.raises(ValueError, match="quantization must be one of"):
        make_simulator(default_array, start_time, quantization=mode)


def test_quantized_block_is_on_the_quantizer_grid(default_array, start_time):
    """Every sample is an integer number of counts, within +-8."""
    block = make_simulator(default_array, start_time, quantization="int4").block(0)
    counts_real = block.data.real / block.quant_scale
    counts_imag = block.data.imag / block.quant_scale
    for counts in (counts_real, counts_imag):
        np.testing.assert_allclose(counts, np.round(counts), atol=1e-3)
        assert counts.min() >= -8.0 - 1e-6
        assert counts.max() <= 7.0 + 1e-6
    assert block.clip_fraction.shape == (default_array.n_antennas,)


def test_quantizer_hits_the_requested_loading(default_array, start_time):
    """The realized counts rms lands on `quant_target_counts`."""
    for target in (DEFAULT_QUANT_TARGET_COUNTS, 2.5):
        block = make_simulator(
            default_array,
            start_time,
            quantization="int4",
            quant_target_counts=target,
            n_chan=8,
            n_time_per_block=4000,
        ).block(0)
        counts = block.data / block.quant_scale
        component_rms = np.sqrt(0.5 * np.mean(np.abs(counts) ** 2))
        assert component_rms == pytest.approx(target, rel=0.02)


def test_quantization_noise_variance_is_scale_squared_over_twelve(default_array, start_time):
    """The residual against the unquantized data is uniform quantizer noise."""
    scale = 0.75
    options = dict(n_chan=8, n_time_per_block=4000, noise_std=1.0)
    plain = make_simulator(default_array, start_time, **options).block(0)
    quantized = make_simulator(
        default_array, start_time, quantization="int4", quant_scale=scale, **options
    ).block(0)
    assert quantized.quant_scale == scale
    residual = quantized.data - plain.data
    for component in (residual.real, residual.imag):
        assert np.var(component) == pytest.approx(scale**2 / 12.0, rel=0.05)
    # With ~1.3 counts rms against +-7 rails, saturation is negligible.
    assert quantized.clip_fraction.max() < 1e-3


def test_gain_scatter_makes_one_antenna_rail(start_time):
    """A strongly over-gained antenna clips far more often than the rest --
    the per-antenna behaviour an unclipped, uniform-gain sim cannot show."""
    array = ArrayConfig(
        antenna_positions_enu_m=np.array(
            [[0.0, 0.0, 0.0], [30.0, 5.0, 0.0], [-20.0, 15.0, 0.0], [10.0, -25.0, 0.0]]
        ),
        latitude_deg=37.234,
        longitude_deg=-118.282,
        height_m=1222.0,
    )
    gains = np.array([1.0, 1.0, 1.0, 8.0], dtype=np.complex128)
    # A fixed scale, as a backend with a fixed digital gain setting has:
    # loading the unit-gain antennas at the default target leaves the
    # over-gained one 8x too hot for the +-7-count rails.
    quiet_scale = 1.0 / np.sqrt(2.0) / DEFAULT_QUANT_TARGET_COUNTS
    sim = make_simulator(
        array,
        start_time,
        instrument=InstrumentModel.from_gains(gains),
        quantization="int4",
        quant_scale=quiet_scale,
        n_chan=8,
        n_time_per_block=2000,
    )
    clip = sim.block(0).clip_fraction
    assert clip[3] > 0.3
    assert clip[:3].max() < 1e-4

    # With a per-block scale chosen from the whole array the strong antenna
    # no longer dominates the rails, but it is still the one that clips.
    auto = make_simulator(
        array,
        start_time,
        instrument=InstrumentModel.from_gains(gains),
        quantization="int4",
        n_chan=8,
        n_time_per_block=2000,
    ).block(0)
    assert auto.clip_fraction[3] > 20.0 * max(auto.clip_fraction[:3].max(), 1e-6)


def test_quantized_blocks_stay_deterministic(default_array, start_time):
    """Quantization does not consume randomness or depend on call order."""
    sim = make_simulator(default_array, start_time, quantization="int4")
    np.testing.assert_array_equal(sim.block(1).data, sim.block(1).data)
    other = make_simulator(default_array, start_time, quantization="int4")
    np.testing.assert_array_equal(sim.block(0).data, other.block(0).data)


def test_quantize_roundtrip_reports_saturation():
    """The io helper flags exactly the samples that railed."""
    voltages = np.array([0.0, 3.0, -4.0, 100.0, -100.0 + 100.0j], dtype=np.complex64)
    dequantized, clipped = quantize_roundtrip(voltages, 1.0)
    np.testing.assert_array_equal(clipped, [False, False, False, True, True])
    np.testing.assert_allclose(dequantized[:3], voltages[:3])
    assert dequantized[3] == pytest.approx(7.0)
    assert dequantized[4] == pytest.approx(-8.0 + 7.0j)


def test_quantize_roundtrip_validates():
    with pytest.raises(ValueError, match="positive finite"):
        quantize_roundtrip(np.ones(4, dtype=np.complex64), 0.0)
    with pytest.raises(ValueError, match="NaN"):
        quantize_roundtrip(np.array([np.nan + 0j]), 1.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"quant_target_counts": 0.0}, "quant_target_counts"),
        ({"quant_scale": -1.0}, "quant_scale"),
    ],
)
def test_quantization_parameters_are_validated(default_array, start_time, kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_simulator(default_array, start_time, quantization="int4", **kwargs)
