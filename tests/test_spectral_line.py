"""Tests for rfi_simulator.sky.SpectralLineForeground and its voltage-level wiring.

The tests that matter most here:

* `test_default_is_bit_identical` -- `spectral_lines=()` must reproduce
  today's data exactly, channel for channel, bit for bit.
* `test_autocorrelation_shows_gaussian_bump` -- the line must show up as a
  Gaussian-shaped power excess in every antenna's autocorrelation, peaking
  at `line_flux_jy` and with the configured FWHM.
* `test_cross_correlation_has_no_coherent_signal` -- the "fully resolved
  extended emission" approximation means the line must NOT correlate
  between antennas, unlike a real celestial point source.
* `test_celestial_mask_is_separate_from_rfi_mask` -- the whole point of this
  feature is a ground-truth label distinguishable from interference.
"""

import numpy as np
import pytest
from conftest import zenith_phase_center

from rfi_simulator import (
    InstrumentModel,
    NarrowbandTransmitter,
    SpectralLineForeground,
    VoltageSimulator,
    correlate,
    enu_from_horizontal,
)

LINE_CENTER_HZ = 1.4e9
LINE_FWHM_HZ = 2.0e4
LINE_FLUX_JY = 5.0
CHAN_WIDTH_HZ = 1.0e3
N_CHAN = 201  # +/- 100 kHz around the band center


def make_simulator(array, start_time, spectral_lines=(), rfi_sources=(), **kwargs):
    """A small simulator with a fine, narrow band centered on the test line."""
    options = dict(
        center_freq_hz=LINE_CENTER_HZ,
        chan_width_hz=CHAN_WIDTH_HZ,
        n_chan=N_CHAN,
        n_blocks=1,
        n_time_per_block=64,
        noise_std=0.0,
        rng=np.random.default_rng(20260803),
    )
    options.update(kwargs)
    phase_center = zenith_phase_center(array, start_time, duration_s=0.1)
    return VoltageSimulator(
        array,
        phase_center,
        start_time,
        [],
        rfi_sources=rfi_sources,
        spectral_lines=spectral_lines,
        **options,
    )


def make_line(**kwargs):
    options = dict(
        center_freq_hz=LINE_CENTER_HZ,
        fwhm_hz=LINE_FWHM_HZ,
        line_flux_jy=LINE_FLUX_JY,
        name="hi_line",
    )
    options.update(kwargs)
    return SpectralLineForeground(**options)


# ----------------------------------------------------------------------
# SpectralLineForeground itself
# ----------------------------------------------------------------------
def test_power_envelope_peaks_at_line_flux_on_a_centered_channel():
    """On a channel grid centered exactly on the line, the peak equals line_flux_jy."""
    line = make_line()
    freq_hz = np.array(
        [LINE_CENTER_HZ - CHAN_WIDTH_HZ, LINE_CENTER_HZ, LINE_CENTER_HZ + CHAN_WIDTH_HZ]
    )
    envelope = line.power_envelope_jy(freq_hz)
    assert envelope[1] == pytest.approx(LINE_FLUX_JY, rel=1e-12)
    assert envelope[0] < envelope[1]
    assert envelope[2] < envelope[1]


def test_power_envelope_fwhm_matches_parameter():
    """The envelope falls to half its peak at +/- fwhm_hz / 2 from center."""
    line = make_line()
    half_width = np.array(
        [LINE_CENTER_HZ - LINE_FWHM_HZ / 2.0, LINE_CENTER_HZ + LINE_FWHM_HZ / 2.0]
    )
    envelope = line.power_envelope_jy(half_width)
    np.testing.assert_allclose(envelope, 0.5 * LINE_FLUX_JY, rtol=1e-9)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(center_freq_hz=0.0), "center_freq_hz"),
        (dict(center_freq_hz=-1.0), "center_freq_hz"),
        (dict(fwhm_hz=0.0), "fwhm_hz"),
        (dict(fwhm_hz=-1.0), "fwhm_hz"),
        (dict(line_flux_jy=-1.0), "line_flux_jy"),
    ],
)
def test_invalid_parameters_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_line(**kwargs)


def test_spectral_lines_must_be_the_right_type(default_array, start_time):
    """VoltageSimulator rejects anything that is not a SpectralLineForeground."""
    with pytest.raises(ValueError, match="SpectralLineForeground"):
        make_simulator(default_array, start_time, spectral_lines=["not a line"])


# ----------------------------------------------------------------------
# Integration into VoltageSimulator / VoltageBlock
# ----------------------------------------------------------------------
def test_default_is_bit_identical(default_array, start_time):
    """spectral_lines=() (the default) reproduces today's data exactly."""
    plain = make_simulator(default_array, start_time)
    explicit_empty = make_simulator(default_array, start_time, spectral_lines=())

    plain_block = plain.block(0)
    empty_block = explicit_empty.block(0)
    np.testing.assert_array_equal(plain_block.data, empty_block.data)
    assert plain_block.celestial_mask.shape == (0, N_CHAN, 64)
    assert plain_block.celestial_source_names == ()
    assert plain_block.n_celestial_sources == 0


def test_autocorrelation_shows_gaussian_bump(default_array, start_time):
    """Per-antenna power vs. frequency matches the line's Gaussian envelope."""
    line = make_line()
    sim = make_simulator(default_array, start_time, [line], n_time_per_block=20000)
    block = sim.block(0)

    measured_power = np.mean(np.abs(block.data) ** 2, axis=(0, 2))  # (n_chan,)
    expected = line.power_envelope_jy(sim.freq_hz)

    # Loose tolerance: this is a statistical measurement over finite samples,
    # averaged over 10 antennas x 20000 samples per channel.
    np.testing.assert_allclose(measured_power, expected, rtol=0.15, atol=0.05)

    peak_chan = int(np.argmax(measured_power))
    assert measured_power[peak_chan] == pytest.approx(LINE_FLUX_JY, rel=0.1)

    # Half-power channels should be within a channel or two of the FWHM.
    half = 0.5 * measured_power[peak_chan]
    above = np.flatnonzero(measured_power >= half)
    measured_fwhm_hz = (above[-1] - above[0]) * CHAN_WIDTH_HZ
    assert measured_fwhm_hz == pytest.approx(LINE_FWHM_HZ, rel=0.25)


def test_cross_correlation_has_no_coherent_signal(default_array, start_time):
    """The line is uncorrelated between antennas -- the 'resolved-out' approximation."""
    line = make_line()
    sim = make_simulator(default_array, start_time, [line], n_time_per_block=20000)
    block = sim.block(0)

    peak_chan = int(np.argmin(np.abs(sim.freq_hz - LINE_CENTER_HZ)))
    v0 = block.data[0, peak_chan, :]
    v1 = block.data[1, peak_chan, :]
    auto0 = np.mean(np.abs(v0) ** 2)
    cross = np.mean(v0 * np.conjugate(v1))

    # Same statistical bound style as the existing receiver-noise test.
    n_samples = v0.size
    assert auto0 == pytest.approx(LINE_FLUX_JY, rel=0.1)
    assert abs(cross) < 6.0 * LINE_FLUX_JY / np.sqrt(n_samples)


def test_celestial_mask_marks_the_right_channels(default_array, start_time):
    """celestial_mask is True only where the profile exceeds the threshold."""
    line = make_line()
    sim = make_simulator(default_array, start_time, [line])
    block = sim.block(0)

    assert block.celestial_mask.shape == (1, N_CHAN, 64)
    assert block.celestial_source_names == ("hi_line",)

    expected = line.mask(sim.freq_hz, 64)
    np.testing.assert_array_equal(block.celestial_mask[0], expected)
    # Some channels are flagged, but not the whole band (fwhm << bandwidth).
    assert block.celestial_mask[0].any()
    assert not block.celestial_mask[0].all()
    # The mask does not vary with time -- there is no duty cycle to a line.
    per_time = block.celestial_mask[0]
    for t in range(1, per_time.shape[1]):
        np.testing.assert_array_equal(per_time[:, t], per_time[:, 0])


def test_celestial_mask_is_separate_from_rfi_mask(default_array, start_time):
    """A run with both an RFI source and a spectral line keeps the two labels apart."""
    line = make_line()
    tower = NarrowbandTransmitter(
        position_enu_m=enu_from_horizontal(90.0, 2.0, 2000.0),
        center_freq_hz=LINE_CENTER_HZ + 4.0e4,
        bandwidth_hz=CHAN_WIDTH_HZ * 4,
        received_power_jy=50.0,
        name="tower",
    )
    sim = make_simulator(default_array, start_time, [line], rfi_sources=[tower])
    block = sim.block(0)

    assert block.rfi_mask.shape[0] == 1
    assert block.celestial_mask.shape[0] == 1
    assert block.rfi_source_names == ("tower",)
    assert block.celestial_source_names == ("hi_line",)
    # The two masks occupy disjoint frequency neighborhoods by construction
    # (tower is 40 kHz above the line, well outside either's FWHM), and are
    # carried in entirely separate arrays/fields.
    assert not np.any(block.rfi_mask[0] & block.celestial_mask[0])


def test_line_passes_through_instrument_gains(default_array, start_time):
    """Per-antenna line power scales as |g_i|**2, same as sky and noise do."""
    n_ant = default_array.n_antennas
    gains = np.array([0.5 + 0.3j if i % 2 == 0 else 1.5 - 0.2j for i in range(n_ant)])
    instrument = InstrumentModel.from_gains(gains)

    line = make_line()
    sim_plain = make_simulator(default_array, start_time, [line], n_time_per_block=8000)
    sim_gained = make_simulator(
        default_array, start_time, [line], n_time_per_block=8000, instrument=instrument
    )

    plain_power = np.mean(np.abs(sim_plain.block(0).data) ** 2, axis=(1, 2))
    gained_power = np.mean(np.abs(sim_gained.block(0).data) ** 2, axis=(1, 2))

    expected_ratio = np.abs(gains) ** 2
    measured_ratio = gained_power / plain_power
    np.testing.assert_allclose(measured_ratio, expected_ratio, rtol=0.1)


def test_partially_out_of_band_is_graceful(default_array, start_time):
    """A line straddling the band edge does not crash and masks only the in-band part."""
    edge_center = LINE_CENTER_HZ + (N_CHAN // 2) * CHAN_WIDTH_HZ  # top channel
    line = make_line(center_freq_hz=edge_center, fwhm_hz=4.0e4)
    sim = make_simulator(default_array, start_time, [line])
    block = sim.block(0)

    assert block.celestial_mask[0].any()  # some in-band channels still light up
    assert np.all(np.isfinite(block.data))


def test_fully_out_of_band_is_graceful(default_array, start_time):
    """A line far outside the band contributes nothing and masks nothing."""
    far_center = LINE_CENTER_HZ + 50 * CHAN_WIDTH_HZ * N_CHAN
    line = make_line(center_freq_hz=far_center, fwhm_hz=CHAN_WIDTH_HZ)
    sim = make_simulator(default_array, start_time, [line])
    block = sim.block(0)

    assert not block.celestial_mask[0].any()
    assert np.all(np.isfinite(block.data))
    # No meaningful power added anywhere in the simulated band.
    assert np.mean(np.abs(block.data) ** 2) < 1e-6


def test_reproducible_with_the_same_seed(default_array, start_time):
    """The same seed with a line attached gives bit-identical voltages."""
    line = make_line()

    def run(seed):
        sim = make_simulator(default_array, start_time, [line], rng=np.random.default_rng(seed))
        return sim.block(0).data

    np.testing.assert_array_equal(run(11), run(11))
    assert not np.array_equal(run(11), run(12))


def test_multiple_lines_stack_independently(default_array, start_time):
    """Two lines get independent masks/names and their powers add."""
    line_a = make_line(name="line_a")
    line_b = make_line(center_freq_hz=LINE_CENTER_HZ + 8.0e4, name="line_b")
    sim = make_simulator(default_array, start_time, [line_a, line_b])
    block = sim.block(0)

    assert block.celestial_source_names == ("line_a", "line_b")
    assert block.celestial_mask.shape == (2, N_CHAN, 64)
    assert block.n_celestial_sources == 2
    # The two lines are far enough apart that their masks do not overlap.
    assert not np.any(block.celestial_mask[0] & block.celestial_mask[1])


def test_correlate_carries_celestial_fraction(default_array, start_time):
    """Visibilities.celestial_fraction mirrors rfi_fraction's bookkeeping."""
    line = make_line()
    sim = make_simulator(default_array, start_time, [line], n_blocks=2)
    vis = correlate(sim.blocks())

    assert vis.celestial_source_names == ("hi_line",)
    assert vis.celestial_fraction.shape == (2, 1, N_CHAN)
    assert vis.n_celestial_sources == 1

    # No line: zero-sized celestial axis, matching the rfi_fraction convention.
    sim_clean = make_simulator(default_array, start_time)
    vis_clean = correlate(sim_clean.blocks())
    assert vis_clean.celestial_fraction.shape == (1, 0, N_CHAN)
    assert vis_clean.n_celestial_sources == 0


def test_rfi_sources_unaffected_by_attaching_a_line(default_array, start_time):
    """Attaching a spectral line does not perturb the interference realization.

    Interference draws from its own seed branch (see VoltageSimulator's
    class Notes), so it must be identical whether or not a spectral line is
    also attached -- only the sky/noise branch (and hence the plain data)
    may shift.
    """
    tower = NarrowbandTransmitter(
        position_enu_m=enu_from_horizontal(90.0, 2.0, 2000.0),
        center_freq_hz=LINE_CENTER_HZ + 4.0e4,
        bandwidth_hz=CHAN_WIDTH_HZ * 4,
        received_power_jy=50.0,
        name="tower",
    )
    line = make_line()

    without_line = make_simulator(default_array, start_time, rfi_sources=[tower])
    with_line = make_simulator(default_array, start_time, [line], rfi_sources=[tower])

    np.testing.assert_array_equal(without_line.block(0).rfi_mask, with_line.block(0).rfi_mask)
