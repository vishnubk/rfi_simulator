"""Tests for rfi_simulator.beam -- primary-beam attenuation of celestial flux.

Covers the beam models in isolation (`GaussianBeam`, `AiryBeam`,
`bessel_j1`), the default-off invariant (attaching no `primary_beam` must
not perturb the simulator's output at all), and the RFIBench-relevant
acceptance test: an off-centre source's *recovered image flux* must equal
its catalog flux times the beam's power response at that offset.
"""

import hashlib

import numpy as np
import pytest
from conftest import zenith_phase_center
from test_channelizer import REFERENCE_DIGESTS
from test_channelizer import reference_simulator as channelizer_reference_simulator

from rfi_simulator import (
    AiryBeam,
    GaussianBeam,
    PointSource,
    PrimaryBeam,
    VoltageSimulator,
    correlate,
    dirty_image,
)
from rfi_simulator.beam import bessel_j1

# Same fine grid test_imaging.py uses.
PIXEL_RAD = 2e-4
GRID = np.arange(-75, 76) * PIXEL_RAD

# First two positive zeros of J1, to 5 significant figures.
J1_ZERO_1 = 3.8317
J1_ZERO_2 = 7.0156


# ----------------------------------------------------------------------
# bessel_j1
# ----------------------------------------------------------------------
def test_bessel_j1_at_known_zeros():
    assert abs(bessel_j1(J1_ZERO_1)) < 1e-5
    assert abs(bessel_j1(J1_ZERO_2)) < 1e-5


def test_bessel_j1_at_known_values():
    # Reference values (Abramowitz & Stegun table / standard references).
    assert bessel_j1(0.0) == pytest.approx(0.0, abs=1e-12)
    assert bessel_j1(1.0) == pytest.approx(0.4400505857, abs=1e-8)
    assert bessel_j1(2.0) == pytest.approx(0.5767248078, abs=1e-8)
    assert bessel_j1(5.0) == pytest.approx(-0.3275791376, abs=1e-8)
    assert bessel_j1(10.0) == pytest.approx(0.0434727462, abs=1e-8)


def test_bessel_j1_is_odd():
    x = np.array([0.3, 1.7, 4.2, 9.9])
    assert np.allclose(bessel_j1(-x), -bessel_j1(x))


# ----------------------------------------------------------------------
# GaussianBeam
# ----------------------------------------------------------------------
def test_gaussian_beam_peaks_at_one_on_axis():
    beam = GaussianBeam(dish_diameter_m=4.5)
    assert beam.power_response(0.0, 1.4e9) == pytest.approx(1.0)


def test_gaussian_beam_is_half_power_at_half_fwhm():
    beam = GaussianBeam(dish_diameter_m=4.5)
    freq_hz = 1.4e9
    fwhm = beam.fwhm_rad(freq_hz)
    assert beam.power_response(0.5 * fwhm, freq_hz) == pytest.approx(0.5, rel=1e-10)


def test_gaussian_beam_fwhm_scales_as_inverse_frequency():
    beam = GaussianBeam(dish_diameter_m=4.5)
    fwhm_1 = beam.fwhm_rad(1.0e9)
    fwhm_2 = beam.fwhm_rad(2.0e9)
    assert fwhm_2 == pytest.approx(0.5 * fwhm_1, rel=1e-12)


def test_gaussian_beam_attenuates_more_at_higher_frequency():
    beam = GaussianBeam(dish_diameter_m=4.5)
    theta = 0.01
    low = beam.power_response(theta, 1.0e9)
    high = beam.power_response(theta, 2.0e9)
    assert high < low


def test_gaussian_beam_broadcasts_over_channels():
    beam = GaussianBeam(dish_diameter_m=4.5)
    theta = np.array([0.0, 0.005, 0.01])[:, np.newaxis]
    freq_hz = np.linspace(1.3e9, 1.5e9, 8)[np.newaxis, :]
    response = beam.power_response(theta, freq_hz)
    assert response.shape == (3, 8)
    assert np.all(response[0] == pytest.approx(1.0))


# ----------------------------------------------------------------------
# AiryBeam
# ----------------------------------------------------------------------
def test_airy_beam_peaks_at_one_on_axis():
    beam = AiryBeam(dish_diameter_m=4.5)
    assert beam.power_response(0.0, 1.4e9) == pytest.approx(1.0, abs=1e-12)


def test_airy_beam_first_null_location():
    beam = AiryBeam(dish_diameter_m=4.5)
    freq_hz = 1.4e9
    wavelength_m = 299792458.0 / freq_hz
    theta_null = np.arcsin(J1_ZERO_1 * wavelength_m / (np.pi * beam.dish_diameter_m))
    assert beam.power_response(theta_null, freq_hz) < 1e-8

    # Just inside/outside the null the response is not (numerically) zero.
    assert beam.power_response(0.9 * theta_null, freq_hz) > 1e-4
    assert beam.power_response(1.1 * theta_null, freq_hz) > 1e-6


def test_airy_beam_x_argument_matches_the_documented_formula():
    beam = AiryBeam(dish_diameter_m=4.5)
    theta, freq_hz = 0.002, 1.4e9
    wavelength_m = 299792458.0 / freq_hz
    expected = np.pi * beam.dish_diameter_m * np.sin(theta) / wavelength_m
    assert beam.x_argument(theta, freq_hz) == pytest.approx(expected)


def test_airy_beam_attenuates_more_at_higher_frequency():
    beam = AiryBeam(dish_diameter_m=4.5)
    theta = 0.005
    low = beam.power_response(theta, 1.0e9)
    high = beam.power_response(theta, 2.0e9)
    assert high < low


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
@pytest.mark.parametrize("cls", [GaussianBeam, AiryBeam])
@pytest.mark.parametrize("bad_diameter", [0.0, -1.0, float("nan"), float("inf")])
def test_beam_rejects_bad_dish_diameter(cls, bad_diameter):
    with pytest.raises(ValueError):
        cls(dish_diameter_m=bad_diameter)


@pytest.mark.parametrize("cls", [GaussianBeam, AiryBeam])
def test_beam_rejects_negative_theta(cls):
    beam = cls(dish_diameter_m=4.5)
    with pytest.raises(ValueError):
        beam.power_response(-0.01, 1.4e9)


@pytest.mark.parametrize("cls", [GaussianBeam, AiryBeam])
@pytest.mark.parametrize("bad_theta", [float("nan"), float("inf")])
def test_beam_rejects_non_finite_theta(cls, bad_theta):
    beam = cls(dish_diameter_m=4.5)
    with pytest.raises(ValueError):
        beam.power_response(bad_theta, 1.4e9)


@pytest.mark.parametrize("cls", [GaussianBeam, AiryBeam])
@pytest.mark.parametrize("bad_freq", [0.0, -1.0e9, float("nan"), float("inf")])
def test_beam_rejects_bad_frequency(cls, bad_freq):
    beam = cls(dish_diameter_m=4.5)
    with pytest.raises(ValueError):
        beam.power_response(0.01, bad_freq)


def test_primary_beam_is_abstract():
    with pytest.raises(TypeError):
        PrimaryBeam()


# ----------------------------------------------------------------------
# Default off: attaching no primary_beam must not move a single byte
# ----------------------------------------------------------------------
def test_no_primary_beam_reproduces_the_channelizer_reference_bytes(default_array, start_time):
    """`primary_beam` defaults to None; the rest of the module must be untouched.

    Reuses `test_channelizer`'s already-hash-pinned reference scenario
    (sky source + spectral line + interference, no filterbank) -- if
    adding the `primary_beam`/`pointing_center` keywords to
    `VoltageSimulator.__init__` perturbed anything about the existing
    synthesis path, this would fail exactly like the channelizer test it
    reuses.
    """
    sim = channelizer_reference_simulator(default_array, start_time)
    assert sim.primary_beam is None
    for index, expected in enumerate(REFERENCE_DIGESTS):
        data = np.ascontiguousarray(sim.block(index).data)
        assert hashlib.sha256(data.tobytes()).hexdigest() == expected


def test_no_primary_beam_leaves_beam_response_none(default_array, start_time):
    phase_center = zenith_phase_center(default_array, start_time, duration_s=1.0)
    source = PointSource.from_lm(phase_center, (0.005, 0.0), flux_jy=1.0)
    sim = VoltageSimulator(
        default_array,
        phase_center,
        start_time,
        [source],
        n_chan=8,
        n_blocks=1,
        n_time_per_block=16,
        rng=np.random.default_rng(0),
    )
    assert sim.beam_response() is None
    assert sim.block(0).beam_response is None


# ----------------------------------------------------------------------
# End-to-end: imaged flux ratio equals the beam's power response
# ----------------------------------------------------------------------
def _image_two_sources(array, start_time, beam, *, n_pol=1, seed=2026):
    """One on-axis, one off-axis source of equal catalog flux, imaged."""
    phase_center = zenith_phase_center(array, start_time, duration_s=1.0)
    l_off, m_off = 0.01, 0.0
    flux_jy = 3.0
    src_on = PointSource.from_lm(phase_center, (0.0, 0.0), flux_jy=flux_jy, name="on")
    src_off = PointSource.from_lm(phase_center, (l_off, m_off), flux_jy=flux_jy, name="off")

    sim = VoltageSimulator(
        array,
        phase_center,
        start_time,
        [src_on, src_off],
        primary_beam=beam,
        n_chan=32,
        n_blocks=6,
        n_time_per_block=400,
        noise_std=0.0,
        n_pol=n_pol,
        rng=np.random.default_rng(seed),
    )
    vis = correlate(sim.blocks())
    image_on, _, _ = dirty_image(vis, np.array([0.0]), np.array([0.0]))
    image_off, _, _ = dirty_image(vis, np.array([l_off]), np.array([m_off]))
    return sim, flux_jy, image_on[0, 0], image_off[0, 0]


@pytest.mark.parametrize("n_pol", [1, 2])
def test_offcentre_flux_matches_catalog_flux_times_power_response(default_array, start_time, n_pol):
    """RFIBench acceptance test: imaged flux == catalog_flux * power_response.

    The on-axis source is a control: its beam response is 1.0, so it
    should image at (approximately) its full catalog flux regardless of
    the beam. The off-axis source's imaged flux, divided by its catalog
    flux, must equal the beam's power response at its offset -- this is
    exactly the effective-SEFD-changing behaviour that motivates the beam
    model in the first place.
    """
    beam = GaussianBeam(dish_diameter_m=4.5)
    sim, flux_jy, image_on, image_off = _image_two_sources(
        default_array, start_time, beam, n_pol=n_pol
    )

    theta = 0.01  # matches l_off, m_off = 0 in _image_two_sources
    expected_response = beam.power_response(theta, sim.freq_hz).mean()

    assert image_on == pytest.approx(flux_jy, rel=0.10)
    assert image_off == pytest.approx(flux_jy * expected_response, rel=0.10)
    assert image_off / image_on == pytest.approx(expected_response, rel=0.10)


def test_beam_response_ground_truth_matches_what_was_applied(default_array, start_time):
    """`VoltageBlock.beam_response` records exactly the factor imaging implies."""
    beam = GaussianBeam(dish_diameter_m=4.5)
    sim, flux_jy, image_on, image_off = _image_two_sources(default_array, start_time, beam)

    block = sim.block(0)
    assert block.beam_response.shape == (2, sim.n_chan)
    # Source 0 is on-axis: response 1.0 at every channel.
    assert np.all(block.beam_response[0] == pytest.approx(1.0))
    # Source 1 is off-axis: matches the beam evaluated directly.
    theta = 0.01
    expected = beam.power_response(theta, sim.freq_hz)
    assert np.allclose(block.beam_response[1], expected)

    # And it is the same array (up to noise/statistics) that explains the
    # imaged flux ratio measured above.
    assert image_off / image_on == pytest.approx(block.beam_response[1].mean(), rel=0.10)


def test_airy_beam_also_attenuates_offcentre_flux_in_imaging(default_array, start_time):
    """Same acceptance test with the sidelobed `AiryBeam`, well inside its main lobe."""
    beam = AiryBeam(dish_diameter_m=4.5)
    sim, flux_jy, image_on, image_off = _image_two_sources(default_array, start_time, beam)

    theta = 0.01
    expected_response = beam.power_response(theta, sim.freq_hz).mean()
    assert image_off / image_on == pytest.approx(expected_response, rel=0.10)
    # A sanity floor: at this offset (well inside the first null for a
    # 4.5 m dish at L band) the beam should attenuate noticeably but not
    # annihilate the source.
    assert 0.05 < expected_response < 0.99


def test_pointing_center_defaults_to_phase_center(default_array, start_time):
    """With no explicit `pointing_center`, the beam is centered on `phase_center`."""
    phase_center = zenith_phase_center(default_array, start_time, duration_s=1.0)
    source = PointSource.from_lm(phase_center, (0.01, 0.0), flux_jy=1.0)
    beam = GaussianBeam(dish_diameter_m=4.5)
    sim = VoltageSimulator(
        default_array,
        phase_center,
        start_time,
        [source],
        primary_beam=beam,
        n_chan=8,
        n_blocks=1,
        n_time_per_block=16,
        rng=np.random.default_rng(0),
    )
    assert sim.pointing_center is phase_center
    expected = beam.power_response(0.01, sim.freq_hz)
    assert np.allclose(sim.beam_response()[0], expected)


def test_explicit_pointing_center_changes_the_offset(default_array, start_time):
    """A `pointing_center` different from `phase_center` shifts the beam, not the sky."""
    from astropy import units as u

    phase_center = zenith_phase_center(default_array, start_time, duration_s=1.0)
    # A source exactly at the phase center...
    source = PointSource.from_lm(phase_center, (0.0, 0.0), flux_jy=1.0)
    beam = GaussianBeam(dish_diameter_m=4.5)
    # ...but the beam is pointed 0.01 rad away in Dec, so the source sees
    # the beam's shoulder instead of its peak.
    pointing_center = phase_center.directional_offset_by(0 * u.deg, 0.01 * u.rad)

    sim = VoltageSimulator(
        default_array,
        phase_center,
        start_time,
        [source],
        primary_beam=beam,
        pointing_center=pointing_center,
        n_chan=8,
        n_blocks=1,
        n_time_per_block=16,
        rng=np.random.default_rng(0),
    )
    response = sim.beam_response()[0]
    assert np.all(response < 1.0)
    assert np.allclose(response, beam.power_response(0.01, sim.freq_hz), atol=1e-3)
