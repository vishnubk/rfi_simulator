"""Tests for rfi_simulator.imaging -- the end-to-end physics checks.

Covers acceptance criteria 1 (localization, on two different antenna
layouts), 2 (sign convention: +l images at +l) and 3 (flux and radiometer
noise).
"""

import warnings

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from conftest import SOURCE_L, SOURCE_M, random_flat_array, zenith_phase_center

from rfi_simulator import (
    PointSource,
    VoltageSimulator,
    correlate,
    dirty_image,
    lm_axis,
    uvw_wavelengths,
)
from rfi_simulator.imaging import w_term_phase_rad

# Grid fine enough to resolve the ~lambda/B = 2e-3 rad synthesized beam
# roughly ten times over, so "within half a pixel" is a real constraint.
PIXEL_RAD = 2e-4
GRID = np.arange(-75, 76) * PIXEL_RAD


def image_a_source(array, start_time, lm, flux_jy=1.0, *, seed=99, noise_std=0.0, **kwargs):
    """Simulate one source, correlate, and image it on `GRID`."""
    options = dict(n_chan=16, n_blocks=4, n_time_per_block=250)
    options.update(kwargs)
    phase_center = zenith_phase_center(array, start_time, duration_s=1.0)
    source = PointSource.from_lm(phase_center, lm, flux_jy=flux_jy)
    sim = VoltageSimulator(
        array,
        phase_center,
        start_time,
        [source],
        noise_std=noise_std,
        rng=np.random.default_rng(seed),
        **options,
    )
    vis = correlate(sim.blocks())
    image, l_grid, m_grid = dirty_image(vis, GRID, GRID)
    return vis, image, l_grid, m_grid, source, phase_center


def peak_lm(image, l_grid, m_grid):
    """Direction cosines of the brightest pixel (row index is m)."""
    i_m, i_l = np.unravel_index(np.argmax(image), image.shape)
    return l_grid[i_l], m_grid[i_m]


@pytest.mark.parametrize("layout", ["default", "random"])
def test_source_images_at_the_input_position(default_array, start_time, layout):
    """Acceptance criterion 1: peak within half a pixel of the input (l, m).

    Run on two different antenna layouts to prove the positions are an
    input and not baked into the physics anywhere.
    """
    array = default_array if layout == "default" else random_flat_array(10, seed=4242)

    _, image, l_grid, m_grid, _, _ = image_a_source(array, start_time, (SOURCE_L, SOURCE_M))
    l_peak, m_peak = peak_lm(image, l_grid, m_grid)

    assert abs(l_peak - SOURCE_L) <= 0.5 * PIXEL_RAD
    assert abs(m_peak - SOURCE_M) <= 0.5 * PIXEL_RAD


def test_peak_is_well_above_the_sidelobes(default_array, start_time):
    """The source is unambiguous, not just marginally the largest pixel.

    A 10-element snapshot has brutal sidelobes (~50% of the peak, since the
    fractional bandwidth is under 1% and 2 s of Earth rotation fills in
    almost nothing), so the bar here is deliberately modest for the default
    array and much tighter for a 24-element one.
    """
    for array, min_ratio in ((default_array, 1.5), (random_flat_array(24, seed=808), 4.0)):
        _, image, l_grid, m_grid, _, _ = image_a_source(array, start_time, (SOURCE_L, SOURCE_M))
        l_peak, m_peak = peak_lm(image, l_grid, m_grid)

        far = (np.abs(l_grid - l_peak)[np.newaxis, :] > 0.004) | (
            np.abs(m_grid - m_peak)[:, np.newaxis] > 0.004
        )
        assert image.max() > min_ratio * np.abs(image[far]).max()


@pytest.mark.parametrize("sign", [+1.0, -1.0])
def test_sign_convention_positive_l_images_at_positive_l(default_array, start_time, sign):
    """Acceptance criterion 2: a source at +l must peak at +l, never -l.

    This is the conjugation test. If ``V_ij = <v_i v_j*>`` were swapped, or
    the delay sign flipped, the image would be mirrored and this test would
    fail -- the fix is the convention, never flipping the image axes.
    """
    l_source = sign * SOURCE_L
    _, image, l_grid, m_grid, _, _ = image_a_source(default_array, start_time, (l_source, 0.0))
    l_peak, m_peak = peak_lm(image, l_grid, m_grid)

    assert np.sign(l_peak) == np.sign(l_source)
    assert abs(l_peak - l_source) <= 0.5 * PIXEL_RAD
    assert abs(m_peak) <= 0.5 * PIXEL_RAD

    # The mirrored position is nowhere near as bright.
    i_mirror = int(np.argmin(np.abs(l_grid + l_source)))
    i_zero_m = int(np.argmin(np.abs(m_grid)))
    assert image[i_zero_m, i_mirror] < 0.5 * image.max()


def test_peak_amplitude_matches_the_source_flux(default_array, start_time):
    """Acceptance criterion 3a: noiseless peak within 10% of the input flux."""
    flux_jy = 4.0
    vis, _, _, _, _, _ = image_a_source(
        default_array,
        start_time,
        (SOURCE_L, SOURCE_M),
        flux_jy=flux_jy,
        n_chan=32,
        n_blocks=4,
        n_time_per_block=500,
    )
    # Evaluate the image exactly at the source, so the answer is not
    # limited by where the grid pixels happen to fall.
    image, _, _ = dirty_image(vis, np.array([SOURCE_L]), np.array([SOURCE_M]))
    assert image[0, 0] == pytest.approx(flux_jy, rel=0.10)


def test_image_rms_follows_the_radiometer_equation(default_array, start_time):
    """Acceptance criterion 3b: image RMS matches the radiometer expectation.

    Derivation (noise-only observation, system power P = noise_std**2 Jy
    per antenna):

    * one visibility sample is ``V = (1/N_t) sum_t v_i v_j*`` over
      ``N_t = t_int * chan_width`` voltage samples, so it is a circular
      complex variable with ``Var(V) = P**2 / N_t`` and hence
      ``Var(Re V) = P**2 / (2 N_t)``;
    * distinct baselines are uncorrelated even when they share an antenna,
      because ``E[v_i v_j* v_i* v_k] = E[|v_i|**2] E[v_j*] E[v_k] = 0``
      for independent zero-mean noise;
    * the dirty image averages ``K = N_base * N_chan * N_int`` such
      independent samples, each rotated by a unit-modulus phase (which
      preserves circularity), so

          sigma_image = P / sqrt(2 * K * N_t)
                      = P / sqrt(2 * N_base * N_chan * chan_width * T_total).

    A factor-2 tolerance is allowed, and the test also checks that doubling
    the channel count drops the RMS by sqrt(2).
    """
    array = default_array
    phase_center = zenith_phase_center(array, start_time, duration_s=0.5)
    noise_std = 1.0
    n_blocks = 8
    n_time = 200

    measured = {}
    for n_chan in (32, 64):
        sim = VoltageSimulator(
            array,
            phase_center,
            start_time,
            [],  # noise only: isolates the radiometer term from sidelobes
            n_chan=n_chan,
            n_blocks=n_blocks,
            n_time_per_block=n_time,
            noise_std=noise_std,
            rng=np.random.default_rng(777),
        )
        vis = correlate(sim.blocks())
        image, _, _ = dirty_image(vis, lm_axis(0.04, 31), lm_axis(0.04, 31))

        n_base = int(vis.cross_mask.sum())
        expected = noise_std**2 / np.sqrt(2 * n_base * n_chan * n_blocks * n_time)
        # Equivalent, stated as a Fourier pair:
        total_time_s = n_blocks * n_time * sim.sample_period_s
        expected_alt = noise_std**2 / np.sqrt(
            2 * n_base * n_chan * sim.chan_width_hz * total_time_s
        )
        assert expected_alt == pytest.approx(expected, rel=1e-12)

        measured[n_chan] = (image.std(), expected)
        assert 0.5 < image.std() / expected < 2.0

    ratio = measured[64][0] / measured[32][0]
    assert ratio == pytest.approx(1.0 / np.sqrt(2.0), rel=0.25)


def test_image_of_two_sources_shows_both(start_time):
    """Two sources land at their own positions with roughly their own fluxes.

    Uses a 24-element array: with only 45 baselines the sidelobes of the
    3 Jy source swamp the 1 Jy one, which is a real property of the dirty
    image rather than a bug, but makes for a useless assertion.
    """
    array = random_flat_array(24, seed=606)
    phase_center = zenith_phase_center(array, start_time, duration_s=1.0)
    sources = [
        PointSource.from_lm(phase_center, (0.006, 0.004), flux_jy=3.0),
        PointSource.from_lm(phase_center, (-0.005, -0.007), flux_jy=1.0),
    ]
    sim = VoltageSimulator(
        array,
        phase_center,
        start_time,
        sources,
        n_chan=32,
        n_blocks=4,
        n_time_per_block=500,
        noise_std=0.0,
        rng=np.random.default_rng(31337),
    )
    vis = correlate(sim.blocks())

    for source in sources:
        lm = source.lm(phase_center)
        image, _, _ = dirty_image(vis, np.array([lm[0]]), np.array([lm[1]]))
        assert image[0, 0] == pytest.approx(source.flux_jy, rel=0.25)


def test_autos_are_excluded_by_default(default_array, start_time):
    """Including autos adds a flat offset equal to the system power."""
    array = default_array
    phase_center = zenith_phase_center(array, start_time, duration_s=0.5)
    sim = VoltageSimulator(
        array,
        phase_center,
        start_time,
        [],
        n_chan=8,
        n_blocks=2,
        n_time_per_block=200,
        noise_std=1.0,
        rng=np.random.default_rng(13),
    )
    vis = correlate(sim.blocks())
    grid = lm_axis(0.02, 11)
    without, _, _ = dirty_image(vis, grid, grid)
    with_autos, _, _ = dirty_image(vis, grid, grid, include_autos=True)
    assert abs(without.mean()) < 0.1
    assert with_autos.mean() > 0.1


def test_uvw_scales_with_frequency(default_array, start_time):
    """u, v are in wavelengths: they scale linearly with channel frequency."""
    array = default_array
    phase_center = zenith_phase_center(array, start_time, duration_s=0.5)
    sim = VoltageSimulator(
        array,
        phase_center,
        start_time,
        [],
        n_chan=8,
        n_blocks=1,
        n_time_per_block=8,
        noise_std=1.0,
        rng=np.random.default_rng(2),
    )
    vis = correlate(sim.blocks())
    u, v, _ = uvw_wavelengths(vis)
    assert u.shape == (1, 55, 8)

    ratio = u[0, 1, -1] / u[0, 1, 0]
    assert ratio == pytest.approx(vis.freq_hz[-1] / vis.freq_hz[0], rel=1e-9)
    np.testing.assert_allclose(u[:, vis.auto_mask, :], 0.0, atol=1e-12)
    np.testing.assert_allclose(v[:, vis.auto_mask, :], 0.0, atol=1e-12)


def test_channel_selection_reduces_work_without_moving_the_source(default_array, start_time):
    """Imaging a channel subset keeps the source in the same place."""
    vis, _, _, _, _, _ = image_a_source(
        default_array, start_time, (SOURCE_L, SOURCE_M), n_chan=32, n_blocks=2
    )
    full, l_grid, m_grid = dirty_image(vis, GRID, GRID)
    subset, _, _ = dirty_image(vis, GRID, GRID, channels=slice(None, None, 4))

    assert peak_lm(full, l_grid, m_grid) == peak_lm(subset, l_grid, m_grid)


def test_dirty_image_rejects_empty_selections(default_array, start_time):
    vis, _, _, _, _, _ = image_a_source(
        default_array,
        start_time,
        (SOURCE_L, SOURCE_M),
        n_chan=4,
        n_blocks=1,
        n_time_per_block=8,
    )
    with pytest.raises(ValueError, match="no visibility samples"):
        dirty_image(vis, GRID, GRID, channels=slice(0, 0))


def test_lm_axis_shape_and_span():
    axis = lm_axis(0.04, 5)
    assert axis.shape == (5,)
    assert axis[0] == pytest.approx(-0.02)
    assert axis[-1] == pytest.approx(0.02)
    with pytest.raises(ValueError, match="n_pix"):
        lm_axis(0.04, 0)


@pytest.mark.slow
def test_full_default_observation_end_to_end(default_array, start_time):
    """The shipped defaults: 10 antennas, 384 channels, 61 blocks, ~2 s.

    This is the run users will actually do, and it has to stay
    laptop-fast; it takes ~15 s here including the DFT imaging.
    """
    array = default_array
    phase_center = zenith_phase_center(array, start_time, duration_s=2.0)
    flux_jy = 5.0
    source = PointSource.from_lm(phase_center, (SOURCE_L, SOURCE_M), flux_jy=flux_jy)
    sim = VoltageSimulator(
        array,
        phase_center,
        start_time,
        [source],
        noise_std=1.0,
        rng=np.random.default_rng(20260930),
    )

    assert sim.n_chan == 384
    assert sim.n_blocks == 61
    assert sim.duration_s == pytest.approx(1.999, abs=0.01)

    vis = correlate(sim.blocks())
    assert vis.data.shape == (61, 55, 384)

    image, l_grid, m_grid = dirty_image(vis, GRID, GRID, channels=slice(None, None, 16))
    l_peak, m_peak = peak_lm(image, l_grid, m_grid)
    assert abs(l_peak - SOURCE_L) <= 0.5 * PIXEL_RAD
    assert abs(m_peak - SOURCE_M) <= 0.5 * PIXEL_RAD

    at_source, _, _ = dirty_image(vis, np.array([SOURCE_L]), np.array([SOURCE_M]))
    assert at_source[0, 0] == pytest.approx(flux_jy, rel=0.10)


def test_lm_axis_single_pixel_is_the_field_center():
    """A one-pixel axis is the map center, not the map edge."""
    axis = lm_axis(0.04, 1)
    assert axis.shape == (1,)
    assert axis[0] == 0.0


def test_w_term_is_zero_for_a_flat_array_at_the_zenith(default_array, start_time):
    """The zenith-pointing configuration used elsewhere has no w term."""
    vis, _, _, _, _, _ = image_a_source(
        default_array, start_time, (SOURCE_L, SOURCE_M), n_chan=8, n_blocks=2
    )
    _, _, w = uvw_wavelengths(vis)
    assert w_term_phase_rad(w, GRID, GRID) < 0.01

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        dirty_image(vis, GRID, GRID)


def test_imaging_far_from_the_zenith_warns_about_the_w_term(default_array, start_time):
    """Pointing 40 degrees off the zenith makes the neglected w term matter."""
    array = default_array
    zenith = zenith_phase_center(array, start_time, duration_s=0.5)
    phase_center = SkyCoord(ra=zenith.ra, dec=zenith.dec - 40.0 * u.deg, frame="icrs")
    source = PointSource.from_lm(phase_center, (0.0, 0.0), flux_jy=1.0)
    sim = VoltageSimulator(
        array,
        phase_center,
        start_time,
        [source],
        n_chan=8,
        n_blocks=2,
        n_time_per_block=64,
        noise_std=0.0,
        rng=np.random.default_rng(17),
    )
    vis = correlate(sim.blocks())

    _, _, w = uvw_wavelengths(vis)
    assert w_term_phase_rad(w, GRID, GRID) > 0.1

    with pytest.warns(UserWarning, match="neglected w term"):
        dirty_image(vis, GRID, GRID)

    # ...and the warning can be silenced deliberately.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        dirty_image(vis, GRID, GRID, warn_on_w_term=False)
