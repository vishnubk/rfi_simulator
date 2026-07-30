"""Tests for rfi_simulator.sky: point sources and the SIN projection."""

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from rfi_simulator import PointSource, lm_from_radec, radec_from_lm

PHASE_CENTER = SkyCoord(ra=45.0 * u.deg, dec=30.0 * u.deg, frame="icrs")


@pytest.mark.parametrize(
    "lm",
    [
        (0.0, 0.0),
        (0.0087, -0.0052),
        (-0.02, 0.03),
        (0.2, -0.15),
    ],
)
def test_lm_radec_round_trip(lm):
    """radec_from_lm and lm_from_radec are exact inverses."""
    coord = radec_from_lm(PHASE_CENTER, np.asarray(lm))
    np.testing.assert_allclose(lm_from_radec(PHASE_CENTER, coord), lm, atol=1e-14)


def test_lm_zero_is_the_phase_center():
    """(l, m) = (0, 0) maps back to the phase center itself."""
    coord = radec_from_lm(PHASE_CENTER, np.array([0.0, 0.0]))
    assert coord.separation(PHASE_CENTER).to_value(u.arcsec) == pytest.approx(0.0, abs=1e-6)


def test_positive_l_is_towards_increasing_ra():
    """The l axis points towards increasing RA, m towards increasing Dec."""
    east = radec_from_lm(PHASE_CENTER, np.array([0.01, 0.0]))
    north = radec_from_lm(PHASE_CENTER, np.array([0.0, 0.01]))

    assert east.ra.to_value(u.deg) > PHASE_CENTER.ra.to_value(u.deg)
    assert east.dec.to_value(u.deg) < PHASE_CENTER.dec.to_value(u.deg) + 1e-6
    assert north.dec.to_value(u.deg) > PHASE_CENTER.dec.to_value(u.deg)


def test_small_offsets_are_close_to_angular_separations():
    """For small offsets, sqrt(l^2 + m^2) is the angular separation."""
    lm = np.array([0.0087, -0.0052])
    coord = radec_from_lm(PHASE_CENTER, lm)
    separation_rad = coord.separation(PHASE_CENTER).to_value(u.rad)
    assert separation_rad == pytest.approx(np.hypot(*lm), rel=1e-4)


def test_radec_from_lm_is_vectorized():
    """A (..., 2) input gives a matching array of coordinates."""
    lm = np.array([[0.0, 0.0], [0.01, 0.0], [0.0, -0.01]])
    coords = radec_from_lm(PHASE_CENTER, lm)
    assert coords.shape == (3,)
    np.testing.assert_allclose(lm_from_radec(PHASE_CENTER, coords), lm, atol=1e-14)


def test_point_source_from_lm_round_trip():
    """PointSource.from_lm / .lm round-trip the direction cosines."""
    lm = (0.0087, -0.0052)
    source = PointSource.from_lm(PHASE_CENTER, lm, flux_jy=3.5, name="test")
    np.testing.assert_allclose(source.lm(PHASE_CENTER), lm, atol=1e-14)
    assert source.flux_jy == pytest.approx(3.5)
    assert source.name == "test"


def test_point_source_accepts_quantity_flux():
    """Flux given as a Quantity converts to plain Jy."""
    source = PointSource(flux_jy=2.0 * u.Jy, coord=PHASE_CENTER)
    source_mjy = PointSource(flux_jy=2000.0 * u.mJy, coord=PHASE_CENTER)
    assert source.flux_jy == pytest.approx(2.0)
    assert source_mjy.flux_jy == pytest.approx(2.0)
    assert isinstance(source.flux_jy, float)


@pytest.mark.parametrize("bad_flux", [-1.0, np.nan])
def test_point_source_rejects_bad_flux(bad_flux):
    with pytest.raises(ValueError):
        PointSource(flux_jy=bad_flux, coord=PHASE_CENTER)


def test_point_source_requires_scalar_coord():
    coords = SkyCoord(ra=[10.0, 20.0] * u.deg, dec=[0.0, 1.0] * u.deg, frame="icrs")
    with pytest.raises(ValueError, match="scalar"):
        PointSource(flux_jy=1.0, coord=coords)


def test_radec_from_lm_rejects_off_sky_direction_cosines():
    with pytest.raises(ValueError, match="off the sky"):
        radec_from_lm(PHASE_CENTER, np.array([0.9, 0.9]))


def test_radec_from_lm_rejects_bad_shape():
    with pytest.raises(ValueError, match=r"\(\.\.\., 2\)"):
        radec_from_lm(PHASE_CENTER, np.array([0.1, 0.2, 0.3]))
