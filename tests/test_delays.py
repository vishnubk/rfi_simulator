"""Tests for rfi_simulator.delays -- the geometric delay sign convention.

The first two tests are the hand-computable ones: a single East-West
baseline with the source on the horizon due east (delay = baseline / c)
and at the zenith (delay = 0).
"""

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord

from rfi_simulator import ArrayConfig
from rfi_simulator.delays import (
    SPEED_OF_LIGHT_M_S,
    earth_location,
    enu_unit_vector,
    geometric_delays_s,
    lm_basis_enu,
    source_unit_vectors_enu,
    zenith_coord,
)
from rfi_simulator.sky import lm_from_radec, radec_from_lm

BASELINE_M = 100.0


@pytest.fixture
def east_west_pair():
    """Two antennas: one at the origin, one 100 m due east."""
    return np.array([[0.0, 0.0, 0.0], [BASELINE_M, 0.0, 0.0]])


def test_source_due_east_at_horizon_gives_baseline_over_c(east_west_pair):
    """Hand-computable: source due east on the horizon, E-W baseline.

    The eastern antenna is 100 m closer to the source, so it sees the
    wavefront 100 m / c earlier: tau = -(r . s_hat) / c = -B / c.
    """
    s_hat = enu_unit_vector(alt_rad=0.0, az_rad=np.pi / 2.0)
    np.testing.assert_allclose(s_hat, [1.0, 0.0, 0.0], atol=1e-12)

    tau_s = geometric_delays_s(east_west_pair, s_hat)

    expected_s = BASELINE_M / SPEED_OF_LIGHT_M_S
    assert tau_s[0] == pytest.approx(0.0, abs=1e-18)
    assert tau_s[1] == pytest.approx(-expected_s, rel=1e-12)
    assert abs(tau_s[1] - tau_s[0]) == pytest.approx(expected_s, rel=1e-12)
    # Sanity on the magnitude: 100 m is about a third of a microsecond.
    assert expected_s == pytest.approx(3.3356e-7, rel=1e-3)


def test_zenith_source_gives_zero_delay_for_flat_array(east_west_pair):
    """Hand-computable: a zenith source has zero delay across a flat array."""
    s_hat = enu_unit_vector(alt_rad=np.pi / 2.0, az_rad=0.0)
    np.testing.assert_allclose(s_hat, [0.0, 0.0, 1.0], atol=1e-12)

    tau_s = geometric_delays_s(east_west_pair, s_hat)
    np.testing.assert_allclose(tau_s, 0.0, atol=1e-18)


def test_source_due_west_flips_the_sign(east_west_pair):
    """Moving the source to the opposite horizon flips the delay sign."""
    s_hat_west = enu_unit_vector(alt_rad=0.0, az_rad=-np.pi / 2.0)
    tau_s = geometric_delays_s(east_west_pair, s_hat_west)
    assert tau_s[1] == pytest.approx(+BASELINE_M / SPEED_OF_LIGHT_M_S, rel=1e-12)


def test_baseline_perpendicular_to_source_has_zero_delay():
    """A North-South baseline sees no delay for a source due east."""
    positions = np.array([[0.0, 0.0, 0.0], [0.0, BASELINE_M, 0.0]])
    s_hat = enu_unit_vector(alt_rad=0.0, az_rad=np.pi / 2.0)
    np.testing.assert_allclose(geometric_delays_s(positions, s_hat), 0.0, atol=1e-18)


def test_delays_broadcast_over_time(default_array, start_time):
    """Delays are evaluated per epoch: Earth rotation must change them."""
    location = earth_location(default_array)
    coord = zenith_coord(location, start_time)
    times = start_time + np.array([0.0, 1.0, 2.0]) * u.s

    s_hat = source_unit_vectors_enu(coord, times, location)
    assert s_hat.shape == (3, 3)

    tau_s = geometric_delays_s(default_array.antenna_positions_enu_m, s_hat)
    assert tau_s.shape == (3, default_array.n_antennas)

    # Over one second the sky rotates ~15 arcsec; with ~50 m baselines that
    # is a few picoseconds -- small, but it must not be exactly zero.
    drift_s = np.abs(tau_s[1] - tau_s[0]).max()
    assert drift_s > 0.0
    assert drift_s < 1e-10


def test_zenith_coord_transforms_back_to_the_zenith(default_array, start_time):
    """`zenith_coord` really is at altitude 90 degrees."""
    location = earth_location(default_array)
    coord = zenith_coord(location, start_time)
    altaz = coord.transform_to(AltAz(obstime=start_time, location=location))
    assert altaz.alt.to_value(u.deg) == pytest.approx(90.0, abs=1e-6)


def test_lm_basis_is_orthonormal(default_array, start_time):
    """The (e_l, e_m, s0) triad is orthonormal to numerical precision."""
    location = earth_location(default_array)
    phase_center = zenith_coord(location, start_time)

    s0_hat, e_l, e_m = lm_basis_enu(phase_center, start_time, location)

    # The triad comes from a central difference of the ICRS->ENU transform,
    # so orthonormality holds to the finite-difference truncation error
    # (~1e-8), not to machine precision.
    for vector in (s0_hat, e_l, e_m):
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-7)
    assert np.dot(e_l, e_m) == pytest.approx(0.0, abs=1e-7)
    assert np.dot(e_l, s0_hat) == pytest.approx(0.0, abs=1e-7)
    assert np.dot(e_m, s0_hat) == pytest.approx(0.0, abs=1e-7)


def test_lm_basis_agrees_with_sky_projection(default_array, start_time):
    """Projecting onto (e_l, e_m) in ENU reproduces sky.lm_from_radec."""
    location = earth_location(default_array)
    phase_center = zenith_coord(location, start_time)

    lm_true = np.array([0.0087, -0.0052])
    source = radec_from_lm(phase_center, lm_true)

    np.testing.assert_allclose(lm_from_radec(phase_center, source), lm_true, atol=1e-12)

    s0_hat, e_l, e_m = lm_basis_enu(phase_center, start_time, location)
    s_hat = source_unit_vectors_enu(source, start_time, location)

    # Aberration makes ICRS->AltAz slightly non-rotational; 1e-6 in
    # direction cosine is ~0.2 arcsec, far below a pixel.
    assert np.dot(s_hat, e_l) == pytest.approx(lm_true[0], abs=1e-6)
    assert np.dot(s_hat, e_m) == pytest.approx(lm_true[1], abs=1e-6)


def test_lm_of_a_fixed_source_is_time_invariant(default_array, start_time):
    """(l, m) of a fixed source does not change as the Earth rotates."""
    location = earth_location(default_array)
    phase_center = zenith_coord(location, start_time)
    source = radec_from_lm(phase_center, np.array([0.0087, -0.0052]))

    times = start_time + np.array([0.0, 2.0]) * u.s
    _, e_l, e_m = lm_basis_enu(phase_center, times, location)
    s_hat = source_unit_vectors_enu(source, times, location)

    l_values = np.einsum("tj,tj->t", s_hat, e_l)
    m_values = np.einsum("tj,tj->t", s_hat, e_m)
    assert l_values[1] == pytest.approx(l_values[0], abs=1e-9)
    assert m_values[1] == pytest.approx(m_values[0], abs=1e-9)


def test_geometric_delays_validates_shapes():
    """Bad input shapes raise ValueError rather than broadcasting silently."""
    with pytest.raises(ValueError, match=r"\(n_antennas, 3\)"):
        geometric_delays_s(np.zeros((4, 2)), np.array([0.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match=r"\(\.\.\., 3\)"):
        geometric_delays_s(np.zeros((4, 3)), np.array([0.0, 1.0]))


def test_earth_location_matches_the_config(default_array):
    """`earth_location` round-trips the configured geodetic origin."""
    location = earth_location(default_array)
    geodetic = location.to_geodetic()
    assert geodetic.lat.to_value(u.deg) == pytest.approx(default_array.latitude_deg, abs=1e-9)
    assert geodetic.lon.to_value(u.deg) == pytest.approx(default_array.longitude_deg, abs=1e-9)
    assert geodetic.height.to_value(u.m) == pytest.approx(default_array.height_m, abs=1e-6)


def test_source_unit_vector_matches_manual_altaz(default_array, start_time):
    """`source_unit_vectors_enu` is just AltAz -> ENU, nothing more."""
    location = earth_location(default_array)
    coord = SkyCoord(ra=120.0 * u.deg, dec=25.0 * u.deg, frame="icrs")
    altaz = coord.transform_to(AltAz(obstime=start_time, location=location))

    expected = enu_unit_vector(altaz.alt.to_value(u.rad), altaz.az.to_value(u.rad))
    np.testing.assert_allclose(
        source_unit_vectors_enu(coord, start_time, location), expected, atol=1e-15
    )


def test_delay_is_linear_in_position():
    """tau is a linear functional of the antenna position vector."""
    positions = np.array([[10.0, 20.0, 5.0], [20.0, 40.0, 10.0]])
    s_hat = enu_unit_vector(alt_rad=0.7, az_rad=1.3)
    tau_s = geometric_delays_s(positions, s_hat)
    assert tau_s[1] == pytest.approx(2.0 * tau_s[0], rel=1e-12)


def test_array_config_positions_feed_delays(default_array, start_time):
    """The default array's delays are bounded by its longest baseline."""
    location = earth_location(default_array)
    coord = zenith_coord(location, start_time)
    s_hat = source_unit_vectors_enu(coord, start_time, location)
    tau_s = geometric_delays_s(default_array.antenna_positions_enu_m, s_hat)
    max_radius_m = np.linalg.norm(default_array.antenna_positions_enu_m, axis=1).max()
    assert np.abs(tau_s).max() <= max_radius_m / SPEED_OF_LIGHT_M_S + 1e-15


def test_array_config_is_unmodified_by_delay_computation(default_array, start_time):
    """Delay evaluation must not mutate the array configuration."""
    before = default_array.antenna_positions_enu_m.copy()
    location = earth_location(default_array)
    geometric_delays_s(
        default_array.antenna_positions_enu_m,
        source_unit_vectors_enu(zenith_coord(location, start_time), start_time, location),
    )
    np.testing.assert_array_equal(default_array.antenna_positions_enu_m, before)


def test_array_config_accepts_quantity_positions():
    """Sanity check that ArrayConfig + delays compose with Quantity input."""
    positions = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]) * u.km
    array = ArrayConfig(
        antenna_positions_enu_m=positions,
        latitude_deg=37.234,
        longitude_deg=-118.282,
        height_m=1222.0,
    )
    s_hat = enu_unit_vector(alt_rad=0.0, az_rad=np.pi / 2.0)
    tau_s = geometric_delays_s(array.antenna_positions_enu_m, s_hat)
    assert tau_s[1] == pytest.approx(-BASELINE_M / SPEED_OF_LIGHT_M_S, rel=1e-12)
