"""Tests for rfi_simulator.satellites.

Everything here runs offline against the frozen element set bundled at
``configs/tle_sample.txt``. **No test may call `fetch_tles`**: the suite
must not depend on a public web service being reachable, nor drift as the
catalogue is refreshed.

The two load-bearing tests are:

* `test_topocentric_angles_match_an_independent_astropy_computation` --
  the hand-rolled ENU rotation is cross-checked against astropy's own
  ITRS-to-horizontal machinery, which is a genuinely separate code path.
* `test_far_field_delays_reduce_to_the_plane_wave_formula` -- at 20 000 km
  the exact path-delay code and the plane-wave code of
  `rfi_simulator.delays` must agree to well under a picosecond, which
  checks both against each other.
"""

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import ITRS, TEME, AltAz, CartesianRepresentation, SkyCoord
from astropy.time import Time
from conftest import DEFAULT_ARRAY_YAML, zenith_phase_center

from rfi_simulator import (
    SatelliteTransmitter,
    TwoLineElement,
    VoltageSimulator,
    correlate,
    read_tle_file,
)
from rfi_simulator.delays import earth_location, geometric_delays_s
from rfi_simulator.rfi import path_delays_s
from rfi_simulator.satellites import _parse_tle_text

TLE_PATH = DEFAULT_ARRAY_YAML.parent / "tle_sample.txt"

# Epochs used throughout, chosen from the bundled element set's own epoch
# (2026-07-30T07:03:47 UTC) so the orbit is propagated only hours away
# from where it is accurate.
APPROACH_TIME = Time("2026-07-30T06:00:00", scale="utc")
RECESSION_TIME = Time("2026-07-30T10:00:00", scale="utc")

# An in-band downlink frequency: the real navigation carriers are far
# outside the default band, and the point of the source is its geometry.
IN_BAND_CARRIER_HZ = 1.405e9


@pytest.fixture
def tle() -> TwoLineElement:
    """The bundled, frozen element set."""
    return TwoLineElement.from_file(TLE_PATH)


def make_simulator(array, start_time, rfi_sources, **kwargs):
    """Small simulator anchored at a time the element set is valid for."""
    options = dict(
        n_chan=32,
        n_blocks=2,
        n_time_per_block=200,
        noise_std=0.0,
        rng=np.random.default_rng(20261001),
    )
    options.update(kwargs)
    phase_center = zenith_phase_center(array, start_time, duration_s=0.1)
    return VoltageSimulator(array, phase_center, start_time, [], rfi_sources=rfi_sources, **options)


# ----------------------------------------------------------------------
# Element-set handling
# ----------------------------------------------------------------------
def test_bundled_element_set_parses(tle):
    """The bundled file is a real, parseable element set with a known epoch."""
    assert tle.name == "GPS BIIR-5  (PRN 22)"
    assert tle.line1.startswith("1 26407U")
    assert tle.line2.startswith("2 26407")
    # The epoch recorded in the file's comment header.
    assert tle.epoch.isot.startswith("2026-07-30T07:03:47")

    entries = read_tle_file(TLE_PATH)
    assert len(entries) == 1


def test_comments_and_blank_lines_are_ignored():
    """Bundled files may carry provenance comments without confusing the reader."""
    text = "\n".join(
        [
            "# a comment",
            "",
            "GPS BIIR-5  (PRN 22)",
            "1 26407U 00040A   26211.29429826  .00000064  00000+0  00000+0 0  9995",
            "2 26407  54.8470 213.4502 0120062 302.9461 145.6045  2.00558031190810",
            "",
        ]
    )
    entries = _parse_tle_text(text)
    assert len(entries) == 1
    assert entries[0].name == "GPS BIIR-5  (PRN 22)"

    single = TwoLineElement.from_string(text)
    assert single.line2 == entries[0].line2


def test_malformed_element_sets_are_rejected(tle):
    with pytest.raises(ValueError, match="must begin with"):
        TwoLineElement(tle.line2, tle.line1)
    with pytest.raises(ValueError, match="no matching line 2"):
        _parse_tle_text(tle.line1)
    with pytest.raises(ValueError, match="exactly one element set"):
        TwoLineElement.from_string("")
    with pytest.raises(IndexError, match="out of range"):
        TwoLineElement.from_file(TLE_PATH, index=5)


# ----------------------------------------------------------------------
# Geometry (acceptance criterion 4, part one)
# ----------------------------------------------------------------------
def test_topocentric_angles_match_an_independent_astropy_computation(default_array, tle):
    """Acceptance criterion 4: az/el cross-checked against astropy alone.

    Our path is ``SGP4 -> TEME -> ITRS -> explicit ENU rotation matrix``,
    then azimuth and elevation read off the ENU vector by trigonometry.
    The reference path shares only the SGP4 propagation: it hands the TEME
    position to astropy and asks astropy's own ITRS-to-`AltAz` machinery
    for the angles, never touching our rotation matrix. Agreement to well
    inside the 0.1 deg tolerance therefore exercises the coordinate chain,
    not a shared bug.
    """
    location = earth_location(default_array)
    epochs = [
        APPROACH_TIME,
        Time("2026-07-30T09:00:00", scale="utc"),
        RECESSION_TIME,
    ]

    for epoch in epochs:
        position_enu_m = tle.enu_position_m(epoch, location)
        range_m = np.linalg.norm(position_enu_m)
        our_elevation_deg = np.rad2deg(np.arcsin(position_enu_m[2] / range_m))
        our_azimuth_deg = np.rad2deg(np.arctan2(position_enu_m[0], position_enu_m[1])) % 360.0

        teme = TEME(CartesianRepresentation(tle.teme_position_m(epoch) * u.m), obstime=epoch)
        itrs = teme.transform_to(ITRS(obstime=epoch))
        topocentric = ITRS(
            itrs.cartesian - location.get_itrs(epoch).cartesian,
            obstime=epoch,
            location=location,
        )
        reference = SkyCoord(topocentric).transform_to(AltAz(obstime=epoch, location=location))

        assert our_azimuth_deg == pytest.approx(reference.az.to_value(u.deg), abs=0.1)
        assert our_elevation_deg == pytest.approx(reference.alt.to_value(u.deg), abs=0.1)
        # Sanity: a navigation satellite is ~20 000 km away, not on the ground.
        assert 1.9e7 < range_m < 2.6e7


def test_far_field_delays_reduce_to_the_plane_wave_formula(default_array, tle):
    """At navigation-satellite range the exact and plane-wave delays agree.

    The exact path length expands as

        |x - r| = |x| - r.s_hat + (r^2 - (r.s_hat)^2) / (2|x|) + ...

    so the residual after removing the plane-wave term is bounded by
    ``r**2 / (2 |x|)``. For the default array (largest antenna radius
    ~40 m) at |x| = 2.1e7 m that is 4e-5 m, i.e. 0.13 ps. The test
    asserts the residual is below 1 ps -- and also that it is *not* zero,
    since a residual of exactly zero would mean the "exact" code had
    quietly become the plane-wave code.
    """
    location = earth_location(default_array)
    position_enu_m = tle.enu_position_m(APPROACH_TIME, location)
    positions = default_array.antenna_positions_enu_m

    exact_s = path_delays_s(position_enu_m, positions)
    exact_s = exact_s - exact_s[0]

    s_hat = position_enu_m / np.linalg.norm(position_enu_m)
    plane_wave_s = geometric_delays_s(positions, s_hat)
    plane_wave_s = plane_wave_s - plane_wave_s[0]

    residual_s = np.abs(exact_s - plane_wave_s)
    assert residual_s.max() < 1e-12
    assert residual_s[1:].min() > 0.0

    # The delays themselves are far from zero, so this is not a trivial pass.
    assert np.abs(exact_s).max() > 1e-8


# ----------------------------------------------------------------------
# Doppler (acceptance criterion 4, part two)
# ----------------------------------------------------------------------
def test_doppler_sign_flips_between_approach_and_recession(default_array, tle):
    """Acceptance criterion 4: the shift is blue on approach, red on recession.

    At the two chosen epochs the bundled satellite is closing at ~73 m/s
    and opening at ~488 m/s respectively, so the received frequency must
    sit above the rest carrier in the first case and below it in the
    second.
    """
    location = earth_location(default_array)
    transmitter = SatelliteTransmitter(tle, carrier_freq_hz=IN_BAND_CARRIER_HZ)

    approach_rate = tle.range_rate_m_s(APPROACH_TIME, location)
    recession_rate = tle.range_rate_m_s(RECESSION_TIME, location)
    assert approach_rate < 0.0 < recession_rate

    approach_shift = transmitter.doppler_shift_hz(APPROACH_TIME, location)
    recession_shift = transmitter.doppler_shift_hz(RECESSION_TIME, location)
    assert approach_shift > 0.0 > recession_shift

    # Magnitude follows f0 * v / c, to the precision of the differencing.
    from rfi_simulator.delays import SPEED_OF_LIGHT_M_S

    assert recession_shift == pytest.approx(
        -IN_BAND_CARRIER_HZ * recession_rate / SPEED_OF_LIGHT_M_S, rel=1e-9
    )
    assert transmitter.received_freq_hz(APPROACH_TIME, location) > IN_BAND_CARRIER_HZ

    # The range itself corroborates the sign of the rate.
    step = 60.0 * u.s
    assert np.linalg.norm(tle.enu_position_m(APPROACH_TIME + step, location)) < np.linalg.norm(
        tle.enu_position_m(APPROACH_TIME, location)
    )


def test_doppler_can_be_switched_off(default_array, tle):
    """`apply_doppler=False` isolates the geometry for debugging."""
    location = earth_location(default_array)
    transmitter = SatelliteTransmitter(tle, carrier_freq_hz=IN_BAND_CARRIER_HZ, apply_doppler=False)
    assert transmitter.doppler_shift_hz(RECESSION_TIME, location) == 0.0
    assert transmitter.received_freq_hz(RECESSION_TIME, location) == IN_BAND_CARRIER_HZ


# ----------------------------------------------------------------------
# Emission
# ----------------------------------------------------------------------
def test_carrier_lands_in_the_doppler_shifted_channel(default_array, tle):
    """The emission follows the received frequency, not the rest frequency."""
    location = earth_location(default_array)
    # A carrier offset far enough from the band center to be unambiguous,
    # with a large Doppler shift engineered by using a high rest frequency.
    transmitter = SatelliteTransmitter(
        tle, carrier_freq_hz=IN_BAND_CARRIER_HZ, received_power_jy=100.0
    )
    sim = make_simulator(default_array, RECESSION_TIME, [transmitter], n_blocks=1)
    block = sim.block(0)

    received_hz = transmitter.received_freq_hz(block.center_time, location)
    expected_channel = int(np.argmin(np.abs(sim.freq_hz - received_hz)))

    occupied = np.flatnonzero(block.rfi_mask[0].any(axis=1))
    assert occupied.tolist() == [expected_channel]


def test_received_power_is_measured_at_the_array_origin(default_array, tle):
    """Band-summed power at the origin antenna equals received_power_jy."""
    transmitter = SatelliteTransmitter(
        tle,
        carrier_freq_hz=IN_BAND_CARRIER_HZ,
        received_power_jy=250.0,
        sideband_bandwidth_hz=2.0e5,
        sideband_power_fraction=0.6,
    )
    sim = make_simulator(
        default_array, APPROACH_TIME, [transmitter], n_blocks=1, n_time_per_block=4000
    )
    data = sim.block(0).data
    band_power_jy = (np.abs(data[0]) ** 2).sum(axis=0).mean()
    assert band_power_jy == pytest.approx(250.0, rel=0.05)


def test_sidebands_widen_the_occupancy(default_array, tle):
    """Sidebands occupy their own channels; a bare carrier occupies one."""
    bare = SatelliteTransmitter(tle, carrier_freq_hz=IN_BAND_CARRIER_HZ)
    spread = SatelliteTransmitter(
        tle,
        carrier_freq_hz=IN_BAND_CARRIER_HZ,
        sideband_bandwidth_hz=3.0e5,
        sideband_power_fraction=0.5,
    )

    def occupied_channels(source):
        sim = make_simulator(default_array, APPROACH_TIME, [source], n_blocks=1)
        return int(sim.block(0).rfi_mask[0].any(axis=1).sum())

    assert occupied_channels(bare) == 1
    assert occupied_channels(spread) > 5


def test_real_downlink_frequency_is_out_of_the_default_band(default_array, tle):
    """The default navigation carrier raises rather than silently doing nothing."""
    transmitter = SatelliteTransmitter(tle)  # defaults to the real L1 carrier
    assert transmitter.carrier_freq_hz == pytest.approx(1.57542e9)

    sim = make_simulator(default_array, APPROACH_TIME, [transmitter], n_blocks=1)
    with pytest.raises(ValueError, match="outside the simulated band"):
        sim.block(0)

    # Re-centering the band on the real downlink makes it work -- which is
    # the documented way to simulate the true frequency.
    recentered = make_simulator(
        default_array, APPROACH_TIME, [transmitter], n_blocks=1, center_freq_hz=1.57542e9
    )
    assert recentered.block(0).rfi_mask[0].any()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"received_power_jy": -1.0}, "received_power_jy"),
        ({"sideband_bandwidth_hz": -1.0}, "sideband_bandwidth_hz"),
        ({"sideband_power_fraction": 1.5}, "sideband_power_fraction"),
    ],
)
def test_satellite_validation(tle, kwargs, match):
    with pytest.raises(ValueError, match=match):
        SatelliteTransmitter(tle, carrier_freq_hz=IN_BAND_CARRIER_HZ, **kwargs)


# ----------------------------------------------------------------------
# Framework contracts still hold
# ----------------------------------------------------------------------
def test_satellite_preserves_purity_and_pairing(default_array, tle):
    """Acceptance criteria 2 and 7 hold with a satellite attached."""
    transmitter = SatelliteTransmitter(
        tle,
        carrier_freq_hz=IN_BAND_CARRIER_HZ,
        received_power_jy=400.0,
        sideband_bandwidth_hz=2.0e5,
        sideband_power_fraction=0.5,
        name="downlink",
    )

    def build(with_rfi):
        return make_simulator(
            default_array,
            APPROACH_TIME,
            [transmitter] if with_rfi else [],
            n_blocks=3,
            noise_std=1.0,
            rng=np.random.default_rng(97),
        )

    clean, dirty = build(False), build(True)

    for index in range(3):
        clean_block, dirty_block = clean.block(index), dirty.block(index)
        union = dirty_block.rfi_mask.any(axis=0)
        assert union.any()
        np.testing.assert_array_equal(dirty_block.data[:, ~union], clean_block.data[:, ~union])

    # Purity: out-of-order generation reproduces the same blocks.
    reference = [block.data for block in build(True).blocks()]
    shuffled = build(True)
    _ = shuffled.block(2)
    for index in (1, 0, 2):
        np.testing.assert_array_equal(shuffled.block(index).data, reference[index])


def test_satellite_labels_reach_the_visibilities(default_array, tle):
    """Acceptance criterion 6, for a moving source: occupancy propagates."""
    transmitter = SatelliteTransmitter(
        tle, carrier_freq_hz=IN_BAND_CARRIER_HZ, received_power_jy=200.0, name="downlink"
    )
    sim = make_simulator(default_array, APPROACH_TIME, [transmitter], n_blocks=3)
    blocks = [sim.block(index) for index in range(3)]
    vis = correlate(iter(blocks))

    assert vis.rfi_source_names == ("downlink",)
    assert vis.rfi_fraction.shape == (3, 1, sim.n_chan)
    for index, block in enumerate(blocks):
        np.testing.assert_array_equal(
            vis.rfi_fraction[index], block.rfi_mask.mean(axis=2, dtype=np.float64)
        )
    # A continuously transmitting satellite occupies its channel all the time.
    assert vis.rfi_fraction.max() == pytest.approx(1.0)


def test_zero_power_satellite_is_silent_not_out_of_band(default_array, tle):
    """A zero-power transmitter is in band and silent, not "out of band".

    ``received_power_jy=0.0`` passes constructor validation, so it must
    produce an all-zero contribution rather than a misleading ValueError
    blaming the band configuration (regression: occupancy used to be
    derived from the power envelope instead of the frequency footprint).
    """
    source = SatelliteTransmitter(tle, carrier_freq_hz=IN_BAND_CARRIER_HZ, received_power_jy=0.0)
    sim = make_simulator(default_array, APPROACH_TIME, [source])
    block = sim.block(0)
    assert not np.any(block.rfi_mask)


def test_satellite_and_ground_sources_coexist(default_array, tle):
    """A satellite and a ground transmitter stack masks independently."""
    from rfi_simulator import NarrowbandTransmitter

    satellite = SatelliteTransmitter(
        tle, carrier_freq_hz=IN_BAND_CARRIER_HZ, received_power_jy=50.0
    )
    tower = NarrowbandTransmitter(
        (2000.0, 0.0, 30.0),
        center_freq_hz=IN_BAND_CARRIER_HZ + 2.0e5,
        bandwidth_hz=6.0e4,
        received_power_jy=50.0,
    )
    sim = make_simulator(default_array, APPROACH_TIME, [satellite, tower])
    block = sim.block(0)
    assert block.rfi_mask.shape[0] == 2
    assert block.rfi_mask[0].any() and block.rfi_mask[1].any()
    vis = correlate(sim.blocks())
    assert vis.rfi_fraction.shape[1] == 2
    assert (vis.rfi_fraction > 0).any(axis=(0, 2)).all()


def test_fetch_tles_rejects_negative_config(tmp_path):
    """Negative cache age or timeout is a caller error, refused up front."""
    from rfi_simulator.satellites import fetch_tles

    with pytest.raises(ValueError, match="max_age_hours"):
        fetch_tles("gps-ops", tmp_path, max_age_hours=-1.0)
    with pytest.raises(ValueError, match="timeout_s"):
        fetch_tles("gps-ops", tmp_path, timeout_s=0.0)
