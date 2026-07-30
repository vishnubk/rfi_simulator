"""Robustness tests for the interference sources and the catalogue fetcher.

These cover the failure modes that produce *plausible wrong answers*
rather than exceptions, which are the expensive kind:

* a satellite or aircraft below the horizon transmitting through the
  Earth at full strength,
* an element set propagated weeks from its epoch, where the propagator
  returns a smooth orbit that has silently drifted,
* a transmitter placed a few millimetres from an antenna, where the
  ``1/r`` model produces a factor of ``1e9`` in received power,
* a catalogue group name that escapes its cache directory.

The catalogue tests stub the network out entirely; nothing here opens a
socket.
"""

import os
import urllib.error
import urllib.request
import warnings
from contextlib import contextmanager

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time
from conftest import DEFAULT_ARRAY_YAML, zenith_phase_center

from rfi_simulator import (
    ADSBTransponder,
    ArrayConfig,
    NarrowbandTransmitter,
    SatelliteTransmitter,
    TwoLineElement,
    VoltageSimulator,
    enu_from_horizontal,
)
from rfi_simulator.delays import earth_location
from rfi_simulator.rfi import MIN_SEPARATION_WAVELENGTHS, elevation_deg
from rfi_simulator.satellites import MAX_TLE_AGE_DAYS, fetch_tles

TLE_PATH = DEFAULT_ARRAY_YAML.parent / "tle_sample.txt"

# The bundled satellite is well below the horizon here: elevation -20.9 deg.
BELOW_HORIZON_TIME = Time("2026-07-30T11:57:47", scale="utc")
# ... and well above it here.
ABOVE_HORIZON_TIME = Time("2026-07-30T06:00:00", scale="utc")
# It rises through the horizon between these two.
RISE_START_TIME = Time("2026-07-30T02:50:00", scale="utc")

IN_BAND_CARRIER_HZ = 1.405e9


@contextmanager
def warnings_as_errors():
    """Turn any warning into an error, so "does not warn" can be asserted."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


@pytest.fixture
def tle() -> TwoLineElement:
    """The bundled, frozen element set."""
    return TwoLineElement.from_file(TLE_PATH)


def make_simulator(array, start_time, rfi_sources, **kwargs):
    """Small simulator; keyword overrides let a test stretch the block length."""
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
# Horizon cut: satellites
# ----------------------------------------------------------------------
def test_satellite_below_the_horizon_is_silent(default_array, tle):
    """A set satellite contributes exactly nothing -- not a faint signal.

    Without this the bundled element set at 2026-07-30T11:57:47, which is
    20.9 deg *below* the horizon, still injected its full received power:
    a transmitter shining through the Earth.
    """
    location = earth_location(default_array)
    position = tle.enu_position_m(BELOW_HORIZON_TIME, location)
    assert elevation_deg(position) == pytest.approx(-20.9, abs=0.2)

    transmitter = SatelliteTransmitter(
        tle, carrier_freq_hz=IN_BAND_CARRIER_HZ, received_power_jy=500.0
    )
    sim = make_simulator(default_array, BELOW_HORIZON_TIME, [transmitter], n_blocks=1)
    block = sim.block(0)

    np.testing.assert_array_equal(block.data, np.zeros_like(block.data))
    assert not block.rfi_mask.any()

    # The same satellite, above the horizon, is loud -- so the silence
    # above is the horizon cut and not a broken configuration.
    visible = make_simulator(default_array, ABOVE_HORIZON_TIME, [transmitter], n_blocks=1).block(0)
    assert np.abs(visible.data).max() > 0.0
    assert visible.rfi_mask.any()


def test_satellite_horizon_cut_is_configurable(default_array, tle):
    """`min_elevation_deg` raises or removes the cut."""
    transmitter = SatelliteTransmitter(
        tle, carrier_freq_hz=IN_BAND_CARRIER_HZ, received_power_jy=500.0
    )
    location = earth_location(default_array)
    elevation = elevation_deg(tle.enu_position_m(ABOVE_HORIZON_TIME, location))
    assert elevation == pytest.approx(65.15, abs=0.1)

    # A mask above the satellite's elevation silences it.
    masked = SatelliteTransmitter(
        tle,
        carrier_freq_hz=IN_BAND_CARRIER_HZ,
        received_power_jy=500.0,
        min_elevation_deg=elevation + 5.0,
    )
    block = make_simulator(default_array, ABOVE_HORIZON_TIME, [masked], n_blocks=1).block(0)
    assert not block.rfi_mask.any()

    # Disabling the cut lets it transmit through the Earth again, which is
    # occasionally what you want for a controlled experiment.
    unmasked = SatelliteTransmitter(
        tle,
        carrier_freq_hz=IN_BAND_CARRIER_HZ,
        received_power_jy=500.0,
        min_elevation_deg=-90.0,
    )
    through_earth = make_simulator(default_array, BELOW_HORIZON_TIME, [unmasked], n_blocks=1).block(
        0
    )
    assert through_earth.rfi_mask.any()

    assert transmitter.min_elevation_deg == 0.0
    with pytest.raises(ValueError, match="min_elevation_deg"):
        SatelliteTransmitter(tle, carrier_freq_hz=IN_BAND_CARRIER_HZ, min_elevation_deg=120.0)


def test_satellite_rising_pass_switches_on(default_array, tle):
    """A rising pass goes from silent to loud within one observation.

    Orbital motion is minutes-scale, so this test deliberately runs an
    unphysical channelization -- 1 Hz channels give a 1 s sample period
    and 120 s blocks -- purely to let a horizon crossing happen inside a
    short simulation. The Doppler is switched off because the resulting
    band is only a few Hz wide, far narrower than the kilohertz shift.
    """
    transmitter = SatelliteTransmitter(
        tle,
        carrier_freq_hz=IN_BAND_CARRIER_HZ,
        received_power_jy=500.0,
        apply_doppler=False,
        name="rising",
    )
    sim = make_simulator(
        default_array,
        RISE_START_TIME,
        [transmitter],
        n_chan=8,
        chan_width_hz=1.0,
        n_time_per_block=120,
        n_blocks=6,
    )
    assert sim.block_duration_s == pytest.approx(120.0)

    occupied = [bool(sim.block(index).rfi_mask.any()) for index in range(sim.n_blocks)]
    assert occupied[0] is False, "the satellite should still be below the horizon"
    assert occupied[-1] is True, "the satellite should have risen by the last block"
    # Exactly one transition, in the rising direction.
    assert occupied == sorted(occupied)
    assert sum(1 for a, b in zip(occupied, occupied[1:]) if a != b) == 1


# ----------------------------------------------------------------------
# Horizon cut: aircraft
# ----------------------------------------------------------------------
def test_aircraft_below_the_horizon_is_silent(default_array, start_time):
    """An aircraft below the array's horizontal plane contributes nothing."""
    below = ADSBTransponder(
        position_enu_m=(30000.0, 0.0, -5000.0),
        velocity_enu_m_s=(0.0, 0.0, 0.0),
        carrier_freq_hz=IN_BAND_CARRIER_HZ,
        bandwidth_hz=2.0e5,
        received_power_jy=1.0e4,
        message_rate_hz=5000.0,
    )
    assert elevation_deg(below.position_enu_m) < 0.0

    block = make_simulator(default_array, start_time, [below], n_blocks=1).block(0)
    np.testing.assert_array_equal(block.data, np.zeros_like(block.data))
    assert not block.rfi_mask.any()

    with pytest.raises(ValueError, match="min_elevation_deg"):
        ADSBTransponder(position_enu_m=(1.0, 0.0, 1.0), min_elevation_deg=-100.0)


def test_aircraft_climbing_through_the_horizon_switches_on(default_array, start_time):
    """A climbing aircraft transitions from silent to emitting.

    As in the satellite case the channelization is stretched (1 s samples,
    60 s blocks) so that a realistic 20 m/s climb crosses the horizontal
    plane inside the simulation.
    """
    climbing = ADSBTransponder(
        position_enu_m=(30000.0, 0.0, -1000.0),
        velocity_enu_m_s=(0.0, 0.0, 20.0),
        carrier_freq_hz=IN_BAND_CARRIER_HZ,
        bandwidth_hz=0.0,
        received_power_jy=1.0e4,
        message_rate_hz=1.0,
    )
    sim = make_simulator(
        default_array,
        start_time,
        [climbing],
        n_chan=8,
        chan_width_hz=1.0,
        center_freq_hz=IN_BAND_CARRIER_HZ,
        n_time_per_block=60,
        n_blocks=2,
    )

    # Block 0 is centred 30 s in (still 400 m below), block 1 at 90 s
    # (800 m above), so the crossing falls between them.
    assert elevation_deg(climbing.position_at(30.0)) < 0.0
    assert elevation_deg(climbing.position_at(90.0)) > 0.0

    assert not sim.block(0).rfi_mask.any()
    np.testing.assert_array_equal(sim.block(0).data, np.zeros_like(sim.block(0).data))
    assert sim.block(1).rfi_mask.any()


def test_ground_sources_have_no_horizon_cut(default_array, start_time):
    """Ground-based sources keep transmitting from below the plane, by design.

    They are specified by an ENU position with an implied line of sight --
    no terrain model exists to say otherwise -- and their docstrings say
    so. This test pins that documented behaviour so a future horizon cut
    cannot be added silently.
    """
    sunken = NarrowbandTransmitter(
        position_enu_m=(2000.0, 0.0, -500.0),
        center_freq_hz=IN_BAND_CARRIER_HZ,
        bandwidth_hz=1.5e5,
        received_power_jy=200.0,
    )
    assert elevation_deg(sunken.position_enu_m) < 0.0

    block = make_simulator(default_array, start_time, [sunken], n_blocks=1).block(0)
    assert block.rfi_mask.any()
    assert np.abs(block.data).max() > 0.0


# ----------------------------------------------------------------------
# Element-set staleness
# ----------------------------------------------------------------------
def test_stale_element_set_warns(default_array, tle):
    """Propagating weeks from the epoch warns; propagating near it does not."""
    stale_time = tle.epoch + (MAX_TLE_AGE_DAYS + 30.0) * u.day
    transmitter = SatelliteTransmitter(
        tle, carrier_freq_hz=IN_BAND_CARRIER_HZ, received_power_jy=100.0, min_elevation_deg=-90.0
    )

    sim = make_simulator(default_array, stale_time, [transmitter], n_blocks=1)
    with pytest.warns(UserWarning, match="element-set epoch"):
        sim.block(0)

    # Well inside the window: no warning at all.
    fresh = make_simulator(default_array, ABOVE_HORIZON_TIME, [transmitter], n_blocks=1)
    with warnings_as_errors():
        fresh.block(0)


def test_stale_warning_is_symmetric_about_the_epoch(default_array, tle):
    """Propagating far *backwards* is just as wrong as far forwards."""
    early_time = tle.epoch - (MAX_TLE_AGE_DAYS + 30.0) * u.day
    transmitter = SatelliteTransmitter(
        tle, carrier_freq_hz=IN_BAND_CARRIER_HZ, received_power_jy=100.0, min_elevation_deg=-90.0
    )
    sim = make_simulator(default_array, early_time, [transmitter], n_blocks=1)
    with pytest.warns(UserWarning, match="days from"):
        sim.block(0)


# ----------------------------------------------------------------------
# Near-field foot-gun
# ----------------------------------------------------------------------
def test_transmitter_inside_the_array_warns(start_time):
    """A transmitter metres from an antenna is almost always a units slip.

    The ``1/r`` amplitude model is normalized at the array origin, so a
    transmitter 1 mm from an antenna hands that antenna ~1e9 times the
    stated received power. Arithmetically correct, physically nonsense,
    and silent until now.
    """
    array = ArrayConfig(
        antenna_positions_enu_m=np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]]),
        latitude_deg=37.234,
        longitude_deg=-118.282,
        height_m=1222.0,
    )
    # Half a metre from antenna 1, i.e. ~2.3 wavelengths at 1.4 GHz.
    too_close = NarrowbandTransmitter(
        position_enu_m=(50.5, 0.0, 0.0),
        center_freq_hz=IN_BAND_CARRIER_HZ,
        bandwidth_hz=1.5e5,
        received_power_jy=100.0,
    )
    sim = make_simulator(array, start_time, [too_close], n_blocks=1)
    with pytest.warns(UserWarning, match="wavelengths"):
        sim.block(0)

    # Ten wavelengths at 1.405 GHz is ~2.13 m; a transmitter comfortably
    # outside that radius is silent about it.
    wavelength_m = 299792458.0 / IN_BAND_CARRIER_HZ
    assert MIN_SEPARATION_WAVELENGTHS * wavelength_m == pytest.approx(2.13, abs=0.05)

    far_enough = NarrowbandTransmitter(
        position_enu_m=enu_from_horizontal(90.0, 0.0, 2000.0),
        center_freq_hz=IN_BAND_CARRIER_HZ,
        bandwidth_hz=1.5e5,
        received_power_jy=100.0,
    )
    quiet_sim = make_simulator(array, start_time, [far_enough], n_blocks=1)
    with warnings_as_errors():
        quiet_sim.block(0)


# ----------------------------------------------------------------------
# Catalogue fetching -- entirely offline
# ----------------------------------------------------------------------
CATALOGUE_TEXT = "\n".join(
    [
        "GPS BIIR-5  (PRN 22)",
        "1 26407U 00040A   26211.29429826  .00000064  00000+0  00000+0 0  9995",
        "2 26407  54.8470 213.4502 0120062 302.9461 145.6045  2.00558031190810",
        "",
    ]
)
REFRESHED_TEXT = CATALOGUE_TEXT.replace("GPS BIIR-5  (PRN 22)", "GPS BIIR-5  REFRESHED")


class _FakeResponse:
    """Minimal stand-in for the object `urlopen` returns."""

    def __init__(self, text: str) -> None:
        self._text = text

    def read(self) -> bytes:
        return self._text.encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc_info) -> bool:
        return False


@pytest.fixture
def no_network(monkeypatch):
    """Fail loudly if anything in a test reaches for the network."""

    def forbidden(*args, **kwargs):
        raise AssertionError("the test suite must never open a network connection")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    return forbidden


def write_cache(cache_dir, group, text, age_hours=0.0):
    """Write a cache file and back-date it by `age_hours`."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{group}.tle"
    path.write_text(text)
    if age_hours:
        old = os.stat(path).st_mtime - age_hours * 3600.0
        os.utime(path, (old, old))
    return path


def test_fetch_uses_a_fresh_cache_without_touching_the_network(tmp_path, no_network):
    """A cache younger than `max_age_hours` short-circuits the download."""
    write_cache(tmp_path, "gps-ops", CATALOGUE_TEXT, age_hours=1.0)

    entries = fetch_tles("gps-ops", tmp_path, max_age_hours=24.0)

    assert len(entries) == 1
    assert entries[0].name == "GPS BIIR-5  (PRN 22)"


def test_fetch_refreshes_a_stale_cache(tmp_path, monkeypatch):
    """A cache older than `max_age_hours` is re-downloaded and rewritten."""
    cache_path = write_cache(tmp_path, "gps-ops", CATALOGUE_TEXT, age_hours=48.0)

    calls = []

    def fake_urlopen(url, timeout=None):
        calls.append(url)
        return _FakeResponse(REFRESHED_TEXT)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    entries = fetch_tles("gps-ops", tmp_path, max_age_hours=24.0)

    assert len(calls) == 1
    assert "GROUP=gps-ops" in calls[0] and "FORMAT=tle" in calls[0]
    assert entries[0].name == "GPS BIIR-5  REFRESHED"
    # The refreshed text replaced the cache, so the next call is cheap.
    assert "REFRESHED" in cache_path.read_text()


def test_fetch_falls_back_to_a_stale_cache_when_offline(tmp_path, monkeypatch):
    """Offline with a stale cache: warn, and use it. Stale beats nothing."""
    write_cache(tmp_path, "gps-ops", CATALOGUE_TEXT, age_hours=100.0)

    def offline(url, timeout=None):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(urllib.request, "urlopen", offline)

    with pytest.warns(UserWarning, match="may be stale"):
        entries = fetch_tles("gps-ops", tmp_path, max_age_hours=1.0)

    assert entries[0].name == "GPS BIIR-5  (PRN 22)"


def test_fetch_without_cache_or_network_raises_clearly(tmp_path, monkeypatch):
    """Offline with no cache: a message naming the path to drop a file at."""

    def offline(url, timeout=None):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(urllib.request, "urlopen", offline)

    with pytest.raises(RuntimeError, match="no cached copy exists"):
        fetch_tles("gps-ops", tmp_path / "empty")

    # The message points at the exact file the user should create.
    with pytest.raises(RuntimeError, match=r"gps-ops\.tle"):
        fetch_tles("gps-ops", tmp_path / "empty")


@pytest.mark.parametrize(
    "group",
    ["../../etc/passwd", "gps-ops/../..", "a b", "gps ops", "", "gps.ops", "gps%2Fops"],
)
def test_fetch_rejects_unsafe_group_names(tmp_path, no_network, group):
    """A group name reaches the filesystem, so anything path-like is refused.

    ``group="../../x"`` previously escaped `cache_dir` for both the read
    and the write. Validation happens before any path or URL is built, so
    the network fixture proves nothing was attempted either.
    """
    with pytest.raises(ValueError, match="invalid catalogue group name"):
        fetch_tles(group, tmp_path)

    # Nothing was created outside (or inside) the cache directory.
    assert not list(tmp_path.rglob("*.tle"))


def test_fetch_accepts_ordinary_group_names(tmp_path, no_network):
    """The validator does not reject the names people actually use."""
    for group in ("gps-ops", "starlink", "galileo", "active", "last_30_days"):
        write_cache(tmp_path, group, CATALOGUE_TEXT, age_hours=0.0)
        assert fetch_tles(group, tmp_path)[0].name == "GPS BIIR-5  (PRN 22)"
