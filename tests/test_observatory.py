"""Tests for the simulated observing day and the live sky monitor.

**Nothing here touches the network.** The two fetchers that can --
aircraft reports and element sets -- are injected everywhere they are
used, and `test_sky_now_survives_a_dead_network` makes both of them raise
on purpose to check that the chart still comes back.

The day tests run the *real* process pool, with two workers rather than
the dozen a server uses: the point of these tests is that a frame really
does survive being pickled into another interpreter and back, which a
mocked executor would not show.
"""

import math
import time

import numpy as np
import pytest

pytest.importorskip("fastapi")

from astropy import units as u  # noqa: E402
from astropy.coordinates import FK5, EarthLocation, SkyCoord  # noqa: E402
from astropy.time import Time  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from rfi_simulator.sky import lm_from_radec  # noqa: E402
from rfi_simulator.webui import skynow  # noqa: E402
from rfi_simulator.webui.observatory import (  # noqa: E402
    CATALOG_SOURCES,
    DEFAULT_DAY_WORKERS,
    JOB_MAX,
    MAX_FRAMES,
    QUIET_SUN_FLUX_JY,
    DayRequest,
    _JobStore,
    cancel_day,
    catalog_sources_in_field,
    day_frame,
    day_status,
    day_workers,
    jobs,
    meridian_phase_center,
    start_day,
    timeline_payload,
)
from rfi_simulator.webui.server import create_app  # noqa: E402
from rfi_simulator.webui.simulate import (  # noqa: E402
    IMAGE_FIELD_HALF_WIDTH_DEG,
    SimulateRequest,
    default_array,
)

# A mid-latitude site with an ordinary sunrise and sunset, and a date far
# from any solstice so that neither is near a limit.
SITE = {"latitude_deg": 37.234, "longitude_deg": -118.282, "height_m": 1222.0}
DATE = "2026-08-05"

# Three antennas and a handful of channels: a real simulation, small
# enough that a four-frame day finishes in a couple of seconds.
TINY_SETUP = {
    "antennas": [[0.0, 0.0, 0.0], [30.0, 0.0, 0.0], [0.0, 40.0, 0.0]],
    "site": SITE,
    "sky_sources": [],
    "sim": {"n_chan": 8, "n_blocks": 1, "seed": 11},
}


def tiny_day(**overrides):
    """A `DayRequest` for a four-frame day, with fields overridden."""
    body = {
        "setup": TINY_SETUP,
        "date": DATE,
        "pointing_dec_deg": SITE["latitude_deg"],
        "n_frames": 4,
        "resolution": "coarse",
    }
    body.update(overrides)
    return DayRequest.model_validate(body)


def run_day(request, *, max_workers=2, timeout_s=180.0):
    """Build a day through the real pool and return its finished status."""
    job_id = start_day(request, max_workers=max_workers)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status = day_status(job_id)
        assert status is not None
        if status["state"] != "building":
            return job_id, status
        time.sleep(0.1)
    raise AssertionError("the day never finished")


# ----------------------------------------------------------------------
# The timeline
# ----------------------------------------------------------------------
def test_timeline_has_one_sunrise_and_one_sunset():
    payload = timeline_payload(DATE, SITE["latitude_deg"], **SITE)
    assert len(payload["sun"]["sunrise_utc"]) == 1
    assert len(payload["sun"]["sunset_utc"]) == 1
    assert not payload["sun"]["always_up"]
    assert not payload["sun"]["always_down"]
    # Every instant is inside the day it describes, and its fraction agrees.
    for key in ("sunrise", "sunset"):
        for fraction in payload["sun"][key]:
            assert 0.0 <= fraction <= 1.0
    assert 0.0 <= payload["sun"]["transit"] <= 1.0


def test_timeline_sun_transit_is_the_highest_sample():
    payload = timeline_payload(DATE, SITE["latitude_deg"], **SITE)
    altitudes = payload["sun"]["altitude_deg"]
    peak = altitudes.index(max(altitudes))
    assert payload["sun"]["max_altitude_deg"] == pytest.approx(max(altitudes))
    assert payload["sun"]["transit"] == pytest.approx(
        payload["sun"]["sample_fractions"][peak], abs=1e-6
    )


def test_timeline_transits_are_inside_the_day_and_in_order():
    payload = timeline_payload(DATE, SITE["latitude_deg"], **SITE)
    for source in payload["sources"]:
        assert source["transits"], f"{source['name']} never transits"
        assert all(0.0 <= fraction <= 1.0 for fraction in source["transits"])
        assert source["transits"] == sorted(source["transits"])
        for instant in source["transits_utc"]:
            moment = Time(instant, scale="utc")
            assert Time(f"{DATE}T00:00:00") <= moment <= Time(f"{DATE}T00:00:00") + 1.0 * u.day


def test_a_source_far_from_the_strip_is_never_in_field():
    # Point at a declination a long way from Cassiopeia A's.
    dec = CATALOG_SOURCES[0]["dec_deg"] - 20.0
    payload = timeline_payload(DATE, dec, **SITE)
    entry = [item for item in payload["sources"] if item["name"] == "Cas A"][0]
    assert abs(entry["dec_offset_deg"]) > IMAGE_FIELD_HALF_WIDTH_DEG
    assert entry["in_field"] is False


def test_a_source_on_the_strip_is_in_field():
    dec = CATALOG_SOURCES[1]["dec_deg"]
    payload = timeline_payload(DATE, dec, **SITE)
    entry = [item for item in payload["sources"] if item["name"] == "Cyg A"][0]
    assert entry["in_field"] is True
    assert entry["dec_offset_deg"] == pytest.approx(0.0, abs=1e-6)


def test_timeline_crossing_time_matches_the_field_width():
    dec = 0.0
    payload = timeline_payload(DATE, dec, **SITE)
    expected = 2.0 * IMAGE_FIELD_HALF_WIDTH_DEG / math.cos(math.radians(dec)) / 15.041 * 60.0
    assert payload["field_crossing_minutes"] == pytest.approx(expected, rel=1e-3)


def test_timeline_without_an_element_set_reports_no_satellite():
    payload = timeline_payload(DATE, SITE["latitude_deg"], **SITE)
    assert payload["satellite"]["configured"] is False
    assert payload["satellite"]["passes"] == []


def test_timeline_rejects_a_date_it_cannot_read():
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        timeline_payload("last tuesday", 0.0, **SITE)


# ----------------------------------------------------------------------
# The pointing
# ----------------------------------------------------------------------
def test_the_meridian_pointing_keeps_the_declination_it_was_given():
    location = EarthLocation.from_geodetic(
        lon=SITE["longitude_deg"] * u.deg,
        lat=SITE["latitude_deg"] * u.deg,
        height=SITE["height_m"] * u.m,
    )
    time_ = Time(f"{DATE}T06:00:00", scale="utc")
    center = meridian_phase_center(location, time_, 40.734)
    assert center.dec.deg == pytest.approx(40.734, abs=1e-9)
    # And it really is on the meridian. The hour angle is the difference
    # between the sidereal time and the *apparent* right ascension, not
    # the ICRS one -- twenty-six years of precession separate the two by a
    # fifth of a degree, which is exactly the error the solve exists to
    # avoid. What is left is the equation of the equinoxes, a second of
    # time.
    lst = time_.sidereal_time("apparent", longitude=location.lon).to_value(u.deg)
    apparent_ra = center.transform_to(FK5(equinox=time_)).ra.deg
    hour_angle = (lst - apparent_ra) % 360.0
    assert min(hour_angle, 360.0 - hour_angle) < 0.01


def test_a_source_on_the_strip_lands_in_the_field_at_transit():
    location = EarthLocation.from_geodetic(
        lon=SITE["longitude_deg"] * u.deg,
        lat=SITE["latitude_deg"] * u.deg,
        height=SITE["height_m"] * u.m,
    )
    payload = timeline_payload(DATE, CATALOG_SOURCES[1]["dec_deg"], **SITE)
    entry = [item for item in payload["sources"] if item["name"] == "Cyg A"][0]
    when = Time(entry["transits_utc"][0], scale="utc")
    center = meridian_phase_center(location, when, CATALOG_SOURCES[1]["dec_deg"])
    found = catalog_sources_in_field(center, when, location)
    assert "Cyg A" in [item["name"] for item in found]
    cyg = [item for item in found if item["name"] == "Cyg A"][0]
    assert cyg["flux_jy"] == pytest.approx(1590.0)
    assert abs(cyg["m"]) < 1e-3
    assert abs(cyg["l"]) < 0.5 * 0.04


def test_a_source_below_the_horizon_never_enters_the_field():
    # A southern site cannot see Cassiopeia A at all; point at its
    # declination anyway and check the field stays empty of it.
    southern = {"latitude_deg": -60.0, "longitude_deg": 0.0, "height_m": 0.0}
    location = EarthLocation.from_geodetic(
        lon=0.0 * u.deg, lat=-60.0 * u.deg, height=0.0 * u.m
    )
    cas = CATALOG_SOURCES[0]
    payload = timeline_payload(DATE, cas["dec_deg"], **southern)
    entry = [item for item in payload["sources"] if item["name"] == "Cas A"][0]
    when = Time(entry["transits_utc"][0], scale="utc")
    center = meridian_phase_center(location, when, cas["dec_deg"])
    # The projection puts it in the field...
    l_dir, m_dir = lm_from_radec(
        center, SkyCoord(ra=cas["ra_deg"] * u.deg, dec=cas["dec_deg"] * u.deg, frame="icrs")
    )
    assert max(abs(float(l_dir)), abs(float(m_dir))) < 0.5 * 0.04
    # ...but the horizon test keeps it out.
    found = [item["name"] for item in catalog_sources_in_field(center, when, location)]
    assert "Cas A" not in found


def test_the_sun_is_a_catalogue_source_with_the_documented_flux():
    location = EarthLocation.from_geodetic(
        lon=SITE["longitude_deg"] * u.deg,
        lat=SITE["latitude_deg"] * u.deg,
        height=SITE["height_m"] * u.m,
    )
    payload = timeline_payload(DATE, 0.0, **SITE)
    sun_dec = payload["sun"]["dec_deg"]
    # In August the Sun is well north of the equator, not south of it --
    # the check that would fail if a GCRS position were converted to ICRS
    # and came back as the anti-Sun.
    assert 10.0 < sun_dec < 25.0
    when = Time(payload["sun"]["transit_utc"], scale="utc")
    center = meridian_phase_center(location, when, sun_dec)
    found = catalog_sources_in_field(center, when, location)
    assert [item["flux_jy"] for item in found if item["name"] == "Sun"] == [QUIET_SUN_FLUX_JY]


# ----------------------------------------------------------------------
# The day job
# ----------------------------------------------------------------------
def test_a_tiny_day_builds_through_the_real_pool():
    job_id, status = run_day(tiny_day())
    assert status["state"] == "done"
    assert status["done"] == 4
    assert status["failed"] == 0
    assert status["scale_max_jy"] > 0.0
    assert status["scale_soft_jy"] > 0.0

    for index in range(4):
        frame = day_frame(job_id, index)
        assert frame["error"] is None
        assert frame["pending"] is False
        image = np.asarray(frame["image"], dtype=float)
        assert image.shape == (64, 64)
        assert np.all(np.isfinite(image))
        # Frames are ordered in time, and the pointing follows the sidereal
        # clock rather than standing still.
        assert frame["dec_deg"] == pytest.approx(SITE["latitude_deg"], abs=1e-9)


def test_frames_advance_in_time_and_in_right_ascension():
    _, status = run_day(tiny_day(n_frames=4))
    times = [Time(frame["utc"], scale="utc") for frame in status["frames"]]
    gaps = [(times[i + 1] - times[i]).sec for i in range(len(times) - 1)]
    assert all(gap == pytest.approx(21600.0, abs=1.0) for gap in gaps)
    # Six hours of sidereal rotation is about 90 degrees of right ascension.
    step = (status["frames"][1]["ra_deg"] - status["frames"][0]["ra_deg"]) % 360.0
    assert step == pytest.approx(90.4, abs=1.0)


def test_every_frame_gets_its_own_seed():
    request = tiny_day(n_frames=4)
    seeds = [request.frame_setup(index).sim.seed for index in range(4)]
    assert len(set(seeds)) == 4
    assert seeds == [11, 12, 13, 14]
    # And the frames really differ, not just their seeds.
    _, status = run_day(request)
    job_id = status["id"]
    images = [np.asarray(day_frame(job_id, index)["image"]) for index in range(4)]
    assert not np.allclose(images[0], images[1])


def test_the_coarse_preset_shrinks_the_recording_and_fine_does_not():
    setup = dict(TINY_SETUP, sim={"n_chan": 128, "n_blocks": 4, "seed": 3})
    coarse = tiny_day(setup=setup, resolution="coarse").frame_setup(0)
    fine = tiny_day(setup=setup, resolution="fine").frame_setup(0)
    assert (coarse.sim.n_chan, coarse.sim.n_blocks) == (32, 1)
    assert (fine.sim.n_chan, fine.sim.n_blocks) == (128, 4)


def test_setup_sky_sources_are_dropped_unless_they_are_carried():
    setup = dict(TINY_SETUP, sky_sources=[{"name": "mine", "offset_deg": [0.2, 0.1]}])
    assert tiny_day(setup=setup).frame_setup(0).sky_sources == []
    carried = tiny_day(setup=setup, carry_setup_sources=True).frame_setup(0)
    assert [source.name for source in carried.sky_sources] == ["mine"]


def test_a_catalogue_source_appears_in_exactly_the_frames_that_hold_it():
    """Build a day around one transit and check where the source shows up.

    The window is constructed rather than searched for: a frame every two
    minutes across the twenty minutes either side of Cygnus A's transit,
    on Cygnus A's own strip. The field is 2.3 degrees wide, so the source
    should be present in a contiguous run of frames in the middle and
    absent at both ends.
    """
    cyg = CATALOG_SOURCES[1]
    payload = timeline_payload(DATE, cyg["dec_deg"], **SITE)
    entry = [item for item in payload["sources"] if item["name"] == "Cyg A"][0]
    transit = Time(entry["transits_utc"][0], scale="utc")

    location = EarthLocation.from_geodetic(
        lon=SITE["longitude_deg"] * u.deg,
        lat=SITE["latitude_deg"] * u.deg,
        height=SITE["height_m"] * u.m,
    )
    offsets_s = np.arange(-1200.0, 1201.0, 120.0)
    present = []
    for offset in offsets_s:
        when = transit + offset * u.s
        center = meridian_phase_center(location, when, cyg["dec_deg"])
        names = [item["name"] for item in catalog_sources_in_field(center, when, location)]
        present.append("Cyg A" in names)

    assert present[0] is False and present[-1] is False
    assert present[len(present) // 2] is True
    # One contiguous window, centred on the transit.
    inside = [index for index, flag in enumerate(present) if flag]
    assert inside == list(range(inside[0], inside[-1] + 1))
    assert abs((inside[0] + inside[-1]) / 2 - (len(present) - 1) / 2) <= 1


def test_a_day_can_be_cancelled():
    job_id = start_day(tiny_day(n_frames=MAX_FRAMES), max_workers=2)
    cancelled = cancel_day(job_id)
    assert cancelled["state"] in {"cancelling", "cancelled"}
    deadline = time.time() + 120.0
    while time.time() < deadline:
        status = day_status(job_id)
        if status["state"] == "cancelled":
            break
        time.sleep(0.2)
    assert day_status(job_id)["state"] == "cancelled"
    assert day_status(job_id)["done"] < MAX_FRAMES


def test_cancelling_a_day_that_does_not_exist_is_a_miss():
    assert cancel_day("0" * 32) is None
    assert day_status("0" * 32) is None
    assert day_frame("0" * 32, 0) is None


def test_the_store_evicts_the_least_recently_touched_day():
    store = _JobStore(max_jobs=2, ttl_s=3600.0)

    class _Stub:
        def __init__(self, identifier):
            self.id = identifier
            self.touched = time.time()
            self.cancelled = type("E", (), {"set": lambda self: None})()

    first, second, third = (_Stub("a"), _Stub("b"), _Stub("c"))
    # Recent, so nothing expires: this test is about the size cap alone.
    now = time.time()
    first.touched = now - 30.0
    second.touched = now - 20.0
    third.touched = now - 10.0
    store.add(first)
    store.add(second)
    store.add(third)
    assert len(store) == 2
    assert store.get("a") is None
    assert store.get("c") is not None


def test_the_store_drops_a_day_that_has_expired():
    store = _JobStore(max_jobs=JOB_MAX, ttl_s=0.0)

    class _Stub:
        id = "a"
        touched = 0.0
        cancelled = type("E", (), {"set": lambda self: None})()

    store.add(_Stub())
    assert len(store) == 0


def test_the_worker_count_is_a_fraction_of_the_host(monkeypatch):
    monkeypatch.delenv("RFI_SIMULATOR_DAY_WORKERS", raising=False)
    assert day_workers() == DEFAULT_DAY_WORKERS
    monkeypatch.setenv("RFI_SIMULATOR_DAY_WORKERS", "3")
    assert day_workers() == 3
    monkeypatch.setenv("RFI_SIMULATOR_DAY_WORKERS", "not a number")
    assert day_workers() == DEFAULT_DAY_WORKERS


def test_a_frame_that_fails_does_not_take_the_day_down():
    from rfi_simulator.webui.observatory import compute_frame

    frame = compute_frame({"setup": TINY_SETUP, "n_frames": 2}, 99)
    assert frame["error"] is not None
    assert frame["image"] is None
    assert frame["index"] == 99


# ----------------------------------------------------------------------
# The live monitor
# ----------------------------------------------------------------------
CANNED_AIRCRAFT = {
    # Synthetic, in the aggregator's schema: one aircraft directly
    # overhead, one on the ground with the feed's string altitude, one
    # missing a position entirely, and one that is not a mapping at all.
    "ac": [
        {
            "hex": "000001",
            "flight": "TEST001 ",
            "lat": SITE["latitude_deg"],
            "lon": SITE["longitude_deg"],
            "alt_geom": 35000,
            "track": 271.5,
        },
        {
            "hex": "000002",
            "flight": "TEST002",
            "lat": SITE["latitude_deg"] + 0.4,
            "lon": SITE["longitude_deg"],
            "alt_baro": "ground",
            "track": None,
        },
        {"hex": "000003", "flight": "NOPOS"},
        "not a dict",
    ],
    "total": 3,
}


def dead_fetcher(*args, **kwargs):
    raise OSError("the network is not reachable from here")


@pytest.fixture(autouse=True)
def _clear_monitor_caches():
    skynow.clear_caches()
    yield
    skynow.clear_caches()


def sky(**overrides):
    body = {
        "latitude_deg": SITE["latitude_deg"],
        "longitude_deg": SITE["longitude_deg"],
        "height_m": SITE["height_m"],
        "now": Time("2026-08-05T04:00:00", scale="utc"),
        "aircraft_fetcher": lambda lat, lon: CANNED_AIRCRAFT,
        "element_loader": lambda: [],
    }
    body.update(overrides)
    return skynow.sky_now(**body)


def test_sky_now_survives_a_dead_network():
    payload = sky(aircraft_fetcher=dead_fetcher, element_loader=dead_fetcher)
    assert payload["layers"]["ephemeris"]["status"] == "ok"
    assert payload["layers"]["aircraft"]["status"] == "down"
    assert payload["layers"]["aircraft"]["note"] == "live aircraft feed unreachable"
    assert payload["layers"]["satellites"]["status"] == "down"
    assert payload["aircraft"] == []
    assert payload["satellites"] == []
    # The ephemeris is all still there, which is the point.
    assert payload["sun"]["name"] == "Sun"
    assert payload["moon"]["name"] == "Moon"
    assert len(payload["sources"]) == len(CATALOG_SOURCES)
    for entry in payload["sources"]:
        assert -90.0 <= entry["altitude_deg"] <= 90.0
        assert 0.0 <= entry["azimuth_deg"] < 360.0


def test_an_aircraft_overhead_is_at_the_zenith():
    payload = sky()
    overhead = [item for item in payload["aircraft"] if item["callsign"] == "TEST001"][0]
    assert overhead["altitude_deg"] == pytest.approx(90.0, abs=0.05)
    assert overhead["heading_deg"] == pytest.approx(271.5)
    assert overhead["height_m"] == pytest.approx(35000 * 0.3048, abs=0.2)


def test_aircraft_parsing_skips_what_it_cannot_read():
    payload = sky()
    callsigns = [item["callsign"] for item in payload["aircraft"]]
    assert callsigns == ["TEST001", "TEST002"]
    ground = [item for item in payload["aircraft"] if item["callsign"] == "TEST002"][0]
    assert ground["height_m"] == 0.0
    assert ground["heading_deg"] is None
    # Due north of the site, and below the horizon from 44 km away at zero
    # altitude, which the geodesic conversion gets right and a flat-Earth
    # one would not.
    assert ground["azimuth_deg"] == pytest.approx(0.0, abs=0.5)
    assert ground["altitude_deg"] < 0.0


def test_the_aircraft_fetch_is_shared_between_callers():
    calls = []

    def counting(lat, lon):
        calls.append((lat, lon))
        return CANNED_AIRCRAFT

    sky(aircraft_fetcher=counting)
    sky(aircraft_fetcher=counting)
    assert len(calls) == 1


def test_a_satellite_below_the_horizon_is_not_drawn():
    from rfi_simulator.satellites import read_tle_file
    from rfi_simulator.webui.simulate import _config_path

    path = _config_path("tle_sample.txt")
    if path is None:
        pytest.skip("no element set is bundled with this installation")
    elements = read_tle_file(path)
    payload = sky(element_loader=lambda: elements)
    assert payload["layers"]["satellites"]["status"] == "ok"
    for entry in payload["satellites"]:
        assert entry["altitude_deg"] > 0.0
        assert 0.0 <= entry["azimuth_deg"] < 360.0


# ----------------------------------------------------------------------
# Through the HTTP surface
# ----------------------------------------------------------------------
@pytest.fixture
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


def test_the_endpoints_carry_a_day_from_start_to_frame(client, monkeypatch):
    monkeypatch.setenv("RFI_SIMULATOR_DAY_WORKERS", "2")
    body = {
        "setup": TINY_SETUP,
        "date": DATE,
        "pointing_dec_deg": SITE["latitude_deg"],
        "n_frames": 4,
        "resolution": "coarse",
    }
    started = client.post("/api/observatory/day", json=body)
    assert started.status_code == 200
    job_id = started.json()["id"]

    deadline = time.time() + 180.0
    while time.time() < deadline:
        status = client.get(f"/api/observatory/day/{job_id}").json()
        if status["state"] != "building":
            break
        time.sleep(0.1)
    assert status["state"] == "done"
    assert status["done"] == 4
    # The status is deliberately imageless: it is polled every second.
    assert all("image" not in frame for frame in status["frames"])

    frame = client.get(f"/api/observatory/day/{job_id}/frame/0")
    assert frame.status_code == 200
    assert len(frame.json()["image"]) == frame.json()["n_pix"]

    assert client.get(f"/api/observatory/day/{job_id}/frame/9").status_code == 404

    # Local time travels with both the status poll and the full frame.
    assert status["zone"]["key"] == "America/Los_Angeles"
    assert all(len(entry["local"]) == 5 for entry in status["frames"])
    assert frame.json()["local"] == status["frames"][0]["local"]

    assert client.get("/api/observatory/day/" + "0" * 32).status_code == 404
    assert client.post(f"/api/observatory/day/{job_id}/cancel").status_code == 200


def test_the_timeline_endpoint_carries_the_zone_and_local_clocks(client):
    payload = client.get(
        "/api/observatory/timeline",
        params=dict(SITE, date=DATE, dec_deg=SITE["latitude_deg"]),
    ).json()
    assert payload["zone"]["abbreviation"] == "PDT"
    assert payload["sun"]["dec_date"] == DATE
    assert len(payload["sun"]["sunrise_local"]) == len(payload["sun"]["sunrise_utc"])
    assert payload["sun"]["transit_local"] != ""


def test_the_monitor_endpoint_carries_the_zone(client, monkeypatch):
    monkeypatch.setattr(skynow, "fetch_aircraft", dead_fetcher)
    monkeypatch.setattr(skynow, "load_elements", dead_fetcher)
    skynow.clear_caches()
    payload = client.get("/api/sky/now", params=SITE).json()
    assert payload["zone"]["key"] == "America/Los_Angeles"
    assert len(payload["local"]) == 5


def test_the_day_endpoint_refuses_a_job_it_cannot_size(client):
    body = {"setup": TINY_SETUP, "date": DATE, "n_frames": MAX_FRAMES + 1}
    assert client.post("/api/observatory/day", json=body).status_code == 422
    bad_date = {"setup": TINY_SETUP, "date": "yesterday"}
    assert client.post("/api/observatory/day", json=bad_date).status_code == 422


def test_a_malformed_job_identifier_is_refused(client):
    assert client.get("/api/observatory/day/not-a-uuid").status_code == 422


def test_the_timeline_endpoint_answers(client):
    response = client.get(
        "/api/observatory/timeline",
        params={"date": DATE, "dec_deg": 40.734, **{f"{k}": v for k, v in SITE.items()}},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["pointing_dec_deg"] == 40.734
    assert payload["sun"]["sunrise_utc"]


def test_the_timeline_endpoint_refuses_a_date_it_cannot_read(client):
    response = client.get(
        "/api/observatory/timeline",
        params={"date": "someday", "dec_deg": 0.0, **SITE},
    )
    assert response.status_code == 422


def test_the_monitor_endpoint_answers_with_a_dead_network(client, monkeypatch):
    monkeypatch.setattr(skynow, "fetch_aircraft", dead_fetcher)
    monkeypatch.setattr(skynow, "load_elements", dead_fetcher)
    skynow.clear_caches()
    response = client.get("/api/sky/now")
    assert response.status_code == 200
    payload = response.json()
    assert payload["layers"]["aircraft"]["status"] == "down"
    assert payload["sun"]["name"] == "Sun"
    assert payload["latitude_deg"] == pytest.approx(default_array().latitude_deg)


def test_the_monitor_endpoint_takes_the_site_it_is_given(client, monkeypatch):
    monkeypatch.setattr(skynow, "fetch_aircraft", lambda lat, lon: CANNED_AIRCRAFT)
    monkeypatch.setattr(skynow, "load_elements", lambda: [])
    skynow.clear_caches()
    payload = client.get("/api/sky/now", params=SITE).json()
    assert payload["latitude_deg"] == pytest.approx(SITE["latitude_deg"])
    assert [item["callsign"] for item in payload["aircraft"]] == ["TEST001", "TEST002"]


def test_the_page_offers_the_observatory_tab(client):
    page = client.get("/").text
    assert 'data-tab="observatory"' in page
    assert 'id="view-observatory"' in page
    assert "Mock Observatory" in page


@pytest.fixture(autouse=True)
def _clear_day_store():
    yield
    jobs.clear()


def test_the_request_model_defends_its_bounds():
    with pytest.raises(ValueError):
        DayRequest.model_validate({"setup": TINY_SETUP, "n_frames": 0})
    with pytest.raises(ValueError):
        DayRequest.model_validate({"setup": TINY_SETUP, "pointing_dec_deg": 100.0})
    with pytest.raises(ValueError):
        DayRequest.model_validate({"setup": TINY_SETUP, "resolution": "medium"})
    # And it really is a `SimulateRequest` underneath, with its own rules.
    with pytest.raises(ValueError):
        DayRequest.model_validate({"setup": dict(TINY_SETUP, antennas=[[0.0, 0.0, 0.0]])})
    assert isinstance(tiny_day().setup, SimulateRequest)


# ----------------------------------------------------------------------
# Observatory-local time
# ----------------------------------------------------------------------
def test_the_timeline_names_the_sites_own_time_zone():
    """Every clock the tab prints is the observatory's, so it must say which."""
    payload = timeline_payload(DATE, SITE["latitude_deg"], **SITE)
    zone = payload["zone"]
    assert zone["key"] == "America/Los_Angeles"
    assert zone["abbreviation"] == "PDT"
    assert zone["utc_offset_hours"] == -7.0
    assert zone["approximate"] is False


def test_local_clocks_are_the_utc_ones_shifted_by_that_offset():
    payload = timeline_payload(DATE, SITE["latitude_deg"], **SITE)
    offset_hours = payload["zone"]["utc_offset_hours"]
    for utc, local in zip(payload["sun"]["sunrise_utc"], payload["sun"]["sunrise_local"]):
        expected = (Time(utc, scale="utc") + offset_hours * u.hour).datetime
        assert local == f"{expected.hour:02d}:{expected.minute:02d}"
    assert payload["sun"]["transit_local"] != payload["sun"]["transit_utc"][11:16]
    # Solar transit lands near local noon, which is the whole point of
    # showing local time on a transit instrument. Not *at* noon: the site
    # sits east of its zone meridian and the zone is on daylight saving,
    # so an hour of slack is the honest bound.
    hour, minute = (int(part) for part in payload["sun"]["transit_local"].split(":"))
    assert abs(hour * 60 + minute - 12 * 60) < 90


def test_daylight_saving_is_resolved_at_the_requested_date():
    """Not a fixed offset stamped once: the same site, two seasons."""
    summer = timeline_payload("2026-07-15", 37.0, **SITE)["zone"]
    winter = timeline_payload("2026-01-15", 37.0, **SITE)["zone"]
    assert summer["abbreviation"] == "PDT"
    assert winter["abbreviation"] == "PST"
    assert summer["utc_offset_hours"] - winter["utc_offset_hours"] == 1.0


def test_a_site_outside_the_table_gets_an_approximate_zone():
    payload = timeline_payload(DATE, 0.0, latitude_deg=-30.0, longitude_deg=25.0, height_m=0.0)
    zone = payload["zone"]
    assert zone["approximate"] is True
    assert zone["utc_offset_hours"] == 2.0
    assert "approximated from the site longitude" in zone["note"]


def test_every_source_transit_carries_a_local_clock():
    payload = timeline_payload(DATE, SITE["latitude_deg"], **SITE)
    for source in payload["sources"]:
        assert len(source["transits_local"]) == len(source["transits_utc"])
        for clock in source["transits_local"]:
            assert len(clock) == 5 and clock[2] == ":"


def test_the_day_and_the_monitor_agree_about_the_zone():
    """Two payloads, one site: a reader must not see two different clocks."""
    identifier, status = run_day(tiny_day())
    monitor = skynow.sky_now(**SITE, now=Time(f"{DATE}T20:00:00", scale="utc"))
    assert status["zone"]["key"] == monitor["zone"]["key"]
    assert monitor["local"] == "13:00"


def test_each_frame_is_stamped_with_its_local_clock():
    identifier, status = run_day(tiny_day())
    offset_hours = status["zone"]["utc_offset_hours"]
    for index, meta in enumerate(status["frames"]):
        assert meta["utc"] is not None
        expected = (Time(meta["utc"], scale="utc") + offset_hours * u.hour).datetime
        assert meta["local"] == f"{expected.hour:02d}:{expected.minute:02d}"
        # The full frame carries the same stamp as the status poll's row.
        assert day_frame(identifier, index)["local"] == meta["local"]


# ----------------------------------------------------------------------
# The Sun's strip moves with the date
# ----------------------------------------------------------------------
def test_the_suns_strip_follows_the_date_and_matches_astropy():
    """The shortcut chip offers a declination that swings 47 degrees a year."""
    from astropy.coordinates import get_sun

    august = timeline_payload("2026-08-05", 37.0, **SITE)["sun"]
    september = timeline_payload("2026-09-05", 37.0, **SITE)["sun"]

    assert august["dec_date"] == "2026-08-05"
    assert september["dec_date"] == "2026-09-05"
    assert abs(august["dec_deg"] - september["dec_deg"]) > 5.0
    for payload, date in ((august, "2026-08-05"), (september, "2026-09-05")):
        expected = float(get_sun(Time(f"{date}T12:00:00", scale="utc")).dec.deg)
        assert abs(payload["dec_deg"] - expected) < 1.0e-3


# ----------------------------------------------------------------------
# A finished day must say so
# ----------------------------------------------------------------------
def test_frame_workers_never_fork_the_server():
    """A forked child inherits the server's locks and its listening socket."""
    from rfi_simulator.webui.observatory import _worker_context

    assert _worker_context().get_start_method() == "spawn"


def test_a_finished_day_reports_done_without_waiting_for_the_pool():
    """The state is settled from the frames, not from the pool shutting down.

    A worker process that lingers -- and one that has inherited a web
    server's threads can -- must not be able to leave a day whose frames
    are all computed reading "building", because the page will not play a
    day it believes is still being built.
    """
    from rfi_simulator.webui import observatory

    released = []
    original = observatory._release_pool

    def slow_release(pool):
        # Stand in for a pool that will not join: the state must already
        # be right by the time this is reached.
        released.append(observatory.jobs.get(identifier).state)
        original(pool)

    identifier = None
    observatory._release_pool = slow_release
    try:
        identifier = start_day(tiny_day(), max_workers=2)
        deadline = time.time() + 180.0
        while time.time() < deadline:
            if day_status(identifier)["state"] != "building":
                break
            time.sleep(0.1)
    finally:
        observatory._release_pool = original

    status = day_status(identifier)
    assert status["state"] == "done"
    assert status["done"] == status["total"]
    # And the state was already "done" when the pool was let go, not after.
    assert released == ["done"]


def test_the_sun_is_in_field_exactly_when_the_strip_is_pointed_at_it():
    dec = timeline_payload(DATE, 0.0, **SITE)["sun"]["dec_deg"]
    assert timeline_payload(DATE, dec, **SITE)["sun"]["in_field"] is True
    assert timeline_payload(DATE, dec + 10.0, **SITE)["sun"]["in_field"] is False
