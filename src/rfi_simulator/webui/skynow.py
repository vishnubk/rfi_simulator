"""The sky over the site right now: ephemeris, satellites, aircraft.

Four layers, and they are not equally trustworthy, so each carries its own
status and the page says which is which.

* **Sun, Moon and the catalogue** are pure ephemeris. They are computed
  from `astropy` and cannot fail for a reachable reason, so they are
  always present.
* **Satellites** come from an element set the process already has -- the
  bundled sample, or a catalogue group cached on disk if
  `TLE_GROUP_ENV_VAR` names one. Elements age (see
  `rfi_simulator.satellites.MAX_TLE_AGE_DAYS`); the layer says how old
  the set it used is.
* **Aircraft** come from a public transponder aggregator over the
  network, which is the only thing here that can be down. It is fetched
  with a hard timeout and cached for a few seconds across every client,
  and when it fails the layer is simply absent with a status saying so.
  The chart still draws.

Degrading honestly is the whole design: a failed layer is reported as
failed, never as an empty sky.

Aircraft geometry
-----------------
An aircraft is given as geodetic latitude, longitude and barometric
altitude. It is converted to horizon coordinates properly rather than on
a flat Earth: the aircraft is an `~astropy.coordinates.EarthLocation` in
its own right, the vector from the site to it is taken in Earth-centred
Earth-fixed coordinates, and that vector is rotated into the site's
East-North-Up frame by the library's own
`rfi_simulator.rfi.enu_from_ecef_offset`. Over a hundred-kilometre radius
the flat-Earth shortcut would misplace a distant aircraft's altitude by
nearly a degree, which is most of a field of view; the proper conversion
costs nothing here because it is arithmetic, not an astropy frame
transform.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time as _time
import urllib.error
import urllib.request
from datetime import timezone as datetime_timezone
from typing import Any, Callable

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, get_sun
from astropy.time import Time

from rfi_simulator.rfi import enu_from_ecef_offset
from rfi_simulator.webui.localtime import local_clock, resolve_zone, zone_payload
from rfi_simulator.webui.observatory import (
    CATALOG_SOURCES,
    QUIET_SUN_FLUX_JY,
)

__all__ = [
    "AIRCRAFT_CACHE_S",
    "AIRCRAFT_RADIUS_NM",
    "AIRCRAFT_URL",
    "MAX_AIRCRAFT",
    "NETWORK_TIMEOUT_S",
    "TLE_CACHE_S",
    "TLE_GROUP_ENV_VAR",
    "fetch_aircraft",
    "load_elements",
    "sky_now",
]

AIRCRAFT_URL = "https://api.adsb.lol/v2/lat/{lat}/lon/{lon}/dist/{dist}"
"""str: Public aggregator of aircraft transponder reports.

Read-only, unauthenticated, and asked for at most once every
`AIRCRAFT_CACHE_S` however many browsers are watching."""

AIRCRAFT_RADIUS_NM = 100
"""int: Search radius, nautical miles -- the aggregator's own unit."""

NETWORK_TIMEOUT_S = 3.0
"""float: Hard timeout on anything this module fetches. A monitor that
refreshes every ten seconds must never block for longer than a refresh."""

AIRCRAFT_CACHE_S = 10.0
"""float: How long one aircraft fetch is shared for. Matches the page's
poll interval, so N browsers cost one request per interval, not N."""

TLE_CACHE_S = 6.0 * 3600.0
"""float: How long a parsed element set is reused for. Elements are
published a few times a day; propagating a six-hour-old set is well inside
what `rfi_simulator.satellites` considers trustworthy."""

TLE_GROUP_ENV_VAR = "RFI_SIMULATOR_SKY_TLE_GROUP"
"""str: Environment variable naming a catalogue group (``gps-ops``,
``starlink``, ...) to show instead of the bundled sample.

Unset by default, and that is deliberate: the default install shows the
one bundled element set and touches no network for it. Setting this opts
into a download, cached on disk by
`rfi_simulator.satellites.fetch_tles`."""

TLE_CACHE_DIR_ENV_VAR = "RFI_SIMULATOR_TLE_CACHE_DIR"

MAX_AIRCRAFT = 120
"""int: Most aircraft returned. A busy corridor can report several
hundred; beyond this the chart is a smear and the response is large, so
the nearest `MAX_AIRCRAFT` are kept."""

MAX_SATELLITES = 60
"""int: Most satellites returned, nearest first by zenith angle."""

FEET_TO_M = 0.3048


# ----------------------------------------------------------------------
# Fetchers -- everything that can touch the network, in one place
# ----------------------------------------------------------------------
def fetch_aircraft(latitude_deg: float, longitude_deg: float) -> dict[str, Any]:
    """One live transponder query. Raises on any network trouble."""
    url = AIRCRAFT_URL.format(
        lat=f"{float(latitude_deg):.4f}",
        lon=f"{float(longitude_deg):.4f}",
        dist=AIRCRAFT_RADIUS_NM,
    )
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=NETWORK_TIMEOUT_S) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def load_elements() -> list[Any]:
    """Element sets to propagate: the bundle, or a catalogue group.

    Raises rather than returning an empty list, so the caller can put the
    reason in the layer's status.
    """
    from rfi_simulator.satellites import fetch_tles, read_tle_file
    from rfi_simulator.webui.simulate import _config_path

    group = os.environ.get(TLE_GROUP_ENV_VAR, "").strip()
    if group:
        cache_dir = os.environ.get(TLE_CACHE_DIR_ENV_VAR) or "~/.cache/rfi-simulator-tles"
        return fetch_tles(group, cache_dir, timeout_s=NETWORK_TIMEOUT_S)

    path = _config_path("tle_sample.txt")
    if path is None:
        raise RuntimeError("no element sets are bundled with this installation")
    return read_tle_file(path)


class _TimedCache:
    """One value, shared between threads, refetched when it goes stale.

    A failure is cached too, for a fraction of the success lifetime: an
    unreachable service must not be retried once per client per poll, but
    it must come back quickly once it recovers.
    """

    FAILURE_FRACTION = 0.5

    def __init__(self, lifetime_s: float) -> None:
        self.lifetime_s = lifetime_s
        self._lock = threading.Lock()
        self._key: Any = None
        self._value: Any = None
        self._error: str | None = None
        self._stamp = 0.0

    def get(self, key: Any, produce: Callable[[], Any]) -> tuple[Any, str | None, float]:
        """Return ``(value, error, age_s)``; at most one of value/error is set."""
        with self._lock:
            now = _time.time()
            age = now - self._stamp
            fresh_for = self.lifetime_s * (self.FAILURE_FRACTION if self._error else 1.0)
            if key == self._key and self._stamp and age < fresh_for:
                return self._value, self._error, age
            try:
                self._value, self._error = produce(), None
            except Exception as exc:  # noqa: BLE001 - any failure is "layer down"
                self._value, self._error = None, f"{type(exc).__name__}: {exc}"
            self._key = key
            self._stamp = _time.time()
            return self._value, self._error, 0.0

    def clear(self) -> None:
        with self._lock:
            self._key = None
            self._value = None
            self._error = None
            self._stamp = 0.0


_aircraft_cache = _TimedCache(AIRCRAFT_CACHE_S)
_element_cache = _TimedCache(TLE_CACHE_S)


def clear_caches() -> None:
    """Forget both cached fetches. For tests, and for a site change."""
    _aircraft_cache.clear()
    _element_cache.clear()


# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------
def _enu_to_altaz(enu: np.ndarray) -> tuple[float, float, float]:
    """``(altitude_deg, azimuth_deg, range_m)`` of an ENU offset vector."""
    east, north, up = (float(value) for value in enu)
    distance = math.sqrt(east * east + north * north + up * up)
    if distance == 0.0:
        return 90.0, 0.0, 0.0
    altitude = math.degrees(math.asin(up / distance))
    # Rounded here rather than at every call site, and taken modulo again
    # afterwards: a bearing a whisker short of due north rounds up to
    # 360.0, which is the same direction written in a way no reader of a
    # compass expects.
    azimuth = round(math.degrees(math.atan2(east, north)) % 360.0, 6) % 360.0
    return altitude, azimuth, distance


def _aircraft_altaz(
    site: EarthLocation, latitude_deg: float, longitude_deg: float, height_m: float
) -> tuple[float, float, float]:
    """Horizon coordinates of a point given geodetically. See module docs."""
    other = EarthLocation.from_geodetic(
        lon=longitude_deg * u.deg, lat=latitude_deg * u.deg, height=height_m * u.m
    )
    delta = np.asarray(
        [
            (other.x - site.x).to_value(u.m),
            (other.y - site.y).to_value(u.m),
            (other.z - site.z).to_value(u.m),
        ],
        dtype=np.float64,
    )
    return _enu_to_altaz(enu_from_ecef_offset(delta, site))


def _parse_aircraft(payload: dict[str, Any], site: EarthLocation) -> list[dict[str, Any]]:
    """Turn one aggregator response into horizon coordinates.

    Defensive throughout: the feed is somebody else's schema, entries with
    no position or a non-numeric one are skipped rather than trusted, and
    ``"ground"`` -- which the feed uses instead of a number for an
    aircraft that is not flying -- becomes zero.
    """
    entries: list[dict[str, Any]] = []
    for item in payload.get("ac") or []:
        if not isinstance(item, dict):
            continue
        try:
            latitude = float(item["lat"])
            longitude = float(item["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(latitude) and math.isfinite(longitude)):
            continue
        altitude_ft = item.get("alt_geom", item.get("alt_baro"))
        if altitude_ft == "ground" or altitude_ft is None:
            altitude_m = 0.0
        else:
            try:
                altitude_m = float(altitude_ft) * FEET_TO_M
            except (TypeError, ValueError):
                altitude_m = 0.0
        altitude_deg, azimuth_deg, distance_m = _aircraft_altaz(
            site, latitude, longitude, altitude_m
        )
        heading = item.get("track")
        try:
            heading_deg = float(heading) if heading is not None else None
        except (TypeError, ValueError):
            heading_deg = None
        entries.append(
            {
                "id": str(item.get("hex") or "")[:12],
                "callsign": str(item.get("flight") or "").strip()[:12],
                "altitude_deg": round(altitude_deg, 3),
                "azimuth_deg": round(azimuth_deg, 3) % 360.0,
                "range_km": round(distance_m / 1000.0, 2),
                "height_m": round(altitude_m, 1),
                "heading_deg": None if heading_deg is None else round(heading_deg % 360.0, 1),
            }
        )
    entries.sort(key=lambda entry: entry["range_km"])
    return entries[:MAX_AIRCRAFT]


def _satellites(elements: list[Any], now: Time, site: EarthLocation) -> list[dict[str, Any]]:
    """Every element set in the loaded catalogue, above the horizon."""
    entries: list[dict[str, Any]] = []
    for element_set in elements:
        try:
            enu = element_set.enu_position_m(now, site)
        except Exception:  # noqa: BLE001 - a decayed orbit is one satellite's fault
            continue
        altitude_deg, azimuth_deg, distance_m = _enu_to_altaz(np.asarray(enu))
        if altitude_deg <= 0.0:
            continue
        entries.append(
            {
                "name": (element_set.name or "satellite")[:32],
                "altitude_deg": round(altitude_deg, 3),
                "azimuth_deg": round(azimuth_deg, 3) % 360.0,
                "range_km": round(distance_m / 1000.0, 1),
                "element_age_days": round(float((now - element_set.epoch).jd), 2),
            }
        )
    entries.sort(key=lambda entry: -entry["altitude_deg"])
    return entries[:MAX_SATELLITES]


# ----------------------------------------------------------------------
# The endpoint's payload
# ----------------------------------------------------------------------
def sky_now(
    latitude_deg: float,
    longitude_deg: float,
    height_m: float = 0.0,
    *,
    now: Time | None = None,
    aircraft_fetcher: Callable[[float, float], dict[str, Any]] | None = None,
    element_loader: Callable[[], list[Any]] | None = None,
) -> dict[str, Any]:
    """The whole chart, computed server-side.

    Parameters
    ----------
    latitude_deg, longitude_deg, height_m : float
        The site, which is the setup's site: changing it moves the chart.
    now : astropy.time.Time, optional
        The instant. Defaults to the real one; the tests pin it.
    aircraft_fetcher, element_loader : callable, optional
        The two things that touch the network, injected so that a test can
        make them fail and check that the chart still comes back. Defaults
        are `fetch_aircraft` and `load_elements`.

    Returns
    -------
    dict
        ``sun``, ``moon``, ``sources``, ``satellites`` and ``aircraft``,
        plus a ``layers`` map giving each one an ``ok``/``down`` status
        and a sentence the page can print verbatim.
    """
    when = Time.now() if now is None else now
    site = EarthLocation.from_geodetic(
        lon=float(longitude_deg) * u.deg,
        lat=float(latitude_deg) * u.deg,
        height=float(height_m) * u.m,
    )
    frame = AltAz(obstime=when, location=site)
    layers: dict[str, dict[str, Any]] = {}

    def horizon(coord: SkyCoord) -> dict[str, Any]:
        altaz = coord.transform_to(frame)
        altitude = float(altaz.alt.to_value(u.deg))
        return {
            "altitude_deg": round(altitude, 3),
            "azimuth_deg": round(float(altaz.az.to_value(u.deg)) % 360.0, 3) % 360.0,
            "up": altitude > 0.0,
        }

    sun_coord = get_sun(when)
    sun = horizon(sun_coord)
    sun.update({"name": "Sun", "flux_jy": QUIET_SUN_FLUX_JY})

    moon_coord = get_body("moon", when, site)
    moon = horizon(moon_coord)
    moon["name"] = "Moon"

    sources = []
    for item in CATALOG_SOURCES:
        coord = SkyCoord(ra=item["ra_deg"] * u.deg, dec=item["dec_deg"] * u.deg, frame="icrs")
        entry = horizon(coord)
        entry.update({"name": item["name"], "flux_jy": item["flux_jy"], "dec_deg": item["dec_deg"]})
        sources.append(entry)
    sources.sort(key=lambda entry: -entry["altitude_deg"])
    layers["ephemeris"] = {"status": "ok", "note": "computed here, no network involved"}

    # --- satellites ---------------------------------------------------
    loader = load_elements if element_loader is None else element_loader
    elements, element_error, element_age = _element_cache.get("elements", loader)
    if element_error is not None:
        satellites: list[dict[str, Any]] = []
        layers["satellites"] = {
            "status": "down",
            "note": "no element set available, so no satellites are drawn",
            "detail": element_error,
        }
    else:
        satellites = _satellites(list(elements or []), when, site)
        layers["satellites"] = {
            "status": "ok",
            "note": f"{len(satellites)} above the horizon",
            "age_s": round(element_age, 1),
        }

    # --- aircraft -----------------------------------------------------
    fetcher = fetch_aircraft if aircraft_fetcher is None else aircraft_fetcher
    key = (round(float(latitude_deg), 3), round(float(longitude_deg), 3))
    payload, aircraft_error, aircraft_age = _aircraft_cache.get(
        key, lambda: fetcher(float(latitude_deg), float(longitude_deg))
    )
    if aircraft_error is not None:
        aircraft: list[dict[str, Any]] = []
        layers["aircraft"] = {
            "status": "down",
            "note": "live aircraft feed unreachable",
            "detail": aircraft_error,
        }
    else:
        aircraft = _parse_aircraft(payload if isinstance(payload, dict) else {}, site)
        layers["aircraft"] = {
            "status": "ok",
            "note": f"{len(aircraft)} within {AIRCRAFT_RADIUS_NM} nautical miles",
            "age_s": round(aircraft_age, 1),
        }

    zone = resolve_zone(float(latitude_deg), float(longitude_deg))
    return {
        "utc": when.isot,
        "local": local_clock(when.isot, zone),
        "zone": zone_payload(
            float(latitude_deg),
            float(longitude_deg),
            when.to_datetime(timezone=datetime_timezone.utc),
        ),
        "latitude_deg": float(latitude_deg),
        "longitude_deg": float(longitude_deg),
        "height_m": float(height_m),
        "lst_deg": round(
            float(when.sidereal_time("apparent", longitude=site.lon).to_value(u.deg)), 4
        ),
        "sun": sun,
        "moon": moon,
        "sources": sources,
        "satellites": satellites,
        "aircraft": aircraft,
        "layers": layers,
    }
