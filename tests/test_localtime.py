"""Tests for observatory-local clock time resolution.

The bundled default site (latitude 37.234, longitude -118.282) is used
throughout as the reference case, matching the site used elsewhere in
the test suite. No network access and no third-party tz database are
involved -- everything here is `zoneinfo` plus the small banded table
in `rfi_simulator.webui.localtime`.
"""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from rfi_simulator.webui.localtime import (
    ZoneInfoResult,
    local_clock,
    local_day_fraction,
    resolve_zone,
    to_local,
    zone_abbreviation,
    zone_payload,
)

SITE = {"latitude_deg": 37.234, "longitude_deg": -118.282}


def test_bundled_site_resolves_to_los_angeles():
    zone = resolve_zone(SITE["latitude_deg"], SITE["longitude_deg"])
    assert zone.key == "America/Los_Angeles"
    assert zone.approximate is False
    assert isinstance(zone.tzinfo, ZoneInfo)


def test_hawaii_band():
    zone = resolve_zone(20.0, -157.0)
    assert zone.key == "Pacific/Honolulu"
    assert zone.approximate is False


def test_alaska_band():
    zone = resolve_zone(61.0, -150.0)
    assert zone.key == "America/Anchorage"
    assert zone.approximate is False


def test_mountain_band():
    zone = resolve_zone(39.7, -105.0)
    assert zone.key == "America/Denver"
    assert zone.approximate is False


def test_central_band():
    zone = resolve_zone(41.8, -93.0)
    assert zone.key == "America/Chicago"
    assert zone.approximate is False


def test_eastern_band():
    zone = resolve_zone(40.7, -74.0)
    assert zone.key == "America/New_York"
    assert zone.approximate is False


def test_etc_gmt_sign_is_inverted_for_western_longitude():
    # -118 degrees is west, so local time is behind UTC by ~8 hours, and
    # the POSIX Etc/GMT+-N convention has the SIGN INVERTED relative to
    # ordinary usage: Etc/GMT+8 means UTC-8, not UTC+8. A coordinate that
    # falls outside every US band but at longitude -118 must therefore
    # fall back to "Etc/GMT+8", never "Etc/GMT-8".
    zone = resolve_zone(0.0, -118.282)
    assert zone.key == "Etc/GMT+8"
    assert zone.approximate is True
    offset = datetime(2026, 1, 15, tzinfo=timezone.utc).astimezone(zone.tzinfo).utcoffset()
    assert offset.total_seconds() / 3600.0 == -8.0


def test_etc_gmt_sign_is_inverted_for_eastern_longitude():
    zone = resolve_zone(0.0, 118.282)
    assert zone.key == "Etc/GMT-8"
    assert zone.approximate is True
    offset = datetime(2026, 1, 15, tzinfo=timezone.utc).astimezone(zone.tzinfo).utcoffset()
    assert offset.total_seconds() / 3600.0 == 8.0


def test_dst_crossing_changes_offset_at_bundled_site():
    zone = resolve_zone(SITE["latitude_deg"], SITE["longitude_deg"])
    january = to_local("2026-01-15T00:00:00.000", zone)
    july = to_local("2026-07-15T00:00:00.000", zone)
    assert january.utcoffset().total_seconds() / 3600.0 == -8.0  # PST
    assert july.utcoffset().total_seconds() / 3600.0 == -7.0  # PDT
    assert zone_abbreviation(zone, january.astimezone(timezone.utc)) == "PST"
    assert zone_abbreviation(zone, july.astimezone(timezone.utc)) == "PDT"


def test_etc_zone_abbreviation_is_utc_offset_style():
    zone = resolve_zone(0.0, -118.282)
    when = datetime(2026, 7, 15, tzinfo=timezone.utc)
    assert zone_abbreviation(zone, when) == "UTC-08:00"


def test_fractional_offset_zone_reports_fractional_hours():
    # Not every zone sits on an hour boundary; this exercises the
    # offset-hours arithmetic (not the fallback table, which is
    # integer-only by construction) against a real half-hour-offset
    # zone to confirm fractional offsets are handled, not rounded away.
    zone = ZoneInfoResult(key="Asia/Kolkata", approximate=False, tzinfo=ZoneInfo("Asia/Kolkata"))
    local = to_local("2026-07-15T00:00:00.000", zone)
    assert local.utcoffset().total_seconds() / 3600.0 == 5.5


def test_local_clock_format():
    zone = resolve_zone(SITE["latitude_deg"], SITE["longitude_deg"])
    assert local_clock("2026-07-15T19:30:00.000", zone) == "12:30"


def test_local_clock_handles_falsy_input():
    zone = resolve_zone(SITE["latitude_deg"], SITE["longitude_deg"])
    assert local_clock("", zone) == "--:--"
    assert local_clock(None, zone) == "--:--"


def test_to_local_accepts_z_suffix_and_fractional_seconds():
    zone = resolve_zone(SITE["latitude_deg"], SITE["longitude_deg"])
    a = to_local("2026-07-15T19:30:00Z", zone)
    b = to_local("2026-07-15T19:30:00.500", zone)
    assert a.hour == 12
    assert b.hour == 12 and b.microsecond == 500000


def test_to_local_raises_on_bad_input():
    zone = resolve_zone(SITE["latitude_deg"], SITE["longitude_deg"])
    with pytest.raises(ValueError):
        to_local("", zone)
    with pytest.raises(ValueError):
        to_local(None, zone)
    with pytest.raises(ValueError):
        to_local("not-a-timestamp", zone)


def test_local_day_fraction_midnight_and_noon():
    zone = resolve_zone(SITE["latitude_deg"], SITE["longitude_deg"])
    # 07:00 UTC on this date is 00:00 local (PDT, UTC-7).
    assert local_day_fraction("2026-07-15T07:00:00.000", zone) == pytest.approx(0.0)
    assert local_day_fraction("2026-07-15T19:00:00.000", zone) == pytest.approx(0.5)


def test_zone_payload_shape_and_note_for_real_zone():
    when = datetime(2026, 7, 15, tzinfo=timezone.utc)
    payload = zone_payload(SITE["latitude_deg"], SITE["longitude_deg"], when=when)
    assert payload["key"] == "America/Los_Angeles"
    assert payload["abbreviation"] == "PDT"
    assert payload["utc_offset_hours"] == -7.0
    assert payload["approximate"] is False
    assert payload["note"] == "Times shown in PDT (America/Los_Angeles)."


def test_zone_payload_note_for_approximate_zone():
    when = datetime(2026, 7, 15, tzinfo=timezone.utc)
    payload = zone_payload(0.0, -118.282, when=when)
    assert payload["key"] == "Etc/GMT+8"
    assert payload["approximate"] is True
    assert payload["utc_offset_hours"] == -8.0
    assert "approximated from the site longitude" in payload["note"]


def test_zone_payload_defaults_when_to_now():
    payload = zone_payload(SITE["latitude_deg"], SITE["longitude_deg"])
    assert payload["key"] == "America/Los_Angeles"
    assert isinstance(payload["utc_offset_hours"], float)
