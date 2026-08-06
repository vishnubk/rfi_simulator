"""Observatory-local clock time, resolved from latitude/longitude alone.

The Mock Observatory only ever knows a site as a (latitude, longitude,
height) triple -- there is no IANA time zone shipped with a site config,
and no tz-boundary database dependency is allowed here (no third-party
packages beyond the standard library). So there is no way to hand back
the *correct* zone for an arbitrary point on Earth; what follows is a
deliberately coarse approximation, in two tiers:

1. A small hardcoded table of longitude/latitude bands that covers the
   handful of US zones (Hawaii, Alaska, Pacific, Mountain, Central,
   Eastern) the bundled site and its likely variants fall in. Coordinates
   landing inside one of these bands get the real IANA key for that zone
   -- correct, including DST, because `zoneinfo` resolves the offset at
   the requested instant rather than from a precomputed constant.
2. Everywhere else, a `round(longitude / 15)`-based fallback to the
   POSIX `Etc/GMT+-N` zones. These have no DST and are only ever an
   approximation of local civil time -- they exist so the UI always has
   *something* better than raw UTC to show.

Every result says which tier produced it (`ZoneInfoResult.approximate`)
so the UI can caption approximate times as such instead of implying a
precision that was never computed.

One sharp edge worth flagging loudly: the POSIX `Etc/GMT+-N` zones use
*inverted* signs relative to ordinary usage -- `Etc/GMT+8` is UTC-8, not
UTC+8. `resolve_zone` accounts for this; a test pins it so it cannot
regress silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

__all__ = [
    "ZoneInfoResult",
    "resolve_zone",
    "zone_abbreviation",
    "to_local",
    "local_clock",
    "local_day_fraction",
    "zone_payload",
]


@dataclass(frozen=True)
class ZoneInfoResult:
    """A resolved time zone plus whether the resolution was a guess."""

    key: str
    approximate: bool
    tzinfo: timezone | ZoneInfo


# Longitude/latitude bands for the handful of US zones the bundled site
# (and its likely siblings) fall in. Edges are approximate -- real zone
# boundaries follow state and county lines, not meridians -- so any
# coordinate landing in one of these bands is treated as "close enough"
# rather than exact, but it is still the *real* IANA zone, not a
# fixed-offset stand-in, so DST is handled correctly.
_US_BANDS: tuple[tuple[float, float, float, float, str], ...] = (
    # (lon_min, lon_max, lat_min, lat_max, IANA key)
    # Adjacent bands share an edge (e.g. both -125..-114 and -114..-102
    # include -114.0 itself); `_lookup_table` returns the first match in
    # this tuple's order, so a longitude landing exactly on a shared edge
    # resolves to whichever band is listed first here (Los_Angeles before
    # Denver, and so on west to east). That is an arbitrary tie-break, not
    # a meaningful one -- real zone boundaries do not run along these
    # meridians anyway, so a site placed exactly on the line is already
    # inside this table's stated approximation, not a case it is trying to
    # get exactly right.
    (-161.0, -154.0, 18.0, 23.0, "Pacific/Honolulu"),
    (-170.0, -129.0, 51.0, 72.0, "America/Anchorage"),
    (-125.0, -114.0, 24.0, 50.0, "America/Los_Angeles"),
    (-114.0, -102.0, 24.0, 50.0, "America/Denver"),
    (-102.0, -87.0, 24.0, 50.0, "America/Chicago"),
    (-87.0, -67.0, 24.0, 50.0, "America/New_York"),
)


def _lookup_table(latitude_deg: float, longitude_deg: float) -> str | None:
    """Return an IANA key from the banded US table, or None if no band matches."""
    for lon_min, lon_max, lat_min, lat_max, key in _US_BANDS:
        if lon_min <= longitude_deg <= lon_max and lat_min <= latitude_deg <= lat_max:
            return key
    return None


def _etc_gmt_key(longitude_deg: float) -> str:
    """Longitude fallback key, honoring the Etc/GMT sign inversion.

    `Etc/GMT+N` is UTC-N (not UTC+N) by POSIX convention, so a site N
    hours *behind* UTC (west, negative longitude) needs `Etc/GMT+N` with
    a positive N. Clamped to [-14, 12] so the key is always one zoneinfo
    actually ships.
    """
    # `n` is the ordinary UTC offset the longitude implies (west of the
    # prime meridian is negative). The Etc/GMT+-N key encodes the
    # NEGATED offset, so it must be flipped here.
    utc_offset_hours = round(longitude_deg / 15.0)
    etc_n = -utc_offset_hours
    etc_n = max(-14, min(12, etc_n))
    sign = "+" if etc_n >= 0 else "-"
    return f"Etc/GMT{sign}{abs(etc_n)}"


def resolve_zone(latitude_deg: float, longitude_deg: float) -> ZoneInfoResult:
    """Resolve a best-effort IANA time zone for a (lat, lon) site.

    Tries the small US longitude/latitude band table first (exact IANA
    zone, DST-aware, `approximate=False`); falls back to an `Etc/GMT+-N`
    fixed-offset zone from the longitude alone (`approximate=True`); and
    if even that key cannot be loaded, falls back to plain UTC.
    """
    key = _lookup_table(latitude_deg, longitude_deg)
    if key is not None:
        try:
            return ZoneInfoResult(key=key, approximate=False, tzinfo=ZoneInfo(key))
        except Exception:
            pass

    etc_key = _etc_gmt_key(longitude_deg)
    try:
        return ZoneInfoResult(key=etc_key, approximate=True, tzinfo=ZoneInfo(etc_key))
    except Exception:
        return ZoneInfoResult(key="UTC", approximate=True, tzinfo=timezone.utc)


def zone_abbreviation(zone: ZoneInfoResult, when: datetime) -> str:
    """The tz abbreviation in effect at `when` (e.g. "PDT", "PST").

    `when` must be an aware datetime (any zone; it is converted). For
    `Etc/*` zones `%Z` yields an unhelpful string like "-08" -- those are
    rewritten as a "UTC-08:00" style string instead.
    """
    local = when.astimezone(zone.tzinfo)
    abbrev = local.strftime("%Z")
    if zone.key.startswith("Etc/") or not abbrev or abbrev.lstrip("+-").isdigit():
        offset = local.utcoffset()
        total_minutes = int(offset.total_seconds() // 60) if offset else 0
        sign = "+" if total_minutes >= 0 else "-"
        total_minutes = abs(total_minutes)
        hh, mm = divmod(total_minutes, 60)
        return f"UTC{sign}{hh:02d}:{mm:02d}"
    return abbrev


def _parse_utc_isot(utc_isot: str) -> datetime:
    """Parse an astropy-style naive UTC ISO string into an aware UTC datetime."""
    if not utc_isot:
        raise ValueError("utc_isot must be a non-empty ISO timestamp string")
    text = utc_isot.strip()
    if text.endswith("Z"):
        text = text[:-1]
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"could not parse {utc_isot!r} as an ISO timestamp") from exc
    if parsed.tzinfo is not None:
        return parsed.astimezone(timezone.utc)
    return parsed.replace(tzinfo=timezone.utc)


def to_local(utc_isot: str, zone: ZoneInfoResult) -> datetime:
    """Convert a naive-UTC astropy ISO string to an aware local datetime."""
    utc_dt = _parse_utc_isot(utc_isot)
    return utc_dt.astimezone(zone.tzinfo)


def local_clock(utc_isot: str, zone: ZoneInfoResult) -> str:
    """ "HH:MM" in local time, or "--:--" for falsy/missing input."""
    if not utc_isot:
        return "--:--"
    local = to_local(utc_isot, zone)
    return local.strftime("%H:%M")


def local_day_fraction(utc_isot: str, zone: ZoneInfoResult) -> float:
    """Fraction (0..1) of the way through the LOCAL calendar day."""
    local = to_local(utc_isot, zone)
    seconds_since_midnight = (
        local.hour * 3600 + local.minute * 60 + local.second + local.microsecond / 1e6
    )
    return seconds_since_midnight / 86400.0


def zone_payload(latitude_deg: float, longitude_deg: float, when: datetime | None = None) -> dict:
    """A JSON-safe summary of the resolved zone, for the API layer.

    `when` defaults to the current instant; pass a specific instant to
    get the offset/abbreviation that would apply then (DST matters).
    """
    zone = resolve_zone(latitude_deg, longitude_deg)
    instant = when if when is not None else datetime.now(timezone.utc)
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)

    abbreviation = zone_abbreviation(zone, instant)
    offset = instant.astimezone(zone.tzinfo).utcoffset()
    utc_offset_hours = offset.total_seconds() / 3600.0 if offset else 0.0

    if zone.approximate:
        note = f"Times shown in {abbreviation}, approximated from the site longitude."
    else:
        note = f"Times shown in {abbreviation} ({zone.key})."

    return {
        "key": zone.key,
        "abbreviation": abbreviation,
        "utc_offset_hours": utc_offset_hours,
        "approximate": zone.approximate,
        "note": note,
    }
