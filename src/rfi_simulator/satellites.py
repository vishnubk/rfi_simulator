r"""Satellite downlinks as interference: TLE propagation, geometry, Doppler.

A satellite differs from the ground-based transmitters in
`rfi_simulator.rfi` in three ways, and this module exists to get all
three right:

1. **It moves, fast.** Position comes from propagating a two-line element
   set (TLE) with SGP4, then converting the result into the array's local
   ENU frame. A low-Earth orbiter crosses the sky in minutes, so the
   geometry is re-evaluated every block, exactly like the sky delays.
2. **It Dopplers.** The received carrier is displaced by
   :math:`-f_0 \dot{r}/c`, and the displacement drifts through the band
   over a pass. The size depends on the orbit: a low-Earth orbiter reaches
   :math:`\pm 7` km/s of range rate, i.e. :math:`\pm 35` kHz at L band --
   more than one 30.5 kHz channel at this package's default resolution --
   while a medium-orbit navigation satellite manages a few hundred m/s and
   a couple of kilohertz. The shift is applied as a time-dependent offset
   of the *received* carrier frequency, evaluated per block.
3. **It is far away.** At 20 000 km the wavefront is planar to well under
   a picosecond across a 100 m array. The exact path delays of
   `rfi_simulator.rfi` are used anyway -- they cost nothing and remain
   correct as the source gets closer, and the agreement with the
   plane-wave formula in that limit is a useful cross-check on both code
   paths.

Coordinate chain
----------------
``SGP4 -> TEME -> ITRS -> local ENU``. SGP4 natively produces positions in
the True Equator Mean Equinox frame, which astropy models directly; the
ITRS step applies Earth rotation and polar motion, and the last step is a
fixed rotation into the array origin's East-North-Up triad
(`rfi_simulator.rfi.enu_rotation_matrix`). Range rate is obtained by
differencing the topocentric range a fraction of a second either side of
the epoch, which automatically includes the observer's motion with the
rotating Earth -- no separate velocity-frame bookkeeping to get wrong.

Frequency is a free parameter
-----------------------------
Real navigation downlinks sit at 1575.42 MHz (L1) and 1227.60 MHz (L2),
both outside this package's default 11.7 MHz band around 1.405 GHz. Rather
than pretend otherwise, `SatelliteTransmitter` raises `ValueError` when its
emission lies wholly outside the simulated band, and `carrier_freq_hz` is
a plain constructor argument: point the simulated band at the real
downlink, or give the satellite a fictitious in-band downlink frequency.
The second option is physically honest -- the geometry, Doppler and
time-variability are what make satellite interference distinctive, not the
particular carrier -- and it is the cheap way to study an in-band
satellite without simulating an 800 MHz-wide band.

Element sets go stale
---------------------
A TLE is accurate for a few days around its epoch and quietly wrong far
from it. `fetch_tles` retrieves current elements from the public
general-perturbations catalogue, cache-first, and is **never** called by
the test suite; ``configs/tle_sample.txt`` bundles one frozen element set
so examples and tests have a real orbit to propagate offline.
"""

from __future__ import annotations

import time as _time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import ITRS, TEME, CartesianRepresentation, EarthLocation
from astropy.time import Time

from rfi_simulator.array_config import _to_value
from rfi_simulator.delays import SPEED_OF_LIGHT_M_S
from rfi_simulator.rfi import (
    BlockContext,
    RFISource,
    band_overlaps,
    channels_within,
    circular_normal,
    enu_from_ecef_offset,
    near_field_phasors,
    occupancy_mask,
    out_of_band_message,
)

__all__ = [
    "CATALOGUE_URL",
    "SatelliteTransmitter",
    "TwoLineElement",
    "fetch_tles",
    "read_tle_file",
]

CATALOGUE_URL = "https://celestrak.org/NORAD/elements/gp.php"
"""str: Public general-perturbations catalogue endpoint used by `fetch_tles`."""

_RANGE_RATE_STEP_S = 0.5
"""float: Half-interval used to difference the topocentric range, seconds."""


class TwoLineElement:
    """A parsed two-line element set, ready to propagate.

    Parameters
    ----------
    line1, line2 : str
        The two 69-character element lines.
    name : str, optional
        Object name, e.g. the line preceding the pair in a three-line
        listing. Defaults to ``""``.

    Attributes
    ----------
    name : str
        Object name.
    line1, line2 : str
        The raw element lines, as given.
    satrec : sgp4.model.Satrec
        The propagator built from the lines.

    Raises
    ------
    ValueError
        If the lines are malformed, or do not start with ``"1 "`` and
        ``"2 "`` respectively.
    ImportError
        If ``sgp4`` is not installed.

    Examples
    --------
    >>> tle = TwoLineElement.from_file("configs/tle_sample.txt")
    >>> tle.name
    'GPS BIIR-5  (PRN 22)'
    """

    def __init__(self, line1: str, line2: str, name: str = "") -> None:
        try:
            from sgp4.api import Satrec
        except ImportError as exc:  # pragma: no cover - depends on install
            raise ImportError(
                "propagating element sets needs the 'sgp4' package; "
                "install it with `pip install sgp4`"
            ) from exc

        self.name = str(name).strip()
        self.line1 = str(line1).rstrip()
        self.line2 = str(line2).rstrip()

        if not self.line1.startswith("1 ") or not self.line2.startswith("2 "):
            raise ValueError(
                "element lines must begin with '1 ' and '2 '; got "
                f"{self.line1[:2]!r} and {self.line2[:2]!r}"
            )
        self.satrec = Satrec.twoline2rv(self.line1, self.line2)

    @classmethod
    def from_string(cls, text: str) -> "TwoLineElement":
        """Parse a two- or three-line element set from text.

        Parameters
        ----------
        text : str
            Either two element lines, or a name line followed by two
            element lines. Blank lines and ``#`` comments are ignored.

        Returns
        -------
        TwoLineElement
            The parsed element set.

        Raises
        ------
        ValueError
            If the text does not contain exactly one element set.
        """
        entries = _parse_tle_text(text)
        if len(entries) != 1:
            raise ValueError(f"expected exactly one element set, found {len(entries)}")
        return entries[0]

    @classmethod
    def from_file(cls, path: str | Path, index: int = 0) -> "TwoLineElement":
        """Read one element set from a file.

        Parameters
        ----------
        path : str or pathlib.Path
            File holding one or more element sets, e.g. the bundled
            ``configs/tle_sample.txt`` or a catalogue download.
        index : int, optional
            Which element set to take, for files holding several.
            Default 0.

        Returns
        -------
        TwoLineElement
            The selected element set.

        Raises
        ------
        IndexError
            If `index` is out of range for the file.
        """
        entries = read_tle_file(path)
        if not -len(entries) <= index < len(entries):
            raise IndexError(
                f"element set index {index} out of range for {path} ({len(entries)} element sets)"
            )
        return entries[index]

    @property
    def epoch(self) -> Time:
        """astropy.time.Time: Epoch of the element set (UTC).

        Notes
        -----
        Element sets degrade away from this epoch -- days, not months.
        Compare it against the observation time before trusting a
        propagated position.
        """
        return Time(self.satrec.jdsatepoch, self.satrec.jdsatepochF, format="jd", scale="utc")

    def teme_position_m(self, time: Time) -> np.ndarray:
        """Propagate to True Equator Mean Equinox coordinates.

        Parameters
        ----------
        time : astropy.time.Time
            Scalar or 1-D UTC time(s) to propagate to.

        Returns
        -------
        numpy.ndarray
            Shape ``time.shape + (3,)`` float64 positions in meters.

        Raises
        ------
        ValueError
            If the propagator reports an error, e.g. a decayed orbit or a
            time absurdly far from the element-set epoch.
        """
        scalar = time.isscalar
        times = time.reshape(1) if scalar else time
        errors, positions_km, _ = self.satrec.sgp4_array(
            np.ascontiguousarray(times.jd1), np.ascontiguousarray(times.jd2)
        )
        if np.any(errors):
            raise ValueError(
                f"element-set propagation failed for {self.name or 'satellite'} "
                f"(sgp4 error code {int(errors[np.flatnonzero(errors)[0]])}); the "
                "requested time may be far from the element-set epoch "
                f"({self.epoch.isot})"
            )
        positions_m = np.asarray(positions_km, dtype=np.float64) * 1000.0
        return positions_m[0] if scalar else positions_m

    def enu_position_m(self, time: Time, location: EarthLocation) -> np.ndarray:
        """Propagate to topocentric ENU coordinates of an observing site.

        Parameters
        ----------
        time : astropy.time.Time
            Scalar or 1-D UTC time(s).
        location : astropy.coordinates.EarthLocation
            Observer location -- for this package, the array origin.

        Returns
        -------
        numpy.ndarray
            Shape ``time.shape + (3,)`` float64 ENU positions in meters,
            relative to `location`.

        Notes
        -----
        Pass all the times you need in one call: the TEME-to-ITRS
        transform carries a fixed astropy overhead per call that dwarfs
        the propagation itself.
        """
        scalar = time.isscalar
        times = time.reshape(1) if scalar else time

        teme_m = self.teme_position_m(times)
        teme = TEME(CartesianRepresentation(teme_m.T * u.m), obstime=times)
        itrs = teme.transform_to(ITRS(obstime=times))

        site_itrs = location.get_itrs(times).cartesian
        delta_m = np.stack(
            [
                (itrs.cartesian.x - site_itrs.x).to_value(u.m),
                (itrs.cartesian.y - site_itrs.y).to_value(u.m),
                (itrs.cartesian.z - site_itrs.z).to_value(u.m),
            ],
            axis=-1,
        )
        enu_m = enu_from_ecef_offset(delta_m, location)
        return enu_m[0] if scalar else enu_m

    def range_rate_m_s(self, time: Time, location: EarthLocation) -> float:
        """Rate of change of the topocentric range, meters per second.

        Parameters
        ----------
        time : astropy.time.Time
            Scalar UTC epoch.
        location : astropy.coordinates.EarthLocation
            Observer location.

        Returns
        -------
        float
            Range rate in m/s: **positive while receding**, negative while
            approaching, which is the sign that makes the Doppler shift
            ``-f0 * range_rate / c``.

        Notes
        -----
        Computed by central-differencing the range half a second either
        side of `time`, in the Earth-fixed frame, so the observer's
        rotation with the Earth is included without any extra velocity
        bookkeeping. Over half a second a low orbiter moves a few
        kilometres along a smooth arc, so the truncation error is
        negligible next to the element set's own accuracy.
        """
        epochs = Time([time - _RANGE_RATE_STEP_S * u.s, time + _RANGE_RATE_STEP_S * u.s])
        positions = self.enu_position_m(epochs, location)
        ranges_m = np.linalg.norm(positions, axis=-1)
        return float((ranges_m[1] - ranges_m[0]) / (2.0 * _RANGE_RATE_STEP_S))

    def __repr__(self) -> str:
        return f"TwoLineElement(name={self.name!r}, epoch={self.epoch.isot})"


def _parse_tle_text(text: str) -> list[TwoLineElement]:
    """Parse every element set in a block of catalogue text.

    Parameters
    ----------
    text : str
        Catalogue text in two- or three-line format. Blank lines and
        lines starting with ``#`` are ignored, so bundled files may carry
        provenance comments.

    Returns
    -------
    list of TwoLineElement
        One entry per element set, in file order.

    Raises
    ------
    ValueError
        If a line beginning ``"1 "`` is not followed by one beginning
        ``"2 "``.
    """
    lines = [
        line.rstrip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    entries: list[TwoLineElement] = []
    index = 0
    pending_name = ""
    while index < len(lines):
        line = lines[index]
        if line.startswith("1 "):
            if index + 1 >= len(lines) or not lines[index + 1].startswith("2 "):
                raise ValueError(f"element line 1 at position {index} has no matching line 2")
            entries.append(TwoLineElement(line, lines[index + 1], pending_name))
            pending_name = ""
            index += 2
        else:
            pending_name = line
            index += 1
    return entries


def read_tle_file(path: str | Path) -> list[TwoLineElement]:
    """Read every element set in a file.

    Parameters
    ----------
    path : str or pathlib.Path
        File in two- or three-line format; ``#`` comments allowed.

    Returns
    -------
    list of TwoLineElement
        One entry per element set, in file order.

    Raises
    ------
    ValueError
        If the file contains no element sets.
    """
    path = Path(path)
    entries = _parse_tle_text(path.read_text())
    if not entries:
        raise ValueError(f"no element sets found in {path}")
    return entries


def fetch_tles(
    group: str,
    cache_dir: str | Path,
    *,
    max_age_hours: float = 24.0,
    url: str = CATALOGUE_URL,
    timeout_s: float = 30.0,
) -> list[TwoLineElement]:
    """Fetch current element sets for a catalogue group, cache first.

    Parameters
    ----------
    group : str
        Catalogue group name, e.g. ``"gps-ops"``, ``"galileo"``,
        ``"starlink"``.
    cache_dir : str or pathlib.Path
        Directory for the cached download. Created if absent. Each group
        is cached as ``<group>.tle``.
    max_age_hours : float, optional
        Reuse the cache without touching the network if it is younger
        than this. Default 24 -- element sets are refreshed a few times a
        day, and hammering a free public service for every simulation run
        is bad manners.
    url : str, optional
        Catalogue endpoint. Default `CATALOGUE_URL`.
    timeout_s : float, optional
        Network timeout, seconds. Default 30.

    Returns
    -------
    list of TwoLineElement
        Element sets for the group.

    Raises
    ------
    RuntimeError
        If the download fails and no cached copy exists. The message
        names the cache path, so the offline fix -- drop a catalogue file
        there by hand -- is obvious.

    Warns
    -----
    UserWarning
        If the download fails but a stale cache exists, which is then
        used. Stale elements are usually better than no elements, but you
        should know it happened.

    Notes
    -----
    **Nothing in the test suite calls this.** Tests use the bundled
    ``configs/tle_sample.txt`` so they neither need the network nor drift
    as the catalogue updates.

    Examples
    --------
    >>> tles = fetch_tles("gps-ops", "~/.cache/tles")  # doctest: +SKIP
    >>> len(tles)  # doctest: +SKIP
    31
    """
    cache_dir = Path(cache_dir).expanduser()
    cache_path = cache_dir / f"{group}.tle"

    if cache_path.exists():
        age_hours = (_time.time() - cache_path.stat().st_mtime) / 3600.0
        if age_hours < max_age_hours:
            return read_tle_file(cache_path)

    query = f"{url}?GROUP={urllib.parse.quote(group)}&FORMAT=tle"
    try:
        with urllib.request.urlopen(query, timeout=timeout_s) as response:
            text = response.read().decode("utf-8")
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        if cache_path.exists():
            warnings.warn(
                f"could not refresh element sets for group {group!r} ({exc}); "
                f"using the cached copy at {cache_path}, which may be stale",
                UserWarning,
                stacklevel=2,
            )
            return read_tle_file(cache_path)
        raise RuntimeError(
            f"could not download element sets for group {group!r} ({exc}) and no "
            f"cached copy exists at {cache_path}. Fetch the group manually while "
            "online and save it there, or pass an element set directly."
        ) from exc

    entries = _parse_tle_text(text)
    if not entries:
        raise RuntimeError(
            f"the catalogue returned no element sets for group {group!r}; check the group name"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text)
    return entries


class SatelliteTransmitter(RFISource):
    r"""A downlink from an orbiting transmitter, propagated from a TLE.

    Emission is a narrowband carrier, optionally surrounded by
    band-limited noise sidebands -- the coarse shape of a navigation
    downlink, where a strong residual carrier sits inside a spread-spectrum
    skirt. Both are placed at the *received* frequency, i.e. after the
    Doppler shift of the block.

    Parameters
    ----------
    tle : TwoLineElement or str
        Element set, or text to parse into one (two lines, optionally
        preceded by a name line).
    carrier_freq_hz : float or astropy.units.Quantity, optional
        Rest-frame carrier frequency, Hz. Default 1575.42 MHz, the real
        navigation L1 downlink -- which lies outside this package's
        default band, so a default-band simulation raises unless you
        re-center the band or choose an in-band frequency. That is
        deliberate; see the module docstring.
    received_power_jy : float or astropy.units.Quantity, optional
        Total power received at the array origin, summed over carrier and
        sidebands, in janskys. Default 500.0.
    sideband_bandwidth_hz : float or astropy.units.Quantity, optional
        Full width of the noise sidebands about the carrier, Hz. Default
        0.0, i.e. a bare carrier.
    sideband_power_fraction : float, optional
        Fraction of `received_power_jy` in the sidebands rather than the
        carrier, in ``[0, 1]``. Default 0.5 when sidebands are present is
        *not* assumed -- the default is 0.0, so a caller who asks for a
        bandwidth but no power split still gets a pure carrier and can see
        why.
    apply_doppler : bool, optional
        If True (default) shift the received frequency by
        ``-carrier_freq_hz * range_rate / c``. Set False to isolate the
        geometry when debugging.
    name : str, optional
        Label for the source. Default ``"satellite"``.

    Raises
    ------
    ValueError
        If any power or bandwidth is negative, or
        `sideband_power_fraction` is outside ``[0, 1]``.

    Notes
    -----
    Position, Doppler and delays are frozen at each block's mid-point, the
    same convention the sky delays use. Over one 32.768 ms block a low
    orbiter moves ~250 m, changing the range by well under a wavelength
    for a high-elevation pass, so the frozen-per-block approximation is
    consistent with how celestial sources are handled here. Very long
    blocks would smear the Doppler within a block; shorten the block
    rather than the observation if that matters.

    Examples
    --------
    >>> tle = TwoLineElement.from_file("configs/tle_sample.txt")
    >>> downlink = SatelliteTransmitter(
    ...     tle, carrier_freq_hz=1.405e9, received_power_jy=300.0
    ... )
    >>> downlink.name
    'satellite'
    """

    def __init__(
        self,
        tle: TwoLineElement | str,
        carrier_freq_hz=1.57542e9,
        received_power_jy=500.0,
        *,
        sideband_bandwidth_hz=0.0,
        sideband_power_fraction: float = 0.0,
        apply_doppler: bool = True,
        name: str = "satellite",
    ) -> None:
        super().__init__(name)
        self.tle = tle if isinstance(tle, TwoLineElement) else TwoLineElement.from_string(tle)
        self.carrier_freq_hz = float(_to_value(carrier_freq_hz, u.Hz))
        self.received_power_jy = float(_to_value(received_power_jy, u.Jy))
        self.sideband_bandwidth_hz = float(_to_value(sideband_bandwidth_hz, u.Hz))
        self.sideband_power_fraction = float(sideband_power_fraction)
        self.apply_doppler = bool(apply_doppler)

        if self.received_power_jy < 0.0:
            raise ValueError(f"received_power_jy must be >= 0, got {self.received_power_jy}")
        if self.sideband_bandwidth_hz < 0.0:
            raise ValueError(
                f"sideband_bandwidth_hz must be >= 0, got {self.sideband_bandwidth_hz}"
            )
        if not 0.0 <= self.sideband_power_fraction <= 1.0:
            raise ValueError(
                f"sideband_power_fraction must be in [0, 1], got {self.sideband_power_fraction}"
            )

    def doppler_shift_hz(self, time: Time, location: EarthLocation) -> float:
        r"""Received-frequency offset at one epoch, Hz.

        Parameters
        ----------
        time : astropy.time.Time
            Scalar UTC epoch.
        location : astropy.coordinates.EarthLocation
            Observer location (the array origin).

        Returns
        -------
        float
            :math:`-f_0 \dot{r}/c`: **negative while the satellite
            recedes** (the classical red shift), positive on approach.
            Zero if `apply_doppler` is False.
        """
        if not self.apply_doppler:
            return 0.0
        range_rate_m_s = self.tle.range_rate_m_s(time, location)
        return -self.carrier_freq_hz * range_rate_m_s / SPEED_OF_LIGHT_M_S

    def received_freq_hz(self, time: Time, location: EarthLocation) -> float:
        """Carrier frequency as seen by the array at one epoch, Hz.

        Parameters
        ----------
        time : astropy.time.Time
            Scalar UTC epoch.
        location : astropy.coordinates.EarthLocation
            Observer location.

        Returns
        -------
        float
            ``carrier_freq_hz + doppler_shift_hz(...)``.
        """
        return self.carrier_freq_hz + self.doppler_shift_hz(time, location)

    def contribution(self, ctx: BlockContext) -> tuple[np.ndarray, np.ndarray]:
        """Voltages and occupancy mask for one block.

        Parameters
        ----------
        ctx : BlockContext
            Block geometry, timing and generator.

        Returns
        -------
        voltages : numpy.ndarray
            Complex64 ``(n_antennas, n_chan, n_time)`` contribution in
            root-Jy.
        mask : numpy.ndarray
            Boolean ``(n_chan, n_time)`` occupancy labels. Constant in
            time within a block, since the emission is continuous; it is
            the *frequency* that moves, block to block, as the Doppler
            shift changes.

        Raises
        ------
        ValueError
            If the received spectrum lies wholly outside the simulated
            band -- the usual case for a real navigation downlink at the
            package defaults. Re-center the band or set an in-band
            `carrier_freq_hz`.
        """
        position_enu_m = self.tle.enu_position_m(ctx.center_time, ctx.location)
        received_hz = self.received_freq_hz(ctx.center_time, ctx.location)
        total_width_hz = max(self.sideband_bandwidth_hz, 0.0)

        if not band_overlaps(received_hz, total_width_hz, ctx.freq_hz, ctx.chan_width_hz):
            raise ValueError(
                out_of_band_message(self.name, received_hz, total_width_hz, ctx.freq_hz)
            )

        carrier_power_jy = self.received_power_jy * (1.0 - self.sideband_power_fraction)
        sideband_power_jy = self.received_power_jy * self.sideband_power_fraction

        envelope = np.zeros((ctx.n_chan, ctx.n_time), dtype=np.float64)

        carrier_channel = channels_within(ctx.freq_hz, received_hz, 0.0)
        if carrier_power_jy > 0.0:
            envelope[carrier_channel, :] += carrier_power_jy

        sideband_channels = np.zeros(ctx.n_chan, dtype=bool)
        if sideband_power_jy > 0.0 and total_width_hz > 0.0:
            sideband_channels = channels_within(ctx.freq_hz, received_hz, total_width_hz)
            n_sideband = int(sideband_channels.sum())
            if n_sideband > 0:
                envelope[sideband_channels, :] += sideband_power_jy / n_sideband

        occupied = envelope[:, 0] > 0.0
        n_occupied = int(occupied.sum())
        if n_occupied == 0:
            raise ValueError(
                out_of_band_message(self.name, received_hz, total_width_hz, ctx.freq_hz)
            )

        # One emitted waveform shared by every antenna, as for the sky.
        waveform = circular_normal(ctx.rng, (n_occupied, ctx.n_time))
        waveform *= np.sqrt(envelope[occupied, 0]).astype(np.float32)[:, np.newaxis]

        phasors = near_field_phasors(
            position_enu_m, ctx.antenna_positions_enu_m, ctx.freq_hz[occupied]
        )
        voltages = np.zeros((ctx.n_antennas, ctx.n_chan, ctx.n_time), dtype=np.complex64)
        voltages[:, occupied, :] = phasors[:, :, np.newaxis] * waveform[np.newaxis, :, :]

        return voltages, occupancy_mask(envelope)
