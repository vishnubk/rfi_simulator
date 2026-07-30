r"""Interference sources injected at the voltage level.

An interference source is anything that adds power to the antennas which
did not come from the tracked sky: a transmitter on a nearby hilltop, an
arcing power line, a satellite. This module defines the plug-in interface
(`RFISource`), the geometry and band helpers every source shares, and the
two stationary ground-based sources; moving platforms live in
`rfi_simulator.satellites` and `rfi_simulator.aircraft`. The simulator
adds every source's contribution to the sky-plus-noise voltages *before*
correlation, so interference goes through exactly the same correlator and
fringe-stopping path as celestial signal.

Geometry: near field, no plane-wave approximation
-------------------------------------------------
Celestial sources are effectively at infinity, so their wavefronts are
planar and the antenna delay is the projection :math:`-(\mathbf{r} \cdot
\hat{s})/c` (see `rfi_simulator.delays`). Terrestrial and orbiting
transmitters are *not* at infinity: a tower 2 km away subtends a curved
wavefront across a 100 m array, and that curvature is one of the few
handles an excision algorithm has for telling interference from sky. So
this module uses the exact absolute path delay

.. math::

    \tau_i = \frac{|\mathbf{x}_{\mathrm{src}} - \mathbf{r}_i|}{c},

with :math:`\mathbf{x}_{\mathrm{src}}` the transmitter position and
:math:`\mathbf{r}_i` the antenna position, both in local ENU meters. The
interferometric observable is the *difference* of two such delays, which
tends to the plane-wave projection only as the source recedes. Phases are
applied at the **RF channel frequency**, :math:`e^{-2\pi i f \tau_i}`,
exactly as the sky path does, so the sign conventions of
`rfi_simulator.delays` and `rfi_simulator.correlator` carry over
unchanged.

Because the correlator stops the fringe on the tracked phase center and a
ground-based transmitter does not move with the sky, interference phase
winds with time in the visibilities. That is real physics, not a bug: it
is what makes interference look different from a source in the field, and
tests here assert the winding is present rather than removing it.

Amplitude convention
--------------------
Sources are specified by the power they *deliver to the array*, not by
transmitter watts: ``received_power_jy`` is the power measured at the
array origin, in the same janskys as `rfi_simulator.sky` fluxes, so it can
be compared directly with a source flux or with ``noise_std**2``. Each
antenna is then scaled by free-space spreading,

.. math::

    a_i = \frac{|\mathbf{x}_{\mathrm{src}}|}{|\mathbf{x}_{\mathrm{src}}
          - \mathbf{r}_i|},

which is within a part in :math:`10^4` of unity for a kilometre-distant
transmitter and a 100 m array, but is kept because it is free and becomes
important for close-in sources.

Labelling convention (ground truth)
-----------------------------------
Every source returns an occupancy mask of shape ``(n_chan, n_time)``
alongside its voltages, and `rfi_simulator.voltages.VoltageBlock` carries
one mask per source. **A cell is labelled occupied when the source's mean
power in that cell, evaluated at the array origin, exceeds 1 % of that
source's peak cell power within the same block.** Two details of that
sentence are deliberate:

* *Mean* power, i.e. the deterministic modulation envelope
  :math:`E|v|^2`, not the single realized noise draw. A band-limited
  noise transmitter has exponentially distributed instantaneous power, so
  thresholding the realization would punch random holes in the labels of
  cells the transmitter is plainly occupying. The envelope is the honest
  answer to "was the transmitter on in this cell".
* *Relative to the block peak*, so the labels are invariant to the
  overall power scaling of the source and only depend on its
  time-frequency structure.

The 1 % (-20 dB) threshold is a convention, not a physical boundary; it
is exposed as `OCCUPANCY_THRESHOLD` so that experiments with stricter or
looser labels are a one-line change.

Randomness
----------
Sources never own a generator. Each `contribution` call receives a
`BlockContext` whose ``rng`` is spawned from the block's own seed, so a
block remains a pure function of ``(seed, block index)``: duty-cycle
frames and event times are reproducible, can be generated out of order,
and do not correlate between blocks except through the rates that
generate them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from rfi_simulator.array_config import ArrayConfig, _to_value
from rfi_simulator.delays import SPEED_OF_LIGHT_M_S, earth_location, enu_unit_vector

__all__ = [
    "OCCUPANCY_THRESHOLD",
    "BlockContext",
    "ImpulsiveBroadband",
    "NarrowbandTransmitter",
    "RFISource",
    "band_overlaps",
    "channels_within",
    "circular_normal",
    "enu_from_ecef_offset",
    "enu_from_geodetic",
    "enu_from_horizontal",
    "enu_rotation_matrix",
    "near_field_phasors",
    "occupancy_mask",
    "out_of_band_message",
    "path_delays_s",
    "path_lengths_m",
    "spreading_amplitudes",
]

OCCUPANCY_THRESHOLD = 0.01
"""float: Mean-power fraction of a block's peak cell above which a cell is
labelled as occupied by a source. See the module docstring."""


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------
def path_lengths_m(position_enu_m, antenna_positions_enu_m: np.ndarray) -> np.ndarray:
    """Exact distances from a transmitter to each antenna.

    Parameters
    ----------
    position_enu_m : array_like
        Transmitter position, shape ``(3,)``, local ENU meters relative to
        the array origin.
    antenna_positions_enu_m : numpy.ndarray
        Antenna positions, shape ``(n_antennas, 3)``, local ENU meters.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_antennas,)`` float64 distances in meters.
    """
    position = np.asarray(position_enu_m, dtype=np.float64).reshape(3)
    antennas = np.asarray(antenna_positions_enu_m, dtype=np.float64)
    return np.linalg.norm(position[np.newaxis, :] - antennas, axis=1)


def path_delays_s(position_enu_m, antenna_positions_enu_m: np.ndarray) -> np.ndarray:
    r"""Absolute near-field propagation delay from a transmitter to each antenna.

    This is :math:`\tau_i = |\mathbf{x}_{\mathrm{src}} - \mathbf{r}_i| / c`
    -- an absolute travel time, positive, with no plane-wave approximation
    and no subtraction of the array-origin delay. Only *differences* of
    these delays are observable, and taking those differences is the
    correlator's job.

    Parameters
    ----------
    position_enu_m : array_like
        Transmitter position, shape ``(3,)``, local ENU meters.
    antenna_positions_enu_m : numpy.ndarray
        Antenna positions, shape ``(n_antennas, 3)``, local ENU meters.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_antennas,)`` float64 delays in seconds.
    """
    return path_lengths_m(position_enu_m, antenna_positions_enu_m) / SPEED_OF_LIGHT_M_S


def spreading_amplitudes(position_enu_m, antenna_positions_enu_m: np.ndarray) -> np.ndarray:
    """Free-space ``1/r`` amplitude of each antenna, normalized at the array origin.

    Parameters
    ----------
    position_enu_m : array_like
        Transmitter position, shape ``(3,)``, local ENU meters.
    antenna_positions_enu_m : numpy.ndarray
        Antenna positions, shape ``(n_antennas, 3)``, local ENU meters.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_antennas,)`` float64 amplitude factors, equal to
        ``|x_src| / |x_src - r_i|``. An antenna at the array origin gets
        exactly 1.0, so a source's stated received power is the power at
        the origin.

    Raises
    ------
    ValueError
        If the transmitter sits exactly at the array origin or exactly on
        an antenna, where the ``1/r`` model diverges.
    """
    position = np.asarray(position_enu_m, dtype=np.float64).reshape(3)
    reference_m = float(np.linalg.norm(position))
    if reference_m == 0.0:
        raise ValueError("transmitter position must not coincide with the array origin")
    distances_m = path_lengths_m(position, antenna_positions_enu_m)
    if np.any(distances_m == 0.0):
        raise ValueError("transmitter position must not coincide with an antenna position")
    return reference_m / distances_m


def enu_from_horizontal(azimuth_deg, elevation_deg, distance_m) -> np.ndarray:
    """Local ENU position of a point given as azimuth, elevation and range.

    Parameters
    ----------
    azimuth_deg : float or astropy.units.Quantity
        Azimuth measured from North through East, degrees.
    elevation_deg : float or astropy.units.Quantity
        Elevation above the horizon, degrees. Small positive values
        (0-2 deg) describe ground-based transmitters seen over the horizon.
    distance_m : float or astropy.units.Quantity
        Slant range from the array origin, meters.

    Returns
    -------
    numpy.ndarray
        Shape ``(3,)`` float64 ENU position in meters.
    """
    azimuth_rad = np.deg2rad(float(_to_value(azimuth_deg, u.deg)))
    elevation_rad = np.deg2rad(float(_to_value(elevation_deg, u.deg)))
    distance = float(_to_value(distance_m, u.m))
    if distance <= 0.0:
        raise ValueError(f"distance_m must be > 0, got {distance}")
    return distance * enu_unit_vector(elevation_rad, azimuth_rad).astype(np.float64)


def enu_from_geodetic(latitude_deg, longitude_deg, height_m, array: ArrayConfig) -> np.ndarray:
    """Local ENU position of a geodetic point relative to an array origin.

    Convenience for specifying a transmitter by map coordinates, e.g. a
    mast whose latitude/longitude is known.

    Parameters
    ----------
    latitude_deg, longitude_deg : float or astropy.units.Quantity
        Geodetic latitude and (East-positive) longitude of the
        transmitter, degrees.
    height_m : float or astropy.units.Quantity
        Height above the WGS84 ellipsoid, meters.
    array : ArrayConfig
        Array whose origin the result is measured from.

    Returns
    -------
    numpy.ndarray
        Shape ``(3,)`` float64 ENU position in meters.

    Notes
    -----
    The conversion is an exact geocentric difference rotated into the
    array origin's East-North-Up triad, so it is valid over the tens of
    kilometres that matter here (and beyond, at the price of Earth
    curvature making "Up" a local notion).
    """
    origin = earth_location(array)
    site = EarthLocation.from_geodetic(
        lon=_to_value(longitude_deg, u.deg) * u.deg,
        lat=_to_value(latitude_deg, u.deg) * u.deg,
        height=_to_value(height_m, u.m) * u.m,
    )
    delta_m = np.array(
        [
            (site.x - origin.x).to_value(u.m),
            (site.y - origin.y).to_value(u.m),
            (site.z - origin.z).to_value(u.m),
        ],
        dtype=np.float64,
    )
    return enu_from_ecef_offset(delta_m, origin)


def enu_rotation_matrix(origin: EarthLocation) -> np.ndarray:
    """Rotation from Earth-centred Earth-fixed axes to a local ENU triad.

    Parameters
    ----------
    origin : astropy.coordinates.EarthLocation
        Point whose local East-North-Up triad is wanted -- for this
        package, the array origin.

    Returns
    -------
    numpy.ndarray
        Shape ``(3, 3)`` float64 orthogonal matrix ``R`` such that
        ``R @ delta_ecef`` is the ENU representation of a geocentric
        offset vector ``delta_ecef``.
    """
    lat_rad = np.deg2rad(float(origin.lat.to_value(u.deg)))
    lon_rad = np.deg2rad(float(origin.lon.to_value(u.deg)))
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
    return np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=np.float64,
    )


def enu_from_ecef_offset(delta_ecef_m: np.ndarray, origin: EarthLocation) -> np.ndarray:
    """Rotate a geocentric offset vector into a local ENU triad.

    Parameters
    ----------
    delta_ecef_m : numpy.ndarray
        Shape ``(..., 3)`` offsets in Earth-centred Earth-fixed meters,
        i.e. target position minus `origin` position.
    origin : astropy.coordinates.EarthLocation
        Point defining the local triad (the array origin).

    Returns
    -------
    numpy.ndarray
        Shape ``(..., 3)`` float64 ENU coordinates in meters.
    """
    delta = np.asarray(delta_ecef_m, dtype=np.float64)
    return delta @ enu_rotation_matrix(origin).T


def band_overlaps(
    center_freq_hz: float, bandwidth_hz: float, freq_hz: np.ndarray, chan_width_hz: float
) -> bool:
    """Whether an emission's spectrum reaches into the simulated band at all.

    Parameters
    ----------
    center_freq_hz : float
        Center frequency of the emission, Hz.
    bandwidth_hz : float
        Full occupied bandwidth of the emission, Hz. May be zero for a
        pure carrier.
    freq_hz : numpy.ndarray
        Shape ``(n_chan,)`` ascending RF channel centers, Hz.
    chan_width_hz : float
        Channel width, Hz.

    Returns
    -------
    bool
        True if any part of ``center +/- bandwidth / 2`` lies within the
        band edges, taken as the outer channel centers extended by half a
        channel.

    Notes
    -----
    Sources call this to decide whether to raise. A transmitter
    configured outside the simulated band is nearly always a mistake in
    the observing setup rather than a deliberate silence, and a source
    that quietly contributes nothing is much harder to debug than one
    that says so.
    """
    low_hz = float(freq_hz[0]) - 0.5 * chan_width_hz
    high_hz = float(freq_hz[-1]) + 0.5 * chan_width_hz
    half_width_hz = 0.5 * bandwidth_hz
    return (center_freq_hz + half_width_hz >= low_hz) and (
        center_freq_hz - half_width_hz <= high_hz
    )


def out_of_band_message(
    name: str, center_freq_hz: float, bandwidth_hz: float, freq_hz: np.ndarray
) -> str:
    """The standard error text for an emission outside the simulated band.

    Parameters
    ----------
    name : str
        Source name.
    center_freq_hz, bandwidth_hz : float
        Emission center frequency and full bandwidth, Hz.
    freq_hz : numpy.ndarray
        Shape ``(n_chan,)`` ascending RF channel centers, Hz.

    Returns
    -------
    str
        A message naming both the emission and the simulated band, so the
        fix (re-center the band, or re-tune the source) is obvious.
    """
    return (
        f"source {name!r} emits at {center_freq_hz / 1e6:.3f} MHz "
        f"+/- {bandwidth_hz / 2e6:.3f} MHz, outside the simulated band "
        f"{freq_hz[0] / 1e6:.3f}-{freq_hz[-1] / 1e6:.3f} MHz. Re-center the "
        f"simulated band on the emission, or re-tune the source."
    )


def channels_within(freq_hz: np.ndarray, center_freq_hz: float, bandwidth_hz: float) -> np.ndarray:
    """Boolean mask of the channels an emission occupies.

    Parameters
    ----------
    freq_hz : numpy.ndarray
        Shape ``(n_chan,)`` RF channel center frequencies, Hz.
    center_freq_hz : float
        Center frequency of the emission, Hz.
    bandwidth_hz : float
        Full occupied bandwidth, Hz. Zero means a pure carrier, which
        lands in the single nearest channel.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(n_chan,)``. May be all False if the
        emission falls between the band edges and the nearest channel is
        far away -- callers decide whether that is an error.
    """
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    half_width_hz = 0.5 * bandwidth_hz
    if half_width_hz > 0.0:
        return np.abs(freq_hz - center_freq_hz) <= half_width_hz
    nearest = int(np.argmin(np.abs(freq_hz - center_freq_hz)))
    mask = np.zeros(freq_hz.shape, dtype=bool)
    mask[nearest] = True
    return mask


# ----------------------------------------------------------------------
# Block context
# ----------------------------------------------------------------------
@dataclass
class BlockContext:
    """Everything a source needs to emit into one block of voltages.

    The simulator builds one of these per block and hands the same object
    to every source (with a per-source `rng`). Sources must treat it as
    read-only.

    Attributes
    ----------
    index : int
        Block index within the observation, counting from zero.
    start_time : astropy.time.Time
        UTC time of the first sample in the block.
    center_time : astropy.time.Time
        UTC time of the block mid-point -- the epoch at which the block's
        sky geometry was frozen.
    sample_times_s : numpy.ndarray
        Shape ``(n_time,)`` float64 offsets of each sample from
        `start_time`, seconds. Sample ``k`` is at
        ``k * sample_period_s``.
    sample_period_s : float
        Post-channelization sample period, seconds.
    freq_hz : numpy.ndarray
        Shape ``(n_chan,)`` float64 RF channel center frequencies, Hz,
        ascending. These are sky frequencies, not baseband offsets.
    chan_width_hz : float
        Channel width, Hz.
    antenna_positions_enu_m : numpy.ndarray
        Shape ``(n_antennas, 3)`` antenna positions, local ENU meters.
    location : astropy.coordinates.EarthLocation
        Geodetic location of the array origin.
    phase_center_s_hat_enu : numpy.ndarray
        Shape ``(3,)`` ENU unit vector towards the tracked phase center at
        `center_time`. Sources do not need it to emit, but it lets a
        source report its angular separation from the field.
    phase_center_delays_s : numpy.ndarray
        Shape ``(n_antennas,)`` phase-center geometric delays of the
        block, seconds -- the delays the correlator will rotate out.
    rng : numpy.random.Generator
        Generator for this ``(block, source)`` pair. Independent of the
        sky and receiver-noise streams, and of every other source.
    """

    index: int
    start_time: Time
    center_time: Time
    sample_times_s: np.ndarray
    sample_period_s: float
    freq_hz: np.ndarray
    chan_width_hz: float
    antenna_positions_enu_m: np.ndarray
    location: EarthLocation
    phase_center_s_hat_enu: np.ndarray
    phase_center_delays_s: np.ndarray
    rng: np.random.Generator

    @property
    def n_antennas(self) -> int:
        """int: Number of antennas."""
        return self.antenna_positions_enu_m.shape[0]

    @property
    def n_chan(self) -> int:
        """int: Number of frequency channels."""
        return self.freq_hz.shape[0]

    @property
    def n_time(self) -> int:
        """int: Number of time samples in the block."""
        return self.sample_times_s.shape[0]

    @property
    def duration_s(self) -> float:
        """float: Block duration, seconds."""
        return self.n_time * self.sample_period_s

    def with_rng(self, rng: np.random.Generator) -> "BlockContext":
        """A shallow copy of this context carrying a different generator.

        Parameters
        ----------
        rng : numpy.random.Generator
            Generator for the new context.

        Returns
        -------
        BlockContext
            Same geometry and timing, new `rng`.
        """
        return BlockContext(
            index=self.index,
            start_time=self.start_time,
            center_time=self.center_time,
            sample_times_s=self.sample_times_s,
            sample_period_s=self.sample_period_s,
            freq_hz=self.freq_hz,
            chan_width_hz=self.chan_width_hz,
            antenna_positions_enu_m=self.antenna_positions_enu_m,
            location=self.location,
            phase_center_s_hat_enu=self.phase_center_s_hat_enu,
            phase_center_delays_s=self.phase_center_delays_s,
            rng=rng,
        )


# ----------------------------------------------------------------------
# Source interface
# ----------------------------------------------------------------------
class RFISource(ABC):
    """Base class for anything that adds interference to the voltages.

    Subclasses implement `contribution`, which is called once per block
    and returns both the voltages the source adds and the ground-truth
    occupancy mask that labels them.

    Parameters
    ----------
    name : str
        Short label carried through to `VoltageBlock.rfi_source_names` and
        `Visibilities.rfi_source_names`. Use something a plot legend can
        show, e.g. ``"tower_east"``.

    Notes
    -----
    Implementing a new source means answering three questions: where is it
    (an ENU position, possibly time-dependent), what does it emit (a
    time-frequency power envelope plus a waveform), and when is it on
    (the mask). Both concrete sources below follow the same three-step
    recipe:

    1. build the mean-power envelope ``(n_chan, n_time)`` at the array
       origin, and turn it into the mask with `occupancy_mask`,
    2. draw the waveform from ``ctx.rng`` and scale it by
       ``sqrt(envelope)``,
    3. apply the per-antenna near-field gains from `near_field_phasors`.

    Keeping step 1 separate is what makes the labels clean -- the mask
    comes from the envelope, never from the noisy realization.
    """

    def __init__(self, name: str) -> None:
        self.name = str(name)

    @abstractmethod
    def contribution(self, ctx: BlockContext) -> tuple[np.ndarray, np.ndarray]:
        """Voltages and occupancy mask for one block.

        Parameters
        ----------
        ctx : BlockContext
            Block geometry, timing and generator.

        Returns
        -------
        voltages : numpy.ndarray
            Complex64 array of shape ``(n_antennas, n_chan, n_time)`` in
            root-Jy, to be *added* to the sky-plus-noise voltages.
        mask : numpy.ndarray
            Boolean array of shape ``(n_chan, n_time)``, True where this
            source occupies the cell (see the module docstring for the
            threshold convention).
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


def near_field_phasors(
    position_enu_m, antenna_positions_enu_m: np.ndarray, freq_hz: np.ndarray
) -> np.ndarray:
    r"""Per-antenna, per-channel complex gain of a near-field transmitter.

    Combines the ``1/r`` amplitude with the exact path-delay phase:
    :math:`a_i e^{-2\pi i f \tau_i}`.

    Parameters
    ----------
    position_enu_m : array_like
        Transmitter position, shape ``(3,)``, local ENU meters.
    antenna_positions_enu_m : numpy.ndarray
        Antenna positions, shape ``(n_antennas, 3)``, local ENU meters.
    freq_hz : numpy.ndarray
        Shape ``(n_chan,)`` RF channel center frequencies, Hz.

    Returns
    -------
    numpy.ndarray
        Complex64 array of shape ``(n_antennas, n_chan)``.

    Notes
    -----
    The phase argument :math:`2\pi f \tau` is large -- a 2 km path at
    1.4 GHz is about ``6e4`` radians -- so it is accumulated in float64
    and only cast to complex64 after the exponential. Doing the
    multiplication in float32 would lose all phase information.
    """
    tau_s = path_delays_s(position_enu_m, antenna_positions_enu_m)
    amplitudes = spreading_amplitudes(position_enu_m, antenna_positions_enu_m)
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    phase = np.exp(-2j * np.pi * freq_hz[np.newaxis, :] * tau_s[:, np.newaxis])
    return (amplitudes[:, np.newaxis] * phase).astype(np.complex64)


def occupancy_mask(envelope_jy: np.ndarray, threshold: float = OCCUPANCY_THRESHOLD) -> np.ndarray:
    """Label the cells a source occupies, from its mean-power envelope.

    Parameters
    ----------
    envelope_jy : numpy.ndarray
        Shape ``(n_chan, n_time)`` mean power per cell at the array
        origin, Jy. Zero where the source emits nothing.
    threshold : float, optional
        Fraction of the block's peak cell power above which a cell counts
        as occupied. Default `OCCUPANCY_THRESHOLD` (1 %).

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(n_chan, n_time)``. All False if the
        source emitted nothing in this block.
    """
    peak = float(envelope_jy.max()) if envelope_jy.size else 0.0
    if peak <= 0.0:
        return np.zeros(envelope_jy.shape, dtype=bool)
    return envelope_jy > threshold * peak


def circular_normal(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Unit-power circular complex Gaussian samples, complex64.

    Parameters
    ----------
    rng : numpy.random.Generator
        Generator to draw from.
    shape : tuple of int
        Output shape.

    Returns
    -------
    numpy.ndarray
        Complex64 array with ``E|z|**2 == 1``.
    """
    parts = rng.standard_normal(size=(*shape, 2), dtype=np.float32)
    parts *= np.float32(1.0 / np.sqrt(2.0))
    return parts.view(np.complex64)[..., 0]


# ----------------------------------------------------------------------
# Concrete sources
# ----------------------------------------------------------------------
class NarrowbandTransmitter(RFISource):
    """A stationary transmitter occupying a contiguous slice of the band.

    Models the mast-on-a-ridge case: a fixed position on the ground, a
    fixed center frequency, a bandwidth covering a handful of channels,
    and a noise-like modulation (the sensible default for a multi-carrier
    digital waveform, whose per-channel statistics are Gaussian). An
    optional duty cycle switches the transmitter on and off in frames,
    which is what makes it look "blocky" in a time-frequency plot.

    Parameters
    ----------
    position_enu_m : array_like
        Transmitter position, shape ``(3,)``, local ENU meters relative to
        the array origin. Build one from map coordinates with
        `enu_from_geodetic`, or from a bearing with `enu_from_horizontal`.
    center_freq_hz : float or astropy.units.Quantity
        Center frequency of the emission, Hz.
    bandwidth_hz : float or astropy.units.Quantity
        Full occupied bandwidth, Hz. Channels whose centers fall within
        ``center_freq_hz +/- bandwidth_hz / 2`` carry the signal.
    received_power_jy : float or astropy.units.Quantity
        Total power received at the array origin while the transmitter is
        on, summed over the occupied channels, in janskys. Compare it with
        a source flux or with ``noise_std**2`` to set an interference-to-noise
        ratio. Default 100.0.
    duty_cycle : float, optional
        Fraction of frames in which the transmitter is on, in ``[0, 1]``.
        Default 1.0 (always on). Frames are drawn independently from the
        block generator, so the realized on-fraction fluctuates about
        `duty_cycle`.
    frame_duration_s : float or astropy.units.Quantity, optional
        Duration of one on/off frame, seconds. Default 0.01 (10 ms).
        Frames are aligned to the start of each block.
    name : str, optional
        Label for the source. Default ``"narrowband"``.

    Raises
    ------
    ValueError
        If `bandwidth_hz` or `received_power_jy` is negative, `duty_cycle`
        is outside ``[0, 1]``, or `frame_duration_s` is not positive.

    Notes
    -----
    The emission is confined to the occupied channels *exactly*: this
    package assumes a perfect channelizer, so there is no spectral
    leakage into neighbouring channels. Real polyphase filter banks leak,
    and an excision algorithm tuned on this simulator will be optimistic
    about how sharply interference is confined until a filter-bank model
    is added.

    Examples
    --------
    >>> import numpy as np
    >>> tower = NarrowbandTransmitter(
    ...     position_enu_m=enu_from_horizontal(90.0, 0.5, 2000.0),
    ...     center_freq_hz=1.4055e9,
    ...     bandwidth_hz=2.0e5,
    ...     received_power_jy=500.0,
    ...     duty_cycle=0.5,
    ... )
    >>> tower.name
    'narrowband'
    """

    def __init__(
        self,
        position_enu_m,
        center_freq_hz,
        bandwidth_hz,
        received_power_jy=100.0,
        *,
        duty_cycle: float = 1.0,
        frame_duration_s=0.01,
        name: str = "narrowband",
    ) -> None:
        super().__init__(name)
        self.position_enu_m = np.asarray(_to_value(position_enu_m, u.m), dtype=np.float64).reshape(
            3
        )
        self.center_freq_hz = float(_to_value(center_freq_hz, u.Hz))
        self.bandwidth_hz = float(_to_value(bandwidth_hz, u.Hz))
        self.received_power_jy = float(_to_value(received_power_jy, u.Jy))
        self.duty_cycle = float(duty_cycle)
        self.frame_duration_s = float(_to_value(frame_duration_s, u.s))

        if self.bandwidth_hz < 0.0:
            raise ValueError(f"bandwidth_hz must be >= 0, got {self.bandwidth_hz}")
        if self.received_power_jy < 0.0:
            raise ValueError(f"received_power_jy must be >= 0, got {self.received_power_jy}")
        if not 0.0 <= self.duty_cycle <= 1.0:
            raise ValueError(f"duty_cycle must be in [0, 1], got {self.duty_cycle}")
        if self.frame_duration_s <= 0.0:
            raise ValueError(f"frame_duration_s must be > 0, got {self.frame_duration_s}")

    def occupied_channels(self, freq_hz: np.ndarray) -> np.ndarray:
        """Boolean mask of the channels this transmitter emits into.

        Parameters
        ----------
        freq_hz : numpy.ndarray
            Shape ``(n_chan,)`` RF channel center frequencies, Hz.

        Returns
        -------
        numpy.ndarray
            Boolean array of shape ``(n_chan,)``.

        Notes
        -----
        A zero-bandwidth transmitter is a pure carrier and lands in the
        single channel nearest its center frequency, provided that channel
        actually contains it.
        """
        return channels_within(freq_hz, self.center_freq_hz, self.bandwidth_hz)

    def on_frames(self, ctx: BlockContext) -> np.ndarray:
        """Per-sample on/off state of the transmitter within a block.

        Parameters
        ----------
        ctx : BlockContext
            Block context; its `rng` supplies the frame draws.

        Returns
        -------
        numpy.ndarray
            Boolean array of shape ``(n_time,)``, True where the
            transmitter is on.
        """
        frame_index = np.floor_divide(
            ctx.sample_times_s, self.frame_duration_s, dtype=np.float64
        ).astype(np.int64)
        n_frames = int(frame_index.max()) + 1 if frame_index.size else 0
        # Always draw, so the generator stream does not depend on the duty
        # cycle value -- only on the number of frames in the block.
        draws = ctx.rng.random(n_frames)
        return draws[frame_index] < self.duty_cycle

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
            Boolean ``(n_chan, n_time)`` occupancy labels.

        Raises
        ------
        ValueError
            If the transmitter's occupied band lies wholly outside the
            simulated band. Silently emitting nothing would look like a
            working configuration, which is worse than an error.
        """
        occupied = self.occupied_channels(ctx.freq_hz)
        n_occupied = int(occupied.sum())
        in_band = band_overlaps(
            self.center_freq_hz, self.bandwidth_hz, ctx.freq_hz, ctx.chan_width_hz
        )
        if n_occupied == 0 or not in_band:
            raise ValueError(
                out_of_band_message(self.name, self.center_freq_hz, self.bandwidth_hz, ctx.freq_hz)
            )

        on = self.on_frames(ctx)

        envelope = np.zeros((ctx.n_chan, ctx.n_time), dtype=np.float64)
        power_per_channel_jy = self.received_power_jy / n_occupied
        envelope[np.ix_(occupied, on)] = power_per_channel_jy

        # One emitted waveform, shared by every antenna; only the occupied
        # channels are drawn, so the cost scales with the occupied band.
        waveform = circular_normal(ctx.rng, (n_occupied, ctx.n_time))
        waveform *= np.sqrt(envelope[occupied], dtype=np.float64).astype(np.float32)

        phasors = near_field_phasors(
            self.position_enu_m, ctx.antenna_positions_enu_m, ctx.freq_hz[occupied]
        )
        voltages = np.zeros((ctx.n_antennas, ctx.n_chan, ctx.n_time), dtype=np.complex64)
        voltages[:, occupied, :] = phasors[:, :, np.newaxis] * waveform[np.newaxis, :, :]

        return voltages, occupancy_mask(envelope)


class ImpulsiveBroadband(RFISource):
    """Short broadband bursts from a fixed direction: arcing, sparks, lightning.

    Events arrive as a Poisson process, each occupying a small number of
    consecutive time samples and the whole simulated band. Event powers
    are drawn from a bounded power law, so most bursts are weak and a few
    are very strong -- the heavy tail is the part that matters for
    excision, since a weak burst hides in the noise and a strong one
    saturates a whole integration.

    Parameters
    ----------
    rate_hz : float or astropy.units.Quantity
        Mean event rate, events per second. The number of events in a
        block is Poisson with mean ``rate_hz * block_duration_s``, so
        blocks are statistically independent and correlate only through
        this rate.
    received_power_jy : float or astropy.units.Quantity
        Power received at the array origin during the *weakest possible*
        event, summed over the whole band, in janskys. Actual events are
        this times a factor in ``[1, max_power_ratio]``. Default 1000.0.
    position_enu_m : array_like, optional
        Transmitter position, shape ``(3,)``, local ENU meters. If
        omitted, a fixed low-elevation position is used:
        `default_azimuth_deg` at 1 deg elevation and 5 km range, i.e. just
        over the horizon. Deliberately deterministic -- a position drawn
        from the block generator would move the source between blocks.
    power_law_index : float, optional
        Index ``alpha`` of the event-power distribution ``p(x) ~
        x**(-alpha)`` on ``[1, max_power_ratio]``. Default 2.0. Must be
        greater than 1 so the distribution is normalizable at the bright
        end without relying on the cutoff alone.
    max_power_ratio : float, optional
        Upper end of the power-law range, i.e. the brightest event is this
        many times `received_power_jy`. Default 30.0. Values far above
        ``1 / OCCUPANCY_THRESHOLD`` push the weakest events below the
        labelling threshold in blocks that also contain a bright one.
    pulse_width_samples : int, optional
        Duration of one event in post-channelization samples. Default 1,
        i.e. a single time sample (32.768 us at the package defaults) --
        the natural width of an impulse that a channelizer has smeared
        over one spectrum.
    name : str, optional
        Label for the source. Default ``"impulsive"``.

    Raises
    ------
    ValueError
        If any rate, power or width is non-positive, if
        `power_law_index` is not greater than 1, or if `max_power_ratio`
        is less than 1.

    Notes
    -----
    Events are placed uniformly over the block's samples and are allowed
    to run off the end of the block, where they are simply truncated. That
    is a small (order ``pulse_width_samples / n_time``) edge effect and it
    keeps blocks independent, which is worth more than exact continuity
    across a block boundary.

    Examples
    --------
    >>> sparks = ImpulsiveBroadband(rate_hz=50.0, received_power_jy=2000.0)
    >>> sparks.name
    'impulsive'
    """

    default_azimuth_deg = 135.0
    default_elevation_deg = 1.0
    default_distance_m = 5000.0

    def __init__(
        self,
        rate_hz,
        received_power_jy=1000.0,
        *,
        position_enu_m=None,
        power_law_index: float = 2.0,
        max_power_ratio: float = 30.0,
        pulse_width_samples: int = 1,
        name: str = "impulsive",
    ) -> None:
        super().__init__(name)
        self.rate_hz = float(_to_value(rate_hz, u.Hz))
        self.received_power_jy = float(_to_value(received_power_jy, u.Jy))
        self.power_law_index = float(power_law_index)
        self.max_power_ratio = float(max_power_ratio)
        self.pulse_width_samples = int(pulse_width_samples)

        if position_enu_m is None:
            self.position_enu_m = enu_from_horizontal(
                self.default_azimuth_deg,
                self.default_elevation_deg,
                self.default_distance_m,
            )
        else:
            self.position_enu_m = np.asarray(
                _to_value(position_enu_m, u.m), dtype=np.float64
            ).reshape(3)

        if self.rate_hz < 0.0:
            raise ValueError(f"rate_hz must be >= 0, got {self.rate_hz}")
        if self.received_power_jy < 0.0:
            raise ValueError(f"received_power_jy must be >= 0, got {self.received_power_jy}")
        if self.power_law_index <= 1.0:
            raise ValueError(f"power_law_index must be > 1, got {self.power_law_index}")
        if self.max_power_ratio < 1.0:
            raise ValueError(f"max_power_ratio must be >= 1, got {self.max_power_ratio}")
        if self.pulse_width_samples < 1:
            raise ValueError(f"pulse_width_samples must be >= 1, got {self.pulse_width_samples}")

    def draw_events(self, ctx: BlockContext) -> tuple[np.ndarray, np.ndarray]:
        """Draw the event times and powers of one block.

        Parameters
        ----------
        ctx : BlockContext
            Block context; its `rng` supplies the Poisson count, the
            start samples and the powers.

        Returns
        -------
        start_samples : numpy.ndarray
            Shape ``(n_events,)`` int array of first-sample indices.
        powers_jy : numpy.ndarray
            Shape ``(n_events,)`` float64 band-summed received powers at
            the array origin, Jy.

        Notes
        -----
        The power law is sampled by inverting its CDF, which is exact and
        needs one uniform per event -- no rejection loop, so the number of
        generator draws depends only on the event count.
        """
        expected = self.rate_hz * ctx.duration_s
        n_events = int(ctx.rng.poisson(expected))
        start_samples = ctx.rng.integers(0, ctx.n_time, size=n_events)

        uniforms = ctx.rng.random(n_events)
        exponent = 1.0 - self.power_law_index
        if self.max_power_ratio == 1.0:
            ratios = np.ones(n_events, dtype=np.float64)
        else:
            ratios = (1.0 + uniforms * (self.max_power_ratio**exponent - 1.0)) ** (1.0 / exponent)
        return start_samples, self.received_power_jy * ratios

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
            Boolean ``(n_chan, n_time)`` occupancy labels. Every channel
            of an event sample is flagged, since the emission is flat
            across the band.
        """
        start_samples, powers_jy = self.draw_events(ctx)

        # Band-summed power -> power per channel, flat across the band.
        per_sample_jy = np.zeros(ctx.n_time, dtype=np.float64)
        for start, power_jy in zip(start_samples, powers_jy):
            stop = min(int(start) + self.pulse_width_samples, ctx.n_time)
            per_sample_jy[int(start) : stop] += power_jy / ctx.n_chan

        envelope = np.broadcast_to(per_sample_jy[np.newaxis, :], (ctx.n_chan, ctx.n_time))
        mask = occupancy_mask(envelope)

        voltages = np.zeros((ctx.n_antennas, ctx.n_chan, ctx.n_time), dtype=np.complex64)
        active = np.flatnonzero(per_sample_jy > 0.0)
        if active.size == 0:
            return voltages, mask

        waveform = circular_normal(ctx.rng, (ctx.n_chan, active.size))
        waveform *= np.sqrt(per_sample_jy[active]).astype(np.float32)[np.newaxis, :]

        phasors = near_field_phasors(self.position_enu_m, ctx.antenna_positions_enu_m, ctx.freq_hz)
        voltages[:, :, active] = phasors[:, :, np.newaxis] * waveform[np.newaxis, :, :]
        return voltages, mask
