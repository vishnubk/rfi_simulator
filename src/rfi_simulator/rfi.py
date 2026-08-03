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

Per-antenna coupling
--------------------
Free-space spreading is not how interference power actually distributes
itself over an array. Measurement campaigns on real arrays find single
antennas carrying an order of magnitude more interference power than their
neighbours -- factors of twenty above the interference-free continuum in
one antenna while the array median sits within a few per cent of it -- with
the affected antennas clustered on physically adjacent positions, and
carriers that are plainly visible in one antenna's spectrum yet invisible
in the array mean. Cable runs, shielding, ground screens, local
oscillators and the local terrain all couple an emitter into a receiver by
paths this simulator does not model.

Rather than model those paths, every source accepts a ``coupling``: a
per-antenna **linear amplitude** factor, so received power scales as
``coupling**2``, applied on top of the near-field geometry above. It can be
uniform (``None``, the default, so nothing changes), an explicit vector
-- which is how a measured, spatially clustered pattern is expressed --
or a lognormal draw. `resolve_coupling` turns any of those into the vector,
and `RFISource.coupling_amplitudes` is the ground truth for it.

Coupling and the occupancy mask are deliberately **separate** ground
truths; see the labelling convention below.

Waveform statistics
-------------------
The default modulation here is band-limited Gaussian noise, which is right
for a multi-carrier digital signal but wrong for a plain carrier. A real
constant-envelope emission -- a phase- or frequency-modulated carrier, of
which there are many -- has *constant instantaneous power*, so the
generalized spectral kurtosis of the occupied channel falls towards zero
(measured values on real carriers run from a few hundredths to about a
half), whereas a Gaussian signal sits at exactly 1 no matter how strong it
is. A kurtosis-based detector therefore cannot see a Gaussian-modelled
carrier at *any* power, which makes the modulation choice a first-order
question for a flagging benchmark, not a cosmetic one.
`NarrowbandTransmitter` and `CombTransmitter` accept
``waveform="constant_envelope"`` for this; `constant_envelope` synthesizes
the unit-modulus samples.

On/off structure
----------------
Real transmitters that duty-cycle do it on a clock: framed emission at a
fixed period (sub-millisecond to a few milliseconds is common) and impulse
trains locked to the mains frequency, rather than independent coin flips
per frame. The i.i.d. ``duty_cycle`` frames remain the default; a
``envelope={"type": "periodic", ...}`` specification gives deterministic,
observation-continuous frames instead, and `ImpulsiveBroadband` accepts a
periodic ``arrival`` for regular pulse trains. The two are mutually
exclusive by construction: a source is either coin-flipped or clocked, and
silently composing the two would produce an on-fraction that matches
neither.

Harmonic combs
--------------
A single misbehaving device rarely emits at one frequency. Driving any
nonlinearity produces a comb: narrow lines at integer multiples of one
fundamental, often many of them spanning hundreds of megahertz, all
sharing one position, one coupling and one on/off pattern because they are
one device. `CombTransmitter` models that as a single source.

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

The mask has **no antenna axis, and per-antenna coupling does not change
it**. A cell is labelled by what the transmitter emitted, evaluated at the
array origin: the label answers "was this source occupying this
time-frequency cell", which is a property of the emitter, not of any one
receiver. So an antenna whose coupling is small -- even exactly zero --
still carries the source's labels. That is a deliberate choice with two
reasons behind it. First, a per-antenna label would need its own detection
threshold per antenna, and the answer would then depend on the arbitrary
`OCCUPANCY_THRESHOLD` rather than on the physics. Second, flagging is
usually decided and applied per baseline or per array, so a label that
disappears on the quietest antenna would penalize an algorithm for
flagging a cell that is genuinely contaminated everywhere else. The
per-antenna information is not lost: it is published separately and
exactly, as the coupling vector
(`RFISource.coupling_amplitudes`, carried on blocks as
`rfi_simulator.voltages.VoltageBlock.rfi_coupling`). An experiment that
wants per-antenna labels can form them from the two by thresholding
``mask[np.newaxis] * coupling[:, np.newaxis, np.newaxis] ** 2`` however it
sees fit.

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

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from rfi_simulator.array_config import ArrayConfig, _to_value
from rfi_simulator.delays import SPEED_OF_LIGHT_M_S, earth_location, enu_unit_vector

__all__ = [
    "COUPLING_TYPES",
    "MIN_SEPARATION_WAVELENGTHS",
    "OCCUPANCY_THRESHOLD",
    "WAVEFORMS",
    "BlockContext",
    "CombTransmitter",
    "ImpulsiveBroadband",
    "NarrowbandTransmitter",
    "RFISource",
    "band_overlaps",
    "channels_within",
    "circular_normal",
    "constant_envelope",
    "enu_from_ecef_offset",
    "enu_from_geodetic",
    "enu_from_horizontal",
    "enu_rotation_matrix",
    "elevation_deg",
    "near_field_phasors",
    "occupancy_mask",
    "out_of_band_message",
    "path_delays_s",
    "path_lengths_m",
    "resolve_coupling",
    "spreading_amplitudes",
]

COUPLING_TYPES = ("lognormal",)
"""tuple of str: Accepted ``type`` values of a coupling specification
dictionary. See `resolve_coupling`."""

WAVEFORMS = ("gaussian", "constant_envelope")
"""tuple of str: Accepted ``waveform`` values of the narrowband sources.
``"gaussian"`` is band-limited noise modulation; ``"constant_envelope"`` is
a phase-modulated carrier of constant instantaneous power (see the module
docstring)."""

OCCUPANCY_THRESHOLD = 0.01
"""float: Mean-power fraction of a block's peak cell above which a cell is
labelled as occupied by a source. See the module docstring."""

MIN_SEPARATION_WAVELENGTHS = 10.0
"""float: Transmitter-to-antenna separation, in wavelengths at the band
center, below which `near_field_phasors` warns that the ``1/r`` amplitude
model has stopped being meaningful."""


def elevation_deg(position_enu_m) -> float:
    """Elevation angle of a local ENU position above the horizontal plane.

    Parameters
    ----------
    position_enu_m : array_like
        Position, shape ``(3,)``, local ENU meters relative to the array
        origin.

    Returns
    -------
    float
        Elevation in degrees: 90 straight up, 0 on the horizontal plane
        through the array origin, negative below it.

    Notes
    -----
    This is a *geometric* elevation in the array's local tangent plane. It
    ignores Earth curvature, so for a distant object it is not quite the
    elevation an observer would measure; the difference matters only for
    sources within a degree or so of the horizon, which is also where
    refraction and terrain dominate anyway.
    """
    position = np.asarray(position_enu_m, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(position))
    if norm == 0.0:
        raise ValueError("cannot take the elevation of the array origin itself")
    return float(np.rad2deg(np.arcsin(position[2] / norm)))


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
# Per-antenna coupling
# ----------------------------------------------------------------------
def _normalize_coupling(coupling) -> np.ndarray | dict | None:
    """Validate a coupling specification without resolving it.

    Parameters
    ----------
    coupling : None, array_like or mapping
        The value a source was constructed with. See `resolve_coupling`
        for the accepted forms.

    Returns
    -------
    None, numpy.ndarray or dict
        ``None`` unchanged; an explicit vector as a read-only float64
        array; a mapping as a validated plain dict.

    Raises
    ------
    ValueError
        If a mapping has an unknown ``type`` or unexpected keys, if
        ``sigma_db`` is negative or non-finite, if a vector is not 1-D, or
        if any factor is negative or non-finite.

    Notes
    -----
    The length of an explicit vector cannot be checked here -- the antenna
    count is only known once the source is handed a block -- so that check
    lives in `resolve_coupling`.
    """
    if coupling is None:
        return None

    if isinstance(coupling, dict):
        spec = dict(coupling)
        kind = spec.pop("type", None)
        if kind not in COUPLING_TYPES:
            raise ValueError(
                f"coupling type must be one of {COUPLING_TYPES}, got {kind!r}. Pass an "
                "explicit per-antenna vector instead to describe a measured pattern."
            )
        sigma_db = float(spec.pop("sigma_db", 0.0))
        seed = spec.pop("seed", None)
        if spec:
            raise ValueError(f"unexpected keys in the coupling specification: {sorted(spec)}")
        if not np.isfinite(sigma_db) or sigma_db < 0.0:
            raise ValueError(f"coupling sigma_db must be finite and >= 0, got {sigma_db}")
        if seed is None:
            raise ValueError(
                "a lognormal coupling specification needs a 'seed': the coupling is a "
                "fixed property of the array, not a per-block random draw, so it is "
                "seeded independently of the simulator's own seed tree"
            )
        return {"type": kind, "sigma_db": sigma_db, "seed": int(seed)}

    factors = np.array(coupling, dtype=np.float64, copy=True)
    if factors.ndim != 1 or factors.size < 1:
        raise ValueError(f"an explicit coupling must have shape (n_antennas,), got {factors.shape}")
    if not np.all(np.isfinite(factors)):
        raise ValueError("coupling contains non-finite values")
    if np.any(factors < 0.0):
        raise ValueError("coupling factors are linear amplitudes and must be >= 0")
    factors.setflags(write=False)
    return factors


def resolve_coupling(coupling, n_antennas: int) -> np.ndarray:
    """Per-antenna linear amplitude factors for a coupling specification.

    Parameters
    ----------
    coupling : None, array_like or mapping
        One of:

        * ``None`` -- uniform coupling: every antenna receives the source
          with the amplitude the near-field geometry gives it.
        * a length-``n_antennas`` sequence of **linear amplitude** factors,
          so antenna power scales as the square. This is how a measured,
          spatially clustered pattern is expressed: one antenna at 4.6
          amplitude is a factor 21 in power above the rest.
        * a mapping ``{"type": "lognormal", "sigma_db": s, "seed": k}``,
          drawing each antenna's factor as ``10 ** (x / 20)`` with ``x``
          from a zero-mean normal of width ``s`` dB. ``s`` is therefore
          the rms of the per-antenna coupling expressed as a power ratio in
          dB, matching the convention of
          `rfi_simulator.instrument.InstrumentModel`. The median factor is
          1, so the source's ``received_power_jy`` still describes a
          typical antenna.
    n_antennas : int
        Number of antennas in the array the source is emitting into.

    Returns
    -------
    numpy.ndarray
        Float64 array of shape ``(n_antennas,)``. Exactly ones for
        ``None``.

    Raises
    ------
    ValueError
        If an explicit vector's length does not match `n_antennas`, or if
        the specification is otherwise invalid (see `_normalize_coupling`).

    Notes
    -----
    A lognormal specification carries its own ``seed`` and is resolved
    without touching the simulator's generators, so switching coupling on
    or off never perturbs any other part of a run -- and the same seed
    always gives the same array.

    Examples
    --------
    >>> import numpy as np
    >>> resolve_coupling(None, 3)
    array([1., 1., 1.])
    >>> factors = resolve_coupling({"type": "lognormal", "sigma_db": 6.0, "seed": 1}, 4)
    >>> factors.shape
    (4,)
    """
    n_antennas = int(n_antennas)
    if n_antennas < 1:
        raise ValueError(f"n_antennas must be >= 1, got {n_antennas}")

    spec = _normalize_coupling(coupling)
    if spec is None:
        return np.ones(n_antennas, dtype=np.float64)
    if isinstance(spec, dict):
        power_db = np.random.default_rng(spec["seed"]).normal(
            loc=0.0, scale=spec["sigma_db"], size=n_antennas
        )
        return 10.0 ** (power_db / 20.0)
    if spec.size != n_antennas:
        raise ValueError(
            f"coupling has {spec.size} entries but the array has {n_antennas} antennas"
        )
    return np.array(spec, dtype=np.float64, copy=True)


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
    coupling : None, array_like or mapping, optional
        Per-antenna linear amplitude coupling, applied on top of the
        near-field geometry (see `resolve_coupling` and the module
        docstring). Default ``None``: uniform coupling, which is a
        bit-for-bit no-op.

    Attributes
    ----------
    name : str
        The source's label.
    coupling : None, numpy.ndarray or dict
        The validated coupling specification, as given. Use
        `coupling_amplitudes` for the resolved vector.

    Notes
    -----
    Implementing a new source means answering three questions: where is it
    (an ENU position, possibly time-dependent), what does it emit (a
    time-frequency power envelope plus a waveform), and when is it on
    (the mask). The concrete sources below follow the same three-step
    recipe:

    1. build the mean-power envelope ``(n_chan, n_time)`` at the array
       origin, and turn it into the mask with `occupancy_mask`,
    2. draw the waveform from ``ctx.rng`` and scale it by
       ``sqrt(envelope)``,
    3. apply the per-antenna gains from `coupled_phasors` -- the
       near-field geometry times this source's coupling.

    Keeping step 1 separate is what makes the labels clean -- the mask
    comes from the envelope, never from the noisy realization. A source
    that builds its phasors with `near_field_phasors` directly still
    works, but silently ignores `coupling`; use `coupled_phasors`.
    """

    def __init__(self, name: str, *, coupling=None) -> None:
        self.name = str(name)
        self.coupling = _normalize_coupling(coupling)
        self._coupling_cache: dict[int, np.ndarray] = {}

    def coupling_amplitudes(self, n_antennas: int) -> np.ndarray:
        """This source's resolved per-antenna coupling -- ground truth.

        Parameters
        ----------
        n_antennas : int
            Number of antennas in the array.

        Returns
        -------
        numpy.ndarray
            Read-only float64 array of shape ``(n_antennas,)`` of linear
            amplitude factors; received power scales as the square. All
            ones for the default uniform coupling.

        Raises
        ------
        ValueError
            If an explicit coupling vector's length does not match
            `n_antennas`.

        Notes
        -----
        Resolved once per antenna count and cached, so a lognormal draw is
        the *same* array for every block of a run -- coupling is a fixed
        property of the installation, not a per-block fluctuation.
        """
        n_antennas = int(n_antennas)
        cached = self._coupling_cache.get(n_antennas)
        if cached is None:
            cached = resolve_coupling(self.coupling, n_antennas)
            cached.setflags(write=False)
            self._coupling_cache[n_antennas] = cached
        return cached

    def coupled_phasors(
        self, position_enu_m, antenna_positions_enu_m: np.ndarray, freq_hz: np.ndarray
    ) -> np.ndarray:
        """Per-antenna complex gains: near-field geometry times coupling.

        Parameters
        ----------
        position_enu_m : array_like
            Emitter position, shape ``(3,)``, local ENU meters.
        antenna_positions_enu_m : numpy.ndarray
            Antenna positions, shape ``(n_antennas, 3)``, local ENU meters.
        freq_hz : numpy.ndarray
            Shape ``(n_freq,)`` RF frequencies, Hz.

        Returns
        -------
        numpy.ndarray
            Complex64 array of shape ``(n_antennas, n_freq)``: exactly
            `near_field_phasors` for uniform coupling, and that times each
            antenna's coupling amplitude otherwise.
        """
        phasors = near_field_phasors(position_enu_m, antenna_positions_enu_m, freq_hz)
        if self.coupling is None:
            return phasors
        coupling = self.coupling_amplitudes(phasors.shape[0])
        return (phasors * coupling[:, np.newaxis]).astype(np.complex64)

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
    _warn_if_too_close(position_enu_m, antenna_positions_enu_m, freq_hz)
    phase = np.exp(-2j * np.pi * freq_hz[np.newaxis, :] * tau_s[:, np.newaxis])
    return (amplitudes[:, np.newaxis] * phase).astype(np.complex64)


def _warn_if_too_close(
    position_enu_m, antenna_positions_enu_m: np.ndarray, freq_hz: np.ndarray
) -> None:
    """Warn when a transmitter is inside the array's reactive-near-field zone.

    Parameters
    ----------
    position_enu_m : array_like
        Transmitter position, shape ``(3,)``, local ENU meters.
    antenna_positions_enu_m : numpy.ndarray
        Antenna positions, shape ``(n_antennas, 3)``, local ENU meters.
    freq_hz : numpy.ndarray
        Channel frequencies, Hz; their mean sets the wavelength.

    Warns
    -----
    UserWarning
        If any antenna is within `MIN_SEPARATION_WAVELENGTHS` wavelengths
        of the transmitter.

    Notes
    -----
    The ``1/r`` amplitude model is normalized at the *array origin*, so
    ``received_power_jy`` means "power at the origin". That contract stops
    being useful once a transmitter is close enough that ``|x - r_i|``
    varies wildly across the array: at 1 mm separation an antenna receives
    a factor of order :math:`10^9` more power than the origin does, which
    is arithmetically correct for a point emitter and physically
    meaningless -- a real receiver that close is in the reactive near
    field, where a scalar ``1/r`` model does not apply at all. The warning
    fires well before that, at ten wavelengths, because the usual cause is
    a units slip (kilometres entered as meters) rather than a deliberate
    choice.
    """
    distances_m = path_lengths_m(position_enu_m, antenna_positions_enu_m)
    wavelength_m = SPEED_OF_LIGHT_M_S / float(np.mean(freq_hz))
    threshold_m = MIN_SEPARATION_WAVELENGTHS * wavelength_m
    closest_m = float(distances_m.min())
    if closest_m < threshold_m:
        warnings.warn(
            f"a transmitter is {closest_m:.4g} m from the nearest antenna, closer "
            f"than {MIN_SEPARATION_WAVELENGTHS:g} wavelengths ({threshold_m:.4g} m) "
            "at the band center. The 1/r amplitude model is normalized at the array "
            "origin, so the per-antenna received power is now wildly non-uniform and "
            "the source's received_power_jy no longer describes what any antenna "
            "sees. Check the position units.",
            UserWarning,
            stacklevel=3,
        )


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


_QPSK_SYMBOLS = np.exp(1j * (np.pi / 4.0 + 0.5 * np.pi * np.arange(4))).astype(np.complex64)
"""numpy.ndarray: The four unit-modulus QPSK constellation points."""


def constant_envelope(
    rng: np.random.Generator, shape: tuple[int, ...], samples_per_symbol: int = 1
) -> np.ndarray:
    r"""Unit-power, unit-*modulus* phase-modulated samples, complex64.

    A random-symbol QPSK carrier: each sample is one of the four unit
    modulus points :math:`e^{i(\pi/4 + k\pi/2)}`, held for
    `samples_per_symbol` consecutive samples along the last axis. Every
    sample has ``|z| == 1`` exactly, so the instantaneous power is
    constant -- which is what distinguishes a real carrier from the
    Gaussian modulation of `circular_normal`, and what a spectral-kurtosis
    detector keys on.

    Parameters
    ----------
    rng : numpy.random.Generator
        Generator to draw the symbols from.
    shape : tuple of int
        Output shape; the last axis is time.
    samples_per_symbol : int, optional
        Number of consecutive time samples one symbol occupies, ``>= 1``.
        Default 1, i.e. a new symbol every sample, which fills the
        available bandwidth.

    Returns
    -------
    numpy.ndarray
        Complex64 array of shape `shape` with ``|z| == 1`` everywhere, so
        ``E|z|**2 == 1`` as for `circular_normal`.

    Notes
    -----
    Symbols are held rectangularly (a zero-order hold) rather than
    pulse-shaped. The approximation this makes is spectral, not
    statistical: a rectangular hold has ``sinc`` skirts, so the occupied
    bandwidth is only *approximately* the symbol rate, with a few per cent
    of the power outside it. The envelope -- the property that matters for
    a kurtosis detector -- is exactly constant either way, whereas a
    root-raised-cosine shaping filter would introduce envelope ripple of
    its own and would need a filter length and a roll-off to argue about.

    Examples
    --------
    >>> import numpy as np
    >>> samples = constant_envelope(np.random.default_rng(0), (2, 8), 4)
    >>> bool(np.allclose(np.abs(samples), 1.0))
    True
    >>> bool(np.all(samples[:, 0] == samples[:, 3]))   # one symbol, four samples
    True
    """
    samples_per_symbol = int(samples_per_symbol)
    if samples_per_symbol < 1:
        raise ValueError(f"samples_per_symbol must be >= 1, got {samples_per_symbol}")
    n_time = int(shape[-1])
    n_symbols = -(-n_time // samples_per_symbol)  # ceiling division
    symbol_shape = (*shape[:-1], n_symbols)
    symbols = _QPSK_SYMBOLS[rng.integers(0, _QPSK_SYMBOLS.size, size=symbol_shape)]
    if samples_per_symbol == 1:
        return symbols.astype(np.complex64, copy=False)
    held = np.repeat(symbols, samples_per_symbol, axis=-1)
    return held[..., :n_time].astype(np.complex64, copy=False)


# ----------------------------------------------------------------------
# Shared narrowband machinery
# ----------------------------------------------------------------------
def _normalize_envelope(envelope) -> dict | None:
    """Validate an on/off envelope specification.

    Parameters
    ----------
    envelope : None or mapping
        ``None`` for the i.i.d. ``duty_cycle`` frames, or a mapping
        ``{"type": "periodic", "period_s": p, "duty": d, "phase": q}``
        with ``phase`` optional (default 0, in cycles).

    Returns
    -------
    None or dict
        The validated specification as a plain dict, or ``None``.

    Raises
    ------
    ValueError
        If the type is unknown, a key is unexpected, ``period_s`` is not
        finite and positive, ``duty`` is outside ``[0, 1]``, or ``phase``
        is not finite.
    """
    if envelope is None:
        return None
    if not isinstance(envelope, dict):
        raise ValueError(
            f"envelope must be None or a mapping, got {type(envelope).__name__}. Use "
            '{"type": "periodic", "period_s": ..., "duty": ...} for clocked frames.'
        )
    spec = dict(envelope)
    kind = spec.pop("type", None)
    if kind != "periodic":
        raise ValueError(f"envelope type must be 'periodic', got {kind!r}")
    period_s = float(_to_value(spec.pop("period_s", 0.0), u.s))
    duty = float(spec.pop("duty", 1.0))
    phase = float(spec.pop("phase", 0.0))
    if spec:
        raise ValueError(f"unexpected keys in the envelope specification: {sorted(spec)}")
    if not np.isfinite(period_s) or not period_s > 0.0:
        raise ValueError(f"envelope period_s must be finite and > 0, got {period_s}")
    if not 0.0 <= duty <= 1.0:
        raise ValueError(f"envelope duty must be in [0, 1], got {duty}")
    if not np.isfinite(phase):
        raise ValueError(f"envelope phase must be finite, got {phase}")
    return {"type": "periodic", "period_s": period_s, "duty": duty, "phase": phase}


def _normalize_arrival(arrival):
    """Validate an impulsive arrival-process specification.

    Parameters
    ----------
    arrival : str or mapping
        ``"poisson"``, or a mapping ``{"type": "periodic", "rate_hz": r,
        "jitter_s": j}`` with ``jitter_s`` optional (default 0).

    Returns
    -------
    str or dict
        ``"poisson"`` unchanged, or the validated specification as a plain
        dict.

    Raises
    ------
    ValueError
        If the string is not ``"poisson"``, the mapping type is not
        ``"periodic"``, a key is unexpected, ``rate_hz`` is not finite and
        positive, or ``jitter_s`` is not finite and non-negative.
    """
    if isinstance(arrival, str):
        if arrival != "poisson":
            raise ValueError(
                f"arrival must be 'poisson' or a periodic specification, got {arrival!r}"
            )
        return "poisson"
    if not isinstance(arrival, dict):
        raise ValueError(f"arrival must be 'poisson' or a mapping, got {type(arrival).__name__}")
    spec = dict(arrival)
    kind = spec.pop("type", None)
    if kind != "periodic":
        raise ValueError(f"arrival type must be 'periodic', got {kind!r}")
    rate_hz = float(_to_value(spec.pop("rate_hz", 0.0), u.Hz))
    jitter_s = float(_to_value(spec.pop("jitter_s", 0.0), u.s))
    if spec:
        raise ValueError(f"unexpected keys in the arrival specification: {sorted(spec)}")
    if not np.isfinite(rate_hz) or not rate_hz > 0.0:
        raise ValueError(f"arrival rate_hz must be > 0 and finite, got {rate_hz}")
    if not np.isfinite(jitter_s) or jitter_s < 0.0:
        raise ValueError(f"arrival jitter_s must be >= 0 and finite, got {jitter_s}")
    return {"type": "periodic", "rate_hz": rate_hz, "jitter_s": jitter_s}


class _NarrowbandDevice(RFISource):
    """Shared synthesis for sources that emit into contiguous channel groups.

    Holds everything `NarrowbandTransmitter` and `CombTransmitter` have in
    common: the modulation choice, the on/off pattern, and the step that
    turns an occupied-channel power envelope into voltages. Not part of the
    public API -- subclass `RFISource` for a new source, or one of the two
    concrete classes to specialize them.

    Parameters
    ----------
    name : str
        Source label.
    coupling : None, array_like or mapping, optional
        Per-antenna coupling; see `resolve_coupling`.
    waveform : {"gaussian", "constant_envelope"}, optional
        Modulation of the emission. Default ``"gaussian"``.
    duty_cycle : float, optional
        Fraction of frames the device is on, drawn independently per frame.
        Default 1.0 (always on). Mutually exclusive with `envelope`.
    frame_duration_s : float or astropy.units.Quantity, optional
        Duration of one i.i.d. frame, seconds. Default 0.01.
    envelope : None or mapping, optional
        Clocked on/off pattern instead of i.i.d. frames; see
        `_normalize_envelope`. Default ``None``.

    Raises
    ------
    ValueError
        If `waveform` is unknown, if `duty_cycle` is outside ``[0, 1]``, if
        `frame_duration_s` is not positive, if `envelope` is invalid, or if
        both a non-trivial `duty_cycle` and an `envelope` are given.
    """

    def __init__(
        self,
        name: str,
        *,
        coupling=None,
        waveform: str = "gaussian",
        duty_cycle: float = 1.0,
        frame_duration_s=0.01,
        envelope=None,
    ) -> None:
        super().__init__(name, coupling=coupling)
        self.waveform = str(waveform)
        self.duty_cycle = float(duty_cycle)
        self.frame_duration_s = float(_to_value(frame_duration_s, u.s))
        self.envelope = _normalize_envelope(envelope)

        if self.waveform not in WAVEFORMS:
            raise ValueError(f"waveform must be one of {WAVEFORMS}, got {self.waveform!r}")
        if not 0.0 <= self.duty_cycle <= 1.0:
            raise ValueError(f"duty_cycle must be in [0, 1], got {self.duty_cycle}")
        if not np.isfinite(self.frame_duration_s) or self.frame_duration_s <= 0.0:
            raise ValueError(
                f"frame_duration_s must be finite and > 0, got {self.frame_duration_s}"
            )
        if self.envelope is not None and self.duty_cycle != 1.0:
            raise ValueError(
                "duty_cycle and envelope both describe when the source is on, and "
                "composing them would give an on-fraction that matches neither. Set "
                "the duty fraction inside the envelope specification instead."
            )

    def on_frames(self, ctx: BlockContext) -> np.ndarray:
        """Per-sample on/off state of the source within a block.

        Parameters
        ----------
        ctx : BlockContext
            Block context; its `rng` supplies the frame draws of the
            i.i.d. mode, and `BlockContext.index` places a clocked pattern
            within the observation.

        Returns
        -------
        numpy.ndarray
            Boolean array of shape ``(n_time,)``, True where the source is
            on.

        Notes
        -----
        The two modes differ in more than their statistics. i.i.d. frames
        are aligned to the start of *each block* and drawn from the block's
        generator, so the pattern neither continues nor repeats across a
        block boundary. A clocked (``"periodic"``) envelope is a function
        of elapsed time since the start of the observation, so its frames
        run continuously through the whole run -- which is what a
        transmitter driven by its own clock does, and what makes the
        period recoverable from data spanning several blocks.
        """
        if self.envelope is None:
            frame_index = np.floor_divide(
                ctx.sample_times_s, self.frame_duration_s, dtype=np.float64
            ).astype(np.int64)
            n_frames = int(frame_index.max()) + 1 if frame_index.size else 0
            # Always draw, so the generator stream does not depend on the
            # duty cycle value -- only on the number of frames in the block.
            draws = ctx.rng.random(n_frames)
            return draws[frame_index] < self.duty_cycle

        spec = self.envelope
        elapsed_s = ctx.index * ctx.duration_s + ctx.sample_times_s
        cycles = elapsed_s / spec["period_s"] - spec["phase"]
        return (cycles - np.floor(cycles)) < spec["duty"]

    def symbol_length_samples(self, ctx: BlockContext, bandwidth_hz: float) -> int:
        """Samples per constant-envelope symbol for an emission bandwidth.

        Parameters
        ----------
        ctx : BlockContext
            Block context; the post-channelization sample rate is
            ``ctx.chan_width_hz``.
        bandwidth_hz : float
            Occupied bandwidth of the emission, Hz. Zero means an
            unmodulated carrier.

        Returns
        -------
        int
            Number of consecutive samples one symbol occupies, ``>= 1``:
            ``round(chan_width / bandwidth)``, i.e. a symbol rate of about
            `bandwidth_hz`. One sample per symbol whenever the emission
            fills a channel or more, and the whole block -- an unmodulated
            carrier -- for zero bandwidth.

        Notes
        -----
        Each channel is synthesized at its own sample rate of
        ``chan_width_hz``, so a symbol rate above that cannot be
        represented in this channelized model: an emission spanning many
        channels is given one symbol per sample **per channel**, with
        independent symbol streams in each. A genuinely wideband
        constant-envelope signal is not constant-envelope *after*
        channelization -- filtering a unit-modulus signal down to one
        channel of its band leaves an amplitude that fluctuates -- so the
        constant-envelope model is faithful for emissions confined to about
        one channel (the carrier case, which is the case that matters for a
        kurtosis detector) and optimistic beyond that.
        """
        if bandwidth_hz <= 0.0:
            return max(int(ctx.n_time), 1)
        return max(int(round(ctx.chan_width_hz / bandwidth_hz)), 1)

    def draw_waveform(self, ctx: BlockContext, n_channels: int, bandwidth_hz: float) -> np.ndarray:
        """Unit-power modulation samples for `n_channels` occupied channels.

        Parameters
        ----------
        ctx : BlockContext
            Block context; its `rng` supplies the samples.
        n_channels : int
            Number of occupied channels.
        bandwidth_hz : float
            Occupied bandwidth, Hz -- only used to set the symbol rate of
            the constant-envelope waveform.

        Returns
        -------
        numpy.ndarray
            Complex64 array of shape ``(n_channels, ctx.n_time)`` with
            ``E|z|**2 == 1``.
        """
        if self.waveform == "constant_envelope":
            return constant_envelope(
                ctx.rng,
                (n_channels, ctx.n_time),
                self.symbol_length_samples(ctx, bandwidth_hz),
            )
        return circular_normal(ctx.rng, (n_channels, ctx.n_time))

    def add_emission(
        self,
        out: np.ndarray,
        ctx: BlockContext,
        phasors: np.ndarray,
        occupied: np.ndarray,
        envelope_jy: np.ndarray,
        bandwidth_hz: float,
    ) -> None:
        """Add one channel group's emission to a voltage array, in place.

        Parameters
        ----------
        out : numpy.ndarray
            Complex64 ``(n_antennas, n_chan, n_time)`` array to add into.
        ctx : BlockContext
            Block context.
        phasors : numpy.ndarray
            Complex64 ``(n_antennas, n_occupied)`` per-antenna gains for
            the occupied channels, from `coupled_phasors`.
        occupied : numpy.ndarray
            Boolean ``(n_chan,)`` mask of the occupied channels.
        envelope_jy : numpy.ndarray
            Float64 ``(n_chan, n_time)`` mean power per cell at the array
            origin, Jy.
        bandwidth_hz : float
            Occupied bandwidth of this emission, Hz.

        Notes
        -----
        One waveform realization is drawn and shared by every antenna --
        it is one emitted signal, and the per-antenna differences are
        entirely in `phasors`. That is what keeps the emission coherent
        across the array, and therefore visible in the visibilities rather
        than only in the autocorrelations.
        """
        waveform = self.draw_waveform(ctx, int(occupied.sum()), bandwidth_hz)
        waveform *= np.sqrt(envelope_jy[occupied], dtype=np.float64).astype(np.float32)
        out[:, occupied, :] += phasors[:, :, np.newaxis] * waveform[np.newaxis, :, :]


# ----------------------------------------------------------------------
# Concrete sources
# ----------------------------------------------------------------------
class NarrowbandTransmitter(_NarrowbandDevice):
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
        `duty_cycle`. Mutually exclusive with `envelope`.
    frame_duration_s : float or astropy.units.Quantity, optional
        Duration of one on/off frame, seconds. Default 0.01 (10 ms).
        Frames are aligned to the start of each block.
    envelope : None or mapping, optional
        Clocked on/off pattern in place of the i.i.d. frames:
        ``{"type": "periodic", "period_s": p, "duty": d, "phase": q}``,
        with ``phase`` in cycles (default 0). The transmitter is on for the
        first fraction ``d`` of every period ``p``, measured from the start
        of the observation, so the frames run continuously across blocks.
        Default ``None`` (i.i.d. frames).
    waveform : {"gaussian", "constant_envelope"}, optional
        Modulation. ``"gaussian"`` (default) is band-limited noise, right
        for a multi-carrier digital signal; ``"constant_envelope"`` is a
        phase-modulated carrier of constant instantaneous power, which is
        what a spectral-kurtosis detector can see (see the module
        docstring). Both are coherent across the array.
    coupling : None, array_like or mapping, optional
        Per-antenna linear amplitude coupling; see `resolve_coupling`.
        Default ``None`` (uniform).
    name : str, optional
        Label for the source. Default ``"narrowband"``.

    Raises
    ------
    ValueError
        If `center_freq_hz`, `bandwidth_hz` or `received_power_jy` is
        non-finite, `bandwidth_hz` or `received_power_jy` is negative,
        `duty_cycle` is outside ``[0, 1]``, `frame_duration_s` is not
        finite and positive, `waveform` or `coupling` or `envelope` is
        invalid, or both `duty_cycle` and `envelope` are given.

    Notes
    -----
    The emission is confined to the occupied channels *exactly*: this
    package assumes a perfect channelizer, so there is no spectral
    leakage into neighbouring channels. Real polyphase filter banks leak,
    and an excision algorithm tuned on this simulator will be optimistic
    about how sharply interference is confined until a filter-bank model
    is added.

    The transmitter is always visible: it is specified by an ENU position
    with an implied line of sight, and **no terrain shadowing, horizon cut
    or diffraction is modelled**. A position placed below the horizontal
    plane through the array origin still transmits at full strength. Place
    ground-based transmitters where the array can actually see them, or
    lower `received_power_jy` to stand in for an obstructed path.

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
        envelope=None,
        waveform: str = "gaussian",
        coupling=None,
        name: str = "narrowband",
    ) -> None:
        super().__init__(
            name,
            coupling=coupling,
            waveform=waveform,
            duty_cycle=duty_cycle,
            frame_duration_s=frame_duration_s,
            envelope=envelope,
        )
        self.position_enu_m = np.asarray(_to_value(position_enu_m, u.m), dtype=np.float64).reshape(
            3
        )
        self.center_freq_hz = float(_to_value(center_freq_hz, u.Hz))
        self.bandwidth_hz = float(_to_value(bandwidth_hz, u.Hz))
        self.received_power_jy = float(_to_value(received_power_jy, u.Jy))

        if not np.isfinite(self.center_freq_hz):
            raise ValueError(f"center_freq_hz must be finite, got {self.center_freq_hz}")
        if not np.isfinite(self.bandwidth_hz) or self.bandwidth_hz < 0.0:
            raise ValueError(f"bandwidth_hz must be finite and >= 0, got {self.bandwidth_hz}")
        if not np.isfinite(self.received_power_jy) or self.received_power_jy < 0.0:
            raise ValueError(
                f"received_power_jy must be finite and >= 0, got {self.received_power_jy}"
            )

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
        phasors = self.coupled_phasors(
            self.position_enu_m, ctx.antenna_positions_enu_m, ctx.freq_hz[occupied]
        )
        voltages = np.zeros((ctx.n_antennas, ctx.n_chan, ctx.n_time), dtype=np.complex64)
        self.add_emission(voltages, ctx, phasors, occupied, envelope, self.bandwidth_hz)

        return voltages, occupancy_mask(envelope)


class ImpulsiveBroadband(RFISource):
    """Short broadband bursts from a fixed direction: arcing, sparks, lightning.

    Events arrive as a Poisson process, each occupying a small number of
    consecutive time samples and the whole simulated band. Event powers
    are drawn from a bounded power law, so most bursts are weak and a few
    are very strong -- the heavy tail is the part that matters for
    excision, since a weak burst hides in the noise and a strong one
    saturates a whole integration.

    Not every impulsive source is Poisson, though. Arcing driven by the
    mains fires in a train locked to the supply frequency, twice per cycle,
    and that regularity is itself a detection handle; ``arrival`` switches
    the source to such a clocked train.

    Parameters
    ----------
    rate_hz : float or astropy.units.Quantity, optional
        Mean event rate, events per second, for the default Poisson
        arrivals. The number of events in a block is Poisson with mean
        ``rate_hz * block_duration_s``, so blocks are statistically
        independent and correlate only through this rate. Required for
        Poisson arrivals and rejected for periodic ones, whose rate lives
        in the `arrival` specification.
    arrival : {"poisson"} or mapping, optional
        Arrival process. ``"poisson"`` (default) uses `rate_hz` above. A
        mapping ``{"type": "periodic", "rate_hz": r, "jitter_s": j}``
        instead fires a regular train at ``r`` pulses per second, phased
        from the start of the observation and continuous across blocks,
        with each pulse displaced by an independent uniform draw in
        ``[-j, +j]`` seconds (``j`` defaults to 0, an exactly regular
        train).
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
    coupling : None, array_like or mapping, optional
        Per-antenna linear amplitude coupling; see `resolve_coupling`.
        Default ``None`` (uniform).
    name : str, optional
        Label for the source. Default ``"impulsive"``.

    Raises
    ------
    ValueError
        If any rate, power or width is non-finite or non-positive, if
        `power_law_index` is not greater than 1, if `max_power_ratio`
        is less than 1, if `arrival` is invalid, or if `rate_hz` and a
        periodic `arrival` are both given (or neither).

    Notes
    -----
    Like `NarrowbandTransmitter`, the source is specified by an ENU
    position with an implied line of sight: **no terrain shadowing,
    horizon cut or diffraction is modelled**, so a position below the
    array's horizontal plane still emits at full strength.

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
        rate_hz=None,
        received_power_jy=1000.0,
        *,
        arrival="poisson",
        position_enu_m=None,
        power_law_index: float = 2.0,
        max_power_ratio: float = 30.0,
        pulse_width_samples: int = 1,
        coupling=None,
        name: str = "impulsive",
    ) -> None:
        super().__init__(name, coupling=coupling)
        self.arrival = _normalize_arrival(arrival)
        if self.arrival == "poisson":
            if rate_hz is None:
                raise ValueError("Poisson arrivals need a rate_hz")
            self.rate_hz = float(_to_value(rate_hz, u.Hz))
        else:
            if rate_hz is not None:
                raise ValueError(
                    "rate_hz and a periodic arrival both set the event rate; give the "
                    "rate inside the arrival specification only"
                )
            self.rate_hz = self.arrival["rate_hz"]
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

        if not np.isfinite(self.rate_hz) or self.rate_hz < 0.0:
            raise ValueError(f"rate_hz must be finite and >= 0, got {self.rate_hz}")
        if not np.isfinite(self.received_power_jy) or self.received_power_jy < 0.0:
            raise ValueError(
                f"received_power_jy must be finite and >= 0, got {self.received_power_jy}"
            )
        if not np.isfinite(self.power_law_index) or self.power_law_index <= 1.0:
            raise ValueError(f"power_law_index must be finite and > 1, got {self.power_law_index}")
        if not np.isfinite(self.max_power_ratio) or self.max_power_ratio < 1.0:
            raise ValueError(f"max_power_ratio must be finite and >= 1, got {self.max_power_ratio}")
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

        For a periodic `arrival` the event *times* are deterministic --
        pulse ``k`` at ``k / rate_hz`` seconds after the start of the
        observation, so the train runs continuously across blocks -- and
        only the jitter and the powers are drawn. Pulses whose jitter
        carries them out of this block are dropped; the neighbouring block
        picks them up, because it considers the same pulse indices.
        """
        if self.arrival == "poisson":
            expected = self.rate_hz * ctx.duration_s
            n_events = int(ctx.rng.poisson(expected))
            start_samples = ctx.rng.integers(0, ctx.n_time, size=n_events)
        else:
            start_samples = self._periodic_start_samples(ctx)

        uniforms = ctx.rng.random(start_samples.size)
        exponent = 1.0 - self.power_law_index
        if self.max_power_ratio == 1.0:
            ratios = np.ones(start_samples.size, dtype=np.float64)
        else:
            ratios = (1.0 + uniforms * (self.max_power_ratio**exponent - 1.0)) ** (1.0 / exponent)
        return start_samples, self.received_power_jy * ratios

    def _periodic_start_samples(self, ctx: BlockContext) -> np.ndarray:
        """First-sample indices of a clocked pulse train within one block.

        Parameters
        ----------
        ctx : BlockContext
            Block context; its `rng` supplies the jitter draws.

        Returns
        -------
        numpy.ndarray
            Int64 array of first-sample indices, ascending in nominal
            pulse order, of the pulses that land inside this block.
        """
        rate_hz = self.arrival["rate_hz"]
        jitter_s = self.arrival["jitter_s"]
        block_start_s = ctx.index * ctx.duration_s
        block_end_s = block_start_s + ctx.duration_s

        # Every pulse index whose nominal time could reach into this block
        # once jittered. Considering the same indices in the neighbouring
        # block is what stops a jittered pulse from being counted twice.
        first = int(np.ceil((block_start_s - jitter_s) * rate_hz))
        last = int(np.floor((block_end_s + jitter_s) * rate_hz))
        indices = np.arange(first, last + 1, dtype=np.int64)
        if indices.size == 0:
            return np.zeros(0, dtype=np.int64)

        # Always draw, so the generator stream depends on the pulse count
        # and not on the jitter value.
        offsets_s = ctx.rng.uniform(-1.0, 1.0, size=indices.size) * jitter_s
        times_s = indices / rate_hz + offsets_s
        inside = (times_s >= block_start_s) & (times_s < block_end_s)
        samples = np.floor((times_s[inside] - block_start_s) / ctx.sample_period_s)
        return np.clip(samples, 0, max(ctx.n_time - 1, 0)).astype(np.int64)

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

        phasors = self.coupled_phasors(
            self.position_enu_m, ctx.antenna_positions_enu_m, ctx.freq_hz
        )
        voltages[:, :, active] = phasors[:, :, np.newaxis] * waveform[np.newaxis, :, :]
        return voltages, mask


class CombTransmitter(_NarrowbandDevice):
    """One device emitting a comb of harmonics of a single fundamental.

    A nonlinearity driven hard -- a switching supply, a failing connector,
    a digital clock radiating through a broken shield -- does not emit one
    line but a series of them, at integer multiples of one fundamental
    frequency, and observed combs can span hundreds of megahertz with
    several harmonics inside a single receiver band. Modelling them as
    independent narrowband transmitters would get the spectrum right and
    the *structure* wrong: the harmonics of one device share its position,
    its per-antenna coupling and its on/off pattern exactly, and an
    excision algorithm can exploit that -- lines that switch on together
    and share a coupling pattern are one device, however far apart in
    frequency they sit.

    Parameters
    ----------
    position_enu_m : array_like
        Device position, shape ``(3,)``, local ENU meters relative to the
        array origin.
    fundamental_hz : float or astropy.units.Quantity
        Fundamental frequency, Hz. The harmonics are at integer multiples
        of it, so it may well lie far below the simulated band.
    harmonic_numbers : sequence of int
        Which harmonics the device emits, as multiples of the fundamental
        (``1`` is the fundamental itself). Sorted ascending and required to
        be unique. Harmonics outside the simulated band are silently
        skipped -- see `in_band_harmonics` and `skipped_harmonics` -- since
        a real comb extends past any one receiver band and clipping it to
        the band is not a configuration error.
    received_powers_jy : float, astropy.units.Quantity or array_like, optional
        Power received at the array origin per harmonic while the device is
        on, in janskys: either one value for every harmonic, or one per
        entry of `harmonic_numbers` (in the *sorted* order). Default 100.0.
    bandwidth_hz : float or astropy.units.Quantity, optional
        Full occupied bandwidth of **each** harmonic, Hz. Default 0.0, a
        set of pure lines, each landing in the single nearest channel.
    duty_cycle : float, optional
        Fraction of i.i.d. frames the whole device is on. Default 1.0.
        Mutually exclusive with `envelope`.
    frame_duration_s : float or astropy.units.Quantity, optional
        Duration of one i.i.d. frame, seconds. Default 0.01.
    envelope : None or mapping, optional
        Clocked on/off pattern for the whole device; see
        `NarrowbandTransmitter`. Default ``None``.
    waveform : {"gaussian", "constant_envelope"}, optional
        Modulation of every harmonic. Default ``"gaussian"``.
    coupling : None, array_like or mapping, optional
        Per-antenna linear amplitude coupling of the device, shared by
        every harmonic; see `resolve_coupling`. Default ``None``.
    name : str, optional
        Label for the device. Default ``"comb"``.

    Raises
    ------
    ValueError
        If `fundamental_hz` or `bandwidth_hz` is non-finite, if
        `fundamental_hz` is not positive, if `harmonic_numbers` is
        empty, non-positive or repeated, if `received_powers_jy` has the
        wrong length, a non-finite entry or a negative entry, or if
        `bandwidth_hz` is negative. `contribution` raises if *no* harmonic
        reaches the simulated band, which is a configuration error rather
        than a clipped comb.

    Notes
    -----
    What the harmonics share and what they do not: position, coupling,
    on/off pattern, waveform family and bandwidth are the device's; the
    modulation *realization* of each harmonic is drawn independently, and
    so is its power. Independent realizations are the honest choice within
    this channelized model -- the code never sees the wideband time series
    in which a real nonlinearity's harmonics are phase-locked to its
    fundamental -- and it is the conservative one: an algorithm cannot
    learn to key on a phase relationship the simulator invents.

    Examples
    --------
    >>> comb = CombTransmitter(
    ...     position_enu_m=enu_from_horizontal(200.0, 1.0, 800.0),
    ...     fundamental_hz=2.81e8,
    ...     harmonic_numbers=(4, 5, 6),
    ...     received_powers_jy=(300.0, 200.0, 120.0),
    ... )
    >>> comb.harmonic_names
    ('comb[4]', 'comb[5]', 'comb[6]')
    """

    def __init__(
        self,
        position_enu_m,
        fundamental_hz,
        harmonic_numbers,
        received_powers_jy=100.0,
        bandwidth_hz=0.0,
        *,
        duty_cycle: float = 1.0,
        frame_duration_s=0.01,
        envelope=None,
        waveform: str = "gaussian",
        coupling=None,
        name: str = "comb",
    ) -> None:
        super().__init__(
            name,
            coupling=coupling,
            waveform=waveform,
            duty_cycle=duty_cycle,
            frame_duration_s=frame_duration_s,
            envelope=envelope,
        )
        self.position_enu_m = np.asarray(_to_value(position_enu_m, u.m), dtype=np.float64).reshape(
            3
        )
        self.fundamental_hz = float(_to_value(fundamental_hz, u.Hz))
        numbers = np.asarray(harmonic_numbers, dtype=np.int64).reshape(-1)
        self.bandwidth_hz = float(_to_value(bandwidth_hz, u.Hz))

        if not np.isfinite(self.fundamental_hz) or not self.fundamental_hz > 0.0:
            raise ValueError(f"fundamental_hz must be finite and > 0, got {self.fundamental_hz}")
        if numbers.size == 0:
            raise ValueError("harmonic_numbers must name at least one harmonic")
        if np.any(numbers < 1):
            raise ValueError(f"harmonic_numbers must all be >= 1, got {numbers.tolist()}")
        if np.unique(numbers).size != numbers.size:
            raise ValueError(f"harmonic_numbers must be unique, got {numbers.tolist()}")
        if not np.isfinite(self.bandwidth_hz) or self.bandwidth_hz < 0.0:
            raise ValueError(f"bandwidth_hz must be finite and >= 0, got {self.bandwidth_hz}")

        order = np.argsort(numbers)
        self.harmonic_numbers = numbers[order]

        powers = np.atleast_1d(np.asarray(_to_value(received_powers_jy, u.Jy), dtype=np.float64))
        if powers.size == 1:
            powers = np.repeat(powers, self.harmonic_numbers.size)
        elif powers.size != numbers.size:
            raise ValueError(
                f"received_powers_jy has {powers.size} entries but there are "
                f"{numbers.size} harmonics; pass one value or one per harmonic"
            )
        else:
            powers = powers[order]
        if not np.all(np.isfinite(powers)):
            raise ValueError("received_powers_jy must all be finite")
        if np.any(powers < 0.0):
            raise ValueError("received_powers_jy must all be >= 0")
        self.received_powers_jy = powers

    @property
    def n_harmonics(self) -> int:
        """int: Number of harmonics the device is configured to emit."""
        return int(self.harmonic_numbers.size)

    @property
    def harmonic_freqs_hz(self) -> np.ndarray:
        """numpy.ndarray: Center frequency of each harmonic, Hz, ascending.

        Shape ``(n_harmonics,)``, equal to ``harmonic_numbers *
        fundamental_hz`` -- including harmonics that fall outside the
        simulated band.
        """
        return self.harmonic_numbers * self.fundamental_hz

    @property
    def harmonic_names(self) -> tuple[str, ...]:
        """tuple of str: Ground-truth label of each harmonic.

        ``"<name>[n]"`` for harmonic number ``n``, e.g. ``"comb[7]"``, in
        the order of `harmonic_numbers`. The device contributes a *single*
        entry to `rfi_simulator.voltages.VoltageBlock.rfi_mask`, under its
        own `name`, holding the union over harmonics -- one source, one
        mask, as for every other source. These per-harmonic labels go with
        `harmonic_masks` for experiments that want the comb resolved line
        by line.
        """
        return tuple(f"{self.name}[{int(n)}]" for n in self.harmonic_numbers)

    def harmonic_channels(self, freq_hz: np.ndarray) -> np.ndarray:
        """Channels occupied by each harmonic.

        Parameters
        ----------
        freq_hz : numpy.ndarray
            Shape ``(n_chan,)`` RF channel center frequencies, Hz.

        Returns
        -------
        numpy.ndarray
            Boolean array of shape ``(n_harmonics, n_chan)``. A row is all
            False for a harmonic that misses the band entirely.
        """
        freq_hz = np.asarray(freq_hz, dtype=np.float64)
        return np.stack(
            [
                channels_within(freq_hz, float(center_hz), self.bandwidth_hz)
                for center_hz in self.harmonic_freqs_hz
            ]
        )

    def _in_band_flags(self, freq_hz: np.ndarray, chan_width_hz: float) -> np.ndarray:
        """Boolean ``(n_harmonics,)`` mask of the harmonics that synthesize."""
        occupied = self.harmonic_channels(freq_hz)
        overlaps = np.array(
            [
                band_overlaps(float(center_hz), self.bandwidth_hz, freq_hz, chan_width_hz)
                for center_hz in self.harmonic_freqs_hz
            ]
        )
        return overlaps & occupied.any(axis=1)

    def in_band_harmonics(self, freq_hz: np.ndarray, chan_width_hz: float) -> np.ndarray:
        """Harmonic numbers that fall inside a simulated band.

        Parameters
        ----------
        freq_hz : numpy.ndarray
            Shape ``(n_chan,)`` RF channel center frequencies, Hz.
        chan_width_hz : float
            Channel width, Hz.

        Returns
        -------
        numpy.ndarray
            Int64 array of the harmonic numbers that occupy at least one
            channel of the band -- exactly the ones `contribution`
            synthesizes.
        """
        return self.harmonic_numbers[self._in_band_flags(freq_hz, float(chan_width_hz))]

    def skipped_harmonics(self, freq_hz: np.ndarray, chan_width_hz: float) -> np.ndarray:
        """Harmonic numbers that fall outside a simulated band.

        Parameters
        ----------
        freq_hz : numpy.ndarray
            Shape ``(n_chan,)`` RF channel center frequencies, Hz.
        chan_width_hz : float
            Channel width, Hz.

        Returns
        -------
        numpy.ndarray
            Int64 array of the harmonic numbers that contribute nothing:
            the complement of `in_band_harmonics`. Recorded rather than
            hidden, so a run can report which lines of the comb it left
            out.
        """
        return self.harmonic_numbers[~self._in_band_flags(freq_hz, float(chan_width_hz))]

    def _envelopes_jy(self, ctx: BlockContext, on: np.ndarray) -> np.ndarray:
        """Mean-power envelope of each harmonic, ``(n_harmonics, n_chan, n_time)``.

        Parameters
        ----------
        ctx : BlockContext
            Block geometry and timing.
        on : numpy.ndarray
            Boolean ``(n_time,)`` on/off state of the device.

        Returns
        -------
        numpy.ndarray
            Float64 powers at the array origin, Jy, zero outside each
            harmonic's occupied channels and outside its on samples.
        """
        occupied = self.harmonic_channels(ctx.freq_hz)
        in_band = self._in_band_flags(ctx.freq_hz, ctx.chan_width_hz)
        envelopes = np.zeros((self.n_harmonics, ctx.n_chan, ctx.n_time), dtype=np.float64)
        for i_harmonic in np.flatnonzero(in_band):
            channels = occupied[i_harmonic]
            n_channels = int(channels.sum())
            envelopes[i_harmonic][np.ix_(channels, on)] = (
                self.received_powers_jy[i_harmonic] / n_channels
            )
        return envelopes

    def harmonic_masks(self, ctx: BlockContext) -> np.ndarray:
        """Per-harmonic ground-truth occupancy labels for one block.

        Parameters
        ----------
        ctx : BlockContext
            Block geometry, timing and generator. The generator is used
            only for an i.i.d. on/off pattern, so call this with the same
            context the block was synthesized with (or a clocked
            `envelope`) to get the labels that go with those voltages.

        Returns
        -------
        numpy.ndarray
            Boolean array of shape ``(n_harmonics, n_chan, n_time)``, in
            the order of `harmonic_numbers` and `harmonic_names`. All False
            for a harmonic outside the band.

        Notes
        -----
        Each harmonic is thresholded against **its own** peak power, so a
        weak line is labelled even next to a strong one. The single mask
        `contribution` returns is thresholded against the device's peak
        instead, following the package-wide convention that a source's
        labels are relative to that source's brightest cell in the block;
        the two therefore differ for a comb whose harmonics span more than
        a factor ``1 / OCCUPANCY_THRESHOLD`` in power.
        """
        envelopes = self._envelopes_jy(ctx, self.on_frames(ctx))
        return np.stack([occupancy_mask(envelope) for envelope in envelopes])

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
            root-Jy, summed over the in-band harmonics.
        mask : numpy.ndarray
            Boolean ``(n_chan, n_time)`` occupancy labels of the device as
            a whole, i.e. the union over its harmonics. `harmonic_masks`
            resolves it line by line.

        Raises
        ------
        ValueError
            If no harmonic reaches the simulated band at all. Individual
            out-of-band harmonics are skipped silently -- a comb is
            expected to extend past the band -- but a comb that is entirely
            outside it emits nothing, which is a configuration error.

        Notes
        -----
        The near-field geometry is evaluated **once** for the whole band,
        because it is one device at one position: the phasors are then
        sliced per harmonic. That is both cheaper than one call per
        harmonic and the reason a comb's harmonics share a coupling
        pattern exactly.
        """
        in_band = self._in_band_flags(ctx.freq_hz, ctx.chan_width_hz)
        if not in_band.any():
            raise ValueError(
                f"source {self.name!r} emits harmonics "
                f"{self.harmonic_numbers.tolist()} of {self.fundamental_hz / 1e6:.3f} MHz, "
                f"none of which reach the simulated band {ctx.freq_hz[0] / 1e6:.3f}-"
                f"{ctx.freq_hz[-1] / 1e6:.3f} MHz. Re-center the simulated band on one "
                "of the harmonics, or re-tune the fundamental."
            )

        on = self.on_frames(ctx)
        envelopes = self._envelopes_jy(ctx, on)
        occupied = self.harmonic_channels(ctx.freq_hz)

        # One device, one position: the phasors are computed once for the
        # whole band and sliced per harmonic, so every line of the comb
        # carries the identical per-antenna coupling.
        phasors = self.coupled_phasors(
            self.position_enu_m, ctx.antenna_positions_enu_m, ctx.freq_hz
        )
        voltages = np.zeros((ctx.n_antennas, ctx.n_chan, ctx.n_time), dtype=np.complex64)
        for i_harmonic in np.flatnonzero(in_band):
            channels = occupied[i_harmonic]
            self.add_emission(
                voltages,
                ctx,
                phasors[:, channels],
                channels,
                envelopes[i_harmonic],
                self.bandwidth_hz,
            )

        return voltages, occupancy_mask(envelopes.sum(axis=0))
