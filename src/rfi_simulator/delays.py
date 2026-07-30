r"""Geometric delays -- the module that owns the delay sign convention.

Binding convention: for a plane wave arriving
from sky direction :math:`\hat{s}` (a unit vector pointing *from the array
towards the source*), the antenna at local ENU position :math:`\mathbf{r}`
(meters, relative to the array origin) sees the signal with geometric delay

.. math::

    \tau = -\frac{\mathbf{r} \cdot \hat{s}}{c}

relative to the array origin, and its voltage is :math:`v(t - \tau)`.

Read the sign like this: an antenna displaced *towards* the source
(:math:`\mathbf{r} \cdot \hat{s} > 0`) is closer to the source, so it sees
the wavefront *earlier*, so its delay is negative. An antenna at the array
origin, or one on a baseline perpendicular to the source direction, has
zero delay.

In the frequency domain this makes the per-antenna voltage spectrum

.. math::

    v(f) = s(f)\, e^{-2\pi i f \tau},

with ``f`` the **RF** (sky) frequency of the channel -- not a baseband
offset. Combined with the visibility definition
:math:`V_{ij} = \langle v_i v_j^* \rangle` (conjugate on the *second*
antenna, see `rfi_simulator.correlator`), this is what makes a source at
:math:`+l` appear at :math:`+l` in the dirty image.

Delays are evaluated per data block with astropy, so Earth rotation over
the observation is honored; there is deliberately no single frozen delay
for a whole observation anywhere in this package.
"""

from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.constants import c as _c
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from rfi_simulator.array_config import ArrayConfig
from rfi_simulator.sky import radec_from_lm

__all__ = [
    "SPEED_OF_LIGHT_M_S",
    "earth_location",
    "enu_unit_vector",
    "geometric_delays_s",
    "lm_basis_enu",
    "source_unit_vectors_enu",
    "zenith_coord",
]

SPEED_OF_LIGHT_M_S = float(_c.to_value(u.m / u.s))
"""float: Speed of light in vacuum, m/s."""


def earth_location(array: ArrayConfig) -> EarthLocation:
    """Geodetic `~astropy.coordinates.EarthLocation` of an array origin.

    Parameters
    ----------
    array : ArrayConfig
        Array whose origin latitude/longitude/height is wanted.

    Returns
    -------
    astropy.coordinates.EarthLocation
        Location of the array origin (the point that ENU antenna
        positions are measured from).
    """
    return EarthLocation.from_geodetic(
        lon=array.longitude_deg * u.deg,
        lat=array.latitude_deg * u.deg,
        height=array.height_m * u.m,
    )


def enu_unit_vector(alt_rad, az_rad) -> np.ndarray:
    """Convert horizontal (altitude, azimuth) angles to an ENU unit vector.

    Parameters
    ----------
    alt_rad : array_like
        Altitude above the horizon, radians.
    az_rad : array_like
        Azimuth in radians, measured from North through East (the astropy
        `~astropy.coordinates.AltAz` convention).

    Returns
    -------
    numpy.ndarray
        Shape ``broadcast(alt_rad, az_rad).shape + (3,)`` float64 array of
        unit vectors in local East-North-Up coordinates.
    """
    alt_rad = np.asarray(alt_rad, dtype=np.float64)
    az_rad = np.asarray(az_rad, dtype=np.float64)
    cos_alt = np.cos(alt_rad)
    east = cos_alt * np.sin(az_rad)
    north = cos_alt * np.cos(az_rad)
    up = np.sin(alt_rad)
    return np.stack(np.broadcast_arrays(east, north, up), axis=-1)


def source_unit_vectors_enu(coord: SkyCoord, time: Time, location: EarthLocation) -> np.ndarray:
    """Unit vectors towards sky coordinates, in the local ENU frame.

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
        Sky coordinate(s). Shapes are broadcast against `time`.
    time : astropy.time.Time
        Observation time(s) (UTC). Shapes are broadcast against `coord`.
    location : astropy.coordinates.EarthLocation
        Observer location (the array origin).

    Returns
    -------
    numpy.ndarray
        Shape ``broadcast(coord, time).shape + (3,)`` float64 array of ENU
        unit vectors pointing from the array towards each coordinate.

    Notes
    -----
    The transform is `~astropy.coordinates.AltAz` with no pressure set, so
    atmospheric refraction is *not* applied (annual aberration and
    precession/nutation are). That is the right choice for a clean
    geometric simulator; a refraction term would belong in a later
    propagation stage.
    """
    altaz = coord.transform_to(AltAz(obstime=time, location=location))
    return enu_unit_vector(altaz.alt.to_value(u.rad), altaz.az.to_value(u.rad))


def geometric_delays_s(positions_enu_m: np.ndarray, s_hat_enu: np.ndarray) -> np.ndarray:
    r"""Geometric delay of each antenna for a given source direction.

    Implements the binding convention :math:`\tau = -(\mathbf{r} \cdot
    \hat{s}) / c` (see the module docstring).

    Parameters
    ----------
    positions_enu_m : numpy.ndarray
        Antenna positions, shape ``(n_antennas, 3)``, local ENU meters
        relative to the array origin.
    s_hat_enu : numpy.ndarray
        Unit vector(s) towards the source in the ENU frame, shape
        ``(..., 3)``.

    Returns
    -------
    numpy.ndarray
        Shape ``s_hat_enu.shape[:-1] + (n_antennas,)`` float64 array of
        delays in seconds. Negative for antennas displaced towards the
        source (they see the wavefront early).
    """
    positions_enu_m = np.asarray(positions_enu_m, dtype=np.float64)
    s_hat_enu = np.asarray(s_hat_enu, dtype=np.float64)
    if positions_enu_m.ndim != 2 or positions_enu_m.shape[1] != 3:
        raise ValueError(
            f"positions_enu_m must have shape (n_antennas, 3), got {positions_enu_m.shape}"
        )
    if s_hat_enu.shape[-1] != 3:
        raise ValueError(f"s_hat_enu must have shape (..., 3), got {s_hat_enu.shape}")

    return -np.einsum("...j,aj->...a", s_hat_enu, positions_enu_m) / SPEED_OF_LIGHT_M_S


def lm_basis_enu(
    phase_center: SkyCoord,
    time: Time,
    location: EarthLocation,
    step_rad: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local ENU triad of the phase center and its (l, m) tangent directions.

    Parameters
    ----------
    phase_center : astropy.coordinates.SkyCoord
        Scalar phase-center coordinate.
    time : astropy.time.Time
        Scalar or 1-D array of observation times (UTC).
    location : astropy.coordinates.EarthLocation
        Observer location (the array origin).
    step_rad : float, optional
        Half-step in direction cosine used for the central-difference
        evaluation of the tangent directions. Default ``1e-5``, which is
        small enough for a ``1e-11``-level truncation error and large
        enough to stay far from float64 round-off.

    Returns
    -------
    s0_hat_enu : numpy.ndarray
        Shape ``time.shape + (3,)`` unit vector(s) towards the phase center.
    e_l_enu : numpy.ndarray
        Shape ``time.shape + (3,)`` unit vector(s) along increasing ``l``.
    e_m_enu : numpy.ndarray
        Shape ``time.shape + (3,)`` unit vector(s) along increasing ``m``.

    Notes
    -----
    The triad is obtained by differencing the *same* ICRS-to-ENU transform
    that positions the sources, so ``l = s_hat . e_l`` computed with these
    vectors agrees with `rfi_simulator.sky.lm_from_radec` by construction.
    """
    scalar_time = time.isscalar
    times = time.reshape(1) if scalar_time else time

    offsets = np.array(
        [
            [0.0, 0.0],
            [step_rad, 0.0],
            [-step_rad, 0.0],
            [0.0, step_rad],
            [0.0, -step_rad],
        ]
    )
    probe_coords = radec_from_lm(phase_center, offsets)

    # One vectorized astropy transform: (5, 1) coords against (n_times,) times.
    vectors = source_unit_vectors_enu(probe_coords.reshape(5, 1), times, location)

    s0_hat = vectors[0]
    e_l = (vectors[1] - vectors[2]) / (2.0 * step_rad)
    e_m = (vectors[3] - vectors[4]) / (2.0 * step_rad)
    e_l = e_l / np.linalg.norm(e_l, axis=-1, keepdims=True)
    e_m = e_m / np.linalg.norm(e_m, axis=-1, keepdims=True)

    if scalar_time:
        return s0_hat[0], e_l[0], e_m[0]
    return s0_hat, e_l, e_m


def zenith_coord(location: EarthLocation, time: Time) -> SkyCoord:
    """Sky coordinate of the local zenith.

    Parameters
    ----------
    location : astropy.coordinates.EarthLocation
        Observer location.
    time : astropy.time.Time
        Scalar observation time (UTC).

    Returns
    -------
    astropy.coordinates.SkyCoord
        The ICRS coordinate that is at the zenith of `location` at `time`.

    Notes
    -----
    Handy as a default phase center: for a flat array (all ``up = 0``) a
    zenith phase center makes the ``w`` term identically zero, so the
    tangent-plane (``l, m``) imaging in `rfi_simulator.imaging` is exact
    rather than approximate.
    """
    zenith = AltAz(alt=90.0 * u.deg, az=0.0 * u.deg, obstime=time, location=location)
    return SkyCoord(zenith).icrs
