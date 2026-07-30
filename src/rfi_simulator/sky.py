"""Sky model: point sources and direction-cosine (l, m) geometry.

Conventions (see ``docs/design_stage2.md``): the phase center is an ICRS
(RA, Dec) coordinate, and source positions are expressed either as ICRS
coordinates or as direction cosines ``(l, m)`` relative to that phase
center.

The direction cosines are the standard orthographic ("SIN") ones. Writing
the phase center as the unit vector :math:`\\hat{s}_0` and building the
orthonormal triad

* :math:`\\hat{e}_l` -- unit vector towards increasing RA at the phase center,
* :math:`\\hat{e}_m` -- unit vector towards increasing Dec at the phase center,
* :math:`\\hat{s}_0` -- the phase center itself,

a source direction :math:`\\hat{s}` has

.. math::

    l = \\hat{s} \\cdot \\hat{e}_l, \\quad
    m = \\hat{s} \\cdot \\hat{e}_m, \\quad
    n = \\hat{s} \\cdot \\hat{s}_0 = \\sqrt{1 - l^2 - m^2}.

Because this triad is carried from ICRS to the local ENU frame by the same
rotation that carries :math:`\\hat{s}`, ``(l, m)`` of a fixed source does
not change as the Earth rotates -- which is why the imaging code can
integrate coherently over a whole snippet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from rfi_simulator.array_config import _to_value

__all__ = ["PointSource", "lm_from_radec", "radec_from_lm"]


@dataclass
class PointSource:
    """An unresolved, unpolarized celestial point source.

    Parameters
    ----------
    flux_jy : float or astropy.units.Quantity
        Flux density of the source in janskys (or a `Quantity` convertible
        to Jy). Assumed flat across the simulated band.
    coord : astropy.coordinates.SkyCoord
        Scalar sky coordinate of the source (ICRS is assumed throughout).
    name : str, optional
        Human-readable label. Defaults to ``""``.

    Attributes
    ----------
    flux_jy : float
        Flux density in Jy.
    coord : astropy.coordinates.SkyCoord
        Source position.
    name : str
        Source label.

    Notes
    -----
    In the simulator the source's voltage spectrum is drawn as circular
    complex Gaussian noise with mean square ``flux_jy``, so that a
    noiseless visibility on any baseline has amplitude ``flux_jy``
    (see `rfi_simulator.voltages.VoltageSimulator`).
    """

    flux_jy: float
    coord: SkyCoord
    name: str = ""

    def __post_init__(self) -> None:
        self.flux_jy = float(_to_value(self.flux_jy, u.Jy))
        if not np.isfinite(self.flux_jy):
            raise ValueError(f"flux_jy must be finite, got {self.flux_jy}")
        if self.flux_jy < 0.0:
            raise ValueError(f"flux_jy must be non-negative, got {self.flux_jy}")
        if not self.coord.isscalar:
            raise ValueError("PointSource.coord must be a scalar SkyCoord")

    @classmethod
    def from_lm(
        cls,
        phase_center: SkyCoord,
        lm: tuple[float, float],
        flux_jy: float,
        name: str = "",
    ) -> "PointSource":
        """Build a `PointSource` from direction cosines relative to a phase center.

        Parameters
        ----------
        phase_center : astropy.coordinates.SkyCoord
            Scalar phase-center coordinate.
        lm : tuple of float
            Direction cosines ``(l, m)`` (dimensionless) of the source
            relative to `phase_center`; ``l`` increases towards increasing
            RA, ``m`` towards increasing Dec.
        flux_jy : float or astropy.units.Quantity
            Flux density in Jy.
        name : str, optional
            Human-readable label.

        Returns
        -------
        PointSource
            Source placed at the sky position corresponding to ``(l, m)``.
        """
        coord = radec_from_lm(phase_center, lm)
        return cls(flux_jy=flux_jy, coord=coord, name=name)

    def lm(self, phase_center: SkyCoord) -> np.ndarray:
        """Direction cosines of this source relative to `phase_center`.

        Parameters
        ----------
        phase_center : astropy.coordinates.SkyCoord
            Scalar phase-center coordinate.

        Returns
        -------
        numpy.ndarray
            Shape ``(2,)`` float64 array ``[l, m]`` (dimensionless).
        """
        return lm_from_radec(phase_center, self.coord)


def radec_from_lm(phase_center: SkyCoord, lm) -> SkyCoord:
    """Convert direction cosines to sky coordinates (inverse SIN projection).

    Parameters
    ----------
    phase_center : astropy.coordinates.SkyCoord
        Scalar phase-center coordinate.
    lm : array_like
        Direction cosines, shape ``(..., 2)``: ``lm[..., 0]`` is ``l``,
        ``lm[..., 1]`` is ``m``. Dimensionless.

    Returns
    -------
    astropy.coordinates.SkyCoord
        Coordinates of shape ``lm.shape[:-1]``, in the same frame as
        `phase_center`.

    Raises
    ------
    ValueError
        If ``l**2 + m**2 > 1`` anywhere (below the tangent-plane horizon).
    """
    lm = np.asarray(lm, dtype=np.float64)
    if lm.shape[-1] != 2:
        raise ValueError(f"lm must have shape (..., 2), got {lm.shape}")

    l_dir = lm[..., 0]
    m_dir = lm[..., 1]
    n_squared = 1.0 - l_dir**2 - m_dir**2
    if np.any(n_squared < 0.0):
        raise ValueError("l**2 + m**2 > 1: direction cosines are off the sky")
    n_dir = np.sqrt(n_squared)

    ra0 = phase_center.ra.to_value(u.rad)
    dec0 = phase_center.dec.to_value(u.rad)

    dec = np.arcsin(m_dir * np.cos(dec0) + n_dir * np.sin(dec0))
    ra = ra0 + np.arctan2(l_dir, n_dir * np.cos(dec0) - m_dir * np.sin(dec0))

    return SkyCoord(
        ra=ra * u.rad,
        dec=dec * u.rad,
        frame=phase_center.frame.replicate_without_data(),
    )


def lm_from_radec(phase_center: SkyCoord, coord: SkyCoord) -> np.ndarray:
    """Convert sky coordinates to direction cosines (forward SIN projection).

    Parameters
    ----------
    phase_center : astropy.coordinates.SkyCoord
        Scalar phase-center coordinate.
    coord : astropy.coordinates.SkyCoord
        Coordinate(s) to project.

    Returns
    -------
    numpy.ndarray
        Shape ``coord.shape + (2,)`` float64 array of ``(l, m)``
        direction cosines (dimensionless).
    """
    ra0 = phase_center.ra.to_value(u.rad)
    dec0 = phase_center.dec.to_value(u.rad)

    coord = coord.transform_to(phase_center.frame.replicate_without_data())
    ra = np.atleast_1d(coord.ra.to_value(u.rad))
    dec = np.atleast_1d(coord.dec.to_value(u.rad))
    delta_ra = ra - ra0

    l_dir = np.cos(dec) * np.sin(delta_ra)
    m_dir = np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(delta_ra)

    out = np.stack([l_dir, m_dir], axis=-1)
    if coord.isscalar:
        return out[0]
    return out
