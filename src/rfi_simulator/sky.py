"""Sky model: point sources and direction-cosine (l, m) geometry.

Conventions: the phase center is an ICRS
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

__all__ = [
    "PointSource",
    "SpectralLineForeground",
    "lm_from_radec",
    "radec_from_lm",
]

_LN2 = np.log(2.0)
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * _LN2))


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


@dataclass
class SpectralLineForeground:
    r"""A celestial spectral line, added as independent per-antenna noise.

    Any observation in this band sits near strong, spatially extended
    celestial line emission -- the classic example is the 21 cm neutral
    hydrogen line at 1420.4058 MHz, present in essentially every pointing.
    A simulator whose celestial content is only compact point sources omits
    this, and a benchmark trained on such data can end up rewarding an
    excision algorithm for flagging a bright, narrowband, *scientifically
    useful* feature simply because it looks the same as a narrowband
    transmitter. This class exists to put the line back in, tagged with a
    ground-truth label distinct from interference (`class "celestial"`, not
    `"rfi"`) so that later scoring can tell the two apart.

    Physics approximation (v1)
    ---------------------------
    A real large-scale line-emitting foreground is only partially resolved
    by any one baseline, and in general has *some* correlated (fringing)
    flux on short baselines. Modelling that properly needs a brightness
    distribution on the sky and a per-baseline visibility function, which
    this class does not attempt. Instead it takes the "fully resolved
    extended emission" limit: each antenna sees an independent, incoherent
    noise-like contribution at the line frequency, exactly as if the
    emission were so extended that no baseline in this array resolves any
    of its structure. Concretely, the same machinery as the receiver noise
    (`rfi_simulator.voltages.VoltageSimulator.noise_std`) is used, except
    the added power is frequency-shaped into a Gaussian bump instead of
    being flat across the band, and drawn independently per antenna, per
    channel, per block. Consequences of the approximation, both
    deliberate: the line shows up in every antenna's autocorrelation
    spectrum (a real system-temperature bump) and is completely absent from
    the imaginary part of every cross-correlation's expectation -- there is
    no coherent signal for the correlator to find. A future version that
    wants correlated line flux needs a genuine sky model for it, not a
    tweak to this one.

    Flux convention
    ----------------
    Chosen to match `noise_std`: `line_flux_jy` is the *peak-channel* added
    power per antenna, i.e. the channel nearest `center_freq_hz` gets
    ``E|v_i|**2 == line_flux_jy`` (on the continuous frequency axis; the
    discretized peak can be marginally lower if `center_freq_hz` falls
    between two channel centers), the same role `noise_std**2` plays for
    the frequency-flat floor. Away from the center the added power tapers
    as a Gaussian in frequency with the given `fwhm_hz`, so `line_flux_jy`
    and `noise_std**2` are both additive, per-antenna, per-channel powers
    in Jy and can be compared or added directly.

    Parameters
    ----------
    center_freq_hz : float or astropy.units.Quantity, optional
        Line center frequency, Hz. Default 1420.4058e6 (neutral hydrogen,
        rest frame -- no Doppler shift is applied here; shift the value if
        a systemic velocity is wanted).
    fwhm_hz : float or astropy.units.Quantity, optional
        Full width at half maximum of the Gaussian frequency profile, Hz.
        Must be positive.
    line_flux_jy : float or astropy.units.Quantity, optional
        Peak-channel added power per antenna, Jy (see Flux convention
        above). Must be non-negative.
    name : str, optional
        Human-readable label, carried through to
        `rfi_simulator.voltages.VoltageBlock.celestial_source_names`.
        Default ``"hi_line"``.

    Raises
    ------
    ValueError
        If `center_freq_hz` or `fwhm_hz` is not finite and positive, or if
        `line_flux_jy` is not finite and non-negative.
    """

    center_freq_hz: float = 1420.4058e6
    fwhm_hz: float = 20e3
    line_flux_jy: float = 1.0
    name: str = "hi_line"

    def __post_init__(self) -> None:
        self.center_freq_hz = float(_to_value(self.center_freq_hz, u.Hz))
        self.fwhm_hz = float(_to_value(self.fwhm_hz, u.Hz))
        self.line_flux_jy = float(_to_value(self.line_flux_jy, u.Jy))
        self.name = str(self.name)
        if not np.isfinite(self.center_freq_hz) or self.center_freq_hz <= 0.0:
            raise ValueError(f"center_freq_hz must be finite and > 0, got {self.center_freq_hz}")
        if not np.isfinite(self.fwhm_hz) or self.fwhm_hz <= 0.0:
            raise ValueError(f"fwhm_hz must be finite and > 0, got {self.fwhm_hz}")
        if not np.isfinite(self.line_flux_jy) or self.line_flux_jy < 0.0:
            raise ValueError(f"line_flux_jy must be finite and >= 0, got {self.line_flux_jy}")

    def power_envelope_jy(self, freq_hz) -> np.ndarray:
        """Per-antenna added power at each RF channel, Jy.

        Parameters
        ----------
        freq_hz : array_like
            Shape ``(n_chan,)`` RF channel center frequencies, Hz.

        Returns
        -------
        numpy.ndarray
            Float64 array of shape ``(n_chan,)``: ``line_flux_jy`` times a
            Gaussian in frequency of width `fwhm_hz`, centered on
            `center_freq_hz`. Evaluated on the continuous frequency axis
            with no band membership check -- a line wholly outside `freq_hz`
            simply gives negligible (numerically underflowing to zero)
            values everywhere, which is the graceful behavior wanted when a
            line is partially or fully out of band.
        """
        freq_hz = np.asarray(freq_hz, dtype=np.float64)
        sigma_hz = self.fwhm_hz * _FWHM_TO_SIGMA
        return self.line_flux_jy * np.exp(-0.5 * ((freq_hz - self.center_freq_hz) / sigma_hz) ** 2)

    def mask(self, freq_hz, n_time: int, threshold: float = 0.01) -> np.ndarray:
        """Ground-truth occupancy mask, celestial label.

        Parameters
        ----------
        freq_hz : array_like
            Shape ``(n_chan,)`` RF channel center frequencies, Hz.
        n_time : int
            Number of time samples in the block; the mask is broadcast
            across this axis because the line's envelope, unlike an
            interference source's, does not vary within a block -- there is
            no duty cycle or on/off pattern to a celestial line.
        threshold : float, optional
            Fraction of the profile's peak channel value above which a
            channel counts as occupied. Default 0.01 (1 %), the same
            convention `rfi_simulator.rfi.OCCUPANCY_THRESHOLD` uses for
            interference labels -- a shared numerical convention, not a
            shared meaning; this mask's class is ``"celestial"``, kept in a
            separate field from any `rfi_mask` (see
            `rfi_simulator.voltages.VoltageBlock`).

        Returns
        -------
        numpy.ndarray
            Boolean array of shape ``(n_chan, n_time)``. All False if the
            line is entirely outside the simulated band (the profile never
            exceeds `threshold` of its own peak in that case only because
            the peak itself is negligible; in practice this is the
            partially/fully-out-of-band case degrading gracefully to an
            empty mask).
        """
        freq_hz = np.asarray(freq_hz, dtype=np.float64)
        envelope = self.power_envelope_jy(freq_hz)
        peak = float(envelope.max()) if envelope.size else 0.0
        if peak <= 0.0:
            occupied_chan = np.zeros(envelope.shape, dtype=bool)
        else:
            occupied_chan = envelope > threshold * peak
        return np.broadcast_to(
            occupied_chan[:, np.newaxis], (envelope.shape[0], int(n_time))
        ).copy()
