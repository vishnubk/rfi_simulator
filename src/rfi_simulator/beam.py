r"""Primary-beam models: attenuation of celestial flux away from the pointing.

A real antenna is not equally sensitive in every direction. Its voltage
response falls off away from the direction it is pointed at, with a shape
and width set by the aperture and the observing wavelength. Without a
model of this, every celestial point source in the simulator is received
at its full catalog flux regardless of how far it sits from the phase
center -- which both overstates the flux of anything off-axis and,
because a virtual-observatory scene's inferred system sensitivity is
calibrated against the (unattenuated) known flux of a source used for
that purpose, biases the inferred sensitivity by the beam's own
attenuation at that source's offset (of order the ~1.5x this module
exists to remove).

This module is deliberately narrow. It answers exactly one question --
given an angular offset from the pointing center and an observing
frequency, what fraction of on-axis power does the antenna respond with
-- and leaves everything else (where the pointing center is, which
sources get attenuated, how the factor enters the voltage equation) to
`rfi_simulator.voltages.VoltageSimulator`, which is the only caller.

Power vs voltage response
--------------------------
Two conventions are in play and it is easy to conflate them:

* the **power response** :math:`B(\theta, f)` is what antenna engineers
  call "the primary beam": the ratio of received power at offset
  :math:`\theta` to received power on-axis, for a fixed source. This is
  what both classes here return from `power_response`, and it is also
  exactly the factor a **correlated visibility's flux** scales by, because
  a visibility is built from *two* antenna voltages and a source's flux
  enters a visibility linearly (see `rfi_simulator.voltages`'s amplitude
  convention) -- so a full derivation would show the beam entering as
  :math:`\sqrt{B_i(\theta)}\sqrt{B_j(\theta)}^{*}=B(\theta)` for identical
  antennas :math:`i,j` pointed the same way, one factor of :math:`B`
  overall, matching the "flux scales by the power response" acceptance
  test.
* the simulator, however, synthesizes per-**antenna voltages**, not
  visibilities directly, so what actually multiplies each antenna's
  voltage sample is the **voltage (amplitude) response**
  :math:`\sqrt{B(\theta, f)}` -- the square root of the power response.
  Squaring that factor on both antennas of a baseline recovers
  :math:`B(\theta, f)` in the correlated visibility, which is the
  quantity this module's docstrings and tests reason about throughout.

Both `GaussianBeam` and `AiryBeam` return the **power** response from
`power_response`; callers that need the voltage-domain factor take the
square root themselves (`rfi_simulator.voltages.VoltageSimulator` does
exactly this).

Offset convention
-----------------
Both models are functions of a single scalar offset ``theta_rad``, the
angle between the source direction and the pointing center. Computing
that angle is the caller's job (`rfi_simulator.voltages.VoltageSimulator`
uses the small-angle approximation
:math:`\theta \approx \sqrt{l^2 + m^2}` from the direction cosines
`rfi_simulator.sky.PointSource.lm` already provides -- exact for
:math:`\theta = 0` and accurate to :math:`O(\theta^4)` at the offsets
(a few degrees at most) any of this package's scenes use). Both beam
models are otherwise expressed in terms of the *true* angle (`AiryBeam`
even takes ``sin(theta)`` internally, so it stays well-behaved at large
offsets even though nothing in this package currently pushes it that
far).

Design
------
`PrimaryBeam` is intentionally a thin abstract base: exactly one method,
`power_response`, that every model must implement and that
`VoltageSimulator` calls without caring which concrete model it got.
Adding a third model (e.g. a measured/tabulated beam) means implementing
that one method against this same signature; nothing else in the package
needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from rfi_simulator.delays import SPEED_OF_LIGHT_M_S

__all__ = ["AiryBeam", "GaussianBeam", "PrimaryBeam", "bessel_j1"]

_LN2 = np.log(2.0)

#: Coefficient relating a filled circular aperture's FWHM to lambda/D.
#: 1.02 is the standard engineering approximation for a uniformly
#: illuminated circular aperture (the Airy pattern's FWHM is ~1.03 lambda/D;
#: 1.02 is the commonly quoted rounded value for a real, mildly tapered
#: illumination, e.g. the ALMA/VLA convention) and is used here, rather than
#: the Gaussian-beam-fit-to-Airy value of ~1.028 lambda/D, because it is the
#: number radio-astronomy documentation for real dishes almost universally
#: quotes; the difference between candidate coefficients here (1.02 vs
#: 1.03) is well under the accuracy this simulator claims for a beam model
#: in the first place.
GAUSSIAN_FWHM_COEFF = 1.02


def _validate_theta_rad(theta_rad) -> np.ndarray:
    """Validate and array-ify an angular-offset argument.

    Parameters
    ----------
    theta_rad : array_like
        Angular offset(s) from the pointing center, radians.

    Returns
    -------
    numpy.ndarray
        Float64 array.

    Raises
    ------
    ValueError
        If any entry is non-finite or negative.
    """
    theta = np.asarray(theta_rad, dtype=np.float64)
    if not np.all(np.isfinite(theta)):
        raise ValueError(f"theta_rad must be finite, got {theta_rad!r}")
    if np.any(theta < 0.0):
        raise ValueError(f"theta_rad must be non-negative, got {theta_rad!r}")
    return theta


def _validate_freq_hz(freq_hz) -> np.ndarray:
    """Validate and array-ify a frequency argument.

    Parameters
    ----------
    freq_hz : array_like
        Frequency/frequencies, Hz.

    Returns
    -------
    numpy.ndarray
        Float64 array.

    Raises
    ------
    ValueError
        If any entry is non-finite or not strictly positive.
    """
    freq = np.asarray(freq_hz, dtype=np.float64)
    if not np.all(np.isfinite(freq)):
        raise ValueError(f"freq_hz must be finite, got {freq_hz!r}")
    if np.any(freq <= 0.0):
        raise ValueError(f"freq_hz must be > 0, got {freq_hz!r}")
    return freq


class PrimaryBeam(ABC):
    """Abstract base for a frequency-dependent primary-beam power pattern.

    A concrete model implements `power_response` only; everything else
    (how the offset is computed, where the factor enters the voltage
    equation, whether it applies to a given source class) lives in
    `rfi_simulator.voltages`.
    """

    @abstractmethod
    def power_response(self, theta_rad, freq_hz) -> np.ndarray:
        """Beam power response at angular offset(s) and frequency/frequencies.

        Parameters
        ----------
        theta_rad : array_like
            Angular offset(s) from the pointing center, radians,
            non-negative and finite.
        freq_hz : array_like
            Observing frequency/frequencies, Hz, finite and positive.
            Broadcastable against `theta_rad` (typically `theta_rad` has
            shape ``(n_src, 1)`` and `freq_hz` has shape ``(n_chan,)`` so
            the result broadcasts to ``(n_src, n_chan)``).

        Returns
        -------
        numpy.ndarray
            Power response(s) in ``[0, 1]``, ``1.0`` on-axis, the same
            broadcast shape as ``theta_rad * freq_hz``.
        """
        raise NotImplementedError


@dataclass
class GaussianBeam(PrimaryBeam):
    r"""A circularly symmetric Gaussian primary beam.

    The frequency-dependent full width at half maximum is the standard
    diffraction-limited approximation for a filled circular aperture,

    .. math::

        \mathrm{FWHM}(f) = \mathtt{GAUSSIAN\_FWHM\_COEFF}
            \, \frac{\lambda}{D} = \mathtt{GAUSSIAN\_FWHM\_COEFF}
            \, \frac{c}{f D},

    and the power pattern is the Gaussian with that FWHM,

    .. math::

        B(\theta, f) = \exp\!\left(-4 \ln 2\,
            \left(\frac{\theta}{\mathrm{FWHM}(f)}\right)^2\right),

    so ``B = 1`` on-axis, ``B = 0.5`` at ``theta = FWHM/2`` by
    construction, and the beam shrinks (attenuates a fixed offset more
    strongly) at higher frequency, since ``FWHM`` scales as ``1/f``. This
    is the standard "Gaussian main-lobe" approximation to a real
    aperture's response: it has no sidelobes and is only accurate near
    the main lobe, which is the regime this simulator's celestial scenes
    (compact sources within a beam or a few of the pointing center) live
    in -- see `AiryBeam` for a model with sidelobes and an exact first
    null.

    Parameters
    ----------
    dish_diameter_m : float
        Aperture (dish) diameter, meters. Must be finite and positive.

    Raises
    ------
    ValueError
        If `dish_diameter_m` is not finite and positive.
    """

    dish_diameter_m: float

    def __post_init__(self) -> None:
        self.dish_diameter_m = float(self.dish_diameter_m)
        if not np.isfinite(self.dish_diameter_m) or self.dish_diameter_m <= 0.0:
            raise ValueError(f"dish_diameter_m must be finite and > 0, got {self.dish_diameter_m}")

    def fwhm_rad(self, freq_hz) -> np.ndarray:
        """Frequency-dependent FWHM, radians.

        Parameters
        ----------
        freq_hz : array_like
            Frequency/frequencies, Hz, finite and positive.

        Returns
        -------
        numpy.ndarray
            FWHM(s) in radians, the same shape as `freq_hz`.
        """
        freq = _validate_freq_hz(freq_hz)
        wavelength_m = SPEED_OF_LIGHT_M_S / freq
        return GAUSSIAN_FWHM_COEFF * wavelength_m / self.dish_diameter_m

    def power_response(self, theta_rad, freq_hz) -> np.ndarray:
        theta = _validate_theta_rad(theta_rad)
        fwhm_rad = self.fwhm_rad(freq_hz)
        return np.exp(-4.0 * _LN2 * (theta / fwhm_rad) ** 2)


@dataclass
class AiryBeam(PrimaryBeam):
    r"""The diffraction pattern of a uniformly illuminated circular aperture.

    .. math::

        x = \frac{\pi D \sin\theta}{\lambda}, \qquad
        B(\theta, f) = \left(\frac{2 J_1(x)}{x}\right)^2,

    with :math:`B \to 1` as :math:`x \to 0` (the removable singularity is
    handled with the small-``x`` Taylor series
    :math:`2 J_1(x)/x = 1 - x^2/8 + O(x^4)`, switched in below
    ``|x| < 1e-4``, well inside float64 precision of the exact ratio).
    Unlike `GaussianBeam` this has real sidelobes and an exact first null
    at :math:`x \approx 3.8317` (the first positive zero of :math:`J_1`),
    i.e. at

    .. math::

        \sin\theta_{\mathrm{null}} = 3.8317\, \frac{\lambda}{\pi D}
            \approx 1.22\, \frac{\lambda}{D}.

    ``sin(theta)`` is used (not the small-angle ``theta``) so the model
    stays well-defined for an offset of any size, though this package's
    scenes only ever use it at small offsets where the two agree to
    :math:`O(\theta^3)`.

    Parameters
    ----------
    dish_diameter_m : float
        Aperture (dish) diameter, meters. Must be finite and positive.

    Raises
    ------
    ValueError
        If `dish_diameter_m` is not finite and positive.
    """

    dish_diameter_m: float

    #: Below this |x|, the exact ratio 2*J1(x)/x is replaced by its
    #: Taylor series to avoid a 0/0 division; the two agree to better
    #: than 1e-16 relative error well above this threshold already, so
    #: the switch point only has to dodge the literal singularity at x=0.
    _SMALL_X = 1e-4

    def __post_init__(self) -> None:
        self.dish_diameter_m = float(self.dish_diameter_m)
        if not np.isfinite(self.dish_diameter_m) or self.dish_diameter_m <= 0.0:
            raise ValueError(f"dish_diameter_m must be finite and > 0, got {self.dish_diameter_m}")

    def x_argument(self, theta_rad, freq_hz) -> np.ndarray:
        """The Airy pattern's dimensionless argument ``pi D sin(theta) / lambda``.

        Parameters
        ----------
        theta_rad : array_like
            Angular offset(s), radians, non-negative and finite.
        freq_hz : array_like
            Frequency/frequencies, Hz, finite and positive.

        Returns
        -------
        numpy.ndarray
            Broadcast of `theta_rad` against `freq_hz`.
        """
        theta = _validate_theta_rad(theta_rad)
        freq = _validate_freq_hz(freq_hz)
        wavelength_m = SPEED_OF_LIGHT_M_S / freq
        return np.pi * self.dish_diameter_m * np.sin(theta) / wavelength_m

    def power_response(self, theta_rad, freq_hz) -> np.ndarray:
        x = self.x_argument(theta_rad, freq_hz)
        small = np.abs(x) < self._SMALL_X
        x_safe = np.where(small, 1.0, x)  # dodge the x=0 division; overwritten below
        ratio = np.where(small, 1.0 - x**2 / 8.0, 2.0 * bessel_j1(x_safe) / x_safe)
        return ratio**2


def bessel_j1(x) -> np.ndarray:
    r"""The Bessel function of the first kind, order 1, :math:`J_1(x)`.

    scipy is not a dependency of this package, so `AiryBeam` needs its own
    :math:`J_1`. This implements the classic rational-function /
    asymptotic-expansion approximation in the style of Abramowitz &
    Stegun Sec. 9.4 (the coefficients below are the widely used numerical
    fit tabulated e.g. in Press et al., *Numerical Recipes*' ``bessj1``,
    itself derived from the same family of Chebyshev/rational fits as A&S
    9.4.4 and 9.4.6): a degree-12 rational approximation for
    :math:`|x| < 8` and an asymptotic cosine/rational form for
    :math:`|x| \geq 8`. Accuracy is better than :math:`1.6\times10^{-8}`
    absolute error over the whole real line -- overkill for a beam model
    but cheap and simple to keep as one vectorized numpy expression.

    Parameters
    ----------
    x : array_like
        Argument(s), any real value(s).

    Returns
    -------
    numpy.ndarray
        :math:`J_1(x)`, same shape as `x`.

    Notes
    -----
    Verified in the test suite against the first two known zeros of
    :math:`J_1`, :math:`x \approx 3.8317` and :math:`x \approx 7.0156`.
    """
    x = np.asarray(x, dtype=np.float64)
    ax = np.abs(x)

    # Rational approximation, |x| < 8.
    y = x * x
    ans1 = x * (
        72362614232.0
        + y
        * (
            -7895059235.0
            + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))
        )
    )
    ans2 = 144725228442.0 + y * (
        2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y)))
    )
    small_branch = ans1 / ans2

    # Asymptotic expansion, |x| >= 8. `ax_safe` only feeds the branch that
    # gets discarded by `np.where` for |x| < 8, so clamping it there is
    # purely to avoid a spurious division warning, not a physical choice.
    ax_safe = np.where(ax < 8.0, 8.0, ax)
    z = 8.0 / ax_safe
    y2 = z * z
    xx = ax_safe - 2.356194491
    p1 = 1.0 + y2 * (
        0.183105e-2 + y2 * (-0.3516396496e-4 + y2 * (0.2457520174e-5 + y2 * -0.240337019e-6))
    )
    p2 = 0.04687499995 + y2 * (
        -0.2002690873e-3 + y2 * (0.8449199096e-5 + y2 * (-0.88228987e-6 + y2 * 0.105787412e-6))
    )
    large_branch = np.sqrt(0.636619772 / ax_safe) * (np.cos(xx) * p1 - z * np.sin(xx) * p2)
    large_branch = np.where(x < 0.0, -large_branch, large_branch)

    return np.where(ax < 8.0, small_branch, large_branch)
