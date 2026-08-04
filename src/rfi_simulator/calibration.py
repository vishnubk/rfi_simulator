r"""Calibration-solution error model: what a calibration pipeline gets wrong.

`rfi_simulator.instrument.InstrumentModel` models the *true* receiver
chain -- the per-antenna complex gain :math:`g_i(f)` that actually sits
between the sky and the correlator. This module models something
different: the *residual error* left behind by an imperfect calibration
pipeline, i.e. the gap between the truth and what a calibration exercise
recovered.

A real pipeline never solves for :math:`g_i(f)` exactly. It estimates
:math:`\hat g_i(f) = g_i(f)\, c_i(f)`, and divides it back out of the
data. If :math:`c_i(f)` were exactly one for every antenna, calibration
would be perfect and this module would be a no-op. It never is: the
solution has finite signal-to-noise, the solve interval is not
infinitesimally short, and the bandpass is not infinitely finely
resolved. `CalibrationErrors` draws a residual :math:`c_i(f)` for every
antenna, built from three independent, physically distinct pieces, each
switched off by default:

* **A residual phase**, `phase_error_deg_rms`, constant across the band --
  the leftover of an imperfect *phase* calibration solved once per
  antenna. Modelled as a per-antenna Gaussian in degrees, matching the
  small-angle regime the coherence-loss analysis below assumes.
* **A residual delay**, `delay_error_ns_rms` -- the leftover of an
  imperfect *delay* (bandpass-slope) calibration, which shows up as a
  phase that is *linear in frequency about the band center*,
  :math:`2\pi (f - f_\mathrm{ref}) \tau_i`, rather than constant. A delay
  error and a phase error look identical at one frequency and completely
  different across a wide band, which is why real pipelines solve for
  both separately and why this module keeps them separate too. Referencing
  the slope to :math:`f_\mathrm{ref}` rather than to zero frequency is
  what makes that separation physical: a delay calibrated at
  :math:`f_\mathrm{ref}` and residual by :math:`\tau_i` afterwards
  contributes *zero* phase there and :math:`\pm\pi B \tau_i` at the edges
  of a band of width :math:`B`, exactly the "the calibration was good at
  one frequency and drifts away from it" error a delay residual actually
  is -- not a large constant phase shared by the whole band, which is what
  referencing to zero frequency would make even a sub-nanosecond residual
  look like at RF.
* **A residual amplitude**, `amplitude_error_db_rms` -- the leftover of an
  imperfect *amplitude* calibration, lognormal in the same dB-of-power
  convention `rfi_simulator.instrument` uses.

Putting the three together, antenna ``i``'s residual factor at RF
frequency ``f`` is

.. math::

    c_i(f) = 10^{a_i / 20}\,
    \exp\left[i \left(\phi_i + 2\pi (f - f_\mathrm{ref}) \tau_i\right)\right],

with :math:`a_i` the amplitude error in dB, :math:`\phi_i` the phase
error in radians, :math:`\tau_i` the delay error in seconds and
:math:`f_\mathrm{ref}` the reference frequency the delay slope is
measured about (`reference_freq_hz`, defaulting to the mean of the band
`factors` is evaluated on). All three error terms default to zero, i.e.
:math:`c_i(f) \equiv 1` -- perfect calibration, bit-identical to not
passing this feature at all.

Application point
------------------
`rfi_simulator.correlator.correlate` accepts a ``calibration_errors=``
argument. It multiplies every baseline's visibility by the
:math:`c_i(f)\, c_j(f)^*` this residual predicts -- exactly the same
baseline structure `InstrumentModel`'s true gains are applied with, just
at the visibility level instead of the voltage level, which is both the
cheapest place to do it and, physically, the right one: a calibration
pipeline's residual error is a property of the *solution it produced*,
not of the antenna's receiver chain, and it is applied at calibration
time -- downstream of correlation in every real pipeline. Applying it
here leaves `rfi_simulator.voltages.VoltageBlock.data` and
`rfi_simulator.instrument.InstrumentModel`'s ground truth completely
untouched: the true gains a calibration exercise has to solve for are
exactly as good (or bad) as they were before, and this module only
corrupts what the pipeline *thinks* it knows about them.

The applied factors are themselves recorded as ground truth, on
`rfi_simulator.correlator.Visibilities.calibration_error_gains` -- a
calibration exercise built against this simulator should be able to
recover them, the same way `VoltageBlock.gains` lets one recover the true
instrument.

Coherence loss
---------------
For small, zero-mean Gaussian phase errors (the `phase_error_deg_rms`
case with the delay and amplitude errors off), the classic result holds:
a baseline's average coherence is reduced by
:math:`\exp\left(-\sigma_\phi^2\right)`, with :math:`\sigma_\phi` in
radians -- one factor of :math:`\exp(-\sigma_\phi^2/2)` from each of the
two independent antenna phases entering the baseline. A dirty-image peak
loses exactly that fraction of its flux, which is the basis of this
module's acceptance test.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["CalibrationErrors"]


def _readonly(array: np.ndarray) -> np.ndarray:
    """Return a read-only view of a fresh copy of `array`."""
    out = np.array(array, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class CalibrationErrors:
    """Per-antenna residual calibration error, relative to the truth.

    Build one with `from_params` (random, seeded) or `identity` (a no-op
    model, every factor exactly one). Instances are frozen and their
    arrays are read-only.

    Attributes
    ----------
    phase_error_rad : numpy.ndarray
        Float64 array of shape ``(n_antennas,)``: the constant
        (frequency-independent) residual phase error of each antenna,
        radians.
    delay_error_s : numpy.ndarray
        Float64 array of shape ``(n_antennas,)``: the residual delay
        error of each antenna, seconds. Contributes a phase linear in RF
        frequency, :math:`2\\pi f \\tau_i`.
    amplitude_error_db : numpy.ndarray
        Float64 array of shape ``(n_antennas,)``: the residual amplitude
        error of each antenna, in dB of power (see the module
        docstring's :math:`c_i(f)` -- ``0`` is unit amplitude).
    reference_freq_hz : float or None, optional
        Frequency, in Hz, the delay-error phase slope is zero at. Default
        ``None``: `factors` uses the mean of whatever `freq_hz` grid it is
        evaluated on, so a residual delay contributes zero phase at band
        center and the slope shows up entirely as a spread across the
        band rather than as a large shared phase offset. Pin an explicit
        value to keep the reference fixed across calls with different
        `freq_hz` grids.

    Examples
    --------
    >>> import numpy as np
    >>> errors = CalibrationErrors.from_params(
    ...     4, seed=3, phase_error_deg_rms=5.0, delay_error_ns_rms=0.2
    ... )
    >>> c = errors.factors(np.array([1.40e9, 1.41e9]))
    >>> c.shape
    (4, 2)
    >>> bool(np.allclose(np.abs(c), 1.0))  # amplitude error is off here
    True
    """

    phase_error_rad: np.ndarray
    delay_error_s: np.ndarray
    amplitude_error_db: np.ndarray
    reference_freq_hz: float | None = None

    def __post_init__(self) -> None:
        phase = np.asarray(self.phase_error_rad, dtype=np.float64)
        delay = np.asarray(self.delay_error_s, dtype=np.float64)
        amplitude = np.asarray(self.amplitude_error_db, dtype=np.float64)
        if phase.ndim != 1 or phase.size < 1:
            raise ValueError(f"phase_error_rad must have shape (n_antennas,), got {phase.shape}")
        if delay.shape != phase.shape:
            raise ValueError(
                f"delay_error_s must have the same shape as phase_error_rad {phase.shape}, "
                f"got {delay.shape}"
            )
        if amplitude.shape != phase.shape:
            raise ValueError(
                "amplitude_error_db must have the same shape as phase_error_rad "
                f"{phase.shape}, got {amplitude.shape}"
            )
        if not (
            np.all(np.isfinite(phase))
            and np.all(np.isfinite(delay))
            and np.all(np.isfinite(amplitude))
        ):
            raise ValueError("CalibrationErrors fields contain non-finite values")
        if self.reference_freq_hz is not None:
            reference_freq_hz = float(self.reference_freq_hz)
            if not np.isfinite(reference_freq_hz):
                raise ValueError(
                    f"reference_freq_hz must be finite or None, got {self.reference_freq_hz}"
                )
            object.__setattr__(self, "reference_freq_hz", reference_freq_hz)
        object.__setattr__(self, "phase_error_rad", _readonly(phase))
        object.__setattr__(self, "delay_error_s", _readonly(delay))
        object.__setattr__(self, "amplitude_error_db", _readonly(amplitude))

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def identity(cls, n_antennas: int) -> "CalibrationErrors":
        """A perfect calibration: every residual factor exactly ``1 + 0j``.

        Parameters
        ----------
        n_antennas : int
            Number of antennas.

        Returns
        -------
        CalibrationErrors
            A model whose `factors` are exactly one, so that applying it
            in `rfi_simulator.correlator.correlate` is a bit-for-bit
            no-op.
        """
        n_antennas = int(n_antennas)
        if n_antennas < 1:
            raise ValueError(f"n_antennas must be >= 1, got {n_antennas}")
        zeros = np.zeros(n_antennas, dtype=np.float64)
        return cls(phase_error_rad=zeros, delay_error_s=zeros, amplitude_error_db=zeros)

    @classmethod
    def from_params(
        cls,
        n_antennas: int,
        *,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
        phase_error_deg_rms: float = 0.0,
        delay_error_ns_rms: float = 0.0,
        amplitude_error_db_rms: float = 0.0,
        reference_freq_hz: float | None = None,
    ) -> "CalibrationErrors":
        """Draw a random, repeatable residual calibration error.

        Parameters
        ----------
        n_antennas : int
            Number of antennas. Must match the dataset this is applied to.
        rng : numpy.random.Generator, optional
            Generator to derive the model's seed from. Drawn from exactly
            once. Mutually exclusive with `seed`.
        seed : int, optional
            Seed to derive the model from. Mutually exclusive with `rng`.
            Either this or `rng` is required as soon as any stochastic
            feature is switched on.
        phase_error_deg_rms : float, optional
            Rms, in degrees, of each antenna's constant residual phase
            error, drawn from a zero-mean normal. Default 0.0 (no residual
            phase error).
        delay_error_ns_rms : float, optional
            Rms, in nanoseconds, of each antenna's residual delay error,
            drawn from a zero-mean normal. Default 0.0 (no residual delay
            error).
        amplitude_error_db_rms : float, optional
            Rms, in dB of power, of each antenna's residual amplitude
            error -- the amplitude of antenna ``i`` is ``10 ** (x_i /
            20)`` with ``x_i`` drawn from a zero-mean normal of this
            width, i.e. a lognormal amplitude, the same convention as
            `rfi_simulator.instrument.InstrumentModel`. Default 0.0 (no
            residual amplitude error).
        reference_freq_hz : float, optional
            See the `CalibrationErrors` attribute of the same name.
            Default ``None``: the delay-error slope is referenced to the
            mean of whatever band `factors` is later evaluated on.

        Returns
        -------
        CalibrationErrors
            An immutable model, fully determined by the seed and the
            antenna count.

        Raises
        ------
        ValueError
            If both `rng` and `seed` are given; if neither is given while
            a stochastic feature is switched on; if `n_antennas` is not
            positive; or if a scatter parameter is non-finite or
            negative.

        Notes
        -----
        The three effects draw from *independent* children of the model's
        root seed sequence, in the fixed order phase, delay, amplitude, so
        switching one on or changing its parameters never perturbs the
        others -- the same pattern `InstrumentModel.from_params` uses, for
        the same reason.
        """
        n_antennas = int(n_antennas)
        if n_antennas < 1:
            raise ValueError(f"n_antennas must be >= 1, got {n_antennas}")
        phase_error_deg_rms = float(phase_error_deg_rms)
        delay_error_ns_rms = float(delay_error_ns_rms)
        amplitude_error_db_rms = float(amplitude_error_db_rms)
        if not np.isfinite(phase_error_deg_rms) or phase_error_deg_rms < 0.0:
            raise ValueError(
                f"phase_error_deg_rms must be finite and >= 0, got {phase_error_deg_rms}"
            )
        if not np.isfinite(delay_error_ns_rms) or delay_error_ns_rms < 0.0:
            raise ValueError(
                f"delay_error_ns_rms must be finite and >= 0, got {delay_error_ns_rms}"
            )
        if not np.isfinite(amplitude_error_db_rms) or amplitude_error_db_rms < 0.0:
            raise ValueError(
                f"amplitude_error_db_rms must be finite and >= 0, got {amplitude_error_db_rms}"
            )
        if rng is not None and seed is not None:
            raise ValueError("give either rng or seed, not both")

        wants_randomness = (
            phase_error_deg_rms > 0.0 or delay_error_ns_rms > 0.0 or amplitude_error_db_rms > 0.0
        )
        if rng is None and seed is None:
            if wants_randomness:
                raise ValueError(
                    "from_params needs an rng or a seed to draw a phase, delay or amplitude "
                    "calibration error; a model with all three switched off needs neither"
                )
            root = np.random.SeedSequence(0)
        elif seed is not None:
            root = np.random.SeedSequence(seed)
        else:
            entropy = rng.integers(0, 2**63 - 1, size=4, dtype=np.int64)
            root = np.random.SeedSequence(entropy.tolist())

        # One independent child per effect, spawned unconditionally and in
        # a fixed order -- see the class Notes.
        phase_seed, delay_seed, amp_seed = root.spawn(3)

        if phase_error_deg_rms > 0.0:
            phase_error_rad = np.random.default_rng(phase_seed).normal(
                loc=0.0, scale=np.deg2rad(phase_error_deg_rms), size=n_antennas
            )
        else:
            phase_error_rad = np.zeros(n_antennas, dtype=np.float64)

        if delay_error_ns_rms > 0.0:
            delay_error_s = np.random.default_rng(delay_seed).normal(
                loc=0.0, scale=delay_error_ns_rms * 1e-9, size=n_antennas
            )
        else:
            delay_error_s = np.zeros(n_antennas, dtype=np.float64)

        if amplitude_error_db_rms > 0.0:
            amplitude_error_db = np.random.default_rng(amp_seed).normal(
                loc=0.0, scale=amplitude_error_db_rms, size=n_antennas
            )
        else:
            amplitude_error_db = np.zeros(n_antennas, dtype=np.float64)

        return cls(
            phase_error_rad=phase_error_rad,
            delay_error_s=delay_error_s,
            amplitude_error_db=amplitude_error_db,
            reference_freq_hz=reference_freq_hz,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @property
    def n_antennas(self) -> int:
        """int: Number of antennas the model describes."""
        return int(self.phase_error_rad.size)

    def factors(self, freq_hz: np.ndarray) -> np.ndarray:
        """Residual complex calibration factor of every antenna, ``c_i(f)``.

        Parameters
        ----------
        freq_hz : numpy.ndarray
            Shape ``(n_chan,)`` RF channel center frequencies, Hz.

        Returns
        -------
        numpy.ndarray
            Complex128 array of shape ``(n_antennas, n_chan)``, the
            :math:`c_i(f)` of the module docstring. `correlate` applies
            it to a baseline as :math:`c_i(f)\\, c_j(f)^*`.

        Raises
        ------
        ValueError
            If `freq_hz` is not a 1-D non-empty array, or if it contains a
            non-finite value.

        Notes
        -----
        The delay-error phase is referenced to `reference_freq_hz` if set,
        or otherwise to the mean of `freq_hz` itself -- evaluated fresh on
        every call, so a fixed reference has to be pinned explicitly if
        `factors` will be called on more than one grid and the two must
        agree on where the delay slope crosses zero.
        """
        freq = np.asarray(freq_hz, dtype=np.float64)
        if freq.ndim != 1 or freq.size < 1:
            raise ValueError(f"freq_hz must have shape (n_chan,), got {freq.shape}")
        if not np.all(np.isfinite(freq)):
            raise ValueError("freq_hz contains non-finite values")

        reference_freq_hz = (
            float(freq.mean()) if self.reference_freq_hz is None else self.reference_freq_hz
        )
        phase = self.phase_error_rad[:, np.newaxis] + 2.0 * np.pi * self.delay_error_s[
            :, np.newaxis
        ] * (freq[np.newaxis, :] - reference_freq_hz)
        amplitude = 10.0 ** (self.amplitude_error_db[:, np.newaxis] / 20.0)
        return (amplitude * np.exp(1j * phase)).astype(np.complex128)


def resolve_calibration_error_models(
    calibration_errors: CalibrationErrors | Sequence[CalibrationErrors],
    n_pol: int,
) -> list[CalibrationErrors]:
    """Validate `calibration_errors` into one `CalibrationErrors` per polarization.

    Parameters
    ----------
    calibration_errors : CalibrationErrors or sequence of CalibrationErrors
        A single model, applied to every polarization, or exactly `n_pol`
        models, one per polarization in the data's polarization order.
    n_pol : int
        Number of polarizations the data carries.

    Returns
    -------
    list of CalibrationErrors
        Length `n_pol`.

    Raises
    ------
    ValueError
        If the argument is neither a `CalibrationErrors` nor a sequence of
        them, or if a sequence has the wrong length.

    Notes
    -----
    Mirrors `rfi_simulator.voltages.VoltageSimulator._instrument_models`:
    passing one model broadcasts it to every polarization -- the "one
    calibration solve serves both receptors" idealization -- while a
    sequence of `n_pol` models gives each polarization an independently
    drawn residual error, which is the realistic case for a pipeline that
    calibrates each polarization's data separately.
    """
    if isinstance(calibration_errors, CalibrationErrors):
        return [calibration_errors] * n_pol
    try:
        models = list(calibration_errors)
    except TypeError as exc:
        raise ValueError(
            "calibration_errors must be a CalibrationErrors, a sequence of them (one per "
            f"polarization), or None, got {type(calibration_errors)!r}"
        ) from exc
    if not models or not all(isinstance(model, CalibrationErrors) for model in models):
        raise ValueError(
            "calibration_errors must be a CalibrationErrors, a sequence of them (one per "
            f"polarization), or None, got {type(calibration_errors)!r}"
        )
    if len(models) != n_pol:
        raise ValueError(
            f"calibration_errors has {len(models)} models but the data has n_pol={n_pol}; "
            "pass one model per polarization, or a single model to give every polarization "
            "the same calibration error"
        )
    return models
