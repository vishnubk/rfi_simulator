r"""Per-antenna instrument model: direction-independent complex gains.

Real receiver chains are not identical. Every antenna has its own low-noise
amplifier, cabling, filters and digital gain setting, so the voltage that
reaches the correlator differs from antenna to antenna by a complex factor
that is the *same for every direction on the sky* -- a
direction-independent gain :math:`g_i(f)`:

.. math::

    v_i^{\mathrm{meas}}(f, t) = g_i(f)\, v_i^{\mathrm{true}}(f, t),
    \qquad
    V_{ij}^{\mathrm{meas}}(f) = g_i(f)\, g_j^*(f)\, V_{ij}^{\mathrm{true}}(f).

This module builds such gains. Three effects are modelled, each
independently switchable and all off by default:

* **Amplitude scatter.** The per-antenna power level differs by a few
  tenths of a dB to a dB or so, which is a *large* effect compared with a
  simulator in which every antenna has exactly unit gain: flaggers and
  calibration algorithms that key on "this antenna's power is unlike its
  neighbours'" have nothing to learn from a perfectly uniform array.
  Modelled as a lognormal amplitude, parameterized by the rms of the
  per-antenna power expressed in dB (`gain_scatter_db`).
* **Phase offsets.** An uncalibrated array has an arbitrary per-antenna
  phase, which destroys coherence: a point source images at reduced peak
  and smeared structure until the phases are solved for.
* **Smooth bandpass ripple.** Filters and cable reflections give each
  antenna a *repeatable*, smoothly frequency-dependent amplitude response
  of order hundredths of a dB rms, modelled here as a sum of a few
  low-order cosines across the band. "Repeatable" is the operative word:
  the ripple is a property of the antenna, fixed by ``(seed, antenna
  index)``, not a per-block random draw, so it survives time averaging
  exactly as a real bandpass does.

Where the gains are applied
---------------------------
`rfi_simulator.voltages.VoltageSimulator` multiplies each antenna's
**total** voltage stream -- sky, interference and receiver noise together
-- by that antenna's gain. This is the physically right place for it: the
gain describes the receiver chain *behind* the antenna, and that chain
amplifies its own noise along with everything it receives. Equivalently,
the gains are referenced at the correlator input, so the per-antenna power
scatter seen in the autocorrelations includes the noise power, which is
what real per-antenna power scatter looks like.

Ground truth
------------
`InstrumentModel` is immutable once built and exposes exactly the
quantities a calibration exercise has to solve for:
`InstrumentModel.gains` (the complex :math:`g_i(f)`),
`InstrumentModel.scalar_gains` (the frequency-independent part) and
`InstrumentModel.bandpass_db` (the ripple, in dB). Simulated blocks carry
the evaluated gain array with them, so a downstream algorithm can be
scored against the truth rather than against another algorithm.

dB convention
-------------
Amplitudes here are *voltage* gains; a "gain in dB" always means the
corresponding **power** ratio, :math:`10 \log_{10} |g|^2 = 20 \log_{10}
|g|`. So `gain_scatter_db` is directly the rms of the per-antenna power in
dB, and a scatter of 0.4 dB corresponds to about 9 % rms fractional power
scatter (in general :math:`\sqrt{e^{s^2} - 1}` with :math:`s =
\sigma_{\mathrm{dB}} \ln 10 / 10`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

__all__ = ["DEFAULT_BANDPASS_N_MODES", "InstrumentModel"]

#: Number of cosine modes used for the per-antenna bandpass ripple unless
#: the caller says otherwise. A handful of modes across the band gives a
#: smooth, few-wiggle shape; many modes would look like noise rather than
#: like a filter response.
DEFAULT_BANDPASS_N_MODES = 3

#: Allowed values of `InstrumentModel.from_params`'s ``phase_offsets``.
_PHASE_MODES = ("zero", "uniform")


def _readonly(array: np.ndarray) -> np.ndarray:
    """Return a read-only view of a fresh copy of `array`.

    Parameters
    ----------
    array : numpy.ndarray
        Array to freeze.

    Returns
    -------
    numpy.ndarray
        A copy of `array` with its ``writeable`` flag cleared, so that
        neither the caller nor a later holder can mutate the stored
        ground truth in place.
    """
    out = np.array(array, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class InstrumentModel:
    """Immutable per-antenna direction-independent complex gains.

    Build one with `from_params` (random, seeded) or `from_gains`
    (explicit user-supplied gains), or get a no-op model from `identity`.
    Instances are frozen and their arrays are read-only, so a model handed
    to a simulator cannot be changed underneath it.

    Attributes
    ----------
    scalar_gains : numpy.ndarray
        Complex128 array of shape ``(n_antennas,)``: the
        frequency-independent part of each antenna's gain, i.e. amplitude
        scatter times phase offset. For a `from_gains` model built from a
        2-D gain table this is all ones and the table carries everything.
    bandpass_cos_db, bandpass_sin_db : numpy.ndarray or None
        Float64 arrays of shape ``(n_antennas, n_modes)``: the cosine and
        sine coefficients, in dB, of the per-antenna bandpass ripple (see
        `bandpass_db`). ``None`` for a flat bandpass.
    band_hz : tuple of float or None
        ``(f_low, f_high)`` reference band the ripple's mode numbers are
        defined against. If ``None``, the band is taken from the frequency
        grid `gains` is evaluated on, which makes the ripple a function of
        that grid -- fine for the common case of evaluating on exactly the
        simulated band, but pass a band explicitly if you intend to
        evaluate sub-bands of one model.
    tabulated_gains : numpy.ndarray or None
        Complex128 array of shape ``(n_antennas, n_chan)`` of explicit
        user-supplied gains, or ``None``. When set, `gains` returns it
        directly and requires the requested frequency grid to match
        `tabulated_freq_hz`.
    tabulated_freq_hz : numpy.ndarray or None
        Float64 array of shape ``(n_chan,)``, the grid `tabulated_gains`
        is defined on.

    Examples
    --------
    >>> import numpy as np
    >>> model = InstrumentModel.from_params(4, seed=7, gain_scatter_db=0.4)
    >>> g = model.gains(np.array([1.40e9, 1.41e9]))
    >>> g.shape
    (4, 2)
    >>> bool(np.allclose(np.abs(g[:, 0]), np.abs(g[:, 1])))   # flat bandpass
    True
    """

    scalar_gains: np.ndarray
    bandpass_cos_db: np.ndarray | None = None
    bandpass_sin_db: np.ndarray | None = None
    band_hz: tuple[float, float] | None = None
    tabulated_gains: np.ndarray | None = None
    tabulated_freq_hz: np.ndarray | None = None

    def __post_init__(self) -> None:
        scalar = np.asarray(self.scalar_gains, dtype=np.complex128)
        if scalar.ndim != 1 or scalar.size < 1:
            raise ValueError(f"scalar_gains must have shape (n_antennas,), got {scalar.shape}")
        if not np.all(np.isfinite(scalar)):
            raise ValueError("scalar_gains contains non-finite values")
        object.__setattr__(self, "scalar_gains", _readonly(scalar))

        cos_db, sin_db = self.bandpass_cos_db, self.bandpass_sin_db
        if (cos_db is None) != (sin_db is None):
            raise ValueError("bandpass_cos_db and bandpass_sin_db must both be given or both None")
        if cos_db is not None:
            cos_db = np.asarray(cos_db, dtype=np.float64)
            sin_db = np.asarray(sin_db, dtype=np.float64)
            if cos_db.shape != sin_db.shape:
                raise ValueError(
                    f"bandpass coefficient shapes differ: {cos_db.shape} vs {sin_db.shape}"
                )
            if cos_db.ndim != 2 or cos_db.shape[0] != scalar.size or cos_db.shape[1] < 1:
                raise ValueError(
                    "bandpass coefficients must have shape (n_antennas, n_modes) = "
                    f"({scalar.size}, >=1), got {cos_db.shape}"
                )
            if not (np.all(np.isfinite(cos_db)) and np.all(np.isfinite(sin_db))):
                raise ValueError("bandpass coefficients contain non-finite values")
            object.__setattr__(self, "bandpass_cos_db", _readonly(cos_db))
            object.__setattr__(self, "bandpass_sin_db", _readonly(sin_db))

        if self.band_hz is not None:
            low, high = (float(v) for v in self.band_hz)
            if not np.isfinite([low, high]).all() or high < low:
                raise ValueError(f"band_hz must be a finite (low, high) pair, got {self.band_hz!r}")
            object.__setattr__(self, "band_hz", (low, high))

        table, table_freq = self.tabulated_gains, self.tabulated_freq_hz
        if (table is None) != (table_freq is None):
            raise ValueError(
                "tabulated_gains and tabulated_freq_hz must both be given or both None"
            )
        if table is not None:
            table = np.asarray(table, dtype=np.complex128)
            table_freq = np.asarray(table_freq, dtype=np.float64)
            if table_freq.ndim != 1 or table_freq.size < 1:
                raise ValueError(
                    f"tabulated_freq_hz must have shape (n_chan,), got {table_freq.shape}"
                )
            if not np.all(np.isfinite(table_freq)):
                raise ValueError("tabulated_freq_hz contains non-finite values")
            if table.shape != (scalar.size, table_freq.size):
                raise ValueError(
                    "tabulated_gains must have shape (n_antennas, n_chan) = "
                    f"({scalar.size}, {table_freq.size}), got {table.shape}"
                )
            if not np.all(np.isfinite(table)):
                raise ValueError("tabulated_gains contains non-finite values")
            object.__setattr__(self, "tabulated_gains", _readonly(table))
            object.__setattr__(self, "tabulated_freq_hz", _readonly(table_freq))

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def identity(cls, n_antennas: int) -> "InstrumentModel":
        """A perfectly uniform array: every gain exactly ``1 + 0j``.

        Parameters
        ----------
        n_antennas : int
            Number of antennas.

        Returns
        -------
        InstrumentModel
            A model whose `gains` are exactly one, so that applying it is
            a bit-for-bit no-op on complex voltages. Useful as the control
            case in tests and as a starting point for `with_scalar_gains`.
        """
        n_antennas = int(n_antennas)
        if n_antennas < 1:
            raise ValueError(f"n_antennas must be >= 1, got {n_antennas}")
        return cls(scalar_gains=np.ones(n_antennas, dtype=np.complex128))

    @classmethod
    def from_params(
        cls,
        n_antennas: int,
        *,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
        gain_scatter_db: float = 0.0,
        phase_offsets: Literal["zero", "uniform"] = "zero",
        bandpass_ripple_db: float = 0.0,
        bandpass_n_modes: int = DEFAULT_BANDPASS_N_MODES,
        freq_hz: np.ndarray | None = None,
    ) -> "InstrumentModel":
        """Draw a random, repeatable instrument model.

        Parameters
        ----------
        n_antennas : int
            Number of antennas. Must match the array the model is used
            with.
        rng : numpy.random.Generator, optional
            Generator to derive the model's seed from. Drawn from exactly
            once. Mutually exclusive with `seed`.
        seed : int, optional
            Seed to derive the model from. Mutually exclusive with `rng`.
            Either this or `rng` is required as soon as any stochastic
            feature is switched on.
        gain_scatter_db : float, optional
            Rms of the per-antenna power gain in dB (see the module
            docstring's dB convention): the amplitude of antenna ``i`` is
            ``10 ** (x_i / 20)`` with ``x_i`` drawn from a zero-mean
            normal of this width, i.e. a lognormal amplitude. Default
            0.0, meaning unit amplitude for every antenna.
        phase_offsets : {"zero", "uniform"}, optional
            ``"zero"`` (default) gives every antenna zero phase -- a
            perfectly phase-calibrated array. ``"uniform"`` draws each
            antenna's phase independently from ``[0, 2 pi)``, i.e. a
            completely uncalibrated array.
        bandpass_ripple_db : float, optional
            Rms, in dB, of the per-antenna smooth bandpass ripple across
            the band, taken over antennas and channels together. Default
            0.0 (flat bandpass).
        bandpass_n_modes : int, optional
            Number of cosine modes in the ripple; mode ``k`` completes
            ``k`` full cycles across the band. Default
            `DEFAULT_BANDPASS_N_MODES`.
        freq_hz : numpy.ndarray, optional
            Frequency grid the ripple's band should be referenced to. Only
            its minimum and maximum are used. If omitted, the band is
            taken from whatever grid `gains` is later evaluated on (see
            `band_hz`).

        Returns
        -------
        InstrumentModel
            An immutable model, fully determined by the seed, the antenna
            count and the reference band.

        Raises
        ------
        ValueError
            If both `rng` and `seed` are given; if neither is given while
            a stochastic feature is switched on; if `n_antennas` or
            `bandpass_n_modes` is not positive; if a scatter parameter is
            non-finite or negative; or if `phase_offsets` is not
            recognised.

        Notes
        -----
        The three effects draw from *independent* children of the model's
        root seed sequence, in the fixed order amplitude, phase, bandpass.
        Switching one effect on or changing its parameters therefore never
        perturbs the others: the same seed with and without a bandpass
        gives the identical amplitude scatter, which is what makes an
        ablation study over these parameters interpretable.
        """
        n_antennas = int(n_antennas)
        if n_antennas < 1:
            raise ValueError(f"n_antennas must be >= 1, got {n_antennas}")
        gain_scatter_db = float(gain_scatter_db)
        bandpass_ripple_db = float(bandpass_ripple_db)
        bandpass_n_modes = int(bandpass_n_modes)
        if not np.isfinite(gain_scatter_db) or gain_scatter_db < 0.0:
            raise ValueError(f"gain_scatter_db must be finite and >= 0, got {gain_scatter_db}")
        if not np.isfinite(bandpass_ripple_db) or bandpass_ripple_db < 0.0:
            raise ValueError(
                f"bandpass_ripple_db must be finite and >= 0, got {bandpass_ripple_db}"
            )
        if bandpass_n_modes < 1:
            raise ValueError(f"bandpass_n_modes must be >= 1, got {bandpass_n_modes}")
        if phase_offsets not in _PHASE_MODES:
            raise ValueError(f"phase_offsets must be one of {_PHASE_MODES}, got {phase_offsets!r}")
        if rng is not None and seed is not None:
            raise ValueError("give either rng or seed, not both")

        wants_randomness = (
            gain_scatter_db > 0.0 or phase_offsets != "zero" or bandpass_ripple_db > 0.0
        )
        if rng is None and seed is None:
            if wants_randomness:
                raise ValueError(
                    "from_params needs an rng or a seed to draw gain scatter, phase offsets "
                    "or bandpass ripple; a model with all three switched off needs neither"
                )
            root = np.random.SeedSequence(0)
        elif seed is not None:
            root = np.random.SeedSequence(seed)
        else:
            entropy = rng.integers(0, 2**63 - 1, size=4, dtype=np.int64)
            root = np.random.SeedSequence(entropy.tolist())

        # One independent child per effect, spawned unconditionally and in
        # a fixed order, so the effects never disturb one another.
        amp_seed, phase_seed, bandpass_seed = root.spawn(3)

        if gain_scatter_db > 0.0:
            power_db = np.random.default_rng(amp_seed).normal(
                loc=0.0, scale=gain_scatter_db, size=n_antennas
            )
            amplitude = 10.0 ** (power_db / 20.0)
        else:
            amplitude = np.ones(n_antennas, dtype=np.float64)

        if phase_offsets == "uniform":
            phase_rad = np.random.default_rng(phase_seed).uniform(0.0, 2.0 * np.pi, size=n_antennas)
        else:
            phase_rad = np.zeros(n_antennas, dtype=np.float64)

        scalar_gains = amplitude * np.exp(1j * phase_rad)

        cos_db = sin_db = None
        if bandpass_ripple_db > 0.0:
            # Coefficients of a real Fourier series in dB. Each mode's
            # contribution has variance (c**2 + s**2) / 2 across the band,
            # so drawing c, s ~ N(0, ripple / sqrt(n_modes)) makes the
            # total ripple variance ripple**2 in expectation.
            bandpass_rng = np.random.default_rng(bandpass_seed)
            per_mode_db = bandpass_ripple_db / np.sqrt(bandpass_n_modes)
            cos_db = bandpass_rng.normal(0.0, per_mode_db, size=(n_antennas, bandpass_n_modes))
            sin_db = bandpass_rng.normal(0.0, per_mode_db, size=(n_antennas, bandpass_n_modes))

        band_hz = None
        if freq_hz is not None:
            freq = np.asarray(freq_hz, dtype=np.float64)
            if freq.ndim != 1 or freq.size < 1:
                raise ValueError(f"freq_hz must have shape (n_chan,), got {freq.shape}")
            band_hz = (float(freq.min()), float(freq.max()))

        return cls(
            scalar_gains=scalar_gains,
            bandpass_cos_db=cos_db,
            bandpass_sin_db=sin_db,
            band_hz=band_hz,
        )

    @classmethod
    def from_gains(cls, gains: np.ndarray, freq_hz: np.ndarray | None = None) -> "InstrumentModel":
        """Wrap explicit, user-supplied gains.

        Parameters
        ----------
        gains : numpy.ndarray
            Either shape ``(n_antennas,)`` -- one frequency-independent
            complex gain per antenna -- or shape ``(n_antennas, n_chan)``
            -- a full gain table, which requires `freq_hz`.
        freq_hz : numpy.ndarray, optional
            Shape ``(n_chan,)`` frequency grid a 2-D `gains` table is
            defined on. Required for a 2-D table, rejected for a 1-D one.

        Returns
        -------
        InstrumentModel
            A model returning exactly these gains. A tabulated model is
            *not* interpolated: `gains` raises unless it is asked for the
            same grid, because silently interpolating a bandpass someone
            measured elsewhere is the kind of convenience that produces
            wrong answers quietly.

        Raises
        ------
        ValueError
            If `gains` is not 1-D or 2-D, if `freq_hz` is missing for a
            2-D table or supplied for a 1-D one, or if any value is
            non-finite.
        """
        gains = np.asarray(gains, dtype=np.complex128)
        if gains.ndim == 1:
            if freq_hz is not None:
                raise ValueError(
                    "freq_hz is meaningless for 1-D (frequency-independent) gains; omit it"
                )
            return cls(scalar_gains=gains)
        if gains.ndim == 2:
            if freq_hz is None:
                raise ValueError("a 2-D gain table requires the freq_hz grid it is defined on")
            return cls(
                scalar_gains=np.ones(gains.shape[0], dtype=np.complex128),
                tabulated_gains=gains,
                tabulated_freq_hz=freq_hz,
            )
        raise ValueError(
            f"gains must have shape (n_antennas,) or (n_antennas, n_chan), got {gains.shape}"
        )

    def with_band(self, freq_hz: np.ndarray) -> "InstrumentModel":
        """Return a copy whose bandpass ripple is referenced to `freq_hz`.

        Parameters
        ----------
        freq_hz : numpy.ndarray
            Shape ``(n_chan,)`` frequency grid; only its extremes are used.

        Returns
        -------
        InstrumentModel
            A new model with `band_hz` pinned, leaving every drawn
            coefficient untouched.
        """
        freq = np.asarray(freq_hz, dtype=np.float64)
        if freq.ndim != 1 or freq.size < 1:
            raise ValueError(f"freq_hz must have shape (n_chan,), got {freq.shape}")
        return replace(self, band_hz=(float(freq.min()), float(freq.max())))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @property
    def n_antennas(self) -> int:
        """int: Number of antennas the model describes."""
        return int(self.scalar_gains.size)

    @property
    def n_bandpass_modes(self) -> int:
        """int: Number of bandpass cosine modes (0 for a flat bandpass)."""
        return 0 if self.bandpass_cos_db is None else int(self.bandpass_cos_db.shape[1])

    @property
    def amplitude_db(self) -> np.ndarray:
        """numpy.ndarray: Frequency-independent power gain per antenna, dB.

        Shape ``(n_antennas,)``, equal to ``20 * log10(abs(scalar_gains))``
        -- the ground truth an amplitude-calibration step should recover
        (up to the usual overall scale degeneracy).
        """
        return 20.0 * np.log10(np.abs(self.scalar_gains))

    @property
    def phase_rad(self) -> np.ndarray:
        """numpy.ndarray: Frequency-independent phase per antenna, radians.

        Shape ``(n_antennas,)`` in ``(-pi, pi]``, the ground truth a
        phase-calibration step should recover (up to a global phase).
        """
        return np.angle(self.scalar_gains)

    def _band_edges(self, freq_hz: np.ndarray) -> tuple[float, float]:
        """Reference band for the bandpass ripple on grid `freq_hz`."""
        if self.band_hz is not None:
            return self.band_hz
        return float(freq_hz.min()), float(freq_hz.max())

    def bandpass_db(self, freq_hz: np.ndarray) -> np.ndarray:
        """Per-antenna bandpass ripple in dB on the grid `freq_hz`.

        Parameters
        ----------
        freq_hz : numpy.ndarray
            Shape ``(n_chan,)`` frequency grid, Hz.

        Returns
        -------
        numpy.ndarray
            Float64 array of shape ``(n_antennas, n_chan)``: the power
            response of each antenna in dB relative to its own mean level.
            All zeros for a flat-bandpass model.

        Notes
        -----
        Mode ``k`` completes ``k`` full cycles across the reference band,
        so every mode integrates to zero over the band and the ripple adds
        no net power -- `amplitude_db` alone sets each antenna's mean
        level.
        """
        freq = np.asarray(freq_hz, dtype=np.float64)
        if freq.ndim != 1 or freq.size < 1:
            raise ValueError(f"freq_hz must have shape (n_chan,), got {freq.shape}")
        if not np.all(np.isfinite(freq)):
            raise ValueError("freq_hz contains non-finite values")
        if self.bandpass_cos_db is None:
            return np.zeros((self.n_antennas, freq.size), dtype=np.float64)

        low, high = self._band_edges(freq)
        span = high - low
        # A zero-width band (one channel, or a degenerate grid) has no
        # frequency structure to express: fall back to the band's start.
        fraction = np.zeros_like(freq) if span == 0.0 else (freq - low) / span
        modes = np.arange(1, self.n_bandpass_modes + 1, dtype=np.float64)
        angle = 2.0 * np.pi * fraction[:, np.newaxis] * modes[np.newaxis, :]  # (n_chan, n_modes)
        return self.bandpass_cos_db @ np.cos(angle).T + self.bandpass_sin_db @ np.sin(angle).T

    def gains(self, freq_hz: np.ndarray) -> np.ndarray:
        """Complex gain of every antenna on the grid `freq_hz`.

        Parameters
        ----------
        freq_hz : numpy.ndarray
            Shape ``(n_chan,)`` RF channel center frequencies, Hz.

        Returns
        -------
        numpy.ndarray
            Complex128 array of shape ``(n_antennas, n_chan)``. This is
            the ground-truth :math:`g_i(f)` of the module docstring:
            measured visibilities are ``g_i g_j*`` times the true ones.

        Raises
        ------
        ValueError
            If `freq_hz` is not a 1-D non-empty array, if it contains a
            non-finite value, or if this is a tabulated model and
            `freq_hz` differs from the grid the table was supplied on.
        """
        freq = np.asarray(freq_hz, dtype=np.float64)
        if freq.ndim != 1 or freq.size < 1:
            raise ValueError(f"freq_hz must have shape (n_chan,), got {freq.shape}")
        if not np.all(np.isfinite(freq)):
            raise ValueError("freq_hz contains non-finite values")

        if self.tabulated_gains is not None:
            if freq.shape != self.tabulated_freq_hz.shape or not np.allclose(
                freq, self.tabulated_freq_hz, rtol=1e-12, atol=0.0
            ):
                raise ValueError(
                    "this InstrumentModel carries a tabulated gain table and does not "
                    "interpolate: gains() must be called with the same freq_hz grid the "
                    f"table was built on ({self.tabulated_freq_hz.size} channels)"
                )
            return np.array(self.tabulated_gains, copy=True)

        out = np.repeat(self.scalar_gains[:, np.newaxis], freq.size, axis=1)
        if self.bandpass_cos_db is not None:
            out = out * 10.0 ** (self.bandpass_db(freq) / 20.0)
        return out
