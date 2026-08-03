r"""FX correlator: channelized voltages in, fringe-stopped visibilities out.

The X step is the binding definition of a visibility in this package:

.. math::

    V_{ij}(f) = \frac{1}{N_t} \sum_t v_i(f, t)\, v_j^*(f, t),

i.e. **the conjugate goes on the second antenna** of the pair, with
``i <= j``. The normalization is a mean rather than a bare sum so that
visibilities come out directly in janskys: a single source of flux
``F`` gives ``|V_ij| = F``, and an autocorrelation gives
``sum(F) + noise_std**2``.

Fringe stopping (tracking the phase center) is applied here, per the
By construction, the voltages carry the absolute geometric delays and the
correlator multiplies by

.. math::

    e^{+2\pi i f (\tau_{0,i} - \tau_{0,j})},

with :math:`\tau_0` the phase-center delays of the block. A source sitting
exactly at the phase center therefore gives a real, positive, constant
visibility.

All ``n_ant * (n_ant + 1) / 2`` pairs are produced, autocorrelations
included.

Polarization
------------
Blocks carrying two receptors (`rfi_simulator.voltages.VoltageSimulator`
built with ``n_pol=2``) correlate into the **parallel-hand** products
``XX`` and ``YY`` only: :math:`V^{pp}_{ij} = \langle v^p_i v^{p*}_j
\rangle` for each receptor ``p`` separately. The cross-hands ``XY`` and
``YX`` are deliberately not formed in this version. They carry the
linear polarization of the *sky*, which this simulator does not model --
every celestial component here is unpolarized by construction -- so
computing them would produce two arrays of pure noise plus whatever
interference leaks into them, at twice the correlation cost and with a
tempting but meaningless calibration target attached.

The schema is nonetheless shaped so that adding them later is additive
rather than breaking: `Visibilities.data` carries an explicit
polarization axis labelled by `Visibilities.pol_names`, so ``("XX",
"YY")`` becomes ``("XX", "XY", "YX", "YY")`` with no change to the field
layout, and every consumer that selects by name keeps working.

As with the voltages, the polarization axis exists only when there is
more than one receptor: a single-polarization run gives exactly the
three-dimensional ``(n_int, n_baselines, n_chan)`` array it always did.
Stokes I is formed by `stokes_i` as ``(XX + YY) / 2``, the convention
under which a dual-polarization run of a scene has the same flux scale
as the single-polarization run of that same scene.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from astropy.time import Time

from rfi_simulator.calibration import CalibrationErrors, resolve_calibration_error_models
from rfi_simulator.voltages import VoltageBlock

__all__ = ["PARALLEL_HAND_NAMES", "Visibilities", "baseline_index_pairs", "correlate"]

PARALLEL_HAND_NAMES = ("XX", "YY")
"""tuple of str: Names of the parallel-hand products, in receptor order.

`correlate` labels a dual-polarization dataset's polarization axis with
these. The cross-hands would slot in between them as ``("XX", "XY",
"YX", "YY")`` -- see the module docstring."""


def baseline_index_pairs(n_antennas: int, include_autos: bool = True) -> np.ndarray:
    """All antenna-pair indices ``(i, j)`` with ``i <= j`` (or ``i < j``).

    Parameters
    ----------
    n_antennas : int
        Number of antennas.
    include_autos : bool, optional
        If True (default) include the ``i == j`` autocorrelations.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_baselines, 2)`` int array of antenna index pairs, in
        row-major upper-triangular order.
    """
    i_idx, j_idx = np.triu_indices(n_antennas, k=0 if include_autos else 1)
    return np.stack([i_idx, j_idx], axis=1)


@dataclass
class Visibilities:
    r"""Fringe-stopped visibilities for a whole observation.

    Attributes
    ----------
    data : numpy.ndarray
        Complex64 array in janskys, with the ``V_ij = <v_i v_j*>``
        convention: shape ``(n_int, n_baselines, n_chan)`` for
        single-polarization data and ``(n_int, n_baselines, n_pol,
        n_chan)`` for dual-polarization data, the polarization axis
        labelled by `pol_names`. Use `pol_data` for a view that always
        carries the axis and `stokes_i` for the pseudo-Stokes-I
        combination.
    ant_1 : numpy.ndarray
        Shape ``(n_baselines,)`` int array of first-antenna indices.
    ant_2 : numpy.ndarray
        Shape ``(n_baselines,)`` int array of second-antenna indices (the
        conjugated one). ``ant_1 <= ant_2`` always.
    freq_hz : numpy.ndarray
        Shape ``(n_chan,)`` RF channel center frequencies, Hz.
    time_mjd : numpy.ndarray
        Shape ``(n_int,)`` float64 MJD (UTC) of each integration center.
    integration_time_s : float
        Duration of one integration, seconds.
    n_samples : int
        Number of voltage samples averaged into each integration
        (``integration_time_s * chan_width_hz``).
    baseline_vectors_enu_m : numpy.ndarray
        Shape ``(n_baselines, 3)`` baseline vectors ``r_i - r_j`` in ENU
        meters -- the same ordering convention as the conjugation.
    e_l_enu : numpy.ndarray
        Shape ``(n_int, 3)`` ENU unit vector along increasing ``l`` at
        each integration center.
    e_m_enu : numpy.ndarray
        Shape ``(n_int, 3)`` ENU unit vector along increasing ``m`` at
        each integration center.
    s0_enu : numpy.ndarray
        Shape ``(n_int, 3)`` ENU unit vector towards the phase center at
        each integration center.
    rfi_fraction : numpy.ndarray
        Float64 ground-truth labels of shape
        ``(n_int, n_interference_sources, n_chan)``: the fraction of the
        integration's voltage samples in which each source occupied each
        channel, in ``[0, 1]``. This is the voltage-resolution occupancy
        mask averaged down to the integration grid, which is the only form
        of it that survives correlation.

        A run with no interference sources gives a **zero-sized** array of
        shape ``(n_int, 0, n_chan)`` rather than a zero-filled one: there
        is no source to report a fraction for, and the source axis then
        stacks and concatenates consistently with contaminated runs. Both
        ``rfi_fraction.max(initial=0.0)`` and ``(rfi_fraction == 0).all()``
        behave as expected on it.
    rfi_source_names : tuple of str
        Names of the interference sources, in the order of
        `rfi_fraction`'s middle axis.
    celestial_fraction : numpy.ndarray
        Float64 ground-truth labels of shape ``(n_int, n_celestial_sources,
        n_chan)``, the same voltage-resolution-to-integration averaging as
        `rfi_fraction` but for `rfi_simulator.sky.SpectralLineForeground`
        celestial labels -- kept in a field of its own so the two classes
        never share an axis. A run with no spectral lines gives a
        zero-sized ``(n_int, 0, n_chan)`` array, matching the
        `rfi_fraction` convention.
    celestial_source_names : tuple of str
        Names of the spectral-line foregrounds, in the order of
        `celestial_fraction`'s middle axis.
    pol_names : tuple of str
        Names of the polarization products, in the order of `data`'s
        polarization axis: ``()`` for single-polarization data and
        ``("XX", "YY")`` for dual-polarization data. The place a future
        cross-hand product would be added.
    rfi_polarization : numpy.ndarray, optional
        Complex128 ground truth of shape ``(n_interference_sources,
        n_pol)``: the per-receptor amplitude each source was received
        with, carried through from `VoltageBlock.rfi_polarization`, or
        ``None`` when the blocks did not carry it. Deliberately *not*
        folded into `rfi_fraction`: occupancy is pol-independent (a
        transmitter occupies a channel whichever receptor hears it best)
        and the amplitude asymmetry is a separate, multiplicative fact
        about the same cell. A per-receptor power weight for source ``s``
        in receptor ``p`` is ``abs(rfi_polarization[s, p])**2``.
    calibration_error_gains : numpy.ndarray, optional
        Complex128 ground truth of the residual calibration error applied
        by `correlate`'s ``calibration_errors=`` argument (see
        `rfi_simulator.calibration.CalibrationErrors`): shape
        ``(n_antennas, n_chan)`` for single-polarization data or
        ``(n_antennas, n_pol, n_chan)`` for dual-polarization data, the
        per-antenna factor :math:`c_i(f)` this dataset's visibilities
        were multiplied by (as :math:`c_i(f)\, c_j(f)^*` on each
        baseline). ``None`` when `correlate` was not given
        `calibration_errors`, i.e. exactly the same data as a perfectly
        calibrated run. This is what a calibration exercise built against
        this simulator is supposed to recover -- and is unrelated to the
        *true* per-antenna gains an `rfi_simulator.instrument`-equipped
        run carries on `rfi_simulator.voltages.VoltageBlock.gains`: this
        field lives entirely downstream of the true instrument, at the
        calibration-solution layer.
    """

    data: np.ndarray
    ant_1: np.ndarray
    ant_2: np.ndarray
    freq_hz: np.ndarray
    time_mjd: np.ndarray
    integration_time_s: float
    n_samples: int
    baseline_vectors_enu_m: np.ndarray
    e_l_enu: np.ndarray
    e_m_enu: np.ndarray
    s0_enu: np.ndarray
    rfi_fraction: np.ndarray | None = None
    rfi_source_names: tuple[str, ...] = field(default_factory=tuple)
    celestial_fraction: np.ndarray | None = None
    celestial_source_names: tuple[str, ...] = field(default_factory=tuple)
    pol_names: tuple[str, ...] = field(default_factory=tuple)
    rfi_polarization: np.ndarray | None = None
    calibration_error_gains: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.data.ndim not in (3, 4):
            raise ValueError(
                "data must have shape (n_int, n_baselines, n_chan) or "
                f"(n_int, n_baselines, n_pol, n_chan), got {self.data.shape}"
            )
        self.pol_names = tuple(str(name) for name in self.pol_names)
        if self.pol_names and len(self.pol_names) != self.n_pol:
            raise ValueError(
                f"pol_names has {len(self.pol_names)} entries but the data carries "
                f"{self.n_pol} polarizations"
            )
        if self.data.ndim == 4 and not self.pol_names:
            raise ValueError(
                'dual-polarization visibilities must be labelled: pass pol_names, e.g. ("XX", "YY")'
            )
        if self.rfi_polarization is not None and self.rfi_polarization.shape != (
            len(self.rfi_source_names),
            self.n_pol,
        ):
            raise ValueError(
                "rfi_polarization must have shape (n_sources, n_pol) = "
                f"({len(self.rfi_source_names)}, {self.n_pol}), "
                f"got {self.rfi_polarization.shape}"
            )
        if self.calibration_error_gains is not None:
            expected_tail = (self.n_chan,) if self.n_pol == 1 else (self.n_pol, self.n_chan)
            if self.calibration_error_gains.shape[1:] != expected_tail:
                raise ValueError(
                    "calibration_error_gains must have shape (n_antennas, n_chan) or "
                    "(n_antennas, n_pol, n_chan) matching this dataset, got "
                    f"{self.calibration_error_gains.shape}"
                )
        if self.rfi_fraction is None:
            self.rfi_fraction = np.zeros((self.n_int, 0, self.n_chan), dtype=np.float64)
        self.rfi_source_names = tuple(self.rfi_source_names)
        if self.celestial_fraction is None:
            self.celestial_fraction = np.zeros((self.n_int, 0, self.n_chan), dtype=np.float64)
        self.celestial_source_names = tuple(self.celestial_source_names)

    @property
    def n_rfi_sources(self) -> int:
        """int: Number of interference sources labelled in this dataset."""
        return self.rfi_fraction.shape[1]

    @property
    def n_celestial_sources(self) -> int:
        """int: Number of spectral-line foregrounds labelled in this dataset."""
        return self.celestial_fraction.shape[1]

    @property
    def n_int(self) -> int:
        """int: Number of integrations."""
        return self.data.shape[0]

    @property
    def n_baselines(self) -> int:
        """int: Number of antenna pairs, autocorrelations included."""
        return self.data.shape[1]

    @property
    def n_chan(self) -> int:
        """int: Number of frequency channels."""
        return self.data.shape[-1]

    @property
    def n_pol(self) -> int:
        """int: Number of polarization products."""
        return 1 if self.data.ndim == 3 else self.data.shape[2]

    @property
    def pol_data(self) -> np.ndarray:
        """numpy.ndarray: `data` with the polarization axis always present.

        Shape ``(n_int, n_baselines, n_pol, n_chan)``, a *view* of `data`
        -- reshaped for single-polarization data, `data` itself otherwise.
        """
        if self.data.ndim == 3:
            return self.data.reshape(*self.data.shape[:2], 1, self.data.shape[2])
        return self.data

    def pol_index(self, pol: str | int) -> int:
        """Index of a polarization product on `data`'s polarization axis.

        Parameters
        ----------
        pol : str or int
            A name from `pol_names`, or an integer index.

        Returns
        -------
        int
            Index into the polarization axis of `pol_data`.

        Raises
        ------
        KeyError
            If a name is not among `pol_names`.
        IndexError
            If an integer index is out of range.
        """
        if isinstance(pol, str):
            if pol not in self.pol_names:
                raise KeyError(f"polarization {pol!r} not present; have {self.pol_names}")
            return self.pol_names.index(pol)
        index = int(pol)
        if not 0 <= index < self.n_pol:
            raise IndexError(f"polarization index {index} out of range [0, {self.n_pol})")
        return index

    def stokes_i(self) -> np.ndarray:
        """Pseudo-Stokes-I visibilities, ``(XX + YY) / 2``.

        Returns
        -------
        numpy.ndarray
            Complex64 array of shape ``(n_int, n_baselines, n_chan)``: the
            mean of the parallel-hand products, which for
            single-polarization data is simply `data` itself.

        Notes
        -----
        The mean, not the sum, because each receptor of this simulator is
        calibrated in Stokes-I janskys: an unpolarized source of flux
        ``F`` gives ``XX = YY = F``, so the mean recovers ``F`` and a
        dual-polarization image has the same flux scale as the
        single-polarization image of the same scene (see
        `rfi_simulator.voltages`). A fully polarized interferer, by
        contrast, is *not* attenuated by this combination: it puts all of
        its Stokes I into one receptor, and half of that survives -- which
        is why polarization-based flagging happens before Stokes I is
        formed, not after.
        """
        if self.data.ndim == 3:
            return self.data
        return self.data.mean(axis=2).astype(np.complex64)

    @property
    def auto_mask(self) -> np.ndarray:
        """numpy.ndarray: Boolean mask of shape ``(n_baselines,)``, True for autos."""
        return self.ant_1 == self.ant_2

    @property
    def cross_mask(self) -> np.ndarray:
        """numpy.ndarray: Boolean mask of shape ``(n_baselines,)``, True for crosses."""
        return self.ant_1 != self.ant_2

    def baseline_index(self, ant_1: int, ant_2: int) -> int:
        """Row index of the ``(ant_1, ant_2)`` pair in `data`.

        Parameters
        ----------
        ant_1, ant_2 : int
            Antenna indices; order does not matter, but the returned row
            always corresponds to the stored ``i <= j`` pair.

        Returns
        -------
        int
            Index into the baseline axis of `data`.

        Raises
        ------
        KeyError
            If the pair is not present.
        """
        low, high = min(ant_1, ant_2), max(ant_1, ant_2)
        matches = np.flatnonzero((self.ant_1 == low) & (self.ant_2 == high))
        if matches.size == 0:
            raise KeyError(f"baseline ({ant_1}, {ant_2}) not present")
        return int(matches[0])


def correlate(
    blocks: Iterable[VoltageBlock],
    *,
    fringe_stop: bool = True,
    include_autos: bool = True,
    calibration_errors: CalibrationErrors | list[CalibrationErrors] | None = None,
) -> Visibilities:
    """Correlate a stream of voltage blocks into visibilities.

    One block becomes one integration.

    Parameters
    ----------
    blocks : iterable of VoltageBlock
        Blocks in time order, e.g. ``VoltageSimulator.blocks()``.
    fringe_stop : bool, optional
        If True (default) rotate out the phase-center geometric delay of
        each block, so that the phase center sits at zero fringe rate.
    include_autos : bool, optional
        If True (default) keep the ``i == j`` autocorrelations.
    calibration_errors : CalibrationErrors or sequence of CalibrationErrors, optional
        Residual calibration error to apply (see
        `rfi_simulator.calibration.CalibrationErrors`), a single model
        applied to every polarization or one model per polarization.
        Default ``None``: perfect calibration, bit-identical to the data
        this function produced before the feature existed. Applied to
        every baseline's visibility as :math:`c_i(f)\\, c_j(f)^*`,
        *after* fringe stopping (the two commute: both are per-baseline,
        per-frequency multiplicative factors) and purely at the
        visibility level -- it never touches the blocks' voltages or
        `rfi_simulator.instrument.InstrumentModel`'s true-gain ground
        truth. The applied factors are recorded on
        `Visibilities.calibration_error_gains`.

    Returns
    -------
    Visibilities
        Shape ``(n_int, n_baselines, n_chan)`` visibilities in Jy, or
        ``(n_int, n_baselines, n_pol, n_chan)`` for dual-polarization
        blocks, whose parallel-hand products are labelled ``("XX",
        "YY")`` -- see the module docstring on why the cross-hands are
        not formed.

    Raises
    ------
    ValueError
        If `blocks` is empty, if the blocks do not all carry the same
        interference-source labels, if they do not all carry the same
        number of polarizations, or if `calibration_errors` describes a
        different number of antennas than the blocks.

    Notes
    -----
    The per-block work is ``n_chan`` small matrix products
    ``v v^H`` of shape ``(n_ant, n_time) @ (n_time, n_ant)``, which BLAS
    handles at full speed; the correlator is not the bottleneck of the
    simulator.
    """
    accumulated: list[np.ndarray] = []
    times_mjd: list[float] = []
    e_l: list[np.ndarray] = []
    e_m: list[np.ndarray] = []
    s0: list[np.ndarray] = []
    rfi_fraction: list[np.ndarray] = []
    celestial_fraction: list[np.ndarray] = []

    pairs = None
    first: VoltageBlock | None = None
    # (n_antennas, n_pol, n_chan) residual calibration factors, evaluated
    # once against the first block's frequency grid -- like the fringe
    # geometry, the residual error is a property of the antennas and the
    # band, not of any one block.
    cal_gains: np.ndarray | None = None

    for block in blocks:
        if first is None:
            first = block
            pairs = baseline_index_pairs(block.n_antennas, include_autos=include_autos)
            if calibration_errors is not None:
                cal_models = resolve_calibration_error_models(calibration_errors, block.n_pol)
                for model in cal_models:
                    if model.n_antennas != block.n_antennas:
                        raise ValueError(
                            f"calibration_errors describes {model.n_antennas} antennas but "
                            f"the data has {block.n_antennas}"
                        )
                cal_gains = np.stack(
                    [model.factors(block.freq_hz).astype(np.complex64) for model in cal_models],
                    axis=1,
                )
        elif block.n_pol != first.n_pol:
            raise ValueError(
                "all blocks must carry the same number of polarizations, got "
                f"{first.n_pol} then {block.n_pol}"
            )
        elif block.rfi_source_names != first.rfi_source_names:
            raise ValueError(
                "all blocks must carry the same interference-source labels, got "
                f"{first.rfi_source_names} then {block.rfi_source_names}"
            )
        elif block.celestial_source_names != first.celestial_source_names:
            raise ValueError(
                "all blocks must carry the same celestial-source labels, got "
                f"{first.celestial_source_names} then {block.celestial_source_names}"
            )

        # (n_pol, n_chan, n_ant, n_time). Each receptor is correlated with
        # itself only -- the parallel hands -- so the polarization axis is
        # just another batch dimension of the same matrix product.
        voltages = np.ascontiguousarray(np.transpose(block.pol_data, (1, 2, 0, 3)))
        # (n_pol, n_chan, n_ant, n_ant): V[p, c, i, j] = sum_t v_i v_j^*
        products = voltages @ np.conjugate(np.transpose(voltages, (0, 1, 3, 2)))
        products /= np.float32(block.n_time)

        vis = products[:, :, pairs[:, 0], pairs[:, 1]]  # (n_pol, n_chan, n_base)
        vis = np.ascontiguousarray(np.transpose(vis, (2, 0, 1)))  # (n_base, n_pol, n_chan)

        if fringe_stop:
            tau = block.phase_center_delays_s
            delta_tau = tau[pairs[:, 0]] - tau[pairs[:, 1]]  # (n_base,)
            stop = np.exp(
                2j * np.pi * delta_tau[:, np.newaxis] * block.freq_hz[np.newaxis, :]
            ).astype(np.complex64)
            # The geometry is the same for both receptors: one fringe, two
            # streams.
            vis = vis * stop[:, np.newaxis, :]

        if cal_gains is not None:
            # (n_base, n_pol, n_chan): the baseline structure calibration
            # divides back out, exactly like InstrumentModel's true gains
            # (see the module docstring), applied here at the visibility
            # level instead of the voltage level.
            c_i = cal_gains[pairs[:, 0]]
            c_j = cal_gains[pairs[:, 1]]
            vis = vis * (c_i * np.conjugate(c_j))

        if block.n_pol == 1:
            vis = vis.reshape(vis.shape[0], vis.shape[2])

        accumulated.append(vis)
        times_mjd.append(float(Time(block.center_time).utc.mjd))
        e_l.append(np.asarray(block.e_l_enu, dtype=np.float64))
        e_m.append(np.asarray(block.e_m_enu, dtype=np.float64))
        s0.append(np.asarray(block.s0_enu, dtype=np.float64))
        # Labels ride along untouched by the correlation itself: the
        # voltage-resolution mask simply averages down the time axis.
        rfi_fraction.append(block.rfi_mask.mean(axis=2, dtype=np.float64))
        celestial_fraction.append(block.celestial_mask.mean(axis=2, dtype=np.float64))

    if first is None:
        raise ValueError("correlate() received no voltage blocks")

    positions = first.antenna_positions_enu_m
    baseline_vectors = positions[pairs[:, 0]] - positions[pairs[:, 1]]
    pol_names = () if first.n_pol == 1 else PARALLEL_HAND_NAMES[: first.n_pol]

    calibration_error_gains = None
    if cal_gains is not None:
        calibration_error_gains = cal_gains.copy()
        if first.n_pol == 1:
            calibration_error_gains = calibration_error_gains.reshape(
                calibration_error_gains.shape[0], calibration_error_gains.shape[2]
            )

    return Visibilities(
        data=np.stack(accumulated, axis=0),
        ant_1=pairs[:, 0].copy(),
        ant_2=pairs[:, 1].copy(),
        freq_hz=np.asarray(first.freq_hz, dtype=np.float64),
        time_mjd=np.asarray(times_mjd, dtype=np.float64),
        integration_time_s=first.duration_s,
        n_samples=first.n_time,
        baseline_vectors_enu_m=baseline_vectors,
        e_l_enu=np.stack(e_l, axis=0),
        e_m_enu=np.stack(e_m, axis=0),
        s0_enu=np.stack(s0, axis=0),
        rfi_fraction=np.stack(rfi_fraction, axis=0),
        rfi_source_names=first.rfi_source_names,
        celestial_fraction=np.stack(celestial_fraction, axis=0),
        celestial_source_names=first.celestial_source_names,
        pol_names=pol_names,
        rfi_polarization=first.rfi_polarization,
        calibration_error_gains=calibration_error_gains,
    )
