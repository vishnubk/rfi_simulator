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
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from astropy.time import Time

from rfi_simulator.voltages import VoltageBlock

__all__ = ["Visibilities", "baseline_index_pairs", "correlate"]


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
    """Fringe-stopped visibilities for a whole observation.

    Attributes
    ----------
    data : numpy.ndarray
        Complex64 array of shape ``(n_int, n_baselines, n_chan)`` in
        janskys, with the ``V_ij = <v_i v_j*>`` convention.
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

    def __post_init__(self) -> None:
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
        return self.data.shape[2]

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

    Returns
    -------
    Visibilities
        Shape ``(n_int, n_baselines, n_chan)`` visibilities in Jy.

    Raises
    ------
    ValueError
        If `blocks` is empty, or if the blocks do not all carry the same
        interference-source labels.

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

    for block in blocks:
        if first is None:
            first = block
            pairs = baseline_index_pairs(block.n_antennas, include_autos=include_autos)
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

        voltages = np.ascontiguousarray(np.transpose(block.data, (1, 0, 2)))
        # (n_chan, n_ant, n_ant): V[c, i, j] = sum_t v_i v_j^*
        products = voltages @ np.conjugate(np.transpose(voltages, (0, 2, 1)))
        products /= np.float32(block.n_time)

        vis = products[:, pairs[:, 0], pairs[:, 1]]  # (n_chan, n_base)
        vis = np.ascontiguousarray(vis.T)  # (n_base, n_chan)

        if fringe_stop:
            tau = block.phase_center_delays_s
            delta_tau = tau[pairs[:, 0]] - tau[pairs[:, 1]]  # (n_base,)
            stop = np.exp(
                2j * np.pi * delta_tau[:, np.newaxis] * block.freq_hz[np.newaxis, :]
            ).astype(np.complex64)
            vis = vis * stop

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
    )
