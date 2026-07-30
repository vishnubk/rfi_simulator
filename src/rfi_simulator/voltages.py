r"""Per-antenna channelized voltage synthesis.

The simulator produces *channelized* complex baseband voltages: for every
antenna, every frequency channel and every post-channelization time sample
it emits one complex number. A perfect channelizer is assumed (no PFB
leakage -- see the deliberate MVP simplifications in
the module docstrings of ``delays`` and ``correlator``).

Synthesis is done in the frequency domain, which is exact for noise-like
celestial sources and needs no fractional-delay filtering:

.. math::

    v_i(f, t) = \sum_{\mathrm{src}} \sqrt{F_{\mathrm{src}}}\,
                S_{\mathrm{src}}(f, t)\, e^{-2\pi i f \tau_{\mathrm{src},i}(t)}
                + \sigma\, n_i(f, t).

Three things in that expression are easy to get wrong and are each covered
by a test:

* ``f`` is the **RF (sky) frequency of the channel**, ``center + offset``,
  never the baseband offset alone. Using the offset alone still produces
  fringes, but places the source at the wrong ``(l, m)`` in a way that
  varies across the band.
* :math:`S_{\mathrm{src}}(f, t)` is **one realization shared by all
  antennas** -- it is a single sky signal. The receiver noise
  :math:`n_i(f, t)` is drawn **independently per antenna**. Swapping these
  either destroys the fringes or manufactures fake correlation.
* :math:`\tau_{\mathrm{src},i}(t)` is re-evaluated **per block** with
  astropy, so Earth rotation is in the code path.

Amplitude convention: the source spectrum is circular complex Gaussian
with mean square ``flux_jy``, so a noiseless, fringe-stopped visibility
has amplitude ``flux_jy`` and an autocorrelation equals
``sum(flux_jy) + noise_std**2``. Everything downstream is therefore in
janskys.

Interference sources (`rfi_simulator.rfi`) add to the same voltages,
after the sky and the receiver noise, and each block carries the
resulting ground-truth occupancy masks. They draw from a **separate
branch** of the block seed sequence, so a run with interference and a run
without it share the identical sky-plus-noise realization for a given
seed -- which is what makes a clean/contaminated pair a usable training
target rather than two unrelated datasets.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from rfi_simulator.array_config import ArrayConfig, _to_value
from rfi_simulator.delays import (
    earth_location,
    geometric_delays_s,
    lm_basis_enu,
    source_unit_vectors_enu,
)
from rfi_simulator.rfi import BlockContext, RFISource
from rfi_simulator.sky import PointSource

__all__ = ["VoltageBlock", "VoltageSimulator"]

DEFAULT_CENTER_FREQ_HZ = 1.405e9
DEFAULT_N_CHAN = 384
DEFAULT_CHAN_WIDTH_HZ = 30517.578125
DEFAULT_N_TIME_PER_BLOCK = 1000
DEFAULT_N_BLOCKS = 61


@dataclass
class VoltageBlock:
    """One block of channelized voltages plus the metadata to correlate it.

    A block is the natural unit of the simulation: delays are frozen
    within a block and re-evaluated between blocks, and one block becomes
    one correlator integration.

    Attributes
    ----------
    data : numpy.ndarray
        Complex64 voltages of shape ``(n_antennas, n_chan, n_time)``, in
        units of sqrt(Jy) (so ``|v|**2`` is in Jy).
    time : astropy.time.Time
        Start time (UTC) of the block.
    center_time : astropy.time.Time
        Mid-point time (UTC) of the block -- the epoch at which the
        block's delays and ``(l, m)`` basis were evaluated.
    freq_hz : numpy.ndarray
        Shape ``(n_chan,)`` float64 array of RF channel center
        frequencies, Hz, ascending.
    sample_period_s : float
        Post-channelization sample period, seconds.
    phase_center_delays_s : numpy.ndarray
        Shape ``(n_antennas,)`` float64 geometric delays towards the phase
        center at `center_time`, seconds. The correlator uses these to
        stop the fringe.
    antenna_positions_enu_m : numpy.ndarray
        Shape ``(n_antennas, 3)`` antenna positions, ENU meters.
    e_l_enu : numpy.ndarray
        Shape ``(3,)`` ENU unit vector along increasing ``l`` at
        `center_time`.
    e_m_enu : numpy.ndarray
        Shape ``(3,)`` ENU unit vector along increasing ``m`` at
        `center_time`.
    s0_enu : numpy.ndarray
        Shape ``(3,)`` ENU unit vector towards the phase center at
        `center_time`. Imaging uses it to size the neglected ``w`` term.
    rfi_mask : numpy.ndarray, optional
        Boolean ground-truth labels of shape
        ``(n_interference_sources, n_chan, n_time)``: ``rfi_mask[s, c, t]``
        is True where source ``s`` occupies cell ``(c, t)``. See
        `rfi_simulator.rfi` for the occupancy convention. Defaults to an
        empty ``(0, n_chan, n_time)`` array, i.e. an interference-free
        block -- the leading axis carries the source count, so a clean
        block is *labelled* clean rather than unlabelled.
    rfi_source_names : tuple of str, optional
        Names of the interference sources, in the order of `rfi_mask`'s
        leading axis. Defaults to ``()``.
    """

    data: np.ndarray
    time: Time
    center_time: Time
    freq_hz: np.ndarray
    sample_period_s: float
    phase_center_delays_s: np.ndarray
    antenna_positions_enu_m: np.ndarray
    e_l_enu: np.ndarray
    e_m_enu: np.ndarray
    s0_enu: np.ndarray
    rfi_mask: np.ndarray | None = None
    rfi_source_names: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.rfi_mask is None:
            self.rfi_mask = np.zeros((0, self.n_chan, self.n_time), dtype=bool)
        self.rfi_source_names = tuple(self.rfi_source_names)
        if self.rfi_mask.shape != (len(self.rfi_source_names), self.n_chan, self.n_time):
            raise ValueError(
                "rfi_mask must have shape (n_sources, n_chan, n_time) = "
                f"({len(self.rfi_source_names)}, {self.n_chan}, {self.n_time}), "
                f"got {self.rfi_mask.shape}"
            )

    @property
    def n_rfi_sources(self) -> int:
        """int: Number of interference sources labelled in this block."""
        return self.rfi_mask.shape[0]

    @property
    def n_antennas(self) -> int:
        """int: Number of antennas."""
        return self.data.shape[0]

    @property
    def n_chan(self) -> int:
        """int: Number of frequency channels."""
        return self.data.shape[1]

    @property
    def n_time(self) -> int:
        """int: Number of time samples in the block."""
        return self.data.shape[2]

    @property
    def duration_s(self) -> float:
        """float: Block duration in seconds."""
        return self.n_time * self.sample_period_s


class VoltageSimulator:
    """Generate channelized voltages for a point-source sky plus receiver noise.

    Parameters
    ----------
    array : ArrayConfig
        Antenna geometry and site location.
    phase_center : astropy.coordinates.SkyCoord
        Scalar phase-center coordinate (ICRS). The array tracks this
        direction; the correlator stops the fringe on it.
    start_time : astropy.time.Time
        Scalar UTC start time of the observation.
    sources : sequence of PointSource, optional
        Sky model. An empty sequence gives a noise-only observation, which
        is what the radiometer test uses. Defaults to ``()``.
    rfi_sources : sequence of RFISource, optional
        Interference sources (see `rfi_simulator.rfi`). Their
        contributions are added to the sky-plus-noise voltages and their
        occupancy masks are attached to every block. Defaults to ``()``.
        Adding or removing them never perturbs the sky-plus-noise
        realization of a given seed.
    center_freq_hz : float or astropy.units.Quantity, optional
        Band center frequency in Hz. Default 1.405 GHz (L band).
    n_chan : int, optional
        Number of frequency channels. Default 384.
    chan_width_hz : float or astropy.units.Quantity, optional
        Channel width in Hz. Default 30517.578125 Hz, so that
        ``n_chan * chan_width`` matches a typical L-band digital backend subband
        (11.71875 MHz).
    n_time_per_block : int, optional
        Post-channelization time samples per block. Default 1000, i.e.
        32.768 ms per block at the default channel width.
    n_blocks : int, optional
        Number of blocks in the observation. Default 61 (~2 s).
    noise_std : float or astropy.units.Quantity, optional
        Standard deviation of the per-antenna complex receiver noise, in
        root-Jy: the noise power added to every autocorrelation is
        ``noise_std**2`` Jy, i.e. ``noise_std**2`` plays the role of an
        SEFD. Default 1.0. Set to 0.0 for noiseless runs.
    rng : numpy.random.Generator
        Seeded random generator. Required -- the package never seeds a
        global generator, so that every run is reproducible. It is drawn
        from exactly once, at construction, to seed an independent
        generator per block (see Notes).

    Attributes
    ----------
    freq_hz : numpy.ndarray
        Shape ``(n_chan,)`` RF channel center frequencies, Hz, ascending.
    sample_period_s : float
        ``1 / chan_width_hz``, seconds.
    seed_sequence : numpy.random.SeedSequence
        Root seed sequence derived from `rng` at construction; the per-block
        generators are spawned from it.

    Notes
    -----
    Blocks are independently seeded, so ``block(i)`` is a pure function of
    the construction-time `rng` state and ``i``. That matters more than it
    looks: with a single shared stream, merely peeking at one block (say,
    in a debugger or a plotting notebook) would shift every later block and
    silently produce a *different* dataset from the same seed. Here blocks
    can be generated in any order, skipped, or regenerated, and the data is
    always the same.

    The seed tree has two branches, and the order they are spawned in is a
    contract, not an implementation detail::

        root -> sky/noise seeds, one per block   (spawned first)
             -> interference seeds, one per block (spawned second)
                  -> one per interference source

    Sky and noise are spawned first and unconditionally, so their stream
    cannot depend on whether -- or how many -- interference sources are
    attached. Each source then gets its own generator per block, so adding
    a second transmitter does not disturb the first one either. Together
    these give exactly reproducible clean/contaminated pairs: run the same
    seed with and without `rfi_sources` and the difference between the two
    datasets is the interference and nothing else.

    Raises
    ------
    ValueError
        If any count is non-positive, `noise_std` is negative, or
        `phase_center` / `start_time` is not scalar.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.time import Time
    >>> from rfi_simulator import ArrayConfig, PointSource, VoltageSimulator
    >>> from rfi_simulator.delays import earth_location, zenith_coord
    >>> array = ArrayConfig.from_yaml("configs/array_default.yaml")
    >>> t0 = Time("2026-10-01T04:00:00", scale="utc")
    >>> center = zenith_coord(earth_location(array), t0)
    >>> src = PointSource.from_lm(center, (0.005, 0.0), flux_jy=1.0)
    >>> sim = VoltageSimulator(array, center, t0, [src], n_chan=8, n_blocks=2,
    ...                        n_time_per_block=16, rng=np.random.default_rng(0))
    >>> blocks = list(sim.blocks())
    >>> blocks[0].data.shape
    (10, 8, 16)
    """

    def __init__(
        self,
        array: ArrayConfig,
        phase_center: SkyCoord,
        start_time: Time,
        sources: Sequence[PointSource] = (),
        *,
        rfi_sources: Sequence[RFISource] = (),
        center_freq_hz=DEFAULT_CENTER_FREQ_HZ,
        n_chan: int = DEFAULT_N_CHAN,
        chan_width_hz=DEFAULT_CHAN_WIDTH_HZ,
        n_time_per_block: int = DEFAULT_N_TIME_PER_BLOCK,
        n_blocks: int = DEFAULT_N_BLOCKS,
        noise_std=1.0,
        rng: np.random.Generator,
    ) -> None:
        if not phase_center.isscalar:
            raise ValueError("phase_center must be a scalar SkyCoord")
        if not start_time.isscalar:
            raise ValueError("start_time must be a scalar Time")

        self.array = array
        self.phase_center = phase_center
        self.start_time = start_time
        self.sources = list(sources)
        self.rfi_sources = list(rfi_sources)

        self.center_freq_hz = float(_to_value(center_freq_hz, u.Hz))
        self.chan_width_hz = float(_to_value(chan_width_hz, u.Hz))
        self.noise_std = float(_to_value(noise_std, u.Jy**0.5))
        self.n_chan = int(n_chan)
        self.n_time_per_block = int(n_time_per_block)
        self.n_blocks = int(n_blocks)

        if self.n_chan < 1:
            raise ValueError(f"n_chan must be >= 1, got {self.n_chan}")
        if self.n_time_per_block < 1:
            raise ValueError(f"n_time_per_block must be >= 1, got {self.n_time_per_block}")
        if self.n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {self.n_blocks}")
        if self.chan_width_hz <= 0.0:
            raise ValueError(f"chan_width_hz must be > 0, got {self.chan_width_hz}")
        if self.noise_std < 0.0:
            raise ValueError(f"noise_std must be >= 0, got {self.noise_std}")

        # RF channel centers, ascending, symmetric about the band center.
        offsets = np.arange(self.n_chan, dtype=np.float64) - 0.5 * (self.n_chan - 1)
        self.freq_hz = self.center_freq_hz + offsets * self.chan_width_hz

        # Draw from the caller's generator exactly once, then give every
        # block its own independent generator spawned from the result.
        entropy = rng.integers(0, 2**63 - 1, size=4, dtype=np.int64)
        self.seed_sequence = np.random.SeedSequence(entropy.tolist())
        # Order matters (see the class Notes): sky/noise first, always, so
        # that attaching interference sources cannot shift their stream.
        self._block_seed_sequences = self.seed_sequence.spawn(self.n_blocks)
        # `SeedSequence.spawn` is stateful, so every spawn happens exactly
        # once, here -- calling it again per block would hand out fresh
        # children each time and destroy the purity of `block(i)`.
        self._rfi_seed_sequences = [
            block_seed.spawn(len(self.rfi_sources))
            for block_seed in self.seed_sequence.spawn(self.n_blocks)
        ]

        self.location = earth_location(array)
        self._precompute_geometry()

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------
    @property
    def sample_period_s(self) -> float:
        """float: Post-channelization sample period, seconds."""
        return 1.0 / self.chan_width_hz

    @property
    def block_duration_s(self) -> float:
        """float: Duration of one block (one integration), seconds."""
        return self.n_time_per_block * self.sample_period_s

    @property
    def duration_s(self) -> float:
        """float: Total duration of the observation, seconds."""
        return self.n_blocks * self.block_duration_s

    @property
    def bandwidth_hz(self) -> float:
        """float: Total simulated bandwidth, Hz."""
        return self.n_chan * self.chan_width_hz

    @property
    def n_antennas(self) -> int:
        """int: Number of antennas."""
        return self.array.n_antennas

    def block_start_times(self) -> Time:
        """Start times of every block.

        Returns
        -------
        astropy.time.Time
            Shape ``(n_blocks,)`` UTC times.
        """
        return self.start_time + np.arange(self.n_blocks) * self.block_duration_s * u.s

    def block_center_times(self) -> Time:
        """Mid-point times of every block (the epochs delays are evaluated at).

        Returns
        -------
        astropy.time.Time
            Shape ``(n_blocks,)`` UTC times.
        """
        return self.block_start_times() + 0.5 * self.block_duration_s * u.s

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def _precompute_geometry(self) -> None:
        """Evaluate per-block source directions and the (l, m) basis, once.

        Two vectorized astropy transforms cover the whole observation;
        each block still gets its *own* delays, so Earth rotation is
        present, but we do not pay astropy's per-call overhead 61 times.
        """
        center_times = self.block_center_times()

        s0_hat, e_l, e_m = lm_basis_enu(self.phase_center, center_times, self.location)
        self._phase_center_s_hat = s0_hat  # (n_blocks, 3)
        self._e_l_enu = e_l  # (n_blocks, 3)
        self._e_m_enu = e_m  # (n_blocks, 3)

        positions = self.array.antenna_positions_enu_m
        # (n_blocks, n_ant)
        self._phase_center_delays_s = geometric_delays_s(positions, s0_hat)

        if self.sources:
            source_coords = SkyCoord([src.coord for src in self.sources])
            # (n_src, 1) coords against (n_blocks,) times -> (n_src, n_blocks, 3)
            source_s_hat = source_unit_vectors_enu(
                source_coords.reshape(len(self.sources), 1), center_times, self.location
            )
            # (n_src, n_blocks, n_ant)
            self._source_delays_s = geometric_delays_s(positions, source_s_hat)
        else:
            self._source_delays_s = np.zeros(
                (0, self.n_blocks, self.array.n_antennas), dtype=np.float64
            )

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------
    def block_rng(self, index: int) -> np.random.Generator:
        """The generator used for block `index`.

        Parameters
        ----------
        index : int
            Block index in ``[0, n_blocks)``.

        Returns
        -------
        numpy.random.Generator
            A fresh generator, seeded only by the construction-time seed
            and `index`. Calling this twice gives two generators that
            produce the same stream.
        """
        if not 0 <= index < self.n_blocks:
            raise ValueError(f"block index {index} out of range [0, {self.n_blocks})")
        return np.random.default_rng(self._block_seed_sequences[index])

    def rfi_block_rngs(self, index: int) -> list[np.random.Generator]:
        """The generators used by the interference sources for block `index`.

        Parameters
        ----------
        index : int
            Block index in ``[0, n_blocks)``.

        Returns
        -------
        list of numpy.random.Generator
            One generator per entry of `rfi_sources`, in the same order.
            These come from a different branch of the seed tree than
            `block_rng`, which is what keeps the sky-plus-noise
            realization independent of the interference configuration.
        """
        if not 0 <= index < self.n_blocks:
            raise ValueError(f"block index {index} out of range [0, {self.n_blocks})")
        return [np.random.default_rng(child) for child in self._rfi_seed_sequences[index]]

    @staticmethod
    def _circular_normal(
        rng: np.random.Generator, shape: tuple[int, ...], scale: float
    ) -> np.ndarray:
        """Draw circular complex Gaussian samples with ``E|z|**2 == scale**2``.

        Parameters
        ----------
        rng : numpy.random.Generator
            Generator to draw from.
        shape : tuple of int
            Output shape.
        scale : float
            Root-mean-square modulus of the samples.

        Returns
        -------
        numpy.ndarray
            Complex64 array of shape `shape`.
        """
        parts = rng.standard_normal(size=(*shape, 2), dtype=np.float32)
        parts *= np.float32(scale / np.sqrt(2.0))
        return parts.view(np.complex64)[..., 0]

    def block(self, index: int) -> VoltageBlock:
        """Synthesize a single block of voltages.

        Parameters
        ----------
        index : int
            Block index in ``[0, n_blocks)``.

        Returns
        -------
        VoltageBlock
            The block's data and metadata.

        Notes
        -----
        Deterministic in ``(seed, index)``: blocks may be generated in any
        order, repeated, or skipped without changing any of them.
        """
        rng = self.block_rng(index)

        n_ant = self.array.n_antennas
        n_chan = self.n_chan
        n_time = self.n_time_per_block

        data = np.zeros((n_ant, n_chan, n_time), dtype=np.complex64)

        for i_src, source in enumerate(self.sources):
            if source.flux_jy == 0.0:
                continue
            # One sky signal, shared by every antenna.
            spectrum = self._circular_normal(rng, (n_chan, n_time), np.sqrt(source.flux_jy))
            tau_s = self._source_delays_s[i_src, index]  # (n_ant,)
            # RF frequency of each channel -- NOT a baseband offset.
            phase = np.exp(-2j * np.pi * self.freq_hz[np.newaxis, :] * tau_s[:, np.newaxis]).astype(
                np.complex64
            )
            data += phase[:, :, np.newaxis] * spectrum[np.newaxis, :, :]

        if self.noise_std > 0.0:
            # Independent receiver noise per antenna.
            data += self._circular_normal(rng, (n_ant, n_chan, n_time), self.noise_std)

        start_time = self.start_time + index * self.block_duration_s * u.s
        rfi_mask = self._add_rfi(data, index)

        return VoltageBlock(
            data=data,
            time=start_time,
            center_time=start_time + 0.5 * self.block_duration_s * u.s,
            freq_hz=self.freq_hz,
            sample_period_s=self.sample_period_s,
            phase_center_delays_s=self._phase_center_delays_s[index],
            antenna_positions_enu_m=self.array.antenna_positions_enu_m,
            e_l_enu=self._e_l_enu[index],
            e_m_enu=self._e_m_enu[index],
            s0_enu=self._phase_center_s_hat[index],
            rfi_mask=rfi_mask,
            rfi_source_names=tuple(source.name for source in self.rfi_sources),
        )

    def block_context(self, index: int, rng: np.random.Generator) -> BlockContext:
        """Build the context handed to an interference source for block `index`.

        Parameters
        ----------
        index : int
            Block index in ``[0, n_blocks)``.
        rng : numpy.random.Generator
            Generator the source should draw from.

        Returns
        -------
        BlockContext
            Block geometry, timing and generator. Sources must treat it
            as read-only.
        """
        start_time = self.start_time + index * self.block_duration_s * u.s
        return BlockContext(
            index=index,
            start_time=start_time,
            center_time=start_time + 0.5 * self.block_duration_s * u.s,
            sample_times_s=np.arange(self.n_time_per_block, dtype=np.float64)
            * self.sample_period_s,
            sample_period_s=self.sample_period_s,
            freq_hz=self.freq_hz,
            chan_width_hz=self.chan_width_hz,
            antenna_positions_enu_m=self.array.antenna_positions_enu_m,
            location=self.location,
            phase_center_s_hat_enu=self._phase_center_s_hat[index],
            phase_center_delays_s=self._phase_center_delays_s[index],
            rng=rng,
        )

    def _add_rfi(self, data: np.ndarray, index: int) -> np.ndarray:
        """Add every interference source to `data` and collect their labels.

        Parameters
        ----------
        data : numpy.ndarray
            Complex64 ``(n_ant, n_chan, n_time)`` sky-plus-noise voltages,
            modified in place.
        index : int
            Block index.

        Returns
        -------
        numpy.ndarray
            Boolean ``(n_sources, n_chan, n_time)`` occupancy masks, in
            the order of `rfi_sources`.

        Raises
        ------
        ValueError
            If a source returns arrays of the wrong shape.
        """
        n_chan, n_time = self.n_chan, self.n_time_per_block
        masks = np.zeros((len(self.rfi_sources), n_chan, n_time), dtype=bool)
        if not self.rfi_sources:
            return masks

        rngs = self.rfi_block_rngs(index)
        expected_voltage_shape = (self.array.n_antennas, n_chan, n_time)
        for i_src, (source, source_rng) in enumerate(zip(self.rfi_sources, rngs)):
            ctx = self.block_context(index, source_rng)
            voltages, mask = source.contribution(ctx)
            if voltages.shape != expected_voltage_shape:
                raise ValueError(
                    f"interference source {source.name!r} returned voltages of shape "
                    f"{voltages.shape}, expected {expected_voltage_shape}"
                )
            if mask.shape != (n_chan, n_time):
                raise ValueError(
                    f"interference source {source.name!r} returned a mask of shape "
                    f"{mask.shape}, expected {(n_chan, n_time)}"
                )
            data += voltages.astype(np.complex64, copy=False)
            masks[i_src] = mask
        return masks

    def blocks(self) -> Iterator[VoltageBlock]:
        """Iterate over the observation, one `VoltageBlock` at a time.

        Yields
        ------
        VoltageBlock
            Successive blocks; the whole observation is never
            materialized in memory at once. This is exactly
            ``block(0), block(1), ...``, so iterating is equivalent to
            asking for the blocks one at a time.
        """
        for index in range(self.n_blocks):
            yield self.block(index)
