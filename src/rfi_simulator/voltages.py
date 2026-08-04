r"""Per-antenna channelized voltage synthesis.

The simulator produces *channelized* complex baseband voltages: for every
antenna, every frequency channel and every post-channelization time sample
it emits one complex number. A perfect channelizer is assumed by default
(no PFB leakage -- see the deliberate MVP simplifications in the module
docstrings of ``delays`` and ``correlator``); pass a `channelizer` to
replace that assumption with a real filterbank response.

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

Optional `rfi_simulator.sky.SpectralLineForeground` foregrounds (e.g. a
Galactic HI-line bump) add independent per-antenna noise, frequency-shaped
into a Gaussian instead of flat, from the **same** seed branch as the sky
and receiver noise -- they are celestial, not interference, so they belong
with the sky/noise stream and are labelled with their own
`VoltageBlock.celestial_mask`, kept separate from `rfi_mask` so scoring can
tell a flagged line apart from flagged interference.

An optional `rfi_simulator.beam.PrimaryBeam` (`primary_beam`, off by
default) attenuates each `PointSource` by its offset from the pointing
center, frequency-dependently -- a source at fixed offset is attenuated
more at higher frequency, because the beam shrinks. Concretely, the
beam's **voltage** (amplitude) response,
:math:`\sqrt{B(\theta_{\mathrm{src}}, f)}` -- see `rfi_simulator.beam` for
why it is the square root of the power response -- multiplies the fringe
phase applied to that source's shared spectrum, per channel, before it is
added to every antenna, so it is common to every antenna (all antennas
are assumed to carry the same dish/beam in this v1) and every
visibility's flux ends up scaled by the **power** response, which is what
the module's acceptance test checks. Only `PointSource` is attenuated:
`SpectralLineForeground` is not, because it already models a fully
resolved, sky-filling emission (see its own docstring) that fills the
whole main lobe rather than sitting at one offset the beam could
discriminate against; near-field `rfi_simulator.rfi.RFISource`
interference is not attenuated either, because it enters the array
through the far sidelobes and the ground plane, not through the main
beam this model describes, and its per-antenna coupling
(`rfi_simulator.rfi.resolve_coupling`) already carries whatever
attenuation that near-field geometry implies. The offset itself is the
small-angle :math:`\theta \approx \sqrt{l^2 + m^2}` from
`rfi_simulator.sky.PointSource.lm` relative to `pointing_center` (the
phase center by default -- see `pointing_center` below); because ``(l,
m)`` of a fixed source is constant across a whole observation (see
`rfi_simulator.sky`), so is its beam response, and it is evaluated once
at construction rather than per block. The applied per-source,
per-channel power response is recorded as ground truth on
`VoltageBlock.beam_response`.

Three optional stages sit at the end of the per-block synthesis, all off by
default so that the expression above is the whole story unless they are
switched on:

* an `rfi_simulator.channelizer.PFBChannelizer`, which replaces the
  perfect-channelizer assumption with a polyphase filterbank's response --
  temporal memory within a channel, correlation between neighbouring
  channels, and continuous carrier frequencies;
* an `rfi_simulator.instrument.InstrumentModel`, whose per-antenna complex
  gain multiplies the antenna's **total** stream -- sky, interference and
  noise together, because the receiver chain amplifies its own noise;
* 4-bit quantization (``quantization="int4"``), which passes the block
  through the quantize/dequantize round trip of the packed on-disk format
  and records which antennas railed.

Polarization axis
-----------------
`VoltageSimulator` is single-polarization by default and grows a second
receptor when built with ``n_pol=2``. The axis convention is deliberate:

* a block's `VoltageBlock.data` is ``(n_antennas, n_chan, n_time)`` when
  there is one receptor and ``(n_antennas, n_pol, n_chan, n_time)`` when
  there are two. The polarization axis is *added*, never permuted in, so
  a single-polarization run is bit-for-bit the data this simulator
  produced before polarization existed and every consumer indexing
  ``data[ant]`` keeps working. `VoltageBlock.pol_data` is the view that
  always has the axis, for code that would rather not branch;
* it sits directly after the antenna axis because a receptor belongs to
  an antenna -- the pair ``(antenna, polarization)`` is what a real
  backend calls an input -- and because it leaves ``(n_chan, n_time)``
  trailing and contiguous, which is the shape every downstream operation
  (the filterbank, the quantizer, the correlator's inner product) works
  on.

**Stokes convention.** Each receptor is calibrated in Stokes-I janskys:
an unpolarized source of flux ``F`` gives ``XX = YY = F``, and Stokes I
is formed as ``I = (XX + YY) / 2``. A dual-polarization run of a scene
therefore images to the *same* flux as the single-polarization run of
that scene, which is the property that makes the two comparable, and one
receptor of a dual-polarization run is statistically what the
single-polarization run models. Sky sources, receiver noise and spectral
lines are unpolarized at this fidelity -- an independent realization per
receptor -- while interference carries whatever state
`rfi_simulator.rfi.resolve_polarization` was given, and a source's
``received_power_jy`` is its Stokes I whatever that state is. Cross-hand
products (XY, YX) are not formed; see `rfi_simulator.correlator`.

They are applied in that order, and the order is a modelling decision
worth stating. The filterbank goes first because it acts on the total
wideband stream, sky and interference and noise alike, and because the
channel mixing it introduces must not be applied to something that has
already been labelled per channel. The gains go *after* it rather than
before, even though a receiver's analog chain physically precedes the
digitizer, so that `VoltageBlock.gains` stays exactly the per-channel
truth: a bandpass applied ahead of the filterbank would be smeared across
channels by it, and the recorded gains would no longer be the numbers a
calibration exercise is supposed to recover. For the smooth bandpasses
`rfi_simulator.instrument` produces the two orders differ negligibly.
Quantization stays last, which is also where a real backend requantizes:
after channelization, not before.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from rfi_simulator.array_config import ArrayConfig, _to_value
from rfi_simulator.beam import PrimaryBeam
from rfi_simulator.channelizer import PFBChannelizer
from rfi_simulator.delays import (
    earth_location,
    geometric_delays_s,
    lm_basis_enu,
    source_unit_vectors_enu,
)
from rfi_simulator.instrument import InstrumentModel
from rfi_simulator.io.packed_voltage import quantize_roundtrip, suggest_quant_scale
from rfi_simulator.rfi import OCCUPANCY_THRESHOLD, BlockContext, RFISource
from rfi_simulator.sky import PointSource, SpectralLineForeground

__all__ = ["QUANTIZATION_MODES", "VoltageBlock", "VoltageSimulator"]

DEFAULT_CENTER_FREQ_HZ = 1.405e9
DEFAULT_N_CHAN = 384
DEFAULT_CHAN_WIDTH_HZ = 30517.578125
DEFAULT_N_TIME_PER_BLOCK = 1000
DEFAULT_N_BLOCKS = 61

#: Accepted values of `VoltageSimulator`'s ``quantization`` argument.
#: ``None`` keeps the voltages in full floating-point precision.
QUANTIZATION_MODES = (None, "int4")

#: Default loading of the 4-bit quantizer, in counts rms per real/imaginary
#: component. Light loading (a little over one count rms against rails at
#: +-7 counts) is what keeps a correlator front end clear of saturation on
#: normal sky-plus-noise data while still leaving quantization noise a
#: modest fraction of the signal; it also means a strong interferer or an
#: over-gained antenna rails, which is exactly the behaviour a flagging
#: benchmark should contain.
DEFAULT_QUANT_TARGET_COUNTS = 1.33


def _widen_channels(mask: np.ndarray, radius: int) -> np.ndarray:
    """Mark a mask's neighbouring channels occupied, out to `radius`.

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean array of shape ``(n_sources, n_chan, n_time)``.
    radius : int
        Number of channels to spread each occupied cell over, on each
        side. ``0`` returns `mask` unchanged.

    Returns
    -------
    numpy.ndarray
        Boolean array of the same shape. The channel axis is treated as
        cyclic, matching the filterbank's own treatment of the band edges
        (see `rfi_simulator.channelizer`).
    """
    if radius <= 0 or mask.size == 0:
        return mask
    widened = mask.copy()
    for shift in range(1, int(radius) + 1):
        widened |= np.roll(mask, shift, axis=1)
        widened |= np.roll(mask, -shift, axis=1)
    return widened


@dataclass
class VoltageBlock:
    """One block of channelized voltages plus the metadata to correlate it.

    A block is the natural unit of the simulation: delays are frozen
    within a block and re-evaluated between blocks, and one block becomes
    one correlator integration.

    Attributes
    ----------
    data : numpy.ndarray
        Complex64 voltages in units of sqrt(Jy) (so ``|v|**2`` is in Jy),
        of shape ``(n_antennas, n_chan, n_time)`` for a
        single-polarization block and ``(n_antennas, n_pol, n_chan,
        n_time)`` for a dual-polarization one. The polarization axis is
        *present only when there is more than one receptor*: a
        single-polarization block is exactly the array this simulator has
        always produced, so every existing consumer of ``data[ant]``
        keeps working, and code that wants one shape for both cases reads
        `pol_data` instead. See the module docstring's "Polarization
        axis" note for why the axis sits between the antennas and the
        channels.
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
    rfi_polarization : numpy.ndarray, optional
        Complex128 ground-truth per-receptor amplitudes of shape
        ``(n_interference_sources, n_pol)``: the amplitude factor each
        source was received with in each polarization, so its power
        contribution scales as ``|a|**2`` (see
        `rfi_simulator.rfi.resolve_polarization`). All ones for an
        unpolarized source -- the same convention as `rfi_coupling`, which
        is all ones for uniform coupling. ``None`` for a block that was
        not built by a simulator. This, and not `rfi_mask`, is where a
        source's polarization lives: occupancy is a property of the
        transmitter's spectrum, identical in both receptors, while the
        *amplitude* each receptor sees is what the polarization state
        sets.
    rfi_coupling : numpy.ndarray, optional
        Float64 ground-truth per-antenna coupling of shape
        ``(n_interference_sources, n_antennas)``: the linear *amplitude*
        factor each source was received with at each antenna, so its power
        contribution scales as the square (see
        `rfi_simulator.rfi.resolve_coupling`). All ones for a source with
        the default uniform coupling. ``None`` (the default) for a block
        that was not built by a simulator. Deliberately separate from
        `rfi_mask`, which has no antenna axis -- see the labelling
        convention in `rfi_simulator.rfi`.
    celestial_mask : numpy.ndarray, optional
        Boolean ground-truth labels of shape
        ``(n_celestial_sources, n_chan, n_time)``, one entry per
        `rfi_simulator.sky.SpectralLineForeground` attached to the
        simulator: ``celestial_mask[s, c, t]`` is True where line ``s``
        occupies cell ``(c, t)``. Defaults to an empty ``(0, n_chan,
        n_time)`` array. Kept entirely separate from `rfi_mask` -- the two
        never share an axis or an index -- because the two label classes
        (``"celestial"`` vs the interference sources' implicit ``"rfi"``)
        must never be confusable, in bookkeeping as much as in meaning: a
        scoring harness that treats a flagged celestial cell as a correctly
        excised one would penalize an algorithm for doing the right thing.
    celestial_source_names : tuple of str, optional
        Names of the spectral-line foregrounds, in the order of
        `celestial_mask`'s leading axis. Defaults to ``()``.
    gains : numpy.ndarray, optional
        Complex ground-truth per-antenna gains of shape ``(n_antennas,
        n_chan)`` -- or ``(n_antennas, n_pol, n_chan)`` for a
        dual-polarization block, whose two receptor chains may have been
        given different models -- that were applied to this block (see
        `rfi_simulator.instrument`), or ``None`` (the default) when the
        block was simulated with no instrument model, i.e. with all gains
        exactly one. Carried so that calibration exercises can be scored
        against the truth.
    clip_fraction : numpy.ndarray, optional
        Float64 array of shape ``(n_antennas,)`` -- or ``(n_antennas,
        n_pol)`` for a dual-polarization block -- holding the fraction of
        this block's complex samples that saturated the quantizer, per
        antenna and receptor, or ``None`` when the block was not
        quantized. Ground truth for antennas driven into their rails; a
        polarized interferer rails one receptor long before the other,
        which is exactly the asymmetry this field has to record.
    quant_scale : float, optional
        The quantization scale (voltage units per count) this block was
        quantized with, or ``None`` when the block was not quantized.
    channelizer : PFBChannelizer, optional
        The filterbank this block was passed through (see
        `rfi_simulator.channelizer`), or ``None`` (the default) for a block
        synthesized under the perfect-channelizer assumption. Carried as
        ground truth, like `gains` and `quant_scale`: the temporal and
        channel-to-channel correlations of the data are a property of this
        object, and a flagger's statistics depend on them.
    beam_response : numpy.ndarray, optional
        Float64 ground-truth per-source, per-channel primary-beam **power**
        response of shape ``(n_sources, n_chan)``, in the order of the
        simulator's `PointSource` sky model: ``beam_response[s, c]`` is
        the fraction of on-axis power source ``s`` was received with at
        channel ``c`` (see `rfi_simulator.beam`), so its true catalog flux
        times this factor is exactly the flux a noiseless correlator
        recovers for that source. ``None`` (the default) for a block
        simulated with no `primary_beam` attached, i.e. every source at
        its full flux -- the same "``None`` means off" convention `gains`
        and `channelizer` use.
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
    rfi_coupling: np.ndarray | None = None
    rfi_polarization: np.ndarray | None = None
    celestial_mask: np.ndarray | None = None
    celestial_source_names: tuple[str, ...] = field(default_factory=tuple)
    gains: np.ndarray | None = None
    clip_fraction: np.ndarray | None = None
    quant_scale: float | None = None
    channelizer: PFBChannelizer | None = None
    beam_response: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.data.ndim not in (3, 4):
            raise ValueError(
                "data must have shape (n_antennas, n_chan, n_time) or "
                f"(n_antennas, n_pol, n_chan, n_time), got {self.data.shape}"
            )
        if self.data.ndim == 4 and self.data.shape[1] < 1:
            raise ValueError(f"data has an empty polarization axis: {self.data.shape}")
        if self.rfi_mask is None:
            self.rfi_mask = np.zeros((0, self.n_chan, self.n_time), dtype=bool)
        self.rfi_source_names = tuple(self.rfi_source_names)
        if self.rfi_mask.shape != (len(self.rfi_source_names), self.n_chan, self.n_time):
            raise ValueError(
                "rfi_mask must have shape (n_sources, n_chan, n_time) = "
                f"({len(self.rfi_source_names)}, {self.n_chan}, {self.n_time}), "
                f"got {self.rfi_mask.shape}"
            )
        if self.rfi_coupling is not None and self.rfi_coupling.shape != (
            len(self.rfi_source_names),
            self.n_antennas,
        ):
            raise ValueError(
                "rfi_coupling must have shape (n_sources, n_antennas) = "
                f"({len(self.rfi_source_names)}, {self.n_antennas}), "
                f"got {self.rfi_coupling.shape}"
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
        if self.celestial_mask is None:
            self.celestial_mask = np.zeros((0, self.n_chan, self.n_time), dtype=bool)
        self.celestial_source_names = tuple(self.celestial_source_names)
        if self.celestial_mask.shape != (
            len(self.celestial_source_names),
            self.n_chan,
            self.n_time,
        ):
            raise ValueError(
                "celestial_mask must have shape (n_celestial_sources, n_chan, n_time) = "
                f"({len(self.celestial_source_names)}, {self.n_chan}, {self.n_time}), "
                f"got {self.celestial_mask.shape}"
            )
        if self.beam_response is not None:
            if self.beam_response.ndim != 2 or self.beam_response.shape[1] != self.n_chan:
                raise ValueError(
                    "beam_response must have shape (n_sources, n_chan) with n_chan = "
                    f"{self.n_chan}, got {self.beam_response.shape}"
                )
            if not np.all(np.isfinite(self.beam_response)):
                raise ValueError("beam_response must be finite")

    @property
    def n_rfi_sources(self) -> int:
        """int: Number of interference sources labelled in this block."""
        return self.rfi_mask.shape[0]

    @property
    def n_celestial_sources(self) -> int:
        """int: Number of spectral-line foregrounds labelled in this block."""
        return self.celestial_mask.shape[0]

    @property
    def n_antennas(self) -> int:
        """int: Number of antennas."""
        return self.data.shape[0]

    @property
    def n_pol(self) -> int:
        """int: Number of polarizations (receptors) in this block."""
        return 1 if self.data.ndim == 3 else self.data.shape[1]

    @property
    def pol_data(self) -> np.ndarray:
        """numpy.ndarray: `data` with the polarization axis always present.

        Shape ``(n_antennas, n_pol, n_chan, n_time)``, a *view* of `data`
        -- reshaped for a single-polarization block, `data` itself
        otherwise. Anything that must handle both cases (the correlator,
        the packed writer, a flagger) should read this instead of
        branching on `data.ndim`.
        """
        if self.data.ndim == 3:
            return self.data.reshape(self.data.shape[0], 1, *self.data.shape[1:])
        return self.data

    @property
    def n_chan(self) -> int:
        """int: Number of frequency channels."""
        return self.data.shape[-2]

    @property
    def n_time(self) -> int:
        """int: Number of time samples in the block."""
        return self.data.shape[-1]

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
    spectral_lines : sequence of SpectralLineForeground, optional
        Celestial spectral-line foregrounds (see `rfi_simulator.sky`),
        e.g. a Galactic HI-line bump. Defaults to ``()``: no line, which is
        bit-for-bit the current behavior. Each line adds independent
        per-antenna noise, frequency-shaped into a Gaussian instead of
        being flat like `noise_std`, drawn from the **same** seed branch as
        the sky and receiver noise (they are celestial, not interference),
        so adding or removing rfi_sources never perturbs a run's spectral
        lines or vice versa. Ground truth is
        `VoltageBlock.celestial_mask`, kept in a field separate from
        `rfi_mask` -- see the module docstring.
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
        SEFD. Default 1.0. Set to 0.0 for noiseless runs. With
        ``n_pol=2`` each receptor gets its own independent realization at
        this same level -- receiver noise is unpolarized.
    n_pol : int, optional
        Number of receptors per antenna: 1 (the default) or 2. The
        default is bit-for-bit the data this simulator produced before
        dual polarization existed. With 2, blocks grow a polarization
        axis (see the module docstring), sky sources, spectral lines and
        receiver noise are drawn independently per receptor, and each
        interference source is split between the receptors according to
        its `rfi_simulator.rfi.RFISource` ``polarization`` state -- which
        is the point of the option: a polarized transmitter against
        unpolarized noise is what a polarization-aware flagger keys on.
    channelizer : PFBChannelizer, optional
        Polyphase-filterbank response (see
        `rfi_simulator.channelizer`). Default ``None``: the perfect
        channelizer, which is bit-for-bit the data this simulator produced
        before the model existed -- each channel white in time and
        uncorrelated with its neighbours. Pass one and the finished block
        is run through the filterbank, which colors every component alike
        (they share one receiver), links neighbouring channels, and lets
        `rfi_simulator.rfi.NarrowbandTransmitter` place a carrier at an
        arbitrary frequency rather than at a channel center. The operator
        is deterministic, so attaching it never perturbs any component's
        realization for a given seed.
    warm_start : bool, optional
        Whether the filterbank's state at the very start of block 0 is
        already full. Default ``True``, which is what a recording taken
        from a running backend looks like: the filter has been fed for
        hours before the dump began, so sample zero is statistically no
        different from any other. The state is built by synthesizing one
        virtual pre-observation block from a dedicated branch of the seed
        tree (see `_warm_start_state`), so block 0 stays a pure function
        of the seed. ``False`` starts the filter from zeros instead --
        the first ``n_taps - 1`` samples of the observation ramp up from
        an exact zero, as switching a backend on does. Only meaningful
        together with `channelizer`.
    instrument : InstrumentModel or sequence of InstrumentModel, optional
        Per-antenna direction-independent complex gains (see
        `rfi_simulator.instrument`). With ``n_pol=2``, pass one model to
        give both receptors the same gains or a sequence of two to give
        them independently drawn ones (see `_instrument_models`). Default
        ``None``: every antenna has
        exactly unit gain, which is bit-for-bit the same data as a run with
        `InstrumentModel.identity`. The gains multiply each antenna's
        **total** stream -- sky, interference and receiver noise together
        -- because they describe the receiver chain behind the antenna,
        which amplifies its own noise too. `InstrumentModel` draws its
        gains from its own seed at construction, never from this
        simulator's seed tree, so attaching one leaves the sky, noise and
        interference realizations untouched.
    quantization : {None, "int4"}, optional
        Digitization applied to each block after the gains. ``None`` (the
        default) keeps full floating-point precision. ``"int4"`` passes the
        block through the signed 4-bit quantize/dequantize round trip of
        `rfi_simulator.io.packed_voltage` -- the same code path as the
        on-disk packed format -- so the data carries real quantization
        noise and real saturation.
    quant_target_counts : float, optional
        Quantizer loading target in counts rms per real/imaginary
        component, used to pick the scale when `quant_scale` is not given.
        Default `DEFAULT_QUANT_TARGET_COUNTS`.
    quant_scale : float, optional
        Fixed quantization scale (voltage units per count). Default
        ``None``: each block gets its own scale from
        `rfi_simulator.io.packed_voltage.suggest_quant_scale` at
        `quant_target_counts`. Pass a value to hold the scale constant
        across blocks, which is what a real backend with a fixed digital
        gain does.
    primary_beam : rfi_simulator.beam.PrimaryBeam, optional
        Frequency-dependent primary-beam model (see
        `rfi_simulator.beam`). Default ``None``: every source is received
        at its full catalog flux regardless of offset from the pointing
        center, which is bit-for-bit the data this simulator produced
        before the model existed. Attached, it attenuates each
        `PointSource` by its offset from `pointing_center` -- see the
        module docstring for exactly where the factor enters and why
        `spectral_lines` and `rfi_sources` are unaffected. Deterministic
        (no randomness of its own), evaluated once at construction, so
        attaching one never perturbs any random draw.
    pointing_center : astropy.coordinates.SkyCoord, optional
        Scalar direction the (optional) primary beam is centered on.
        Default ``None``: the pointing center is `phase_center` itself --
        the ordinary case of an array phased up and pointed at the same
        direction. Pass a different scalar `SkyCoord` for an
        offset-pointing scene (phased on one direction, physically
        pointed at another); only meaningful together with
        `primary_beam`, but accepted either way.
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
             -> warm-start seed, one per run      (spawned last)
                  -> one per interference source

    Sky and noise are spawned first and unconditionally, so their stream
    cannot depend on whether -- or how many -- interference sources are
    attached. Each source then gets its own generator per block, so adding
    a second transmitter does not disturb the first one either. Together
    these give exactly reproducible clean/contaminated pairs: run the same
    seed with and without `rfi_sources` and the difference between the two
    datasets is the interference and nothing else. The warm-start branch
    hangs off the end for the same reason: the virtual pre-observation
    block that fills a filterbank's state at ``t = 0`` (see `warm_start`)
    must not be able to move any real block's realization, whether or not
    a channelizer is attached.

    Polarization does not add a branch to that tree. A second receptor
    only widens the *shape* of each existing draw -- the sky spectrum
    becomes ``(n_pol, n_chan, n_time)``, the noise ``(n_ant, n_pol,
    n_chan, n_time)``, and an unpolarized interference source draws one
    waveform realization per receptor -- so the generator is consumed in
    the same order, and ``n_pol=1`` is bit-for-bit unchanged. A receptor
    of a dual-polarization run is therefore *statistically*, not
    literally, the single-polarization run of the same seed.

    `spectral_lines` draw from the sky/noise branch's own generator,
    *after* the sky sources and receiver noise of a block (see `block`),
    the same way adding another `PointSource` would: they are celestial,
    so they belong on that branch rather than the interference one, at the
    cost that attaching a line does shift the noise realization drawn
    after it -- exactly as attaching another sky source already does.
    `rfi_sources` are never affected either way.

    Raises
    ------
    ValueError
        If any count is non-positive, if `center_freq_hz`, `chan_width_hz`,
        `noise_std`, `quant_target_counts` or `quant_scale` is not finite
        (or is otherwise out of its documented range, e.g. `noise_std`
        negative), if `phase_center` / `start_time` is not scalar, or if
        `n_time_per_block` is shorter than `channelizer.n_taps - 1`.

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
        spectral_lines: Sequence[SpectralLineForeground] = (),
        center_freq_hz=DEFAULT_CENTER_FREQ_HZ,
        n_chan: int = DEFAULT_N_CHAN,
        chan_width_hz=DEFAULT_CHAN_WIDTH_HZ,
        n_time_per_block: int = DEFAULT_N_TIME_PER_BLOCK,
        n_blocks: int = DEFAULT_N_BLOCKS,
        noise_std=1.0,
        n_pol: int = 1,
        channelizer: PFBChannelizer | None = None,
        warm_start: bool = True,
        instrument: InstrumentModel | Sequence[InstrumentModel] | None = None,
        quantization: str | None = None,
        quant_target_counts: float = DEFAULT_QUANT_TARGET_COUNTS,
        quant_scale: float | None = None,
        primary_beam: PrimaryBeam | None = None,
        pointing_center: SkyCoord | None = None,
        rng: np.random.Generator,
    ) -> None:
        if not phase_center.isscalar:
            raise ValueError("phase_center must be a scalar SkyCoord")
        if not start_time.isscalar:
            raise ValueError("start_time must be a scalar Time")
        if pointing_center is not None and not pointing_center.isscalar:
            raise ValueError("pointing_center must be a scalar SkyCoord")
        if primary_beam is not None and not isinstance(primary_beam, PrimaryBeam):
            raise ValueError(
                f"primary_beam must be a PrimaryBeam or None, got {type(primary_beam)!r}"
            )

        self.array = array
        self.phase_center = phase_center
        self.start_time = start_time
        self.sources = list(sources)
        self.rfi_sources = list(rfi_sources)
        self.spectral_lines = list(spectral_lines)

        self.center_freq_hz = float(_to_value(center_freq_hz, u.Hz))
        self.chan_width_hz = float(_to_value(chan_width_hz, u.Hz))
        self.noise_std = float(_to_value(noise_std, u.Jy**0.5))
        self.n_chan = int(n_chan)
        self.n_time_per_block = int(n_time_per_block)
        self.n_blocks = int(n_blocks)
        self.n_pol = int(n_pol)
        if self.n_pol not in (1, 2):
            raise ValueError(f"n_pol must be 1 or 2, got {self.n_pol}")

        if self.n_chan < 1:
            raise ValueError(f"n_chan must be >= 1, got {self.n_chan}")
        if self.n_time_per_block < 1:
            raise ValueError(f"n_time_per_block must be >= 1, got {self.n_time_per_block}")
        if self.n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {self.n_blocks}")
        if not np.isfinite(self.center_freq_hz):
            raise ValueError(f"center_freq_hz must be finite, got {self.center_freq_hz}")
        if not np.isfinite(self.chan_width_hz) or self.chan_width_hz <= 0.0:
            raise ValueError(f"chan_width_hz must be finite and > 0, got {self.chan_width_hz}")
        if not np.isfinite(self.noise_std) or self.noise_std < 0.0:
            raise ValueError(f"noise_std must be finite and >= 0, got {self.noise_std}")
        for line in self.spectral_lines:
            if not isinstance(line, SpectralLineForeground):
                raise ValueError(
                    "spectral_lines must contain SpectralLineForeground instances, "
                    f"got {type(line)!r}"
                )

        # RF channel centers, ascending, symmetric about the band center.
        offsets = np.arange(self.n_chan, dtype=np.float64) - 0.5 * (self.n_chan - 1)
        self.freq_hz = self.center_freq_hz + offsets * self.chan_width_hz

        self.primary_beam = primary_beam
        self.pointing_center = phase_center if pointing_center is None else pointing_center
        # A source's (l, m) relative to a fixed pointing center does not
        # change as the Earth rotates (see rfi_simulator.sky), so its beam
        # response is a property of the source, not the block, and is
        # evaluated exactly once here -- never per block.
        if self.primary_beam is None:
            self._beam_power = None
        elif self.sources:
            offsets_lm = np.stack(
                [source.lm(self.pointing_center) for source in self.sources], axis=0
            )  # (n_src, 2)
            theta_rad = np.sqrt(np.sum(offsets_lm**2, axis=-1))  # (n_src,), small-angle approx
            self._beam_power = self.primary_beam.power_response(
                theta_rad[:, np.newaxis], self.freq_hz[np.newaxis, :]
            )  # (n_src, n_chan)
        else:
            self._beam_power = np.zeros((0, self.n_chan), dtype=np.float64)

        if channelizer is not None and not isinstance(channelizer, PFBChannelizer):
            raise ValueError(
                f"channelizer must be a PFBChannelizer or None, got {type(channelizer)!r}"
            )
        if channelizer is not None and self.n_time_per_block < channelizer.n_taps - 1:
            raise ValueError(
                f"n_time_per_block={self.n_time_per_block} is shorter than "
                f"channelizer.n_taps - 1={channelizer.n_taps - 1}; the seam state at a "
                "block's leading edge is rebuilt from its predecessor's trailing "
                "n_taps - 1 samples (see _seam_state), which a block shorter than that "
                "cannot supply, so a block generated out of order would silently see a "
                "different (short) history than one generated in sequence. Use a longer "
                "n_time_per_block or a channelizer with fewer taps."
            )
        self.channelizer = channelizer
        self.warm_start = bool(warm_start)
        # Filter state at the seam between block i-1 and block i, cached so
        # that iterating over blocks in order costs nothing extra; see
        # `_seam_state`.
        self._seam_cache: tuple[int, np.ndarray] | None = None
        # The state at the leading edge of block 0, built once on demand.
        self._warm_cache: np.ndarray | None = None

        self.instrument = instrument
        if instrument is None:
            self._gains = None
        else:
            models = self._instrument_models(instrument)
            for model in models:
                if model.n_antennas != self.array.n_antennas:
                    raise ValueError(
                        f"instrument describes {model.n_antennas} antennas but the array has "
                        f"{self.array.n_antennas}"
                    )
            # Evaluated once: the gains are a property of the receivers, not
            # of the block, so every block sees the identical bandpass.
            # (n_antennas, n_pol, n_chan), one plane per receptor.
            self._gains = np.stack(
                [model.gains(self.freq_hz).astype(np.complex64) for model in models], axis=1
            )

        if quantization not in QUANTIZATION_MODES:
            raise ValueError(
                f"quantization must be one of {QUANTIZATION_MODES}, got {quantization!r}"
            )
        self.quantization = quantization
        self.quant_target_counts = float(quant_target_counts)
        self.quant_scale = None if quant_scale is None else float(quant_scale)
        if not np.isfinite(self.quant_target_counts) or self.quant_target_counts <= 0.0:
            raise ValueError(
                f"quant_target_counts must be finite and > 0, got {self.quant_target_counts}"
            )
        if self.quant_scale is not None and (
            not np.isfinite(self.quant_scale) or not self.quant_scale > 0.0
        ):
            raise ValueError(f"quant_scale must be finite and > 0, got {self.quant_scale}")

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
        # The pre-observation block that warms the filterbank up (see
        # `_warm_start_state`) gets its own branch, spawned last and
        # unconditionally, so that neither attaching a channelizer nor
        # switching `warm_start` off can move any block's realization.
        warm_seed = self.seed_sequence.spawn(1)[0]
        self._warm_seed_sequence = warm_seed
        self._warm_rfi_seed_sequences = warm_seed.spawn(len(self.rfi_sources))

        self.location = earth_location(array)
        self._precompute_geometry()

    def _instrument_models(self, instrument) -> list[InstrumentModel]:
        """Validate `instrument` into one `InstrumentModel` per receptor.

        Parameters
        ----------
        instrument : InstrumentModel or sequence of InstrumentModel
            A single model, applied to every receptor, or exactly `n_pol`
            models, one per receptor in polarization order.

        Returns
        -------
        list of InstrumentModel
            Length `n_pol`.

        Raises
        ------
        ValueError
            If the argument is neither an `InstrumentModel` nor a sequence
            of them, or if a sequence has the wrong length.

        Notes
        -----
        `InstrumentModel` itself stays single-feed on purpose: a receptor
        chain is exactly what it already describes, and a dual-polarization
        array is two of them. Passing one model broadcasts it to both
        receptors -- the "the two feeds share a receiver" idealization,
        and the only behavior that keeps a `n_pol=2` run comparable to the
        `n_pol=1` run of the same instrument -- while passing two
        independently drawn models (different seeds) gives the two
        receptors genuinely different bandpasses, which is the realistic
        case.
        """
        if isinstance(instrument, InstrumentModel):
            return [instrument] * self.n_pol
        try:
            models = list(instrument)
        except TypeError as exc:
            raise ValueError(
                "instrument must be an InstrumentModel, a sequence of them (one per "
                f"polarization), or None, got {type(instrument)!r}"
            ) from exc
        if not models or not all(isinstance(model, InstrumentModel) for model in models):
            raise ValueError(
                "instrument must be an InstrumentModel, a sequence of them (one per "
                f"polarization), or None, got {type(instrument)!r}"
            )
        if len(models) != self.n_pol:
            raise ValueError(
                f"instrument has {len(models)} models but the simulator has "
                f"n_pol={self.n_pol}; pass one model per polarization, or a single "
                "model to give both receptors the same gains"
            )
        return models

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

    @property
    def instrument_gains(self) -> np.ndarray | None:
        """numpy.ndarray or None: Applied gains, shape ``(n_antennas, n_chan)``.

        The `instrument` model evaluated on `freq_hz`, complex64 -- shape
        ``(n_antennas, n_pol, n_chan)`` for a dual-polarization run, whose
        receptors may carry different models -- or ``None`` for a run with
        no instrument model. A copy, so the simulator's own copy cannot be
        edited through it.
        """
        return None if self._gains is None else self._block_gains()

    def _block_gains(self) -> np.ndarray:
        """A copy of the applied gains in the shape a block carries them.

        Returns
        -------
        numpy.ndarray
            Complex64 ``(n_antennas, n_chan)`` for a single-polarization
            run -- the polarization axis is dropped for the same reason
            `VoltageBlock.data`'s is -- and ``(n_antennas, n_pol,
            n_chan)`` otherwise.
        """
        gains = self._gains
        if self.n_pol == 1:
            gains = gains.reshape(gains.shape[0], gains.shape[2])
        return gains.copy()

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

    def _ideal_block(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Synthesize one block's voltages under the perfect channelizer.

        Parameters
        ----------
        index : int
            Block index in ``[0, n_blocks)``.

        Returns
        -------
        data : numpy.ndarray
            Complex64 voltages -- sky, receiver noise, spectral lines and
            interference, with neither the filterbank, nor the gains, nor
            quantization applied -- in the same shape convention as
            `VoltageBlock.data`: ``(n_antennas, n_chan, n_time)`` for a
            single-polarization run and ``(n_antennas, n_pol, n_chan,
            n_time)`` otherwise. Synthesis itself always carries the
            polarization axis (`_with_pol_axis`); it is dropped here, once.
        rfi_mask : numpy.ndarray
            Boolean ``(n_rfi_sources, n_chan, n_time)`` occupancy labels.
        celestial_mask : numpy.ndarray
            Boolean ``(n_celestial_sources, n_chan, n_time)`` labels.

        Notes
        -----
        Split out from `block` because it is also what the *previous*
        block contributes to a filterbank's state at a block seam: the
        filter is FIR, so the seam depends on the neighbour's ideal
        voltages and on nothing further back.
        """
        return self._synthesize_ideal(index, self.block_rng(index), self.rfi_block_rngs(index))

    def _synthesize_ideal(
        self,
        index: int,
        rng: np.random.Generator,
        rfi_rngs: Sequence[np.random.Generator],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Synthesize one block from explicitly supplied generators.

        Parameters
        ----------
        index : int
            Block index, which fixes the *geometry and timing* of the
            synthesis (delays, sample times, source positions).
        rng : numpy.random.Generator
            Generator for the sky/noise branch.
        rfi_rngs : sequence of numpy.random.Generator
            One generator per entry of `rfi_sources`.

        Returns
        -------
        tuple
            As `_ideal_block`.

        Notes
        -----
        The randomness is a parameter rather than a lookup on `index` so
        that the same synthesis can also produce the *virtual*
        pre-observation block a warm filterbank start needs, from its own
        branch of the seed tree -- see `_warm_start_state`. `_ideal_block`
        is this method with the block's own generators.
        """
        n_ant = self.array.n_antennas
        n_pol = self.n_pol
        n_chan = self.n_chan
        n_time = self.n_time_per_block

        data = np.zeros((n_ant, n_pol, n_chan, n_time), dtype=np.complex64)

        for i_src, source in enumerate(self.sources):
            if source.flux_jy == 0.0:
                continue
            # One sky signal, shared by every antenna -- but an independent
            # realization per receptor at the *same* level: the sky is
            # unpolarized here, so each receptor sees the source at its
            # full Stokes-I flux and the two are uncorrelated. That is the
            # convention that keeps a dual-polarization image's flux equal
            # to a single-polarization one (see the module docstring).
            spectrum = self._circular_normal(rng, (n_pol, n_chan, n_time), np.sqrt(source.flux_jy))
            tau_s = self._source_delays_s[i_src, index]  # (n_ant,)
            # RF frequency of each channel -- NOT a baseband offset.
            phase = np.exp(-2j * np.pi * self.freq_hz[np.newaxis, :] * tau_s[:, np.newaxis]).astype(
                np.complex64
            )
            if self.primary_beam is not None:
                # Voltage-domain factor is the square root of the power
                # response (see rfi_simulator.beam); common to every
                # antenna, so it lands in the visibility as the power
                # response itself once both antennas' factors multiply.
                beam_amplitude = np.sqrt(self._beam_power[i_src]).astype(np.complex64)
                phase = phase * beam_amplitude[np.newaxis, :]
            data += phase[:, np.newaxis, :, np.newaxis] * spectrum[np.newaxis]

        if self.noise_std > 0.0:
            # Independent receiver noise per antenna and per receptor.
            data += self._circular_normal(rng, (n_ant, n_pol, n_chan, n_time), self.noise_std)

        celestial_mask = self._add_spectral_lines(data, rng)
        rfi_mask = self._add_rfi(data, index, rfi_rngs)
        return self._without_pol_axis(data), rfi_mask, celestial_mask

    def _with_pol_axis(self, data: np.ndarray) -> np.ndarray:
        """View of `data` with the polarization axis always present.

        Parameters
        ----------
        data : numpy.ndarray
            Voltages in the `VoltageBlock.data` shape convention.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_antennas, n_pol, n_chan, n_time)``; a reshaped view
            for a single-polarization run, `data` itself otherwise.
        """
        if self.n_pol == 1:
            return data.reshape(data.shape[0], 1, *data.shape[1:])
        return data

    def _without_pol_axis(self, data: np.ndarray) -> np.ndarray:
        """View of `data` in the `VoltageBlock.data` shape convention.

        Parameters
        ----------
        data : numpy.ndarray
            Voltages of shape ``(n_antennas, n_pol, n_chan, n_time)``.

        Returns
        -------
        numpy.ndarray
            The same array without its length-one polarization axis for a
            single-polarization run, and unchanged otherwise.
        """
        if self.n_pol == 1:
            return data.reshape(data.shape[0], *data.shape[2:])
        return data

    def _seam_state(self, index: int) -> np.ndarray | None:
        """The filterbank state left over from the block before `index`.

        Parameters
        ----------
        index : int
            Block index in ``[0, n_blocks)``.

        Returns
        -------
        numpy.ndarray or None
            State to hand to `rfi_simulator.channelizer.PFBChannelizer.apply`
            -- `_warm_start_state` for block 0, or ``None`` for block 0 of
            a ``warm_start=False`` run (the backend has just been switched
            on, so the first ``n_taps - 1`` samples of the observation ramp
            up) and for a memoryless filterbank.

        Notes
        -----
        This is what keeps a filterbank's temporal correlation *continuous
        across block boundaries* while `block` stays a pure function of
        ``(seed, index)``: the state is the tail of block ``index - 1``'s
        ideal voltages, which is itself a pure function of the seed and
        ``index - 1``. Regenerating it costs one extra ideal block, so
        `blocks` caches the tail it produces on the way past and the common
        case -- iterating in order -- pays nothing.
        """
        if self.channelizer is None or self.channelizer.n_taps == 1:
            return None
        if index == 0:
            return self._warm_start_state() if self.warm_start else None
        cached = self._seam_cache
        if cached is not None and cached[0] == index - 1:
            return cached[1]
        previous, _, _ = self._ideal_block(index - 1)
        return self.channelizer.trailing_state(self._with_pol_axis(previous))

    def _warm_start_state(self) -> np.ndarray:
        """The filterbank state at the leading edge of block 0.

        Returns
        -------
        numpy.ndarray
            State to hand to `rfi_simulator.channelizer.PFBChannelizer.apply`
            for block 0, built from a *virtual pre-observation block* --
            one full block of ideal voltages synthesized exactly like any
            other, from a dedicated branch of the seed tree, of which only
            the last ``n_taps - 1`` samples survive into the filter state.

        Notes
        -----
        Why this exists: a filter started from zeros makes the first
        output sample of the observation strongly attenuated (identically
        zero for windows with zero endpoints) and the next few ramp up,
        in every channel at once. That is what switching a
        backend on looks like, and it is *not* what a dump taken from a
        running telescope looks like -- there the filterbank has been fed
        for hours before the recording started. The all-channel ramp is
        both an unphysical artifact and a strong statistical tell: it
        correlates every channel pair over the first few samples, which
        inflates any measurement of cross-channel correlation made on the
        block (by half again at the package defaults, and worse for
        longer prototypes).

        The pre-observation block is drawn from its own seed branch, so
        block 0 remains a pure function of ``(seed, 0)`` -- generated
        first, last or twice, it is the same data. It is synthesized with
        block 0's geometry rather than a genuine block ``-1``'s: over one
        block the delays move by parts in :math:`10^{5}`, and the state
        only has to be *statistically* indistinguishable from the stream
        that would have preceded the observation, not to continue a
        particular one. Cost: one extra ideal block, once, cached here --
        the same price `_seam_state` already pays for a block generated
        out of order. A run whose blocks were shorter than the filter
        needs to look back further than one block holds, which
        `VoltageSimulator.__init__` rejects outright (see its
        `n_time_per_block` validation), so this always has a full
        ``n_taps - 1`` rows of history to draw the state from.
        """
        if self._warm_cache is None:
            data, _, _ = self._synthesize_ideal(
                0,
                np.random.default_rng(self._warm_seed_sequence),
                [np.random.default_rng(child) for child in self._warm_rfi_seed_sequences],
            )
            state = self.channelizer.trailing_state(self._with_pol_axis(data))
            self._warm_cache = state
        return self._warm_cache

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
        order, repeated, or skipped without changing any of them. With a
        `channelizer` attached, a block out of sequence costs twice as much
        as one in sequence, because the filter state at its leading edge
        has to be regenerated from the previous block (see `_seam_state`).
        """
        data, rfi_mask, celestial_mask = self._ideal_block(index)
        data = self._with_pol_axis(data)
        start_time = self.start_time + index * self.block_duration_s * u.s

        if self.channelizer is not None:
            # The filterbank's leading axes are just independent streams,
            # so (antenna, receptor) pairs pass through it exactly as
            # antennas alone used to: each receptor is its own signal
            # chain and is filtered independently.
            data, tail = self.channelizer.apply(data, self._seam_state(index))
            if tail is not None:
                self._seam_cache = (index, tail)
            # Ground truth must follow the data: a filterbank that spreads
            # a channel-centered signal into its neighbours makes those
            # neighbours occupied too. Zero for any reasonably long
            # prototype, which leaves the labels untouched.
            radius = self.channelizer.leakage_radius(self.n_chan, OCCUPANCY_THRESHOLD)
            rfi_mask = _widen_channels(rfi_mask, radius)
            celestial_mask = _widen_channels(celestial_mask, radius)

        # The receiver chain sees sky, interference and noise alike, so the
        # gains go on last, on the total stream.
        if self._gains is not None:
            data *= self._gains[:, :, :, np.newaxis]

        clip_fraction: np.ndarray | None = None
        quant_scale: float | None = None
        if self.quantization == "int4":
            quant_scale = (
                self.quant_scale
                if self.quant_scale is not None
                else suggest_quant_scale(data, target_counts=self.quant_target_counts)
            )
            # One scale for the whole block, as a real backend applies one
            # digital gain setting: per-antenna gain scatter then shows up
            # as per-antenna saturation instead of being normalized away.
            data, clipped = quantize_roundtrip(data, quant_scale)
            # Per antenna *and* per receptor: one polarized interferer can
            # rail one receptor while the other stays clean, and a single
            # per-antenna number would average that signature away.
            clip_fraction = clipped.reshape(self.n_antennas, self.n_pol, -1).mean(
                axis=2, dtype=np.float64
            )
            if self.n_pol == 1:
                clip_fraction = clip_fraction[:, 0]

        # The polarization axis exists only inside the synthesis; a
        # single-receptor block carries the historical 3-D shape.
        data = self._without_pol_axis(data)

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
            rfi_coupling=self.rfi_coupling(),
            rfi_polarization=self.rfi_polarization(),
            celestial_mask=celestial_mask,
            celestial_source_names=tuple(line.name for line in self.spectral_lines),
            gains=None if self._gains is None else self._block_gains(),
            clip_fraction=clip_fraction,
            quant_scale=quant_scale,
            channelizer=self.channelizer,
            beam_response=self.beam_response(),
        )

    def _add_spectral_lines(self, data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Add every spectral-line foreground to `data` and collect their labels.

        Parameters
        ----------
        data : numpy.ndarray
            Complex64 ``(n_ant, n_pol, n_chan, n_time)`` voltages, modified
            in place. Called after the sky sources and receiver noise, before
            interference and before the instrument gains, so a line passes
            through gains exactly like every other component of the total
            stream.
        rng : numpy.random.Generator
            This block's sky/noise generator (see the class Notes) --
            spectral lines are celestial, so they draw from the same
            branch as the sky and receiver noise, never the interference
            branch.

        Returns
        -------
        numpy.ndarray
            Boolean ``(n_lines, n_chan, n_time)`` occupancy masks, in the
            order of `spectral_lines`, class ``"celestial"`` -- see
            `VoltageBlock.celestial_mask`.
        """
        n_ant, n_pol = self.array.n_antennas, self.n_pol
        n_chan, n_time = self.n_chan, self.n_time_per_block
        masks = np.zeros((len(self.spectral_lines), n_chan, n_time), dtype=bool)
        for i_line, line in enumerate(self.spectral_lines):
            envelope_jy = line.power_envelope_jy(self.freq_hz)  # (n_chan,)
            scale = np.sqrt(envelope_jy).astype(np.float32)
            # Independent per antenna, exactly like the receiver noise --
            # the "fully resolved extended emission" approximation (see
            # rfi_simulator.sky.SpectralLineForeground): unit-power circular
            # Gaussian samples scaled by the line's frequency profile, so no
            # correlated component reaches the correlator.
            # Unpolarized, like the receiver noise: an independent
            # realization per receptor at the same level.
            unit = self._circular_normal(rng, (n_ant, n_pol, n_chan, n_time), 1.0)
            data += unit * scale[np.newaxis, np.newaxis, :, np.newaxis]
            masks[i_line] = line.mask(self.freq_hz, n_time)
        return masks

    def beam_response(self) -> np.ndarray | None:
        """Per-source, per-channel primary-beam power response -- ground truth.

        Returns
        -------
        numpy.ndarray or None
            Float64 array of shape ``(n_sources, n_chan)`` -- see
            `VoltageBlock.beam_response` -- or ``None`` if no
            `primary_beam` is attached, so a run with no beam still
            carries the correct ("nothing was attenuated") truth rather
            than an all-ones array standing in for "not applicable".
            Constant across blocks, like `rfi_coupling`, since a source's
            offset from the pointing center does not change (see
            `rfi_simulator.sky`).
        """
        return None if self._beam_power is None else self._beam_power.copy()

    def rfi_coupling(self) -> np.ndarray:
        """Per-antenna coupling of every interference source -- ground truth.

        Returns
        -------
        numpy.ndarray
            Float64 array of shape ``(n_interference_sources, n_antennas)``
            of linear amplitude factors, in the order of `rfi_sources`;
            received power scales as the square. All ones for a source with
            uniform coupling, so a run with no coupling configured still
            gets a well-defined (and correct) ground truth rather than
            ``None``. Shape ``(0, n_antennas)`` for a run with no
            interference.

        Notes
        -----
        Coupling is a fixed property of the installation, not of a block,
        so this is the same array for every block of a run. It is attached
        to each block anyway (`VoltageBlock.rfi_coupling`), for the same
        reason the gains are: a block that has been written out and read
        back should carry its own truth.
        """
        coupling = np.ones((len(self.rfi_sources), self.array.n_antennas), dtype=np.float64)
        for i_src, source in enumerate(self.rfi_sources):
            coupling[i_src] = source.coupling_amplitudes(self.array.n_antennas)
        return coupling

    def rfi_polarization(self) -> np.ndarray:
        """Per-receptor amplitudes of every interference source -- ground truth.

        Returns
        -------
        numpy.ndarray
            Complex128 array of shape ``(n_interference_sources, n_pol)``
            of amplitude factors, in the order of `rfi_sources`; received
            power scales as ``|a|**2`` and Stokes I as the mean over
            receptors (see `rfi_simulator.rfi.resolve_polarization`). All
            ones for an unpolarized source and for every source of a
            single-polarization run, so a run that configures no
            polarization still gets a well-defined -- and correct --
            ground truth rather than ``None``, exactly as `rfi_coupling`
            does. Shape ``(0, n_pol)`` for a run with no interference.

        Notes
        -----
        Kept separate from `VoltageBlock.rfi_mask`, which has no receptor
        axis: which channels a transmitter occupies does not depend on the
        receptor, only how strongly each one hears it does. A flagger that
        exploits the polarization asymmetry is scored against the
        occupancy labels and calibrated against these amplitudes.
        """
        amplitudes = np.ones((len(self.rfi_sources), self.n_pol), dtype=np.complex128)
        for i_src, source in enumerate(self.rfi_sources):
            amplitudes[i_src] = source.polarization_amplitudes(self.n_pol)
        return amplitudes

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
            channelizer=self.channelizer,
            n_pol=self.n_pol,
        )

    def _add_rfi(
        self, data: np.ndarray, index: int, rngs: Sequence[np.random.Generator]
    ) -> np.ndarray:
        """Add every interference source to `data` and collect their labels.

        Parameters
        ----------
        data : numpy.ndarray
            Complex64 ``(n_ant, n_pol, n_chan, n_time)`` sky-plus-noise
            voltages, modified in place.
        index : int
            Block index, which fixes the geometry and timing handed to
            each source.
        rngs : sequence of numpy.random.Generator
            One generator per entry of `rfi_sources`, in the same order --
            ordinarily `rfi_block_rngs`, but the warm-start branch for the
            virtual pre-observation block.

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

        n_ant = self.array.n_antennas
        pol_shape = (n_ant, self.n_pol, n_chan, n_time)
        expected_voltage_shape = (n_ant, n_chan, n_time) if self.n_pol == 1 else pol_shape
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
            data += voltages.astype(np.complex64, copy=False).reshape(pol_shape)
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
