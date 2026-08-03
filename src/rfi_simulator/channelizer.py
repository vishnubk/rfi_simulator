r"""Polyphase-filterbank channel response applied to channelized voltages.

`rfi_simulator.voltages` synthesizes voltages *already channelized*, and by
default it assumes a **perfect channelizer**: every channel is an ideal,
brick-wall, critically sampled slice of the band, so each channel's time
series is white and no two channels are correlated. Real backends do not
work that way. They digitize a wideband stream and run it through a
polyphase filterbank (PFB), whose prototype filter spans several output
samples. Two consequences dominate the statistics of the resulting data:

* **temporal memory** -- consecutive samples of one channel are correlated,
  because the filter's impulse response covers several output samples;
* **spectral leakage** -- neighbouring channels are correlated, because the
  channel responses overlap, and a carrier that sits between two channel
  centers appears in both with a definite amplitude ratio and phase.

This module models both, exactly, without ever leaving the channelized
representation's sample count.

The model
---------
A critically sampled analysis filterbank with :math:`M` channels and a
prototype filter :math:`h[n]` of length :math:`L = P M` (``P = n_taps``)
produces

.. math::

    y_k[m] = \frac{1}{\sqrt{M}} \sum_{n=0}^{L-1} h[n]\, x[mM + n]\,
             e^{-2\pi i k n / M},

for a wideband complex-baseband input :math:`x`. Written in polyphase form
this is a length-``n_taps`` FIR filter along the output-sample axis (one
set of taps per polyphase branch) followed by a length-``M`` DFT, which is
how `PFBChannelizer.apply` evaluates it.

The point of the module is that this can be applied to data the simulator
has *already* channelized. Given an ideal (perfect-channelizer) block
``Z[k, m]``, the unitary inverse DFT along the channel axis reconstructs
exactly the wideband stream ``x`` whose ideal channelization is ``Z``; the
PFB is then run on that stream. So the operator is

    ideal synthesis (inverse channelization)  ->  PFB analysis

which is a *physically correct re-channelization*, not an imposed
covariance: white noise comes out with the right coloring, a carrier at a
channel center comes out with the right leakage into its neighbours, and a
carrier between two channel centers comes out with the right amplitude
ratio and phase -- all from one linear operator, applied once to the total
signal, exactly as a real backend channelizes sky, interference and
receiver noise together.

Because the inverse DFT is unitary, the operator costs two length-``M``
FFTs and ``n_taps`` complex multiply-adds per output sample: the same
order as the synthesis it follows, with no increase in sample count.

Second-order statistics
-----------------------
For white input the output statistics follow from :math:`h` alone. With
:math:`\sum_n h[n]^2 = M` (the normalization this module uses, so that
channel power is preserved exactly):

* per-channel temporal autocorrelation at lag :math:`p`,
  :math:`\rho[p] = \sum_n h[n] h[n + pM] / \sum_n h[n]^2`, the same in
  every channel (`temporal_autocorrelation`);
* channel-to-channel coherence at lag zero,
  :math:`\gamma_{\Delta k} = \sum_n h[n]^2 e^{2\pi i \Delta k\, n/M} /
  \sum_n h[n]^2` (`adjacent_channel_coherence` for
  :math:`\Delta k = 1`);
* the response to a tone offset by :math:`\delta` channels from a channel
  center, :math:`H(\delta) = M^{-1/2} \sum_n h[n] e^{2\pi i \delta n/M}`
  (`channel_response`).

Both correlations vanish as the prototype approaches an ideal brick wall,
which is the perfect-channelizer limit and the reason the default model is
*off*.

Band edges
----------
The DFT pair is cyclic in the channel index, so the lowest and highest
simulated channels are each other's neighbours. That is not an
approximation: the reconstructed stream is a complex-baseband signal of
bandwidth exactly ``n_chan * chan_width``, sampled critically, and its
spectrum is periodic. A filterbank fed such a stream really does alias its
edge channels into one another. It also makes channel power preservation
exact everywhere, including at the edges.

Prototype filter
----------------
``h[n] = w[n] * sinc(sinc_bandwidth * (n - (L-1)/2) / M)``, i.e. the usual
windowed-sinc PFB prototype: the sinc sets the nominal channel width, the
window (`WINDOWS`) sets the stopband, and `sinc_bandwidth` trims the
main-lobe width. Longer filters and narrower main lobes approach the ideal
brick wall, so they *reduce* every statistic above; short filters leak.
The package defaults (`DEFAULT_N_TAPS` taps, a `DEFAULT_WINDOW` window and
a sinc bandwidth of `DEFAULT_SINC_BANDWIDTH`) sit where the coloring is
comparable to what wideband receiver backends typically show -- see the
class docstring for the predicted numbers and for what they trade off.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DEFAULT_N_TAPS",
    "DEFAULT_SINC_BANDWIDTH",
    "DEFAULT_WINDOW",
    "WINDOWS",
    "PFBChannelizer",
    "ideal_channel_weights",
]

#: Prototype-filter windows this module knows how to build, as symmetric
#: (not periodic) windows of length ``n_taps * n_chan``.
WINDOWS = ("hann", "hamming", "blackman")

#: Default prototype length, in output samples (taps per polyphase branch).
#: Four taps puts the temporal and adjacent-channel correlations in the
#: range wideband backends show; eight taps is a sharper filter that leaves
#: the data nearly white again.
DEFAULT_N_TAPS = 4

#: Default prototype window.
DEFAULT_WINDOW = "hamming"

#: Default sinc main-lobe width, in channels. ``1.0`` -- nominal channel
#: width equal to the channel spacing -- is the textbook choice; the
#: default is a shade wider, which is what puts both default correlations
#: where measured wideband backends put them (see `PFBChannelizer`).
DEFAULT_SINC_BANDWIDTH = 1.025

_WINDOW_BUILDERS = {
    "hann": np.hanning,
    "hamming": np.hamming,
    "blackman": np.blackman,
}


def ideal_channel_weights(offset_channels, n_chan: int) -> np.ndarray:
    r"""Perfect-channelizer amplitudes of a tone offset from a channel center.

    A tone at :math:`u` channels from the center of channel ``k`` does not
    land in channel ``k`` alone even under a *perfect* (plain-DFT)
    channelizer: the finite transform length spreads it over the band with
    the Dirichlet kernel

    .. math::

        c(u) = \frac{1}{M} \sum_{j=0}^{M-1} e^{2\pi i u j / M}
             = \frac{1}{M} e^{i\pi u (M-1)/M}
               \frac{\sin(\pi u)}{\sin(\pi u / M)} .

    Feeding these amplitudes to `PFBChannelizer.apply` reconstructs the
    tone exactly, which is how `rfi_simulator.rfi` places a carrier at an
    arbitrary frequency rather than snapping it to a channel center.

    Parameters
    ----------
    offset_channels : array_like
        Tone frequency minus channel center frequency, in units of the
        channel width, one entry per channel.
    n_chan : int
        Number of channels, ``>= 1``.

    Returns
    -------
    numpy.ndarray
        Complex128 array of the same shape as `offset_channels`, with
        ``sum(abs(c)**2) == 1`` over a full band of channel offsets: the
        spreading conserves power, it does not create it. Exactly
        ``1`` at zero offset and ``0`` at every nonzero integer offset, so
        a channel-centered tone reduces to the single-channel convention
        used when no channelizer is attached.

    Examples
    --------
    >>> import numpy as np
    >>> w = ideal_channel_weights([-1.0, 0.0, 1.0], 16)
    >>> bool(np.allclose(np.abs(w), [0.0, 1.0, 0.0]))
    True
    """
    n_chan = int(n_chan)
    if n_chan < 1:
        raise ValueError(f"n_chan must be >= 1, got {n_chan}")
    u = np.asarray(offset_channels, dtype=np.float64)
    denominator = np.sin(np.pi * u / n_chan)
    # u == 0 (mod n_chan) is the single removable singularity, where the
    # kernel is exactly one; every other integer offset gives exactly zero.
    at_center = np.isclose(denominator, 0.0, atol=1e-12)
    safe = np.where(at_center, 1.0, denominator)
    weights = np.exp(1j * np.pi * u * (n_chan - 1) / n_chan) * np.sin(np.pi * u) / (n_chan * safe)
    # The kernel is exactly one at every multiple of n_chan (the band is
    # cyclic), which is the removable singularity the closed form misses.
    return np.where(at_center, np.ones_like(weights), weights)


class PFBChannelizer:
    """Polyphase-filterbank channel response for channelized voltages.

    An instance is a *shape-free* description of a filterbank: the number
    of taps, the prototype window and the sinc main-lobe width. Everything
    that depends on the channel count -- the prototype filter itself and
    the statistics derived from it -- is a method taking ``n_chan``, so one
    channelizer can be attached to simulators of different widths. Derived
    filters are cached per channel count.

    Parameters
    ----------
    n_taps : int, optional
        Prototype filter length in output samples, i.e. taps per polyphase
        branch; the filter is ``n_taps * n_chan`` samples long. Must be
        ``>= 1``. Default `DEFAULT_N_TAPS`. ``1`` is a plain windowed DFT
        (no temporal memory at all, heavy leakage); large values approach
        an ideal brick wall and reproduce the perfect-channelizer limit.
    window : {"hann", "hamming", "blackman"}, optional
        Prototype window, symmetric, of length ``n_taps * n_chan``. Default
        `DEFAULT_WINDOW`.
    sinc_bandwidth : float, optional
        Width of the prototype's sinc main lobe, in channels. Default
        `DEFAULT_SINC_BANDWIDTH`; ``1.0``, nominal channel width equal to
        the channel spacing, is the textbook choice. Below 1 the channels
        are narrower than their spacing, which suppresses leakage and
        raises the temporal correlation; above 1 they overlap more, which
        does the reverse. Together with `n_taps` and `window` this is the
        knob for matching a measured filterbank once its correlations are
        known -- it trades the two correlations against each other along a
        curve that neither of the other two can move you along as finely.

    Attributes
    ----------
    n_taps : int
    window : str
    sinc_bandwidth : float

    Raises
    ------
    ValueError
        If `n_taps` is not an integer ``>= 1``, `window` is not one of
        `WINDOWS`, or `sinc_bandwidth` is not finite and positive.

    Notes
    -----
    At the package defaults (4 taps, hamming window, sinc bandwidth 1.025)
    the filter predicts, essentially independently of the channel count:

    ==========================================  =========
    lag-1 temporal autocorrelation              ``0.136``
    adjacent-channel coherence magnitude        ``0.133``
    tone leakage ratio at 0.40 channels offset  ``0.45``
    ==========================================  =========

    Those are the numbers the defaults exist to produce: they are what
    wideband receiver backends show, against ``0.0005`` and ``0.007`` for
    the perfect channelizer. The three cannot be matched simultaneously by
    any windowed-sinc prototype -- a filter leaky enough to share a tone
    between two channels in a large ratio is also one whose neighbouring
    channels are strongly coherent -- so the defaults sit at the best
    joint fit for the two *correlations*, and the leakage ratio comes out
    somewhat lower than a real bank's.

    Cost: two length-``n_chan`` FFTs and ``n_taps`` complex multiply-adds
    per output sample, per antenna, applied once to the whole block, plus
    of order 50 MB of working memory at the package defaults (10 antennas,
    384 channels, 1000 samples per block). Measured there, it roughly
    doubles the time to synthesize a block -- about 0.2 s on a laptop-class
    core -- and it does not change the number of samples the simulator
    carries. Generating blocks *out of order* doubles that again, because
    the filter state at a block's leading edge has to be regenerated from
    its predecessor; iterating with
    `rfi_simulator.voltages.VoltageSimulator.blocks` does not.

    Examples
    --------
    >>> import numpy as np
    >>> pfb = PFBChannelizer()
    >>> float(round(pfb.temporal_autocorrelation(64)[1], 3))
    0.136
    >>> float(round(abs(pfb.adjacent_channel_coherence(64)), 3))
    0.133
    """

    def __init__(
        self,
        n_taps: int = DEFAULT_N_TAPS,
        window: str = DEFAULT_WINDOW,
        sinc_bandwidth: float = DEFAULT_SINC_BANDWIDTH,
    ) -> None:
        if isinstance(n_taps, bool) or not isinstance(n_taps, (int, np.integer)):
            raise ValueError(f"n_taps must be an integer, got {n_taps!r}")
        self.n_taps = int(n_taps)
        if self.n_taps < 1:
            raise ValueError(f"n_taps must be >= 1, got {self.n_taps}")

        self.window = str(window)
        if self.window not in WINDOWS:
            raise ValueError(f"window must be one of {WINDOWS}, got {self.window!r}")

        self.sinc_bandwidth = float(sinc_bandwidth)
        if not np.isfinite(self.sinc_bandwidth) or self.sinc_bandwidth <= 0.0:
            raise ValueError(f"sinc_bandwidth must be finite and > 0, got {self.sinc_bandwidth}")

        self._prototype_cache: dict[int, np.ndarray] = {}

    def __repr__(self) -> str:
        return (
            f"PFBChannelizer(n_taps={self.n_taps}, window={self.window!r}, "
            f"sinc_bandwidth={self.sinc_bandwidth!r})"
        )

    # ------------------------------------------------------------------
    # Prototype filter
    # ------------------------------------------------------------------
    def prototype_filter(self, n_chan: int) -> np.ndarray:
        """The prototype filter for a given channel count.

        Parameters
        ----------
        n_chan : int
            Number of channels, ``>= 1``.

        Returns
        -------
        numpy.ndarray
            Read-only float64 array of length ``n_taps * n_chan``,
            normalized so that ``sum(h**2) == n_chan``. That normalization
            -- not unit sum, not unit norm -- is what makes `apply`
            preserve the mean power of every channel exactly.

        Raises
        ------
        ValueError
            If `n_chan` is not a positive integer.
        """
        n_chan = int(n_chan)
        if n_chan < 1:
            raise ValueError(f"n_chan must be >= 1, got {n_chan}")
        cached = self._prototype_cache.get(n_chan)
        if cached is not None:
            return cached

        length = self.n_taps * n_chan
        index = np.arange(length, dtype=np.float64)
        window = _WINDOW_BUILDERS[self.window](length)
        taper = np.sinc(self.sinc_bandwidth * (index - 0.5 * (length - 1)) / n_chan)
        h = window * taper
        power = float(np.sum(h**2))
        if power <= 0.0:  # pragma: no cover - only reachable for degenerate windows
            raise ValueError(
                f"prototype filter for window={self.window!r}, n_taps={self.n_taps}, "
                f"n_chan={n_chan} has zero energy"
            )
        h *= np.sqrt(n_chan / power)
        h.setflags(write=False)
        self._prototype_cache[n_chan] = h
        return h

    def polyphase_taps(self, n_chan: int) -> np.ndarray:
        """The prototype reshaped into its polyphase branches.

        Parameters
        ----------
        n_chan : int
            Number of channels.

        Returns
        -------
        numpy.ndarray
            Float64 array of shape ``(n_taps, n_chan)``: entry ``[t, j]``
            is ``h[t * n_chan + j]``, the tap branch ``j`` applies to the
            input block ``t`` samples back.
        """
        return self.prototype_filter(n_chan).reshape(self.n_taps, int(n_chan))

    # ------------------------------------------------------------------
    # Derived second-order statistics
    # ------------------------------------------------------------------
    def temporal_autocorrelation(self, n_chan: int) -> np.ndarray:
        r"""Per-channel autocorrelation of channelized white noise.

        Parameters
        ----------
        n_chan : int
            Number of channels.

        Returns
        -------
        numpy.ndarray
            Float64 array of length `n_taps`: entry ``p`` is
            :math:`\rho[p] = \sum_n h[n] h[n+pM] / \sum_n h[n]^2`, the
            normalized autocorrelation of one channel's complex time
            series at lag ``p`` output samples. ``rho[0] == 1``, and every
            lag ``>= n_taps`` is exactly zero because the prototype is that
            long. The same for every channel -- the filterbank is
            stationary in the channel index.

        Notes
        -----
        Real, not complex: the lag-``p`` correlation carries a factor
        :math:`e^{2\pi i k p M / M} = 1`, so it is independent of the
        channel and of any frequency offset.
        """
        h = self.prototype_filter(n_chan)
        n_chan = int(n_chan)
        energy = float(np.sum(h**2))
        lags = np.empty(self.n_taps, dtype=np.float64)
        for p in range(self.n_taps):
            shift = p * n_chan
            lags[p] = float(np.sum(h[: h.size - shift] * h[shift:])) / energy
        return lags

    def channel_coherence(self, n_chan: int, delta_channels: int = 1) -> complex:
        r"""Zero-lag complex coherence between channels `delta_channels` apart.

        Parameters
        ----------
        n_chan : int
            Number of channels.
        delta_channels : int, optional
            Channel separation. Default 1 (adjacent channels).

        Returns
        -------
        complex
            :math:`\langle v_k v_{k+\Delta}^* \rangle /
            \sqrt{P_k P_{k+\Delta}}` for white input:
            :math:`\sum_n h[n]^2 e^{2\pi i \Delta n / M} / \sum_n h[n]^2`.
            Its magnitude is the quantity a coherence measurement returns;
            its phase is the fixed phase the leakage arrives with.
        """
        h = self.prototype_filter(n_chan)
        n_chan = int(n_chan)
        index = np.arange(h.size, dtype=np.float64)
        weights = h**2
        phasor = np.exp(2j * np.pi * int(delta_channels) * index / n_chan)
        return complex(np.sum(weights * phasor) / np.sum(weights))

    def adjacent_channel_coherence(self, n_chan: int) -> complex:
        """Zero-lag complex coherence between neighbouring channels.

        Parameters
        ----------
        n_chan : int
            Number of channels.

        Returns
        -------
        complex
            ``channel_coherence(n_chan, 1)``.
        """
        return self.channel_coherence(n_chan, 1)

    def channel_response(self, offset_channels, n_chan: int) -> np.ndarray:
        r"""Complex response of a channel to a tone offset from its center.

        Parameters
        ----------
        offset_channels : array_like
            Tone frequency minus channel center frequency, in units of the
            channel width.
        n_chan : int
            Number of channels.

        Returns
        -------
        numpy.ndarray
            Complex128 array of the same shape as `offset_channels`:
            :math:`H(\delta) = M^{-1/2} \sum_n h[n] e^{2\pi i \delta n/M}`,
            the amplitude a unit-amplitude wideband tone produces in a
            channel whose center it misses by `offset_channels`.

        Notes
        -----
        Only *ratios* of this function are convention-independent, and
        ratios are what the simulator uses: the fraction of a carrier's
        power that lands in each channel is
        ``abs(H(delta_k))**2 / sum_k abs(H(delta_k))**2``. A tone also
        acquires a phase rotation of ``exp(2j*pi*delta)`` per output
        sample, which `PFBChannelizer.apply` reproduces on its own.
        """
        h = self.prototype_filter(n_chan)
        n_chan = int(n_chan)
        offsets = np.asarray(offset_channels, dtype=np.float64)
        index = np.arange(h.size, dtype=np.float64)
        phases = np.exp(2j * np.pi * offsets[..., np.newaxis] * index / n_chan)
        return (phases * h).sum(axis=-1) / np.sqrt(n_chan)

    def leakage_radius(self, n_chan: int, threshold: float) -> int:
        """How many channels away a channel-centered signal stays detectable.

        Parameters
        ----------
        n_chan : int
            Number of channels.
        threshold : float
            Power fraction, relative to the on-center response, above which
            a neighbouring channel counts as occupied. Pass
            `rfi_simulator.rfi.OCCUPANCY_THRESHOLD` to match the package's
            ground-truth labelling convention.

        Returns
        -------
        int
            The largest ``r`` with ``abs(H(r))**2 / abs(H(0))**2 >
            threshold``, capped at ``n_chan // 2``; ``0`` when the
            filterbank confines a channel-centered signal to its own
            channel at that threshold, which is the case for every
            reasonably long prototype.

        Notes
        -----
        Used to widen ground-truth occupancy labels so that they still
        cover every channel that actually receives power once the
        filterbank has spread it -- see `rfi_simulator.voltages`.
        """
        threshold = float(threshold)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError(f"threshold must be finite and > 0, got {threshold}")
        limit = int(n_chan) // 2
        if limit < 1:
            return 0
        offsets = np.arange(limit + 1, dtype=np.float64)
        response = np.abs(self.channel_response(offsets, n_chan)) ** 2
        above = response > threshold * response[0]
        # Contiguous run from the center outwards: a deep stopband can dip
        # below the threshold and come back up in a sidelobe, and labelling
        # a far sidelobe while skipping the gap would be nonsense.
        radius = 0
        for r in range(1, limit + 1):
            if not above[r]:
                break
            radius = r
        return radius

    # ------------------------------------------------------------------
    # The operator
    # ------------------------------------------------------------------
    def state_shape(self, leading_shape: tuple[int, ...], n_chan: int) -> tuple[int, ...]:
        """Shape of the filter state carried between consecutive blocks.

        Parameters
        ----------
        leading_shape : tuple of int
            The leading (e.g. antenna) axes of the data `apply` will see.
        n_chan : int
            Number of channels.

        Returns
        -------
        tuple of int
            ``(*leading_shape, n_taps - 1, n_chan)``; the state is empty
            along its second-to-last axis for a single-tap filterbank.
        """
        return (*tuple(leading_shape), self.n_taps - 1, int(n_chan))

    def trailing_state(self, data: np.ndarray) -> np.ndarray | None:
        """The filter state a block leaves behind, without filtering it.

        Parameters
        ----------
        data : numpy.ndarray
            Complex array of shape ``(..., n_chan, n_time)``, as for
            `apply`.

        Returns
        -------
        numpy.ndarray or None
            The same state `apply` would return for `data`, or ``None`` for
            a single-tap filterbank. It depends only on the *last*
            ``n_taps - 1`` samples of `data`, and not at all on the state
            `data` itself was filtered with -- the filterbank is FIR -- so
            this reconstructs the seam between two blocks without filtering
            the earlier one. The one exception is a block shorter than the
            filter, ``n_time < n_taps - 1``, whose missing rows come out as
            zeros rather than from the block before last.
        """
        n_history = self.n_taps - 1
        if n_history == 0:
            return None
        data = np.asarray(data)
        if data.ndim < 2:
            raise ValueError(f"data must have shape (..., n_chan, n_time), got {data.shape}")
        n_time = int(data.shape[-1])
        tail = np.swapaxes(data[..., -min(n_history, n_time) :], -1, -2)
        grid = np.fft.ifft(tail, axis=-1, norm="ortho").astype(np.complex64)
        if n_time >= n_history:
            return grid
        pad = np.zeros((*grid.shape[:-2], n_history - n_time, grid.shape[-1]), dtype=np.complex64)
        return np.concatenate([pad, grid], axis=-2)

    def apply(
        self, data: np.ndarray, state: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Re-channelize ideally channelized voltages through this filterbank.

        Parameters
        ----------
        data : numpy.ndarray
            Complex array of shape ``(..., n_chan, n_time)`` holding
            *perfect-channelizer* voltages -- what
            `rfi_simulator.voltages.VoltageSimulator` synthesizes. The
            leading axes are carried through untouched (one independent
            stream each, e.g. one per antenna).
        state : numpy.ndarray, optional
            Filter state returned by the previous call on the *preceding*
            block of the same streams, shape `state_shape`. Default
            ``None``: start cold, with zeros, which attenuates the first
            ``n_taps - 1`` output samples exactly as switching a real
            backend on does.

        Returns
        -------
        out : numpy.ndarray
            Complex64 array of the same shape as `data`.
        state : numpy.ndarray or None
            State to hand to the next block, or ``None`` for a single-tap
            filterbank, which has no memory.

        Raises
        ------
        ValueError
            If `data` has fewer than two dimensions or `state` has the
            wrong shape.

        Notes
        -----
        The operator is deterministic and linear: it draws no random
        numbers, so attaching a channelizer never perturbs any component's
        realization for a given seed.
        """
        data = np.asarray(data)
        if data.ndim < 2:
            raise ValueError(f"data must have shape (..., n_chan, n_time), got {data.shape}")
        n_chan, n_time = int(data.shape[-2]), int(data.shape[-1])
        leading = data.shape[:-2]
        n_history = self.n_taps - 1

        # Ideal synthesis: the unitary inverse DFT along the channel axis
        # turns each output sample into the n_chan wideband samples whose
        # perfect channelization is exactly this block. Unitary, so white
        # channelized noise maps to white wideband noise.
        grid = np.fft.ifft(np.swapaxes(data, -1, -2), axis=-1, norm="ortho")
        grid = grid.astype(np.complex64, copy=False)

        expected_state = self.state_shape(leading, n_chan)
        if n_history == 0:
            padded = grid
            new_state = None
        else:
            if state is None:
                history = np.zeros(expected_state, dtype=np.complex64)
            else:
                history = np.asarray(state)
                if history.shape != expected_state:
                    raise ValueError(f"state must have shape {expected_state}, got {history.shape}")
                history = history.astype(np.complex64, copy=False)
            padded = np.concatenate([history, grid], axis=-2)
            # The tail of the padded stream is the next block's history:
            # the filter runs continuously across the seam. Taken from
            # `padded`, not `grid`, so that a block shorter than the filter
            # keeps carrying the samples that are still in flight.
            new_state = np.array(padded[..., n_time:, :], dtype=np.complex64)

        taps = self.polyphase_taps(n_chan).astype(np.float32)
        # Polyphase FIR along the output-sample axis: branch j of tap t
        # weights the wideband row (n_taps - 1 - t) samples in the past.
        filtered = padded[..., :n_time, :] * taps[0]
        for t in range(1, self.n_taps):
            filtered += padded[..., t : t + n_time, :] * taps[t]

        out = np.fft.fft(filtered, axis=-1, norm="ortho")
        return np.swapaxes(out, -1, -2).astype(np.complex64), new_state
