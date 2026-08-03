r"""Direct-DFT dirty imaging -- a verification tool, not a science imager.

There is no gridding, no FFT, no deconvolution and no ``w`` handling: the
image is evaluated by brute force on whatever ``(l, m)`` grid you ask for.
That is deliberate. The point of this module is to check that a source put
in at :math:`(l_0, m_0)` comes back out at :math:`(l_0, m_0)`, and a direct
DFT has no gridding conventions of its own to hide a sign error behind.

Conventions
-----------
The baseline coordinates are, for the stored pair ``(i, j)`` with baseline
vector :math:`\mathbf{b} = \mathbf{r}_i - \mathbf{r}_j`,

.. math::

    u = \frac{f\, \mathbf{b} \cdot \hat{e}_l}{c}, \qquad
    v = \frac{f\, \mathbf{b} \cdot \hat{e}_m}{c}, \qquad
    w = \frac{f\, \mathbf{b} \cdot \hat{s}_0}{c},

all in wavelengths at the channel's RF frequency ``f``. With the delay
convention of `rfi_simulator.delays` and the conjugation convention of
`rfi_simulator.correlator`, a fringe-stopped point source of flux ``F`` at
direction cosines :math:`(l_0, m_0)` produces

.. math::

    V = F\, e^{+2\pi i (u l_0 + v m_0 + w(n_0 - 1))},

so the matching inverse transform -- the one implemented here -- is

.. math::

    I(l, m) = \frac{1}{K} \sum_k \mathrm{Re}\left[
        V_k\, e^{-2\pi i (u_k l + v_k m)} \right],

summed over all ``K`` (integration, baseline, channel) samples with
natural weighting. The normalization is chosen so that a noiseless point
source of flux ``F`` peaks at exactly ``F``.

The neglected :math:`w(n_0 - 1)` term is why the tests phase up on the
zenith: for a flat array (all ``up = 0``) a zenith phase center makes
:math:`\mathbf{b} \cdot \hat{s}_0 = 0`, hence ``w = 0``, and the
tangent-plane transform above is exact. Point somewhere else and the term
comes back; `dirty_image` measures it and raises a `UserWarning` rather
than quietly returning a smeared map.
"""

from __future__ import annotations

import warnings

import numpy as np

from rfi_simulator.correlator import Visibilities
from rfi_simulator.delays import SPEED_OF_LIGHT_M_S

__all__ = ["dirty_image", "lm_axis", "uvw_wavelengths", "w_term_phase_rad"]

_MAX_PHASE_ELEMENTS = 1_000_000

W_TERM_WARN_PHASE_RAD = 0.1
"""float: Neglected w-term phase above which `dirty_image` warns, radians.

0.1 rad is roughly a 0.5% amplitude loss at the map edge -- small enough
to ignore, large enough that anything above it deserves a look.
"""


def lm_axis(field_of_view_rad: float, n_pix: int) -> np.ndarray:
    """Build a symmetric direction-cosine axis.

    Parameters
    ----------
    field_of_view_rad : float
        Full width of the axis in direction cosine (approximately
        radians for small fields).
    n_pix : int
        Number of pixels along the axis.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_pix,)`` float64 axis running from
        ``-field_of_view_rad / 2`` to ``+field_of_view_rad / 2``. A
        single-pixel axis is the field *center*, ``[0.0]``, not an edge.
    """
    if n_pix < 1:
        raise ValueError(f"n_pix must be >= 1, got {n_pix}")
    if n_pix == 1:
        return np.zeros(1, dtype=np.float64)
    return np.linspace(-0.5 * field_of_view_rad, 0.5 * field_of_view_rad, n_pix)


def uvw_wavelengths(vis: Visibilities) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Baseline coordinates in wavelengths for every (integration, baseline, channel).

    Parameters
    ----------
    vis : Visibilities
        Visibilities carrying baseline vectors, per-integration basis
        vectors and channel frequencies.

    Returns
    -------
    u, v, w : numpy.ndarray
        Shape ``(n_int, n_baselines, n_chan)`` float64 arrays of the
        baseline projections onto ``e_l``, ``e_m`` and the phase-center
        direction, in wavelengths. Only ``u`` and ``v`` enter the image;
        ``w`` is returned so callers can check how much they are ignoring.
    """
    scale = vis.freq_hz / SPEED_OF_LIGHT_M_S  # (n_chan,) 1/m
    b_l = np.einsum("bj,tj->tb", vis.baseline_vectors_enu_m, vis.e_l_enu)
    b_m = np.einsum("bj,tj->tb", vis.baseline_vectors_enu_m, vis.e_m_enu)
    b_s = np.einsum("bj,tj->tb", vis.baseline_vectors_enu_m, vis.s0_enu)
    u = b_l[:, :, np.newaxis] * scale[np.newaxis, np.newaxis, :]
    v = b_m[:, :, np.newaxis] * scale[np.newaxis, np.newaxis, :]
    w = b_s[:, :, np.newaxis] * scale[np.newaxis, np.newaxis, :]
    return u, v, w


def w_term_phase_rad(w: np.ndarray, l_grid: np.ndarray, m_grid: np.ndarray) -> float:
    """Largest phase error this imager throws away, in radians.

    The exact visibility of a point source carries a term
    ``exp(2 pi i w (n - 1))`` with ``n = sqrt(1 - l**2 - m**2)``, which the
    two-dimensional transform in `dirty_image` ignores. This returns its
    worst-case size over the given baselines and map extent.

    Parameters
    ----------
    w : numpy.ndarray
        Baseline projections onto the phase-center direction, wavelengths.
    l_grid, m_grid : numpy.ndarray
        The direction-cosine axes of the map.

    Returns
    -------
    float
        ``2 pi max|w| max|n - 1|`` in radians. Zero for a flat array
        pointed at the zenith, where every baseline has ``w = 0``.
    """
    if w.size == 0:
        return 0.0
    l_max = float(np.abs(l_grid).max())
    m_max = float(np.abs(m_grid).max())
    n_edge = np.sqrt(max(0.0, 1.0 - l_max**2 - m_max**2))
    return float(2.0 * np.pi * np.abs(w).max() * abs(n_edge - 1.0))


def dirty_image(
    vis: Visibilities,
    l_grid: np.ndarray | None = None,
    m_grid: np.ndarray | None = None,
    *,
    field_of_view_rad: float = 0.04,
    n_pix: int = 64,
    channels: slice | np.ndarray | None = None,
    include_autos: bool = False,
    warn_on_w_term: bool = True,
    pol: str | int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Naturally-weighted dirty image by direct DFT.

    Parameters
    ----------
    vis : Visibilities
        Fringe-stopped visibilities from `rfi_simulator.correlator.correlate`.
    l_grid : numpy.ndarray, optional
        1-D array of ``l`` direction cosines to evaluate. If omitted, built
        from `field_of_view_rad` and `n_pix`.
    m_grid : numpy.ndarray, optional
        1-D array of ``m`` direction cosines. Defaults to `l_grid`.
    field_of_view_rad : float, optional
        Full width of the default grid, default 0.04 (~2.3 deg).
    n_pix : int, optional
        Pixels per side of the default grid, default 64.
    channels : slice or numpy.ndarray, optional
        Channel selection, e.g. ``slice(None, None, 16)`` to image every
        16th channel. Default: all channels. Subsampling channels is the
        cheap way to keep this DFT fast; it costs sensitivity and raises
        sidelobes but does not move the source.
    include_autos : bool, optional
        If True, include autocorrelations. Default False -- autos carry no
        fringe and only add a flat offset equal to the total system power.
    warn_on_w_term : bool, optional
        If True (default) emit a `UserWarning` when the neglected ``w``
        term exceeds `W_TERM_WARN_PHASE_RAD` at the map edge, which is the
        usual symptom of imaging far from the zenith with a flat array.
    pol : str or int, optional
        Which polarization product to image. Default ``None``: Stokes I,
        i.e. `rfi_simulator.correlator.Visibilities.stokes_i` -- for
        single-polarization data that is the data itself, and for
        dual-polarization data it is ``(XX + YY) / 2``, which puts the two
        on the same flux scale (see `rfi_simulator.voltages`). Pass a name
        from ``vis.pol_names`` (e.g. ``"XX"``) or an integer index to image
        one receptor instead, which is what a polarization diagnostic
        wants.

    Returns
    -------
    image : numpy.ndarray
        Shape ``(len(m_grid), len(l_grid))`` float64 dirty image in Jy.
        Row index is ``m``, column index is ``l``.
    l_grid : numpy.ndarray
        The ``l`` axis actually used.
    m_grid : numpy.ndarray
        The ``m`` axis actually used.

    Raises
    ------
    ValueError
        If the baseline/channel selection is empty.
    KeyError
        If `pol` names a product this dataset does not carry.
    """
    if l_grid is None:
        l_grid = lm_axis(field_of_view_rad, n_pix)
    l_grid = np.asarray(l_grid, dtype=np.float64).ravel()
    if m_grid is None:
        m_grid = l_grid
    m_grid = np.asarray(m_grid, dtype=np.float64).ravel()

    baseline_sel = np.ones(vis.n_baselines, dtype=bool) if include_autos else vis.cross_mask
    if not np.any(baseline_sel):
        raise ValueError("no baselines selected for imaging")

    chan_sel = slice(None) if channels is None else channels

    selected = vis.stokes_i() if pol is None else vis.pol_data[:, :, vis.pol_index(pol), :]

    u, v, w = uvw_wavelengths(vis)
    u = u[:, baseline_sel, :][:, :, chan_sel].ravel()
    v = v[:, baseline_sel, :][:, :, chan_sel].ravel()
    w = w[:, baseline_sel, :][:, :, chan_sel].ravel()
    data = selected[:, baseline_sel, :][:, :, chan_sel].ravel()

    n_terms = data.size
    if n_terms == 0:
        raise ValueError("no visibility samples selected for imaging")

    if warn_on_w_term:
        phase_rad = w_term_phase_rad(w, l_grid, m_grid)
        if phase_rad > W_TERM_WARN_PHASE_RAD:
            warnings.warn(
                f"neglected w term reaches {phase_rad:.2f} rad at the edge of this "
                f"map (threshold {W_TERM_WARN_PHASE_RAD} rad): this two-dimensional "
                "transform will smear and shift sources. Phase up closer to the "
                "zenith, shrink the field of view, or use a w-aware imager.",
                UserWarning,
                stacklevel=2,
            )

    n_l = l_grid.size
    n_m = m_grid.size

    # image_lm[i_l, i_m] = sum_k exp(-2 pi i u_k l_i) V_k exp(-2 pi i v_k m_j)
    # evaluated as one matrix product per chunk of k, which keeps the
    # expensive np.exp calls at O(K * (n_l + n_m)) instead of O(K * n_l * n_m).
    image_lm = np.zeros((n_l, n_m), dtype=np.complex128)
    chunk = max(1, _MAX_PHASE_ELEMENTS // max(n_l, n_m))
    for start in range(0, n_terms, chunk):
        stop = min(start + chunk, n_terms)
        phase_l = np.exp(-2j * np.pi * np.outer(u[start:stop], l_grid))
        phase_m = np.exp(-2j * np.pi * np.outer(v[start:stop], m_grid))
        phase_m *= data[start:stop, np.newaxis]
        image_lm += phase_l.T @ phase_m

    image = np.real(image_lm.T) / n_terms
    return image, l_grid, m_grid
