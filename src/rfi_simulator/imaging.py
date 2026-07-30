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
tangent-plane transform above is exact.
"""

from __future__ import annotations

import numpy as np

from rfi_simulator.correlator import Visibilities
from rfi_simulator.delays import SPEED_OF_LIGHT_M_S

__all__ = ["dirty_image", "lm_axis", "uvw_wavelengths"]

_MAX_PHASE_ELEMENTS = 1_000_000


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
        ``-field_of_view_rad / 2`` to ``+field_of_view_rad / 2``.
    """
    if n_pix < 1:
        raise ValueError(f"n_pix must be >= 1, got {n_pix}")
    return np.linspace(-0.5 * field_of_view_rad, 0.5 * field_of_view_rad, n_pix)


def uvw_wavelengths(vis: Visibilities) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Baseline coordinates in wavelengths for every (integration, baseline, channel).

    Parameters
    ----------
    vis : Visibilities
        Visibilities carrying baseline vectors, per-integration ``(l, m)``
        basis vectors and channel frequencies.

    Returns
    -------
    u, v : numpy.ndarray
        Shape ``(n_int, n_baselines, n_chan)`` float64 arrays of ``u`` and
        ``v`` in wavelengths.
    scale : numpy.ndarray
        Shape ``(n_chan,)`` array ``freq_hz / c``, in inverse meters --
        returned because callers often want it for ``w`` as well.
    """
    scale = vis.freq_hz / SPEED_OF_LIGHT_M_S  # (n_chan,) 1/m
    b_l = np.einsum("bj,tj->tb", vis.baseline_vectors_enu_m, vis.e_l_enu)
    b_m = np.einsum("bj,tj->tb", vis.baseline_vectors_enu_m, vis.e_m_enu)
    u = b_l[:, :, np.newaxis] * scale[np.newaxis, np.newaxis, :]
    v = b_m[:, :, np.newaxis] * scale[np.newaxis, np.newaxis, :]
    return u, v, scale


def dirty_image(
    vis: Visibilities,
    l_grid: np.ndarray | None = None,
    m_grid: np.ndarray | None = None,
    *,
    field_of_view_rad: float = 0.04,
    n_pix: int = 64,
    channels: slice | np.ndarray | None = None,
    include_autos: bool = False,
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

    u, v, _ = uvw_wavelengths(vis)
    u = u[:, baseline_sel, :][:, :, chan_sel].ravel()
    v = v[:, baseline_sel, :][:, :, chan_sel].ravel()
    data = vis.data[:, baseline_sel, :][:, :, chan_sel].ravel()

    n_terms = data.size
    if n_terms == 0:
        raise ValueError("no visibility samples selected for imaging")

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
