r"""Packed 4-bit complex voltage block format.

FX-correlator packet streams commonly move channelized antenna voltages as
a headerless stream of fixed-size blocks, one byte per complex sample: a
signed 4-bit two's-complement real part in one nibble and a signed 4-bit
two's-complement imaginary part in the other, dequantized by multiplying
by a fixed scale factor. This module implements that on-disk convention
as a small, dependency-free (beyond numpy) reader/writer, independent of
any particular correlator's block dimensions -- every axis length is a
parameter of `PackedVoltageLayout`.

Nibble convention
------------------
Given a raw byte interpreted as a signed 8-bit two's-complement value
``b``::

    real = sign_extend_4bit(b & 0x0F)         # low (least-significant) nibble
    imag = sign_extend_4bit((b >> 4) & 0x0F)  # high (most-significant) nibble

i.e. **the real part lives in the low nibble and the imaginary part in the
high nibble**, each in the signed range ``[-8, 7]``. This is the
arithmetic-shift trick ``real = (b << 4) >> 4`` / ``imag = b >> 4`` used by
FX-correlator "fluff" kernels, transcribed here in pure numpy via a
256-entry lookup table instead of per-byte shifts.

Both nibbles are dequantized with a single scalar ``scale`` (typically of
order 0.01-0.1, matching the small dynamic range of a 4-bit sample):

.. math::

    v = \mathrm{scale} \times (\mathrm{real} + i\,\mathrm{imag})

Block layout
------------
One block is a dense tensor of shape ``(n_packets, n_antennas, n_channels,
n_times_per_packet, n_pols)``, C-contiguous, one byte per element. The
packet axis and the intra-packet time axis both index time; this module
merges them into a single contiguous time axis of length
``n_packets * n_times_per_packet`` on unpack (and splits it back out on
pack), with the packet index as the *outer* (slower-varying) component and
the intra-packet index as the *inner* (faster-varying) component:
``t = packet_index * n_times_per_packet + intra_packet_index``.

Files are headerless concatenations of blocks with no padding, so a valid
file's size is an exact multiple of the block's byte count (one byte per
sample).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "DEFAULT_QUANT_SCALE",
    "PackedVoltageLayout",
    "pack_block",
    "pack_from_voltage_block",
    "quantize_roundtrip",
    "read_packed_file",
    "suggest_quant_scale",
    "unpack_block",
]

#: Signed 4-bit two's-complement range, inclusive.
_NIBBLE_MIN = -8
_NIBBLE_MAX = 7

#: A reasonable default dequantization scale for demonstration / test use.
#: There is nothing physically special about this value -- callers reading
#: a real packet stream must supply the scale that stream was quantized
#: with; callers writing one out should generally use
#: `suggest_quant_scale` instead of this constant.
DEFAULT_QUANT_SCALE = 0.05


def _build_nibble_lut() -> np.ndarray:
    """Build the 256-entry byte -> (real, imag) int8 lookup table.

    Returns
    -------
    numpy.ndarray
        int8 array of shape ``(256, 2)``; ``lut[byte, 0]`` is the signed
        real nibble (low nibble), ``lut[byte, 1]`` is the signed imaginary
        nibble (high nibble).
    """
    bytes_ = np.arange(256, dtype=np.uint8)
    low = (bytes_ & 0x0F).astype(np.int16)
    high = ((bytes_ >> 4) & 0x0F).astype(np.int16)
    # Sign-extend a 4-bit nibble: values >= 8 represent negative numbers.
    real = np.where(low >= 8, low - 16, low).astype(np.int8)
    imag = np.where(high >= 8, high - 16, high).astype(np.int8)
    return np.stack([real, imag], axis=-1)


#: Module-level LUT, built once. `unpack_block` indexes into this with the
#: raw byte array to vectorize the nibble split -- no Python-level bit
#: twiddling per sample.
_NIBBLE_LUT = _build_nibble_lut()


@dataclass(frozen=True)
class PackedVoltageLayout:
    """Dimensions of one packed 4-bit complex voltage block.

    Attributes
    ----------
    n_packets : int
        Number of packets per block (outer time axis).
    n_antennas : int
        Number of antennas.
    n_channels : int
        Number of frequency channels.
    n_times_per_packet : int
        Number of time samples carried by a single packet (inner time
        axis).
    n_pols : int
        Number of polarizations.

    Notes
    -----
    The on-disk byte order for one block is
    ``(n_packets, n_antennas, n_channels, n_times_per_packet, n_pols)``,
    C-contiguous, one byte per complex sample.
    """

    n_packets: int
    n_antennas: int
    n_channels: int
    n_times_per_packet: int
    n_pols: int

    def __post_init__(self) -> None:
        for name in (
            "n_packets",
            "n_antennas",
            "n_channels",
            "n_times_per_packet",
            "n_pols",
        ):
            value = getattr(self, name)
            if int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

    @property
    def n_time(self) -> int:
        """int: Merged time-axis length, ``n_packets * n_times_per_packet``."""
        return self.n_packets * self.n_times_per_packet

    @property
    def raw_shape(self) -> tuple[int, int, int, int, int]:
        """tuple of int: On-disk block shape (packet axis and intra-packet
        time axis kept separate), as bytes/samples."""
        return (
            self.n_packets,
            self.n_antennas,
            self.n_channels,
            self.n_times_per_packet,
            self.n_pols,
        )

    @property
    def unpacked_shape(self) -> tuple[int, int, int, int]:
        """tuple of int: Shape of an unpacked complex block,
        ``(n_antennas, n_channels, n_time, n_pols)``."""
        return (self.n_antennas, self.n_channels, self.n_time, self.n_pols)

    @property
    def samples_per_block(self) -> int:
        """int: Total complex samples per block (== bytes per block, one
        byte per sample)."""
        return (
            self.n_packets
            * self.n_antennas
            * self.n_channels
            * self.n_times_per_packet
            * self.n_pols
        )

    @property
    def bytes_per_block(self) -> int:
        """int: Bytes per block. One byte per complex sample, so this
        equals `samples_per_block`."""
        return self.samples_per_block


def _require_positive_finite_scale(scale: float, *, where: str) -> None:
    """Validate that `scale` is a positive, finite number.

    Parameters
    ----------
    scale : float
        Value to validate.
    where : str
        Name of the calling function, used only to make the error message
        identify which entry point rejected the value.

    Raises
    ------
    ValueError
        If `scale` is not a finite number strictly greater than zero. This
        also rejects ``float('nan')`` and ``float('inf')``/``float('-inf')``
        explicitly: a naive ``scale <= 0.0`` check does *not* catch
        ``nan`` (every comparison with ``nan`` is ``False``, so
        ``nan <= 0.0`` is ``False`` and would silently slip through), so
        this helper checks `numpy.isfinite` in addition to positivity.
    """
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{where}: scale must be a positive finite number, got {scale!r}")


def _reject_nan(voltages: np.ndarray, *, nan_policy: str, where: str) -> None:
    """Raise if `voltages` has any NaN real or imaginary component.

    Parameters
    ----------
    voltages : numpy.ndarray
        Complex voltages about to be quantized.
    nan_policy : str
        The caller's ``nan_policy`` value, quoted in the error message.
    where : str
        Name of the calling function, quoted in the error message.

    Raises
    ------
    ValueError
        If `nan_policy` is not ``"raise"`` (the only currently implemented
        policy), or if any component is NaN. See `pack_block`'s Notes for
        why NaN is rejected outright while ``+-Inf`` is allowed to
        saturate.
    """
    if nan_policy != "raise":
        raise ValueError(
            f"{where}: nan_policy must be 'raise' (the only implemented policy), got {nan_policy!r}"
        )
    real_nan = np.isnan(voltages.real)
    imag_nan = np.isnan(voltages.imag)
    if np.any(real_nan) or np.any(imag_nan):
        parts = []
        if np.any(real_nan):
            parts.append(f"real ({int(np.count_nonzero(real_nan))} element(s))")
        if np.any(imag_nan):
            parts.append(f"imag ({int(np.count_nonzero(imag_nan))} element(s))")
        raise ValueError(
            "voltages contains NaN component(s) in "
            + " and ".join(parts)
            + f"; {where}(nan_policy={nan_policy!r}) refuses to silently quantize NaN "
            "to an unspecified nibble value (+-Inf is fine and saturates correctly -- only "
            "NaN is rejected)"
        )


def _quantize_components(
    voltages: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Round and saturate complex voltages to signed 4-bit counts.

    This is the single implementation of the quantization rule -- round
    half-to-even, then saturate to ``[-8, 7]`` -- shared by `pack_block`
    and `quantize_roundtrip`, so the on-disk format and the in-simulation
    quantization stage can never drift apart.

    Parameters
    ----------
    voltages : numpy.ndarray
        Complex voltages, any shape.
    scale : float
        Quantization scale: one count corresponds to `scale` in the units
        of `voltages`.

    Returns
    -------
    real_q, imag_q : numpy.ndarray
        int8 arrays shaped like `voltages`, in ``[-8, 7]``.
    clipped : numpy.ndarray
        Boolean array shaped like `voltages`, True where *either*
        component's rounded value fell outside the representable range and
        was therefore saturated -- i.e. where the sample railed.
    """
    real_r = np.round(voltages.real / scale)
    imag_r = np.round(voltages.imag / scale)
    clipped = (
        (real_r < _NIBBLE_MIN)
        | (real_r > _NIBBLE_MAX)
        | (imag_r < _NIBBLE_MIN)
        | (imag_r > _NIBBLE_MAX)
    )
    real_q = np.clip(real_r, _NIBBLE_MIN, _NIBBLE_MAX).astype(np.int8)
    imag_q = np.clip(imag_r, _NIBBLE_MIN, _NIBBLE_MAX).astype(np.int8)
    return real_q, imag_q, clipped


def quantize_roundtrip(
    voltages: np.ndarray, scale: float, *, nan_policy: str = "raise"
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize to 4 bits and dequantize again, in memory.

    This is the pack/unpack round trip without the bytes: it applies
    exactly the quantization rule of `pack_block` (`_quantize_components`)
    and then the dequantization of `unpack_block`, but skips the nibble
    packing and the block-layout reshaping, so it works on an array of any
    shape. Use it to put a signal through the dynamic range of a 4-bit
    correlator front end -- the quantization noise and the saturation
    behaviour are identical to what a packed file would give.

    Parameters
    ----------
    voltages : numpy.ndarray
        Complex voltages, any shape.
    scale : float
        Quantization scale, in the units of `voltages` per count. See
        `suggest_quant_scale` for choosing one from a target loading.
    nan_policy : {"raise"}, optional
        As `pack_block`: NaN components are rejected.

    Returns
    -------
    dequantized : numpy.ndarray
        complex64 array shaped like `voltages`, holding
        ``scale * (round(re / scale) + 1j * round(im / scale))`` with both
        components saturated to ``[-8, 7]``.
    clipped : numpy.ndarray
        Boolean array shaped like `voltages`, True where the sample railed
        (see `_quantize_components`). ``clipped.mean()`` over an antenna's
        samples is that antenna's clipped-sample fraction.

    Raises
    ------
    ValueError
        If `scale` is not a positive finite number, if `nan_policy` is not
        ``"raise"``, or if `voltages` contains a NaN component.
    """
    _require_positive_finite_scale(scale, where="quantize_roundtrip")
    _reject_nan(voltages, nan_policy=nan_policy, where="quantize_roundtrip")
    real_q, imag_q, clipped = _quantize_components(voltages, scale)
    dequantized = (real_q.astype(np.float32) + 1j * imag_q.astype(np.float32)) * np.float32(scale)
    return dequantized.astype(np.complex64), clipped


def unpack_block(
    raw_bytes: bytes | bytearray | memoryview | np.ndarray,
    layout: PackedVoltageLayout,
    scale: float = DEFAULT_QUANT_SCALE,
) -> np.ndarray:
    """Dequantize one packed block into complex voltages.

    Parameters
    ----------
    raw_bytes : bytes-like or numpy.ndarray
        Exactly `PackedVoltageLayout.bytes_per_block` bytes (uint8), in
        `PackedVoltageLayout.raw_shape` C order.
    layout : PackedVoltageLayout
        Block dimensions.
    scale : float, optional
        Dequantization scale applied to both nibbles. Default
        `DEFAULT_QUANT_SCALE`.

    Returns
    -------
    numpy.ndarray
        complex64 array of shape `PackedVoltageLayout.unpacked_shape` =
        ``(n_antennas, n_channels, n_time, n_pols)``, with
        ``n_time = n_packets * n_times_per_packet`` and the packet index
        as the outer (slower) component of that merged axis.

    Raises
    ------
    ValueError
        If `raw_bytes` is not exactly `PackedVoltageLayout.bytes_per_block`
        bytes, or if `scale` is not a positive finite number.
    """
    _require_positive_finite_scale(scale, where="unpack_block")
    arr = (
        np.frombuffer(bytes(raw_bytes), dtype=np.uint8)
        if isinstance(raw_bytes, (bytes, bytearray, memoryview))
        else np.asarray(raw_bytes, dtype=np.uint8).reshape(-1)
    )

    expected = layout.bytes_per_block
    if arr.size != expected:
        raise ValueError(f"raw block size {arr.size} != expected {expected} bytes")

    raw = arr.reshape(layout.raw_shape)  # (pkt, ant, chan, t_sub, pol)
    nibbles = _NIBBLE_LUT[raw]  # (pkt, ant, chan, t_sub, pol, 2) int8
    real = nibbles[..., 0]
    imag = nibbles[..., 1]

    # Merge (pkt, t_sub) into one time axis, packet outer / t_sub inner:
    # transpose to (ant, chan, pkt, t_sub, pol) then collapse (pkt, t_sub).
    real = np.transpose(real, (1, 2, 0, 3, 4))
    imag = np.transpose(imag, (1, 2, 0, 3, 4))
    new_shape = (layout.n_antennas, layout.n_channels, layout.n_time, layout.n_pols)
    real = real.reshape(new_shape)
    imag = imag.reshape(new_shape)

    out = (real.astype(np.float32) + 1j * imag.astype(np.float32)) * np.float32(scale)
    return out.astype(np.complex64)


def pack_block(
    voltages: np.ndarray,
    layout: PackedVoltageLayout,
    scale: float = DEFAULT_QUANT_SCALE,
    *,
    nan_policy: str = "raise",
) -> bytes:
    """Quantize complex voltages into one packed 4-bit block.

    Parameters
    ----------
    voltages : numpy.ndarray
        Complex array of shape `PackedVoltageLayout.unpacked_shape` =
        ``(n_antennas, n_channels, n_time, n_pols)``, with the merged time
        axis ordered packet-outer / intra-packet-inner (the same
        convention `unpack_block` produces).
    layout : PackedVoltageLayout
        Block dimensions.
    scale : float, optional
        Quantization scale: each nibble encodes ``round(component /
        scale)``. Default `DEFAULT_QUANT_SCALE`.
    nan_policy : {"raise"}, optional
        What to do about NaN components in `voltages`. Currently only
        ``"raise"`` (the default) is implemented: any NaN real or
        imaginary component raises `ValueError` before quantization. The
        keyword exists so a future policy (e.g. replacing NaNs with zero)
        can be added without changing the call signature; there is
        currently no way to opt out of the NaN check.

    Returns
    -------
    bytes
        `PackedVoltageLayout.bytes_per_block` bytes in
        `PackedVoltageLayout.raw_shape` C order.

    Raises
    ------
    ValueError
        If `voltages` is not shaped `PackedVoltageLayout.unpacked_shape`,
        if `scale` is not a positive finite number, if `nan_policy` is not
        ``"raise"``, or if `voltages` contains any NaN component.

    Notes
    -----
    Values are rounded half-to-even (`numpy.round`'s default) and then
    **saturated** (clipped) to the symmetric signed range ``[-8, 7]``
    rather than wrapped -- an out-of-range input clamps to the nearest
    representable nibble instead of aliasing to a wildly different value.
    This saturation applies to ``+-inf`` too: an infinite component divides
    by `scale` and rounds to ``+-inf``, which `numpy.clip` saturates to the
    nibble rails exactly like any other too-large finite value, so
    ``+-inf`` is handled correctly by construction and is *not* rejected
    by `nan_policy` -- only NaN is.

    NaN is a different case from Inf and is always rejected regardless of
    `nan_policy`'s value: `numpy.round` propagates NaN unchanged, and
    `numpy.clip` also propagates NaN unchanged (a NaN compares False
    against both clip bounds), so a NaN component reaches
    ``.astype(np.int8)`` as a float NaN. The result of casting NaN to an
    integer dtype is *undefined by the C standard* and numpy inherits that
    -- in practice it is usually silently ``0``, but the point is it is
    unspecified. Left unchecked, a NaN anywhere upstream (e.g. a divide by
    zero earlier in a physics pipeline) would silently masquerade as a
    valid, plausible-looking zero sample in the packed output with no
    error and, at best, a `RuntimeWarning`. `pack_block` therefore checks
    for NaN explicitly and raises before that cast can happen.
    """
    _require_positive_finite_scale(scale, where="pack_block")
    expected_shape = layout.unpacked_shape
    if voltages.shape != expected_shape:
        raise ValueError(f"voltages shape {voltages.shape} != expected {expected_shape}")

    _reject_nan(voltages, nan_policy=nan_policy, where="pack_block")
    real_q, imag_q, _clipped = _quantize_components(voltages, scale)

    # Split merged time axis back into (n_packets, n_times_per_packet),
    # packet outer / intra-packet inner, then move to raw_shape's axis
    # order (pkt, ant, chan, t_sub, pol).
    n_ant, n_chan, _n_time, n_pol = expected_shape
    real_q = real_q.reshape(n_ant, n_chan, layout.n_packets, layout.n_times_per_packet, n_pol)
    imag_q = imag_q.reshape(n_ant, n_chan, layout.n_packets, layout.n_times_per_packet, n_pol)
    real_q = np.transpose(real_q, (2, 0, 1, 3, 4))
    imag_q = np.transpose(imag_q, (2, 0, 1, 3, 4))

    real_u8 = real_q.view(np.uint8) & 0x0F
    imag_u8 = imag_q.view(np.uint8) & 0x0F
    byte = (imag_u8 << 4) | real_u8  # imag = high nibble, real = low nibble
    return np.ascontiguousarray(byte, dtype=np.uint8).tobytes()


def read_packed_file(
    path: str | Path,
    layout: PackedVoltageLayout,
    scale: float = DEFAULT_QUANT_SCALE,
    block_indices: Iterable[int] | None = None,
) -> Iterator[np.ndarray]:
    """Iterate over the blocks of a headerless packed voltage file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the file. Its size must be an exact multiple of
        `PackedVoltageLayout.bytes_per_block`.
    layout : PackedVoltageLayout
        Block dimensions.
    scale : float, optional
        Dequantization scale, passed through to `unpack_block`.
    block_indices : iterable of int, optional
        Zero-based block indices to yield, in the given order. Defaults
        to every block in the file, in order.

    Yields
    ------
    numpy.ndarray
        One complex64 array per requested block, shaped
        `PackedVoltageLayout.unpacked_shape` (see `unpack_block`).

    Raises
    ------
    ValueError
        If the file size is not an exact multiple of
        `PackedVoltageLayout.bytes_per_block`, if a requested block index
        is out of range, or if `scale` is not a positive finite number.

    Notes
    -----
    Uses `numpy.memmap` so only the requested blocks are ever paged into
    memory -- suitable for multi-gigabyte files that must not be loaded
    whole. `scale` is validated up front (before the memmap is opened) so
    a bad scale fails fast with a clear error instead of only surfacing
    once `unpack_block` runs on the first requested block. A 0-byte file
    is a valid (empty) file under this format -- it has zero blocks, and
    this function yields nothing for it, rather than raising.
    """
    _require_positive_finite_scale(scale, where="read_packed_file")
    path = Path(path)
    bytes_per_block = layout.bytes_per_block
    file_size = path.stat().st_size
    if file_size % bytes_per_block != 0:
        raise ValueError(
            f"file size {file_size} bytes is not an exact multiple of "
            f"bytes_per_block ({bytes_per_block}); path={path}"
        )
    n_blocks = file_size // bytes_per_block

    if block_indices is None:
        indices: Iterable[int] = range(n_blocks)
    else:
        indices = list(block_indices)
        for idx in indices:
            if not 0 <= idx < n_blocks:
                raise ValueError(f"block index {idx} out of range [0, {n_blocks})")

    if n_blocks == 0:
        # A 0-byte file has zero blocks by construction (0 is a multiple of
        # bytes_per_block); `numpy.memmap` refuses to map an empty file at
        # all, so that generic case must be short-circuited here rather
        # than surfacing memmap's unrelated "cannot mmap an empty file"
        # error. Empty iteration is the correct, documented behavior.
        return
    mmap = np.memmap(path, dtype=np.uint8, mode="r", shape=(n_blocks, bytes_per_block))
    for idx in indices:
        yield unpack_block(mmap[idx], layout, scale)


def pack_from_voltage_block(
    data: np.ndarray,
    layout: PackedVoltageLayout,
    scale: float = DEFAULT_QUANT_SCALE,
    *,
    pol_mode: str = "duplicate",
) -> bytes:
    """Pack a simulator `VoltageBlock`-style array into one raw block.

    Parameters
    ----------
    data : numpy.ndarray
        Complex voltages, either shaped ``(n_antennas, n_channels,
        n_time)`` (single-pol, as produced by
        `rfi_simulator.voltages.VoltageBlock.data`) or ``(n_antennas,
        n_channels, n_time, n_pols)`` (already carrying a polarization
        axis).
    layout : PackedVoltageLayout
        Target block dimensions. ``n_time = layout.n_packets *
        layout.n_times_per_packet`` must equal `data`'s time axis length.
    scale : float, optional
        Quantization scale, passed through to `pack_block`.
    pol_mode : {"duplicate", "stack"}, optional
        How to handle the polarization axis:

        ``"duplicate"``
            `data` is 3-D, single-pol; the same voltage is copied into
            every polarization of the packed block (only sensible for
            `layout.n_pols == 1`, or as a placeholder for an
            unpolarized/Stokes-I-only simulation run through a dual-pol
            layout).
        ``"stack"``
            `data` is already 4-D with a trailing polarization axis of
            length `layout.n_pols` and is used as-is.

    Returns
    -------
    bytes
        `PackedVoltageLayout.bytes_per_block` bytes, ready to append to a
        packed voltage file.

    Raises
    ------
    ValueError
        If `pol_mode` is not recognised, if `data`'s shape does not match
        `layout` under the requested `pol_mode`, or if `data`'s time axis
        does not equal `layout.n_time`.
    """
    if pol_mode == "duplicate":
        if data.ndim != 3:
            raise ValueError(
                f"pol_mode='duplicate' expects a 3-D (n_ant, n_chan, n_time) array, "
                f"got shape {data.shape}"
            )
        n_ant, n_chan, n_time = data.shape
        if n_time != layout.n_time:
            raise ValueError(f"data n_time {n_time} != layout.n_time {layout.n_time}")
        voltages = np.repeat(data[..., np.newaxis], layout.n_pols, axis=-1)
    elif pol_mode == "stack":
        if data.ndim != 4:
            raise ValueError(
                f"pol_mode='stack' expects a 4-D (n_ant, n_chan, n_time, n_pols) array, "
                f"got shape {data.shape}"
            )
        voltages = data
    else:
        raise ValueError(f"pol_mode must be 'duplicate' or 'stack', got {pol_mode!r}")

    if voltages.shape != layout.unpacked_shape:
        raise ValueError(
            f"voltage shape {voltages.shape} != layout.unpacked_shape "
            f"{layout.unpacked_shape} (check n_antennas/n_channels/n_time/n_pols)"
        )
    return pack_block(voltages.astype(np.complex64, copy=False), layout, scale)


#: Variance (in quantization-count units, i.e. LSBs^2) contributed by
#: uniform quantization noise for a step size of 1 count: Delta^2 / 12 with
#: Delta = 1. Used by `suggest_quant_scale` to correct for the fact that
#: quantizing a signal *adds* this much variance to each real/imaginary
#: component, on top of the signal's own variance.
_QUANT_NOISE_VARIANCE_COUNTS2 = 1.0 / 12.0


def suggest_quant_scale(
    voltages: np.ndarray,
    target_counts: float = 2.5,
    *,
    correct_for_quantization_noise: bool = True,
) -> float:
    """Pick a quantization scale so the input's RMS lands near `target_counts`.

    Parameters
    ----------
    voltages : numpy.ndarray
        Complex voltages to be packed (any shape).
    target_counts : float, optional
        Desired RMS magnitude of each real/imaginary quantization count
        *after* quantization (i.e. the RMS a caller would measure on the
        packed-and-unpacked, dequantized samples). Default 2.5, matching
        the light loading (a few LSBs of RMS, well clear of both the 0
        floor and the +-7 saturation ceiling) typical of correlator 4-bit
        voltage quantization -- it leaves headroom for the occasional
        large sample (RFI, gain drift) without saturating every
        high-power event, while still using enough of the dynamic range
        that quantization noise stays a small fraction of the signal.
    correct_for_quantization_noise : bool, optional
        If True (the default), solve for the scale that makes the
        *post*-quantization RMS equal `target_counts`, correcting for the
        variance that quantization itself adds (see Notes). If False, use
        the older, simpler formula that targets `target_counts` against
        the *input* (pre-quantization) RMS directly -- because it ignores
        the extra 1/12-count^2 variance quantization itself adds, the
        actual (realized) post-quantization RMS then *overshoots*
        `target_counts` by a factor ``sqrt(1 + 1/(12 * target_counts**2))``:
        about 0.7% high at the default ``target_counts=2.5``, growing to
        about 2.4% high at ``target_counts=1.315`` and larger still at
        smaller targets, since the fixed 1/12-count^2 quantization-noise
        contribution is a bigger fraction of a smaller target.

    Returns
    -------
    float
        A scale such that, after quantizing with it, ``rms(quantized
        real)`` and ``rms(quantized imag)`` are approximately
        `target_counts` (or, with `correct_for_quantization_noise=False`,
        such that the *input*'s RMS divided by the scale is approximately
        `target_counts`). Returns `DEFAULT_QUANT_SCALE` if `voltages` is
        all-zero (RMS 0), since any positive scale is equally valid there.

    Raises
    ------
    ValueError
        If `target_counts` is not finite and positive, if `voltages` is
        empty, if `voltages` contains a NaN component, or (when
        `correct_for_quantization_noise` is True) if `target_counts` is at
        or below the quantization noise floor
        (``sqrt(1/12) ~= 0.2887`` counts) -- below that floor, no scale
        can make the post-quantization RMS that small, since quantization
        alone already contributes at least that much variance.

    Notes
    -----
    Quantizing a signal to the nearest integer count adds uniform
    quantization noise with variance ``1/12`` count^2 per component (a
    standard result for a quantization step of 1 count). So the *total*
    post-quantization variance of one component is approximately
    ``signal_variance_in_counts + 1/12``, not just
    ``signal_variance_in_counts``. Solving for the scale that makes the
    post-quantization RMS equal `target_counts` (rather than the simpler,
    but biased, calculation that ignores this extra term) means dividing
    the input RMS not by `target_counts` directly but by
    ``sqrt(target_counts**2 - 1/12)``:

    .. math::

        \\mathrm{scale} = \\frac{\\mathrm{rms\\_in}}
        {\\sqrt{\\mathrm{target\\_counts}^2 - 1/12}}

    Ignoring the ``-1/12`` term (the `correct_for_quantization_noise=False`
    path, and this function's behavior prior to this correction) makes the
    realized post-quantization RMS come in *higher* than `target_counts`,
    by a factor ``sqrt(1 + 1/(12 * target_counts**2))`` -- about 0.7% at
    the default ``target_counts=2.5``, growing to roughly 2.4% at
    ``target_counts=1.315`` (the bias is larger at smaller `target_counts`,
    where the fixed 1/12-count^2 term is a bigger fraction of the target).
    """
    if not np.isfinite(target_counts) or target_counts <= 0.0:
        raise ValueError(f"target_counts must be finite and > 0, got {target_counts}")
    voltages = np.asarray(voltages)
    if voltages.size == 0:
        raise ValueError("voltages must not be empty")
    _reject_nan(voltages, nan_policy="raise", where="suggest_quant_scale")
    # Per-component (real and imaginary treated together) RMS.
    component_rms = np.sqrt(
        0.5 * np.mean(voltages.real.astype(np.float64) ** 2 + voltages.imag.astype(np.float64) ** 2)
    )
    if component_rms == 0.0:
        return DEFAULT_QUANT_SCALE

    if not correct_for_quantization_noise:
        return float(component_rms / target_counts)

    residual = target_counts**2 - _QUANT_NOISE_VARIANCE_COUNTS2
    if residual <= 0.0:
        raise ValueError(
            f"target_counts={target_counts!r} is at or below the quantization noise floor "
            f"(sqrt(1/12) ~= {np.sqrt(_QUANT_NOISE_VARIANCE_COUNTS2):.4f} counts); "
            "quantization alone already contributes this much RMS per component, so no "
            "scale can hit a smaller post-quantization target. Pick a larger target_counts "
            "or set correct_for_quantization_noise=False to use the uncorrected formula."
        )
    return float(component_rms / np.sqrt(residual))
