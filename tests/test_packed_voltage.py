"""Tests for rfi_simulator.io.packed_voltage.

Layouts here are deliberately tiny (a handful of packets/antennas/channels)
-- never the multi-hundred-megabyte block size a real correlator uses --
so the whole suite runs in a fraction of a second. The golden-vector tests
are hand-computed from the documented nibble convention (real = low
nibble, imag = high nibble, both signed 4-bit two's complement) rather
than derived from the implementation, so they would catch a swapped-nibble
or wrong-sign-extension bug even if the rest of the module shared that
bug.
"""

import numpy as np
import pytest

from rfi_simulator.io.packed_voltage import (
    PackedVoltageLayout,
    pack_block,
    pack_from_voltage_block,
    read_packed_file,
    suggest_quant_scale,
    unpack_block,
)

SMALL_LAYOUT = PackedVoltageLayout(
    n_packets=4, n_antennas=3, n_channels=8, n_times_per_packet=2, n_pols=2
)


def test_layout_derived_fields():
    layout = SMALL_LAYOUT
    assert layout.n_time == 8
    assert layout.raw_shape == (4, 3, 8, 2, 2)
    assert layout.unpacked_shape == (3, 8, 8, 2)
    assert layout.samples_per_block == 4 * 3 * 8 * 2 * 2
    assert layout.bytes_per_block == layout.samples_per_block


def test_layout_rejects_non_positive_dims():
    with pytest.raises(ValueError):
        PackedVoltageLayout(n_packets=0, n_antennas=3, n_channels=8, n_times_per_packet=2, n_pols=2)


# ---------------------------------------------------------------------
# Golden byte-level vectors.
#
# byte = (imag_nibble << 4) | (real_nibble & 0x0F), each nibble a signed
# 4-bit two's-complement value in [-8, 7] (16 wraps to -16.. range via
# the top bit). Hand-computed below, independent of the implementation.
# ---------------------------------------------------------------------
GOLDEN = [
    # (byte, expected_real, expected_imag)
    (0x00, 0, 0),  # 0b0000_0000
    (0x78, -8, 7),  # low nibble 0b1000 = -8, high nibble 0b0111 = 7
    (0x87, 7, -8),  # low nibble 0b0111 = 7,  high nibble 0b1000 = -8
    (0xFF, -1, -1),  # low nibble 0b1111 = -1, high nibble 0b1111 = -1
    (0xF1, 1, -1),  # low nibble 0b0001 = 1,  high nibble 0b1111 = -1
    (0x1F, -1, 1),  # low nibble 0b1111 = -1, high nibble 0b0001 = 1
]


@pytest.mark.parametrize("byte,real,imag", GOLDEN)
def test_golden_nibble_vectors(byte, real, imag):
    layout = PackedVoltageLayout(
        n_packets=1, n_antennas=1, n_channels=1, n_times_per_packet=1, n_pols=1
    )
    raw = bytes([byte])
    scale = 1.0  # no dequantization scaling, so we can compare integer nibbles exactly
    out = unpack_block(raw, layout, scale)
    assert out.shape == (1, 1, 1, 1)
    assert out[0, 0, 0, 0] == complex(real, imag)


def test_golden_vectors_batched_in_one_block():
    """The same golden bytes, packed into one multi-sample block."""
    n = len(GOLDEN)
    layout = PackedVoltageLayout(
        n_packets=n, n_antennas=1, n_channels=1, n_times_per_packet=1, n_pols=1
    )
    raw = bytes(byte for byte, _, _ in GOLDEN)
    out = unpack_block(raw, layout, scale=1.0)
    assert out.shape == (1, 1, n, 1)
    for i, (_byte, real, imag) in enumerate(GOLDEN):
        assert out[0, 0, i, 0] == complex(real, imag)


def test_unpack_applies_scale():
    layout = PackedVoltageLayout(
        n_packets=1, n_antennas=1, n_channels=1, n_times_per_packet=1, n_pols=1
    )
    out = unpack_block(bytes([0x78]), layout, scale=0.05)
    assert out[0, 0, 0, 0] == pytest.approx(complex(-8 * 0.05, 7 * 0.05), abs=1e-7)


# ---------------------------------------------------------------------
# Round trip, saturation, axis ordering.
# ---------------------------------------------------------------------


def test_round_trip_within_quantization_error(rng=np.random.default_rng(0)):
    layout = SMALL_LAYOUT
    scale = 0.05
    voltages = (
        rng.standard_normal(layout.unpacked_shape) + 1j * rng.standard_normal(layout.unpacked_shape)
    ).astype(np.complex64)
    # Keep amplitudes well inside the +-7 nibble range so round trip error
    # is pure quantization noise, not saturation.
    voltages *= 1.0 * scale

    raw = pack_block(voltages, layout, scale)
    assert len(raw) == layout.bytes_per_block
    recovered = unpack_block(raw, layout, scale)

    assert recovered.shape == voltages.shape
    # Quantization error is at most +-0.5 count per component.
    assert np.max(np.abs(recovered.real - voltages.real)) <= 0.5 * scale + 1e-6
    assert np.max(np.abs(recovered.imag - voltages.imag)) <= 0.5 * scale + 1e-6


def test_saturation_clips_symmetrically():
    layout = PackedVoltageLayout(
        n_packets=1, n_antennas=1, n_channels=1, n_times_per_packet=1, n_pols=1
    )
    scale = 1.0
    huge = np.array([[[[complex(1000.0, -1000.0)]]]], dtype=np.complex64)
    raw = pack_block(huge, layout, scale)
    out = unpack_block(raw, layout, scale)
    assert out[0, 0, 0, 0] == complex(7.0, -8.0)


def test_saturation_negative_extreme():
    layout = PackedVoltageLayout(
        n_packets=1, n_antennas=1, n_channels=1, n_times_per_packet=1, n_pols=1
    )
    scale = 1.0
    huge = np.array([[[[complex(-1000.0, -1000.0)]]]], dtype=np.complex64)
    raw = pack_block(huge, layout, scale)
    out = unpack_block(raw, layout, scale)
    assert out[0, 0, 0, 0] == complex(-8.0, -8.0)


def test_axis_ordering_packet_outer_intrapacket_inner():
    """Encode a distinctive per-time-sample pattern and check placement.

    Puts a different, easily-recognised real value at every (packet,
    intra-packet) pair for a single (antenna, channel, pol) and checks it
    lands at `t = packet_index * n_times_per_packet + intra_packet_index`
    after unpacking -- catching a transpose bug that would otherwise
    silently interleave time samples in the wrong order.
    """
    n_packets, n_times_per_packet = 3, 2
    layout = PackedVoltageLayout(
        n_packets=n_packets,
        n_antennas=1,
        n_channels=1,
        n_times_per_packet=n_times_per_packet,
        n_pols=1,
    )
    raw = np.zeros(layout.raw_shape, dtype=np.uint8)
    expected_real = {}
    value = -8
    for pkt in range(n_packets):
        for t_sub in range(n_times_per_packet):
            # Encode `value` (cycling through the nibble range) as the low
            # nibble; imag nibble left at 0.
            nibble = value & 0x0F
            raw[pkt, 0, 0, t_sub, 0] = nibble
            expected_real[(pkt, t_sub)] = value
            value = value + 1 if value < 7 else -8

    out = unpack_block(raw.tobytes(), layout, scale=1.0)
    assert out.shape == (1, 1, n_packets * n_times_per_packet, 1)
    for pkt in range(n_packets):
        for t_sub in range(n_times_per_packet):
            t = pkt * n_times_per_packet + t_sub
            assert out[0, 0, t, 0].real == expected_real[(pkt, t_sub)]


def test_unpack_rejects_wrong_byte_count():
    layout = SMALL_LAYOUT
    with pytest.raises(ValueError):
        unpack_block(bytes(layout.bytes_per_block - 1), layout, 0.05)


def test_pack_rejects_wrong_shape():
    layout = SMALL_LAYOUT
    with pytest.raises(ValueError):
        pack_block(np.zeros((1, 1, 1, 1), dtype=np.complex64), layout, 0.05)


def test_pack_rejects_non_positive_scale():
    layout = SMALL_LAYOUT
    voltages = np.zeros(layout.unpacked_shape, dtype=np.complex64)
    with pytest.raises(ValueError):
        pack_block(voltages, layout, 0.0)


# ---------------------------------------------------------------------
# File-level reader.
# ---------------------------------------------------------------------


def _write_blocks(path, layout, scale, n_blocks, rng):
    blocks = []
    with open(path, "wb") as f:
        for _ in range(n_blocks):
            voltages = (
                rng.standard_normal(layout.unpacked_shape)
                + 1j * rng.standard_normal(layout.unpacked_shape)
            ).astype(np.complex64) * (1.0 * scale)
            blocks.append(voltages)
            f.write(pack_block(voltages, layout, scale))
    return blocks


def test_read_packed_file_round_trip(tmp_path):
    layout = SMALL_LAYOUT
    scale = 0.05
    rng = np.random.default_rng(1)
    path = tmp_path / "voltages.dat"
    blocks = _write_blocks(path, layout, scale, n_blocks=3, rng=rng)

    read_back = list(read_packed_file(path, layout, scale))
    assert len(read_back) == 3
    for original, recovered in zip(blocks, read_back):
        assert recovered.shape == original.shape
        assert np.max(np.abs(recovered - original)) <= 0.5 * scale * np.sqrt(2) + 1e-6


def test_read_packed_file_block_indices(tmp_path):
    layout = SMALL_LAYOUT
    scale = 0.05
    rng = np.random.default_rng(2)
    path = tmp_path / "voltages.dat"
    blocks = _write_blocks(path, layout, scale, n_blocks=4, rng=rng)

    selected = list(read_packed_file(path, layout, scale, block_indices=[3, 0]))
    assert len(selected) == 2
    assert np.max(np.abs(selected[0] - blocks[3])) <= 0.5 * scale * np.sqrt(2) + 1e-6
    assert np.max(np.abs(selected[1] - blocks[0])) <= 0.5 * scale * np.sqrt(2) + 1e-6


def test_read_packed_file_bad_index(tmp_path):
    layout = SMALL_LAYOUT
    scale = 0.05
    rng = np.random.default_rng(3)
    path = tmp_path / "voltages.dat"
    _write_blocks(path, layout, scale, n_blocks=2, rng=rng)

    with pytest.raises(ValueError):
        list(read_packed_file(path, layout, scale, block_indices=[5]))


def test_read_packed_file_rejects_bad_file_size(tmp_path):
    layout = SMALL_LAYOUT
    path = tmp_path / "truncated.dat"
    path.write_bytes(bytes(layout.bytes_per_block - 1))
    with pytest.raises(ValueError):
        list(read_packed_file(path, layout, 0.05))


# ---------------------------------------------------------------------
# Converter helper.
# ---------------------------------------------------------------------


def test_pack_from_voltage_block_duplicate_pol():
    layout = SMALL_LAYOUT  # n_pols = 2
    n_ant, n_chan, n_time, n_pol = layout.unpacked_shape
    rng = np.random.default_rng(4)
    data = (rng.standard_normal((n_ant, n_chan, n_time)) * 0.1).astype(np.complex64)

    raw = pack_from_voltage_block(data, layout, scale=0.05, pol_mode="duplicate")
    out = unpack_block(raw, layout, scale=0.05)
    assert out.shape == (n_ant, n_chan, n_time, n_pol)
    # Both polarizations should be (quantized) copies of the same input.
    assert np.array_equal(out[..., 0], out[..., 1])


def test_pack_from_voltage_block_stack_pol():
    layout = SMALL_LAYOUT
    shape = layout.unpacked_shape
    rng = np.random.default_rng(5)
    data = (rng.standard_normal(shape) * 0.1).astype(np.complex64)

    raw = pack_from_voltage_block(data, layout, scale=0.05, pol_mode="stack")
    out = unpack_block(raw, layout, scale=0.05)
    assert out.shape == shape
    assert np.max(np.abs(out - data)) <= 0.5 * 0.05 * np.sqrt(2) + 1e-6


def test_pack_from_voltage_block_rejects_bad_pol_mode():
    layout = SMALL_LAYOUT
    data = np.zeros(layout.unpacked_shape[:3], dtype=np.complex64)
    with pytest.raises(ValueError):
        pack_from_voltage_block(data, layout, pol_mode="nonsense")


def test_pack_from_voltage_block_rejects_wrong_ndim_for_mode():
    layout = SMALL_LAYOUT
    data_4d = np.zeros(layout.unpacked_shape, dtype=np.complex64)
    with pytest.raises(ValueError):
        pack_from_voltage_block(data_4d, layout, pol_mode="duplicate")

    data_3d = np.zeros(layout.unpacked_shape[:3], dtype=np.complex64)
    with pytest.raises(ValueError):
        pack_from_voltage_block(data_3d, layout, pol_mode="stack")


# ---------------------------------------------------------------------
# Scale suggestion.
# ---------------------------------------------------------------------


def test_suggest_quant_scale_matches_target_counts():
    rng = np.random.default_rng(6)
    voltages = rng.standard_normal((1000,)) + 1j * rng.standard_normal((1000,))
    scale = suggest_quant_scale(voltages, target_counts=2.5)
    component_rms = np.sqrt(0.5 * np.mean(voltages.real**2 + voltages.imag**2))
    assert scale == pytest.approx(component_rms / 2.5, rel=1e-9)


def test_suggest_quant_scale_zero_input_falls_back():
    voltages = np.zeros(16, dtype=np.complex64)
    scale = suggest_quant_scale(voltages)
    assert scale > 0.0


def test_suggest_quant_scale_rejects_non_positive_target():
    with pytest.raises(ValueError):
        suggest_quant_scale(np.ones(4, dtype=np.complex64), target_counts=0.0)
