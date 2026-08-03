"""File I/O for on-disk voltage formats.

Currently one format is supported: a packed 4-bit complex voltage block
format used by FX-correlator packet streams (see `rfi_simulator.io.packed_voltage`).
"""

from rfi_simulator.io.packed_voltage import (
    DEFAULT_QUANT_SCALE,
    PackedVoltageLayout,
    pack_block,
    pack_from_voltage_block,
    quantize_roundtrip,
    read_packed_file,
    suggest_quant_scale,
    unpack_block,
)

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
