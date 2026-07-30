"""rfi_simulator: voltage-level radio-frequency-interference simulator.

Public API is re-exported from submodules here; each submodule's docstring
states the physics conventions it owns.

The pipeline is four steps long::

    array = ArrayConfig.from_yaml("configs/array_default.yaml")
    sim   = VoltageSimulator(array, phase_center, start_time, sources, rng=rng)
    vis   = correlate(sim.blocks())
    image, l_grid, m_grid = dirty_image(vis)
"""

from rfi_simulator.array_config import ArrayConfig
from rfi_simulator.correlator import Visibilities, baseline_index_pairs, correlate
from rfi_simulator.delays import (
    SPEED_OF_LIGHT_M_S,
    earth_location,
    geometric_delays_s,
    lm_basis_enu,
    source_unit_vectors_enu,
    zenith_coord,
)
from rfi_simulator.imaging import dirty_image, lm_axis, uvw_wavelengths
from rfi_simulator.rfi import (
    OCCUPANCY_THRESHOLD,
    BlockContext,
    ImpulsiveBroadband,
    NarrowbandTransmitter,
    RFISource,
    enu_from_geodetic,
    enu_from_horizontal,
    path_delays_s,
)
from rfi_simulator.sky import PointSource, lm_from_radec, radec_from_lm
from rfi_simulator.voltages import VoltageBlock, VoltageSimulator

__version__ = "0.1.0.dev0"

__all__ = [
    "OCCUPANCY_THRESHOLD",
    "SPEED_OF_LIGHT_M_S",
    "ArrayConfig",
    "BlockContext",
    "ImpulsiveBroadband",
    "NarrowbandTransmitter",
    "PointSource",
    "RFISource",
    "Visibilities",
    "VoltageBlock",
    "VoltageSimulator",
    "__version__",
    "baseline_index_pairs",
    "correlate",
    "dirty_image",
    "earth_location",
    "enu_from_geodetic",
    "enu_from_horizontal",
    "geometric_delays_s",
    "lm_axis",
    "lm_basis_enu",
    "lm_from_radec",
    "path_delays_s",
    "radec_from_lm",
    "source_unit_vectors_enu",
    "uvw_wavelengths",
    "zenith_coord",
]
