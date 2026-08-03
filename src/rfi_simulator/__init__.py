"""rfi_simulator: voltage-level radio-frequency-interference simulator.

Public API is re-exported from submodules here; each submodule's docstring
states the physics conventions it owns.

The pipeline is four steps long::

    array = ArrayConfig.from_yaml("configs/array_default.yaml")
    sim   = VoltageSimulator(array, phase_center, start_time, sources, rng=rng)
    vis   = correlate(sim.blocks())
    image, l_grid, m_grid = dirty_image(vis)
"""

from rfi_simulator.aircraft import ADSB_FREQ_HZ, ADSBTransponder
from rfi_simulator.array_config import ArrayConfig
from rfi_simulator.binning import bin_any, bin_mean, block_any
from rfi_simulator.channelizer import PFBChannelizer
from rfi_simulator.correlator import Visibilities, baseline_index_pairs, correlate
from rfi_simulator.delays import (
    SPEED_OF_LIGHT_M_S,
    earth_location,
    geometric_delays_s,
    lm_basis_enu,
    source_unit_vectors_enu,
    zenith_coord,
)
from rfi_simulator.flaggers import (
    mad_clip_mask,
    spectral_kurtosis_mask,
    sumthreshold_mask,
)
from rfi_simulator.imaging import dirty_image, lm_axis, uvw_wavelengths
from rfi_simulator.instrument import InstrumentModel
from rfi_simulator.metrics import (
    confusion_counts,
    flag_scores,
    pool_truth,
    pool_truth_accumulations,
)
from rfi_simulator.rfi import (
    OCCUPANCY_THRESHOLD,
    BlockContext,
    CombTransmitter,
    ImpulsiveBroadband,
    NarrowbandTransmitter,
    RFISource,
    constant_envelope,
    enu_from_geodetic,
    enu_from_horizontal,
    path_delays_s,
    resolve_coupling,
)
from rfi_simulator.satellites import (
    SatelliteTransmitter,
    TwoLineElement,
    fetch_tles,
    read_tle_file,
)
from rfi_simulator.sky import (
    PointSource,
    SpectralLineForeground,
    lm_from_radec,
    radec_from_lm,
)
from rfi_simulator.voltages import VoltageBlock, VoltageSimulator

__version__ = "0.1.0.dev0"

__all__ = [
    "ADSB_FREQ_HZ",
    "OCCUPANCY_THRESHOLD",
    "SPEED_OF_LIGHT_M_S",
    "ADSBTransponder",
    "ArrayConfig",
    "BlockContext",
    "CombTransmitter",
    "ImpulsiveBroadband",
    "InstrumentModel",
    "NarrowbandTransmitter",
    "PFBChannelizer",
    "PointSource",
    "RFISource",
    "SatelliteTransmitter",
    "SpectralLineForeground",
    "TwoLineElement",
    "Visibilities",
    "VoltageBlock",
    "VoltageSimulator",
    "__version__",
    "baseline_index_pairs",
    "bin_any",
    "bin_mean",
    "block_any",
    "confusion_counts",
    "constant_envelope",
    "correlate",
    "dirty_image",
    "earth_location",
    "enu_from_geodetic",
    "enu_from_horizontal",
    "fetch_tles",
    "flag_scores",
    "geometric_delays_s",
    "lm_axis",
    "lm_basis_enu",
    "lm_from_radec",
    "mad_clip_mask",
    "path_delays_s",
    "pool_truth",
    "pool_truth_accumulations",
    "radec_from_lm",
    "read_tle_file",
    "resolve_coupling",
    "source_unit_vectors_enu",
    "spectral_kurtosis_mask",
    "sumthreshold_mask",
    "uvw_wavelengths",
    "zenith_coord",
]
