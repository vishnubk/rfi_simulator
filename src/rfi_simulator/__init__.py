"""rfi_simulator: voltage-level radio-frequency-interference simulator.

Public API is re-exported from submodules here; see ``docs/design_stage2.md``
for the package layout and physics conventions.
"""

from rfi_simulator.array_config import ArrayConfig

__version__ = "0.1.0.dev0"

__all__ = ["ArrayConfig", "__version__"]
