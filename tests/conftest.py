"""Shared fixtures and helpers for the rfi_simulator test suite.

Most tests deliberately run a *small* version of the default observation
(fewer channels, fewer samples, fewer blocks) so that the whole suite stays
laptop-fast; one end-to-end test in ``test_imaging.py`` uses the real
Package defaults (384 channels, 61 blocks of 1000 samples, ~2 s).
"""

from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time

from rfi_simulator import ArrayConfig
from rfi_simulator.delays import earth_location, zenith_coord

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARRAY_YAML = REPO_ROOT / "configs" / "array_default.yaml"

# A fixed, arbitrary UTC epoch: the tests must not depend on wall-clock time.
START_TIME = Time("2026-10-01T04:00:00", scale="utc")

# Standard test source offset used across the suite.
SOURCE_L = float(np.sin(np.deg2rad(0.5)))
SOURCE_M = float(np.sin(np.deg2rad(-0.3)))


@pytest.fixture
def default_array() -> ArrayConfig:
    """The shipped 10-antenna array configuration."""
    return ArrayConfig.from_yaml(DEFAULT_ARRAY_YAML)


@pytest.fixture
def start_time() -> Time:
    """Fixed UTC start time of the simulated observation."""
    return START_TIME


def zenith_phase_center(array: ArrayConfig, start_time: Time, duration_s: float) -> "object":
    """Phase center at the zenith of `array` at the middle of the observation.

    A zenith phase center over a flat (``up = 0``) array makes ``w``
    identically zero, so the tangent-plane imaging in
    `rfi_simulator.imaging` is exact and the acceptance tests measure the
    delay/conjugation conventions rather than a ``w``-term approximation.
    """
    return zenith_coord(earth_location(array), start_time + 0.5 * duration_s * u.s)


def random_flat_array(n_antennas: int, seed: int, max_radius_m: float = 60.0) -> ArrayConfig:
    """An irregular flat array at the default site, for layout-independence tests."""
    rng = np.random.default_rng(seed)
    east_north = rng.uniform(-max_radius_m, max_radius_m, size=(n_antennas, 2))
    positions = np.hstack([east_north, np.zeros((n_antennas, 1))])
    return ArrayConfig(
        antenna_positions_enu_m=positions,
        latitude_deg=37.234,
        longitude_deg=-118.282,
        height_m=1222.0,
        name=f"random_{seed}",
    )
