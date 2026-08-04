"""Shared fixtures and helpers for the rfi_simulator test suite.

Most tests deliberately run a *small* version of the default observation
(fewer channels, fewer samples, fewer blocks) so that the whole suite stays
laptop-fast; one end-to-end test in ``test_imaging.py`` uses the real
Package defaults (384 channels, 61 blocks of 1000 samples, ~2 s).
"""

import hashlib
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


# ----------------------------------------------------------------------
# Bit-reference tests: strict on the reference platform, portable elsewhere
# ----------------------------------------------------------------------
#
# A handful of tests pin the sha256 digest of simulated arrays to guard the
# "this code path produces exactly the bytes it always did" contract. The
# numpy Generator's own output stream is bit-stable across platforms (NEP
# 19), but the *downstream* float arithmetic is not: sin/cos/exp and FFTs
# are evaluated by different library code depending on the numpy build, the
# BLAS backend and the CPU's SIMD path, so their last-bit rounding differs
# from machine to machine even though every run on a *given* machine is
# perfectly deterministic. A hard-coded digest is therefore only meaningful
# on the platform it was recorded on.
#
# `environment_fingerprint` hashes a small canonical computation exercising
# those same operations (transcendentals, an FFT, an RNG draw), and
# `REFERENCE_PLATFORM_FINGERPRINT` is that hash as measured on the platform
# the reference digests below were recorded on. Where the two agree, the
# strict digest is asserted unchanged; where they don't, `assert_bit_reference`
# falls back to a portable check that is still physically meaningful: the
# same scene, built twice from scratch with the same seed, must produce
# byte-identical output (the simulator is a pure function of its inputs),
# and, where supplied, an alternative spelling of "feature disabled" must
# produce the same bytes as the default spelling (pinning the API contract
# rather than the literal bytes).
def environment_fingerprint() -> str:
    """sha256 fingerprint of this platform's transcendental/FFT/RNG behavior."""
    x = np.linspace(-3.0, 3.0, 257)
    trig = np.sin(x) * np.cos(2.0 * x) + np.exp(-0.5 * x**2)
    spectrum = np.fft.fft(np.exp(1j * x) * np.hanning(x.size))
    draw = np.random.default_rng(12345).standard_normal(64)
    payload = trig.tobytes() + spectrum.tobytes() + draw.tobytes()
    return hashlib.sha256(payload).hexdigest()


#: `environment_fingerprint()` as measured on the platform the sha256
#: reference digests in this suite were recorded on.
REFERENCE_PLATFORM_FINGERPRINT = "e736300bdad1b41a75eb7df89a039d5b3e86094c0d66b6d34e1625063f092c92"


def bit_reference_mode() -> str:
    """ "strict" on the reference platform, "fallback" everywhere else."""
    return "strict" if environment_fingerprint() == REFERENCE_PLATFORM_FINGERPRINT else "fallback"


def assert_bit_reference(rebuild, expected_digests, *, alt_rebuild=None) -> str:
    """Assert recorded digests (strict) or a portable equivalent (fallback).

    ``rebuild`` is a zero-argument callable returning a sequence of
    array-likes; it is called once (twice in fallback mode) to build the
    scene from scratch. ``expected_digests`` are the sha256 hex digests
    recorded on the reference platform, one per returned array, checked
    only in strict mode.

    If ``alt_rebuild`` is given, it must return the same shape of sequence
    via an alternative spelling of the same configuration (e.g. an
    explicit ``channelizer=None`` versus an omitted keyword); its output is
    always compared byte-for-byte against ``rebuild()``'s, in both modes,
    which pins the "these spellings are the same code path" contract
    independent of platform.

    Returns the mode string ("strict" or "fallback") so a test can assert
    on it directly if it wants to confirm which path actually ran.
    """
    mode = bit_reference_mode()
    first = [np.ascontiguousarray(value) for value in rebuild()]

    if mode == "strict":
        for index, (value, expected) in enumerate(zip(first, expected_digests)):
            digest = hashlib.sha256(value.tobytes()).hexdigest()
            assert digest == expected, (
                f"[{mode}] item {index} does not match the digest recorded on "
                "the platform the reference digests were recorded on"
            )
    else:
        second = [np.ascontiguousarray(value) for value in rebuild()]
        for index, (a, b) in enumerate(zip(first, second)):
            assert np.array_equal(a, b), (
                f"[{mode}] item {index}: two independent builds of the same "
                "scene (same seed) must be byte-identical -- exact digests "
                "aren't portable across numpy builds/BLAS/SIMD, but "
                "determinism of the simulator is"
            )

    if alt_rebuild is not None:
        alt = [np.ascontiguousarray(value) for value in alt_rebuild()]
        for index, (a, b) in enumerate(zip(first, alt)):
            assert np.array_equal(a, b), (
                f"[{mode}] item {index}: the alternative spelling must produce "
                "byte-identical output to the default spelling"
            )

    return mode
