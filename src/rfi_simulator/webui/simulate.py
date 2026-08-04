"""Request models and the request -> library -> response glue.

Everything the browser can ask for is expressed here as a pydantic model,
turned into ordinary library objects, and reduced to small nested lists.
There is deliberately no HTTP in this module: `run_simulation` takes a
`SimulateRequest` and returns a plain dictionary, so the whole surface can
be exercised from a REPL or a test without a server.

Three conventions are worth knowing before reading on.

**Reduction, not recomputation.** The waterfall the browser draws is the
per-antenna autocorrelation power ``|v|**2`` block-averaged down to at
most `MAX_BINS` cells per axis; the interference overlays are the
library's own ``rfi_mask`` pooled onto the *same* grid with an ANY rule (a
displayed cell is flagged if any voltage-resolution cell inside it was).
Nothing is re-derived from a threshold in the browser, so what is drawn is
ground truth by construction.

**Decibels have a floor.** Power is converted with
``10 log10(P)`` after clamping to `DYNAMIC_RANGE_DB` below the observation
peak, which keeps empty channels from taking the colour scale to minus
infinity and makes the scale comparable between runs of the same setup.

**Units at the boundary.** The library speaks hertz, metres, seconds and
janskys, and so does this API. Field descriptors in `defaults_payload`
carry a display unit and a multiplier so the browser can show megahertz
without either side inventing a second convention.
"""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from astropy.time import Time
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rfi_simulator import (
    ADSBTransponder,
    AiryBeam,
    ArrayConfig,
    CalibrationErrors,
    CombTransmitter,
    GaussianBeam,
    ImpulsiveBroadband,
    InstrumentModel,
    NarrowbandTransmitter,
    PFBChannelizer,
    PointSource,
    RFISource,
    SatelliteTransmitter,
    SpectralLineForeground,
    TwoLineElement,
    VoltageSimulator,
    correlate,
    dirty_image,
    enu_from_horizontal,
    uvw_wavelengths,
)
from rfi_simulator.binning import bin_any, bin_mean
from rfi_simulator.delays import earth_location, zenith_coord
from rfi_simulator.voltages import DEFAULT_CHAN_WIDTH_HZ, DEFAULT_QUANT_TARGET_COUNTS

__all__ = [
    "DEFAULT_CENTER_FREQ_HZ",
    "DEFAULT_N_BLOCKS",
    "DEFAULT_N_CHAN",
    "MAX_ANTENNAS",
    "MAX_COORDINATE_M",
    "MAX_N_BLOCKS",
    "MAX_N_CHAN",
    "MAX_RFI_SOURCES",
    "MAX_SKY_SOURCES",
    "MAX_TOTAL_SAMPLES",
    "START_TIME_UTC",
    "SimulateRequest",
    "build_simulator",
    "default_array",
    "defaults_payload",
    "run_simulation",
    "sample_tle_text",
]

# ----------------------------------------------------------------------
# Fixed choices and guard rails
# ----------------------------------------------------------------------
START_TIME_UTC = "2026-07-30T06:00:00"
"""str: UTC start of every observation this front end runs.

Fixed rather than "now" for two reasons: a run must be reproducible from
its seed alone, and the bundled element set is only accurate for a few
days either side of its own epoch -- this instant sits within hours of it,
with the sample object high in the sky.
"""

N_TIME_PER_BLOCK = 1000
"""int: Time samples per block. Not user-tunable: it fixes the cost of a
run, and with the default channel width one block is 32.768 ms."""

DEFAULT_N_CHAN = 128
DEFAULT_N_BLOCKS = 8
DEFAULT_CENTER_FREQ_HZ = 1.405e9
DEFAULT_NOISE_STD = 1.0
DEFAULT_SEED = 20260730

MAX_ANTENNAS = 32
MAX_N_CHAN = 512
MAX_N_BLOCKS = 32
MAX_SKY_SOURCES = 8
MAX_RFI_SOURCES = 6
MAX_SPECTRAL_LINES = 4

MAX_COORDINATE_M = 1.0e6
"""float: Largest antenna coordinate accepted, metres.

The same bound the aircraft trajectory uses. Beyond it the geometry is no
longer a local array, and coordinates near the floating-point ceiling
overflow into infinities that leave the response full of nulls."""

MAX_TOTAL_SAMPLES = 48_000_000
"""int: Most voltage samples one run may generate.

The count is ``n_antennas * n_chan * n_blocks * N_TIME_PER_BLOCK``. Each
of the individual caps is modest on its own, but their product is not:
taking every one of them at once would allocate several gigabytes and run
for about a minute. This budget keeps a run to a few hundred megabytes and
a few seconds while leaving room for the largest setups the page offers in
any one direction -- a 32-antenna array at the default width, or a
512-channel band on a handful of antennas."""

MAX_BINS = 256
"""int: Most cells the browser is ever sent along one axis of a waterfall."""

MAX_WATERFALL_CELLS = 400_000
"""int: Budget for all antennas' waterfalls together. The time axis is
thinned until the whole response fits, so a 32-antenna run stays a
few megabytes rather than a few tens."""

DYNAMIC_RANGE_DB = 60.0
"""float: Decibels below the observation peak at which power is clamped."""

DISPLAY_PERCENTILES = (0.5, 99.5)
"""tuple of float: Percentiles of the decibel map used as the colour-scale
ends. Clipping the loudest half-percent is what keeps a single bright
interference cell from flattening the whole display."""

IMAGE_N_PIX = 64
IMAGE_FIELD_OF_VIEW_RAD = 0.04
IMAGE_MAX_CHANNELS = 64
"""int: Channels the direct-DFT image uses. Above this the channels are
evenly subsampled, which costs sensitivity but does not move sources."""


# ----------------------------------------------------------------------
# Packaged inputs
# ----------------------------------------------------------------------
def _config_path(filename: str) -> Path | None:
    """Locate a file in the repository's ``configs`` directory, if present.

    The search only ever looks inside a checkout: it climbs from this
    module to the first directory holding a ``pyproject.toml`` and stops
    there. An installed copy has no such ancestor and finds nothing, which
    is the intended answer -- better a known-good built-in default than
    whatever ``configs`` directory happens to sit above the install root
    on a machine shared with other people.
    """
    here = Path(__file__).resolve()
    parents = list(here.parents)
    root_index = next(
        (index for index, parent in enumerate(parents) if (parent / "pyproject.toml").is_file()),
        None,
    )
    if root_index is None:
        return None
    for parent in parents[: root_index + 1]:
        candidate = parent / "configs" / filename
        if candidate.is_file():
            return candidate
    return None


_FALLBACK_ANTENNAS = [
    [0.0, 0.0, 0.0],
    [12.4, -8.7, 0.0],
    [-19.3, 5.2, 0.0],
    [27.8, 21.6, 0.0],
    [-33.1, -14.9, 0.0],
    [8.6, 44.3, 0.0],
    [-45.7, 27.0, 0.0],
    [51.2, -12.5, 0.0],
    [-6.9, -52.8, 0.0],
    [38.4, 9.1, 0.0],
]
_FALLBACK_SITE = {"latitude_deg": 37.234, "longitude_deg": -118.282, "height_m": 1222.0}
_FALLBACK_TLE = (
    "GPS BIIR-5  (PRN 22)\n"
    "1 26407U 00040A   26211.29429826  .00000064  00000+0  00000+0 0  9995\n"
    "2 26407  54.8470 213.4502 0120062 302.9461 145.6045  2.00558031190810\n"
)


def default_array() -> ArrayConfig:
    """The array the page opens with.

    Returns
    -------
    ArrayConfig
        Loaded from ``configs/array_default.yaml`` when that file is
        reachable, otherwise from an identical copy kept here so an
        installed wheel still starts.
    """
    path = _config_path("array_default.yaml")
    if path is not None:
        return ArrayConfig.from_yaml(path)
    return ArrayConfig(
        antenna_positions_enu_m=np.asarray(_FALLBACK_ANTENNAS, dtype=np.float64),
        name="array_default",
        **_FALLBACK_SITE,
    )


def sample_tle_text() -> str:
    """Text of the bundled element set, for the "use the sample" option."""
    path = _config_path("tle_sample.txt")
    if path is None:
        return _FALLBACK_TLE
    tle = TwoLineElement.from_file(path)
    return f"{tle.name}\n{tle.line1}\n{tle.line2}\n"


# ----------------------------------------------------------------------
# Field descriptors: one source of truth for the forms the browser builds
# ----------------------------------------------------------------------
def _num(
    name: str,
    label: str,
    default: float,
    *,
    unit: str = "",
    factor: float = 1.0,
    minimum: float | None = None,
    maximum: float | None = None,
    step: float | None = None,
    help_text: str = "",
) -> dict[str, Any]:
    """One numeric form field.

    `factor` is what the browser multiplies a displayed value by to get
    the value this API expects, so a frequency can be typed in megahertz
    and sent in hertz.

    `default`, `minimum` and `maximum` are in the API's units -- they are
    the validator's own bounds, and the browser divides them by `factor`
    to display them. `step` is the one exception: it is a property of the
    input box, so it is given in the *displayed* unit already.
    """
    field: dict[str, Any] = {
        "name": name,
        "label": label,
        "kind": "number",
        "default": default,
        "unit": unit,
        "factor": factor,
        "help": help_text,
    }
    if minimum is not None:
        field["min"] = minimum
    if maximum is not None:
        field["max"] = maximum
    if step is not None:
        field["step"] = step
    return field


def _choice(name, label, default, options, *, help_text="") -> dict[str, Any]:
    """One dropdown field; `options` is a list of ``{value, label}``."""
    return {
        "name": name,
        "label": label,
        "kind": "choice",
        "default": default,
        "options": options,
        "help": help_text,
    }


def _toggle(name, label, default, *, help_text="") -> dict[str, Any]:
    """One checkbox field."""
    return {
        "name": name,
        "label": label,
        "kind": "toggle",
        "default": default,
        "help": help_text,
    }


def _text(name, label, default, *, multiline=False, help_text="", placeholder="") -> dict[str, Any]:
    """One free-text field."""
    return {
        "name": name,
        "label": label,
        "kind": "text",
        "default": default,
        "multiline": multiline,
        "help": help_text,
        "placeholder": placeholder,
    }


MHZ = 1.0e6
KHZ = 1.0e3

SKY_SOURCE_FIELDS = [
    _num(
        "l",
        "Offset east (l)",
        0.0087,
        minimum=-0.5,
        maximum=0.5,
        step=0.001,
        unit="direction cosine",
        help_text="Direction cosine towards increasing right ascension.",
    ),
    _num(
        "m",
        "Offset north (m)",
        -0.0052,
        minimum=-0.5,
        maximum=0.5,
        step=0.001,
        unit="direction cosine",
        help_text="Direction cosine towards increasing declination.",
    ),
    _num("flux_jy", "Flux density", 5.0, minimum=0.0, maximum=1.0e4, step=0.5, unit="Jy"),
]

# `waveform` is a real scalar field on `TowerParams`/`CombParams`, so it is
# safe to describe with an ordinary field descriptor (see RFI_TYPES) --
# unlike `coupling`, `polarization`, `envelope` and `arrival`, which are
# nested objects and so are not: the schema-driven card form only knows
# how to build one control per scalar field (`buildField` in app.js), and
# every RFI type's `defaults` dict must round-trip through the request
# model unchanged (see `test_schema_defaults_are_accepted_by_the_request_
# models`). Those four stay API-only, reachable directly through
# `/api/simulate`; app.js adds small hand-written controls for them next
# to the schema-driven fields instead of describing them here.
_WAVEFORM_FIELD = _choice(
    "waveform",
    "Waveform",
    "gaussian",
    [
        {"value": "gaussian", "label": "Band-limited noise (gaussian)"},
        {"value": "constant_envelope", "label": "Constant-envelope carrier"},
    ],
    help_text="Constant-envelope is what a spectral-kurtosis detector keys on.",
)

_TOWER_FIELDS = [
    _num(
        "azimuth_deg",
        "Bearing",
        90.0,
        unit="deg",
        minimum=0.0,
        maximum=360.0,
        step=1.0,
        help_text="Compass bearing from the array origin, north through east.",
    ),
    _num("elevation_deg", "Elevation", 0.5, unit="deg", minimum=-5.0, maximum=90.0, step=0.1),
    _num(
        "distance_m",
        "Range",
        4000.0,
        unit="km",
        factor=1000.0,
        minimum=10.0,
        maximum=5.0e5,
        step=0.1,
    ),
    _num(
        "center_freq_hz",
        "Centre frequency",
        1.4055e9,
        unit="MHz",
        factor=MHZ,
        minimum=1.0e6,
        maximum=1.0e11,
        step=0.001,
    ),
    _num(
        "bandwidth_hz",
        "Bandwidth",
        2.0e5,
        unit="kHz",
        factor=KHZ,
        minimum=0.0,
        maximum=1.0e9,
        step=1.0,
    ),
    _num(
        "received_power_jy",
        "Received power",
        500.0,
        unit="Jy",
        minimum=0.0,
        maximum=1.0e9,
        step=10.0,
        help_text="Power at the array origin while the transmitter is on.",
    ),
    _num(
        "duty_cycle",
        "Duty cycle",
        0.5,
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        help_text="Fraction of frames the transmitter is on.",
    ),
    _num(
        "frame_duration_s",
        "Frame length",
        0.01,
        unit="ms",
        factor=1.0e-3,
        minimum=1.0e-5,
        maximum=10.0,
        step=1.0,
    ),
    _WAVEFORM_FIELD,
]

_IMPULSIVE_FIELDS = [
    _num("rate_hz", "Event rate", 200.0, unit="events/s", minimum=0.0, maximum=1.0e5, step=10.0),
    _num(
        "received_power_jy",
        "Weakest event",
        2000.0,
        unit="Jy",
        minimum=0.0,
        maximum=1.0e9,
        step=100.0,
        help_text="Power of the faintest burst; brighter ones follow a power law.",
    ),
    _num("azimuth_deg", "Bearing", 135.0, unit="deg", minimum=0.0, maximum=360.0, step=1.0),
    _num("elevation_deg", "Elevation", 1.0, unit="deg", minimum=-5.0, maximum=90.0, step=0.1),
    _num(
        "distance_m",
        "Range",
        5000.0,
        unit="km",
        factor=1000.0,
        minimum=10.0,
        maximum=5.0e5,
        step=0.1,
    ),
    _num("power_law_index", "Power-law index", 2.0, minimum=1.01, maximum=6.0, step=0.1),
    _num("max_power_ratio", "Brightest / faintest", 30.0, minimum=1.0, maximum=1.0e4, step=1.0),
    _num("pulse_width_samples", "Burst length", 1, unit="samples", minimum=1, maximum=64, step=1),
]

_SATELLITE_FIELDS = [
    _choice(
        "tle_source",
        "Element set",
        "sample",
        [
            {"value": "sample", "label": "Bundled sample object"},
            {"value": "custom", "label": "Paste your own"},
        ],
        help_text="Nothing is fetched from the network; paste the lines yourself.",
    ),
    _text(
        "tle_text",
        "Pasted element set",
        "",
        multiline=True,
        placeholder="OBJECT NAME\n1 …\n2 …",
        help_text="Two 69-character lines, optionally preceded by a name line.",
    ),
    _num(
        "carrier_freq_hz",
        "Carrier frequency",
        1.405e9,
        unit="MHz",
        factor=MHZ,
        minimum=1.0e6,
        maximum=1.0e11,
        step=0.001,
        help_text="Rest-frame; the received frequency is Doppler shifted.",
    ),
    _num(
        "received_power_jy",
        "Received power",
        400.0,
        unit="Jy",
        minimum=0.0,
        maximum=1.0e9,
        step=10.0,
    ),
    _num(
        "sideband_bandwidth_hz",
        "Sideband width",
        2.0e5,
        unit="kHz",
        factor=KHZ,
        minimum=0.0,
        maximum=1.0e9,
        step=1.0,
    ),
    _num("sideband_power_fraction", "Power in sidebands", 0.5, minimum=0.0, maximum=1.0, step=0.05),
    _num(
        "min_elevation_deg", "Horizon cut", 0.0, unit="deg", minimum=-90.0, maximum=90.0, step=1.0
    ),
    _toggle("apply_doppler", "Apply Doppler shift", True),
]

_AIRCRAFT_FIELDS = [
    _num(
        "east_m",
        "Start east",
        -40000.0,
        unit="km",
        factor=1000.0,
        minimum=-1.0e6,
        maximum=1.0e6,
        step=1.0,
    ),
    _num(
        "north_m",
        "Start north",
        15000.0,
        unit="km",
        factor=1000.0,
        minimum=-1.0e6,
        maximum=1.0e6,
        step=1.0,
    ),
    _num("altitude_m", "Altitude", 11000.0, unit="m", minimum=0.0, maximum=1.0e5, step=100.0),
    _num(
        "velocity_east_m_s",
        "Speed east",
        250.0,
        unit="m/s",
        minimum=-1.0e3,
        maximum=1.0e3,
        step=10.0,
    ),
    _num(
        "velocity_north_m_s",
        "Speed north",
        0.0,
        unit="m/s",
        minimum=-1.0e3,
        maximum=1.0e3,
        step=10.0,
    ),
    _num(
        "carrier_freq_hz",
        "Carrier frequency",
        1.4052e9,
        unit="MHz",
        factor=MHZ,
        minimum=1.0e6,
        maximum=1.0e11,
        step=0.001,
    ),
    _num(
        "bandwidth_hz",
        "Burst bandwidth",
        2.0e5,
        unit="kHz",
        factor=KHZ,
        minimum=0.0,
        maximum=1.0e9,
        step=1.0,
    ),
    _num(
        "received_power_jy",
        "Received power",
        1.0e4,
        unit="Jy",
        minimum=0.0,
        maximum=1.0e9,
        step=100.0,
    ),
    _num(
        "message_rate_hz",
        "Message rate",
        500.0,
        unit="messages/s",
        minimum=0.0,
        maximum=1.0e5,
        step=10.0,
    ),
    _num("pulse_width_samples", "Burst length", 1, unit="samples", minimum=1, maximum=64, step=1),
    _num(
        "min_elevation_deg", "Horizon cut", 0.0, unit="deg", minimum=-90.0, maximum=90.0, step=1.0
    ),
]

_COMB_FIELDS = [
    _num("azimuth_deg", "Bearing", 200.0, unit="deg", minimum=0.0, maximum=360.0, step=1.0),
    _num("elevation_deg", "Elevation", 2.0, unit="deg", minimum=-5.0, maximum=90.0, step=0.1),
    _num(
        "distance_m", "Range", 3000.0, unit="km", factor=1000.0,
        minimum=10.0, maximum=5.0e5, step=0.1,
    ),
    _num(
        "fundamental_hz", "Fundamental frequency", 1.405e6, unit="MHz", factor=MHZ,
        minimum=1.0e3, maximum=1.0e10, step=0.000001,
        help_text="May sit far below the simulated band; only in-band harmonics show up.",
    ),
    _text(
        "harmonic_numbers",
        "Harmonics (comma-separated)",
        "999,1000,1001",
        help_text="Which multiples of the fundamental the device emits, e.g. '999,1000,1001'.",
    ),
    _num(
        "received_power_jy", "Received power per harmonic", 200.0, unit="Jy",
        minimum=0.0, maximum=1.0e9, step=10.0,
    ),
    _num(
        "bandwidth_hz", "Bandwidth per harmonic", 0.0, unit="kHz", factor=KHZ,
        minimum=0.0, maximum=1.0e9, step=1.0,
        help_text="0 (default) makes every harmonic a pure line, one channel wide.",
    ),
    _num("duty_cycle", "Duty cycle", 1.0, minimum=0.0, maximum=1.0, step=0.05),
    _num(
        "frame_duration_s", "Frame length", 0.01, unit="ms", factor=1.0e-3,
        minimum=1.0e-5, maximum=10.0, step=1.0,
    ),
    _WAVEFORM_FIELD,
]

RFI_TYPES = [
    {
        "type": "tower",
        "label": "Ground transmitter",
        "summary": (
            "A mast on a fixed bearing filling a slice of the band, switching on and off in frames."
        ),
        "fields": _TOWER_FIELDS,
    },
    {
        "type": "impulsive",
        "label": "Impulsive broadband",
        "summary": "Arcing and sparks: short bursts across the whole band, most of them faint.",
        "fields": _IMPULSIVE_FIELDS,
    },
    {
        "type": "satellite",
        "label": "Satellite downlink",
        "summary": (
            "A carrier from an orbiting transmitter, propagated from an "
            "element set and Doppler shifted."
        ),
        "fields": _SATELLITE_FIELDS,
    },
    {
        "type": "aircraft",
        "label": "Aircraft transponder",
        "summary": (
            "Bursts from an aircraft on a straight-line course, its delays moving block to block."
        ),
        "fields": _AIRCRAFT_FIELDS,
    },
    {
        "type": "comb",
        "label": "Harmonic comb",
        "summary": (
            "One device -- a switching supply, a broken shield -- emitting several "
            "harmonics of one fundamental, sharing a position and an on/off pattern."
        ),
        "fields": _COMB_FIELDS,
    },
]


def _schema_defaults(fields: list[dict[str, Any]]) -> dict[str, Any]:
    """The default value of every field, keyed by field name."""
    return {field["name"]: field["default"] for field in fields}


# ----------------------------------------------------------------------
# Shared realism specifications: coupling, polarization, envelope, arrival
# ----------------------------------------------------------------------
# These mirror the dict-shaped specifications `rfi_simulator.rfi` accepts
# (`resolve_coupling`, `resolve_polarization`, `_normalize_envelope`,
# `_normalize_arrival`). Pydantic gives them proper field-level validation
# at the API boundary; `.build()`/the module-level `_build_*` helpers turn
# a validated model back into the plain dict or list the library expects,
# so the library's own validation still has the last word.
class LognormalCoupling(BaseModel):
    """A per-antenna coupling scatter, drawn lognormal in dB of power."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["lognormal"] = "lognormal"
    sigma_db: float = Field(default=3.0, ge=0.0, le=60.0)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)


CouplingSpec = list[float] | LognormalCoupling
"""An explicit per-antenna amplitude vector, or a lognormal draw."""


def _build_coupling(coupling: CouplingSpec | None) -> Any:
    """`coupling` as the plain value/dict `resolve_coupling` accepts."""
    if coupling is None:
        return None
    if isinstance(coupling, LognormalCoupling):
        return {"type": "lognormal", "sigma_db": coupling.sigma_db, "seed": coupling.seed}
    return list(coupling)


class LinearPolarization(BaseModel):
    """Fully (or partially) linearly polarized emission."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["linear"] = "linear"
    angle_deg: float = Field(default=45.0, ge=-360.0, le=360.0)
    fraction: float = Field(default=1.0, ge=0.0, le=1.0)


class FullPolarization(BaseModel):
    """An explicit Jones-vector polarization state."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["full"] = "full"
    jones_re: list[float] = Field(min_length=2, max_length=2)
    jones_im: list[float] = Field(default=[0.0, 0.0], min_length=2, max_length=2)
    fraction: float = Field(default=1.0, ge=0.0, le=1.0)


PolarizationSpec = Annotated[
    LinearPolarization | FullPolarization, Field(discriminator="type")
]


def _build_polarization(polarization: PolarizationSpec | None) -> dict[str, Any] | None:
    """`polarization` as the plain dict `resolve_polarization` accepts."""
    if polarization is None:
        return None
    if isinstance(polarization, LinearPolarization):
        return {
            "type": "linear",
            "angle_deg": polarization.angle_deg,
            "fraction": polarization.fraction,
        }
    jones = [
        complex(polarization.jones_re[0], polarization.jones_im[0]),
        complex(polarization.jones_re[1], polarization.jones_im[1]),
    ]
    return {"type": "full", "jones": jones, "fraction": polarization.fraction}


class PeriodicEnvelope(BaseModel):
    """A clocked on/off pattern, in place of i.i.d. duty-cycle frames."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["periodic"] = "periodic"
    period_s: float = Field(gt=0.0, le=100.0)
    duty: float = Field(default=1.0, ge=0.0, le=1.0)
    phase: float = Field(default=0.0, ge=-1.0e6, le=1.0e6)


def _build_envelope(envelope: PeriodicEnvelope | None) -> dict[str, Any] | None:
    """`envelope` as the plain dict `_normalize_envelope` accepts."""
    if envelope is None:
        return None
    return {
        "type": "periodic",
        "period_s": envelope.period_s,
        "duty": envelope.duty,
        "phase": envelope.phase,
    }


class PeriodicArrival(BaseModel):
    """A regular, jittered pulse train for `ImpulsiveBroadband`."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["periodic"] = "periodic"
    rate_hz: float = Field(gt=0.0, le=1.0e5)
    jitter_s: float = Field(default=0.0, ge=0.0, le=10.0)


ArrivalSpec = Literal["poisson"] | PeriodicArrival


def _build_arrival(arrival: ArrivalSpec) -> Any:
    """`arrival` as the plain string/dict `_normalize_arrival` accepts."""
    if arrival == "poisson":
        return "poisson"
    return {"type": "periodic", "rate_hz": arrival.rate_hz, "jitter_s": arrival.jitter_s}


# ----------------------------------------------------------------------
# Request models
# ----------------------------------------------------------------------
class SkySource(BaseModel):
    """One celestial point source, placed by direction cosines."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="source", max_length=40)
    l: float = Field(default=0.0087, ge=-0.5, le=0.5)  # noqa: E741 - the standard symbol
    m: float = Field(default=-0.0052, ge=-0.5, le=0.5)
    flux_jy: float = Field(default=5.0, ge=0.0, le=1.0e4)

    @model_validator(mode="after")
    def _check_on_sky(self) -> "SkySource":
        if self.l**2 + self.m**2 >= 1.0:
            raise ValueError("l and m place this source off the sky: l^2 + m^2 must be below 1")
        return self


class TowerParams(BaseModel):
    """A stationary ground transmitter, placed by bearing and range."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["tower"] = "tower"
    name: str = Field(default="tower", max_length=40)
    azimuth_deg: float = Field(default=90.0, ge=0.0, le=360.0)
    elevation_deg: float = Field(default=0.5, ge=-5.0, le=90.0)
    distance_m: float = Field(default=4000.0, ge=10.0, le=5.0e5)
    center_freq_hz: float = Field(default=1.4055e9, ge=1.0e6, le=1.0e11)
    bandwidth_hz: float = Field(default=2.0e5, ge=0.0, le=1.0e9)
    received_power_jy: float = Field(default=500.0, ge=0.0, le=1.0e9)
    duty_cycle: float = Field(default=0.5, ge=0.0, le=1.0)
    frame_duration_s: float = Field(default=0.01, gt=0.0, le=10.0)
    waveform: Literal["gaussian", "constant_envelope"] = "gaussian"
    envelope: PeriodicEnvelope | None = None
    coupling: CouplingSpec | None = None
    polarization: PolarizationSpec | None = None

    def build(self) -> RFISource:
        """The library source this describes.

        `envelope` and `duty_cycle` are mutually exclusive in the library
        (see `NarrowbandTransmitter`): when an envelope is given, this
        passes `duty_cycle=1.0` regardless of the request's own value, so
        the form's duty-cycle field never has to be reset by hand to turn
        the periodic envelope on.
        """
        return NarrowbandTransmitter(
            position_enu_m=enu_from_horizontal(
                self.azimuth_deg, self.elevation_deg, self.distance_m
            ),
            center_freq_hz=self.center_freq_hz,
            bandwidth_hz=self.bandwidth_hz,
            received_power_jy=self.received_power_jy,
            duty_cycle=1.0 if self.envelope is not None else self.duty_cycle,
            frame_duration_s=self.frame_duration_s,
            envelope=_build_envelope(self.envelope),
            waveform=self.waveform,
            coupling=_build_coupling(self.coupling),
            polarization=_build_polarization(self.polarization),
            name=self.name,
        )


class ImpulsiveParams(BaseModel):
    """Short broadband bursts from a fixed direction."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["impulsive"] = "impulsive"
    name: str = Field(default="impulsive", max_length=40)
    rate_hz: float = Field(default=200.0, ge=0.0, le=1.0e5)
    received_power_jy: float = Field(default=2000.0, ge=0.0, le=1.0e9)
    azimuth_deg: float = Field(default=135.0, ge=0.0, le=360.0)
    elevation_deg: float = Field(default=1.0, ge=-5.0, le=90.0)
    distance_m: float = Field(default=5000.0, ge=10.0, le=5.0e5)
    power_law_index: float = Field(default=2.0, gt=1.0, le=6.0)
    max_power_ratio: float = Field(default=30.0, ge=1.0, le=1.0e4)
    pulse_width_samples: int = Field(default=1, ge=1, le=64)
    arrival: ArrivalSpec = "poisson"
    coupling: CouplingSpec | None = None
    polarization: PolarizationSpec | None = None

    def build(self) -> RFISource:
        """The library source this describes.

        `rate_hz` and a periodic `arrival` are mutually exclusive in the
        library (see `ImpulsiveBroadband`): `rate_hz` is only forwarded
        for the default Poisson arrivals, where the event rate has
        nowhere else to live.
        """
        return ImpulsiveBroadband(
            rate_hz=self.rate_hz if self.arrival == "poisson" else None,
            received_power_jy=self.received_power_jy,
            arrival=_build_arrival(self.arrival),
            position_enu_m=enu_from_horizontal(
                self.azimuth_deg, self.elevation_deg, self.distance_m
            ),
            power_law_index=self.power_law_index,
            max_power_ratio=self.max_power_ratio,
            pulse_width_samples=self.pulse_width_samples,
            coupling=_build_coupling(self.coupling),
            polarization=_build_polarization(self.polarization),
            name=self.name,
        )


class SatelliteParams(BaseModel):
    """A downlink propagated from a two-line element set."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["satellite"] = "satellite"
    name: str = Field(default="satellite", max_length=40)
    tle_source: Literal["sample", "custom"] = "sample"
    tle_text: str = Field(default="", max_length=4000)
    carrier_freq_hz: float = Field(default=1.405e9, ge=1.0e6, le=1.0e11)
    received_power_jy: float = Field(default=400.0, ge=0.0, le=1.0e9)
    sideband_bandwidth_hz: float = Field(default=2.0e5, ge=0.0, le=1.0e9)
    sideband_power_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    apply_doppler: bool = True
    min_elevation_deg: float = Field(default=0.0, ge=-90.0, le=90.0)
    coupling: CouplingSpec | None = None
    polarization: PolarizationSpec | None = None

    @model_validator(mode="after")
    def _check_elements(self) -> "SatelliteParams":
        """Parse the element set now, so a typo is a form error not a crash."""
        self.element_set()
        return self

    def element_set(self) -> TwoLineElement:
        """The element set to propagate, parsed from text or the bundle."""
        text = sample_tle_text() if self.tle_source == "sample" else self.tle_text
        if not text.strip():
            raise ValueError(
                "paste an element set: two 69-character lines, optionally preceded by a name line"
            )
        try:
            return TwoLineElement.from_string(text)
        except Exception as exc:
            raise ValueError(
                f"this is not a readable element set ({exc}). It must be two "
                "69-character lines beginning '1 ' and '2 ', optionally "
                "preceded by a name line"
            ) from exc

    def build(self) -> RFISource:
        """The library source this describes."""
        return SatelliteTransmitter(
            self.element_set(),
            carrier_freq_hz=self.carrier_freq_hz,
            received_power_jy=self.received_power_jy,
            sideband_bandwidth_hz=self.sideband_bandwidth_hz,
            sideband_power_fraction=self.sideband_power_fraction,
            apply_doppler=self.apply_doppler,
            min_elevation_deg=self.min_elevation_deg,
            coupling=_build_coupling(self.coupling),
            polarization=_build_polarization(self.polarization),
            name=self.name,
        )


class AircraftParams(BaseModel):
    """An aircraft transponder on a straight-line course."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["aircraft"] = "aircraft"
    name: str = Field(default="transponder", max_length=40)
    east_m: float = Field(default=-40000.0, ge=-1.0e6, le=1.0e6)
    north_m: float = Field(default=15000.0, ge=-1.0e6, le=1.0e6)
    altitude_m: float = Field(default=11000.0, ge=0.0, le=1.0e5)
    velocity_east_m_s: float = Field(default=250.0, ge=-1.0e3, le=1.0e3)
    velocity_north_m_s: float = Field(default=0.0, ge=-1.0e3, le=1.0e3)
    carrier_freq_hz: float = Field(default=1.4052e9, ge=1.0e6, le=1.0e11)
    bandwidth_hz: float = Field(default=2.0e5, ge=0.0, le=1.0e9)
    received_power_jy: float = Field(default=1.0e4, ge=0.0, le=1.0e9)
    message_rate_hz: float = Field(default=500.0, ge=0.0, le=1.0e5)
    pulse_width_samples: int = Field(default=1, ge=1, le=64)
    min_elevation_deg: float = Field(default=0.0, ge=-90.0, le=90.0)
    coupling: CouplingSpec | None = None
    polarization: PolarizationSpec | None = None

    def build(self) -> RFISource:
        """The library source this describes."""
        return ADSBTransponder(
            position_enu_m=(self.east_m, self.north_m, self.altitude_m),
            velocity_enu_m_s=(self.velocity_east_m_s, self.velocity_north_m_s, 0.0),
            carrier_freq_hz=self.carrier_freq_hz,
            bandwidth_hz=self.bandwidth_hz,
            received_power_jy=self.received_power_jy,
            message_rate_hz=self.message_rate_hz,
            pulse_width_samples=self.pulse_width_samples,
            min_elevation_deg=self.min_elevation_deg,
            coupling=_build_coupling(self.coupling),
            polarization=_build_polarization(self.polarization),
            name=self.name,
        )


class CombParams(BaseModel):
    """One device emitting a comb of harmonics of a single fundamental."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["comb"] = "comb"
    name: str = Field(default="comb", max_length=40)
    azimuth_deg: float = Field(default=200.0, ge=0.0, le=360.0)
    elevation_deg: float = Field(default=2.0, ge=-5.0, le=90.0)
    distance_m: float = Field(default=3000.0, ge=10.0, le=5.0e5)
    # 999/1000/1001 x 1.405 MHz sit at 1403.6/1405.0/1406.4 MHz -- inside
    # the default ~3.9 MHz band around the default 1.405 GHz centre, so a
    # freshly added comb source is visible without retuning anything.
    fundamental_hz: float = Field(default=1.405e6, gt=0.0, le=1.0e10)
    harmonic_numbers: list[int] = Field(default=[999, 1000, 1001], min_length=1, max_length=32)
    received_power_jy: float = Field(default=200.0, ge=0.0, le=1.0e9)
    bandwidth_hz: float = Field(default=0.0, ge=0.0, le=1.0e9)
    duty_cycle: float = Field(default=1.0, ge=0.0, le=1.0)
    frame_duration_s: float = Field(default=0.01, gt=0.0, le=10.0)
    waveform: Literal["gaussian", "constant_envelope"] = "gaussian"
    envelope: PeriodicEnvelope | None = None
    coupling: CouplingSpec | None = None
    polarization: PolarizationSpec | None = None

    @field_validator("harmonic_numbers", mode="before")
    @classmethod
    def _parse_harmonics(cls, value: Any) -> Any:
        """Accept the browser's comma-separated text field as well as a list.

        The card form has one control per scalar field (see
        `_COMB_FIELDS`'s ``harmonic_numbers`` text field), so the browser
        sends ``"999,1000,1001"`` rather than a JSON array; a request
        built directly against the API (as the tests do) can still send a
        real list.
        """
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",") if part.strip()]
            try:
                return [int(part) for part in parts]
            except ValueError as exc:
                raise ValueError(
                    f"harmonic_numbers must be a comma-separated list of integers, got {value!r}"
                ) from exc
        return value

    @field_validator("harmonic_numbers")
    @classmethod
    def _check_harmonics(cls, value: list[int]) -> list[int]:
        if any(number < 1 for number in value):
            raise ValueError(f"harmonic_numbers must all be >= 1, got {value}")
        if len(set(value)) != len(value):
            raise ValueError(f"harmonic_numbers must be unique, got {value}")
        return value

    def build(self) -> RFISource:
        """The library source this describes. See `TowerParams.build`."""
        return CombTransmitter(
            position_enu_m=enu_from_horizontal(
                self.azimuth_deg, self.elevation_deg, self.distance_m
            ),
            fundamental_hz=self.fundamental_hz,
            harmonic_numbers=self.harmonic_numbers,
            received_powers_jy=self.received_power_jy,
            bandwidth_hz=self.bandwidth_hz,
            duty_cycle=1.0 if self.envelope is not None else self.duty_cycle,
            frame_duration_s=self.frame_duration_s,
            envelope=_build_envelope(self.envelope),
            waveform=self.waveform,
            coupling=_build_coupling(self.coupling),
            polarization=_build_polarization(self.polarization),
            name=self.name,
        )


RFIParams = Annotated[
    TowerParams | ImpulsiveParams | SatelliteParams | AircraftParams | CombParams,
    Field(discriminator="type"),
]


SPECTRAL_LINE_FIELDS = [
    _num(
        "center_freq_hz",
        "Line centre",
        1420.4058e6,
        unit="MHz",
        factor=MHZ,
        minimum=1.0e6,
        maximum=1.0e11,
        step=0.0001,
        help_text="Default: the 21 cm neutral hydrogen line, rest frame.",
    ),
    _num(
        "fwhm_hz", "Line width (FWHM)", 20.0e3, unit="kHz", factor=KHZ,
        minimum=1.0, maximum=1.0e9, step=1.0,
    ),
    _num(
        "line_flux_jy", "Peak-channel power", 1.0, unit="Jy",
        minimum=0.0, maximum=1.0e6, step=0.1,
        help_text="Added per antenna, like noise_std^2, tapering as a Gaussian in frequency.",
    ),
]


class SpectralLineParams(BaseModel):
    """A celestial spectral line -- ground truth labelled "celestial", not "rfi"."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="hi_line", max_length=40)
    center_freq_hz: float = Field(default=1420.4058e6, gt=0.0, le=1.0e11)
    fwhm_hz: float = Field(default=20.0e3, gt=0.0, le=1.0e9)
    line_flux_jy: float = Field(default=1.0, ge=0.0, le=1.0e6)

    def build(self) -> SpectralLineForeground:
        """The library foreground this describes."""
        return SpectralLineForeground(
            center_freq_hz=self.center_freq_hz,
            fwhm_hz=self.fwhm_hz,
            line_flux_jy=self.line_flux_jy,
            name=self.name,
        )


class InstrumentParams(BaseModel):
    """Per-antenna direction-independent gain realism (see `InstrumentModel`).

    Field names match `InstrumentModel.from_params`'s keyword arguments
    exactly, so `build` can forward them with a single ``model_dump()``.
    """

    model_config = ConfigDict(extra="forbid")

    gain_scatter_db: float = Field(default=0.4, ge=0.0, le=10.0)
    phase_offsets: Literal["zero", "uniform"] = "zero"
    bandpass_ripple_db: float = Field(default=0.05, ge=0.0, le=5.0)
    bandpass_n_modes: int = Field(default=3, ge=1, le=16)
    band_slope_db: float = Field(default=0.0, ge=0.0, le=10.0)
    band_slope_n_modes: int = Field(default=2, ge=1, le=16)
    subband_scatter_db: float = Field(default=0.0, ge=0.0, le=10.0)
    n_subbands: int = Field(default=1, ge=1, le=64)

    def build(self, n_antennas: int, seed: int) -> InstrumentModel:
        """The library model this describes.

        `freq_hz` is deliberately omitted: leaving `band_hz` unset makes
        the ripple/slope/subband reference band default to whatever grid
        `InstrumentModel.gains` is later evaluated on, which is exactly
        the simulated band here -- see `InstrumentModel.from_params`.
        """
        return InstrumentModel.from_params(n_antennas, seed=seed, **self.model_dump())


class CalibrationErrorParams(BaseModel):
    """Residual calibration error (see `CalibrationErrors`), applied at `correlate`."""

    model_config = ConfigDict(extra="forbid")

    phase_error_deg_rms: float = Field(default=5.0, ge=0.0, le=180.0)
    delay_error_ns_rms: float = Field(default=0.0, ge=0.0, le=1000.0)
    amplitude_error_db_rms: float = Field(default=0.0, ge=0.0, le=10.0)

    def build(self, n_antennas: int, seed: int) -> CalibrationErrors:
        """The library model this describes."""
        return CalibrationErrors.from_params(n_antennas, seed=seed, **self.model_dump())


class ChannelizerParams(BaseModel):
    """Polyphase-filterbank channel response (see `PFBChannelizer`)."""

    model_config = ConfigDict(extra="forbid")

    n_taps: int = Field(default=4, ge=1, le=32)
    window: Literal["hann", "hamming", "blackman"] = "hamming"
    sinc_bandwidth: float = Field(default=1.025, gt=0.0, le=8.0)

    def build(self) -> PFBChannelizer:
        """The library channelizer this describes."""
        return PFBChannelizer(
            n_taps=self.n_taps, window=self.window, sinc_bandwidth=self.sinc_bandwidth
        )


class PrimaryBeamParams(BaseModel):
    """A primary beam attenuating celestial flux away from the pointing."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["gaussian", "airy"] = "gaussian"
    dish_diameter_m: float = Field(default=4.5, gt=0.0, le=1000.0)

    def build(self) -> GaussianBeam | AiryBeam:
        """The library beam this describes."""
        cls = GaussianBeam if self.type == "gaussian" else AiryBeam
        return cls(dish_diameter_m=self.dish_diameter_m)


class QuantizationParams(BaseModel):
    """4-bit quantization of the synthesized voltages.

    Presence of this object on the request (as opposed to ``None``) is
    what turns quantization on: there is currently one supported mode,
    ``"int4"``, so there is nothing else for a `type` field to select.
    """

    model_config = ConfigDict(extra="forbid")

    quant_target_counts: float = Field(default=DEFAULT_QUANT_TARGET_COUNTS, gt=0.0, le=100.0)
    quant_scale: float | None = Field(default=None, gt=0.0)


class SimParams(BaseModel):
    """Observation size and the receiver noise level."""

    model_config = ConfigDict(extra="forbid")

    n_chan: int = Field(default=DEFAULT_N_CHAN, ge=4, le=MAX_N_CHAN)
    n_blocks: int = Field(default=DEFAULT_N_BLOCKS, ge=1, le=MAX_N_BLOCKS)
    center_freq_hz: float = Field(default=DEFAULT_CENTER_FREQ_HZ, ge=1.0e6, le=1.0e11)
    noise_std: float = Field(default=DEFAULT_NOISE_STD, ge=0.0, le=1.0e4)
    seed: int = Field(default=DEFAULT_SEED, ge=0, le=2**31 - 1)


class SimulateRequest(BaseModel):
    """Everything one run needs.

    Antenna positions are local East-North-Up metres relative to the
    array origin, which is the site of the default array.
    """

    model_config = ConfigDict(extra="forbid")

    antennas: list[list[float]] = Field(default_factory=list, max_length=MAX_ANTENNAS)
    sky_sources: list[SkySource] = Field(default_factory=list, max_length=MAX_SKY_SOURCES)
    rfi_sources: list[RFIParams] = Field(default_factory=list, max_length=MAX_RFI_SOURCES)
    spectral_lines: list[SpectralLineParams] = Field(
        default_factory=list, max_length=MAX_SPECTRAL_LINES
    )
    sim: SimParams = Field(default_factory=SimParams)
    n_pol: Literal[1, 2] = 1
    instrument: InstrumentParams | None = None
    calibration_errors: CalibrationErrorParams | None = None
    channelizer: ChannelizerParams | None = None
    primary_beam: PrimaryBeamParams | None = None
    quantization: QuantizationParams | None = None

    @field_validator("antennas")
    @classmethod
    def _check_antennas(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) < 2:
            raise ValueError("place at least 2 antennas: one antenna forms no baseline")
        # An over-long list never reaches this validator: the Field's
        # max_length rejects it before the elements are even coerced.
        for index, position in enumerate(value):
            if len(position) != 3:
                raise ValueError(
                    f"antenna {index} must be [east_m, north_m, up_m], got {len(position)} numbers"
                )
            if not all(math.isfinite(coordinate) for coordinate in position):
                raise ValueError(f"antenna {index} has a position that is not a finite number")
            if any(abs(coordinate) > MAX_COORDINATE_M for coordinate in position):
                raise ValueError(
                    f"antenna {index} lies further than {MAX_COORDINATE_M:g} m from the array "
                    "origin: give east, north and up in metres, not in another unit"
                )
        return value

    @model_validator(mode="after")
    def _check_total_size(self) -> "SimulateRequest":
        """Refuse a run whose *product* of sizes is too large.

        Antennas, channels and integrations are each capped on their own,
        but the cost of a run is their product; this is the check that
        keeps a request that is legal in every field from allocating
        gigabytes.
        """
        total = len(self.antennas) * self.sim.n_chan * self.sim.n_blocks * N_TIME_PER_BLOCK
        if total > MAX_TOTAL_SAMPLES:
            raise ValueError(
                f"this run needs {total:,} voltage samples "
                f"({len(self.antennas)} antennas x {self.sim.n_chan} channels x "
                f"{self.sim.n_blocks} integrations x {N_TIME_PER_BLOCK} samples), more than the "
                f"{MAX_TOTAL_SAMPLES:,} this front end allows: reduce the number of antennas, "
                "n_chan, or n_blocks"
            )
        return self


def default_request() -> SimulateRequest:
    """The run the page performs before anything is edited."""
    array = default_array()
    return SimulateRequest(
        antennas=[list(row) for row in array.antenna_positions_enu_m],
        sky_sources=[SkySource()],
        rfi_sources=[],
        sim=SimParams(),
    )


# ----------------------------------------------------------------------
# Defaults payload
# ----------------------------------------------------------------------
def defaults_payload() -> dict[str, Any]:
    """Everything the page needs to draw itself before the first run.

    Returns
    -------
    dict
        Default array and site, default observation parameters, the guard
        rails, and a field descriptor list per source type so the browser
        builds its forms from this rather than from hard-coded copies.
    """
    array = default_array()
    return {
        "array": {
            "name": array.name,
            "latitude_deg": array.latitude_deg,
            "longitude_deg": array.longitude_deg,
            "height_m": array.height_m,
            "antennas": [[float(x) for x in row] for row in array.antenna_positions_enu_m],
        },
        "sim": {
            "n_chan": DEFAULT_N_CHAN,
            "n_blocks": DEFAULT_N_BLOCKS,
            "center_freq_hz": DEFAULT_CENTER_FREQ_HZ,
            "noise_std": DEFAULT_NOISE_STD,
            "seed": DEFAULT_SEED,
            "chan_width_hz": DEFAULT_CHAN_WIDTH_HZ,
            "n_time_per_block": N_TIME_PER_BLOCK,
            "block_duration_s": N_TIME_PER_BLOCK / DEFAULT_CHAN_WIDTH_HZ,
            "start_time_utc": START_TIME_UTC,
        },
        "limits": {
            "max_antennas": MAX_ANTENNAS,
            "max_n_chan": MAX_N_CHAN,
            "max_n_blocks": MAX_N_BLOCKS,
            "max_sky_sources": MAX_SKY_SOURCES,
            "max_rfi_sources": MAX_RFI_SOURCES,
            "max_spectral_lines": MAX_SPECTRAL_LINES,
            "max_total_samples": MAX_TOTAL_SAMPLES,
            "dynamic_range_db": DYNAMIC_RANGE_DB,
        },
        "sky_source": {
            "label": "Sky source",
            "fields": SKY_SOURCE_FIELDS,
            "defaults": _schema_defaults(SKY_SOURCE_FIELDS),
        },
        "spectral_line": {
            "label": "Spectral line",
            "fields": SPECTRAL_LINE_FIELDS,
            "defaults": _schema_defaults(SPECTRAL_LINE_FIELDS),
        },
        "rfi_types": [
            dict(entry, defaults=_schema_defaults(entry["fields"])) for entry in RFI_TYPES
        ],
        "sample_tle": sample_tle_text(),
    }


# ----------------------------------------------------------------------
# Array reduction helpers
# ----------------------------------------------------------------------
def _waterfall_shape(n_antennas: int, n_chan: int, n_blocks: int) -> tuple[int, int]:
    """Channel and per-block time bin counts that fit the response budget."""
    chan_bins = min(n_chan, MAX_BINS)
    budget = max(1, MAX_WATERFALL_CELLS // (n_antennas * chan_bins))
    time_bins = min(MAX_BINS, max(n_blocks, budget))
    per_block = max(1, time_bins // n_blocks)
    per_block = min(per_block, N_TIME_PER_BLOCK)
    return chan_bins, per_block


def _round_grid(values: np.ndarray, decimals: int) -> list[list[float]]:
    """A 2-D array as nested lists, rounded to keep the response small."""
    return np.round(values, decimals).astype(float).tolist()


# ----------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------
def build_simulator(request: SimulateRequest) -> VoltageSimulator:
    """Assemble the library objects a request describes.

    Parameters
    ----------
    request : SimulateRequest
        A validated request.

    Returns
    -------
    VoltageSimulator
        Phase-centred on the zenith of the array at the start of the
        observation, which is what makes the neglected ``w`` term
        vanish for a flat array and keeps a default run warning-free.

    Notes
    -----
    Separated from `run_simulation` on purpose: a test can build the same
    simulator, correlate it itself, and compare with what the response
    claims, rather than trusting the response to check itself.
    """
    site = default_array()
    array = ArrayConfig(
        antenna_positions_enu_m=np.asarray(request.antennas, dtype=np.float64),
        latitude_deg=site.latitude_deg,
        longitude_deg=site.longitude_deg,
        height_m=site.height_m,
        name=site.name,
    )
    start_time = Time(START_TIME_UTC, scale="utc")
    phase_center = zenith_coord(earth_location(array), start_time)

    sources = [
        PointSource.from_lm(phase_center, (source.l, source.m), source.flux_jy, name=source.name)
        for source in request.sky_sources
    ]
    rfi_sources = [source.build() for source in request.rfi_sources]
    spectral_lines = [line.build() for line in request.spectral_lines]

    n_antennas = len(request.antennas)
    # All the realism models below are re-seeded from the run's own seed:
    # there is no separate seed field per feature, so switching one on
    # never disturbs another, and re-running with the same seed is still
    # byte-identical (the same guarantee `sim.seed` already gives the sky,
    # noise and RFI draws).
    instrument = (
        None
        if request.instrument is None
        else request.instrument.build(n_antennas, request.sim.seed)
    )
    channelizer = None if request.channelizer is None else request.channelizer.build()
    primary_beam = None if request.primary_beam is None else request.primary_beam.build()

    quantization = None if request.quantization is None else "int4"
    quant_kwargs: dict[str, Any] = {}
    if request.quantization is not None:
        quant_kwargs["quant_target_counts"] = request.quantization.quant_target_counts
        if request.quantization.quant_scale is not None:
            quant_kwargs["quant_scale"] = request.quantization.quant_scale

    return VoltageSimulator(
        array,
        phase_center,
        start_time,
        sources,
        rfi_sources=rfi_sources,
        spectral_lines=spectral_lines,
        center_freq_hz=request.sim.center_freq_hz,
        n_chan=request.sim.n_chan,
        n_time_per_block=N_TIME_PER_BLOCK,
        n_blocks=request.sim.n_blocks,
        noise_std=request.sim.noise_std,
        n_pol=request.n_pol,
        channelizer=channelizer,
        instrument=instrument,
        quantization=quantization,
        primary_beam=primary_beam,
        rng=np.random.default_rng(request.sim.seed),
        **quant_kwargs,
    )


class _WaterfallReducer:
    """Reduce blocks onto the display grid as they stream past.

    A block at the largest sizes this front end allows is over a hundred
    megabytes, so nothing keeps them: `stream` hands each block to the
    correlator and folds its power and its interference masks into the
    display grid on the way through, leaving one block's worth of voltages
    alive at a time rather than the whole observation's.
    """

    def __init__(self, simulator: VoltageSimulator, pol: int = 0) -> None:
        self._simulator = simulator
        self._pol = pol
        self.chan_bins, self.time_bins_per_block = _waterfall_shape(
            simulator.n_antennas, simulator.n_chan, simulator.n_blocks
        )
        self._power_columns: list[np.ndarray] = []
        self._mask_columns: list[np.ndarray] = []
        self._occupied_cells = np.zeros(len(simulator.rfi_sources), dtype=np.int64)

    @property
    def time_samples_per_cell(self) -> int:
        """int: Voltage time samples pooled into one displayed column."""
        return -(-N_TIME_PER_BLOCK // self.time_bins_per_block)

    def stream(self) -> Iterator[Any]:
        """Yield every block of the observation, reducing it in passing."""
        for block in self._simulator.blocks():
            self._absorb(block)
            yield block

    def _absorb(self, block: Any) -> None:
        # `pol_data` always carries the polarization axis (length 1 for a
        # single-polarization run), so selecting `self._pol` here works
        # identically whether or not the simulator was built with n_pol=2;
        # the waterfall always shows exactly one receptor.
        data = block.pol_data[:, self._pol]
        power = data.real.astype(np.float64) ** 2 + data.imag.astype(np.float64) ** 2
        power = bin_mean(power, axis=2, n_bins=self.time_bins_per_block)
        power = bin_mean(power, axis=1, n_bins=self.chan_bins)
        self._power_columns.append(power)

        self._occupied_cells += block.rfi_mask.sum(axis=(1, 2))
        mask = bin_any(block.rfi_mask, axis=2, n_bins=self.time_bins_per_block)
        mask = bin_any(mask, axis=1, n_bins=self.chan_bins)
        self._mask_columns.append(mask)

    def reduced(self) -> dict[str, Any]:
        """The pooled power, masks, occupancies and axes of the whole run."""
        simulator = self._simulator
        # (n_ant, chan_bins, n_blocks * time_bins_per_block)
        waterfall = np.concatenate(self._power_columns, axis=2)
        masks = np.concatenate(self._mask_columns, axis=2)

        total_cells = simulator.n_blocks * simulator.n_chan * simulator.n_time_per_block
        occupancy = self._occupied_cells / float(total_cells)

        freq_hz = bin_mean(simulator.freq_hz, axis=0, n_bins=self.chan_bins)
        n_columns = waterfall.shape[2]
        column_duration_s = simulator.duration_s / n_columns
        time_s = (np.arange(n_columns) + 0.5) * column_duration_s

        return {
            "waterfall": waterfall,
            "masks": masks,
            "occupancy": occupancy,
            "freq_hz": freq_hz,
            "time_s": time_s,
            "time_samples_per_cell": self.time_samples_per_cell,
        }


def _to_decibels(power: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Convert power to decibels and choose a colour range that shows structure.

    Parameters
    ----------
    power : numpy.ndarray
        Mean power per displayed cell, Jy.

    Returns
    -------
    decibels : numpy.ndarray
        ``10 log10(P)`` with `P` clamped `DYNAMIC_RANGE_DB` below the
        observation peak, so an empty channel cannot reach minus infinity.
    low, high : float
        Colour-scale ends, taken from `DISPLAY_PERCENTILES` rather than
        from the extremes. A single bright interference cell would
        otherwise push every other cell into the top decibel of the ramp
        and the map would read as blank; clipping the loudest half-percent
        keeps the receiver noise floor legible, which is the background
        an excision algorithm has to work against.
    peak_db : float
        The true maximum, reported so the clipping is visible rather than
        silent.
    """
    peak = float(power.max()) if power.size else 0.0
    if not np.isfinite(peak) or peak <= 0.0:
        zeros = np.zeros(power.shape, dtype=np.float64)
        return zeros, 0.0, 1.0, 0.0
    floor = peak * 10.0 ** (-DYNAMIC_RANGE_DB / 10.0)
    decibels = 10.0 * np.log10(np.maximum(power, floor))
    peak_db = 10.0 * math.log10(peak)
    low, high = (float(x) for x in np.percentile(decibels, DISPLAY_PERCENTILES))
    if high - low < 1.0:
        low, high = peak_db - 1.0, peak_db + 1.0
    return decibels, low, high, peak_db


def run_simulation(request: SimulateRequest, *, pol: int = 0) -> dict[str, Any]:
    """Run one observation and reduce it to what a browser can draw.

    Parameters
    ----------
    request : SimulateRequest
        A validated request.
    pol : int, optional
        Which receptor the waterfall display shows, ``0`` or ``1``.
        Meaningless (and harmless) for a single-polarization run, which
        has only receptor ``0``. Default 0. The dirty image is unaffected
        by this: it always images Stokes I (see `dirty_image`'s own
        ``pol=None`` default), so a polarization comparison belongs in the
        waterfall, not the image.

    Returns
    -------
    dict
        JSON-ready: ``waterfall`` (per-antenna power in decibels on a
        shared grid), ``sources`` (one pooled ground-truth mask and one
        exact occupancy fraction each), ``image`` (dirty image and its
        peak), ``uv`` (baseline coordinates in wavelengths), any
        ``warnings`` the library raised, and the wall time. The waterfall
        also reports ``time_samples_per_cell``, the number of voltage
        samples pooled into one displayed column, so the page can say how
        coarse the picture it is drawing really is.

    Raises
    ------
    ValueError
        Straight from the library, e.g. when a transmitter is tuned
        outside the simulated band. Callers turn it into a form error.
    """
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        simulator = build_simulator(request)
        pol = int(pol) if simulator.n_pol > 1 else 0
        reducer = _WaterfallReducer(simulator, pol=pol)
        calibration_errors = (
            None
            if request.calibration_errors is None
            else request.calibration_errors.build(simulator.n_antennas, request.sim.seed)
        )
        visibilities = correlate(reducer.stream(), calibration_errors=calibration_errors)
        reduced = reducer.reduced()

        channel_step = max(1, simulator.n_chan // IMAGE_MAX_CHANNELS)
        image, l_grid, m_grid = dirty_image(
            visibilities,
            field_of_view_rad=IMAGE_FIELD_OF_VIEW_RAD,
            n_pix=IMAGE_N_PIX,
            channels=slice(None, None, channel_step),
        )
        u_lambda, v_lambda, _ = uvw_wavelengths(visibilities)
    wall_time_s = time.perf_counter() - started

    decibels, vmin_db, vmax_db, peak_db = _to_decibels(reduced["waterfall"])

    peak_index = int(np.argmax(image))
    peak_row, peak_col = np.unravel_index(peak_index, image.shape)

    center_channel = visibilities.n_chan // 2
    cross = visibilities.cross_mask
    u_points = u_lambda[:, cross, center_channel].ravel()
    v_points = v_lambda[:, cross, center_channel].ravel()

    messages = []
    for entry in caught:
        text = str(entry.message)
        if text not in messages:
            messages.append(text)

    return {
        "waterfall": {
            "antennas": [_round_grid(plane, 2) for plane in decibels],
            "freq_mhz": (reduced["freq_hz"] / 1.0e6).round(6).tolist(),
            "time_s": reduced["time_s"].round(6).tolist(),
            "vmin_db": round(vmin_db, 3),
            "vmax_db": round(vmax_db, 3),
            "peak_db": round(peak_db, 3),
            "dynamic_range_db": DYNAMIC_RANGE_DB,
            "time_samples_per_cell": int(reduced["time_samples_per_cell"]),
            "unit": "dB (Jy per cell)",
        },
        "sources": [
            {
                "name": source.name,
                "type": params.type,
                "occupancy": float(reduced["occupancy"][index]),
                "mask": reduced["masks"][index].astype(np.uint8).tolist(),
            }
            for index, (params, source) in enumerate(
                zip(request.rfi_sources, simulator.rfi_sources)
            )
        ],
        "image": {
            "values": _round_grid(image, 6),
            "l": l_grid.round(8).tolist(),
            "m": m_grid.round(8).tolist(),
            "vmin_jy": float(np.round(image.min(), 6)),
            "vmax_jy": float(np.round(image.max(), 6)),
            "peak": {
                "l": float(l_grid[peak_col]),
                "m": float(m_grid[peak_row]),
                "value_jy": float(image[peak_row, peak_col]),
            },
            "unit": "Jy per beam",
        },
        "uv": {
            "u": u_points.round(3).tolist(),
            "v": v_points.round(3).tolist(),
            "max_lambda": float(
                np.round(np.sqrt(u_points**2 + v_points**2).max(), 3) if u_points.size else 0.0
            ),
        },
        "observation": {
            "n_antennas": simulator.n_antennas,
            "n_baselines": int(np.count_nonzero(cross)),
            "n_chan": simulator.n_chan,
            "n_blocks": simulator.n_blocks,
            "duration_s": simulator.duration_s,
            "bandwidth_hz": simulator.bandwidth_hz,
            "center_freq_hz": simulator.center_freq_hz,
            "start_time_utc": START_TIME_UTC,
            "seed": request.sim.seed,
            "n_pol": simulator.n_pol,
            "pol_names": list(visibilities.pol_names),
            "waterfall_pol": pol,
        },
        "warnings": messages,
        "wall_time_s": round(wall_time_s, 3),
    }
