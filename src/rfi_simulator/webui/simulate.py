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

**Three levels, one run.** A response describes the same observation at
the three places an excision method can work: the per-antenna voltages
(`_WaterfallReducer`), the correlated visibilities
(`_visibility_payload`) and the dirty image. Each level is reduced under
its own budget, and each carries the ground truth *of that level* --
occupancy masks at voltage resolution, ``rfi_fraction`` at the
integration grid -- rather than one level's truth redrawn on another's
axes. `run_flaggers` adds a fourth thing to compare against: what the
classical methods actually catch.

**Units at the boundary.** The library speaks hertz, metres, seconds and
janskys, and so does this API. Field descriptors in `defaults_payload`
carry a display unit and a multiplier so the browser can show megahertz
without either side inventing a second convention.
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

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
from rfi_simulator.channelizer import (
    DEFAULT_N_TAPS,
    DEFAULT_SINC_BANDWIDTH,
    DEFAULT_WINDOW,
)
from rfi_simulator.delays import earth_location, zenith_coord
from rfi_simulator.flaggers import mad_clip_mask, spectral_kurtosis_mask, sumthreshold_mask
from rfi_simulator.metrics import flag_scores, pool_truth_accumulations
from rfi_simulator.sky import lm_from_radec
from rfi_simulator.voltages import DEFAULT_CHAN_WIDTH_HZ, DEFAULT_QUANT_TARGET_COUNTS

_log = logging.getLogger(__name__)

_SLUG_STRIP_RE = re.compile(r"[^a-z0-9]+")

ARRAY_DIR_ENV_VAR = "RFI_SIMULATOR_ARRAY_DIR"
"""str: Environment variable naming a directory of extra array
configurations, offered alongside the bundled ones. `main` sets it from
``--array-dir`` so the value survives into the process the reloader
starts."""

__all__ = [
    "ARRAY_DIR_ENV_VAR",
    "DEFAULT_CENTER_FREQ_HZ",
    "DEFAULT_N_BLOCKS",
    "DEFAULT_N_CHAN",
    "FLAG_DOMAINS",
    "FLAG_METHODS",
    "FlagRequest",
    "MAX_ANTENNAS",
    "MAX_COORDINATE_M",
    "MAX_N_BLOCKS",
    "MAX_N_CHAN",
    "MAX_RFI_SOURCES",
    "MAX_SKY_SOURCES",
    "MAX_TOTAL_SAMPLES",
    "START_TIME_UTC",
    "SimulateRequest",
    "array_catalogue",
    "array_detail",
    "array_summaries",
    "build_simulator",
    "default_array",
    "defaults_payload",
    "pointing_payload",
    "run_flaggers",
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

MAX_ANTENNAS = 128
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

MAX_TOTAL_SAMPLES = 100_000_000
"""int: Most voltage samples one run may generate.

The count is ``n_antennas * n_chan * n_blocks * N_TIME_PER_BLOCK``. Each
of the individual caps is modest on its own, but their product is not:
taking every one of them at once would allocate tens of gigabytes and run
for many minutes. This budget leaves room for the largest setups the page
offers in any one direction -- a hundred-element array at the default
width, or a 512-channel band on a handful of antennas -- while keeping the
memory a run holds at once (one block of voltages, not the whole
observation) to something a laptop has. The largest runs it allows take
tens of seconds rather than the few seconds a default run takes, which is
what the page warns about when a large array is loaded."""

MAX_BINS = 256
"""int: Most cells the browser is ever sent along one axis of a waterfall."""

MAX_WATERFALL_CELLS = 400_000
"""int: Budget for all antennas' waterfalls together. The time axis is
thinned until the whole response fits, so a many-antenna run stays a
few megabytes rather than a few tens."""

DYNAMIC_RANGE_DB = 60.0
"""float: Decibels below the observation peak at which power is clamped."""

VIS_MAX_CHAN_BINS = 256
"""int: Channel bins in the baseline-averaged visibility waterfall.

The integration axis needs no budget of its own: there are at most
`MAX_N_BLOCKS` integrations in a run, so the whole map is a few thousand
numbers however wide the band is."""

VIS_SPECTRA_MAX_CHAN_BINS = 128
"""int: Channel bins in the per-baseline amplitude/phase spectra."""

MIN_SPECTRA_BASELINES = 16
"""int: Baselines the per-baseline spectra always offer, however large the
array is. Below this the picker stops being a picker."""

MAX_VIS_SPECTRUM_VALUES = 200_000
"""int: Budget for the per-baseline spectra, counting amplitudes and
phases together.

One baseline of a default run is 8 integrations x 128 channels x 2
quantities, so the default array's 45 baselines all fit with room to
spare. A large array does not: `_visibility_spectra` first offers a
subset of baselines evenly spaced through the uv-distance ordering, and
only if that is still not enough averages integrations together. Either
reduction is reported alongside the numbers, so the page can print what
it is drawing."""

FLAG_DEFAULT_M = 250
"""int: Default accumulation length of the flagger endpoint, in voltage
time samples. Divides `N_TIME_PER_BLOCK`, so accumulations never straddle
a block boundary and four of them fit in each block."""

MAX_FLAG_CELLS = 4_000_000
"""int: Largest ``n_chan x n_accumulations`` grid a flagging run may
build. Reached only by asking for a short accumulation on a wide band;
the message says which knob to turn."""

FLAG_OVERLAY_MAX_CHAN_BINS = 256
"""int: Channel bins the flagger overlay is pooled onto before it is sent.
The *scores* are always computed on the native grid -- pooling a mask can
only make it look better -- so this affects the picture and nothing else.
"""

DISPLAY_PERCENTILES = (0.5, 99.5)
"""tuple of float: Percentiles of the decibel map used as the colour-scale
ends. Clipping the loudest half-percent is what keeps a single bright
interference cell from flattening the whole display."""

IMAGE_N_PIX = 64
IMAGE_FIELD_OF_VIEW_RAD = 0.04
IMAGE_MAX_CHANNELS = 64
"""int: Channels the direct-DFT image uses. Above this the channels are
evenly subsampled, which costs sensitivity but does not move sources."""

IMAGE_FIELD_HALF_WIDTH_DEG = math.degrees(math.asin(0.5 * IMAGE_FIELD_OF_VIEW_RAD))
"""float: Angular half-width of the imaged field, degrees.

`IMAGE_FIELD_OF_VIEW_RAD` is the *full* width of the direction-cosine grid
(`rfi_simulator.imaging.lm_axis`), so its half is the largest ``l`` or
``m`` the image covers, and the angle that corresponds to is its arcsine.
The two differ by a part in ten thousand at this size; the arcsine is used
anyway so that the number the page quotes is the same angle the source
placement uses."""

DEFAULT_OFFSET_EAST_DEG = 0.5
DEFAULT_OFFSET_NORTH_DEG = -0.3
"""float: Where a freshly added sky source sits, degrees east and north of
the pointing centre. Comfortably inside `IMAGE_FIELD_HALF_WIDTH_DEG`, and
off-centre in both axes so a mirrored sign convention would be visible in
the image rather than hidden by symmetry."""

MAX_OFFSET_DEG = 30.0
"""float: Largest tangent-plane offset a source may be given, degrees.

``sin(30 deg)`` is 0.5, which is the same bound `MAX_LM` puts on the
direction cosines themselves."""

MAX_LM = 0.5
"""float: Largest direction cosine a source may be given."""


# ----------------------------------------------------------------------
# Packaged inputs
# ----------------------------------------------------------------------
def _config_dir() -> Path | None:
    """The repository's ``configs`` directory, if this is a checkout.

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
        candidate = parent / "configs"
        if candidate.is_dir():
            return candidate
    return None


def _config_path(filename: str) -> Path | None:
    """Locate a file in the repository's ``configs`` directory, if present."""
    directory = _config_dir()
    if directory is None:
        return None
    candidate = directory / filename
    return candidate if candidate.is_file() else None


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


def _slugify(text: str) -> str:
    """A short identifier made only of letters, digits and hyphens."""
    slug = _SLUG_STRIP_RE.sub("-", text.lower()).strip("-")
    return slug[:60] or "array"


def _array_directories(extra_dir: str | Path | None = None) -> list[Path]:
    """Directories scanned for array configurations, in listing order.

    The bundled ``configs`` directory comes first so the default array is
    always the first entry; `extra_dir` (the ``--array-dir`` flag, or
    `ARRAY_DIR_ENV_VAR` in the environment) is appended when it is set and
    exists. Nothing else on the filesystem is ever read: the operator names
    one directory, and only that directory is offered.
    """
    directories: list[Path] = []
    bundled = _config_dir()
    if bundled is not None:
        directories.append(bundled)
    if extra_dir is None:
        extra_dir = os.environ.get(ARRAY_DIR_ENV_VAR) or None
    if extra_dir:
        extra = Path(extra_dir).expanduser()
        if extra.is_dir() and extra.resolve() not in {d.resolve() for d in directories}:
            directories.append(extra)
    return directories


def array_catalogue(extra_dir: str | Path | None = None) -> list[tuple[str, ArrayConfig]]:
    """Every array configuration this server can offer.

    Parameters
    ----------
    extra_dir : str or pathlib.Path, optional
        A second directory to scan, in addition to the bundled one.
        Defaults to `ARRAY_DIR_ENV_VAR` in the environment.

    Returns
    -------
    list of (str, ArrayConfig)
        Identifier and loaded configuration, in listing order. The
        identifier is derived from the file name *here*, on the server; a
        client never names a path, so no request can point this at a file
        the operator did not offer. A YAML that `ArrayConfig.from_yaml`
        refuses -- anything from an unrelated YAML file sharing the
        directory to a malformed array -- is skipped without comment
        beyond a debug log line.
    """
    entries: list[tuple[str, ArrayConfig]] = []
    taken: set[str] = set()
    for directory in _array_directories(extra_dir):
        paths = sorted(path for path in directory.iterdir() if path.suffix in {".yaml", ".yml"})
        for path in paths:
            try:
                array = ArrayConfig.from_yaml(path)
            except Exception:  # noqa: BLE001 - any unreadable file is simply not offered
                _log.debug("not offering %s: it is not a readable array configuration", path)
                continue
            identifier = _slugify(path.stem)
            if identifier in taken:
                suffix = 2
                while f"{identifier}-{suffix}" in taken:
                    suffix += 1
                identifier = f"{identifier}-{suffix}"
            taken.add(identifier)
            entries.append((identifier, array))
    return entries


def _array_payload(identifier: str, array: ArrayConfig) -> dict[str, Any]:
    """One array in the shape the page's array section holds it."""
    positions = np.asarray(array.antenna_positions_enu_m, dtype=np.float64)
    return {
        "id": identifier,
        "name": array.name or identifier,
        "n_antennas": int(positions.shape[0]),
        "latitude_deg": float(array.latitude_deg),
        "longitude_deg": float(array.longitude_deg),
        "height_m": float(array.height_m),
        "antennas": [[float(value) for value in row] for row in positions],
        "runnable": int(positions.shape[0]) <= MAX_ANTENNAS,
    }


def array_summaries(extra_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """The catalogue without the antenna positions, for the dropdown."""
    return [
        {
            key: value
            for key, value in _array_payload(identifier, array).items()
            if key != "antennas"
        }
        for identifier, array in array_catalogue(extra_dir)
    ]


def array_detail(identifier: str, extra_dir: str | Path | None = None) -> dict[str, Any] | None:
    """One catalogue entry in full, or ``None`` if there is no such id."""
    for candidate, array in array_catalogue(extra_dir):
        if candidate == identifier:
            return _array_payload(candidate, array)
    return None


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
    _num("flux_jy", "Flux density", 5.0, minimum=0.0, maximum=1.0e4, step=0.5, unit="Jy"),
]
"""list of dict: The schema-driven part of a sky source's form.

Position is deliberately not in here: it is one quantity expressed three
different ways (see `SkySource`), which the one-control-per-field form
builder cannot render, so the page draws a small unit switcher for it by
hand and describes the three modes from `SKY_POSITION_MODES`."""

SKY_POSITION_MODES = [
    {
        "value": "offset",
        "label": "Offset from pointing (degrees E/N)",
        "fields": ["east_deg", "north_deg"],
        "unit": "deg",
        "step": 0.05,
        "limit": MAX_OFFSET_DEG,
        "labels": ["East of pointing", "North of pointing"],
    },
    {
        "value": "radec",
        "label": "Right ascension / declination (degrees)",
        "fields": ["ra_deg", "dec_deg"],
        "unit": "deg",
        "step": 0.01,
        "limit": 360.0,
        "labels": ["Right ascension (ICRS)", "Declination (ICRS)"],
    },
    {
        "value": "lm",
        "label": "Direction cosines l/m (advanced)",
        "fields": ["l", "m"],
        "unit": "direction cosine",
        "step": 0.001,
        "limit": MAX_LM,
        "labels": ["l (east)", "m (north)"],
    },
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
        "distance_m",
        "Range",
        3000.0,
        unit="km",
        factor=1000.0,
        minimum=10.0,
        maximum=5.0e5,
        step=0.1,
    ),
    _num(
        "fundamental_hz",
        "Fundamental frequency",
        1.405e6,
        unit="MHz",
        factor=MHZ,
        minimum=1.0e3,
        maximum=1.0e10,
        step=0.000001,
        help_text="May sit far below the simulated band; only in-band harmonics show up.",
    ),
    _text(
        "harmonic_numbers",
        "Harmonics (comma-separated)",
        "999,1000,1001",
        help_text="Which multiples of the fundamental the device emits, e.g. '999,1000,1001'.",
    ),
    _num(
        "received_power_jy",
        "Received power per harmonic",
        200.0,
        unit="Jy",
        minimum=0.0,
        maximum=1.0e9,
        step=10.0,
    ),
    _num(
        "bandwidth_hz",
        "Bandwidth per harmonic",
        0.0,
        unit="kHz",
        factor=KHZ,
        minimum=0.0,
        maximum=1.0e9,
        step=1.0,
        help_text="0 (default) makes every harmonic a pure line, one channel wide.",
    ),
    _num("duty_cycle", "Duty cycle", 1.0, minimum=0.0, maximum=1.0, step=0.05),
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


def _check_finite_floats(values: list[float]) -> list[float]:
    """Reject NaN/inf, the same way the antenna-position validator does."""
    if not all(math.isfinite(value) for value in values):
        raise ValueError("values must all be finite numbers")
    return values


FiniteFloatList = Annotated[list[float], AfterValidator(_check_finite_floats)]
"""A `list[float]` that has already been checked for NaN/inf."""

CouplingSpec = FiniteFloatList | LognormalCoupling
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
    jones_re: FiniteFloatList = Field(min_length=2, max_length=2)
    jones_im: FiniteFloatList = Field(default=[0.0, 0.0], min_length=2, max_length=2)
    fraction: float = Field(default=1.0, ge=0.0, le=1.0)


PolarizationSpec = Annotated[LinearPolarization | FullPolarization, Field(discriminator="type")]


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
def _pair(name: str, value: list[float] | None, limit: float) -> list[float] | None:
    """Validate a two-number position spec: finite, in range, exactly two."""
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError(f"{name} must be two numbers, got {len(value)}")
    for number in value:
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite numbers, got {value}")
        if abs(number) > limit:
            raise ValueError(f"{name} must lie between -{limit:g} and {limit:g}, got {number}")
    return [float(number) for number in value]


class SkySource(BaseModel):
    """One celestial point source and where on the sky it sits.

    The position may be given in any one of three ways, and giving more
    than one is an error rather than a silent precedence rule:

    ``offset_deg``
        ``[east_deg, north_deg]`` from the pointing centre, as a rigorous
        tangent-plane offset: `~astropy.coordinates.SkyCoord.spherical_offsets_by`
        moves the phase centre by those two angles, and the result is
        projected to ``(l, m)`` by the same `rfi_simulator.sky.lm_from_radec`
        `radec_deg` uses. This is *not* the independent-sine shortcut
        (``l = sin(east_deg)``, ``m = sin(north_deg)``) -- that shortcut is
        exact only along a single axis (one of the two offsets zero); with
        both nonzero it and the rigorous answer diverge at a nonzero
        declination, by an amount of order ``east * north * tan(dec)``.
        Going through the same projection `radec_deg` uses means the two
        notations agree exactly by construction, not approximately.
    ``radec_deg``
        ``[ra_deg, dec_deg]``, absolute ICRS, projected onto the same
        tangent plane by `rfi_simulator.sky.lm_from_radec` -- the
        library's own forward SIN projection, which is the exact inverse
        of the `PointSource.from_lm` used to build the source.
    ``l``/``m``
        Direction cosines, the library's native coordinates: ``l``
        increases east (towards increasing right ascension), ``m`` north
        (towards increasing declination).

    With none of them given the source lands at
    (`DEFAULT_OFFSET_EAST_DEG`, `DEFAULT_OFFSET_NORTH_DEG`).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="source", max_length=40)
    l: float | None = Field(default=None, ge=-MAX_LM, le=MAX_LM)  # noqa: E741 - standard symbol
    m: float | None = Field(default=None, ge=-MAX_LM, le=MAX_LM)
    offset_deg: list[float] | None = None
    radec_deg: list[float] | None = None
    flux_jy: float = Field(default=5.0, ge=0.0, le=1.0e4)

    @field_validator("offset_deg")
    @classmethod
    def _check_offset(cls, value: list[float] | None) -> list[float] | None:
        return _pair("offset_deg", value, MAX_OFFSET_DEG)

    @field_validator("radec_deg")
    @classmethod
    def _check_radec(cls, value: list[float] | None) -> list[float] | None:
        value = _pair("radec_deg", value, 360.0)
        if value is not None and abs(value[1]) > 90.0:
            raise ValueError(f"radec_deg declination must lie in [-90, 90], got {value[1]}")
        return value

    @model_validator(mode="after")
    def _check_one_position(self) -> "SkySource":
        given = []
        if self.l is not None or self.m is not None:
            if self.l is None or self.m is None:
                raise ValueError("give both l and m, or neither")
            given.append("l/m")
        if self.offset_deg is not None:
            given.append("offset_deg")
        if self.radec_deg is not None:
            given.append("radec_deg")
        if len(given) > 1:
            raise ValueError(
                "give this source one position only, not " + " and ".join(given) + " together"
            )
        if self.l is not None and self.l**2 + self.m**2 >= 1.0:
            raise ValueError("l and m place this source off the sky: l^2 + m^2 must be below 1")
        return self

    def resolve_lm(self, phase_center: SkyCoord) -> tuple[float, float]:
        """Direction cosines of this source relative to `phase_center`.

        Parameters
        ----------
        phase_center : astropy.coordinates.SkyCoord
            Where the run points; only `radec_deg` sources depend on it.

        Returns
        -------
        tuple of float
            ``(l, m)``, whichever way the position was given.
        """
        if self.l is not None and self.m is not None:
            return (float(self.l), float(self.m))
        if self.radec_deg is not None:
            coord = SkyCoord(
                ra=self.radec_deg[0] * u.deg, dec=self.radec_deg[1] * u.deg, frame="icrs"
            )
            l_dir, m_dir = lm_from_radec(phase_center, coord)
            return (float(l_dir), float(m_dir))
        east_deg, north_deg = self.offset_deg or (
            DEFAULT_OFFSET_EAST_DEG,
            DEFAULT_OFFSET_NORTH_DEG,
        )
        # Rigorous, and deliberately routed through the same projection
        # `radec_deg` uses: an offset applied independently on each axis
        # (`l = sin(east)`, `m = sin(north)`) is only exact when one of the
        # two is zero, and quietly diverges from the true tangent-plane
        # position once both are nonzero at a nonzero declination. Going
        # through an actual sky position first is what makes the two
        # notations agree exactly rather than approximately.
        offset_coord = phase_center.spherical_offsets_by(east_deg * u.deg, north_deg * u.deg)
        l_dir, m_dir = lm_from_radec(phase_center, offset_coord)
        return (float(l_dir), float(m_dir))

    def build(self, phase_center: SkyCoord) -> PointSource:
        """This source as a library `PointSource`.

        A source given in right ascension and declination is built at that
        coordinate directly rather than round-tripped through ``(l, m)``:
        the projection is many-to-one over the whole sphere, so a position
        far outside the field would otherwise come back mirrored onto the
        near hemisphere instead of simply being absent from the image.
        """
        if self.radec_deg is not None:
            return PointSource(
                flux_jy=self.flux_jy,
                coord=SkyCoord(
                    ra=self.radec_deg[0] * u.deg, dec=self.radec_deg[1] * u.deg, frame="icrs"
                ),
                name=self.name,
            )
        return PointSource.from_lm(
            phase_center, self.resolve_lm(phase_center), self.flux_jy, name=self.name
        )


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
        "fwhm_hz",
        "Line width (FWHM)",
        20.0e3,
        unit="kHz",
        factor=KHZ,
        minimum=1.0,
        maximum=1.0e9,
        step=1.0,
    ),
    _num(
        "line_flux_jy",
        "Peak-channel power",
        1.0,
        unit="Jy",
        minimum=0.0,
        maximum=1.0e6,
        step=0.1,
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

    def build(self, n_antennas: int, rng: np.random.Generator) -> InstrumentModel:
        """The library model this describes.

        `freq_hz` is deliberately omitted: leaving `band_hz` unset makes
        the ripple/slope/subband reference band default to whatever grid
        `InstrumentModel.gains` is later evaluated on, which is exactly
        the simulated band here -- see `InstrumentModel.from_params`.

        `rng` must already be independent of any other model's generator
        (see `build_simulator`'s seed derivation) -- passing the same
        seed both here and to `CalibrationErrorParams.build` would draw
        the same standard-normal vector for the gain scatter and the
        calibration phase error, perfectly correlating two effects the
        rest of the simulator treats as independent.
        """
        return InstrumentModel.from_params(n_antennas, rng=rng, **self.model_dump())


class CalibrationErrorParams(BaseModel):
    """Residual calibration error (see `CalibrationErrors`), applied at `correlate`."""

    model_config = ConfigDict(extra="forbid")

    phase_error_deg_rms: float = Field(default=5.0, ge=0.0, le=180.0)
    delay_error_ns_rms: float = Field(default=0.0, ge=0.0, le=10.0)
    amplitude_error_db_rms: float = Field(default=0.0, ge=0.0, le=10.0)

    def build(self, n_antennas: int, rng: np.random.Generator) -> CalibrationErrors:
        """The library model this describes.

        `rng` must already be independent of any other model's generator
        -- see `InstrumentParams.build`.
        """
        return CalibrationErrors.from_params(n_antennas, rng=rng, **self.model_dump())


class ChannelizerParams(BaseModel):
    """Polyphase-filterbank channel response (see `PFBChannelizer`)."""

    model_config = ConfigDict(extra="forbid")

    n_taps: int = Field(default=DEFAULT_N_TAPS, ge=1, le=32)
    window: Literal["hann", "hamming", "blackman"] = DEFAULT_WINDOW
    sinc_bandwidth: float = Field(default=DEFAULT_SINC_BANDWIDTH, gt=0.0, le=8.0)

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


class SiteParams(BaseModel):
    """Where on Earth the array origin stands.

    Optional: a request that leaves it out observes from the default
    array's site. It matters because the phase centre is the zenith of
    *this* point at the start of the observation, so loading another
    array's antennas without its site would point the telescope somewhere
    that array never looks.
    """

    model_config = ConfigDict(extra="forbid")

    latitude_deg: float = Field(default=0.0, ge=-90.0, le=90.0)
    longitude_deg: float = Field(default=0.0, ge=-360.0, le=360.0)
    height_m: float = Field(default=0.0, ge=-500.0, le=1.0e4)


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
    array origin, which is `site` when it is given and the default
    array's site otherwise.
    """

    model_config = ConfigDict(extra="forbid")

    antennas: list[list[float]] = Field(default_factory=list, max_length=MAX_ANTENNAS)
    site: SiteParams | None = None
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
def phase_center_for_site(latitude_deg: float, longitude_deg: float, height_m: float) -> SkyCoord:
    """Where a run from this site points: the zenith at `START_TIME_UTC`.

    Built through the library's own `earth_location`/`zenith_coord` pair,
    on a throwaway two-antenna array, so that the coordinate the page
    quotes and the coordinate `build_simulator` fringe-stops on cannot
    drift apart.
    """
    array = ArrayConfig(
        antenna_positions_enu_m=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        height_m=height_m,
    )
    return zenith_coord(earth_location(array), Time(START_TIME_UTC, scale="utc"))


def pointing_payload(
    latitude_deg: float | None = None,
    longitude_deg: float | None = None,
    height_m: float | None = None,
) -> dict[str, Any]:
    """Where the telescope points, and how far out the image reaches.

    Parameters
    ----------
    latitude_deg, longitude_deg, height_m : float, optional
        The site to answer for. Any left out is taken from the default
        array, so the no-argument call describes the run the page opens
        with.

    Returns
    -------
    dict
        ``ra_deg``/``dec_deg`` of the phase centre in ICRS, the site it
        was computed for, the start time, and ``field_half_width_deg`` --
        how far from the pointing a source can sit and still land inside
        the dirty image.
    """
    site = default_array()
    latitude = site.latitude_deg if latitude_deg is None else float(latitude_deg)
    longitude = site.longitude_deg if longitude_deg is None else float(longitude_deg)
    height = site.height_m if height_m is None else float(height_m)
    center = phase_center_for_site(latitude, longitude, height)
    return {
        "ra_deg": float(center.ra.deg),
        "dec_deg": float(center.dec.deg),
        "start_time_utc": START_TIME_UTC,
        "latitude_deg": latitude,
        "longitude_deg": longitude,
        "height_m": height,
        "field_half_width_deg": IMAGE_FIELD_HALF_WIDTH_DEG,
        "field_of_view_rad": IMAGE_FIELD_OF_VIEW_RAD,
        "image_n_pix": IMAGE_N_PIX,
        "tracking": True,
    }


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
            "position": {
                "modes": SKY_POSITION_MODES,
                "default_mode": "offset",
                "default_offset_deg": [DEFAULT_OFFSET_EAST_DEG, DEFAULT_OFFSET_NORTH_DEG],
            },
        },
        "pointing": pointing_payload(),
        "spectral_line": {
            "label": "Spectral line",
            "fields": SPECTRAL_LINE_FIELDS,
            "defaults": _schema_defaults(SPECTRAL_LINE_FIELDS),
        },
        "rfi_types": [
            dict(entry, defaults=_schema_defaults(entry["fields"])) for entry in RFI_TYPES
        ],
        "flaggers": {
            "methods": FLAG_METHODS,
            "max_methods": MAX_FLAG_METHODS,
            "domains": FLAG_DOMAINS,
            "defaults": FlagParams().model_dump(),
        },
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


def _finite(value: float) -> float | None:
    """A float the JSON encoder will accept: ``None`` for NaN and infinity.

    `rfi_simulator.metrics.flag_scores` reports an undefined score as NaN
    -- precision when nothing was flagged, recall when the truth holds no
    interference -- and NaN is not representable in JSON, so the response
    would fail to encode rather than fail to answer. ``None`` is the
    honest transport for "this score is not defined for this run"; the
    page prints it as a dash.
    """
    number = float(value)
    return number if math.isfinite(number) else None


def _beam_half_power_rad(beam: GaussianBeam | AiryBeam, freq_hz: float) -> float | None:
    """Angle at which this beam's power response has fallen to one half.

    Found by bisection on the beam's own `power_response` rather than
    from a closed form, so the number drawn on the image is the same
    model the sources were attenuated by whichever beam is fitted. Both
    beams fall monotonically through 0.5 well inside their first null
    (`AiryBeam`'s brightest sidelobe reaches under 2 % of the peak), so
    there is exactly one crossing to find.

    Parameters
    ----------
    beam : GaussianBeam or AiryBeam
        The fitted primary beam.
    freq_hz : float
        Frequency to evaluate at, Hz -- the band centre, since the beam
        narrows across a band.

    Returns
    -------
    float or None
        The half-power radius in radians, or ``None`` for a beam so wide
        that it has not reached half power by the horizon, where there is
        no circle to draw.
    """
    low, high = 0.0, 0.5 * math.pi
    if float(beam.power_response(high, freq_hz)) >= 0.5:
        return None
    for _ in range(60):
        middle = 0.5 * (low + high)
        if float(beam.power_response(middle, freq_hz)) >= 0.5:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _visibility_spectra(visibilities: Any, order: np.ndarray, pol: int) -> dict[str, Any]:
    """Amplitude and phase against frequency, per baseline, per integration.

    Parameters
    ----------
    visibilities : rfi_simulator.correlator.Visibilities
        The correlated observation.
    order : numpy.ndarray
        Indices of the cross-correlation baselines, shortest first. The
        subset that survives the payload budget is taken from this
        ordering, so a reduced response still spans the whole range of
        baseline lengths.
    pol : int
        Receptor to report, matching the waterfall's own choice.

    Returns
    -------
    dict
        ``baselines`` (which entries of `order` are included),
        ``amplitude`` and ``phase_deg`` (baseline, integration, channel),
        the two axes, and how the two reductions were applied.

    Notes
    -----
    Amplitude is the mean of ``|V|`` over the cells of a bin and phase is
    the argument of the mean of ``V``. They are deliberately not taken
    from the same complex average: a bin whose phase turns over its width
    has a small vector mean, which would read as an amplitude null that
    the individual channels do not have.
    """
    data = visibilities.pol_data[:, :, pol, :]
    n_int, _, n_chan = data.shape
    chan_bins = min(n_chan, VIS_SPECTRA_MAX_CHAN_BINS)

    # Baselines are thinned before integrations are averaged together, and
    # only down to `MIN_SPECTRA_BASELINES`: offering fewer antenna pairs
    # costs a reader nothing but choice, while averaging the integrations
    # away costs them the one thing this plot is for -- watching a
    # baseline's phase turn from one integration to the next.
    int_bins = n_int
    selected = order

    def values() -> int:
        return 2 * len(selected) * int_bins * chan_bins

    if values() > MAX_VIS_SPECTRUM_VALUES:
        keep = max(MIN_SPECTRA_BASELINES, MAX_VIS_SPECTRUM_VALUES // (2 * int_bins * chan_bins))
        if len(selected) > keep:
            selected = order[:: -(-len(selected) // keep)]
    while int_bins > 1 and values() > MAX_VIS_SPECTRUM_VALUES:
        int_bins = max(1, int_bins // 2)

    block = data[:, selected, :]
    amplitude = bin_mean(np.abs(block).astype(np.float64), axis=2, n_bins=chan_bins)
    complex_mean = bin_mean(
        block.real.astype(np.float64), axis=2, n_bins=chan_bins
    ) + 1j * bin_mean(block.imag.astype(np.float64), axis=2, n_bins=chan_bins)
    if int_bins != n_int:
        amplitude = bin_mean(amplitude, axis=0, n_bins=int_bins)
        complex_mean = bin_mean(complex_mean.real, axis=0, n_bins=int_bins) + 1j * bin_mean(
            complex_mean.imag, axis=0, n_bins=int_bins
        )

    freq_hz = bin_mean(np.asarray(visibilities.freq_hz, dtype=np.float64), axis=0, n_bins=chan_bins)
    centres = (np.arange(n_int) + 0.5) * visibilities.integration_time_s
    time_s = bin_mean(centres, axis=0, n_bins=int_bins)

    # (integration, baseline, channel) -> (baseline, integration, channel):
    # one entry per baseline is what the page indexes into.
    return {
        "baselines": [int(index) for index in selected],
        "freq_mhz": (freq_hz / 1.0e6).round(6).tolist(),
        "time_s": time_s.round(6).tolist(),
        "amplitude": np.round(np.moveaxis(amplitude, 0, 1), 5).tolist(),
        "phase_deg": np.round(np.degrees(np.angle(np.moveaxis(complex_mean, 0, 1))), 3).tolist(),
        "integrations_per_bin": int(round(n_int / max(1, int_bins))),
        "channels_per_bin": int(round(n_chan / max(1, chan_bins))),
        "n_baselines_offered": int(len(selected)),
    }


def _visibility_payload(visibilities: Any, pol: int) -> dict[str, Any]:
    """What the correlator saw, reduced to what a browser can draw.

    Parameters
    ----------
    visibilities : rfi_simulator.correlator.Visibilities
        The correlated observation.
    pol : int
        Receptor to report -- the same one the voltage waterfall shows,
        so the two panels describe the same signal path.

    Returns
    -------
    dict
        ``amplitude``: ``|V|`` averaged over every cross-correlation
        baseline, on a (channel, integration) grid, which is where a
        visibility-domain flagger works. ``sources``: the library's own
        ``rfi_fraction`` -- the fraction of each integration's samples a
        source occupied -- pooled onto that same grid with an ANY rule,
        plus its exact mean and maximum. ``baselines``: one row per
        antenna pair. ``spectra``: see `_visibility_spectra`.

    Notes
    -----
    Occupancy is a property of a time-frequency cell, not of a baseline:
    interference reaches every antenna, so `rfi_fraction` has no baseline
    axis to summarise and the per-baseline rows carry only amplitudes.
    """
    data = visibilities.pol_data[:, :, pol, :]
    cross = visibilities.cross_mask
    n_int, _, n_chan = data.shape
    chan_bins = min(n_chan, VIS_MAX_CHAN_BINS)

    amplitude = np.abs(data[:, cross, :]).astype(np.float64).mean(axis=1)  # (n_int, n_chan)
    amplitude = bin_mean(amplitude.T, axis=0, n_bins=chan_bins)  # (chan_bins, n_int)

    low, high = (float(value) for value in np.percentile(amplitude, DISPLAY_PERCENTILES))
    if high - low <= 0.0:
        low, high = float(amplitude.min()), float(amplitude.max()) or 1.0

    fraction = np.asarray(visibilities.rfi_fraction, dtype=np.float64)  # (n_int, n_src, n_chan)
    sources = []
    for index, name in enumerate(visibilities.rfi_source_names):
        plane = fraction[:, index, :]
        mask = bin_any(plane.T > 0.0, axis=0, n_bins=chan_bins)
        sources.append(
            {
                "name": str(name),
                "mask": mask.astype(np.uint8).tolist(),
                "mean_fraction": float(plane.mean()) if plane.size else 0.0,
                "max_fraction": float(plane.max()) if plane.size else 0.0,
            }
        )

    lengths = np.linalg.norm(np.asarray(visibilities.baseline_vectors_enu_m), axis=1)
    cross_indices = np.flatnonzero(cross)
    order = cross_indices[np.argsort(lengths[cross_indices], kind="stable")]
    mean_amplitude = np.abs(data).astype(np.float64).mean(axis=(0, 2))

    baselines = [
        {
            "index": int(index),
            "ant_1": int(visibilities.ant_1[index]),
            "ant_2": int(visibilities.ant_2[index]),
            "length_m": float(np.round(lengths[index], 2)),
            "mean_amp_jy": float(np.round(mean_amplitude[index], 5)),
        }
        for index in order
    ]

    return {
        "amplitude": _round_grid(amplitude, 5),
        "freq_mhz": (
            bin_mean(np.asarray(visibilities.freq_hz, dtype=np.float64), axis=0, n_bins=chan_bins)
            / 1.0e6
        )
        .round(6)
        .tolist(),
        "time_s": ((np.arange(n_int) + 0.5) * visibilities.integration_time_s).round(6).tolist(),
        "vmin_jy": round(low, 5),
        "vmax_jy": round(high, 5),
        "peak_jy": round(float(amplitude.max()) if amplitude.size else 0.0, 5),
        "n_baselines": int(np.count_nonzero(cross)),
        "n_integrations": int(n_int),
        "integration_time_s": float(visibilities.integration_time_s),
        "sources": sources,
        "baselines": baselines,
        "spectra": _visibility_spectra(visibilities, order, pol),
        "unit": "Jy",
    }


# ----------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------
def _feature_seed_sequences(seed: int) -> tuple[np.random.SeedSequence, np.random.SeedSequence]:
    """Independent seed sequences for the instrument model and calibration errors.

    Both `InstrumentModel.from_params` and `CalibrationErrors.from_params`
    accept a raw `seed` and, given one, do the same thing internally:
    build `SeedSequence(seed)` and spawn a fixed number of children in a
    fixed order for their own effects. `SeedSequence.spawn`'s first child
    depends only on the seed, not on how many children are requested, so
    handing both models the same `request.sim.seed` directly would give
    `InstrumentModel`'s gain-scatter draw and `CalibrationErrors`'s phase
    error draw the *same* underlying standard-normal vector -- perfectly
    correlating two effects the rest of the simulator treats as
    independent. Spawning two children from one root here, and passing
    each through `rng=` instead of `seed=`, keeps the run fully
    reproducible from `sim.seed` alone while decorrelating the two
    models. (`VoltageSimulator`'s own `rng` -- see `build_simulator` --
    draws entropy from the generator before building its internal seed
    sequence, a different code path, so it does not re-collide with
    either of these despite starting from the same `sim.seed`.)
    """
    root = np.random.SeedSequence(seed)
    instrument_seq, calibration_seq = root.spawn(2)
    return instrument_seq, calibration_seq


def build_simulator(
    request: SimulateRequest,
    *,
    start_time: Time | None = None,
    phase_center: SkyCoord | None = None,
    extra_sources: list[PointSource] | None = None,
) -> VoltageSimulator:
    """Assemble the library objects a request describes.

    Parameters
    ----------
    request : SimulateRequest
        A validated request.
    start_time : astropy.time.Time, optional
        When the observation starts. Defaults to `START_TIME_UTC`, which
        is what the single-run endpoint uses so that a run is reproducible
        from its seed alone. The observatory day passes one instant per
        frame instead.
    phase_center : astropy.coordinates.SkyCoord, optional
        Where the array points. Defaults to the zenith of the site at
        `start_time`. The observatory day passes a meridian pointing at a
        fixed declination, which is not the zenith unless that
        declination is the site latitude.
    extra_sources : list of PointSource, optional
        Celestial sources to observe in addition to `request.sky_sources`,
        already built at absolute coordinates. The observatory day passes
        its catalogue this way, so that the catalogue never has to be
        expressed as an offset from a pointing that moves.

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
        latitude_deg=site.latitude_deg if request.site is None else request.site.latitude_deg,
        longitude_deg=site.longitude_deg if request.site is None else request.site.longitude_deg,
        height_m=site.height_m if request.site is None else request.site.height_m,
        name=site.name,
    )
    if start_time is None:
        start_time = Time(START_TIME_UTC, scale="utc")
    if phase_center is None:
        phase_center = zenith_coord(earth_location(array), start_time)

    sources = [source.build(phase_center) for source in request.sky_sources]
    sources.extend(extra_sources or ())
    rfi_sources = [source.build() for source in request.rfi_sources]
    spectral_lines = [line.build() for line in request.spectral_lines]

    n_antennas = len(request.antennas)
    # All the realism models below are re-seeded from the run's own seed:
    # there is no separate seed field per feature, so switching one on
    # never disturbs another, and re-running with the same seed is still
    # byte-identical (the same guarantee `sim.seed` already gives the sky,
    # noise and RFI draws). Instrument and calibration errors additionally
    # go through `_feature_seed_sequences` rather than the raw seed, so
    # that the two models -- which both spawn children from a seed
    # sequence the same way -- do not draw the same underlying random
    # vector for two supposedly-independent effects.
    instrument_seq, _ = _feature_seed_sequences(request.sim.seed)
    instrument = (
        None
        if request.instrument is None
        else request.instrument.build(n_antennas, np.random.default_rng(instrument_seq))
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
        # The bandpass: total power per antenna, per receptor, per channel,
        # summed over every time sample of the run. Unlike the waterfall
        # this keeps *both* receptors, because comparing them is the whole
        # point of a bandpass plot; it costs n_ant x n_pol x n_chan floats,
        # which is a few tens of kilobytes at the largest size this front
        # end runs and is never pooled in time -- a time-averaged spectrum
        # is exactly what is wanted.
        self._bandpass_sum = np.zeros(
            (simulator.n_antennas, simulator.n_pol, simulator.n_chan), dtype=np.float64
        )
        self._bandpass_samples = 0

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
        # Both receptors, every channel, summed over time: the bandpass.
        # Taken before the single-receptor slice below so that a dual
        # polarization run ships XX and YY spectra from one pass.
        all_pol = block.pol_data
        all_power = all_pol.real.astype(np.float64) ** 2 + all_pol.imag.astype(np.float64) ** 2
        self._bandpass_sum += all_power.sum(axis=3)
        self._bandpass_samples += all_power.shape[3]

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

        # Mean power per (antenna, receptor, channel), then pooled onto the
        # display's channel bins so the bandpass shares the waterfall's
        # frequency axis and the two plots can be read against each other.
        samples = max(1, self._bandpass_samples)
        bandpass = bin_mean(self._bandpass_sum / samples, axis=2, n_bins=self.chan_bins)

        return {
            "waterfall": waterfall,
            "masks": masks,
            "occupancy": occupancy,
            "freq_hz": freq_hz,
            "time_s": time_s,
            "time_samples_per_cell": self.time_samples_per_cell,
            "bandpass": bandpass,
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


def _bandpass_payload(bandpass: np.ndarray, pol_names: list[str]) -> dict[str, Any]:
    """Time-averaged spectra, in decibels, ready to plot as lines.

    Parameters
    ----------
    bandpass : numpy.ndarray
        Mean power per ``(antenna, receptor, channel)``, Jy.
    pol_names : list of str
        The receptor names the correlator reports, ``["XX"]`` or
        ``["XX", "YY"]``, so the page can label the traces without
        guessing from an index.

    Returns
    -------
    dict
        ``antennas``: a ``(n_antenna, n_pol, chan_bins)`` nested list in
        decibels, sharing the waterfall's frequency axis. ``vmin_db`` and
        ``vmax_db`` are the true ends of the data with a little headroom,
        not percentiles: a bandpass is read for its shape, and clipping
        the very feature -- a narrowband spike -- that the reader is
        looking for would be the wrong kindness. `DYNAMIC_RANGE_DB` below
        the peak is still floored so an empty channel cannot reach minus
        infinity.
    """
    peak = float(bandpass.max()) if bandpass.size else 0.0
    if not np.isfinite(peak) or peak <= 0.0:
        zeros = np.zeros(bandpass.shape, dtype=np.float64)
        return {
            "antennas": [_round_grid(plane, 3) for plane in zeros],
            "pol_names": list(pol_names),
            "vmin_db": 0.0,
            "vmax_db": 1.0,
            "unit": "dB (mean Jy per channel)",
        }
    floor = peak * 10.0 ** (-DYNAMIC_RANGE_DB / 10.0)
    decibels = 10.0 * np.log10(np.maximum(bandpass, floor))
    low = float(decibels.min())
    high = float(decibels.max())
    if high - low < 1.0:
        low, high = low - 0.5, high + 0.5
    pad = 0.05 * (high - low)
    return {
        "antennas": [_round_grid(plane, 3) for plane in decibels],
        "pol_names": list(pol_names),
        "vmin_db": round(low - pad, 3),
        "vmax_db": round(high + pad, 3),
        "unit": "dB (mean Jy per channel)",
    }


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
        exact occupancy fraction each), ``visibilities`` (what the
        correlator saw, see `_visibility_payload`), ``image`` (dirty
        image, its peak, and the fitted primary beam's half-power radius
        when there is one), ``sky_sources`` (each with its band-centre
        ``beam_response`` when a beam is fitted),
        ``uv`` (baseline coordinates in wavelengths), any
        ``warnings`` the library raised, and the wall time. The waterfall
        also reports ``time_samples_per_cell``, the number of voltage
        samples pooled into one displayed column, so the page can say how
        coarse the picture it is drawing really is, and ``bandpass``, the
        time-averaged spectrum of every antenna and *every* receptor (see
        `_bandpass_payload`) -- both receptors regardless of which one the
        waterfall is showing, because a bandpass plot exists to compare
        them.

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
        _, calibration_seq = _feature_seed_sequences(request.sim.seed)
        calibration_errors = (
            None
            if request.calibration_errors is None
            else request.calibration_errors.build(
                simulator.n_antennas, np.random.default_rng(calibration_seq)
            )
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
        visibility_payload = _visibility_payload(visibilities, pol)
        beam_response = simulator.beam_response()
    wall_time_s = time.perf_counter() - started

    decibels, vmin_db, vmax_db, peak_db = _to_decibels(reduced["waterfall"])
    bandpass_payload = _bandpass_payload(reduced["bandpass"], list(visibilities.pol_names))

    peak_index = int(np.argmax(image))
    peak_row, peak_col = np.unravel_index(peak_index, image.shape)

    center_channel = visibilities.n_chan // 2
    cross = visibilities.cross_mask
    u_points = u_lambda[:, cross, center_channel].ravel()
    v_points = v_lambda[:, cross, center_channel].ravel()

    # Band-centre primary-beam response per sky source -- the library's
    # own ground truth (`VoltageSimulator.beam_response`), not a second
    # evaluation of the beam here. `None` throughout when no beam is
    # fitted, which is a different statement from "attenuated by 1.0".
    beam_centre = (
        None if beam_response is None else beam_response[:, simulator.n_chan // 2].astype(float)
    )
    sky_payload = []
    for index, source in enumerate(simulator.sources):
        lm = source.lm(simulator.phase_center)
        sky_payload.append(
            {
                "name": source.name,
                "flux_jy": float(source.flux_jy),
                "l": float(lm[0]),
                "m": float(lm[1]),
                "ra_deg": float(source.coord.icrs.ra.deg),
                "dec_deg": float(source.coord.icrs.dec.deg),
                "in_field": bool(
                    max(abs(float(lm[0])), abs(float(lm[1]))) <= 0.5 * IMAGE_FIELD_OF_VIEW_RAD
                ),
                "beam_response": None if beam_centre is None else float(beam_centre[index]),
            }
        )

    beam_payload = None
    if request.primary_beam is not None:
        half_power_rad = _beam_half_power_rad(
            request.primary_beam.build(), simulator.center_freq_hz
        )
        beam_payload = {
            "type": request.primary_beam.type,
            "dish_diameter_m": request.primary_beam.dish_diameter_m,
            "half_power_rad": None if half_power_rad is None else round(half_power_rad, 9),
            "center_freq_hz": float(simulator.center_freq_hz),
        }

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
            "bandpass": bandpass_payload,
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
        "sky_sources": sky_payload,
        "visibilities": visibility_payload,
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
            "beam": beam_payload,
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


# ----------------------------------------------------------------------
# Classical flaggers: the floor an excision algorithm has to beat
# ----------------------------------------------------------------------
MAX_FLAG_METHODS = 2
"""int: Methods one flagging request may run.

Two, because the point of the panel is a head-to-head: one column each,
scored on the same cells of the same simulated data. Three columns would
not fit the table and would not be read."""

FLAG_METHODS = [
    {
        "value": "sk",
        "label": "Spectral kurtosis",
        "summary": (
            "Tests each accumulation's power distribution against the exponential "
            "one Gaussian noise gives. Catches carriers and short bursts, including "
            "ones no louder than the noise."
        ),
        "grid": "pre-detection voltages, one decision per accumulation",
    },
    {
        "value": "mad",
        "label": "MAD clipping",
        "summary": (
            "Per-channel robust sigma clipping of the power spectrogram: median and "
            "median absolute deviation over time, then a cut either side."
        ),
        "grid": "detected power, one decision per accumulated cell",
    },
    {
        "value": "sumthreshold",
        "label": "SumThreshold",
        "summary": (
            "Thresholds runs of neighbouring cells with a limit that falls as the run "
            "lengthens, on the MAD-normalized residual. Finds faint but contiguous "
            "interference a single-cell cut misses."
        ),
        "grid": "detected power, one decision per accumulated cell",
    },
]
"""list of dict: The flagging methods the page offers, as descriptors the
browser builds its controls from -- the same one-source-of-truth rule the
form fields follow."""

FlagMethod = Literal["sk", "mad", "sumthreshold"]

FlagDomain = Literal["voltage", "visibility"]
"""Where a flagging request is answered.

``voltage`` flags one antenna's own accumulated spectra, before
correlation; ``visibility`` flags the baseline-averaged amplitude the
correlator produced, which is where a visibility-domain method such as
AOFlagger's SumThreshold actually runs on a real telescope. The two see
different grids and different ground truth (per-sample interference
masks against the correlator's per-integration `rfi_fraction`), so the
domain travels in the response and the page prints which one it is."""

FLAG_DOMAINS = [
    {
        "value": "voltage",
        "label": "Voltages",
        "summary": "One antenna's accumulated spectra, before correlation.",
        "methods": ["sk", "mad", "sumthreshold"],
    },
    {
        "value": "visibility",
        "label": "Visibilities",
        "summary": (
            "The baseline-averaged amplitude the correlator produced. Spectral "
            "kurtosis needs pre-detection samples and does not apply here."
        ),
        "methods": ["mad", "sumthreshold"],
    },
]
"""list of dict: The two places the page can ask for flags, as
descriptors the browser builds its controls from -- including which
methods each domain admits, so a chip can be greyed out with a reason
instead of a request being refused after the click."""


class FlagParams(BaseModel):
    """Tuning of the classical flaggers.

    One model for all three methods rather than one per method: they
    share the accumulation grid, and a page that lets a reader switch
    method should not silently discard the settings of the one they
    switched away from. Each method reads only the fields that mean
    something to it.
    """

    model_config = ConfigDict(extra="forbid")

    m: int = Field(default=FLAG_DEFAULT_M, ge=2, le=N_TIME_PER_BLOCK)
    pfa: float = Field(default=0.0027, gt=0.0, lt=1.0)
    n_sigma: float = Field(default=5.0, gt=0.0, le=50.0)
    chi_1: float = Field(default=6.0, gt=0.0, le=100.0)
    iterations: int = Field(default=5, ge=1, le=10)

    @field_validator("m")
    @classmethod
    def _check_accumulation(cls, value: int) -> int:
        """Insist the accumulation divides a block.

        Blocks are generated one at a time and never held together, so an
        accumulation that straddled a block boundary could not be formed
        at all; one that did not divide the block would silently drop the
        remainder of every block instead of the remainder of the run.
        """
        if N_TIME_PER_BLOCK % value:
            raise ValueError(
                f"m must divide the {N_TIME_PER_BLOCK} time samples in a block, and "
                f"{value} does not: accumulations are formed inside one block"
            )
        return value


class FlagRequest(BaseModel):
    """Run one or two classical flaggers over one antenna of a run.

    The observation is described by `request`, exactly as
    ``POST /api/simulate`` takes it, and is *re-simulated* here. That
    costs a second pass over the voltages, and buys the two properties
    that matter: the server keeps no per-client state, and the data
    flagged is bit-identical to the data the page is displaying because
    both are drawn from `SimParams.seed`.
    """

    model_config = ConfigDict(extra="forbid")

    request: SimulateRequest
    methods: list[FlagMethod] = Field(min_length=1, max_length=MAX_FLAG_METHODS)
    antenna: int = Field(default=0, ge=0, le=MAX_ANTENNAS - 1)
    pol: Literal[0, 1] = 0
    domain: FlagDomain = "voltage"
    params: FlagParams = Field(default_factory=FlagParams)

    @field_validator("methods")
    @classmethod
    def _check_unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError(f"name each method once, got {value}")
        return value

    @model_validator(mode="after")
    def _check_domain(self) -> FlagRequest:
        """Refuse spectral kurtosis after correlation.

        The estimator is a ratio of the second and fourth moments of the
        *pre-detection* samples inside one accumulation. Baseline-averaged
        visibility amplitudes have already had both the polarization and
        the baseline axes collapsed and the samples integrated, so those
        moments are gone; running the same formula on them would produce a
        number, and the number would mean nothing. Say so rather than
        return it.
        """
        if self.domain == "visibility" and "sk" in self.methods:
            raise ValueError(
                "spectral kurtosis is a pre-detection test and cannot be run on "
                "baseline-averaged visibility amplitudes: use it in the voltage "
                "domain, or pick MAD clipping or SumThreshold here"
            )
        return self


def _flag_overlay(predicted: np.ndarray, truth: np.ndarray, chan_bins: int) -> dict[str, Any]:
    """The three ways a decision can turn out, pooled for drawing.

    Parameters
    ----------
    predicted, truth : numpy.ndarray
        Boolean masks on the flagger's own grid, ``(n_chan, n_accum)``.
    chan_bins : int
        Channel bins to pool onto. Time is never pooled: the accumulation
        grid is already coarse in time and its columns are what the
        reader is being shown.

    Returns
    -------
    dict
        ``caught``, ``missed`` and ``false_alarm`` as 0/1 grids. Pooling
        is an ANY rule per category, so one displayed cell can be more
        than one colour where a bin holds both outcomes -- which is the
        truth about that bin, and the reason the scores in the same
        response are computed on the unpooled masks.
    """
    categories = {
        "caught": predicted & truth,
        "missed": truth & ~predicted,
        "false_alarm": predicted & ~truth,
    }
    return {
        name: bin_any(mask, axis=0, n_bins=chan_bins).astype(np.uint8).tolist()
        for name, mask in categories.items()
    }


def _flag_fraction(mask: np.ndarray, chan_bins: int) -> list[float]:
    """The fraction of the run each channel spent flagged.

    Parameters
    ----------
    mask : numpy.ndarray
        A boolean mask on the flagger's own grid, ``(n_chan, n_time)``.
    chan_bins : int
        Channel bins to pool onto, matching the overlay's.

    Returns
    -------
    list of float
        One number per displayed channel, 0 to 1. Pooling here is a
        *mean*, not the ANY rule the overlay uses: this is the flag
        occupancy of a channel and averaging is what it means, whereas
        the overlay answers "was anything in this cell flagged".
    """
    if not mask.size:
        return []
    per_channel = mask.astype(np.float64).mean(axis=1)
    return bin_mean(per_channel, axis=0, n_bins=chan_bins).round(5).tolist()


def _visibility_flag_grids(request: FlagRequest) -> dict[str, Any]:
    """Correlate the run and hand back the grid a visibility flagger sees.

    Parameters
    ----------
    request : FlagRequest
        A validated request whose ``domain`` is ``"visibility"``.

    Returns
    -------
    dict
        ``spectrogram``: ``|V|`` averaged over every cross-correlation
        baseline, ``(n_chan, n_integrations)`` -- the same quantity the
        visibility panel draws, at full channel resolution. ``truth``: a
        boolean mask of the cells any interference source touched, from
        the correlator's own ``rfi_fraction``. ``freq_hz``, ``time_s``,
        ``integration_time_s`` and ``n_integrations`` describe the axes.

    Notes
    -----
    Ground truth after correlation is a different statement from ground
    truth before it. Pre-correlation, a cell is contaminated if any
    voltage sample in it was; post-correlation, the library reports the
    *fraction* of an integration a source occupied, and a cell counts as
    contaminated here when that fraction is non-zero. A source that lit
    one sample in a thousand therefore marks the whole integration --
    which is the honest reading, because that integration's amplitude
    really is contaminated, however slightly.
    """
    simulator = build_simulator(request.request)
    pol = request.pol if simulator.n_pol > 1 else 0
    _, calibration_seq = _feature_seed_sequences(request.request.sim.seed)
    calibration_errors = (
        None
        if request.request.calibration_errors is None
        else request.request.calibration_errors.build(
            simulator.n_antennas, np.random.default_rng(calibration_seq)
        )
    )
    visibilities = correlate(simulator.blocks(), calibration_errors=calibration_errors)

    cross = visibilities.cross_mask
    data = visibilities.pol_data[:, :, pol, :]
    spectrogram = np.abs(data[:, cross, :]).astype(np.float64).mean(axis=1).T

    fraction = np.asarray(visibilities.rfi_fraction, dtype=np.float64)
    if fraction.size:
        truth = (fraction > 0.0).any(axis=1).T
    else:
        truth = np.zeros(spectrogram.shape, dtype=bool)

    n_int = int(visibilities.n_int)
    integration_time_s = float(visibilities.integration_time_s)
    return {
        "simulator": simulator,
        "pol": pol,
        "spectrogram": spectrogram,
        "truth": truth,
        "freq_hz": np.asarray(visibilities.freq_hz, dtype=np.float64),
        "time_s": (np.arange(n_int) + 0.5) * integration_time_s,
        "integration_time_s": integration_time_s,
        "n_integrations": n_int,
    }


def run_flaggers(request: FlagRequest) -> dict[str, Any]:
    """Re-simulate one run and score classical flaggers against its truth.

    Parameters
    ----------
    request : FlagRequest
        A validated request: the observation, the methods, which antenna
        and receptor to look at, and which ``domain`` -- the voltages of
        one antenna, or the baseline-averaged visibility amplitude.

    Returns
    -------
    dict
        JSON-ready: one entry per method under ``methods``, each with the
        scores `rfi_simulator.metrics.flag_scores` gives on the flagger's
        own grid and an overlay of caught / missed / false alarm for
        drawing; plus the shared ``grid`` those decisions live on and the
        axes to draw it against.

    Raises
    ------
    ValueError
        If the antenna does not exist in this array, or the requested
        accumulation would build a grid larger than `MAX_FLAG_CELLS`.

    Notes
    -----
    **One grid for every method.** Spectral kurtosis is defined on
    accumulations of ``m`` pre-detection samples and cannot be computed
    from a spectrogram at all; the power-based methods are therefore run
    on the mean power of the *same* accumulations rather than on the
    voltage-resolution grid. That is both realistic -- a real time-domain
    flagger runs on integrated spectra -- and the only way two methods
    can be put in adjacent columns honestly, since scores computed on
    different grids are not comparable. Ground truth is brought onto that
    grid with `rfi_simulator.metrics.pool_truth_accumulations`, the
    partition the kurtosis estimator itself uses.
    """
    if request.domain == "visibility":
        return _run_visibility_flaggers(request)
    started = time.perf_counter()
    params = request.params
    methods = list(request.methods)
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        simulator = build_simulator(request.request)
        if request.antenna >= simulator.n_antennas:
            raise ValueError(
                f"this array has {simulator.n_antennas} antennas, so there is no "
                f"antenna {request.antenna}"
            )
        pol = request.pol if simulator.n_pol > 1 else 0

        accum_per_block = simulator.n_time_per_block // params.m
        n_accum = accum_per_block * simulator.n_blocks
        if simulator.n_chan * n_accum > MAX_FLAG_CELLS:
            raise ValueError(
                f"flagging this run on accumulations of {params.m} samples would build a "
                f"{simulator.n_chan} x {n_accum} grid, more than the {MAX_FLAG_CELLS:,} "
                "cells this front end flags at once: lengthen the accumulation or "
                "record fewer channels"
            )

        sk_columns: list[np.ndarray] = []
        power_columns: list[np.ndarray] = []
        truth_columns: list[np.ndarray] = []
        for block in simulator.blocks():
            data = block.pol_data[request.antenna, pol]
            if "sk" in methods:
                sk_columns.append(spectral_kurtosis_mask(data, params.m, pfa=params.pfa))
            trimmed = data[:, : accum_per_block * params.m]
            power = (
                trimmed.real.astype(np.float64) ** 2 + trimmed.imag.astype(np.float64) ** 2
            ).reshape(simulator.n_chan, accum_per_block, params.m)
            power_columns.append(power.mean(axis=2))
            truth_columns.append(pool_truth_accumulations(block.rfi_mask.any(axis=0), params.m))

        spectrogram = np.concatenate(power_columns, axis=1)
        truth = np.concatenate(truth_columns, axis=1)

        predictions: dict[str, np.ndarray] = {}
        if "sk" in methods:
            predictions["sk"] = np.concatenate(sk_columns, axis=1)
        if "mad" in methods:
            predictions["mad"] = mad_clip_mask(spectrogram, params.n_sigma)
        if "sumthreshold" in methods:
            _, residual = mad_clip_mask(spectrogram, params.n_sigma, return_statistic=True)
            predictions["sumthreshold"] = sumthreshold_mask(
                residual, params.chi_1, params.iterations
            )
    wall_time_s = time.perf_counter() - started

    chan_bins = min(simulator.n_chan, FLAG_OVERLAY_MAX_CHAN_BINS)
    accumulation_s = simulator.duration_s / n_accum
    labels = {entry["value"]: entry["label"] for entry in FLAG_METHODS}
    grids = {entry["value"]: entry["grid"] for entry in FLAG_METHODS}

    results = [
        {
            "method": method,
            "label": labels[method],
            "grid": grids[method],
            "scores": {
                name: _finite(value)
                for name, value in flag_scores(predictions[method], truth).items()
            },
            "overlay": _flag_overlay(predictions[method], truth, chan_bins),
            "flag_fraction": _flag_fraction(predictions[method], chan_bins),
        }
        for method in methods
    ]

    messages = []
    for entry in caught_warnings:
        text = str(entry.message)
        if text not in messages:
            messages.append(text)

    return {
        "methods": results,
        "grid": {
            "domain": "voltage",
            "m": params.m,
            "n_chan": int(simulator.n_chan),
            "n_accumulations": int(n_accum),
            "accumulation_s": round(float(accumulation_s), 9),
            "chan_bins": int(chan_bins),
            "freq_mhz": (bin_mean(simulator.freq_hz, axis=0, n_bins=chan_bins) / 1.0e6)
            .round(6)
            .tolist(),
            "time_s": ((np.arange(n_accum) + 0.5) * accumulation_s).round(6).tolist(),
            "truth_fraction": _flag_fraction(truth, chan_bins),
        },
        "domain": "voltage",
        "antenna": int(request.antenna),
        "pol": int(pol),
        "params": params.model_dump(),
        "warnings": messages,
        "wall_time_s": round(wall_time_s, 3),
    }


def _run_visibility_flaggers(request: FlagRequest) -> dict[str, Any]:
    """Score the power-based flaggers on the correlated amplitudes.

    Parameters
    ----------
    request : FlagRequest
        A validated request whose ``domain`` is ``"visibility"``. Its
        ``antenna`` is ignored -- interference reaches every antenna and
        the grid here is already averaged over every baseline -- and is
        echoed back so the caller can see that it was not used.

    Returns
    -------
    dict
        The same shape `run_flaggers` returns in the voltage domain, so
        the page draws it with the same code: ``methods`` with scores, a
        caught / missed / false-alarm overlay and a per-channel flag
        fraction, and the ``grid`` those decisions live on.

    Notes
    -----
    The grid is coarse in time by construction: there is one column per
    correlator integration, and a short recording has only a handful.
    Robust statistics over a handful of samples are weak, and the scores
    will say so -- which is itself the lesson, since a real
    visibility-domain flagger runs on hours of integrations, not on
    milliseconds.
    """
    started = time.perf_counter()
    params = request.params
    methods = list(request.methods)
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        grids = _visibility_flag_grids(request)
        spectrogram = grids["spectrogram"]
        truth = grids["truth"]

        # Both directions, unioned. A per-channel median over time -- the
        # background the voltage domain uses -- is blind here: a
        # continuous transmitter is *constant* in a channel across a
        # recording this short, so it becomes the channel's own median and
        # deviates from itself by nothing. The spectral direction is what
        # sees it, and is what an astronomer looking at a snapshot
        # spectrum uses. Running only one of the two would report a recall
        # of zero on interference plainly visible in the panel above.
        time_mask, time_residual = mad_clip_mask(spectrogram, params.n_sigma, return_statistic=True)
        freq_mask, freq_residual = mad_clip_mask(
            spectrogram.T, params.n_sigma, return_statistic=True
        )

        predictions: dict[str, np.ndarray] = {}
        if "mad" in methods:
            predictions["mad"] = time_mask | freq_mask.T
        if "sumthreshold" in methods:
            predictions["sumthreshold"] = (
                sumthreshold_mask(time_residual, params.chi_1, params.iterations)
                | sumthreshold_mask(freq_residual, params.chi_1, params.iterations).T
            )
    wall_time_s = time.perf_counter() - started

    simulator = grids["simulator"]
    chan_bins = min(int(simulator.n_chan), FLAG_OVERLAY_MAX_CHAN_BINS)
    labels = {entry["value"]: entry["label"] for entry in FLAG_METHODS}

    results = [
        {
            "method": method,
            "label": labels[method],
            "grid": (
                "baseline-averaged visibility amplitude, one decision per "
                "integration, background removed along time and frequency"
            ),
            "scores": {
                name: _finite(value)
                for name, value in flag_scores(predictions[method], truth).items()
            },
            "overlay": _flag_overlay(predictions[method], truth, chan_bins),
            "flag_fraction": _flag_fraction(predictions[method], chan_bins),
        }
        for method in methods
    ]

    messages = []
    for entry in caught_warnings:
        text = str(entry.message)
        if text not in messages:
            messages.append(text)

    return {
        "methods": results,
        "grid": {
            "domain": "visibility",
            "m": None,
            "n_chan": int(simulator.n_chan),
            "n_accumulations": int(grids["n_integrations"]),
            "accumulation_s": round(float(grids["integration_time_s"]), 9),
            "chan_bins": int(chan_bins),
            "freq_mhz": (bin_mean(grids["freq_hz"], axis=0, n_bins=chan_bins) / 1.0e6)
            .round(6)
            .tolist(),
            "time_s": np.asarray(grids["time_s"]).round(6).tolist(),
            "truth_fraction": _flag_fraction(truth, chan_bins),
        },
        "domain": "visibility",
        "antenna": int(request.antenna),
        "pol": int(grids["pol"]),
        "params": params.model_dump(),
        "warnings": messages,
        "wall_time_s": round(wall_time_s, 3),
    }
