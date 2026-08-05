"""A simulated day of drift-scan observing, and the sky as it is right now.

Two things live here, and they share a coordinate convention rather than
any code path.

**The day.** The array is treated as a transit instrument: it stares at
the meridian at one fixed declination and lets the sky drift through.
Frame *i* of a day is therefore one short snapshot phase-centred on
``(RA = the right ascension crossing the meridian at that instant,
Dec = the declination the user chose)``, simulated with the user's own
setup -- the same antennas, the same interference, the same instrument --
plus a small catalogue of real radio sources laid in at their true
coordinates. A source contributes to a frame only when it is inside that
frame's imaged field and above the horizon, so watching the day play is
watching the catalogue drift through the beam.

Frames are independent, so they are computed in a process pool
(`day_workers`) and stored reduced: a 64x64 image and a line of metadata
each, never a whole response.

**Now.** `sky_now` in `rfi_simulator.webui.skynow` is the live half; the
catalogue and the horizon geometry it needs are defined here so that the
movie and the monitor cannot disagree about where Cygnus A is.

Coordinate conventions
----------------------
The pointing declination is read as **ICRS**, the frame the catalogue is
tabulated in, so that "Cygnus A's strip" really does put Cygnus A through
the centre of the field. The right ascension is *not* ICRS to begin with
-- the meridian is a direction on the equator of date -- so
`meridian_phase_center` solves for the ICRS right ascension that lands on
the meridian at the requested declination. Nutation is ignored in the
pairing (apparent sidereal time against a mean-of-date frame), which
displaces the pointing by at most 17 arcseconds, four hundred times less
than the field's half width.
"""

from __future__ import annotations

import math
import os
import threading
import time as _time
import uuid
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Literal

import numpy as np
from astropy import units as u
from astropy.coordinates import FK5, AltAz, EarthLocation, SkyCoord, get_sun
from astropy.time import Time, TimeDelta
from pydantic import BaseModel, ConfigDict, Field, field_validator

from rfi_simulator import ArrayConfig, PointSource, correlate, dirty_image
from rfi_simulator.delays import earth_location
from rfi_simulator.sky import lm_from_radec
from rfi_simulator.webui.simulate import (
    IMAGE_FIELD_HALF_WIDTH_DEG,
    IMAGE_FIELD_OF_VIEW_RAD,
    IMAGE_MAX_CHANNELS,
    IMAGE_N_PIX,
    SimulateRequest,
    build_simulator,
    default_array,
)

__all__ = [
    "CATALOG_SOURCES",
    "DAY_WORKERS_ENV_VAR",
    "DEFAULT_DAY_WORKERS",
    "DayRequest",
    "JOB_MAX",
    "JOB_TTL_S",
    "MAX_FRAMES",
    "QUIET_SUN_FLUX_JY",
    "cancel_day",
    "catalog_sources_in_field",
    "day_frame",
    "day_status",
    "day_workers",
    "jobs",
    "meridian_phase_center",
    "site_location",
    "start_day",
    "timeline_payload",
]


# ----------------------------------------------------------------------
# The catalogue
# ----------------------------------------------------------------------
CATALOG_SOURCES: tuple[dict[str, Any], ...] = (
    # The four brightest compact sources of the L band, with standard
    # L-band flux-scale values at 1.4 GHz. They are quoted to three
    # figures because that is all a flux scale is good for, and they are
    # held flat across the simulated band -- a few megahertz wide, over
    # which every one of them varies by well under a percent.
    {"name": "Cas A", "ra_deg": 350.858, "dec_deg": 58.815, "flux_jy": 1720.0},
    {"name": "Cyg A", "ra_deg": 299.868, "dec_deg": 40.734, "flux_jy": 1590.0},
    {"name": "Tau A", "ra_deg": 83.633, "dec_deg": 22.015, "flux_jy": 875.0},
    {"name": "Vir A", "ra_deg": 187.706, "dec_deg": 12.391, "flux_jy": 212.0},
)
"""tuple of dict: The movie's sky, in ICRS degrees and janskys.

Positions are the standard catalogue ones; the flux densities are
standard L-band flux-scale values at 1.4 GHz. All four are treated as
unresolved point sources, which they are not -- Cas A and Tau A are
arcminutes across -- but the array this front end simulates has a
resolution measured in degrees, so nothing in the image could tell the
difference."""

QUIET_SUN_FLUX_JY = 5.0e4
"""float: Flux density of the quiet Sun at L band, janskys.

A documented round number for the quiet (spot-free) disc near 1.4 GHz,
held constant: the real value follows the solar cycle over a factor of a
few and a flare can beat it by orders of magnitude, so a single constant
is honest only as an order of magnitude. It is the right order, and it is
what makes the Sun the loudest thing in the movie by a factor of thirty.
"""

SUN_NAME = "Sun"


# ----------------------------------------------------------------------
# Sizes, budgets and shared-host etiquette
# ----------------------------------------------------------------------
DEFAULT_DAY_WORKERS = 12
"""int: Processes a day is computed in.

A fraction of the available cores -- roughly 30% of a forty-core host --
rather than all of them: shared-host etiquette. Frames are independent, so
the pool scales almost linearly up to whatever this is set to."""

DAY_WORKERS_ENV_VAR = "RFI_SIMULATOR_DAY_WORKERS"
"""str: Environment variable overriding `DEFAULT_DAY_WORKERS`."""

MAX_FRAMES = 288
"""int: Most frames one day may have -- five-minute cadence."""

DEFAULT_FRAMES = 96
"""int: Frames a day has by default: a quarter-hour cadence."""

JOB_MAX = 4
"""int: Days kept in memory at once. A finished day is a few megabytes of
reduced images; beyond this the least recently touched one is dropped."""

JOB_TTL_S = 3600.0
"""float: Age at which a day is dropped whether or not the store is full."""

COARSE_N_CHAN = 32
COARSE_N_BLOCKS = 1
"""int: The coarse preset's recording size.

Ninety-six frames at the setup's own size would be a ninety-six-fold run;
coarse trades channels and integrations -- which cost sensitivity and
sidelobe level, not source position -- for a day that builds in a minute
rather than an hour."""

FRAME_IMAGE_DECIMALS = 5


def day_workers() -> int:
    """Processes to compute a day in, honouring `DAY_WORKERS_ENV_VAR`."""
    raw = os.environ.get(DAY_WORKERS_ENV_VAR)
    if not raw:
        return DEFAULT_DAY_WORKERS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_DAY_WORKERS
    return max(1, value)


# ----------------------------------------------------------------------
# Geometry shared by the movie and the monitor
# ----------------------------------------------------------------------
def site_location(request: SimulateRequest | None = None) -> EarthLocation:
    """The observing site, from a request's ``site`` or the default array.

    Built through the library's own `earth_location` on a throwaway array
    so that the site the day observes from is the same object the single
    run observes from.
    """
    site = default_array()
    if request is None or request.site is None:
        latitude, longitude, height = site.latitude_deg, site.longitude_deg, site.height_m
    else:
        latitude = request.site.latitude_deg
        longitude = request.site.longitude_deg
        height = request.site.height_m
    array = ArrayConfig(
        antenna_positions_enu_m=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        latitude_deg=latitude,
        longitude_deg=longitude,
        height_m=height,
    )
    return earth_location(array)


def meridian_phase_center(location: EarthLocation, time: Time, dec_deg: float) -> SkyCoord:
    """Where a transit instrument at `dec_deg` points at `time`.

    Parameters
    ----------
    location : astropy.coordinates.EarthLocation
        The site.
    time : astropy.time.Time
        Scalar UTC instant.
    dec_deg : float
        Pointing declination, ICRS degrees.

    Returns
    -------
    astropy.coordinates.SkyCoord
        Scalar ICRS coordinate on the local meridian, at exactly
        `dec_deg`. See the module docstring for why the declination is
        carried through unchanged while the right ascension is precessed.

    Notes
    -----
    The right ascension is *solved for* rather than converted. Precession
    mixes right ascension and declination, so simply converting the pair
    ``(sidereal time, dec)`` from the equator of date to ICRS and keeping
    its right ascension leaves the pointing a few arcminutes off the
    meridian -- a tenth of the field. Instead this asks the opposite
    question: which ICRS right ascension, paired with this declination,
    *lands* on the meridian? Two Newton steps answer it to well under an
    arcsecond, because the map from one to the other is the identity plus
    a slowly-varying offset.
    """
    dec = float(dec_deg) * u.deg
    lst_deg = float(time.sidereal_time("apparent", longitude=location.lon).to_value(u.deg))
    ra_deg = lst_deg
    for _ in range(2):
        guess = SkyCoord(ra=ra_deg * u.deg, dec=dec, frame="icrs")
        of_date = guess.transform_to(FK5(equinox=time))
        residual = (lst_deg - float(of_date.ra.deg) + 180.0) % 360.0 - 180.0
        ra_deg = (ra_deg + residual) % 360.0
    return SkyCoord(ra=ra_deg * u.deg, dec=dec, frame="icrs")


def _altitude_deg(coord: SkyCoord, time: Time, location: EarthLocation) -> np.ndarray:
    """Altitude above the horizon, degrees. No refraction (see `delays`)."""
    altaz = coord.transform_to(AltAz(obstime=time, location=location))
    return np.atleast_1d(altaz.alt.to_value(u.deg))


def catalog_sources_in_field(
    phase_center: SkyCoord, time: Time, location: EarthLocation
) -> list[dict[str, Any]]:
    """Which catalogue sources this frame actually sees.

    A source counts when it lands inside the imaged field -- the same
    ``max(|l|, |m|) <= half the field of view`` test the single run's
    ``in_field`` uses -- *and* is above the horizon. The horizon test is
    not redundant: a pointing declination far from the site latitude puts
    the whole field under the ground, and a source there must not appear.

    Returns
    -------
    list of dict
        ``name``, ``flux_jy``, ``ra_deg``, ``dec_deg``, ``l``, ``m`` and
        ``altitude_deg`` for each source in the field, brightest first.
    """
    entries: list[dict[str, Any]] = []
    candidates = [
        (
            SkyCoord(ra=item["ra_deg"] * u.deg, dec=item["dec_deg"] * u.deg, frame="icrs"),
            item["name"],
            item["flux_jy"],
        )
        for item in CATALOG_SOURCES
    ]
    # `get_sun` answers in GCRS -- geocentric, on axes aligned with ICRS.
    # Its right ascension and declination are therefore used as they
    # stand, and NOT converted with `.icrs`: that conversion shifts the
    # origin to the solar-system barycentre and returns the direction of
    # the anti-Sun. Reading GCRS angles as ICRS ones costs the ~20
    # arcseconds of aberration and light deflection, four hundred times
    # less than this field's half width.
    sun = get_sun(time)
    candidates.append((SkyCoord(ra=sun.ra, dec=sun.dec, frame="icrs"), SUN_NAME, QUIET_SUN_FLUX_JY))

    for coord, name, flux_jy in candidates:
        l_dir, m_dir = lm_from_radec(phase_center, coord)
        if max(abs(float(l_dir)), abs(float(m_dir))) > 0.5 * IMAGE_FIELD_OF_VIEW_RAD:
            continue
        altitude = float(_altitude_deg(coord, time, location)[0])
        if altitude <= 0.0:
            continue
        entries.append(
            {
                "name": name,
                "flux_jy": float(flux_jy),
                "ra_deg": float(coord.ra.deg),
                "dec_deg": float(coord.dec.deg),
                "l": float(l_dir),
                "m": float(m_dir),
                "altitude_deg": round(altitude, 3),
            }
        )
    entries.sort(key=lambda entry: -entry["flux_jy"])
    return entries


# ----------------------------------------------------------------------
# The request
# ----------------------------------------------------------------------
def _check_date(value: str) -> str:
    try:
        Time(f"{value}T00:00:00", scale="utc")
    except Exception as exc:
        raise ValueError(f"give the date as YYYY-MM-DD, not {value!r} ({exc})") from exc
    return value


class DayRequest(BaseModel):
    """One simulated day: a setup, a strip of sky, and a date."""

    model_config = ConfigDict(extra="forbid")

    setup: SimulateRequest
    date: str = Field(default="2026-07-30", max_length=32)
    pointing_dec_deg: float = Field(default=37.234, ge=-90.0, le=90.0)
    n_frames: int = Field(default=DEFAULT_FRAMES, ge=1, le=MAX_FRAMES)
    resolution: Literal["coarse", "fine"] = "coarse"
    carry_setup_sources: bool = False
    """bool: Whether the setup's own sky sources ride along.

    Off by default, and that is a physics decision rather than a taste
    one: a setup source placed as an offset from the pointing would follow
    the pointing around the sky all day, which no real source does. The
    catalogue is the movie's sky."""

    @field_validator("date")
    @classmethod
    def _validate_date(cls, value: str) -> str:
        return _check_date(value)

    def frame_times(self) -> Time:
        """UTC instants of every frame: the calendar day, evenly split."""
        start = Time(f"{self.date}T00:00:00", scale="utc")
        step = TimeDelta(86400.0 / self.n_frames, format="sec")
        return start + step * np.arange(self.n_frames)

    def frame_setup(self, index: int) -> SimulateRequest:
        """The setup frame `index` is simulated with.

        Every frame is an independent seeded simulation: the seed is the
        setup's own seed plus the frame index, so two frames never draw
        the same noise and the whole day is reproducible.
        """
        data = self.setup.model_dump()
        data["sim"] = dict(data["sim"])
        data["sim"]["seed"] = (int(self.setup.sim.seed) + int(index)) % (2**31 - 1)
        if self.resolution == "coarse":
            data["sim"]["n_chan"] = min(int(self.setup.sim.n_chan), COARSE_N_CHAN)
            data["sim"]["n_blocks"] = COARSE_N_BLOCKS
        if not self.carry_setup_sources:
            data["sky_sources"] = []
        return SimulateRequest.model_validate(data)


# ----------------------------------------------------------------------
# One frame
# ----------------------------------------------------------------------
def compute_frame(spec: dict[str, Any], index: int) -> dict[str, Any]:
    """Simulate, correlate and image one frame of a day.

    Parameters
    ----------
    spec : dict
        A `DayRequest` as plain data -- this runs in another process, so
        the argument has to pickle without dragging pydantic models or
        astropy objects across.
    index : int
        Which frame.

    Returns
    -------
    dict
        The reduced frame: a 64x64 image, the instant, the pointing, and
        which catalogue sources were in the field. A frame that fails
        returns the same shape with ``error`` set and no image, so one bad
        frame costs one frame rather than the day.
    """
    request = DayRequest.model_validate(spec)
    started = _time.perf_counter()
    try:
        time = request.frame_times()[index]
        location = site_location(request.setup)
        phase_center = meridian_phase_center(location, time, request.pointing_dec_deg)
        in_field = catalog_sources_in_field(phase_center, time, location)
        extra = [
            PointSource(
                flux_jy=entry["flux_jy"],
                coord=SkyCoord(
                    ra=entry["ra_deg"] * u.deg, dec=entry["dec_deg"] * u.deg, frame="icrs"
                ),
                name=entry["name"],
            )
            for entry in in_field
        ]

        setup = request.frame_setup(index)
        with warnings.catch_warnings():
            # A meridian pointing away from the zenith leaves a real w
            # term, which the direct-DFT imager warns about once per
            # frame. It is expected here rather than a fault, and the
            # frame reports the pointing's zenith angle instead so the
            # page can say how far off-zenith the strip is.
            warnings.simplefilter("ignore")
            simulator = build_simulator(
                setup, start_time=time, phase_center=phase_center, extra_sources=extra
            )
            visibilities = correlate(simulator.blocks())
            step = max(1, simulator.n_chan // IMAGE_MAX_CHANNELS)
            image, _, _ = dirty_image(
                visibilities,
                field_of_view_rad=IMAGE_FIELD_OF_VIEW_RAD,
                n_pix=IMAGE_N_PIX,
                channels=slice(None, None, step),
                warn_on_w_term=False,
            )
        altitude = float(_altitude_deg(phase_center, time, location)[0])
        return {
            "index": index,
            "utc": time.isot,
            "lst_deg": float(
                time.sidereal_time("apparent", longitude=location.lon).to_value(u.deg)
            ),
            "ra_deg": float(phase_center.ra.deg),
            "dec_deg": float(phase_center.dec.deg),
            "altitude_deg": round(altitude, 3),
            "sources": in_field,
            # Kept as a float32 array rather than nested lists: a day of
            # 288 frames is 4.7 MB this way and 37 MB as Python floats,
            # and four days are held at once. It becomes lists in
            # `day_frame`, one frame at a time, on its way to the browser.
            "image": image.astype(np.float32),
            "vmin_jy": float(np.round(image.min(), FRAME_IMAGE_DECIMALS)),
            "vmax_jy": float(np.round(image.max(), FRAME_IMAGE_DECIMALS)),
            # The root-mean-square of the map. On an empty frame this is
            # the noise level; it is what the day's brightness stretch is
            # anchored to, so that a frame with nothing in it still shows
            # its own structure rather than going flat black.
            "rms_jy": float(np.sqrt(np.mean(np.square(image, dtype=np.float64)))),
            "wall_time_s": round(_time.perf_counter() - started, 3),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - one frame's fault, not the day's
        return {
            "index": index,
            "utc": None,
            "sources": [],
            "image": None,
            "vmin_jy": None,
            "vmax_jy": None,
            "rms_jy": None,
            "wall_time_s": round(_time.perf_counter() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


# ----------------------------------------------------------------------
# The job store
# ----------------------------------------------------------------------
class _DayJob:
    """One day being built, or built. Guarded by `_JobStore`'s lock."""

    def __init__(self, identifier: str, request: DayRequest) -> None:
        self.id = identifier
        self.request = request
        self.total = request.n_frames
        self.frames: list[dict[str, Any] | None] = [None] * self.total
        self.state = "building"
        self.done = 0
        self.failed = 0
        self.created = _time.time()
        self.touched = self.created
        self.cancelled = threading.Event()
        self.error: str | None = None


class _JobStore:
    """A handful of days, oldest and least recently touched dropped first.

    Not a cache with a background sweeper: expiry is checked whenever the
    store is touched, which is enough for a store whose entries are only
    ever created by a request.
    """

    def __init__(self, max_jobs: int = JOB_MAX, ttl_s: float = JOB_TTL_S) -> None:
        self.max_jobs = max_jobs
        self.ttl_s = ttl_s
        self._jobs: dict[str, _DayJob] = {}
        self._lock = threading.RLock()

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def _evict(self) -> None:
        now = _time.time()
        for identifier, job in list(self._jobs.items()):
            if now - job.touched > self.ttl_s:
                job.cancelled.set()
                del self._jobs[identifier]
        while len(self._jobs) > self.max_jobs:
            oldest = min(self._jobs.values(), key=lambda job: job.touched)
            oldest.cancelled.set()
            del self._jobs[oldest.id]

    def add(self, job: _DayJob) -> None:
        with self._lock:
            self._jobs[job.id] = job
            self._evict()

    def get(self, identifier: str) -> _DayJob | None:
        with self._lock:
            job = self._jobs.get(identifier)
            if job is None:
                return None
            job.touched = _time.time()
            self._evict()
            return self._jobs.get(identifier)

    def clear(self) -> None:
        with self._lock:
            for job in self._jobs.values():
                job.cancelled.set()
            self._jobs.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._jobs)


jobs = _JobStore()
"""_JobStore: The process-wide store of simulated days."""


def _drive(job: _DayJob, max_workers: int) -> None:
    """Compute every frame of `job` in a pool, filling slots as they land."""
    spec = job.request.model_dump()
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(compute_frame, spec, index): index for index in range(job.total)
            }
            # Frames are collected as they land rather than in order, so
            # the progress the page shows is the work actually finished.
            for future in as_completed(futures):
                index = futures[future]
                if job.cancelled.is_set():
                    for pending in futures:
                        pending.cancel()
                    break
                try:
                    frame = future.result()
                except Exception as exc:  # noqa: BLE001 - a worker that died
                    frame = {
                        "index": index,
                        "utc": None,
                        "sources": [],
                        "image": None,
                        "vmin_jy": None,
                        "vmax_jy": None,
                        "rms_jy": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                with jobs.lock:
                    job.frames[index] = frame
                    job.done += 1
                    if frame.get("error"):
                        job.failed += 1
    except Exception as exc:  # noqa: BLE001 - the pool itself failed
        with jobs.lock:
            job.state = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
        return
    with jobs.lock:
        if job.cancelled.is_set():
            job.state = "cancelled"
        elif job.state == "building":
            job.state = "done"


def start_day(request: DayRequest, *, max_workers: int | None = None) -> str:
    """Start building a day in the background.

    Parameters
    ----------
    request : DayRequest
        Validated.
    max_workers : int, optional
        Processes to use. Defaults to `day_workers`; the tests pass a
        small number so the suite does not fork a dozen interpreters.

    Returns
    -------
    str
        The job identifier to poll.
    """
    job = _DayJob(uuid.uuid4().hex, request)
    jobs.add(job)
    workers = day_workers() if max_workers is None else max(1, int(max_workers))
    thread = threading.Thread(
        target=_drive, args=(job, min(workers, job.total)), name=f"day-{job.id[:8]}", daemon=True
    )
    thread.start()
    return job.id


def _frame_meta(frame: dict[str, Any] | None) -> dict[str, Any] | None:
    """One frame without its image -- what the status poll carries."""
    if frame is None:
        return None
    return {key: value for key, value in frame.items() if key != "image"}


def day_status(identifier: str) -> dict[str, Any] | None:
    """Progress and per-frame metadata, without any images.

    The colour scale the page draws with is fixed for the whole day, so
    the running maximum over the frames finished so far travels here: the
    page can start playing before the day is built and still be drawing on
    a scale that only ever grows.

    ``scale_soft_jy`` goes with it. A day spans six orders of magnitude --
    an empty field is a fraction of a millijansky of noise and Cygnus A is
    fifteen hundred janskys -- so a linear scale that can show the source
    shows every other frame as black. It is the median of the frames'
    root-mean-square levels, i.e. the noise of a typical frame, and the
    page uses it as the soft point of an arcsinh stretch. The stretch is
    one fixed mapping for the whole day, so a frame going brighter still
    means the sky went brighter.
    """
    job = jobs.get(identifier)
    if job is None:
        return None
    with jobs.lock:
        frames = [_frame_meta(frame) for frame in job.frames]
        maxima = [
            frame["vmax_jy"] for frame in job.frames if frame and frame.get("vmax_jy") is not None
        ]
        levels = [
            frame["rms_jy"]
            for frame in job.frames
            if frame and frame.get("rms_jy") is not None and frame["rms_jy"] > 0.0
        ]
        soft = float(np.median(levels)) if levels else None
        return {
            "id": job.id,
            "state": job.state,
            "done": job.done,
            "failed": job.failed,
            "total": job.total,
            "error": job.error,
            "date": job.request.date,
            "pointing_dec_deg": job.request.pointing_dec_deg,
            "resolution": job.request.resolution,
            "carry_setup_sources": job.request.carry_setup_sources,
            "field_half_width_deg": IMAGE_FIELD_HALF_WIDTH_DEG,
            "scale_max_jy": max(maxima) if maxima else None,
            "scale_soft_jy": soft,
            "frames": frames,
        }


def day_frame(identifier: str, index: int) -> dict[str, Any] | None:
    """One finished frame in full, image included.

    ``None`` when there is no such job; a frame that is not finished yet
    comes back as ``{"pending": True}`` rather than as an error, because
    "not yet" is the normal answer while a day is still building.
    """
    job = jobs.get(identifier)
    if job is None:
        return None
    with jobs.lock:
        if not 0 <= index < job.total:
            return None
        frame = job.frames[index]
        if frame is None:
            return {"index": index, "pending": True}
        payload = dict(frame)
        image = payload.get("image")
        payload["image"] = (
            None if image is None else np.round(np.asarray(image), FRAME_IMAGE_DECIMALS).tolist()
        )
        payload["pending"] = False
        payload["n_pix"] = IMAGE_N_PIX
        payload["field_of_view_rad"] = IMAGE_FIELD_OF_VIEW_RAD
        return payload


def cancel_day(identifier: str) -> dict[str, Any] | None:
    """Stop building a day. Frames already finished stay readable."""
    job = jobs.get(identifier)
    if job is None:
        return None
    with jobs.lock:
        job.cancelled.set()
        if job.state == "building":
            job.state = "cancelling"
        return {"id": job.id, "state": job.state, "done": job.done, "total": job.total}


# ----------------------------------------------------------------------
# The timeline
# ----------------------------------------------------------------------
SUN_SAMPLE_STEP_S = 300.0
"""float: Spacing of the sun-altitude curve the timeline is shaded from,
seconds. Five minutes resolves sunrise to about a minute after linear
interpolation, which is finer than a band a thousand pixels wide can
draw."""

SATELLITE_SAMPLE_STEP_S = 60.0
"""float: Spacing at which an element set is propagated across the day.

A low orbiter crosses a 2.3-degree field in roughly twenty seconds, so a
one-minute grid will miss some passes outright. That is a deliberate
trade: it is a marker strip, not a scheduling tool, and propagating every
twenty seconds costs four thousand transforms per request."""

SIDEREAL_DAY_HOURS = 23.9344696


def _crossings(times: Time, values: np.ndarray, rising: bool) -> list[str]:
    """Instants where a sampled curve crosses zero, linearly interpolated."""
    out: list[str] = []
    for index in range(len(values) - 1):
        low, high = float(values[index]), float(values[index + 1])
        if rising and not (low < 0.0 <= high):
            continue
        if not rising and not (low >= 0.0 > high):
            continue
        span = high - low
        fraction = 0.0 if span == 0.0 else (0.0 - low) / span
        out.append((times[index] + (times[index + 1] - times[index]) * fraction).isot)
    return out


def timeline_payload(
    date: str,
    dec_deg: float,
    latitude_deg: float,
    longitude_deg: float,
    height_m: float = 0.0,
    *,
    tle_text: str = "",
) -> dict[str, Any]:
    """Everything the day's timeline band draws, in one cheap call.

    Parameters
    ----------
    date : str
        Calendar day, ``YYYY-MM-DD`` UTC.
    dec_deg : float
        Pointing declination of the strip, ICRS degrees.
    latitude_deg, longitude_deg, height_m : float
        The site.
    tle_text : str, optional
        A two-line element set from the setup's satellite source, if there
        is one. Its passes through the strip come back as intervals.

    Returns
    -------
    dict
        ``sun`` (an altitude curve to shade night from, plus sunrise,
        sunset and transit), ``sources`` (each catalogue source's transit
        instants and whether this strip ever sees it), ``satellite``
        (field-crossing intervals) and the day's bounds. Every instant is
        also given as ``fraction``, its position in the day from 0 to 1,
        because that is what the band actually needs.
    """
    _check_date(date)
    array = ArrayConfig(
        antenna_positions_enu_m=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        latitude_deg=float(latitude_deg),
        longitude_deg=float(longitude_deg),
        height_m=float(height_m),
    )
    location = earth_location(array)

    start = Time(f"{date}T00:00:00", scale="utc")
    end = start + TimeDelta(86400.0, format="sec")

    def fraction_of(instant: Time | str) -> float:
        moment = Time(instant, scale="utc") if isinstance(instant, str) else instant
        return float((moment - start).sec / 86400.0)

    # --- the sun ------------------------------------------------------
    n_samples = int(round(86400.0 / SUN_SAMPLE_STEP_S)) + 1
    sample_times = start + TimeDelta(np.linspace(0.0, 86400.0, n_samples), format="sec")
    sun = get_sun(sample_times)
    sun_altaz = sun.transform_to(AltAz(obstime=sample_times, location=location))
    sun_alt_deg = sun_altaz.alt.to_value(u.deg)

    peak = int(np.argmax(sun_alt_deg))
    sun_payload = {
        "altitude_deg": [round(float(value), 3) for value in sun_alt_deg],
        "sample_fractions": [round(index / (n_samples - 1), 6) for index in range(n_samples)],
        "sunrise_utc": _crossings(sample_times, sun_alt_deg, rising=True),
        "sunset_utc": _crossings(sample_times, sun_alt_deg, rising=False),
        "transit_utc": sample_times[peak].isot,
        "max_altitude_deg": round(float(sun_alt_deg[peak]), 3),
        "always_up": bool(np.all(sun_alt_deg > 0.0)),
        "always_down": bool(np.all(sun_alt_deg <= 0.0)),
    }
    sun_payload["sunrise"] = [fraction_of(value) for value in sun_payload["sunrise_utc"]]
    sun_payload["sunset"] = [fraction_of(value) for value in sun_payload["sunset_utc"]]
    sun_payload["transit"] = fraction_of(sun_payload["transit_utc"])
    # The Sun's declination moves a fraction of a degree a day, so which
    # strip it is in is a property of the date; the page offers it as a
    # shortcut chip. Taken at the middle of the day, and read straight off
    # the GCRS answer -- see `catalog_sources_in_field` for why it is not
    # converted to ICRS.
    sun_payload["dec_deg"] = round(
        float(get_sun(start + TimeDelta(43200.0, format="sec")).dec.deg), 4
    )
    sun_payload["in_field"] = bool(
        abs(sun_payload["dec_deg"] - float(dec_deg)) <= IMAGE_FIELD_HALF_WIDTH_DEG
    )

    # --- the catalogue ------------------------------------------------
    lst_start_deg = float(start.sidereal_time("apparent", longitude=location.lon).to_value(u.deg))
    sources = []
    for item in CATALOG_SOURCES:
        # A source transits when the local sidereal time reaches its right
        # ascension. Sidereal time runs fast against the clock by the ratio
        # of the solar to the sidereal day, so a calendar day holds one
        # transit, or two when the first falls in the first four minutes.
        offset_deg = (item["ra_deg"] - lst_start_deg) % 360.0
        first_hours = offset_deg / 360.0 * SIDEREAL_DAY_HOURS
        transits_hours = [first_hours]
        if first_hours + SIDEREAL_DAY_HOURS < 24.0:
            transits_hours.append(first_hours + SIDEREAL_DAY_HOURS)
        instants = [start + TimeDelta(hours * 3600.0, format="sec") for hours in transits_hours]
        # At transit the source sits at l = 0, so whether the strip ever
        # sees it is entirely a question of declination.
        separation_deg = abs(item["dec_deg"] - float(dec_deg))
        in_field = separation_deg <= IMAGE_FIELD_HALF_WIDTH_DEG
        altitudes = [
            float(
                _altitude_deg(
                    SkyCoord(
                        ra=item["ra_deg"] * u.deg, dec=item["dec_deg"] * u.deg, frame="icrs"
                    ),
                    instant,
                    location,
                )[0]
            )
            for instant in instants
        ]
        sources.append(
            {
                "name": item["name"],
                "ra_deg": item["ra_deg"],
                "dec_deg": item["dec_deg"],
                "flux_jy": item["flux_jy"],
                "in_field": in_field,
                "dec_offset_deg": round(item["dec_deg"] - float(dec_deg), 4),
                "transits_utc": [instant.isot for instant in instants],
                "transits": [fraction_of(instant) for instant in instants],
                "transit_altitude_deg": [round(value, 3) for value in altitudes],
            }
        )

    payload = {
        "date": date,
        "start_utc": start.isot,
        "end_utc": end.isot,
        "pointing_dec_deg": float(dec_deg),
        "latitude_deg": float(latitude_deg),
        "longitude_deg": float(longitude_deg),
        "height_m": float(height_m),
        "field_half_width_deg": IMAGE_FIELD_HALF_WIDTH_DEG,
        # How long anything at this declination stays in the field as the
        # sky drifts through it. The sky turns 15.041 degrees of right
        # ascension an hour, and a degree of right ascension is
        # ``cos(dec)`` degrees on the sky, so a narrow strip near the pole
        # is crossed slowly and one on the equator quickly. This is the
        # number that decides whether a given frame cadence can see
        # anything at all, so the page prints it next to the frame count.
        "field_crossing_minutes": round(
            2.0
            * IMAGE_FIELD_HALF_WIDTH_DEG
            / max(math.cos(math.radians(float(dec_deg))), 1.0e-6)
            / 15.041
            * 60.0,
            2,
        ),
        "pointing_zenith_angle_deg": round(abs(float(latitude_deg) - float(dec_deg)), 4),
        "sun": sun_payload,
        "sources": sources,
        "satellite": _satellite_passes(
            tle_text, start, location, float(latitude_deg), float(dec_deg)
        ),
    }
    return payload


def _pointing_enu(latitude_deg: float, dec_deg: float) -> np.ndarray:
    """Unit vector the strip points along, in the site's ENU frame.

    A transit pointing does not move: the meridian at declination `dec_deg`
    sits at zenith angle ``|latitude - dec|``, due south when the strip is
    south of the zenith and due north when it is north. That is why a
    satellite's passes through the field can be found from geometry alone,
    without an image.
    """
    zenith_angle = math.radians(abs(latitude_deg - dec_deg))
    azimuth = math.radians(180.0 if dec_deg < latitude_deg else 0.0)
    altitude = math.pi / 2.0 - zenith_angle
    return np.asarray(
        [
            math.cos(altitude) * math.sin(azimuth),
            math.cos(altitude) * math.cos(azimuth),
            math.sin(altitude),
        ]
    )


def _satellite_passes(
    tle_text: str, start: Time, location: EarthLocation, latitude_deg: float, dec_deg: float
) -> dict[str, Any]:
    """When the setup's satellite crosses the strip, as intervals."""
    if not tle_text.strip():
        return {"configured": False, "name": None, "passes": [], "note": None}

    from rfi_simulator.satellites import TwoLineElement

    try:
        element_set = TwoLineElement.from_string(tle_text)
    except Exception as exc:  # noqa: BLE001 - a paste, not a program
        return {
            "configured": True,
            "name": None,
            "passes": [],
            "note": f"could not read the element set ({exc})",
        }

    n_samples = int(round(86400.0 / SATELLITE_SAMPLE_STEP_S)) + 1
    times = start + TimeDelta(np.linspace(0.0, 86400.0, n_samples), format="sec")
    try:
        positions = element_set.enu_position_m(times, location)
    except Exception as exc:  # noqa: BLE001 - a decayed or far-from-epoch orbit
        return {
            "configured": True,
            "name": element_set.name or None,
            "passes": [],
            "note": f"could not propagate this element set across the day ({exc})",
        }

    ranges = np.linalg.norm(positions, axis=-1)
    ranges[ranges == 0.0] = 1.0
    directions = positions / ranges[:, None]
    pointing = _pointing_enu(latitude_deg, dec_deg)
    cosines = np.clip(directions @ pointing, -1.0, 1.0)
    inside = np.degrees(np.arccos(cosines)) <= IMAGE_FIELD_HALF_WIDTH_DEG

    passes: list[dict[str, Any]] = []
    index = 0
    while index < len(inside):
        if not inside[index]:
            index += 1
            continue
        run_start = index
        while index + 1 < len(inside) and inside[index + 1]:
            index += 1
        passes.append(
            {
                "start_utc": times[run_start].isot,
                "end_utc": times[index].isot,
                "start": float((times[run_start] - start).sec / 86400.0),
                "end": float((times[index] - start).sec / 86400.0),
            }
        )
        index += 1

    note = None
    if not passes:
        note = "this element set never crosses the strip on this date"
    return {
        "configured": True,
        "name": element_set.name or None,
        "passes": passes,
        "sample_step_s": SATELLITE_SAMPLE_STEP_S,
        "note": note,
    }
