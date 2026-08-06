"""HTTP surface for the browser front end.

`create_app` builds the application without starting anything, which is
what the tests use; `main` is the ``rfi-simulator-ui`` console entry
point.

The server binds to the loopback interface by default. The page it serves
never reaches the network: every asset comes from this process, and the
interactive API documentation, which would pull its viewer from a content
delivery network, is switched off. The machine-readable schema stays at
``/api/openapi.json``.

The server reaches the network in exactly one place -- ``GET
/api/sky/now``, whose live aircraft layer is fetched from a public
aggregator with a hard timeout and degrades to an empty layer with a
status when it cannot be reached (see `rfi_simulator.webui.skynow`).
Nothing a simulation depends on is ever fetched: element sets are pasted
in or taken from the bundle.
"""

from __future__ import annotations

import argparse
import os
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi import Path as PathParam
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from rfi_simulator import __version__
from rfi_simulator.webui.observatory import (
    DayBusyError,
    DayRequest,
    cancel_day,
    day_frame,
    day_status,
    start_day,
    timeline_payload,
)
from rfi_simulator.webui.simulate import (
    ARRAY_DIR_ENV_VAR,
    FlagRequest,
    SimulateRequest,
    array_detail,
    array_summaries,
    default_array,
    defaults_payload,
    pointing_payload,
    run_flaggers,
    run_simulation,
)
from rfi_simulator.webui.skynow import sky_now

__all__ = ["create_app", "main"]

STATIC_DIR = Path(__file__).resolve().parent / "static"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

HOST_ENV_VAR = "RFI_SIMULATOR_UI_HOST"
"""str: Environment variable naming the bound interface.

`main` sets it before handing the application to the server by import
string, which is the only way the value survives into the process the
reloader starts."""

LOCAL_HOSTS = ["127.0.0.1", "localhost", "testserver"]
"""list of str: Host headers served without argument. Anything else has to
be the interface the server was actually asked to bind."""

WILDCARD_HOSTS = {"0.0.0.0", "::", ""}  # noqa: S104 - recognised, not a default
"""set of str: Bind addresses that mean "every interface", for which no
host check is possible."""

MAX_REQUEST_BYTES = 2_000_000
"""int: Largest request body accepted. A full array of the largest size
this front end runs is a few kilobytes, so this is generous by three
orders of magnitude and still refuses a body big enough to matter."""

MAX_CONCURRENT_SIMULATIONS = 1
"""int: Runs allowed in flight at once.

One run is bounded by the request model's size cap; unbounded concurrency
would multiply that bound by however many requests happen to arrive, so
the rest queue instead. Single flight also keeps warning capture correct:
`warnings.catch_warnings` swaps interpreter-global state, so two runs
recording at once could attribute one request's warnings to the other.

The slot is taken non-blocking, not queued: a client that cannot get it
is told so with a 429 rather than parked on a worker thread. Threaded
servers have a finite thread pool, and a client that queues rather than
being refused holds one of those threads for as long as the run in front
of it takes; forty queued clicks was enough to freeze the page behind
threads that were all just waiting.

Deliberately NOT shared with the observatory day pool
(`rfi_simulator.webui.observatory._day_slots`, gating
``POST /api/observatory/day``): a day build is minutes of work in a
background process pool, and making an interactive run wait behind it --
which is what one shared semaphore would do -- would be worse than the
alternative. The day pool has its own, independent admission control
(`day_max_concurrent`, default one build at a time) plus its own cost
budget (`DAY_COST_BUDGET`) bounding what one build may cost. The two
pools' worst case is therefore not unbounded: it is one interactive run
here plus one day build there, each bounded on its own terms, running at
once. See `DEFAULT_DAY_MAX_CONCURRENT` for the other half of this
accounting."""

_simulation_slots = threading.BoundedSemaphore(MAX_CONCURRENT_SIMULATIONS)


def _busy_response(message: str) -> JSONResponse:
    """A 429 in the same ``{detail: [...]}`` shape as a 422, for the page."""
    return JSONResponse(
        status_code=429,
        content={"detail": [{"loc": ["body"], "msg": message, "type": "value_error"}]},
    )


class ContentLengthLimitMiddleware:
    """Refuse an over-long request body, however it is framed.

    A declared ``Content-Length`` beyond the limit is answered from the
    header alone, before any body is read. A body without a declared
    length (chunked transfer encoding, or a header that lies) is counted
    as it streams and cut off at the same limit -- the framing must not
    matter, because the point is to bound what a request can make this
    process buffer.
    """

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_REQUEST_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    def _refusal(self, received: int) -> JSONResponse:
        return JSONResponse(
            status_code=413,
            content={
                "detail": [
                    {
                        "loc": ["body"],
                        "msg": (
                            f"this request is {received} bytes, more than the "
                            f"{self.max_bytes} this server accepts"
                        ),
                        "type": "value_error",
                    }
                ]
            },
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        for name, value in scope.get("headers", ()):
            if name.lower() != b"content-length":
                continue
            try:
                declared = int(value)
            except ValueError:
                break
            if declared > self.max_bytes:
                await self._refusal(declared)(scope, receive, send)
                return
            break

        received = 0

        async def counted_receive() -> Any:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    # HTTPException, because the body is read inside the
                    # routing layer, which turns any other exception into
                    # a generic 400 before it could reach this middleware.
                    raise HTTPException(
                        status_code=413,
                        detail=[
                            {
                                "loc": ["body"],
                                "msg": (
                                    f"this request is over {received} bytes, more than "
                                    f"the {self.max_bytes} this server accepts"
                                ),
                                "type": "value_error",
                            }
                        ],
                    )
            return message

        await self.app(scope, counted_receive, send)


def create_app(host: str | None = None, array_dir: str | Path | None = None) -> FastAPI:
    """Build the application.

    Parameters
    ----------
    host : str, optional
        The interface the server is bound to, added to the accepted
        ``Host`` headers. Loopback names are always accepted. Defaults to
        `HOST_ENV_VAR` in the environment, which is how `main` passes the
        bound interface through the reloader's fresh process.
    array_dir : str or pathlib.Path, optional
        A directory of extra array configurations to offer alongside the
        bundled ones. Defaults to `ARRAY_DIR_ENV_VAR` in the environment.
        Only this one directory is ever read, and clients name entries by
        an identifier this process assigned, never by path.

    Returns
    -------
    fastapi.FastAPI
        With ``GET /api/defaults``, ``GET /api/pointing``,
        ``GET /api/arrays``, ``GET /api/arrays/{array_id}``,
        ``POST /api/simulate``, ``POST /api/flag``, the observatory day
        (``POST /api/observatory/day``, ``GET /api/observatory/day/{id}``,
        ``GET /api/observatory/day/{id}/frame/{i}``,
        ``POST /api/observatory/day/{id}/cancel``,
        ``GET /api/observatory/timeline``), the live monitor
        (``GET /api/sky/now``), and the page itself at ``/``.
    """
    app = FastAPI(
        title="Interference simulator",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )

    if host is None:
        host = os.environ.get(HOST_ENV_VAR) or None
    if array_dir is None:
        array_dir = os.environ.get(ARRAY_DIR_ENV_VAR) or None

    allowed_hosts = list(LOCAL_HOSTS)
    if host in WILDCARD_HOSTS:
        # Binding every interface is a deliberate choice to be reachable
        # under whatever name the operator uses; there is nothing left to
        # check the header against.
        allowed_hosts = ["*"]
    elif host and host not in allowed_hosts:
        allowed_hosts.append(host)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
    app.add_middleware(ContentLengthLimitMiddleware)

    @app.exception_handler(RequestValidationError)
    def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Report where and what, and nothing else.

        The default handler echoes the offending input back, which is both
        noise for the form and a hazard: a NaN in the request is not
        representable in JSON and the response itself then fails to
        encode. Only the location and the message are returned.
        """
        detail = [
            {
                "loc": [str(part) for part in error.get("loc", ())],
                "msg": error.get("msg", "invalid value"),
                "type": error.get("type", "value_error"),
            }
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": detail})

    @app.get("/api/defaults")
    def get_defaults() -> dict[str, Any]:
        """Default array, default observation, guard rails and form schemas."""
        return defaults_payload()

    @app.get("/api/pointing")
    def get_pointing(
        latitude_deg: float | None = Query(default=None, ge=-90.0, le=90.0),
        longitude_deg: float | None = Query(default=None, ge=-360.0, le=360.0),
        height_m: float | None = Query(default=None, ge=-500.0, le=1.0e4),
    ) -> dict[str, Any]:
        """Where a run from this site points, and how wide the image is.

        The page asks again whenever the site changes -- loading another
        array moves the zenith, and the honest bounds it quotes for source
        placement move with it.
        """
        return pointing_payload(latitude_deg, longitude_deg, height_m)

    @app.get("/api/arrays")
    def get_arrays() -> list[dict[str, Any]]:
        """Array layouts this server can offer, without their positions."""
        return array_summaries(array_dir)

    @app.get("/api/arrays/{array_id}")
    def get_array(
        array_id: str = PathParam(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$"),
    ) -> dict[str, Any]:
        """One layout in full: antennas and the site they stand on.

        `array_id` is matched against identifiers this process handed out
        in `get_arrays`; it never reaches the filesystem, so there is no
        path for a request to traverse. Anything unknown is a 404.
        """
        payload = array_detail(array_id, array_dir)
        if payload is None:
            raise HTTPException(status_code=404, detail=f"no array layout called {array_id!r}")
        return payload

    @app.post("/api/simulate")
    def post_simulate(
        request: SimulateRequest,
        pol: int = Query(default=0, ge=0, le=1, description="Receptor the waterfall shows"),
    ) -> Any:
        """Run one observation and return everything the page draws.

        A `ValueError` from the library -- a transmitter tuned outside the
        band is the common one -- comes back as a 422 with the library's
        own message, because it is a fault in the setup the user typed,
        not in the server.

        Bounded by `MAX_CONCURRENT_SIMULATIONS`: the request model bounds
        the memory of one run, and this is what keeps several arriving at
        once from multiplying that bound. A request that arrives while the
        slot is taken is refused with a 429 rather than queued, so it never
        holds a server thread waiting for someone else's run.

        `pol` selects which receptor the waterfall display shows for a
        dual-polarization run (``n_pol=2``); it is ignored (fixed to 0)
        for a single-polarization one. The dirty image always images
        Stokes I regardless of `pol` -- see `run_simulation`.
        """
        if not _simulation_slots.acquire(blocking=False):
            return _busy_response(
                "a simulation is already running -- wait for it to finish and try again"
            )
        try:
            return run_simulation(request, pol=pol)
        except ValueError as exc:
            return JSONResponse(
                status_code=422,
                content={"detail": [{"loc": ["body"], "msg": str(exc), "type": "value_error"}]},
            )
        finally:
            _simulation_slots.release()

    @app.post("/api/flag")
    def post_flag(request: FlagRequest) -> Any:
        """Score one or two classical flaggers against a run's ground truth.

        The body carries the whole observation again rather than the
        identifier of an earlier one: this server keeps no per-client
        state, and a run is reproducible from its seed, so re-simulating
        is what makes the flagged data provably the same data the page is
        already showing. It shares the same `MAX_CONCURRENT_SIMULATIONS`
        slot as a run, because it is one, and the same
        429-rather-than-queue rule: a slot already taken is refused, not
        waited for.

        A `ValueError` -- an antenna that does not exist in this array, an
        accumulation that would build too large a grid, or anything the
        library refuses -- comes back as a 422 with its own message.
        """
        if not _simulation_slots.acquire(blocking=False):
            return _busy_response(
                "a simulation is already running -- wait for it to finish and try again"
            )
        try:
            return run_flaggers(request)
        except ValueError as exc:
            return JSONResponse(
                status_code=422,
                content={"detail": [{"loc": ["body"], "msg": str(exc), "type": "value_error"}]},
            )
        finally:
            _simulation_slots.release()

    @app.post("/api/observatory/day")
    def post_observatory_day(request: DayRequest) -> Any:
        """Start building a simulated day, and answer with its identifier.

        The work happens in a process pool behind this call rather than in
        it: a day is ninety-six independent simulations, which is minutes
        of work, and no browser should hold a request open for that. The
        page polls `get_observatory_day` and reads finished frames one at
        a time.

        Refused with a 422 if the request's cost estimate is over budget
        (see `rfi_simulator.webui.observatory.DAY_COST_BUDGET`), and with a
        429 if a day is already building (see `day_max_concurrent`) --
        cancel it or wait, then try again.
        """
        try:
            job_id = start_day(request)
        except DayBusyError as exc:
            return _busy_response(str(exc))
        except ValueError as exc:
            return JSONResponse(
                status_code=422,
                content={"detail": [{"loc": ["body"], "msg": str(exc), "type": "value_error"}]},
            )
        return {"id": job_id, "total": request.n_frames, "state": "building"}

    @app.get("/api/observatory/day/{job_id}")
    def get_observatory_day(
        job_id: str = PathParam(pattern=r"^[0-9a-f]{32}$"),
    ) -> dict[str, Any]:
        """How far the day has got, and every finished frame's metadata.

        Deliberately without the images: this is polled once a second
        while a day builds, and the images are what make a day large.
        """
        payload = day_status(job_id)
        if payload is None:
            raise HTTPException(
                status_code=404,
                detail="no such day: it may have finished over an hour ago and been dropped",
            )
        return payload

    @app.get("/api/observatory/day/{job_id}/frame/{index}")
    def get_observatory_frame(
        job_id: str = PathParam(pattern=r"^[0-9a-f]{32}$"),
        index: int = PathParam(ge=0, le=10_000),
    ) -> dict[str, Any]:
        """One frame's image and metadata."""
        payload = day_frame(job_id, index)
        if payload is None:
            raise HTTPException(status_code=404, detail=f"no frame {index} in this day")
        return payload

    @app.post("/api/observatory/day/{job_id}/cancel")
    def post_observatory_cancel(
        job_id: str = PathParam(pattern=r"^[0-9a-f]{32}$"),
    ) -> dict[str, Any]:
        """Stop building. Frames already finished stay readable."""
        payload = cancel_day(job_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="no such day")
        return payload

    @app.get("/api/observatory/timeline")
    def get_observatory_timeline(
        date: str = Query(max_length=32),
        dec_deg: float = Query(ge=-90.0, le=90.0),
        latitude_deg: float = Query(ge=-90.0, le=90.0),
        longitude_deg: float = Query(ge=-360.0, le=360.0),
        height_m: float = Query(default=0.0, ge=-500.0, le=1.0e4),
        tle_text: str = Query(default="", max_length=4000),
    ) -> Any:
        """Everything the day's timeline band draws, for one date and strip.

        Cheap enough to answer while the controls are still being edited:
        no simulation happens here, only ephemeris and, when the setup has
        a satellite, one propagation across the day.
        """
        try:
            return timeline_payload(
                date, dec_deg, latitude_deg, longitude_deg, height_m, tle_text=tle_text
            )
        except ValueError as exc:
            return JSONResponse(
                status_code=422,
                content={
                    "detail": [{"loc": ["query", "date"], "msg": str(exc), "type": "value_error"}]
                },
            )

    @app.get("/api/sky/now")
    def get_sky_now(
        latitude_deg: float | None = Query(default=None, ge=-90.0, le=90.0),
        longitude_deg: float | None = Query(default=None, ge=-360.0, le=360.0),
        height_m: float | None = Query(default=None, ge=-500.0, le=1.0e4),
    ) -> dict[str, Any]:
        """What is over the site at this instant.

        This is the one endpoint in the server that reaches the network,
        and it is the one endpoint that is allowed to fail in part: the
        ephemeris layers are always there, and a layer whose source is
        unreachable comes back empty with a status saying so rather than
        taking the response down with it.
        """
        site = default_array()
        return sky_now(
            site.latitude_deg if latitude_deg is None else latitude_deg,
            site.longitude_deg if longitude_deg is None else longitude_deg,
            site.height_m if height_m is None else height_m,
        )

    @app.get("/")
    def get_index() -> FileResponse:
        """The console page."""
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def main(argv: list[str] | None = None) -> int:
    """Serve the page.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status.
    """
    parser = argparse.ArgumentParser(
        prog="rfi-simulator-ui",
        description="Serve the interference simulator console in a browser.",
        epilog=(
            "The server listens on 127.0.0.1 so that nothing outside the "
            "machine can reach it, which matters on a shared host. To use it "
            "from a laptop while the simulator runs elsewhere, forward the "
            "port over SSH rather than changing --host:\n"
            "    ssh -N -L 8765:127.0.0.1:8765 user@host\n"
            "then open http://127.0.0.1:8765/ on the laptop."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="interface to bind (default: %(default)s, loopback only)",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="port to listen on (default: %(default)s)"
    )
    parser.add_argument(
        "--array-dir",
        default=None,
        help=(
            "directory of extra array layout YAML files to offer in the page's "
            "layout picker, alongside the bundled ones (default: the "
            f"{ARRAY_DIR_ENV_VAR} environment variable, if set)"
        ),
    )
    parser.add_argument(
        "--reload", action="store_true", help="restart when the source changes (development)"
    )
    args = parser.parse_args(argv)

    import uvicorn

    os.environ[HOST_ENV_VAR] = args.host
    if args.array_dir:
        # Same reason as the host: an import-string application is built
        # in a process this one does not otherwise get to configure.
        os.environ[ARRAY_DIR_ENV_VAR] = str(Path(args.array_dir).expanduser())
    uvicorn.run(
        "rfi_simulator.webui.server:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
