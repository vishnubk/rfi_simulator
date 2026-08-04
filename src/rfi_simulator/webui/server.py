"""HTTP surface for the browser front end.

`create_app` builds the application without starting anything, which is
what the tests use; `main` is the ``rfi-simulator-ui`` console entry
point.

The server binds to the loopback interface by default and deliberately
never reaches the network itself -- neither on the server side nor in the
page it serves: element sets are pasted in or taken from the bundled
sample, every asset is served from this process, and the interactive API
documentation, which would pull its viewer from a content delivery
network, is switched off. The machine-readable schema stays at
``/api/openapi.json``.
"""

from __future__ import annotations

import argparse
import os
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from rfi_simulator import __version__
from rfi_simulator.webui.simulate import SimulateRequest, defaults_payload, run_simulation

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
recording at once could attribute one request's warnings to the other."""

_simulation_slots = threading.BoundedSemaphore(MAX_CONCURRENT_SIMULATIONS)


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


def create_app(host: str | None = None) -> FastAPI:
    """Build the application.

    Parameters
    ----------
    host : str, optional
        The interface the server is bound to, added to the accepted
        ``Host`` headers. Loopback names are always accepted. Defaults to
        `HOST_ENV_VAR` in the environment, which is how `main` passes the
        bound interface through the reloader's fresh process.

    Returns
    -------
    fastapi.FastAPI
        With ``GET /api/defaults``, ``POST /api/simulate``, and the page
        itself at ``/``.
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

        Runs queue behind `MAX_CONCURRENT_SIMULATIONS`: the request model
        bounds the memory of one run, and this is what keeps several
        arriving at once from multiplying that bound.

        `pol` selects which receptor the waterfall display shows for a
        dual-polarization run (``n_pol=2``); it is ignored (fixed to 0)
        for a single-polarization one. The dirty image always images
        Stokes I regardless of `pol` -- see `run_simulation`.
        """
        with _simulation_slots:
            try:
                return run_simulation(request, pol=pol)
            except ValueError as exc:
                return JSONResponse(
                    status_code=422,
                    content={"detail": [{"loc": ["body"], "msg": str(exc), "type": "value_error"}]},
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
        "--reload", action="store_true", help="restart when the source changes (development)"
    )
    args = parser.parse_args(argv)

    import uvicorn

    os.environ[HOST_ENV_VAR] = args.host
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
