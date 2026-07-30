"""HTTP surface for the browser front end.

`create_app` builds the application without starting anything, which is
what the tests use; `main` is the ``rfi-simulator-ui`` console entry
point.

The server binds to the loopback interface by default and deliberately
never reaches the network itself: element sets are pasted in or taken
from the bundled sample, so a run needs nothing but this process.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from rfi_simulator import __version__
from rfi_simulator.webui.simulate import SimulateRequest, defaults_payload, run_simulation

__all__ = ["create_app", "main"]

STATIC_DIR = Path(__file__).resolve().parent / "static"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def create_app() -> FastAPI:
    """Build the application.

    Returns
    -------
    fastapi.FastAPI
        With ``GET /api/defaults``, ``POST /api/simulate``, and the page
        itself at ``/``.
    """
    app = FastAPI(
        title="Interference simulator",
        version=__version__,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

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
    def post_simulate(request: SimulateRequest) -> Any:
        """Run one observation and return everything the page draws.

        A `ValueError` from the library -- a transmitter tuned outside the
        band is the common one -- comes back as a 422 with the library's
        own message, because it is a fault in the setup the user typed,
        not in the server.
        """
        try:
            return run_simulation(request)
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
