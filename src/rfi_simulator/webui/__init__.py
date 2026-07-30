"""Browser front end for the simulator.

The package is a *thin* layer: it validates a request, calls the library
exactly as a script would, and reduces the result to small JSON-friendly
arrays. No physics lives here -- if a number in the browser cannot be
traced back to a library call, that is a bug.

Two modules:

``simulate``
    Request models and the request -> library -> response glue.
    Importable and testable without starting a web server.
``server``
    The HTTP surface: ``create_app()`` for tests, ``main()`` for the
    ``rfi-simulator-ui`` console entry point. Import it explicitly
    (``from rfi_simulator.webui.server import create_app``) so that
    ``simulate`` stays usable without a web framework installed.
"""
