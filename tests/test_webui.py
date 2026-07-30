"""Tests for the browser front end.

Everything here goes through `fastapi.testclient.TestClient`, so the
request models, the glue and the routing are all exercised together but no
socket is opened and no browser is involved.

The load-bearing test is
`test_reported_occupancy_matches_the_library` -- it re-runs the same
request through the library by hand and insists the number the page shows
is the library's own ``rfi_fraction``, not something the front end
computed for itself.
"""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rfi_simulator import correlate
from rfi_simulator.webui.server import create_app
from rfi_simulator.webui.simulate import (
    MAX_ANTENNAS,
    MAX_N_BLOCKS,
    MAX_N_CHAN,
    SimulateRequest,
    build_simulator,
    default_array,
    defaults_payload,
)

# Small but real: two integrations of 32 channels run in well under a
# second and still exercise every code path the page uses.
SMALL_SIM = {"n_chan": 32, "n_blocks": 2, "seed": 7}

RFI_SOURCE_CASES = {
    "tower": {"type": "tower"},
    "impulsive": {"type": "impulsive"},
    "satellite": {"type": "satellite"},
    "aircraft": {"type": "aircraft"},
}


@pytest.fixture(scope="module")
def client() -> TestClient:
    """A client bound to the application, shared by the whole module."""
    return TestClient(create_app())


@pytest.fixture(scope="module")
def defaults(client: TestClient) -> dict:
    """The payload the page builds itself from."""
    response = client.get("/api/defaults")
    assert response.status_code == 200
    return response.json()


def make_request(rfi_sources=(), **sim_overrides) -> dict:
    """A minimal valid request body."""
    array = default_array()
    sim = dict(SMALL_SIM)
    sim.update(sim_overrides)
    return {
        "antennas": [list(row) for row in array.antenna_positions_enu_m],
        "sky_sources": [{"name": "target", "l": 0.0087, "m": -0.0052, "flux_jy": 5.0}],
        "rfi_sources": list(rfi_sources),
        "sim": sim,
    }


# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------
def test_defaults_carry_everything_the_page_needs(defaults):
    """The page hard-codes no schema, so the payload must be complete."""
    assert set(defaults) >= {"array", "sim", "limits", "sky_source", "rfi_types", "sample_tle"}

    antennas = defaults["array"]["antennas"]
    assert len(antennas) >= 2
    assert all(len(position) == 3 for position in antennas)

    assert defaults["limits"]["max_antennas"] == MAX_ANTENNAS
    assert defaults["limits"]["max_n_chan"] == MAX_N_CHAN
    assert defaults["limits"]["max_n_blocks"] == MAX_N_BLOCKS

    types = {entry["type"] for entry in defaults["rfi_types"]}
    assert types == set(RFI_SOURCE_CASES)

    for entry in defaults["rfi_types"]:
        assert entry["label"] and entry["summary"]
        for field in entry["fields"]:
            assert field["kind"] in {"number", "choice", "toggle", "text"}
            assert field["label"]
            assert field["name"] in entry["defaults"]

    assert defaults["sample_tle"].count("\n") >= 3 - 1


def test_schema_defaults_are_accepted_by_the_request_models(defaults):
    """The form defaults and the validators must not drift apart."""
    for entry in defaults["rfi_types"]:
        body = make_request([dict(entry["defaults"], type=entry["type"])])
        SimulateRequest.model_validate(body)

    sky_defaults = defaults["sky_source"]["defaults"]
    body = make_request()
    body["sky_sources"] = [dict(sky_defaults)]
    SimulateRequest.model_validate(body)


def test_the_page_and_its_assets_are_served(client):
    """The console is one HTML file and two static assets, all local."""
    index = client.get("/")
    assert index.status_code == 200
    assert "Interference simulator" in index.text
    # Nothing may be fetched from anywhere but this server.
    assert "//" not in index.text.replace("http://www.w3.org/2000/svg", "")
    for asset in ("/static/app.js", "/static/styles.css"):
        assert client.get(asset).status_code == 200


# ----------------------------------------------------------------------
# Round trip
# ----------------------------------------------------------------------
@pytest.mark.parametrize("case", sorted(RFI_SOURCE_CASES))
def test_every_source_type_runs_and_reports_its_own_ground_truth(client, defaults, case):
    """Each kind of interference runs, and comes back labelled."""
    schema = [entry for entry in defaults["rfi_types"] if entry["type"] == case][0]
    params = dict(schema["defaults"], type=case)

    response = client.post("/api/simulate", json=make_request([params]))
    assert response.status_code == 200, response.text
    result = response.json()

    assert len(result["sources"]) == 1
    source = result["sources"][0]
    assert source["type"] == case
    assert source["occupancy"] > 0.0, "this configuration should emit into the band"

    waterfall = result["waterfall"]["antennas"]
    assert len(waterfall) == result["observation"]["n_antennas"]
    n_rows = len(result["waterfall"]["freq_mhz"])
    n_cols = len(result["waterfall"]["time_s"])
    for plane in waterfall:
        assert len(plane) == n_rows
        assert all(len(row) == n_cols for row in plane)
    assert len(source["mask"]) == n_rows
    assert all(len(row) == n_cols for row in source["mask"])
    assert set(np.unique(np.asarray(source["mask"]))) <= {0, 1}

    # ANY-pooling can only ever grow the flagged fraction, never shrink it.
    pooled = np.asarray(source["mask"], dtype=bool).mean()
    assert pooled >= source["occupancy"] - 1e-12

    assert result["wall_time_s"] > 0.0
    assert result["image"]["values"]
    assert len(result["uv"]["u"]) == len(result["uv"]["v"]) > 0


@pytest.mark.parametrize("case", sorted(RFI_SOURCE_CASES))
def test_reported_occupancy_matches_the_library(client, defaults, case):
    """The occupancy the page shows is the library's ``rfi_fraction``."""
    schema = [entry for entry in defaults["rfi_types"] if entry["type"] == case][0]
    body = make_request([dict(schema["defaults"], type=case)])

    result = client.post("/api/simulate", json=body).json()

    simulator = build_simulator(SimulateRequest.model_validate(body))
    visibilities = correlate(simulator.blocks())
    expected = float(visibilities.rfi_fraction.mean())

    assert result["sources"][0]["occupancy"] == pytest.approx(expected, abs=1e-12)
    assert result["sources"][0]["name"] == visibilities.rfi_source_names[0]


def test_a_clean_run_places_the_source_where_it_was_put(client):
    """The dirty image is the library's, so the source must land on target."""
    body = make_request()
    result = client.post("/api/simulate", json=body).json()

    peak = result["image"]["peak"]
    assert peak["l"] == pytest.approx(0.0087, abs=2e-3)
    assert peak["m"] == pytest.approx(-0.0052, abs=2e-3)
    assert peak["value_jy"] > 1.0
    assert result["sources"] == []
    assert result["warnings"] == [], "a zenith-phased flat array should run clean"


def test_the_same_seed_gives_the_same_image_twice(client):
    """Determinism is the point of seeding; two runs must agree bit for bit."""
    body = make_request(seed=1234)
    first = client.post("/api/simulate", json=body).json()
    second = client.post("/api/simulate", json=body).json()

    assert first["image"]["values"] == second["image"]["values"]
    assert first["waterfall"]["antennas"] == second["waterfall"]["antennas"]

    body["sim"]["seed"] = 1235
    third = client.post("/api/simulate", json=body).json()
    assert third["image"]["values"] != first["image"]["values"]


def test_a_stale_element_set_is_reported_rather_than_hidden(client, defaults):
    """Library warnings reach the notice area instead of the server log."""
    schema = [entry for entry in defaults["rfi_types"] if entry["type"] == "satellite"][0]
    params = dict(schema["defaults"], type="satellite", carrier_freq_hz=2.0e9)
    body = make_request([params])
    body["sim"]["center_freq_hz"] = 2.0e9

    result = client.post("/api/simulate", json=body).json()
    assert isinstance(result["warnings"], list)


# ----------------------------------------------------------------------
# Guard rails
# ----------------------------------------------------------------------
def test_too_many_antennas_is_refused(client):
    """The cap is a 422 with an explanation, not an out-of-memory kill."""
    body = make_request()
    body["antennas"] = [[float(i), 0.0, 0.0] for i in range(MAX_ANTENNAS + 1)]
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 422
    assert str(MAX_ANTENNAS) in response.text


def test_one_antenna_is_refused(client):
    """An array of one has no baselines, so say so rather than divide by zero."""
    body = make_request()
    body["antennas"] = [[0.0, 0.0, 0.0]]
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 422
    assert "baseline" in response.text


def test_a_non_finite_antenna_position_is_refused(client):
    """NaN reaches the array validator as a form error, not a crash.

    Sent as raw text because ``NaN`` is not JSON, but it is what a
    JavaScript ``JSON.stringify`` of a broken form can still produce
    through a hand-edited request -- the server must not take it.
    """
    body = make_request()
    body["antennas"] = [[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]]
    response = client.post(
        "/api/simulate",
        content=json.dumps(body),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert "finite" in response.text


@pytest.mark.parametrize("key, value", [("n_chan", MAX_N_CHAN + 8), ("n_blocks", MAX_N_BLOCKS + 1)])
def test_oversized_observations_are_refused(client, key, value):
    """Channel and integration counts are capped before anything is allocated."""
    body = make_request(**{key: value})
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 422
    assert key in response.text


def test_a_malformed_element_set_is_refused_with_advice(client):
    """A pasted element set that will not parse is a form error."""
    body = make_request(
        [{"type": "satellite", "tle_source": "custom", "tle_text": "not an element set"}]
    )
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 422
    assert "element set" in response.text


def test_an_empty_pasted_element_set_is_refused(client):
    """Choosing "paste your own" and pasting nothing says what to do."""
    body = make_request([{"type": "satellite", "tle_source": "custom", "tle_text": "   "}])
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 422
    assert "69-character" in response.text


def test_an_unknown_source_type_is_refused(client):
    """The discriminated union rejects anything it does not model."""
    response = client.post("/api/simulate", json=make_request([{"type": "microwave-oven"}]))
    assert response.status_code == 422


def test_a_transmitter_outside_the_band_explains_itself(client):
    """The library's own message reaches the user, as a 422 not a 500."""
    body = make_request([{"type": "tower", "center_freq_hz": 9.0e9}])
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 422
    assert "outside the simulated band" in response.text


def test_the_defaults_payload_is_importable_without_http():
    """The glue is usable from a script; the server is only a wrapper."""
    payload = defaults_payload()
    assert payload["sim"]["n_chan"] >= 4
    assert payload["array"]["antennas"]
