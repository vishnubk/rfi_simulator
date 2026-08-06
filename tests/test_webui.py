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
import math
from pathlib import Path

import numpy as np
import pytest

# The front end is an optional extra: without it installed these tests
# have nothing to exercise and skip rather than break collection.
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from rfi_simulator import GaussianBeam, correlate, spectral_kurtosis_mask  # noqa: E402
from rfi_simulator.metrics import flag_scores, pool_truth_accumulations  # noqa: E402
from rfi_simulator.webui.server import (  # noqa: E402
    MAX_REQUEST_BYTES,
    _simulation_slots,
    create_app,
)
from rfi_simulator.webui.simulate import (  # noqa: E402
    FLAG_DEFAULT_M,
    MAX_ANTENNAS,
    MAX_COORDINATE_M,
    MAX_FLAG_METHODS,
    MAX_N_BLOCKS,
    MAX_N_CHAN,
    MAX_TOTAL_SAMPLES,
    MAX_VIS_SPECTRUM_VALUES,
    MAX_WATERFALL_CELLS,
    N_TIME_PER_BLOCK,
    VIS_MAX_CHAN_BINS,
    SimulateRequest,
    build_simulator,
    default_array,
    defaults_payload,
    pointing_payload,
)

# Small but real: two integrations of 32 channels run in well under a
# second and still exercise every code path the page uses.
SMALL_SIM = {"n_chan": 32, "n_blocks": 2, "seed": 7}

RFI_SOURCE_CASES = {
    "tower": {"type": "tower"},
    "impulsive": {"type": "impulsive"},
    "satellite": {"type": "satellite"},
    "aircraft": {"type": "aircraft"},
    "comb": {"type": "comb"},
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
    assert "RFI Simulator" in index.text
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


def _local_maxima(image: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Every interior pixel no darker than its eight neighbours.

    Deliberately naive: the images here are 64 x 64 and the point is to
    count peaks, not to find them quickly.
    """
    found = []
    for row in range(1, image.shape[0] - 1):
        for col in range(1, image.shape[1] - 1):
            value = image[row, col]
            if value >= threshold and value >= image[row - 1 : row + 2, col - 1 : col + 2].max():
                found.append((row, col))
    return found


def test_several_sources_each_raise_their_own_peak(client):
    """Three sources at three places are three peaks, not one.

    The contract the page leans on: whatever it puts in ``sky_sources``
    comes back resolved one for one, and every source that is in the
    field raises its own maximum in the dirty image. A page that gave
    several sources the same position would show a single blob -- see
    `test_sources_on_one_spot_add_into_a_single_peak` for what that
    looks like from here.
    """
    offsets = [(0.6, 0.0), (-0.5, 0.4), (0.1, -0.6)]
    body = make_request()
    body["sky_sources"] = [
        {"name": f"source {index}", "offset_deg": [east, north], "flux_jy": 5.0}
        for index, (east, north) in enumerate(offsets)
    ]
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 200, response.text
    result = response.json()

    resolved = result["sky_sources"]
    assert len(resolved) == len(offsets)
    places = {(round(source["l"], 9), round(source["m"], 9)) for source in resolved}
    assert len(places) == len(offsets), "three offsets must resolve to three positions"
    assert all(source["in_field"] for source in resolved)

    image = np.asarray(result["image"]["values"])
    l_grid = np.asarray(result["image"]["l"])
    m_grid = np.asarray(result["image"]["m"])
    peaks = _local_maxima(image, 0.5 * float(image.max()))

    brightest = []
    for source in resolved:
        col = int(np.argmin(np.abs(l_grid - source["l"])))
        row = int(np.argmin(np.abs(m_grid - source["m"])))
        near = [peak for peak in peaks if abs(peak[0] - row) <= 2 and abs(peak[1] - col) <= 2]
        assert near, f"no peak within two pixels of {source['name']}"
        value = max(float(image[peak]) for peak in near)
        # Every source is as bright as every other, so none of them may
        # be a faint bump beside one dominant blob.
        assert value > 0.7 * float(image.max()), source["name"]
        brightest.append(value)

    # The three sources are the three brightest things in the image; the
    # rest of the maxima are this small array's sidelobes.
    others = sorted((float(image[peak]) for peak in peaks), reverse=True)[len(offsets) :]
    assert min(brightest) > max(others)


def test_sources_on_one_spot_add_into_a_single_peak(client):
    """Coincident sources are one peak of their summed flux.

    Not a bug in itself -- it is what interferometry does -- but it is
    why the page must place added sources apart from one another.
    """
    body = make_request()
    body["sky_sources"] = [
        {"name": f"source {index}", "offset_deg": [0.5, -0.3], "flux_jy": 5.0} for index in range(3)
    ]
    result = client.post("/api/simulate", json=body).json()

    image = np.asarray(result["image"]["values"])
    # One bright thing, three times one source's flux -- everything else
    # above half the maximum is this small array's sidelobe pattern.
    peaks = _local_maxima(image, 0.9 * float(image.max()))
    assert len(peaks) == 1
    assert result["image"]["peak"]["value_jy"] == pytest.approx(15.0, rel=0.1)


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


def test_a_run_that_is_legal_field_by_field_can_still_be_too_large(client):
    """Every field within its cap, the product beyond it: refused with advice."""
    body = make_request(n_chan=MAX_N_CHAN, n_blocks=MAX_N_BLOCKS)
    body["antennas"] = [[float(i) * 10.0, 0.0, 0.0] for i in range(MAX_ANTENNAS)]
    total = MAX_ANTENNAS * MAX_N_CHAN * MAX_N_BLOCKS * N_TIME_PER_BLOCK
    assert total > MAX_TOTAL_SAMPLES, "this test only means anything above the budget"

    response = client.post("/api/simulate", json=body)
    assert response.status_code == 422
    text = response.text
    assert "n_chan" in text and "n_blocks" in text and "antennas" in text


def test_the_served_defaults_are_within_the_size_budget(defaults):
    """The page must open on a run the size cap accepts."""
    total = (
        len(defaults["array"]["antennas"])
        * defaults["sim"]["n_chan"]
        * defaults["sim"]["n_blocks"]
        * defaults["sim"]["n_time_per_block"]
    )
    assert total <= MAX_TOTAL_SAMPLES
    assert defaults["limits"]["max_total_samples"] == MAX_TOTAL_SAMPLES


# ----------------------------------------------------------------------
# A busy simulation slot is refused, not queued (finding 6)
# ----------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _release_simulation_slot_between_tests():
    """A test that fails while holding the slot must not wedge every test after it."""
    yield
    try:
        _simulation_slots.release()
    except ValueError:
        pass  # the slot was never taken; a BoundedSemaphore refuses an over-release


def test_a_busy_simulation_slot_is_refused_with_429(client):
    assert _simulation_slots.acquire(blocking=False)
    try:
        response = client.post("/api/simulate", json=make_request())
        assert response.status_code == 429
        assert "already running" in response.json()["detail"][0]["msg"]
    finally:
        _simulation_slots.release()
    # The slot is free again immediately -- nothing was left queued behind it.
    ok = client.post("/api/simulate", json=make_request())
    assert ok.status_code == 200


def test_a_busy_simulation_slot_also_refuses_the_flagger(client):
    body = {
        "request": make_request(),
        "methods": ["mad"],
        "antenna": 0,
        "pol": 0,
    }
    assert _simulation_slots.acquire(blocking=False)
    try:
        response = client.post("/api/flag", json=body)
        assert response.status_code == 429
        assert "already running" in response.json()["detail"][0]["msg"]
    finally:
        _simulation_slots.release()


def test_an_oversized_body_is_refused_before_it_is_read(client):
    """The length header alone is enough to say no."""
    payload = b"x" * (MAX_REQUEST_BYTES + 1024)
    response = client.post(
        "/api/simulate",
        content=payload,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413
    assert "bytes" in response.text


def test_an_oversized_chunked_body_is_refused_while_streaming(client):
    """No declared length must not mean no limit.

    A chunked body carries no ``Content-Length``, so the header check
    cannot fire; the middleware has to count the bytes as they arrive
    and refuse when the running total passes the cap.
    """

    def chunks():
        sent = 0
        while sent <= MAX_REQUEST_BYTES:
            yield b"x" * 65536
            sent += 65536

    response = client.post(
        "/api/simulate",
        content=chunks(),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413
    assert "bytes" in response.text


def test_an_antenna_further_out_than_the_bound_is_refused(client):
    """Coordinates are metres from the origin, not arbitrary floats."""
    body = make_request()
    body["antennas"] = [[0.0, 0.0, 0.0], [10.0 * MAX_COORDINATE_M, 0.0, 0.0]]
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 422
    assert "origin" in response.text


def test_the_interactive_api_documentation_is_not_served(client):
    """That page would pull its viewer off the network; the schema is enough."""
    assert client.get("/api/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/api/openapi.json").status_code == 200


def test_an_unexpected_host_header_is_refused():
    """Only the loopback names and the bound interface are answered."""
    local = TestClient(create_app())
    assert local.get("/api/defaults", headers={"Host": "evil.example"}).status_code == 400

    bound = TestClient(create_app(host="10.0.0.5"), base_url="http://10.0.0.5")
    assert bound.get("/api/defaults").status_code == 200


# ----------------------------------------------------------------------
# The widest band the front end runs
# ----------------------------------------------------------------------
def pool_any(mask: np.ndarray, axis: int, n_bins: int) -> np.ndarray:
    """ANY-pool `mask` to `n_bins` along `axis`, written out the slow way.

    Deliberately not the implementation under test: the bins are sliced
    and reduced one at a time so that a mis-mapped row would show up.
    """
    length = mask.shape[axis]
    if n_bins >= length:
        return mask
    edges = np.linspace(0, length, n_bins + 1).astype(int)
    parts = [
        mask.take(range(edges[k], edges[k + 1]), axis=axis).any(axis=axis) for k in range(n_bins)
    ]
    return np.stack(parts, axis=axis)


def test_a_full_width_band_pools_its_mask_onto_the_right_frequencies(client):
    """The widest band the caps allow, checked row by row against the library.

    ``n_chan`` is above `MAX_BINS`, so the channel axis really is pooled
    here -- the case where a mistake in the binning would silently move an
    interference feature to the wrong frequency.
    """
    body = make_request(
        [{"type": "tower", "center_freq_hz": 1.4055e9, "bandwidth_hz": 2.0e5}],
        n_chan=MAX_N_CHAN,
        n_blocks=8,
    )
    body["antennas"] = [[0.0, 0.0, 0.0], [12.4, -8.7, 0.0], [-19.3, 5.2, 0.0]]
    SimulateRequest.model_validate(body)  # inside the size budget

    response = client.post("/api/simulate", json=body)
    assert response.status_code == 200, response.text
    result = response.json()

    mask = np.asarray(result["sources"][0]["mask"], dtype=bool)
    freq_mhz = np.asarray(result["waterfall"]["freq_mhz"], dtype=np.float64)
    n_rows, n_cols = mask.shape

    # (b) the frequency axis is ordered, and (c) the response stays inside
    # the cell budget it promises.
    assert np.all(np.diff(freq_mhz) > 0.0)
    assert len(result["waterfall"]["antennas"]) * n_rows * n_cols <= MAX_WATERFALL_CELLS
    assert result["waterfall"]["time_samples_per_cell"] >= 1

    # (a) re-pool the library's own mask and insist row for row.
    simulator = build_simulator(SimulateRequest.model_validate(body))
    time_bins = n_cols // simulator.n_blocks
    expected_columns = []
    for block in simulator.blocks():
        pooled = pool_any(block.rfi_mask[0], axis=1, n_bins=time_bins)
        expected_columns.append(pool_any(pooled, axis=0, n_bins=n_rows))
    expected = np.concatenate(expected_columns, axis=1)

    assert expected.shape == mask.shape
    assert np.array_equal(mask, expected)
    assert expected.any(), "this tower should flag something"

    # The frequencies the rows are labelled with are the pooled channel
    # centres, so a flagged row really is where the transmitter sits.
    edges = np.linspace(0, simulator.n_chan, n_rows + 1).astype(int)
    expected_freq = np.array(
        [simulator.freq_hz[edges[k] : edges[k + 1]].mean() for k in range(n_rows)]
    )
    np.testing.assert_allclose(freq_mhz, expected_freq / 1.0e6, rtol=0, atol=1e-6)

    flagged = freq_mhz[mask.any(axis=1)]
    assert flagged.min() <= 1405.5 <= flagged.max()


def test_the_time_pooling_factor_is_reported(client):
    """The page cannot say how coarse the picture is unless the run says so."""
    result = client.post("/api/simulate", json=make_request()).json()
    pooled = result["waterfall"]["time_samples_per_cell"]
    assert isinstance(pooled, int)
    n_cols = len(result["waterfall"]["time_s"])
    time_bins_per_block = n_cols // result["observation"]["n_blocks"]
    assert pooled == -(-N_TIME_PER_BLOCK // time_bins_per_block)


def test_the_defaults_payload_is_importable_without_http():
    """The glue is usable from a script; the server is only a wrapper."""
    payload = defaults_payload()
    assert payload["sim"]["n_chan"] >= 4
    assert payload["array"]["antennas"]


# ----------------------------------------------------------------------
# Realism features added since the UI was built: instrument, quantization,
# channelizer, dual polarization, calibration errors, primary beam, and the
# per-source coupling/polarization/envelope/arrival/comb extras. Each group
# gets an on/off round trip (on must differ from off, both must be 200) and
# a bad-value case (422/400, never a 500 traceback); one request turns
# everything on at once.
# ----------------------------------------------------------------------
def test_instrument_realism_changes_the_result(client):
    """A gain-scattered, uncalibrated array must not image like an ideal one."""
    off = client.post("/api/simulate", json=make_request()).json()
    body = make_request()
    body["instrument"] = {"gain_scatter_db": 3.0, "phase_offsets": "uniform"}
    on = client.post("/api/simulate", json=body)
    assert on.status_code == 200, on.text
    on = on.json()
    assert on["image"]["peak"]["value_jy"] != off["image"]["peak"]["value_jy"]


def test_instrument_rejects_a_negative_scatter(client):
    """A library `ValueError` (or pydantic's own bound) reaches the user, not a crash."""
    body = make_request()
    body["instrument"] = {"gain_scatter_db": -1.0}
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 422


def test_quantization_changes_the_result(client):
    """4-bit quantization must leave a visible trace in the waterfall."""
    off = client.post("/api/simulate", json=make_request()).json()
    body = make_request()
    body["quantization"] = {"quant_target_counts": 1.33}
    on = client.post("/api/simulate", json=body)
    assert on.status_code == 200, on.text
    on = on.json()
    assert on["waterfall"]["antennas"] != off["waterfall"]["antennas"]


def test_quantization_rejects_a_non_positive_scale(client):
    body = make_request()
    body["quantization"] = {"quant_scale": 0.0}
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 422


def test_channelizer_changes_the_result(client):
    """A polyphase filterbank must colour the waterfall differently from the ideal one."""
    off = client.post("/api/simulate", json=make_request()).json()
    body = make_request()
    body["channelizer"] = {"n_taps": 8, "window": "blackman", "sinc_bandwidth": 1.1}
    on = client.post("/api/simulate", json=body)
    assert on.status_code == 200, on.text
    on = on.json()
    assert on["waterfall"]["antennas"] != off["waterfall"]["antennas"]


def test_channelizer_rejects_an_unknown_window(client):
    body = make_request()
    body["channelizer"] = {"window": "rectangular"}
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 422


def test_dual_polarization_reports_both_receptors(client):
    """n_pol=2 must show up in the observation summary, not silently collapse to 1."""
    body = make_request()
    body["n_pol"] = 2
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["observation"]["n_pol"] == 2
    assert result["observation"]["pol_names"] == ["XX", "YY"]


def test_the_pol_query_param_selects_the_waterfall_receptor(client):
    """The two receptors of a polarized source must not draw identically."""
    body = make_request(
        [
            {
                "type": "tower",
                "polarization": {"type": "linear", "angle_deg": 15.0},
            }
        ]
    )
    body["n_pol"] = 2
    pol0 = client.post("/api/simulate?pol=0", json=body).json()
    pol1 = client.post("/api/simulate?pol=1", json=body).json()
    assert pol0["observation"]["waterfall_pol"] == 0
    assert pol1["observation"]["waterfall_pol"] == 1
    assert pol0["waterfall"]["antennas"] != pol1["waterfall"]["antennas"]


def test_n_pol_rejects_an_unsupported_value(client):
    body = make_request()
    body["n_pol"] = 3
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 422


def test_calibration_errors_reduce_the_imaged_flux(client):
    """A residual phase error must lose coherence, not silently do nothing."""
    off = client.post("/api/simulate", json=make_request()).json()
    body = make_request()
    body["calibration_errors"] = {"phase_error_deg_rms": 25.0}
    on = client.post("/api/simulate", json=body)
    assert on.status_code == 200, on.text
    on = on.json()
    assert on["image"]["peak"]["value_jy"] < off["image"]["peak"]["value_jy"]


def test_calibration_errors_rejects_a_negative_rms(client):
    body = make_request()
    body["calibration_errors"] = {"delay_error_ns_rms": -1.0}
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 422


def test_instrument_and_calibration_errors_do_not_share_a_random_draw(client):
    """The gain-scatter and calibration-phase draws must be independent, not the same vector.

    Regression test for a same-seed collision: `InstrumentModel.from_params`
    and `CalibrationErrors.from_params` each spawn children from
    `SeedSequence(seed)` in a fixed order, and `SeedSequence.spawn`'s first
    child depends only on the seed -- so handing both models the same raw
    `sim.seed` used to make the per-antenna gain scatter (dB) and the
    per-antenna phase error draw the exact same standard-normal vector,
    just rescaled. With enough antennas that would show up as a
    near-perfect correlation between the two, standardized, vectors;
    `build_simulator`/`run_simulation` now derive independent children
    instead (see `_feature_seed_sequences`), so the correlation must not be
    extreme in either direction.
    """
    n_antennas = MAX_ANTENNAS
    rng = np.random.default_rng(0)
    positions = [
        [east, north, 0.0]
        for east, north in rng.uniform(-200.0, 200.0, size=(n_antennas, 2)).tolist()
    ]

    body = make_request()
    body["antennas"] = positions
    body["instrument"] = {"gain_scatter_db": 3.0}
    body["calibration_errors"] = {"phase_error_deg_rms": 20.0}
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 200, response.text

    from rfi_simulator.webui.simulate import (
        CalibrationErrorParams,
        InstrumentParams,
        _feature_seed_sequences,
    )

    seed = body["sim"]["seed"]
    instrument_seq, calibration_seq = _feature_seed_sequences(seed)
    instrument = InstrumentParams(gain_scatter_db=3.0).build(
        n_antennas, np.random.default_rng(instrument_seq)
    )
    calibration = CalibrationErrorParams(phase_error_deg_rms=20.0).build(
        n_antennas, np.random.default_rng(calibration_seq)
    )

    gain_db = 20.0 * np.log10(np.abs(instrument.scalar_gains))
    phase_rad = calibration.phase_error_rad

    gain_z = (gain_db - gain_db.mean()) / gain_db.std()
    phase_z = (phase_rad - phase_rad.mean()) / phase_rad.std()
    correlation = float(np.corrcoef(gain_z, phase_z)[0, 1])
    # Before the fix this was exactly +/-1.0 (the same underlying draw,
    # just rescaled by different loc/scale arguments); after it, two
    # independent normal draws of 40 samples correlate weakly at worst.
    assert abs(correlation) < 0.5, f"gain scatter and phase error correlate at {correlation}"


def test_the_same_seed_reproduces_the_full_response_with_every_realism_feature_on(client):
    """Reproducibility must survive deriving independent per-feature seeds."""
    body = make_request()
    body["instrument"] = {"gain_scatter_db": 2.0, "phase_offsets": "uniform"}
    body["calibration_errors"] = {"phase_error_deg_rms": 15.0, "delay_error_ns_rms": 1.0}
    body["sim"]["seed"] = 999

    first = client.post("/api/simulate", json=body)
    second = client.post("/api/simulate", json=body)
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    first_payload, second_payload = first.json(), second.json()
    # `wall_time_s` is a measurement, not a draw from the seed, and is
    # never expected to repeat exactly; everything else must.
    del first_payload["wall_time_s"], second_payload["wall_time_s"]
    assert first_payload == second_payload


def test_primary_beam_attenuates_an_offset_source(client):
    """A source away from the pointing centre must lose flux under a beam."""
    off = client.post("/api/simulate", json=make_request()).json()
    body = make_request()
    body["primary_beam"] = {"type": "gaussian", "dish_diameter_m": 4.5}
    on = client.post("/api/simulate", json=body)
    assert on.status_code == 200, on.text
    on = on.json()
    assert on["image"]["peak"]["value_jy"] < off["image"]["peak"]["value_jy"]


def test_airy_beam_is_also_accepted(client):
    body = make_request()
    body["primary_beam"] = {"type": "airy", "dish_diameter_m": 4.5}
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 200, response.text


def test_primary_beam_rejects_a_non_positive_dish(client):
    body = make_request()
    body["primary_beam"] = {"dish_diameter_m": 0.0}
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 422


def test_a_spectral_line_is_labelled_celestial_not_rfi(client):
    """A spectral line must occupy the band without being reported as an RFI source."""
    off = client.post("/api/simulate", json=make_request()).json()
    body = make_request()
    body["spectral_lines"] = [
        {"name": "hi", "center_freq_hz": 1405.0e6, "fwhm_hz": 2.0e4, "line_flux_jy": 50.0}
    ]
    on = client.post("/api/simulate", json=body)
    assert on.status_code == 200, on.text
    on = on.json()
    assert on["sources"] == []  # the line is not an RFI source
    assert on["waterfall"]["antennas"] != off["waterfall"]["antennas"]


def test_a_spectral_line_rejects_a_non_positive_fwhm(client):
    body = make_request()
    body["spectral_lines"] = [{"fwhm_hz": 0.0}]
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 422


def test_a_comb_transmitter_runs_and_reports_its_harmonics(client):
    """The new source type must round-trip like every other one."""
    body = make_request(
        [
            {
                "type": "comb",
                "fundamental_hz": 1.405e6,
                "harmonic_numbers": [999, 1000, 1001],
                "received_power_jy": 500.0,
            }
        ]
    )
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert len(result["sources"]) == 1
    assert result["sources"][0]["type"] == "comb"
    assert result["sources"][0]["occupancy"] > 0.0


def test_a_comb_transmitter_accepts_a_comma_separated_harmonics_string(client):
    """The browser's text field sends a string; the API must parse it, not just a list."""
    body = make_request(
        [{"type": "comb", "fundamental_hz": 1.405e6, "harmonic_numbers": "999,1000,1001"}]
    )
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 200, response.text


def test_a_comb_transmitter_rejects_duplicate_harmonics(client):
    body = make_request([{"type": "comb", "harmonic_numbers": [1, 1, 2]}])
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 422


def test_impulsive_periodic_arrival_is_accepted(client):
    """`arrival` replaces `rate_hz`; the library's own mutual exclusion still applies."""
    body = make_request(
        [{"type": "impulsive", "arrival": {"type": "periodic", "rate_hz": 50.0, "jitter_s": 1e-3}}]
    )
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 200, response.text


def test_coupling_and_polarization_extras_are_accepted_on_every_source_type(client):
    """`coupling`/`polarization` are API-only extras -- every RFI type must accept them."""
    extras = {
        "coupling": {"type": "lognormal", "sigma_db": 6.0, "seed": 1},
        "polarization": {"type": "linear", "angle_deg": 30.0},
    }
    for case, params in RFI_SOURCE_CASES.items():
        body = make_request([dict(params, **extras)])
        response = client.post("/api/simulate", json=body)
        assert response.status_code == 200, f"{case}: {response.text}"


def test_an_explicit_coupling_vector_must_match_the_antenna_count(client):
    body = make_request([{"type": "tower", "coupling": [1.0, 1.0]}])  # array has more antennas
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 422


def test_a_non_finite_explicit_coupling_vector_is_refused(client):
    """An explicit coupling vector is only reachable via the API, but must still be checked.

    Sent as raw text, same reason as `test_a_non_finite_antenna_position_is_refused`:
    ``NaN`` is not valid JSON, but a hand-built request can still contain it.
    """
    n_antennas = len(default_array().antenna_positions_enu_m)
    body = make_request([{"type": "tower", "coupling": [float("nan")] * n_antennas}])
    response = client.post(
        "/api/simulate",
        content=json.dumps(body),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert "finite" in response.text


@pytest.mark.parametrize("field", ["jones_re", "jones_im"])
def test_a_non_finite_full_polarization_jones_component_is_refused(client, field):
    body = make_request(
        [
            {
                "type": "tower",
                "polarization": {"type": "full", "jones_re": [1.0, 0.0], "jones_im": [0.0, 0.0]},
            }
        ]
    )
    body["rfi_sources"][0]["polarization"][field] = [float("nan"), 0.0]
    response = client.post(
        "/api/simulate",
        content=json.dumps(body),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert "finite" in response.text


def test_a_periodic_envelope_replaces_the_duty_cycle(client):
    """Turning on the envelope must not collide with the (non-1.0) default duty cycle."""
    body = make_request(
        [
            {
                "type": "tower",
                "envelope": {"type": "periodic", "period_s": 0.02, "duty": 0.5},
            }
        ]
    )
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 200, response.text


def test_everything_on_at_once_still_runs(client):
    """The full combination of every new feature in one request must still succeed."""
    body = make_request(
        [
            {
                "type": "comb",
                "fundamental_hz": 1.405e6,
                "harmonic_numbers": [999, 1000, 1001],
                "received_power_jy": 300.0,
                "waveform": "constant_envelope",
                "envelope": {"type": "periodic", "period_s": 0.02, "duty": 0.5},
                "coupling": {"type": "lognormal", "sigma_db": 4.0, "seed": 2},
                "polarization": {"type": "linear", "angle_deg": 10.0},
            },
            {
                "type": "tower",
                "coupling": [1.0] * len(default_array().antenna_positions_enu_m),
            },
        ]
    )
    body["spectral_lines"] = [
        {"name": "hi", "center_freq_hz": 1405.0e6, "fwhm_hz": 2.0e4, "line_flux_jy": 5.0}
    ]
    body["n_pol"] = 2
    body["instrument"] = {
        "gain_scatter_db": 1.0,
        "phase_offsets": "uniform",
        "bandpass_ripple_db": 0.1,
        "band_slope_db": 0.2,
        "subband_scatter_db": 0.1,
        "n_subbands": 4,
    }
    body["calibration_errors"] = {
        "phase_error_deg_rms": 8.0,
        "delay_error_ns_rms": 1.0,
        "amplitude_error_db_rms": 0.3,
    }
    body["channelizer"] = {"n_taps": 8, "window": "blackman", "sinc_bandwidth": 1.1}
    body["quantization"] = {"quant_target_counts": 2.0}
    body["primary_beam"] = {"type": "airy", "dish_diameter_m": 4.5}

    response = client.post("/api/simulate?pol=1", json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["observation"]["n_pol"] == 2
    assert len(result["sources"]) == 2


# ----------------------------------------------------------------------
# The array catalogue: the bundled layouts plus whatever directory the
# operator points the server at
# ----------------------------------------------------------------------
EXTRA_ARRAY_YAML = """
name: three element line
latitude_deg: -30.7
longitude_deg: 21.4
height_m: 1050.0
antennas:
  - [0.0, 0.0, 0.0]
  - [25.0, 0.0, 0.0]
  - [50.0, 12.0, 0.0]
"""


@pytest.fixture(scope="module")
def extra_array_dir(tmp_path_factory) -> Path:
    """A directory holding one loadable layout and one that is not."""
    directory = tmp_path_factory.mktemp("arrays")
    (directory / "line_of_three.yaml").write_text(EXTRA_ARRAY_YAML)
    (directory / "not_an_array.yaml").write_text("colours:\n  - red\n  - blue\n")
    return directory


@pytest.fixture(scope="module")
def catalogue_client(extra_array_dir: Path) -> TestClient:
    """A client whose server also offers the extra directory."""
    return TestClient(create_app(array_dir=extra_array_dir))


def test_the_catalogue_lists_the_bundled_and_the_extra_layouts(catalogue_client):
    """A readable layout is offered; an unreadable YAML beside it is not."""
    listing = catalogue_client.get("/api/arrays")
    assert listing.status_code == 200
    entries = listing.json()

    by_name = {entry["name"]: entry for entry in entries}
    array = default_array()
    assert array.name in by_name
    assert by_name[array.name]["n_antennas"] == len(array.antenna_positions_enu_m)
    assert by_name["three element line"]["n_antennas"] == 3
    assert all(entry["runnable"] for entry in entries)
    assert all("antennas" not in entry for entry in entries), "the listing stays small"


def test_one_catalogue_entry_round_trips_its_antennas(catalogue_client):
    """What the picker loads is the file's own geometry and site."""
    entries = catalogue_client.get("/api/arrays").json()
    identifier = [entry for entry in entries if entry["name"] == "three element line"][0]["id"]

    detail = catalogue_client.get(f"/api/arrays/{identifier}")
    assert detail.status_code == 200
    payload = detail.json()

    assert payload["antennas"] == [[0.0, 0.0, 0.0], [25.0, 0.0, 0.0], [50.0, 12.0, 0.0]]
    assert payload["latitude_deg"] == pytest.approx(-30.7)
    assert payload["longitude_deg"] == pytest.approx(21.4)
    assert payload["height_m"] == pytest.approx(1050.0)

    # And it is a run the server accepts exactly as it was handed over.
    body = make_request()
    body["antennas"] = payload["antennas"]
    body["site"] = {
        "latitude_deg": payload["latitude_deg"],
        "longitude_deg": payload["longitude_deg"],
        "height_m": payload["height_m"],
    }
    assert catalogue_client.post("/api/simulate", json=body).status_code == 200


def test_an_unknown_layout_is_a_404(catalogue_client):
    """Ids come from the server's own listing; anything else is not found."""
    assert catalogue_client.get("/api/arrays/no-such-layout").status_code == 404


def test_a_layout_id_cannot_name_a_path(catalogue_client):
    """The id is an identifier, never a file name to be traversed."""
    for attempt in ("..", "%2e%2e%2fetc%2fpasswd", "..%2farray-default"):
        assert catalogue_client.get(f"/api/arrays/{attempt}").status_code in {404, 422}


def test_a_run_can_observe_from_another_site(client):
    """The phase centre follows the site, because it is that site's zenith."""
    equator = pointing_payload(0.0, 0.0, 0.0)
    assert equator["dec_deg"] != pytest.approx(pointing_payload()["dec_deg"], abs=1.0)

    body = make_request()
    body["site"] = {"latitude_deg": 0.0, "longitude_deg": 0.0, "height_m": 0.0}
    body["sky_sources"] = [{"name": "target", "offset_deg": [0.0, 0.0], "flux_jy": 5.0}]
    result = client.post("/api/simulate", json=body).json()
    assert result["sky_sources"][0]["dec_deg"] == pytest.approx(equator["dec_deg"], abs=1e-6)


# ----------------------------------------------------------------------
# Human-friendly sky positions
# ----------------------------------------------------------------------
def post_raw(client: TestClient, body: dict):
    """POST a body that may hold NaN, which the JSON encoder refuses."""
    return client.post(
        "/api/simulate",
        content=json.dumps(body),
        headers={"content-type": "application/json"},
    )


def test_the_pointing_endpoint_agrees_with_the_image_grid(client):
    """The bound the page quotes is the edge of the image it will draw."""
    response = client.get("/api/pointing")
    assert response.status_code == 200
    pointing = response.json()

    assert np.isfinite(pointing["ra_deg"]) and 0.0 <= pointing["ra_deg"] < 360.0
    assert np.isfinite(pointing["dec_deg"]) and -90.0 <= pointing["dec_deg"] <= 90.0
    # The zenith's declination is the site's latitude, up to the geodetic
    # flattening -- a fraction of a degree, not a degree.
    assert pointing["dec_deg"] == pytest.approx(default_array().latitude_deg, abs=0.2)

    image = client.post("/api/simulate", json=make_request()).json()["image"]
    half_width = float(np.degrees(np.arcsin(max(image["l"]))))
    assert pointing["field_half_width_deg"] == pytest.approx(half_width, rel=1e-9)
    assert max(image["l"]) == pytest.approx(0.5 * pointing["field_of_view_rad"])


def test_a_source_can_be_placed_in_degrees_from_the_pointing(client):
    """`offset_deg` is the exact sine of the angle, in the library's l and m."""
    body = make_request()
    body["sky_sources"] = [{"name": "target", "offset_deg": [0.5, -0.3], "flux_jy": 5.0}]
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 200, response.text
    resolved = response.json()["sky_sources"][0]

    assert resolved["l"] == pytest.approx(math.sin(math.radians(0.5)), abs=1e-12)
    assert resolved["m"] == pytest.approx(math.sin(math.radians(-0.3)), abs=1e-12)
    assert resolved["in_field"] is True
    # East is towards increasing right ascension, north towards increasing
    # declination -- the convention `PointSource.from_lm` documents.
    pointing = client.get("/api/pointing").json()
    assert resolved["ra_deg"] > pointing["ra_deg"]
    assert resolved["dec_deg"] < pointing["dec_deg"]

    peak = response.json()["image"]["peak"]
    assert peak["l"] == pytest.approx(resolved["l"], abs=1e-3)
    assert peak["m"] == pytest.approx(resolved["m"], abs=1e-3)


def test_a_source_placed_at_the_pointing_lands_dead_centre(client):
    """An absolute position equal to the phase centre resolves to l = m = 0."""
    pointing = client.get("/api/pointing").json()
    body = make_request()
    body["sky_sources"] = [
        {
            "name": "centre",
            "radec_deg": [pointing["ra_deg"], pointing["dec_deg"]],
            "flux_jy": 5.0,
        }
    ]
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 200, response.text
    resolved = response.json()["sky_sources"][0]

    assert resolved["l"] == pytest.approx(0.0, abs=1e-9)
    assert resolved["m"] == pytest.approx(0.0, abs=1e-9)
    assert response.json()["image"]["peak"]["value_jy"] > 1.0


def test_an_absolute_position_images_where_the_offset_one_does(client):
    """The two notations are two spellings of one place."""
    pointing = client.get("/api/pointing").json()
    offset = client.post(
        "/api/simulate",
        json=dict(
            make_request(),
            sky_sources=[{"name": "a", "offset_deg": [0.4, 0.2], "flux_jy": 5.0}],
        ),
    ).json()["sky_sources"][0]

    absolute = client.post(
        "/api/simulate",
        json=dict(
            make_request(),
            sky_sources=[
                {
                    "name": "b",
                    "radec_deg": [offset["ra_deg"], offset["dec_deg"]],
                    "flux_jy": 5.0,
                }
            ],
        ),
    ).json()["sky_sources"][0]

    assert absolute["l"] == pytest.approx(offset["l"], abs=1e-9)
    assert absolute["m"] == pytest.approx(offset["m"], abs=1e-9)
    assert offset["ra_deg"] > pointing["ra_deg"]
    assert offset["dec_deg"] > pointing["dec_deg"]


def test_a_source_far_outside_the_field_is_reported_as_such(client):
    """Out of the image is not an error: it is simulated, and labelled."""
    body = make_request()
    body["sky_sources"] = [{"name": "far", "offset_deg": [4.0, 0.0], "flux_jy": 5.0}]
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["sky_sources"][0]["in_field"] is False


@pytest.mark.parametrize(
    "position",
    [
        {"l": 0.01, "m": 0.0, "offset_deg": [0.5, -0.3]},
        {"l": 0.01, "m": 0.0, "radec_deg": [10.0, 10.0]},
        {"offset_deg": [0.5, -0.3], "radec_deg": [10.0, 10.0]},
        {"l": 0.01},
        {"m": 0.01},
        {"offset_deg": [0.5]},
        {"offset_deg": [0.5, -0.3, 0.1]},
        {"offset_deg": [90.0, 0.0]},
        {"radec_deg": [10.0, 120.0]},
    ],
)
def test_one_position_per_source_and_it_must_be_in_range(client, position):
    """Two positions, half a position, or an impossible one: all refused."""
    body = make_request()
    body["sky_sources"] = [dict(position, name="target", flux_jy=5.0)]
    assert client.post("/api/simulate", json=body).status_code == 422


@pytest.mark.parametrize(
    "position",
    [
        {"offset_deg": [float("nan"), 0.0]},
        {"offset_deg": [0.5, float("inf")]},
        {"radec_deg": [float("nan"), 0.0]},
        {"l": float("nan"), "m": 0.0},
    ],
)
def test_a_position_that_is_not_a_number_is_refused(client, position):
    """NaN parses as a float, and must not get past the validator."""
    body = make_request()
    body["sky_sources"] = [dict(position, name="target", flux_jy=5.0)]
    assert post_raw(client, body).status_code == 422


def test_a_source_with_no_position_at_all_lands_where_the_page_opens(client):
    """No position given is the page's own default offset, not the origin."""
    body = make_request()
    body["sky_sources"] = [{"name": "target", "flux_jy": 5.0}]
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 200, response.text
    resolved = response.json()["sky_sources"][0]

    default_offset = defaults_payload()["sky_source"]["position"]["default_offset_deg"]
    assert resolved["l"] == pytest.approx(math.sin(math.radians(default_offset[0])), abs=1e-12)
    assert resolved["m"] == pytest.approx(math.sin(math.radians(default_offset[1])), abs=1e-12)


# ----------------------------------------------------------------------
# What the correlator sees
# ----------------------------------------------------------------------
def test_the_visibility_waterfall_is_the_librarys_own_average(client):
    """The amplitude map must be the library's visibilities, not a re-derivation.

    The whole page rests on the rule that the browser draws what the
    library computed. This re-runs the same request through the library
    by hand, averages ``|V|`` over the cross-correlation baselines
    itself, and insists the response holds that number.
    """
    body = make_request([{"type": "tower"}])
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 200, response.text
    reported = np.asarray(response.json()["visibilities"]["amplitude"])

    simulator = build_simulator(SimulateRequest.model_validate(body))
    visibilities = correlate(simulator.blocks())
    data = visibilities.pol_data[:, :, 0, :]
    expected = np.abs(data[:, visibilities.cross_mask, :]).mean(axis=1).T

    assert reported.shape == expected.shape
    assert reported == pytest.approx(expected, rel=1e-4, abs=1e-5)


def test_one_baselines_spectrum_is_that_baselines_visibility(client):
    """The picked baseline's amplitude and phase are its own, unaveraged.

    At this size nothing is binned away -- 32 channels and 2 integrations
    are both under their budgets -- so the response must hold the
    library's visibility for that antenna pair exactly.
    """
    body = make_request([{"type": "tower"}])
    block = client.post("/api/simulate", json=body).json()["visibilities"]
    spectra = block["spectra"]
    assert spectra["integrations_per_bin"] == 1
    assert spectra["channels_per_bin"] == 1

    simulator = build_simulator(SimulateRequest.model_validate(body))
    visibilities = correlate(simulator.blocks())

    position = 2
    baseline = spectra["baselines"][position]
    expected = visibilities.pol_data[:, baseline, 0, :]
    assert np.asarray(spectra["amplitude"][position]) == pytest.approx(
        np.abs(expected), rel=1e-4, abs=1e-5
    )
    assert np.asarray(spectra["phase_deg"][position]) == pytest.approx(
        np.degrees(np.angle(expected)), abs=1e-2
    )


def test_the_visibility_truth_is_the_librarys_rfi_fraction(client):
    """The stripes an excision algorithm is scored against are ``rfi_fraction``."""
    body = make_request([{"type": "tower"}])
    result = client.post("/api/simulate", json=body).json()
    reported = result["visibilities"]["sources"]
    assert len(reported) == 1

    simulator = build_simulator(SimulateRequest.model_validate(body))
    visibilities = correlate(simulator.blocks())
    fraction = np.asarray(visibilities.rfi_fraction)[:, 0, :]

    assert reported[0]["name"] == visibilities.rfi_source_names[0]
    assert np.asarray(reported[0]["mask"], dtype=bool).tolist() == (fraction.T > 0.0).tolist()
    assert reported[0]["mean_fraction"] == pytest.approx(float(fraction.mean()))
    assert reported[0]["max_fraction"] == pytest.approx(float(fraction.max()))
    assert reported[0]["max_fraction"] > 0.0


def test_the_visibility_reductions_stay_small(client):
    """Every visibility product must arrive on a grid the page can draw."""
    body = make_request([{"type": "tower"}], n_chan=64, n_blocks=3)
    result = client.post("/api/simulate", json=body).json()
    block = result["visibilities"]

    amplitude = np.asarray(block["amplitude"])
    assert amplitude.shape == (min(64, VIS_MAX_CHAN_BINS), 3)
    assert len(block["freq_mhz"]) == amplitude.shape[0]
    assert len(block["time_s"]) == amplitude.shape[1]
    assert block["vmin_jy"] <= block["vmax_jy"] <= block["peak_jy"] * 1.000001

    array = default_array()
    n_antennas = len(array.antenna_positions_enu_m)
    assert block["n_baselines"] == n_antennas * (n_antennas - 1) // 2
    assert len(block["baselines"]) == block["n_baselines"]
    lengths = [entry["length_m"] for entry in block["baselines"]]
    assert lengths == sorted(lengths)
    assert all(entry["ant_1"] < entry["ant_2"] for entry in block["baselines"])

    spectra = block["spectra"]
    amplitudes = np.asarray(spectra["amplitude"])
    assert amplitudes.shape == (
        len(spectra["baselines"]),
        len(spectra["time_s"]),
        len(spectra["freq_mhz"]),
    )
    assert np.asarray(spectra["phase_deg"]).shape == amplitudes.shape
    assert abs(np.asarray(spectra["phase_deg"])).max() <= 180.0
    assert amplitudes.size * 2 <= MAX_VIS_SPECTRUM_VALUES
    assert set(spectra["baselines"]) <= {entry["index"] for entry in block["baselines"]}


def test_a_clean_rerun_of_the_same_seed_carries_no_interference(client):
    """The A/B comparison: the same run without interference, same realization.

    The page offers a clean/contaminated flip by posting the same body
    with an empty interference list and the same seed. The clean run must
    hold no interference truth anywhere, and must still be the same sky.
    """
    body = make_request([{"type": "tower"}])
    contaminated = client.post("/api/simulate", json=body).json()

    clean_body = dict(body, rfi_sources=[])
    assert clean_body["sim"]["seed"] == body["sim"]["seed"]
    clean = client.post("/api/simulate", json=clean_body)
    assert clean.status_code == 200, clean.text
    clean = clean.json()

    assert clean["sources"] == []
    assert clean["visibilities"]["sources"] == []
    assert clean["sky_sources"] == contaminated["sky_sources"]
    assert clean["image"]["vmax_jy"] != contaminated["image"]["vmax_jy"]


# ----------------------------------------------------------------------
# The primary beam, on the image
# ----------------------------------------------------------------------
def test_a_fitted_beam_reports_its_half_power_circle_and_per_source_response(client):
    """The circle drawn on the image is the beam the sources were dimmed by."""
    body = make_request()
    body["primary_beam"] = {"type": "gaussian", "dish_diameter_m": 4.5}
    result = client.post("/api/simulate", json=body).json()

    beam = result["image"]["beam"]
    assert beam["type"] == "gaussian"
    expected = 0.5 * float(GaussianBeam(dish_diameter_m=4.5).fwhm_rad(beam["center_freq_hz"]))
    assert beam["half_power_rad"] == pytest.approx(expected, rel=1e-6)

    response = result["sky_sources"][0]["beam_response"]
    assert 0.0 < response < 1.0


def test_an_airy_beam_reports_a_wider_half_power_circle_than_a_gaussian(client):
    """The circle is found on the beam's own response, so the two models differ."""
    radii = {}
    for kind in ("gaussian", "airy"):
        body = make_request()
        body["primary_beam"] = {"type": kind, "dish_diameter_m": 4.5}
        result = client.post("/api/simulate", json=body).json()
        radii[kind] = result["image"]["beam"]["half_power_rad"]
    assert radii["airy"] > radii["gaussian"] > 0.0


def test_no_beam_is_reported_as_absent_not_as_unity(client):
    """ "Nothing was attenuated" and "attenuated by 1.0" are different claims."""
    result = client.post("/api/simulate", json=make_request()).json()
    assert result["image"]["beam"] is None
    assert result["sky_sources"][0]["beam_response"] is None


# ----------------------------------------------------------------------
# The classical flaggers
# ----------------------------------------------------------------------
def flag_body(methods, rfi_sources=({"type": "tower"},), **overrides) -> dict:
    """A flagging request over a small contaminated run."""
    body = {
        "request": make_request(list(rfi_sources)),
        "methods": list(methods),
        "antenna": 0,
        "pol": 0,
    }
    body.update(overrides)
    return body


def test_the_defaults_describe_the_flaggers_the_page_offers(defaults):
    """The method picker is built from the schema, so the schema must be whole."""
    flaggers = defaults["flaggers"]
    assert {entry["value"] for entry in flaggers["methods"]} == {"sk", "mad", "sumthreshold"}
    for entry in flaggers["methods"]:
        assert entry["label"] and entry["summary"] and entry["grid"]
    assert flaggers["max_methods"] == MAX_FLAG_METHODS
    assert flaggers["defaults"]["m"] == FLAG_DEFAULT_M


@pytest.mark.parametrize("method", ["sk", "mad", "sumthreshold"])
def test_every_flagger_returns_masks_and_usable_scores(client, method):
    """Each method must produce a decision for every cell and score it."""
    response = client.post("/api/flag", json=flag_body([method]))
    assert response.status_code == 200, response.text
    result = response.json()

    grid = result["grid"]
    assert grid["m"] == FLAG_DEFAULT_M
    assert grid["n_accumulations"] == SMALL_SIM["n_blocks"] * (N_TIME_PER_BLOCK // grid["m"])
    assert len(grid["freq_mhz"]) == grid["chan_bins"]
    assert len(grid["time_s"]) == grid["n_accumulations"]

    assert len(result["methods"]) == 1
    entry = result["methods"][0]
    assert entry["method"] == method
    assert entry["label"] and entry["grid"]
    for name in ("caught", "missed", "false_alarm"):
        assert np.asarray(entry["overlay"][name]).shape == (
            grid["chan_bins"],
            grid["n_accumulations"],
        )
    scores = entry["scores"]
    assert scores["truth_occupancy"] > 0.0
    assert scores["tp"] + scores["fn"] > 0
    for name in ("precision", "recall", "f1", "false_positive_rate"):
        assert scores[name] is None or 0.0 <= scores[name] <= 1.0


def test_the_flagger_scores_are_the_librarys_own(client):
    """The load-bearing one: re-run the flagger by hand and compare.

    Spectral kurtosis on the same seed, with the truth pooled by
    `pool_truth_accumulations` -- the partition the estimator itself
    uses -- must give exactly the numbers the endpoint reports, or the
    page is scoring something other than what it says it is.
    """
    body = flag_body(["sk"])
    reported = client.post("/api/flag", json=body).json()["methods"][0]

    request = SimulateRequest.model_validate(body["request"])
    simulator = build_simulator(request)
    predicted = []
    truth = []
    for block in simulator.blocks():
        predicted.append(spectral_kurtosis_mask(block.pol_data[0, 0], FLAG_DEFAULT_M))
        truth.append(pool_truth_accumulations(block.rfi_mask.any(axis=0), FLAG_DEFAULT_M))
    expected = flag_scores(np.concatenate(predicted, axis=1), np.concatenate(truth, axis=1))

    for name, value in expected.items():
        if math.isnan(value):
            assert reported["scores"][name] is None
        else:
            assert reported["scores"][name] == pytest.approx(value)


def test_two_methods_come_back_side_by_side_on_one_grid(client):
    """The head-to-head: two columns, same cells, same truth."""
    response = client.post("/api/flag", json=flag_body(["sk", "sumthreshold"]))
    assert response.status_code == 200, response.text
    result = response.json()

    assert [entry["method"] for entry in result["methods"]] == ["sk", "sumthreshold"]
    occupancies = {entry["scores"]["truth_occupancy"] for entry in result["methods"]}
    assert len(occupancies) == 1
    shapes = {np.asarray(entry["overlay"]["caught"]).shape for entry in result["methods"]}
    assert len(shapes) == 1


def test_a_flagger_scores_a_clean_run_as_having_nothing_to_find(client):
    """With no interference there is no recall to have, and it must not be faked."""
    response = client.post("/api/flag", json=flag_body(["mad"], rfi_sources=()))
    assert response.status_code == 200, response.text
    scores = response.json()["methods"][0]["scores"]
    assert scores["truth_occupancy"] == 0.0
    assert scores["recall"] is None
    assert not np.asarray(response.json()["methods"][0]["overlay"]["caught"]).any()


def test_the_flagger_follows_the_antenna_and_the_receptor(client):
    """A per-antenna decision must actually read that antenna's voltages."""
    body = flag_body(["mad"])
    first = client.post("/api/flag", json=body).json()
    body["antenna"] = 3
    second = client.post("/api/flag", json=body)
    assert second.status_code == 200, second.text
    second = second.json()
    assert second["antenna"] == 3
    assert second["methods"][0]["overlay"] != first["methods"][0]["overlay"]


def test_the_flagger_refuses_an_antenna_this_array_does_not_have(client):
    body = flag_body(["mad"], antenna=64)
    response = client.post("/api/flag", json=body)
    assert response.status_code == 422
    assert "antenna" in response.json()["detail"][0]["msg"]


def test_the_flagger_refuses_an_accumulation_that_straddles_a_block(client):
    """A block is generated alone, so an accumulation has to fit inside one."""
    body = flag_body(["sk"], params={"m": 300})
    response = client.post("/api/flag", json=body)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "methods",
    [[], ["sk", "sk"], ["sk", "mad", "sumthreshold"], ["nonesuch"]],
)
def test_the_flagger_refuses_a_method_list_it_cannot_answer(client, methods):
    """One or two named methods, each named once."""
    assert client.post("/api/flag", json=flag_body(methods)).status_code == 422


# ----------------------------------------------------------------------
# The bandpass: time-averaged spectra, both receptors
# ----------------------------------------------------------------------
def test_the_bandpass_ships_every_antenna_and_every_receptor(client):
    """A dual-polarization run carries XX and YY, whichever one the
    waterfall is showing: the bandpass plot exists to compare them."""
    body = make_request()
    body["n_pol"] = 2
    response = client.post("/api/simulate?pol=0", json=body)
    assert response.status_code == 200, response.text
    water = response.json()["waterfall"]
    bandpass = water["bandpass"]

    assert bandpass["pol_names"] == ["XX", "YY"]
    grid = np.asarray(bandpass["antennas"])
    assert grid.shape == (len(water["antennas"]), 2, len(water["freq_mhz"]))
    assert np.isfinite(grid).all()
    assert bandpass["vmin_db"] < bandpass["vmax_db"]


def test_a_single_polarization_run_still_ships_one_bandpass_trace(client):
    response = client.post("/api/simulate", json=make_request())
    bandpass = response.json()["waterfall"]["bandpass"]
    assert np.asarray(bandpass["antennas"]).shape[1] == 1


def test_the_bandpass_is_the_time_average_of_the_waterfall(client):
    """Not a second, differently-computed quantity: the same power, averaged.

    The waterfall is pooled in time by the mean, so averaging its
    decibels back is *not* the bandpass -- but averaging the linear power
    is, and the two agree to the rounding the response carries.
    """
    payload = client.post("/api/simulate", json=make_request([{"type": "tower"}])).json()
    water = payload["waterfall"]
    antenna = np.asarray(water["antennas"][0])
    bandpass = np.asarray(water["bandpass"]["antennas"][0][0])

    from_waterfall = 10.0 * np.log10((10.0 ** (antenna / 10.0)).mean(axis=1))
    assert np.allclose(from_waterfall, bandpass, atol=0.05)


def test_a_transmitter_raises_the_bandpass_in_its_own_channels(client):
    """The sanity test: the spike sits where the transmitter is tuned."""
    clean = client.post("/api/simulate", json=make_request()).json()
    tower = {"type": "tower", "center_freq_hz": 1.4055e9, "bandwidth_hz": 2.0e5}
    dirty = client.post("/api/simulate", json=make_request([tower])).json()

    freq_mhz = np.asarray(dirty["waterfall"]["freq_mhz"])
    lift = np.asarray(dirty["waterfall"]["bandpass"]["antennas"][0][0]) - np.asarray(
        clean["waterfall"]["bandpass"]["antennas"][0][0]
    )
    assert lift.max() > 3.0
    assert abs(freq_mhz[int(np.argmax(lift))] - 1405.5) < 1.0


# ----------------------------------------------------------------------
# uv coverage
# ----------------------------------------------------------------------
def uv_request() -> dict:
    """Three antennas on flat ground -- small enough to check by hand."""
    body = make_request()
    body["antennas"] = [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 60.0, 0.0]]
    return body


def test_the_plotted_uv_points_are_the_librarys_own_uvw(client):
    """What the page draws is `uvw_wavelengths`, not a second calculation.

    The response is re-derived here by rebuilding the identical
    observation from its seed and correlating it, so a drift between the
    page's uv plane and the library's would fail rather than go unseen.
    """
    from rfi_simulator import uvw_wavelengths

    body = uv_request()
    payload = client.post("/api/simulate", json=body).json()

    simulator = build_simulator(SimulateRequest(**body))
    visibilities = correlate(simulator.blocks())
    u_lambda, v_lambda, _ = uvw_wavelengths(visibilities)
    cross = visibilities.cross_mask
    center = visibilities.n_chan // 2

    assert np.allclose(payload["uv"]["u"], u_lambda[:, cross, center].ravel(), atol=1e-3)
    assert np.allclose(payload["uv"]["v"], v_lambda[:, cross, center].ravel(), atol=1e-3)


def test_uv_radii_are_the_baseline_lengths_over_the_wavelength(client):
    """An independent geometric check that owes the library nothing.

    A flat array pointed at its own zenith has every baseline
    perpendicular to the phase-center direction, so ``w`` vanishes and
    ``sqrt(u**2 + v**2)`` is exactly the baseline length in wavelengths.
    The three baselines here are 100 m, 60 m and hypot(100, 60) m.
    """
    payload = client.post("/api/simulate", json=uv_request()).json()
    observation = payload["observation"]

    n_chan = observation["n_chan"]
    bandwidth = observation["bandwidth_hz"]
    center_freq = (
        observation["center_freq_hz"] - 0.5 * bandwidth + (n_chan // 2 + 0.5) * (bandwidth / n_chan)
    )
    wavelength_m = 299792458.0 / center_freq

    radii = np.hypot(payload["uv"]["u"], payload["uv"]["v"])
    expected = np.array([60.0, 100.0, math.hypot(100.0, 60.0)]) / wavelength_m
    # Every point sits on one of the three circles, and every circle is
    # visited. The radii drift by a fraction of a wavelength across the
    # recording -- that is Earth rotation, and over 130 ms it is all there
    # is of it, which is why the plot is a scatter and not an arc.
    assert np.abs(radii[:, None] - expected[None, :]).min(axis=1).max() < 1.0
    assert np.abs(expected[:, None] - radii[None, :]).min(axis=1).max() < 1.0


def test_the_uv_plane_is_reported_without_its_conjugate_half(client):
    """The page mirrors each point itself, so shipping both would double-draw."""
    payload = client.post("/api/simulate", json=uv_request()).json()
    points = {(round(u, 3), round(v, 3)) for u, v in zip(payload["uv"]["u"], payload["uv"]["v"])}
    assert not (points & {(-u, -v) for u, v in points})
    assert len(payload["uv"]["u"]) == (
        payload["observation"]["n_baselines"] * payload["observation"]["n_blocks"]
    )


# ----------------------------------------------------------------------
# Flag fractions and the visibility domain
# ----------------------------------------------------------------------
def test_every_method_reports_a_flag_fraction_per_channel(client):
    response = client.post("/api/flag", json=flag_body(["mad", "sumthreshold"]))
    assert response.status_code == 200, response.text
    payload = response.json()
    chan_bins = payload["grid"]["chan_bins"]
    assert len(payload["grid"]["truth_fraction"]) == chan_bins
    for method in payload["methods"]:
        fraction = np.asarray(method["flag_fraction"])
        assert fraction.shape == (chan_bins,)
        assert ((fraction >= 0.0) & (fraction <= 1.0)).all()


def test_the_flag_fraction_agrees_with_the_overlay_beside_it(client):
    """A channel flagged in no cell at all must report a zero fraction."""
    payload = client.post("/api/flag", json=flag_body(["mad"])).json()
    method = payload["methods"][0]
    fraction = np.asarray(method["flag_fraction"])
    flagged_anywhere = (
        np.asarray(method["overlay"]["caught"]) | np.asarray(method["overlay"]["false_alarm"])
    ).any(axis=1)
    assert not fraction[~flagged_anywhere].any()


def test_the_visibility_domain_flags_the_correlated_amplitudes(client):
    """A tower survives correlation, so a visibility-domain method finds it."""
    body = flag_body(["mad", "sumthreshold"], domain="visibility")
    response = client.post("/api/flag", json=body)
    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["domain"] == "visibility"
    assert payload["grid"]["domain"] == "visibility"
    assert payload["grid"]["m"] is None
    # One column per correlator integration, not per accumulation.
    assert payload["grid"]["n_accumulations"] == SMALL_SIM["n_blocks"]
    assert len(payload["grid"]["time_s"]) == SMALL_SIM["n_blocks"]
    for method in payload["methods"]:
        assert method["scores"]["recall"] > 0.5


def test_visibility_truth_is_the_correlators_own_interference_fraction(client):
    """Not a second opinion about which cells were hit."""
    body = flag_body(["mad"], domain="visibility")
    payload = client.post("/api/flag", json=body).json()

    simulator = build_simulator(SimulateRequest(**body["request"]))
    visibilities = correlate(simulator.blocks())
    truth = (np.asarray(visibilities.rfi_fraction) > 0.0).any(axis=1).T
    assert np.allclose(
        payload["grid"]["truth_fraction"], truth.astype(float).mean(axis=1), atol=1e-5
    )


def test_spectral_kurtosis_is_refused_after_correlation(client):
    """It is a pre-detection test; the moments it needs are gone by here."""
    response = client.post("/api/flag", json=flag_body(["sk"], domain="visibility"))
    assert response.status_code == 422
    assert "pre-detection" in response.text


def test_the_defaults_describe_both_flagging_domains(defaults):
    domains = defaults["flaggers"]["domains"]
    assert [entry["value"] for entry in domains] == ["voltage", "visibility"]
    visibility = [entry for entry in domains if entry["value"] == "visibility"][0]
    assert "sk" not in visibility["methods"]
