/* Interference simulator console -- vanilla JS, no framework, no network
 * beyond this server's own endpoints (defaults, arrays, pointing, simulate,
 * flag).
 *
 * Shape of the file:
 *   1. constants and small helpers          7. the run
 *   2. state                                8. result displays
 *   3. request wrapper                      9. the flagger overlay
 *   4. site plan                           10. tabs and navigation
 *   5. source and observation forms        11. tooltips
 *   6. scenario presets                    12. wiring and boot
 *
 * Rule of the house: the browser never computes physics. Masks, images,
 * scores, occupancies and warnings all arrive from the server as ground
 * truth; this file only arranges pixels. The one arithmetic it does own is
 * drawing: colour mapping, data-to-plot coordinate transforms, and scaling
 * a server-sent grid down into a thumbnail.
 *
 * The page is two tabs. Setup is the forms; Results is the three levels the
 * data passes through -- voltages, visibilities, image -- one panel each,
 * each showing one plot and one primary toggle with everything else folded
 * away, because the panels are read far more often than they are adjusted.
 */
(function () {
  "use strict";

  /* ------------------------------------------------------------ 1. bits */

  // Magma-like sequential ramp, dark-low to bright-high: perceptually
  // near-uniform, safe in greyscale, and not a rainbow.
  var MAGMA = [
    [0, 0, 4], [24, 15, 61], [68, 15, 118], [114, 31, 129],
    [158, 47, 127], [205, 64, 113], [240, 96, 93], [251, 147, 91],
    [254, 201, 141], [252, 253, 191]
  ];

  // Plot-surface colours. These live on the dark canvases, so they are the
  // dark-surface members of the page palette: mask red = --error-line,
  // truth marker = --amber, sky swatch = --plot-point.
  var MASK_RGB = "214, 48, 49";
  var MASK_ALPHA = 0.45;
  var SKY_MARKER = "#74b9ff";
  var TRUTH_MARKER = "#fdcb6e";

  // The flagger overlay's three outcomes. Green is the only place on the
  // page that means "right"; the red is the same red the ground-truth mask
  // uses, because a missed cell is exactly a truth cell nobody caught, and
  // the amber is the page's "look here" colour for a cell wrongly flagged.
  var FLAG_COLOURS = {
    caught: "0, 184, 148",
    missed: "214, 48, 49",
    false_alarm: "253, 203, 110"
  };
  var FLAG_ALPHA = 0.55;
  var AXIS_STROKE = "#39404b";
  var GRID_STROKE = "#262c35";
  var AXIS_TEXT = "#9aa1ac";
  var PLOT_FONT = '11px Menlo, Consolas, ui-monospace, monospace';

  var VIEW = 400;          // site-plan internal coordinate box, px
  var SHEET_PAD = 30;      // margin inside it, px
  var MIN_SPACING_M = 5;   // how far apart a dropped antenna tries to land
  var LARGE_ARRAY = 32;    // above this, a loaded layout gets a "slower run" note
  var HISTORY_MAX = 5;     // completed runs kept in memory for the run strip
  var NICE_STEPS = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000];

  function $(id) { return document.getElementById(id); }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) { node.className = className; }
    if (text !== undefined && text !== null) { node.textContent = text; }
    return node;
  }

  // A readout table: uppercase column heads, monospace right-aligned
  // values, one hairline per row. `headers` is a list of strings, `rows` a
  // list of equal-length lists of already-formatted strings. An empty
  // `rows` empties the host, which CSS then collapses.
  function fillMetricTable(host, headers, rows) {
    while (host.firstChild) { host.removeChild(host.firstChild); }
    if (!rows.length) { return; }
    var table = el("table", "metric-table");
    var head = el("tr");
    headers.forEach(function (label) {
      var cell = el("th", null, label);
      cell.scope = "col";
      head.appendChild(cell);
    });
    var thead = el("thead");
    thead.appendChild(head);
    table.appendChild(thead);
    var body = el("tbody");
    rows.forEach(function (values) {
      var line = el("tr");
      values.forEach(function (value) { line.appendChild(el("td", null, value)); });
      body.appendChild(line);
    });
    table.appendChild(body);
    host.appendChild(table);
  }

  function clamp(value, low, high) {
    return Math.min(high, Math.max(low, value));
  }

  // The primary toggles are real buttons carrying their own state, so the
  // pressed attribute is the state and the class is only what it looks
  // like. One helper keeps the two from drifting apart.
  function setPressed(button, on) {
    button.setAttribute("aria-pressed", String(Boolean(on)));
  }

  // A score the server could not define -- precision with nothing flagged,
  // say -- arrives as null. It is printed as a dash, never as "null".
  function score(value, digits) {
    if (value === null || value === undefined || !isFinite(value)) { return "—"; }
    return value.toFixed(digits === undefined ? 3 : digits);
  }

  function fmt(value, digits) {
    if (!isFinite(value)) { return "--"; }
    return value.toFixed(digits === undefined ? 1 : digits);
  }

  function magma(t) {
    var x = clamp(t, 0, 1) * (MAGMA.length - 1);
    var i = Math.min(MAGMA.length - 2, Math.floor(x));
    var f = x - i;
    var a = MAGMA[i], b = MAGMA[i + 1];
    return [
      Math.round(a[0] + (b[0] - a[0]) * f),
      Math.round(a[1] + (b[1] - a[1]) * f),
      Math.round(a[2] + (b[2] - a[2]) * f)
    ];
  }

  function niceStep(target) {
    for (var i = 0; i < NICE_STEPS.length; i += 1) {
      if (NICE_STEPS[i] >= target) { return NICE_STEPS[i]; }
    }
    return NICE_STEPS[NICE_STEPS.length - 1];
  }

  function ticks(low, high, count) {
    var out = [];
    for (var i = 0; i <= count; i += 1) {
      out.push(low + (high - low) * (i / count));
    }
    return out;
  }

  /* ----------------------------------------------------------- 2. state */

  var state = {
    defaults: null,
    antennas: [],
    site: null,           // {latitude_deg, longitude_deg, height_m, name}
    pointing: null,       // {ra_deg, dec_deg, field_half_width_deg, ...}
    arrays: [],           // catalogue from /api/arrays
    loadedArray: null,    // JSON of the antennas as last loaded or reset
    skySources: [],
    rfiSources: [],
    rfiOpen: [],         // whether each rfi card's "More details" fold is open
    spectralLines: [],
    sim: {},
    realism: {},
    result: null,
    running: false,
    abRunning: false,     // the clean comparison is a second server run
    tab: "setup",         // "setup" or "results"; mirrored in location.hash
    booted: false,        // false until the first automatic run has landed
    waterfallAntenna: 0,
    waterfallPol: 0,
    allAntennas: true,    // the thumbnail wall is where a run lands; see below
    truthVisible: false,  // the voltage panel's master ground-truth switch
    maskVisible: [],
    hatch: false,
    visTruth: false,      // the visibility panel's own ground-truth switch
    visBaseline: null,    // `index` of the baseline whose spectra are drawn
    flagMethods: [],      // which methods the flagger control has ticked
    flagRunning: false,
    flagError: false,     // keeps a refusal on screen until something changes
    notices: [],
    obs: null,            // the mock observatory's day; see resetObservatory
    sky: { data: null, timer: null, error: null },   // the live monitor
    history: [],          // last HISTORY_MAX completed runs, oldest first
    historyIndex: -1,     // which of them the result displays are showing
    elapsedTimer: null,
    drag: null
  };

  /* --------------------------------------------------------- 3. request */

  function request(path, options) {
    return fetch(path, options).then(function (response) {
      return response.json().catch(function () { return null; }).then(function (body) {
        if (response.ok) { return body; }
        throw new Error(readProblem(body, response.status));
      });
    });
  }

  // FastAPI reports validation problems as a list of {loc, msg}; turn that
  // into one sentence a person can act on.
  function readProblem(body, status) {
    if (!body || !body.detail) { return "The server refused the run (status " + status + ")."; }
    if (typeof body.detail === "string") { return body.detail; }
    return body.detail.map(function (item) {
      var where = (item.loc || []).filter(function (part) {
        return part !== "body";
      }).join(" › ");
      return where ? where + ": " + item.msg : item.msg;
    }).join("\n");
  }

  /* ------------------------------------------------------- 4. site plan */

  function extentMetres() {
    var largest = 50;
    state.antennas.forEach(function (antenna) {
      largest = Math.max(largest, Math.abs(antenna[0]), Math.abs(antenna[1]));
    });
    return Math.ceil(largest * 1.25 / 25) * 25;
  }

  function planScale() {
    return (VIEW / 2 - SHEET_PAD) / extentMetres();
  }

  function toView(east, north) {
    var s = planScale();
    return [VIEW / 2 + east * s, VIEW / 2 - north * s];
  }

  function fromView(x, y) {
    var s = planScale();
    return [(x - VIEW / 2) / s, (VIEW / 2 - y) / s];
  }

  function svgNode(name, attributes) {
    var node = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.keys(attributes || {}).forEach(function (key) {
      node.setAttribute(key, attributes[key]);
    });
    return node;
  }

  function renderSitePlan() {
    var svg = $("site-plan");
    svg.setAttribute("viewBox", "0 0 " + VIEW + " " + VIEW);
    while (svg.firstChild) { svg.removeChild(svg.firstChild); }

    var extent = extentMetres();
    // Labelled interval first, then five fine divisions inside it, so the
    // sheet always carries at least one labelled line either side of zero.
    var major = niceStep(extent / 2);
    var minor = major / 5;
    var s = planScale();

    var grid = svgNode("g", {});
    for (var v = -Math.ceil(extent / minor) * minor; v <= extent; v += minor) {
      var isMajor = Math.abs(v % major) < 1e-6;
      var isAxis = Math.abs(v) < 1e-6;
      var cls = isAxis ? "grid-axis" : (isMajor ? "grid-major" : "grid-minor");
      var p = toView(v, v);
      grid.appendChild(svgNode("line", {
        x1: p[0], y1: SHEET_PAD - 6, x2: p[0], y2: VIEW - SHEET_PAD + 6, "class": cls
      }));
      grid.appendChild(svgNode("line", {
        x1: SHEET_PAD - 6, y1: p[1], x2: VIEW - SHEET_PAD + 6, y2: p[1], "class": cls
      }));
      if (isMajor && !isAxis) {
        var xLabel = svgNode("text", {
          x: p[0], y: VIEW - SHEET_PAD + 18, "class": "grid-label", "text-anchor": "middle"
        });
        xLabel.textContent = v;
        grid.appendChild(xLabel);
        var yLabel = svgNode("text", {
          x: SHEET_PAD - 10, y: p[1] + 3, "class": "grid-label", "text-anchor": "end"
        });
        yLabel.textContent = v;
        grid.appendChild(yLabel);
      }
    }
    svg.appendChild(grid);

    // Scale bar, bottom left: one major grid interval, labelled in metres.
    var barLength = major * s;
    var barY = VIEW - 12;
    var barX = SHEET_PAD;
    var bar = svgNode("g", {});
    bar.appendChild(svgNode("line", {
      x1: barX, y1: barY, x2: barX + barLength, y2: barY, "class": "scalebar-line"
    }));
    [0, barLength].forEach(function (offset) {
      bar.appendChild(svgNode("line", {
        x1: barX + offset, y1: barY - 4, x2: barX + offset, y2: barY + 2, "class": "scalebar-line"
      }));
    });
    var barText = svgNode("text", {
      x: barX + barLength + 6, y: barY + 3, "class": "scalebar-text"
    });
    barText.textContent = major + " m";
    bar.appendChild(barText);
    svg.appendChild(bar);

    // Compass rose, top right.
    var cx = VIEW - 26, cy = 26;
    var rose = svgNode("g", {});
    rose.appendChild(svgNode("circle", { cx: cx, cy: cy, r: 11, "class": "compass-ring" }));
    rose.appendChild(svgNode("line", {
      x1: cx, y1: cy + 8, x2: cx, y2: cy - 11, "class": "compass-needle"
    }));
    rose.appendChild(svgNode("line", {
      x1: cx, y1: cy, x2: cx + 8, y2: cy, "class": "compass-needle"
    }));
    var north = svgNode("text", {
      x: cx, y: cy - 14, "class": "compass-text", "text-anchor": "middle"
    });
    north.textContent = "N";
    rose.appendChild(north);
    var east = svgNode("text", { x: cx + 12, y: cy + 3.5, "class": "compass-text" });
    east.textContent = "E";
    rose.appendChild(east);
    svg.appendChild(rose);

    // Antennas: a theodolite cross inside a ring.
    state.antennas.forEach(function (antenna, index) {
      var p = toView(antenna[0], antenna[1]);
      var group = svgNode("g", {
        "class": "antenna" + (state.drag && state.drag.index === index ? " dragging" : ""),
        "data-index": index
      });
      group.appendChild(svgNode("circle", { cx: p[0], cy: p[1], r: 9, "class": "antenna-hit" }));
      group.appendChild(svgNode("line", {
        x1: p[0] - 7, y1: p[1], x2: p[0] + 7, y2: p[1], "class": "antenna-cross"
      }));
      group.appendChild(svgNode("line", {
        x1: p[0], y1: p[1] - 7, x2: p[0], y2: p[1] + 7, "class": "antenna-cross"
      }));
      group.appendChild(svgNode("circle", {
        cx: p[0], cy: p[1], r: 3.1, "class": "antenna-ring"
      }));
      var label = svgNode("text", { x: p[0] + 8, y: p[1] - 6, "class": "antenna-label" });
      label.textContent = index;
      group.appendChild(label);
      svg.appendChild(group);
    });

    $("sheet-empty").hidden = state.antennas.length > 0;
    renderArraySummary();
    renderAntennaTable();
  }

  function longestBaseline() {
    var longest = 0;
    for (var i = 0; i < state.antennas.length; i += 1) {
      for (var j = i + 1; j < state.antennas.length; j += 1) {
        var de = state.antennas[i][0] - state.antennas[j][0];
        var dn = state.antennas[i][1] - state.antennas[j][1];
        longest = Math.max(longest, Math.sqrt(de * de + dn * dn));
      }
    }
    return longest;
  }

  function renderArraySummary() {
    var n = state.antennas.length;
    var baselines = n * (n - 1) / 2;
    $("array-summary").textContent =
      n + " antennas make " + baselines + " baselines; the longest is "
      + fmt(longestBaseline(), 1) + " m.";
  }

  function renderAntennaTable() {
    var body = $("antenna-rows");
    while (body.firstChild) { body.removeChild(body.firstChild); }
    state.antennas.forEach(function (antenna, index) {
      var row = el("tr");
      row.appendChild(el("td", null, String(index)));
      row.appendChild(el("td", null, fmt(antenna[0], 1)));
      row.appendChild(el("td", null, fmt(antenna[1], 1)));
      var cell = el("td");
      var remove = el("button", "row-remove", "×");
      remove.type = "button";
      remove.title = "Remove antenna " + index;
      remove.setAttribute("aria-label", "Remove antenna " + index);
      remove.addEventListener("click", function () {
        state.antennas.splice(index, 1);
        renderSitePlan();
      });
      cell.appendChild(remove);
      row.appendChild(cell);
      body.appendChild(row);
    });
  }

  function planPoint(event) {
    var svg = $("site-plan");
    var rect = svg.getBoundingClientRect();
    var size = Math.min(rect.width, rect.height);
    var offsetX = rect.left + (rect.width - size) / 2;
    var offsetY = rect.top + (rect.height - size) / 2;
    return [
      (event.clientX - offsetX) / size * VIEW,
      (event.clientY - offsetY) / size * VIEW
    ];
  }

  function addAntenna(east, north) {
    var limit = state.defaults.limits.max_antennas;
    if (state.antennas.length >= limit) {
      showNotice("error", "That is the most antennas this front end runs ("
        + limit + "). Remove one before adding another.");
      return;
    }
    state.antennas.push([
      Math.round(east * 10) / 10,
      Math.round(north * 10) / 10,
      0
    ]);
    renderSitePlan();
  }

  function nearestAntennaGap(east, north) {
    var nearest = Infinity;
    state.antennas.forEach(function (antenna) {
      var de = antenna[0] - east;
      var dn = antenna[1] - north;
      nearest = Math.min(nearest, Math.sqrt(de * de + dn * dn));
    });
    return nearest;
  }

  // The discoverable route to a new antenna: drop one on free ground
  // somewhere inside the piece of site the plan already shows, then let
  // the user drag it. Math.random is honest here -- where a marker lands
  // is a convenience of the editor, not part of the simulated physics, and
  // nothing downstream is meant to be reproducible from it. Candidates
  // land at least MIN_SPACING_M from their neighbours where possible, both
  // so the two markers can be told apart and grabbed separately and
  // because coincident antennas make a zero-length baseline the library
  // warns about.
  function addAntennaAtRandom() {
    var extent = extentMetres() * 0.85;
    var best = [0, 0];
    var bestGap = -1;
    for (var attempt = 0; attempt < 60; attempt += 1) {
      var candidate = [
        (Math.random() * 2 - 1) * extent,
        (Math.random() * 2 - 1) * extent
      ];
      var gap = nearestAntennaGap(candidate[0], candidate[1]);
      if (gap > bestGap) { bestGap = gap; best = candidate; }
      if (gap >= MIN_SPACING_M) { break; }
    }
    addAntenna(best[0], best[1]);
  }

  /* --- known layouts -------------------------------------------------- */

  function markArrayLoaded() {
    state.loadedArray = JSON.stringify(state.antennas);
  }

  function arrayIsEdited() {
    return state.loadedArray !== null && JSON.stringify(state.antennas) !== state.loadedArray;
  }

  function renderArrayChoices() {
    var select = $("array-choice");
    while (select.firstChild) { select.removeChild(select.firstChild); }
    state.arrays.forEach(function (entry) {
      var text = entry.name + " — " + entry.n_antennas + " antennas";
      if (!entry.runnable) {
        text += " (more than this front end runs)";
      }
      var option = el("option", null, text);
      option.value = entry.id;
      select.appendChild(option);
    });
    var empty = state.arrays.length === 0;
    if (empty) { select.appendChild(el("option", null, "no layouts available")); }
    select.disabled = empty;
    $("load-array").disabled = empty;
  }

  // Loading a big array can put the recording settings past the size the
  // server accepts, so the number of integrations and then of channels is
  // halved until the product fits. Better to run something smaller at once
  // than to hand back a validation error for a button press.
  function fitRecordingToArray() {
    var limits = state.defaults.limits;
    var perBlock = state.antennas.length * state.defaults.sim.n_time_per_block;
    var changed = false;
    function total() { return perBlock * state.sim.n_chan * state.sim.n_blocks; }
    while (total() > limits.max_total_samples && state.sim.n_blocks > 1) {
      state.sim.n_blocks = Math.max(1, Math.floor(state.sim.n_blocks / 2));
      changed = true;
    }
    while (total() > limits.max_total_samples && state.sim.n_chan > 4) {
      state.sim.n_chan = Math.max(4, Math.floor(state.sim.n_chan / 2));
      changed = true;
    }
    return changed;
  }

  function showArrayNote(array, trimmed) {
    var note = $("array-load-note");
    var n = array.n_antennas;
    var baselines = n * (n - 1) / 2;
    var lines = [];
    if (n > LARGE_ARRAY) {
      lines.push("Loaded " + array.name + ": " + n + " antennas, " + baselines
        + " baselines. Baselines grow as the square of the antenna count, so this run"
        + " will take appreciably longer than the default array's — tens of seconds"
        + " rather than a few. Fewer channels or integrations in section 5 make it quick"
        + " again.");
    } else {
      lines.push("Loaded " + array.name + ": " + n + " antennas, " + baselines + " baselines.");
    }
    if (trimmed) {
      lines.push("Recording settings were reduced to " + state.sim.n_chan + " channels and "
        + state.sim.n_blocks + " integrations to keep the run within the server's size budget.");
    }
    if (!array.runnable) {
      lines.push("This layout has more antennas than this front end will run ("
        + state.defaults.limits.max_antennas + "); remove some before running.");
    }
    // A layout that costs real time, was trimmed to fit, or will not run
    // at all is a warning; anything else is plain information.
    var warn = n > LARGE_ARRAY || trimmed || !array.runnable;
    note.className = "banner" + (warn ? " banner-warn" : "");
    note.textContent = lines.join(" ");
    note.hidden = false;
  }

  function loadArray(id) {
    var entry = state.arrays.filter(function (candidate) { return candidate.id === id; })[0];
    if (!entry) { return; }
    if (arrayIsEdited()
        && !window.confirm("Loading " + entry.name + " replaces the antennas you have moved."
          + " Go ahead?")) {
      return;
    }
    request("/api/arrays/" + encodeURIComponent(id)).then(function (array) {
      state.antennas = array.antennas.map(function (row) { return row.slice(); });
      state.site = {
        name: array.name,
        latitude_deg: array.latitude_deg,
        longitude_deg: array.longitude_deg,
        height_m: array.height_m
      };
      markArrayLoaded();
      var trimmed = fitRecordingToArray();
      renderSitePlan();
      renderSiteMeta();
      renderSimFields();
      showArrayNote(array, trimmed);
      refreshPointing();
    }).catch(function (error) {
      showNotice("error", "Could not load that layout: " + error.message);
    });
  }

  function bindSitePlan() {
    var svg = $("site-plan");
    var readout = $("cursor-readout");

    svg.addEventListener("pointerdown", function (event) {
      var marker = event.target.closest(".antenna");
      var point = planPoint(event);
      if (marker) {
        state.drag = {
          index: Number(marker.getAttribute("data-index")),
          moved: false,
          pointerId: event.pointerId
        };
        svg.setPointerCapture(event.pointerId);
        renderSitePlan();
      } else {
        state.drag = { index: -1, moved: false, start: point, pointerId: event.pointerId };
      }
      event.preventDefault();
    });

    svg.addEventListener("pointermove", function (event) {
      var point = planPoint(event);
      var metres = fromView(point[0], point[1]);
      readout.textContent = "E " + fmt(metres[0], 1) + " m   N " + fmt(metres[1], 1) + " m";
      if (!state.drag) { return; }
      if (state.drag.index >= 0) {
        var extent = extentMetres();
        state.antennas[state.drag.index][0] =
          Math.round(clamp(metres[0], -extent, extent) * 10) / 10;
        state.antennas[state.drag.index][1] =
          Math.round(clamp(metres[1], -extent, extent) * 10) / 10;
        state.drag.moved = true;
        renderSitePlan();
      } else if (state.drag.start) {
        var dx = point[0] - state.drag.start[0];
        var dy = point[1] - state.drag.start[1];
        if (dx * dx + dy * dy > 9) { state.drag.moved = true; }
      }
    });

    svg.addEventListener("pointerup", function (event) {
      var drag = state.drag;
      state.drag = null;
      if (svg.hasPointerCapture && svg.hasPointerCapture(event.pointerId)) {
        svg.releasePointerCapture(event.pointerId);
      }
      if (drag && drag.index === -1 && !drag.moved) {
        var point = planPoint(event);
        var metres = fromView(point[0], point[1]);
        addAntenna(metres[0], metres[1]);
      } else {
        renderSitePlan();
      }
    });

    svg.addEventListener("pointerleave", function () {
      readout.textContent = "";
    });

    svg.addEventListener("dblclick", function (event) {
      var marker = event.target.closest(".antenna");
      if (!marker) { return; }
      state.antennas.splice(Number(marker.getAttribute("data-index")), 1);
      renderSitePlan();
    });

    // Keyboard route to the same action, so the plan is not mouse-only.
    // New markers walk outwards on a small spiral rather than piling up on
    // the origin, where duplicate positions would be flagged.
    svg.addEventListener("keydown", function (event) {
      if (event.key !== "a" && event.key !== "A") { return; }
      var n = state.antennas.length;
      var radius = 15 + 6 * n;
      addAntenna(radius * Math.cos(n * 2.4), radius * Math.sin(n * 2.4));
      event.preventDefault();
    });

    $("reset-array").addEventListener("click", function () {
      state.antennas = state.defaults.array.antennas.map(function (row) {
        return row.slice();
      });
      state.site = defaultSite();
      markArrayLoaded();
      $("array-load-note").hidden = true;
      renderSitePlan();
      renderSiteMeta();
      refreshPointing();
    });

    $("clear-array").addEventListener("click", function () {
      state.antennas = [];
      renderSitePlan();
    });
  }

  /* ----------------------------------------------------------- 5. forms */

  /* Front-end wording, keyed by source kind and then by the API's own
   * field name. The names are the contract with the server and are left
   * alone; only what a reader sees is rewritten here, in plainer words
   * and with one honest line about what the number does to the data.
   * Anything missing falls back to the descriptor's own label and its
   * `help` text (see simulate.py's field descriptors). */
  var FIELD_COPY = {
    sky: {
      flux_jy: {
        label: "Brightness",
        hint: "How bright the source is. The dirty image should peak at this value,"
          + " at the offsets above."
      }
    },
    line: {
      center_freq_hz: {
        label: "Line centre frequency",
        hint: "Rest frame. The default is the 21 cm hydrogen line at 1420.4 MHz, which"
          + " sits outside the default band — move it inside the recorded band to see it."
      },
      fwhm_hz: {
        label: "Line width",
        hint: "Full width at half maximum of the Gaussian line profile."
      },
      line_flux_jy: {
        label: "Peak-channel power",
        hint: "Added to every antenna's own power, like receiver noise, tapering as a"
          + " Gaussian in frequency."
      }
    },
    tower: {
      azimuth_deg: {
        label: "Bearing from the array",
        hint: "Compass bearing, north through east."
      },
      elevation_deg: {
        label: "Elevation above the horizon",
        hint: "Ground transmitters sit near 0°, so they enter through the far sidelobes."
      },
      distance_m: {
        label: "Distance from the array",
        hint: "Sets how much the delay to the transmitter differs between antennas."
      },
      center_freq_hz: {
        label: "Transmit frequency",
        hint: "Keep it inside the recorded band (printed under Recording settings) or"
          + " nothing will show."
      },
      bandwidth_hz: {
        label: "Bandwidth it occupies",
        hint: "How wide a slice of the band it fills; 200 kHz is a few channels."
      },
      received_power_jy: {
        label: "Received power",
        hint: "How loud this transmitter is at the array while it is on; the default is"
          + " far above the receiver noise and easy to see."
      },
      duty_cycle: {
        label: "Fraction of the time it transmits",
        hint: "0.5 gives the on/off stripes in the waterfall; 1 transmits continuously."
      },
      frame_duration_s: {
        label: "Length of one on/off frame",
        hint: "The period of the stripes in time."
      },
      waveform: {
        label: "Signal shape",
        hint: "Band-limited noise looks like a raised noise floor; a constant-envelope"
          + " carrier is what a spectral-kurtosis detector keys on."
      }
    },
    impulsive: {
      rate_hz: {
        label: "Bursts per second",
        hint: "Average rate; arrival times are random (Poisson) unless you change it below."
      },
      received_power_jy: {
        label: "Power of the faintest burst",
        hint: "Brighter bursts follow a power law up to the ratio set below."
      },
      azimuth_deg: { label: "Bearing from the array", hint: "Compass bearing, north through east." },
      elevation_deg: {
        label: "Elevation above the horizon",
        hint: "Sparking hardware is usually on the ground, so near 0°."
      },
      distance_m: {
        label: "Distance from the array",
        hint: "Sets how much the delay differs between antennas."
      },
      power_law_index: {
        label: "Brightness power-law index",
        hint: "Larger means proportionally more faint bursts and fewer bright ones."
      },
      max_power_ratio: {
        label: "Brightest burst / faintest burst",
        hint: "The dynamic range of the burst population."
      },
      pulse_width_samples: {
        label: "Burst length",
        hint: "In voltage samples; 1 is a single spike smeared across the whole band."
      }
    },
    satellite: {
      tle_source: {
        label: "Where the orbit comes from",
        hint: "Nothing is fetched from the network: use the bundled object or paste"
          + " your own element set."
      },
      tle_text: {
        label: "Pasted element set",
        hint: "Two 69-character lines, optionally preceded by a name line."
      },
      carrier_freq_hz: {
        label: "Carrier frequency",
        hint: "As transmitted. What arrives is Doppler shifted by the satellite's"
          + " motion, which is what draws the drifting track."
      },
      received_power_jy: {
        label: "Received power",
        hint: "Power at the array while the satellite is above the horizon cut."
      },
      sideband_bandwidth_hz: {
        label: "Width of the sidebands",
        hint: "Noise-like power either side of the carrier."
      },
      sideband_power_fraction: {
        label: "Share of the power in the sidebands",
        hint: "0 is a pure carrier line; 1 puts all of it into the sidebands."
      },
      min_elevation_deg: {
        label: "Horizon cut",
        hint: "Below this elevation the satellite counts as set and transmits nothing."
      },
      apply_doppler: {
        label: "Apply the Doppler shift",
        hint: "Turn it off to hold the carrier at one frequency and see what the drift"
          + " was worth."
      }
    },
    aircraft: {
      east_m: {
        label: "Starting position, east",
        hint: "Where the aircraft is at the first sample, relative to the array."
      },
      north_m: { label: "Starting position, north", hint: "Same, towards north." },
      altitude_m: {
        label: "Altitude",
        hint: "High enough and it is well above the horizon, unlike ground transmitters."
      },
      velocity_east_m_s: {
        label: "Ground speed, east",
        hint: "The course is a straight line; the delays move from block to block as"
          + " it flies."
      },
      velocity_north_m_s: { label: "Ground speed, north", hint: "Together these set the heading." },
      carrier_freq_hz: {
        label: "Transponder frequency",
        hint: "Keep it inside the recorded band to see the bursts."
      },
      bandwidth_hz: { label: "Width of each burst", hint: "How wide a slice of the band a reply fills." },
      received_power_jy: {
        label: "Received power",
        hint: "Transponders are loud: the default sits far above the noise."
      },
      message_rate_hz: {
        label: "Replies per second",
        hint: "Each reply is one short burst."
      },
      pulse_width_samples: { label: "Burst length", hint: "In voltage samples." },
      min_elevation_deg: {
        label: "Horizon cut",
        hint: "Below this elevation the aircraft is out of sight and transmits nothing."
      }
    },
    comb: {
      azimuth_deg: { label: "Bearing from the array", hint: "Compass bearing, north through east." },
      elevation_deg: {
        label: "Elevation above the horizon",
        hint: "Interfering hardware is usually low on the horizon."
      },
      distance_m: {
        label: "Distance from the array",
        hint: "Sets how much the delay differs between antennas."
      },
      fundamental_hz: {
        label: "Fundamental frequency",
        hint: "The device's base frequency. It may sit far below the band; only its"
          + " in-band harmonics show up."
      },
      harmonic_numbers: {
        label: "Which harmonics it emits",
        hint: "Multiples of the fundamental, comma separated — e.g. 999,1000,1001 makes"
          + " three lines a fundamental apart."
      },
      received_power_jy: {
        label: "Received power per harmonic",
        hint: "Every listed harmonic arrives with this much power."
      },
      bandwidth_hz: {
        label: "Width of each harmonic",
        hint: "0 makes every harmonic a pure line, one channel wide."
      },
      duty_cycle: {
        label: "Fraction of the time it emits",
        hint: "1 is continuous; less than 1 chops the comb into frames."
      },
      frame_duration_s: { label: "Length of one on/off frame", hint: "The period of the chopping." }
    }
  };

  // One field descriptor -> one labelled control. `values` is the object
  // the control writes back into, in API units; the descriptor's `factor`
  // is what the person sees divided by.
  //
  // `copy` is an optional {name: {label, hint}} map of front-end wording
  // that overrides the descriptor's own: the server's field names are the
  // API contract and never change, but what a reader is shown may say the
  // same thing in plainer words, with one line underneath saying what the
  // number does to the data. See FIELD_COPY.
  function buildField(descriptor, values, onChange, copy) {
    var words = (copy && copy[descriptor.name]) || {};
    var isToggle = descriptor.kind === "toggle";
    var wide = (descriptor.kind === "text" && descriptor.multiline) || isToggle;
    var wrap = el("div", "field" + (wide ? " field-wide" : "")
      + (isToggle ? " field-toggle" : ""));
    var id = "f" + Math.random().toString(36).slice(2, 9);
    var label = el("label", isToggle ? "toggle-label" : "field-label");
    label.htmlFor = id;
    var labelText = el("span", null, words.label || descriptor.label);
    if (descriptor.unit) {
      labelText.appendChild(document.createTextNode(" "));
      labelText.appendChild(el("span", "field-unit", "(" + descriptor.unit + ")"));
    }

    var input;
    if (descriptor.kind === "choice") {
      input = el("select", "input");
      descriptor.options.forEach(function (option) {
        var node = el("option", null, option.label);
        node.value = option.value;
        input.appendChild(node);
      });
      input.value = values[descriptor.name];
    } else if (descriptor.kind === "toggle") {
      input = el("input", "");
      input.type = "checkbox";
      input.checked = Boolean(values[descriptor.name]);
    } else if (descriptor.kind === "text") {
      input = el(descriptor.multiline ? "textarea" : "input", "input");
      if (descriptor.placeholder) { input.placeholder = descriptor.placeholder; }
      input.value = values[descriptor.name] || "";
    } else {
      input = el("input", "input");
      input.type = "number";
      var factor = descriptor.factor || 1;
      if (descriptor.step !== undefined) { input.step = descriptor.step; }
      if (descriptor.min !== undefined) { input.min = descriptor.min / factor; }
      if (descriptor.max !== undefined) { input.max = descriptor.max / factor; }
      input.value = String(values[descriptor.name] / factor);
    }
    input.id = id;
    if (descriptor.help) { input.title = descriptor.help; }

    input.addEventListener("change", function () {
      if (descriptor.kind === "toggle") {
        values[descriptor.name] = input.checked;
      } else if (descriptor.kind === "number") {
        var parsed = parseFloat(input.value);
        values[descriptor.name] = isFinite(parsed)
          ? parsed * (descriptor.factor || 1)
          : descriptor.default;
      } else {
        values[descriptor.name] = input.value;
      }
      if (onChange) { onChange(); }
    });

    // A checkbox reads as a sentence with the box in front of it; every
    // other control reads as a caption above the box.
    if (isToggle) {
      label.appendChild(input);
      label.appendChild(labelText);
      wrap.appendChild(label);
    } else {
      label.appendChild(labelText);
      wrap.appendChild(label);
      wrap.appendChild(input);
    }

    var hint = words.hint || descriptor.help;
    if (hint) { wrap.appendChild(el("p", "field-hint", hint)); }
    return wrap;
  }

  /* --- where a sky source sits ----------------------------------------
   *
   * The same position written three ways: degrees east/north of the
   * pointing, absolute right ascension and declination, or the library's
   * direction cosines. The request carries whichever the user chose and
   * the server does the authoritative conversion (see `SkySource` in
   * simulate.py); everything below exists only so that flipping the unit
   * switcher shows the same place in the new units, and so the page can
   * warn before a run that a source is off the edge of the image. It is
   * the same projection the library uses (`rfi_simulator.sky`), written
   * out here for display and nothing else.
   */
  function rad(degrees) { return degrees * Math.PI / 180; }
  function deg(radians) { return radians * 180 / Math.PI; }

  function lmFromRadec(ra0, dec0, ra, dec) {
    var delta = rad(ra - ra0);
    return [
      Math.cos(rad(dec)) * Math.sin(delta),
      Math.sin(rad(dec)) * Math.cos(rad(dec0))
        - Math.cos(rad(dec)) * Math.sin(rad(dec0)) * Math.cos(delta)
    ];
  }

  function radecFromLm(ra0, dec0, l, m) {
    var n = Math.sqrt(Math.max(0, 1 - l * l - m * m));
    var dec = Math.asin(clamp(m * Math.cos(rad(dec0)) + n * Math.sin(rad(dec0)), -1, 1));
    var ra = rad(ra0) + Math.atan2(l, n * Math.cos(rad(dec0)) - m * Math.sin(rad(dec0)));
    return [((deg(ra) % 360) + 360) % 360, deg(dec)];
  }

  function sourceLm(source) {
    if (source.mode === "lm") { return [source.l, source.m]; }
    if (source.mode === "radec") {
      if (!state.pointing) { return [0, 0]; }
      return lmFromRadec(state.pointing.ra_deg, state.pointing.dec_deg,
        source.ra_deg, source.dec_deg);
    }
    return [Math.sin(rad(source.east_deg)), Math.sin(rad(source.north_deg))];
  }

  function setSourceMode(source, mode) {
    var lm = sourceLm(source);
    if (mode === "lm") {
      source.l = Math.round(lm[0] * 1e6) / 1e6;
      source.m = Math.round(lm[1] * 1e6) / 1e6;
    } else if (mode === "radec") {
      var radec = state.pointing
        ? radecFromLm(state.pointing.ra_deg, state.pointing.dec_deg, lm[0], lm[1])
        : [0, 0];
      source.ra_deg = Math.round(radec[0] * 1e5) / 1e5;
      source.dec_deg = Math.round(radec[1] * 1e5) / 1e5;
    } else {
      source.east_deg = Math.round(deg(Math.asin(clamp(lm[0], -1, 1))) * 1e5) / 1e5;
      source.north_deg = Math.round(deg(Math.asin(clamp(lm[1], -1, 1))) * 1e5) / 1e5;
    }
    source.mode = mode;
  }

  function positionMode(value) {
    return state.defaults.sky_source.position.modes.filter(function (mode) {
      return mode.value === value;
    })[0];
  }

  // One line under the position inputs: the same place in the other two
  // notations, plus an amber word of warning when it falls outside the
  // piece of sky the dirty image covers.
  function describePosition(source, note) {
    var lm = sourceLm(source);
    var eastDeg = deg(Math.asin(clamp(lm[0], -1, 1)));
    var northDeg = deg(Math.asin(clamp(lm[1], -1, 1)));
    var radec = state.pointing
      ? radecFromLm(state.pointing.ra_deg, state.pointing.dec_deg, lm[0], lm[1])
      : null;
    var parts = [
      fmt(eastDeg, 3) + "° E, " + fmt(northDeg, 3) + "° N of the pointing",
      "l " + lm[0].toFixed(5) + ", m " + lm[1].toFixed(5)
    ];
    if (radec) {
      parts.push("RA " + fmt(radec[0], 4) + "°, Dec " + fmt(radec[1], 4) + "°");
    }
    note.textContent = parts.join("   ·   ");

    var half = state.pointing ? state.pointing.field_half_width_deg : Infinity;
    var outside = Math.max(Math.abs(eastDeg), Math.abs(northDeg)) > half;
    note.className = "position-note mono" + (outside ? " position-warn" : "");
    if (outside) {
      note.textContent = "This sits outside the ±" + fmt(half, 1) + "° imaged field."
        + " The simulator still records it, and it still lands in the voltages — it"
        + " simply will not appear in the dirty image.   ·   " + note.textContent;
    }
  }

  function positionBlock(source) {
    var block = el("div", "position-block");
    var modeField = el("div", "field");
    var modeLabel = el("label", "field-label", "Position given as");
    var modeId = "p" + Math.random().toString(36).slice(2, 9);
    modeLabel.htmlFor = modeId;
    var select = el("select", "input");
    select.id = modeId;
    state.defaults.sky_source.position.modes.forEach(function (mode) {
      var option = el("option", null, mode.label);
      option.value = mode.value;
      select.appendChild(option);
    });
    select.value = source.mode;
    modeField.appendChild(modeLabel);
    modeField.appendChild(select);
    block.appendChild(modeField);

    var inputs = el("div", "position-inputs");
    var note = el("p", "position-note mono");
    var mode = positionMode(source.mode);

    mode.fields.forEach(function (key, which) {
      var field = el("div", "field");
      var id = "p" + Math.random().toString(36).slice(2, 9);
      var label = el("label", "field-label");
      label.htmlFor = id;
      var text = el("span", null, mode.labels[which]);
      text.appendChild(document.createTextNode(" "));
      text.appendChild(el("span", "field-unit", "(" + mode.unit + ")"));
      label.appendChild(text);
      var input = el("input", "input");
      input.type = "number";
      input.id = id;
      input.step = mode.step;
      input.value = source[key];
      input.addEventListener("input", function () {
        var parsed = parseFloat(input.value);
        if (isFinite(parsed)) {
          source[key] = parsed;
          describePosition(source, note);
        }
      });
      field.appendChild(label);
      field.appendChild(input);
      inputs.appendChild(field);
    });

    select.addEventListener("change", function () {
      setSourceMode(source, select.value);
      renderSkyCards();
    });

    block.appendChild(inputs);
    describePosition(source, note);
    block.appendChild(note);
    return block;
  }

  function renderSkyCards() {
    var host = $("sky-cards");
    while (host.firstChild) { host.removeChild(host.firstChild); }
    $("sky-empty").hidden = state.skySources.length > 0;

    state.skySources.forEach(function (source, index) {
      var card = el("div", "card");
      var head = el("div", "card-head");
      var swatch = el("span", "card-swatch");
      swatch.style.background = SKY_MARKER;
      head.appendChild(swatch);
      head.appendChild(el("span", "card-kind", "Sky source"));
      var name = el("input", "input card-name");
      name.value = source.name;
      name.setAttribute("aria-label", "Sky source name");
      name.addEventListener("change", function () { source.name = name.value; });
      head.appendChild(name);
      var remove = el("button", "card-remove", "×");
      remove.type = "button";
      remove.setAttribute("aria-label", "Remove this sky source");
      remove.addEventListener("click", function () {
        state.skySources.splice(index, 1);
        renderSkyCards();
      });
      head.appendChild(remove);
      card.appendChild(head);

      var body = el("div", "card-body");
      body.appendChild(positionBlock(source));
      var grid = el("div", "field-grid");
      state.defaults.sky_source.fields.forEach(function (descriptor) {
        grid.appendChild(buildField(descriptor, source, null, FIELD_COPY.sky));
      });
      body.appendChild(grid);
      card.appendChild(body);
      host.appendChild(card);
    });
  }

  function rfiType(typeName) {
    return state.defaults.rfi_types.filter(function (entry) {
      return entry.type === typeName;
    })[0];
  }

  /* --- the interference picker ----------------------------------------
   *
   * The server's own labels for the source types are precise but written
   * for someone who already knows the field ("Ground transmitter",
   * "Impulsive broadband"). What a first-time reader is offered instead
   * is one tick box per kind, named after the thing in the world and
   * described in a line: ticking one adds a source with the schema's own
   * defaults, and everything that kind can be tuned by is one "More
   * details" click away.
   *
   * `keys` names the two or three numbers worth printing on the compact
   * card -- frequency, power, and whatever third quantity that kind is
   * really about. They are field names of that type's own schema, so the
   * card reads its values and units straight out of the descriptors.
   *
   * Kinds the server offers but this list does not name still appear,
   * described by the server's own label and summary; nothing here is
   * required for a type to work.
   */
  var RFI_KINDS = [
    { type: "tower", title: "Cell tower",
      blurb: "a narrowband transmitter on the horizon",
      keys: ["center_freq_hz", "received_power_jy", "duty_cycle"] },
    { type: "satellite", title: "Satellite",
      blurb: "a moving transmitter crossing the sky (real orbit)",
      keys: ["carrier_freq_hz", "received_power_jy", "min_elevation_deg"] },
    { type: "aircraft", title: "Aircraft",
      blurb: "ADS-B transponder flying over",
      keys: ["carrier_freq_hz", "received_power_jy", "message_rate_hz"] },
    { type: "impulsive", title: "Broadband bursts",
      blurb: "short wideband crackles (sparking, radar-like)",
      keys: ["rate_hz", "received_power_jy", "pulse_width_samples"] },
    { type: "comb", title: "Harmonic comb",
      blurb: "one device polluting many frequencies at once",
      keys: ["fundamental_hz", "received_power_jy", "duty_cycle"] }
  ];

  function rfiKind(typeName) {
    var known = RFI_KINDS.filter(function (kind) { return kind.type === typeName; })[0];
    if (known) { return known; }
    var entry = rfiType(typeName);
    return { type: typeName, title: entry.label, blurb: entry.summary, keys: [] };
  }

  // Every kind the server offers, the named ones first and in the order
  // a newcomer meets them rather than the schema's.
  function rfiKinds() {
    var named = RFI_KINDS.filter(function (kind) { return Boolean(rfiType(kind.type)); });
    var namedTypes = named.map(function (kind) { return kind.type; });
    var rest = state.defaults.rfi_types.filter(function (entry) {
      return namedTypes.indexOf(entry.type) === -1;
    }).map(function (entry) { return rfiKind(entry.type); });
    return named.concat(rest);
  }

  function countOfKind(typeName) {
    return state.rfiSources.filter(function (source) {
      return source.type === typeName;
    }).length;
  }

  function addRfiSource(typeName) {
    if (state.rfiSources.length >= state.defaults.limits.max_rfi_sources) {
      showNotice("error", "That is the most interference sources one run takes ("
        + state.defaults.limits.max_rfi_sources + ").");
      return false;
    }
    state.rfiSources.push(newRfiSource(typeName));
    state.rfiOpen.push(false);
    return true;
  }

  // "Edited" means anything about the source differs from what ticking
  // the box would have made, its name aside -- that is what is worth
  // asking about before it is thrown away.
  function rfiSourceIsEdited(source) {
    var fresh = newRfiSource(source.type);
    return Object.keys(fresh).some(function (key) {
      if (key === "name") { return false; }
      return JSON.stringify(source[key]) !== JSON.stringify(fresh[key]);
    });
  }

  function removeRfiKind(typeName) {
    var kind = rfiKind(typeName);
    var edited = state.rfiSources.some(function (source) {
      return source.type === typeName && rfiSourceIsEdited(source);
    });
    if (edited && !window.confirm("Remove the " + kind.title.toLowerCase()
        + " you have changed? The settings you typed will be lost.")) {
      return false;
    }
    for (var i = state.rfiSources.length - 1; i >= 0; i -= 1) {
      if (state.rfiSources[i].type === typeName) {
        state.rfiSources.splice(i, 1);
        state.rfiOpen.splice(i, 1);
      }
    }
    return true;
  }

  function renderRfiKinds() {
    var host = $("rfi-kinds");
    while (host.firstChild) { host.removeChild(host.firstChild); }

    rfiKinds().forEach(function (kind) {
      var count = countOfKind(kind.type);
      var card = el("label", "kind-card" + (count ? " kind-on" : ""));
      var box = el("input", "kind-box");
      box.type = "checkbox";
      box.checked = count > 0;

      var text = el("span", "kind-text");
      var title = el("span", "kind-title");
      title.appendChild(document.createTextNode(kind.title));
      // A "1" badge on a single source says nothing the tick does not;
      // the count earns its place once there is more than one.
      if (count > 1) { title.appendChild(el("span", "kind-count", "×" + count)); }
      text.appendChild(title);
      text.appendChild(el("span", "kind-blurb", kind.blurb));

      card.appendChild(box);
      card.appendChild(text);
      box.addEventListener("change", function () {
        if (box.checked) { addRfiSource(kind.type); } else { removeRfiKind(kind.type); }
        renderRfi();
      });
      host.appendChild(card);
    });
  }

  // Numbers on a compact card are printed, not editable: the same field
  // has a control inside the fold below, and two live inputs on one
  // value would disagree the moment either was typed into.
  function showNumber(value) {
    if (typeof value !== "number") { return String(value); }
    if (!isFinite(value)) { return "--"; }
    return String(Math.round(value * 1e6) / 1e6);
  }

  function fillKeyNumbers(source, host) {
    while (host.firstChild) { host.removeChild(host.firstChild); }
    var descriptors = rfiType(source.type).fields;
    var copy = FIELD_COPY[source.type] || {};
    rfiKind(source.type).keys.forEach(function (name) {
      var descriptor = descriptors.filter(function (entry) {
        return entry.name === name;
      })[0];
      if (!descriptor) { return; }
      var words = copy[name] || {};
      var shown = descriptor.kind === "number"
        ? showNumber(source[name] / (descriptor.factor || 1))
        : String(source[name]);
      host.appendChild(el("dt", null, words.label || descriptor.label));
      host.appendChild(el("dd", null, shown + (descriptor.unit ? " " + descriptor.unit : "")));
    });
  }

  // A spectral line is ground truth labelled "celestial", not "rfi" (see
  // rfi_simulator.sky.SpectralLineForeground); its card follows the same
  // pattern as a sky source's, just with its own field list and colour.
  function renderLineCards() {
    var host = $("line-cards");
    while (host.firstChild) { host.removeChild(host.firstChild); }
    $("line-empty").hidden = state.spectralLines.length > 0;

    state.spectralLines.forEach(function (source, index) {
      var card = el("div", "card");
      var head = el("div", "card-head");
      var swatch = el("span", "card-swatch");
      swatch.style.background = SKY_MARKER;
      head.appendChild(swatch);
      head.appendChild(el("span", "card-kind", "Spectral line"));
      var name = el("input", "input card-name");
      name.value = source.name;
      name.setAttribute("aria-label", "Spectral line name");
      name.addEventListener("change", function () { source.name = name.value; });
      head.appendChild(name);
      var remove = el("button", "card-remove", "×");
      remove.type = "button";
      remove.setAttribute("aria-label", "Remove this spectral line");
      remove.addEventListener("click", function () {
        state.spectralLines.splice(index, 1);
        renderLineCards();
      });
      head.appendChild(remove);
      card.appendChild(head);

      var body = el("div", "card-body");
      var grid = el("div", "field-grid");
      state.defaults.spectral_line.fields.forEach(function (descriptor) {
        grid.appendChild(buildField(descriptor, source, null, FIELD_COPY.line));
      });
      body.appendChild(grid);
      card.appendChild(body);
      host.appendChild(card);
    });
  }

  function newSpectralLine() {
    var values = Object.assign({}, state.defaults.spectral_line.defaults);
    values.name = "hi_line " + (state.spectralLines.length + 1);
    return values;
  }

  // A handful of nested-object request fields (`coupling`, `polarization`,
  // `envelope`, `arrival`) have no scalar field of their own to describe
  // with a `buildField` descriptor -- see simulate.py's comment on
  // `_WAVEFORM_FIELD`. This builds a small row of hand-written controls
  // for them instead, writing straight into the source object; `run()`'s
  // request serialization already forwards every property a source has,
  // so nothing else has to change to send what this writes.
  function extraNumber(value, step, min, max, title) {
    var input = el("input", "input input-small");
    input.type = "number";
    input.step = step;
    input.min = min;
    input.max = max;
    input.title = title;
    input.value = value;
    return input;
  }

  function extraToggle(text, checked) {
    var label = el("label", "checkline");
    var box = el("input");
    box.type = "checkbox";
    box.checked = checked;
    label.appendChild(box);
    label.appendChild(document.createTextNode(" " + text));
    return { label: label, box: box };
  }

  // One labelled block inside the card's Advanced fold: a checkbox line,
  // the numbers it reveals, and a sentence saying what it changes.
  function extraBlock(row, toggle, inputs, hint) {
    var block = el("div", "extra");
    block.appendChild(toggle.label);
    var numbers = el("div", "extra-numbers");
    inputs.forEach(function (entry) {
      var field = el("label", "extra-field");
      field.appendChild(el("span", "extra-field-label", entry[0]));
      field.appendChild(entry[1]);
      numbers.appendChild(field);
    });
    block.appendChild(numbers);
    block.appendChild(el("p", "field-hint", hint));
    row.appendChild(block);
    return numbers;
  }

  function buildRfiExtras(source) {
    var fold = el("details", "advanced advanced-card");
    var summary = el("summary", null, "Advanced: how it couples into the antennas");
    fold.appendChild(summary);
    var row = el("div", "card-extras");
    fold.appendChild(row);

    var coupling = extraToggle(
      "Every antenna receives it a little differently", Boolean(source.coupling)
    );
    var couplingSigma = extraNumber(
      source.coupling ? source.coupling.sigma_db : 3.0, 0.5, 0, 60,
      "Per-antenna lognormal coupling scatter, dB"
    );
    var couplingNumbers = extraBlock(
      row, coupling, [["Spread between antennas (dB)", couplingSigma]],
      "Off, the transmitter arrives at exactly the same strength everywhere. On, each"
        + " antenna's gain towards it is drawn from a lognormal spread — nearer masts"
        + " and local screening do this."
    );
    couplingNumbers.hidden = !source.coupling;
    function syncCoupling() {
      source.coupling = coupling.box.checked
        ? { type: "lognormal", sigma_db: parseFloat(couplingSigma.value) || 0, seed: Math.round(state.sim.seed) }
        : null;
      couplingNumbers.hidden = !coupling.box.checked;
    }
    coupling.box.addEventListener("change", syncCoupling);
    couplingSigma.addEventListener("change", syncCoupling);

    var polarized = extraToggle("The signal is polarized", Boolean(source.polarization));
    var polAngle = extraNumber(
      source.polarization ? source.polarization.angle_deg : 45.0, 1, -360, 360,
      "Linear polarization angle, degrees, first receptor towards the second"
    );
    var polNumbers = extraBlock(
      row, polarized, [["Angle (deg)", polAngle]],
      "Splits the power between the two receptors instead of sharing it evenly."
        + " Only visible with two polarizations recorded (section 4); 0° goes into the"
        + " first receptor, 90° into the second."
    );
    polNumbers.hidden = !source.polarization;
    function syncPolarization() {
      source.polarization = polarized.box.checked
        ? { type: "linear", angle_deg: parseFloat(polAngle.value) || 0 }
        : null;
      polNumbers.hidden = !polarized.box.checked;
    }
    polarized.box.addEventListener("change", syncPolarization);
    polAngle.addEventListener("change", syncPolarization);

    if (source.type === "tower" || source.type === "comb") {
      var envelope = extraToggle(
        "Add a second, slower on/off cycle", Boolean(source.envelope)
      );
      var envPeriod = extraNumber(
        source.envelope ? source.envelope.period_s * 1000 : 20.0, 1, 0.1, 1e5, "Envelope period, ms"
      );
      var envDuty = extraNumber(
        source.envelope ? source.envelope.duty : 0.5, 0.05, 0, 1, "Envelope duty (on-fraction)"
      );
      var envNumbers = extraBlock(
        row, envelope,
        [["Cycle length (ms)", envPeriod], ["Fraction on", envDuty]],
        "On top of the frame pattern above: a long duty cycle over the short one, as a"
          + " transmitter that bursts in scheduled slots would give."
      );
      envNumbers.hidden = !source.envelope;
      function syncEnvelope() {
        source.envelope = envelope.box.checked
          ? {
            type: "periodic",
            period_s: (parseFloat(envPeriod.value) || 20.0) / 1000,
            duty: parseFloat(envDuty.value) || 0.5
          }
          : null;
        envNumbers.hidden = !envelope.box.checked;
      }
      envelope.box.addEventListener("change", syncEnvelope);
      envPeriod.addEventListener("change", syncEnvelope);
      envDuty.addEventListener("change", syncEnvelope);
    }

    if (source.type === "impulsive") {
      var periodic = typeof source.arrival === "object" && source.arrival !== null;
      var arrival = extraToggle("Bursts arrive on a clock, not at random", periodic);
      var arrRate = extraNumber(
        periodic ? source.arrival.rate_hz : 100.0, 10, 0, 1e5, "Arrival rate, events/s"
      );
      var arrJitter = extraNumber(
        periodic ? source.arrival.jitter_s * 1000 : 0.5, 0.1, 0, 1e4, "Arrival jitter, ms"
      );
      var arrNumbers = extraBlock(
        row, arrival,
        [["Bursts per second", arrRate], ["Wobble on each arrival (ms)", arrJitter]],
        "Off, arrivals are Poisson — genuinely random gaps. On, they are evenly spaced"
          + " with a little jitter, like mains-synchronised arcing."
      );
      arrNumbers.hidden = !periodic;
      function syncArrival() {
        source.arrival = arrival.box.checked
          ? {
            type: "periodic",
            rate_hz: parseFloat(arrRate.value) || 100.0,
            jitter_s: (parseFloat(arrJitter.value) || 0) / 1000
          }
          : "poisson";
        arrNumbers.hidden = !arrival.box.checked;
      }
      arrival.box.addEventListener("change", syncArrival);
      arrRate.addEventListener("change", syncArrival);
      arrJitter.addEventListener("change", syncArrival);
    }

    return fold;
  }

  /* One card per interference source: a compact face with the two or
   * three numbers that kind is about, and the whole schema-driven form
   * -- every field, plus the Advanced coupling and polarization fold --
   * behind "More details". Which folds are open is remembered in
   * `state.rfiOpen`, so re-rendering does not shut a card the user had
   * opened. Nothing about what is sent changes here: every control still
   * writes into the same source object, under the same field name, that
   * `buildRequest` serializes. */
  function renderRfiCards() {
    var host = $("rfi-cards");
    while (host.firstChild) { host.removeChild(host.firstChild); }
    // Presets and resets replace the source list wholesale, so the fold
    // state is squared up against it here rather than at every writer.
    while (state.rfiOpen.length < state.rfiSources.length) { state.rfiOpen.push(false); }
    state.rfiOpen.length = state.rfiSources.length;
    $("rfi-empty").hidden = state.rfiSources.length > 0;

    state.rfiSources.forEach(function (source, index) {
      var descriptorSet = rfiType(source.type);
      var kind = rfiKind(source.type);
      var card = el("div", "card");
      var head = el("div", "card-head");
      var swatch = el("span", "card-swatch");
      swatch.style.background = "rgba(" + MASK_RGB + ", 1)";
      head.appendChild(swatch);
      head.appendChild(el("span", "card-kind", kind.title));
      var name = el("input", "input card-name");
      name.value = source.name;
      name.setAttribute("aria-label", "Interference source name");
      name.addEventListener("change", function () {
        source.name = name.value;
      });
      head.appendChild(name);
      var remove = el("button", "card-remove", "×");
      remove.type = "button";
      remove.setAttribute("aria-label", "Remove this interference source");
      remove.addEventListener("click", function () {
        state.rfiSources.splice(index, 1);
        state.rfiOpen.splice(index, 1);
        renderRfi();
      });
      head.appendChild(remove);
      card.appendChild(head);

      var body = el("div", "card-body");
      body.appendChild(el("p", "card-intro", descriptorSet.summary));
      var numbers = el("dl", "card-summary mono");
      fillKeyNumbers(source, numbers);
      body.appendChild(numbers);
      card.appendChild(body);

      var fold = el("details", "card-details");
      fold.open = Boolean(state.rfiOpen[index]);
      fold.appendChild(el("summary", null, "More details"));
      fold.addEventListener("toggle", function () {
        state.rfiOpen[index] = fold.open;
      });

      var detail = el("div", "card-body");
      var grid = el("div", "field-grid");
      descriptorSet.fields.forEach(function (descriptor) {
        // The pasted element set only matters when it is going to be used.
        if (descriptor.name === "tle_text" && source.tle_source !== "custom") { return; }
        grid.appendChild(buildField(descriptor, source, function () {
          if (descriptor.name === "tle_source") { renderRfi(); return; }
          fillKeyNumbers(source, numbers);
        }, FIELD_COPY[source.type]));
      });
      detail.appendChild(grid);
      fold.appendChild(detail);
      fold.appendChild(buildRfiExtras(source));

      var foot = el("div", "card-foot");
      var another = el("button", "button button-small",
        "+ Add another " + kind.title.toLowerCase());
      another.type = "button";
      another.addEventListener("click", function () {
        addRfiSource(source.type);
        renderRfi();
      });
      foot.appendChild(another);
      fold.appendChild(foot);

      card.appendChild(fold);
      host.appendChild(card);
    });
  }

  function renderRfi() {
    renderRfiKinds();
    renderRfiCards();
  }

  function renderSimFields() {
    var host = $("sim-fields");
    while (host.firstChild) { host.removeChild(host.firstChild); }
    var limits = state.defaults.limits;
    var sim = state.defaults.sim;
    var descriptors = [
      {
        name: "n_chan", label: "Frequency channels", kind: "number", factor: 1,
        min: 4, max: limits.max_n_chan, step: 4, default: sim.n_chan,
        help: "Channels are a fixed " + fmt(sim.chan_width_hz / 1e3, 2)
          + " kHz wide, so this sets how much bandwidth is recorded."
      },
      {
        name: "n_blocks", label: "Time blocks", kind: "number", factor: 1,
        min: 1, max: limits.max_n_blocks, step: 1, default: sim.n_blocks,
        help: "Each block is " + fmt(sim.block_duration_s * 1000, 1)
          + " ms long, so this sets how long the observation lasts."
      },
      {
        name: "center_freq_hz", label: "Centre of the band", kind: "number", unit: "MHz",
        factor: 1e6, min: 1e6, max: 1e11, step: 0.1,
        default: sim.center_freq_hz,
        help: "Where the recorded band sits. Sources outside the band shown below"
          + " will not appear."
      },
      {
        name: "noise_std", label: "Receiver noise level", kind: "number", unit: "√Jy",
        factor: 1, min: 0, max: 1e4, step: 0.1, default: sim.noise_std,
        help: "Its square is the noise power added to every antenna. Leave it at 1 and"
          + " read every source power as a multiple of it."
      },
      {
        name: "seed", label: "Random seed", kind: "number", factor: 1,
        min: 0, max: 2147483647, step: 1, default: sim.seed,
        help: "The same seed reproduces byte-identical data; change it for a fresh"
          + " noise realisation."
      }
    ];
    descriptors.forEach(function (descriptor) {
      host.appendChild(buildField(descriptor, state.sim, renderSimSummary));
    });
    renderSimSummary();
  }

  function renderSimSummary() {
    var sim = state.defaults.sim;
    var duration = state.sim.n_blocks * sim.block_duration_s;
    var bandwidth = state.sim.n_chan * sim.chan_width_hz;
    var half = bandwidth / 2;
    fillMetricTable($("sim-summary"), ["this run records", ""], [
      ["duration", fmt(duration * 1000, 1) + " ms"],
      ["band", fmt((state.sim.center_freq_hz - half) / 1e6, 3) + " – "
        + fmt((state.sim.center_freq_hz + half) / 1e6, 3) + " MHz"],
      ["bandwidth", fmt(bandwidth / 1e6, 3) + " MHz"],
      ["channel width", fmt(sim.chan_width_hz / 1e3, 2) + " kHz"],
      ["starts", sim.start_time_utc + " UTC"]
    ]);
  }

  // The realism panel's fields have no server-side schema of their own
  // (see simulate.py -- these groups are Optional request fields, on only
  // when their `*_enabled` toggle is checked): the list below is this
  // file's own field descriptors, the same pattern `renderSimFields`
  // already uses for the observation fields. Each entry's `section` opens
  // a new labelled subgroup within the one field-grid.
  // `section` opens a labelled subgroup with its own one-line summary;
  // `advanced: true` moves a field into that subgroup's Advanced fold, for
  // knobs whose meaning only lands once you already know the hardware.
  var REALISM_FIELDS = [
    { name: "n_pol", label: "How many polarizations to record", kind: "choice", default: "1",
      options: [{ value: "1", label: "1 — a single receptor" },
        { value: "2", label: "2 — both receptors (XX and YY)" }],
      section: "Polarization",
      sectionIntro: "A real dish has two perpendicular receptors. Recording both lets a"
        + " polarized transmitter look different in each.",
      help: "With two, the waterfall below gains a control for which receptor it shows." },

    { name: "instrument_enabled", label: "Give every antenna its own gain and bandpass",
      kind: "toggle", default: false, section: "Antenna-to-antenna differences",
      sectionIntro: "By default every antenna is identical. Real ones are not: their"
        + " gains differ and their passbands are not flat.",
      help: "Nothing below this line applies until it is ticked." },
    { name: "gain_scatter_db", label: "Gain spread between antennas", kind: "number", unit: "dB",
      factor: 1, min: 0, max: 10, step: 0.05, default: 0.4,
      help: "How much overall sensitivity varies from antenna to antenna." },
    { name: "phase_offsets", label: "Antenna phase offsets", kind: "choice", default: "zero",
      options: [{ value: "zero", label: "None — the array is perfectly calibrated" },
        { value: "uniform", label: "Random — an uncalibrated array" }],
      help: "Random offsets scatter the phases, and the dirty image loses its clean peak." },
    { name: "bandpass_ripple_db", label: "Ripple across the band", kind: "number", unit: "dB",
      factor: 1, min: 0, max: 5, step: 0.01, default: 0.05,
      help: "A gentle wiggle in gain with frequency, different for each antenna." },
    { name: "band_slope_db", label: "Tilt across the band", kind: "number", unit: "dB", factor: 1,
      min: 0, max: 10, step: 0.05, default: 0.0,
      help: "One edge of the band more sensitive than the other." },
    { name: "subband_scatter_db", label: "Gain steps between sub-bands", kind: "number",
      unit: "dB", factor: 1, min: 0, max: 10, step: 0.05, default: 0.0, advanced: true,
      help: "Step changes in gain, as separate digitiser boards covering slices of the"
        + " band would give." },
    { name: "n_subbands", label: "How many sub-bands", kind: "number", factor: 1,
      min: 1, max: 64, step: 1, default: 1, advanced: true,
      help: "How many equal slices the band is cut into for the step above." },

    { name: "quantization_enabled", label: "Round the voltages to 4-bit samples",
      kind: "toggle", default: false, section: "Digitisation",
      sectionIntro: "Recorders write a few bits per sample, not exact numbers. Strong"
        + " interference then spills across the whole band, which is what makes it hard"
        + " to remove after the fact.",
      help: "Off, the voltages keep full precision — an idealisation." },
    { name: "quant_target_counts", label: "Where the noise sits in the 4-bit range",
      kind: "number", unit: "counts", factor: 1, min: 0.1, max: 20, step: 0.01, default: 1.33,
      help: "Rms level the digitiser aims for; about 1.33 counts loses the least"
        + " information for Gaussian noise." },

    { name: "channelizer_enabled", label: "Split the band with a real filterbank",
      kind: "toggle", default: false, section: "Channelisation",
      sectionIntro: "By default channels are formed by a perfect Fourier transform."
        + " Real hardware uses a polyphase filterbank, so neighbouring channels leak"
        + " into each other and a narrow transmitter smears sideways.",
      help: "Off, each channel is perfectly isolated from its neighbours." },
    { name: "n_taps", label: "Filter taps", kind: "number", factor: 1, min: 1, max: 32, step: 1,
      default: 4, advanced: true,
      help: "More taps mean sharper channel edges and less leakage." },
    { name: "window", label: "Window function", kind: "choice", default: "hamming",
      options: [{ value: "hann", label: "Hann" }, { value: "hamming", label: "Hamming" },
        { value: "blackman", label: "Blackman" }], advanced: true,
      help: "The taper applied along the filter; it trades channel width against how"
        + " far the leakage reaches." },
    { name: "sinc_bandwidth", label: "Channel width factor", kind: "number", factor: 1,
      min: 0.1, max: 8, step: 0.01, default: 1.01, advanced: true,
      help: "Width of the prototype filter in channels; 1.01 keeps channels roughly"
        + " one channel wide." },

    { name: "calibration_enabled", label: "Leave residual calibration errors",
      kind: "toggle", default: false, section: "Calibration residuals",
      sectionIntro: "Even a calibrated array is never exactly calibrated. These are the"
        + " leftovers, applied per antenna.",
      help: "Off, calibration is assumed perfect." },
    { name: "phase_error_deg_rms", label: "Leftover phase error", kind: "number", unit: "deg rms",
      factor: 1, min: 0, max: 180, step: 0.5, default: 5.0,
      help: "A constant phase slip per antenna; it smears the image peak." },
    // max matches CalibrationErrorParams.delay_error_ns_rms's `le` bound in
    // simulate.py -- keep the two in sync if either changes.
    { name: "delay_error_ns_rms", label: "Leftover delay error", kind: "number", unit: "ns rms",
      factor: 1, min: 0, max: 10, step: 0.1, default: 0.0, advanced: true,
      help: "A phase error that grows across the band, rather than a constant one." },
    { name: "amplitude_error_db_rms", label: "Leftover amplitude error", kind: "number",
      unit: "dB rms", factor: 1, min: 0, max: 10, step: 0.05, default: 0.0, advanced: true,
      help: "Per-antenna gain left over after calibration." },

    { name: "beam_enabled", label: "Dim sources away from where the dish points",
      kind: "toggle", default: false, section: "Primary beam",
      sectionIntro: "A dish is most sensitive straight ahead. This attenuates celestial"
        + " sources by their offset from the pointing centre; interference is not"
        + " attenuated by it.",
      help: "Off, the dish is equally sensitive in every direction." },
    { name: "beam_type", label: "Beam shape", kind: "choice", default: "gaussian",
      options: [{ value: "gaussian", label: "Gaussian — a smooth main lobe" },
        { value: "airy", label: "Airy — a real aperture, with rings" }],
      help: "The Airy pattern adds the sidelobe rings a circular dish really has." },
    { name: "dish_diameter_m", label: "Dish diameter", kind: "number", unit: "m", factor: 1,
      min: 0.1, max: 1000, step: 0.1, default: 4.5,
      help: "Sets the beam width: a bigger dish sees a narrower patch of sky." }
  ];

  function renderRealismFields() {
    var host = $("realism-fields");
    while (host.firstChild) { host.removeChild(host.firstChild); }
    var grid = null;
    var advancedGrid = null;

    REALISM_FIELDS.forEach(function (descriptor) {
      if (descriptor.section) {
        var block = el("div", "realism-section");
        block.appendChild(el("h4", "subgroup-label", descriptor.section));
        if (descriptor.sectionIntro) {
          block.appendChild(el("p", "group-intro", descriptor.sectionIntro));
        }
        grid = el("div", "field-grid");
        block.appendChild(grid);
        var fold = el("details", "advanced");
        fold.appendChild(el("summary", null, "Advanced"));
        advancedGrid = el("div", "field-grid");
        fold.appendChild(advancedGrid);
        block.appendChild(fold);
        host.appendChild(block);
      }
      (descriptor.advanced ? advancedGrid : grid)
        .appendChild(buildField(descriptor, state.realism));
    });

    // A section with nothing expert-only in it should not advertise a fold.
    Array.prototype.forEach.call(host.querySelectorAll("details.advanced"),
      function (fold) {
        if (!fold.querySelector(".field")) { fold.hidden = true; }
      });
  }

  function defaultRealism() {
    var values = {};
    REALISM_FIELDS.forEach(function (descriptor) { values[descriptor.name] = descriptor.default; });
    return values;
  }

  /* Where the nth added source lands when the user does not say.
   *
   * The first one keeps the server's advertised default offset. Every
   * one after it steps round a golden-angle spiral, growing as the
   * square root of the count so the points stay evenly spread, and
   * capped inside the imaged field. Sources dropped on one another add
   * coherently into a single peak, so a page that gave every new source
   * the same default made two clicks of "Add a source" look like one
   * source in the dirty image.
   */
  var GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

  function defaultOffsetDeg(index) {
    var base = state.defaults.sky_source.position.default_offset_deg;
    if (index <= 0) { return [base[0], base[1]]; }
    var radius = Math.sqrt(base[0] * base[0] + base[1] * base[1]) || 0.5;
    var half = state.pointing ? state.pointing.field_half_width_deg : radius * 2;
    var reach = Math.min(0.75 * half, radius * Math.sqrt(index + 1));
    var angle = Math.atan2(base[1], base[0]) + index * GOLDEN_ANGLE;
    return [
      Math.round(reach * Math.cos(angle) * 1e3) / 1e3,
      Math.round(reach * Math.sin(angle) * 1e3) / 1e3
    ];
  }

  // `state.skySources` must already hold every source that comes before
  // this one: both the name and the spiral placement count it.
  function newSkySource(east_deg, north_deg) {
    var position = state.defaults.sky_source.position;
    var index = state.skySources.length;
    var spot = defaultOffsetDeg(index);
    var values = Object.assign({}, state.defaults.sky_source.defaults);
    values.name = "source " + (index + 1);
    values.mode = position.default_mode;
    values.east_deg = east_deg === undefined ? spot[0] : east_deg;
    values.north_deg = north_deg === undefined ? spot[1] : north_deg;
    // The other two notations are filled in from these the moment the unit
    // switcher is used; seeding them keeps the inputs from starting empty.
    // The offsets themselves are left exactly as typed rather than
    // round-tripped back through the projection.
    values.mode = "offset";
    setSourceMode(values, "lm");
    setSourceMode(values, "radec");
    values.mode = position.default_mode;
    return values;
  }

  function newRfiSource(typeName) {
    var descriptorSet = rfiType(typeName);
    var values = Object.assign({ type: typeName }, descriptorSet.defaults);
    // Named after the tick box that made it, so the ground-truth chips
    // under the waterfall read in the same words as the picker.
    values.name = rfiKind(typeName).title.toLowerCase();
    var same = state.rfiSources.filter(function (source) {
      return source.type === typeName;
    }).length;
    if (same > 0) { values.name += " " + (same + 1); }
    // Extras with no scalar field of their own (see buildRfiExtras): off
    // by default, exactly as the library's own defaults are.
    values.coupling = null;
    values.polarization = null;
    if (typeName === "tower" || typeName === "comb") { values.envelope = null; }
    if (typeName === "impulsive") { values.arrival = "poisson"; }
    return values;
  }

  /* --------------------------------------------------------- 6. presets */

  /* Canned scenarios, built out of the same state the forms edit: loading
   * one resets everything to the server's defaults, applies the scenario,
   * redraws every form, and runs. Nothing here is a separate code path --
   * whatever a preset sets, a reader can find and change in the sections
   * above. */
  var PRESETS = [
    {
      id: "clean",
      label: "Clean sky — one source, no interference",
      // Placed in degrees east and north of the pointing, like every other
      // source the page makes; the server resolves it to direction cosines.
      apply: function () {
        state.skySources = [];
        state.skySources.push(newSkySource(0.5, -0.3));
      }
    },
    {
      id: "tower",
      label: "Cell tower — a narrowband transmitter switching on and off",
      apply: function () {
        state.rfiSources = [newRfiSource("tower")];
      }
    },
    {
      id: "satellite",
      label: "Satellite pass — a carrier drifting with Doppler",
      apply: function () {
        state.rfiSources = [newRfiSource("satellite")];
      }
    },
    {
      id: "busy",
      label: "Busy band — a tower, a harmonic comb and sparking hardware",
      apply: function () {
        state.rfiSources = [
          newRfiSource("tower"),
          newRfiSource("comb"),
          newRfiSource("impulsive")
        ];
      }
    },
    {
      id: "realistic",
      label: "Realistic instrument — the same tower through imperfect hardware",
      apply: function () {
        state.rfiSources = [newRfiSource("tower")];
        state.realism.n_pol = "2";
        state.realism.instrument_enabled = true;
        state.realism.quantization_enabled = true;
        state.realism.channelizer_enabled = true;
        state.realism.calibration_enabled = true;
        state.realism.beam_enabled = true;
      }
    }
  ];

  function defaultSite() {
    var array = state.defaults.array;
    return {
      name: array.name,
      latitude_deg: array.latitude_deg,
      longitude_deg: array.longitude_deg,
      height_m: array.height_m
    };
  }

  function resetToDefaults() {
    var defaults = state.defaults;
    state.antennas = defaults.array.antennas.map(function (row) { return row.slice(); });
    state.site = defaultSite();
    state.pointing = defaults.pointing;
    markArrayLoaded();
    state.sim = {
      n_chan: defaults.sim.n_chan,
      n_blocks: defaults.sim.n_blocks,
      center_freq_hz: defaults.sim.center_freq_hz,
      noise_std: defaults.sim.noise_std,
      seed: defaults.sim.seed
    };
    // Cleared first: `newSkySource` counts what is already there.
    state.skySources = [];
    state.skySources.push(newSkySource());
    state.rfiSources = [];
    state.rfiOpen = [];
    state.spectralLines = [];
    state.realism = defaultRealism();
    state.waterfallPol = 0;
  }

  function renderAllForms() {
    renderSitePlan();
    renderSiteMeta();
    renderPointingHint();
    renderSkyCards();
    renderRfi();
    renderLineCards();
    renderSimFields();
    renderRealismFields();
  }

  function applyPreset(id) {
    var preset = PRESETS.filter(function (entry) { return entry.id === id; })[0];
    if (!preset) { return; }
    resetToDefaults();
    preset.apply();
    renderAllForms();
    $("array-load-note").hidden = true;
    $("waterfall-pol").value = "0";
    run();   // which lands on the Results tab when it finishes
  }

  /* ------------------------------------------------------------- 7. run */

  function showNotice(kind, text) {
    state.notices.push({ kind: kind, text: text });
    renderNotices();
  }

  function renderNotices() {
    var host = $("notices");
    while (host.firstChild) { host.removeChild(host.firstChild); }
    state.notices.forEach(function (notice) {
      var node = el("div", "banner" + (notice.kind === "error" ? " banner-error" : " banner-warn"));
      node.appendChild(el("span", "banner-kind",
        notice.kind === "error" ? "Problem" : "Note"));
      node.appendChild(el("span", null, notice.text));
      host.appendChild(node);
    });
  }

  function buildRequest() {
    var realism = state.realism;
    return {
      antennas: state.antennas.map(function (antenna) {
        return [antenna[0], antenna[1], antenna[2] || 0];
      }),
      site: state.site ? {
        latitude_deg: state.site.latitude_deg,
        longitude_deg: state.site.longitude_deg,
        height_m: state.site.height_m
      } : null,
      // Whichever notation the user chose travels as it stands: the server
      // resolves it against the run's own phase centre, so there is one
      // conversion and it is the library's.
      sky_sources: state.skySources.map(function (source) {
        var payload = { name: source.name, flux_jy: source.flux_jy };
        if (source.mode === "lm") {
          payload.l = source.l;
          payload.m = source.m;
        } else if (source.mode === "radec") {
          payload.radec_deg = [source.ra_deg, source.dec_deg];
        } else {
          payload.offset_deg = [source.east_deg, source.north_deg];
        }
        return payload;
      }),
      rfi_sources: state.rfiSources.map(function (source) {
        var payload = {};
        Object.keys(source).forEach(function (key) {
          if (key === "tle_text" && source.tle_source !== "custom") { return; }
          payload[key] = source[key];
        });
        return payload;
      }),
      spectral_lines: state.spectralLines.map(function (source) {
        return {
          name: source.name,
          center_freq_hz: source.center_freq_hz,
          fwhm_hz: source.fwhm_hz,
          line_flux_jy: source.line_flux_jy
        };
      }),
      sim: {
        n_chan: Math.round(state.sim.n_chan),
        n_blocks: Math.round(state.sim.n_blocks),
        center_freq_hz: state.sim.center_freq_hz,
        noise_std: state.sim.noise_std,
        seed: Math.round(state.sim.seed)
      },
      n_pol: Number(realism.n_pol) === 2 ? 2 : 1,
      instrument: realism.instrument_enabled ? {
        gain_scatter_db: realism.gain_scatter_db,
        phase_offsets: realism.phase_offsets,
        bandpass_ripple_db: realism.bandpass_ripple_db,
        band_slope_db: realism.band_slope_db,
        subband_scatter_db: realism.subband_scatter_db,
        n_subbands: Math.round(realism.n_subbands)
      } : null,
      calibration_errors: realism.calibration_enabled ? {
        phase_error_deg_rms: realism.phase_error_deg_rms,
        delay_error_ns_rms: realism.delay_error_ns_rms,
        amplitude_error_db_rms: realism.amplitude_error_db_rms
      } : null,
      channelizer: realism.channelizer_enabled ? {
        n_taps: Math.round(realism.n_taps),
        window: realism.window,
        sinc_bandwidth: realism.sinc_bandwidth
      } : null,
      primary_beam: realism.beam_enabled ? {
        type: realism.beam_type,
        dish_diameter_m: realism.dish_diameter_m
      } : null,
      quantization: realism.quantization_enabled ? {
        quant_target_counts: realism.quant_target_counts
      } : null
    };
  }

  // One Run button, in the topbar, so it is reachable from either tab.
  function runButtons() { return [$("run")]; }

  // The status reads a four-state cell: info while idle, amber while a run
  // is in flight, green when it lands, red when the server refuses it.
  function setRunStatus(kind, text) {
    var node = $("run-status");
    node.className = "status-cell status-" + kind;
    node.textContent = text;
  }

  function startElapsed() {
    stopElapsed();
    var started = Date.now();
    setRunStatus("warn", "running   0 s");
    state.elapsedTimer = window.setInterval(function () {
      var seconds = Math.round((Date.now() - started) / 1000);
      setRunStatus("warn", "running   " + seconds + " s");
    }, 1000);
  }

  function stopElapsed() {
    if (state.elapsedTimer !== null) {
      window.clearInterval(state.elapsedTimer);
      state.elapsedTimer = null;
    }
  }

  function run() {
    if (state.running) { return; }
    state.running = true;
    state.notices = [];
    renderNotices();
    runButtons().forEach(function (button) { button.disabled = true; });
    startElapsed();
    $("waterfall-sweep").hidden = state.result === null;

    // The exact payload is kept with the run: the clean comparison and the
    // flagger both re-send it, and they are only honest if they send the
    // request this run was made from rather than whatever the forms hold by
    // the time the user asks.
    var payload = buildRequest();
    request("/api/simulate?pol=" + state.waterfallPol, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).then(function (result) {
      stopElapsed();
      rememberRun(result, payload);
      result.warnings.forEach(function (message) { showNotice("note", message); });
      setRunStatus("ok", "done   " + fmt(result.wall_time_s, 2) + " s");
      $("wall-time").textContent = fmt(result.wall_time_s, 2) + " s wall";
      // A finished run is a result to look at. The very first run is the
      // one the page makes for itself on load, and jumping the reader
      // straight past the forms they have not seen yet would be wrong, so
      // that one leaves the tab where the hash put it.
      if (state.booted) { goToTab("results"); }
      state.booted = true;
    }).catch(function (error) {
      stopElapsed();
      showNotice("error", error.message);
      setRunStatus("error", "failed   " + shorten(error.message, 60));
    }).then(function () {
      state.running = false;
      state.booted = true;
      runButtons().forEach(function (button) { button.disabled = false; });
      $("waterfall-sweep").hidden = true;
    });
  }

  function shorten(text, limit) {
    var one = String(text).replace(/\s+/g, " ").trim();
    return one.length > limit ? one.slice(0, limit - 1) + "…" : one;
  }

  /* --------------------------------------------- the recent-run strip */
  /* The last few completed runs are kept in memory -- the response object
     itself, nothing persisted and nothing re-requested. Sliding back
     re-renders a stored response into the same three panels; the newest
     run appends and jumps to latest. There is no polling: the simulator
     only ever runs when asked. */

  function rememberRun(result, payload) {
    state.history.push({
      result: result,
      request: payload,   // what produced it, for the clean twin and the flagger
      clean: null,        // the interference-free twin, once asked for
      showClean: false,   // which of the two the image panel is showing
      flag: null,         // the last flagger response for this run
      at: new Date(),
      n_antennas: state.antennas.length,
      n_sky: state.skySources.length,
      n_rfi: state.rfiSources.length
    });
    while (state.history.length > HISTORY_MAX) { state.history.shift(); }
    showRun(state.history.length - 1);
  }

  // Point the displays at one stored run. Everything the panels read off
  // `state` (which masks are shown, which antenna, whether there are two
  // receptors) is re-derived from that run, not carried over from another.
  function showRun(index) {
    if (index < 0 || index >= state.history.length) { return; }
    var entry = state.history[index];
    var result = entry.result;
    state.historyIndex = index;
    state.result = result;
    state.maskVisible = result.sources.map(function () { return true; });
    state.waterfallAntenna = clamp(
      state.waterfallAntenna, 0, result.waterfall.antennas.length - 1
    );
    state.visBaseline = null;   // baseline indices are this run's, not the last one's
    $("waterfall-pol-group").hidden = result.observation.n_pol !== 2;
    showAbNote("");
    renderRunStrip();
    renderResults();
  }

  function clockOf(date) {
    function pad(value) { return (value < 10 ? "0" : "") + value; }
    return pad(date.getHours()) + ":" + pad(date.getMinutes())
      + ":" + pad(date.getSeconds());
  }

  function plural(count, word) {
    return count + " " + word + (count === 1 ? "" : "s");
  }

  function renderRunStrip() {
    var strip = $("runs-strip");
    var dots = $("runs-dots");
    var total = state.history.length;
    strip.hidden = total < 2;
    // The dots are rebuilt on every change, so a keyboard user's place in
    // them has to be handed back afterwards.
    var hadFocus = dots.contains(document.activeElement);
    while (dots.firstChild) { dots.removeChild(dots.firstChild); }
    if (!total) {
      $("runs-banner").hidden = true;
      return;
    }

    state.history.forEach(function (entry, index) {
      var dot = el("button", "run-dot" + (index === state.historyIndex ? " is-active" : ""));
      dot.type = "button";
      dot.title = "run " + (index + 1) + " of " + total + " — " + clockOf(entry.at);
      dot.setAttribute("aria-label", dot.title);
      dot.setAttribute("aria-pressed", String(index === state.historyIndex));
      dot.addEventListener("click", function () { showRun(index); });
      dot.addEventListener("keydown", function (event) {
        // showRun rebuilds the dots and hands focus back to the new one.
        if (event.key === "ArrowLeft" && index > 0) {
          event.preventDefault();
          showRun(index - 1);
        } else if (event.key === "ArrowRight" && index < total - 1) {
          event.preventDefault();
          showRun(index + 1);
        }
      });
      dots.appendChild(dot);
    });
    if (hadFocus) { focusDot(state.historyIndex); }

    var current = state.history[state.historyIndex];
    $("runs-meta").textContent = "run " + (state.historyIndex + 1) + " of " + total
      + " — " + clockOf(current.at) + ", " + plural(current.n_antennas, "antenna")
      + ", " + plural(current.n_sky, "source") + ", " + current.n_rfi + " RFI";

    var stale = state.historyIndex < total - 1;
    $("runs-banner").hidden = !stale;
    if (stale) {
      $("runs-banner-text").textContent = "Viewing an earlier run — run "
        + (state.historyIndex + 1) + " of " + total + ", recorded at "
        + clockOf(current.at) + ". The panels below show that run, not the newest one.";
    }
  }

  function focusDot(index) {
    var dot = $("runs-dots").children[index];
    if (dot) { dot.focus(); }
  }

  /* -------------------------------------------------------- 8. displays */

  function prepareCanvas(canvas) {
    var rect = canvas.getBoundingClientRect();
    var ratio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(rect.width * ratio));
    canvas.height = Math.max(1, Math.round(rect.height * ratio));
    var ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    return { ctx: ctx, width: rect.width, height: rect.height };
  }

  // One heatmap renderer for both the waterfall and the dirty image:
  // sequential ramp, recessive axes, colourbar with its two ends labelled,
  // and an optional ANY-pooled ground-truth overlay in the reserved red.
  function drawHeatmap(canvas, spec) {
    var surface = prepareCanvas(canvas);
    var ctx = surface.ctx;
    ctx.font = PLOT_FONT;

    // Margins follow the labels rather than the other way round: a
    // frequency axis needs more room than a direction cosine, and neither
    // may sit on top of the other's digits.
    function widest(values, format) {
      return values.reduce(function (most, value) {
        return Math.max(most, ctx.measureText(format(value)).width);
      }, 0);
    }
    var margin = {
      left: Math.ceil(widest(ticks(spec.yLow, spec.yHigh, 4), spec.yFormat)) + 24,
      right: Math.ceil(widest([spec.vmin, spec.vmax], spec.barFormat)) + 32,
      top: 8,
      bottom: 30
    };
    var plot = {
      x: margin.left,
      y: margin.top,
      w: Math.max(10, surface.width - margin.left - margin.right),
      h: Math.max(10, surface.height - margin.top - margin.bottom)
    };

    var rows = spec.values.length;
    var cols = rows ? spec.values[0].length : 0;
    if (!rows || !cols) { return; }

    // Data coordinates to plot pixels. The axes run linearly between the
    // centre of the first cell and the centre of the last one; everything
    // drawn on top of the picture goes through these two so that markers,
    // circles and foreign-grid overlays all agree with the tick labels.
    function toPlotX(value) {
      return plot.x + (value - spec.xLow) / (spec.xHigh - spec.xLow) * plot.w;
    }
    function toPlotY(value) {
      return plot.y + plot.h - (value - spec.yLow) / (spec.yHigh - spec.yLow) * plot.h;
    }

    var off = document.createElement("canvas");
    off.width = cols;
    off.height = rows;
    var offCtx = off.getContext("2d");
    var pixels = offCtx.createImageData(cols, rows);
    var span = spec.vmax - spec.vmin || 1;
    for (var r = 0; r < rows; r += 1) {
      var row = spec.values[r];
      for (var c = 0; c < cols; c += 1) {
        var colour = magma((row[c] - spec.vmin) / span);
        // Row 0 is the low end of the y axis, drawn at the bottom.
        var index = (((rows - 1 - r) * cols) + c) * 4;
        pixels.data[index] = colour[0];
        pixels.data[index + 1] = colour[1];
        pixels.data[index + 2] = colour[2];
        pixels.data[index + 3] = 255;
      }
    }
    offCtx.putImageData(pixels, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, cols, rows, plot.x, plot.y, plot.w, plot.h);

    (spec.overlays || []).forEach(function (overlay) {
      var path = new Path2D();
      var cellW = plot.w / cols;
      var cellH = plot.h / rows;
      for (var mr = 0; mr < rows; mr += 1) {
        var maskRow = overlay.mask[mr];
        for (var mc = 0; mc < cols; mc += 1) {
          if (!maskRow[mc]) { continue; }
          path.rect(
            plot.x + mc * cellW,
            plot.y + (rows - 1 - mr) * cellH,
            Math.ceil(cellW) + 0.5,
            Math.ceil(cellH) + 0.5
          );
        }
      }
      ctx.save();
      ctx.fillStyle = "rgba(" + MASK_RGB + ", " + MASK_ALPHA + ")";
      ctx.fill(path);
      if (spec.hatch) {
        ctx.clip(path);
        ctx.strokeStyle = "rgba(" + MASK_RGB + ", 0.95)";
        ctx.lineWidth = 1;
        for (var d = -plot.h; d < plot.w; d += 6) {
          ctx.beginPath();
          ctx.moveTo(plot.x + d, plot.y + plot.h);
          ctx.lineTo(plot.x + d + plot.h, plot.y);
          ctx.stroke();
        }
      }
      ctx.restore();
    });

    /* An overlay whose grid is not the picture's grid. The flagger decides
       on its own channels and accumulations, so its cells are placed by the
       frequency and time they cover rather than by row and column index,
       and sized from the spacing of its own axes. Anything falling outside
       the plot rectangle is clipped rather than drawn over the labels. */
    (spec.coordOverlays || []).forEach(function (overlay) {
      var freq = overlay.freq_mhz;
      var time = overlay.time_s;
      var nRows = freq.length;
      var nCols = time.length;
      if (!nRows || !nCols) { return; }
      var stepF = nRows > 1
        ? (freq[nRows - 1] - freq[0]) / (nRows - 1)
        : (spec.yHigh - spec.yLow);
      var stepT = nCols > 1
        ? (time[nCols - 1] - time[0]) / (nCols - 1)
        : (spec.xHigh - spec.xLow);
      var cellW = Math.max(1, Math.abs(stepT / (spec.xHigh - spec.xLow) * plot.w));
      var cellH = Math.max(1, Math.abs(stepF / (spec.yHigh - spec.yLow) * plot.h));
      ctx.save();
      ctx.beginPath();
      ctx.rect(plot.x, plot.y, plot.w, plot.h);
      ctx.clip();
      ctx.fillStyle = "rgba(" + overlay.colour + ", "
        + (overlay.alpha === undefined ? FLAG_ALPHA : overlay.alpha) + ")";
      for (var orow = 0; orow < nRows; orow += 1) {
        var line = overlay.mask[orow];
        for (var ocol = 0; ocol < nCols; ocol += 1) {
          if (!line[ocol]) { continue; }
          ctx.fillRect(
            toPlotX(time[ocol]) - cellW / 2,
            toPlotY(freq[orow]) - cellH / 2,
            cellW,
            cellH
          );
        }
      }
      ctx.restore();
    });

    /* A ring at a radius given in the picture's own units -- the primary
       beam's half-power circle on the dirty image. The two axes need not
       share a scale, so each radius is converted on its own axis and the
       ring is drawn as an ellipse. */
    (spec.circles || []).forEach(function (circle) {
      var rx = Math.abs(circle.r / (spec.xHigh - spec.xLow) * plot.w);
      var ry = Math.abs(circle.r / (spec.yHigh - spec.yLow) * plot.h);
      ctx.save();
      ctx.beginPath();
      ctx.rect(plot.x, plot.y, plot.w, plot.h);
      ctx.clip();
      ctx.strokeStyle = circle.colour || TRUTH_MARKER;
      ctx.lineWidth = 1.2;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.ellipse(toPlotX(circle.x), toPlotY(circle.y), rx, ry, 0, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.restore();
    });

    // Markers: every simulated source's true (l, m), drawn as a labelled
    // diamond distinct from the brightest-pixel crosshair below — with RFI
    // present the brightest pixel is not always a real source.
    (spec.truthMarkers || []).forEach(function (truth) {
      var tx = toPlotX(truth.x);
      var ty = toPlotY(truth.y);
      ctx.save();
      ctx.strokeStyle = TRUTH_MARKER;
      ctx.fillStyle = TRUTH_MARKER;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(tx, ty - 7);
      ctx.lineTo(tx + 7, ty);
      ctx.lineTo(tx, ty + 7);
      ctx.lineTo(tx - 7, ty);
      ctx.closePath();
      ctx.stroke();
      if (truth.label) {
        ctx.font = PLOT_FONT;
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        ctx.fillText(truth.label, tx + 9, ty - 4);
      }
      ctx.restore();
    });

    // Marker: the image's peak, in the sky-source colour.
    if (spec.marker) {
      var mx = toPlotX(spec.marker.x);
      var my = toPlotY(spec.marker.y);
      ctx.save();
      ctx.strokeStyle = SKY_MARKER;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(mx, my, 6, 0, 2 * Math.PI);
      ctx.moveTo(mx - 10, my);
      ctx.lineTo(mx - 7, my);
      ctx.moveTo(mx + 7, my);
      ctx.lineTo(mx + 10, my);
      ctx.moveTo(mx, my - 10);
      ctx.lineTo(mx, my - 7);
      ctx.moveTo(mx, my + 7);
      ctx.lineTo(mx, my + 10);
      ctx.stroke();
      ctx.restore();
    }

    ctx.strokeStyle = AXIS_STROKE;
    ctx.lineWidth = 1;
    ctx.strokeRect(plot.x + 0.5, plot.y + 0.5, plot.w - 1, plot.h - 1);

    ctx.fillStyle = AXIS_TEXT;
    ctx.font = PLOT_FONT;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ticks(spec.xLow, spec.xHigh, 4).forEach(function (value, i) {
      var x = plot.x + (i / 4) * plot.w;
      ctx.fillText(spec.xFormat(value), x, plot.y + plot.h + 5);
    });
    ctx.fillText(spec.xLabel, plot.x + plot.w / 2, plot.y + plot.h + 17);

    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ticks(spec.yLow, spec.yHigh, 4).forEach(function (value, i) {
      var y = plot.y + plot.h - (i / 4) * plot.h;
      ctx.fillText(spec.yFormat(value), plot.x - 6, y);
    });
    ctx.save();
    ctx.translate(11, plot.y + plot.h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(spec.yLabel, 0, 0);
    ctx.restore();

    // Colourbar.
    var barX = plot.x + plot.w + 14;
    var barW = 10;
    for (var by = 0; by < plot.h; by += 1) {
      var colourBar = magma(1 - by / plot.h);
      ctx.fillStyle = "rgb(" + colourBar.join(",") + ")";
      ctx.fillRect(barX, plot.y + by, barW, 1);
    }
    ctx.strokeStyle = AXIS_STROKE;
    ctx.strokeRect(barX + 0.5, plot.y + 0.5, barW, plot.h - 1);
    ctx.fillStyle = AXIS_TEXT;
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(spec.barFormat(spec.vmax), barX + barW + 4, plot.y);
    ctx.textBaseline = "bottom";
    ctx.fillText(spec.barFormat(spec.vmin), barX + barW + 4, plot.y + plot.h);
    ctx.textBaseline = "top";
    ctx.fillText(spec.barLabel, barX + barW + 4, plot.y + plot.h / 2 - 6);

    canvas.plotSpec = { plot: plot, rows: rows, cols: cols, spec: spec };
  }

  /* The featured single-antenna waterfall. Two things can be painted over
     it and they are deliberately separate: the ground-truth mask, which is
     what the simulator knows, and the flagger overlay, which is what a
     method decided. The master switch gates the first; the second only
     appears once a flagger has run on this very antenna and receptor. */
  function renderWaterfall() {
    var canvas = $("waterfall-canvas");
    var result = state.result;
    if (!result) { return; }
    var water = result.waterfall;
    var values = water.antennas[state.waterfallAntenna];
    var overlays = state.truthVisible
      ? result.sources.filter(function (source, index) {
        return state.maskVisible[index];
      }).map(function (source) {
        return { mask: source.mask };
      })
      : [];
    var flag = activeFlag();
    var coordOverlays = [];
    if (flag) {
      var chosen = flag.response.methods[flag.active];
      var grid = flag.response.grid;
      // Caught first, false alarm next, missed last: where a coarse cell
      // holds more than one outcome the worst one is what stays visible.
      ["caught", "false_alarm", "missed"].forEach(function (key) {
        coordOverlays.push({
          mask: chosen.overlay[key],
          freq_mhz: grid.freq_mhz,
          time_s: grid.time_s,
          colour: FLAG_COLOURS[key]
        });
      });
    }

    drawHeatmap(canvas, {
      values: values,
      vmin: water.vmin_db,
      vmax: water.vmax_db,
      xLow: water.time_s[0],
      xHigh: water.time_s[water.time_s.length - 1],
      yLow: water.freq_mhz[0],
      yHigh: water.freq_mhz[water.freq_mhz.length - 1],
      xLabel: "time (s)",
      yLabel: "frequency (MHz)",
      barLabel: "dB",
      xFormat: function (v) { return v.toFixed(3); },
      yFormat: function (v) { return v.toFixed(2); },
      barFormat: function (v) { return v.toFixed(0); },
      overlays: overlays,
      coordOverlays: coordOverlays,
      hatch: state.hatch,
      readCell: function (row, col) {
        return "antenna " + state.waterfallAntenna + "\n"
          + water.freq_mhz[row].toFixed(3) + " MHz\n"
          + water.time_s[col].toFixed(4) + " s\n"
          + values[row][col].toFixed(2) + " dB"
          + maskedBy(row, col);
      }
    });

    $("waterfall-sub").textContent =
      water.freq_mhz.length + " × " + water.time_s.length + " cells · "
      + water.vmin_db.toFixed(0) + " to " + water.vmax_db.toFixed(0) + " dB";
    $("waterfall-empty").hidden = true;
  }

  function maskedBy(row, col) {
    if (!state.truthVisible) { return ""; }
    var names = state.result.sources.filter(function (source, index) {
      return state.maskVisible[index] && source.mask[row][col];
    }).map(function (source) { return source.name; });
    return names.length ? "\nflagged: " + names.join(", ") : "";
  }

  /* --- the thumbnail wall ---------------------------------------------
   *
   * The wall is where a run lands, not an extra behind a switch: what an
   * array does to interference is a per-antenna story -- one antenna
   * coupling harder than its neighbours, one clipping, one dead -- and none
   * of that is visible one antenna at a time. Clicking a tile features that
   * antenna full size, with a way back; the antenna picker does the same
   * thing for anyone who would rather type a number. The choice sticks for
   * the session, so re-running while looking at antenna 7 keeps antenna 7.
   *
   * Every tile comes from the waterfalls the run already shipped -- the
   * server sends all of them under one global cell budget, so nothing is
   * re-requested and the per-antenna resolution falls as the array grows.
   * Each tile is the same magma ramp on the run's shared vmin/vmax, which
   * is the whole point: antennas are only comparable stretched identically.
   * Drawing at cell resolution into an offscreen canvas and letting the
   * browser scale it up with smoothing off is the only arithmetic here.
   */

  // How many tiles across. Roughly the square root of the antenna count,
  // leaned wide because a dynamic spectrum is wider than it is tall: 10
  // antennas make 5 columns of generous tiles, 32 make 8, and anything
  // from ~70 up sits at the 12-column floor on tile size. The grid template
  // is `repeat(var(--ant-cols), minmax(0, 1fr))`, so whatever the count,
  // the tiles divide the panel width exactly and nothing scrolls sideways.
  function thumbnailColumns(count) {
    return Math.round(clamp(Math.sqrt(count) * 1.45, 3, 12));
  }

  function drawThumbnail(canvas, values, vmin, vmax) {
    var rows = values.length;
    var cols = rows ? values[0].length : 0;
    if (!rows || !cols) { return; }
    var ratio = window.devicePixelRatio || 1;
    var rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round((rect.width || 120) * ratio));
    canvas.height = Math.max(1, Math.round((rect.height || 64) * ratio));

    var off = document.createElement("canvas");
    off.width = cols;
    off.height = rows;
    var offCtx = off.getContext("2d");
    var pixels = offCtx.createImageData(cols, rows);
    var span = vmax - vmin || 1;
    for (var r = 0; r < rows; r += 1) {
      var row = values[r];
      for (var c = 0; c < cols; c += 1) {
        var colour = magma((row[c] - vmin) / span);
        var index = (((rows - 1 - r) * cols) + c) * 4;
        pixels.data[index] = colour[0];
        pixels.data[index + 1] = colour[1];
        pixels.data[index + 2] = colour[2];
        pixels.data[index + 3] = 255;
      }
    }
    offCtx.putImageData(pixels, 0, 0);
    var ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, cols, rows, 0, 0, canvas.width, canvas.height);
  }

  // The one control that switches between the two views. On the wall it is
  // a pressed switch; on a featured antenna it is the way back, and says so.
  function renderAllAntennasControl() {
    var button = $("all-antennas-toggle");
    setPressed(button, state.allAntennas);
    $("all-antennas-label").textContent = state.allAntennas
      ? "All antennas"
      : "← All antennas";
    button.title = state.allAntennas
      ? "Showing every antenna; click a tile to feature one"
      : "Back to every antenna";
  }

  function featureAntenna(index) {
    state.waterfallAntenna = index;
    state.allAntennas = false;
    state.flagError = false;   // the flagger's scores were for another antenna
    $("waterfall-antenna").value = String(index);
    renderAllAntennasControl();
    renderThumbnails();
    renderFlagger();
    renderWaterfall();
  }

  function renderThumbnails() {
    var panel = $("antenna-thumbnails-panel");
    panel.hidden = !state.allAntennas;
    $("waterfall-wrap").hidden = state.allAntennas;
    var host = $("antenna-thumbnails");
    while (host.firstChild) { host.removeChild(host.firstChild); }
    if (!state.allAntennas || !state.result) { return; }

    var water = state.result.waterfall;
    // Set before anything is measured: the tiles are sized by the grid, and
    // each canvas draws itself at whatever width the grid gave it.
    host.style.setProperty("--ant-cols", String(thumbnailColumns(water.antennas.length)));

    water.antennas.forEach(function (values, index) {
      // `data-status` is where a per-antenna verdict (clipping, dead) would
      // tint the tile; nothing sets it yet.
      var button = el("button", "thumb"
        + (index === state.waterfallAntenna ? " is-active" : ""));
      button.type = "button";
      button.setAttribute("data-status", "ok");
      button.title = "Feature antenna " + index;
      button.setAttribute("aria-label", "Feature antenna " + index);
      var canvas = el("canvas", "thumb-canvas");
      button.appendChild(canvas);
      button.appendChild(el("span", "thumb-label mono", String(index)));
      button.addEventListener("click", function () { featureAntenna(index); });
      host.appendChild(button);
      drawThumbnail(canvas, values, water.vmin_db, water.vmax_db);
    });

    $("antenna-thumbnails-sub").textContent =
      water.antennas.length + " antennas · " + water.freq_mhz.length + " × "
      + water.time_s.length + " cells each · pooled over "
      + water.time_samples_per_cell + " voltage samples per pixel";
  }

  /* --- level 2: what the correlator sees -------------------------------
   *
   * The visibility panel is drawn by the same machinery as the waterfall --
   * one heatmap, one ground-truth switch -- because the point being made is
   * that the same interference survives correlation and is still a
   * time-frequency picture, just a smaller one. The per-baseline spectra
   * are a line plot rather than a heatmap: phase against frequency is the
   * quantity a visibility-domain method actually keys on, and it has no
   * sensible sequential colour.
   */

  // A stack of small line plots sharing one x axis, in the same axis and
  // tick idiom as drawHeatmap. No library: each panel is a set of
  // equal-length series drawn as polylines, autoscaled per panel unless the
  // caller pins the range.
  function drawLineStack(canvas, spec) {
    var surface = prepareCanvas(canvas);
    var ctx = surface.ctx;
    ctx.font = PLOT_FONT;
    if (!spec.x.length || !spec.panels.length) { return; }

    function widest(values, format) {
      return values.reduce(function (most, value) {
        return Math.max(most, ctx.measureText(format(value)).width);
      }, 0);
    }

    var left = 0;
    var ranges = spec.panels.map(function (panel) {
      var low = Infinity;
      var high = -Infinity;
      panel.series.forEach(function (series) {
        series.forEach(function (value) {
          if (!isFinite(value)) { return; }
          low = Math.min(low, value);
          high = Math.max(high, value);
        });
      });
      if (!isFinite(low)) { low = 0; high = 1; }
      if (high - low < 1e-12) { high = low + 1; }
      var pad = (high - low) * 0.06;
      var range = [low - pad, high + pad];
      left = Math.max(left, widest(ticks(range[0], range[1], 3), panel.yFormat));
      return range;
    });

    var margin = { left: Math.ceil(left) + 26, right: 12, top: 10, bottom: 34 };
    var gap = 26;
    var plotW = Math.max(10, surface.width - margin.left - margin.right);
    var plotH = Math.max(
      10,
      (surface.height - margin.top - margin.bottom - gap * (spec.panels.length - 1))
        / spec.panels.length
    );
    var xLow = spec.x[0];
    var xHigh = spec.x[spec.x.length - 1];
    if (xHigh - xLow === 0) { xHigh = xLow + 1; }

    spec.panels.forEach(function (panel, panelIndex) {
      var top = margin.top + panelIndex * (plotH + gap);
      var range = ranges[panelIndex];

      ctx.strokeStyle = GRID_STROKE;
      ctx.lineWidth = 1;
      ticks(range[0], range[1], 3).forEach(function (value, i) {
        var y = top + plotH - (i / 3) * plotH;
        ctx.beginPath();
        ctx.moveTo(margin.left, y);
        ctx.lineTo(margin.left + plotW, y);
        ctx.stroke();
      });

      panel.series.forEach(function (series, seriesIndex) {
        // Bright end of the ramp only: the dark end of magma vanishes into
        // the plotting surface.
        var shade = panel.series.length > 1
          ? seriesIndex / (panel.series.length - 1)
          : 1;
        var rgb = magma(0.3 + 0.65 * shade);
        ctx.strokeStyle = "rgb(" + rgb.join(",") + ")";
        ctx.lineWidth = 1;
        ctx.beginPath();
        series.forEach(function (value, i) {
          var x = margin.left + (spec.x[i] - xLow) / (xHigh - xLow) * plotW;
          var y = top + plotH - (value - range[0]) / (range[1] - range[0]) * plotH;
          if (i === 0) { ctx.moveTo(x, y); } else { ctx.lineTo(x, y); }
        });
        ctx.stroke();
      });

      ctx.strokeStyle = AXIS_STROKE;
      ctx.strokeRect(margin.left + 0.5, top + 0.5, plotW - 1, plotH - 1);

      ctx.fillStyle = AXIS_TEXT;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ticks(range[0], range[1], 3).forEach(function (value, i) {
        ctx.fillText(panel.yFormat(value), margin.left - 6, top + plotH - (i / 3) * plotH);
      });
      ctx.save();
      ctx.translate(11, top + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(panel.yLabel, 0, 0);
      ctx.restore();

      var last = panelIndex === spec.panels.length - 1;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ticks(xLow, xHigh, 4).forEach(function (value, i) {
        ctx.fillText(spec.xFormat(value), margin.left + (i / 4) * plotW, top + plotH + 5);
      });
      if (last) {
        ctx.fillText(spec.xLabel, margin.left + plotW / 2, top + plotH + 18);
      }
    });
  }

  function visBaselines() {
    var vis = state.result ? state.result.visibilities : null;
    if (!vis) { return []; }
    var offered = vis.spectra.baselines;
    return vis.baselines.filter(function (baseline) {
      return offered.indexOf(baseline.index) >= 0;
    });
  }

  function renderVisBaselineSelect() {
    var select = $("vis-baseline");
    while (select.firstChild) { select.removeChild(select.firstChild); }
    var offered = visBaselines();
    if (!offered.length) {
      select.appendChild(el("option", null, "no baselines offered"));
      select.disabled = true;
      return;
    }
    select.disabled = false;
    var known = offered.some(function (baseline) {
      return baseline.index === state.visBaseline;
    });
    if (!known) { state.visBaseline = offered[0].index; }
    offered.forEach(function (baseline) {
      var option = el("option", null, "baseline " + baseline.ant_1 + "–" + baseline.ant_2
        + " · " + baseline.length_m.toFixed(1) + " m");
      option.value = String(baseline.index);
      select.appendChild(option);
    });
    select.value = String(state.visBaseline);
  }

  function renderVisSpectrum() {
    if (!state.result || !$("vis-more").open) { return; }
    var vis = state.result.visibilities;
    var spectra = vis.spectra;
    var which = spectra.baselines.indexOf(state.visBaseline);
    var note = $("vis-spectra-note");
    if (which < 0) {
      note.textContent = "no spectra for this baseline";
      return;
    }
    drawLineStack($("vis-spectrum"), {
      x: spectra.freq_mhz,
      xLabel: "frequency (MHz)",
      xFormat: function (v) { return v.toFixed(2); },
      panels: [
        {
          series: spectra.amplitude[which],
          yLabel: "amplitude (Jy)",
          yFormat: function (v) { return v.toFixed(1); }
        },
        {
          series: spectra.phase_deg[which],
          yLabel: "phase (deg)",
          yFormat: function (v) { return v.toFixed(0); }
        }
      ]
    });

    var lines = ["one line per integration · " + spectra.amplitude[which].length
      + " integrations × " + spectra.freq_mhz.length + " channels"];
    if (spectra.integrations_per_bin > 1 || spectra.channels_per_bin > 1) {
      lines.push("averaged " + spectra.integrations_per_bin + " integrations and "
        + spectra.channels_per_bin + " channels per point");
    }
    if (spectra.n_baselines_offered < vis.n_baselines) {
      lines.push("spectra offered for " + spectra.n_baselines_offered + " of "
        + vis.n_baselines + " baselines, shortest first");
    }
    note.textContent = lines.join(" · ");
  }

  function renderVisibilities() {
    var result = state.result;
    if (!result || !result.visibilities) { return; }
    var vis = result.visibilities;
    var overlays = state.visTruth
      ? vis.sources.map(function (source) { return { mask: source.mask }; })
      : [];

    drawHeatmap($("vis-canvas"), {
      values: vis.amplitude,
      vmin: vis.vmin_jy,
      vmax: vis.vmax_jy,
      xLow: vis.time_s[0],
      xHigh: vis.time_s[vis.time_s.length - 1],
      yLow: vis.freq_mhz[0],
      yHigh: vis.freq_mhz[vis.freq_mhz.length - 1],
      xLabel: "time (s)",
      yLabel: "frequency (MHz)",
      barLabel: "Jy",
      xFormat: function (v) { return v.toFixed(3); },
      yFormat: function (v) { return v.toFixed(2); },
      barFormat: function (v) { return v.toFixed(1); },
      overlays: overlays,
      readCell: function (row, col) {
        return vis.freq_mhz[row].toFixed(3) + " MHz\n"
          + vis.time_s[col].toFixed(4) + " s\n"
          + vis.amplitude[row][col].toFixed(3) + " Jy"
          + visMaskedBy(row, col);
      }
    });

    $("vis-sub").textContent = vis.n_baselines + " baselines · "
      + vis.n_integrations + " integrations of "
      + (vis.integration_time_s * 1000).toFixed(2) + " ms · peak "
      + vis.peak_jy.toFixed(1) + " Jy";
    $("vis-empty").hidden = true;

    renderVisBaselineSelect();
    renderVisSpectrum();
  }

  function visMaskedBy(row, col) {
    if (!state.visTruth) { return ""; }
    var names = state.result.visibilities.sources.filter(function (source) {
      return source.mask[row][col];
    }).map(function (source) { return source.name; });
    return names.length ? "\nflagged: " + names.join(", ") : "";
  }

  function nearestIndex(sortedValues, target) {
    var best = 0;
    var bestDist = Infinity;
    for (var i = 0; i < sortedValues.length; i += 1) {
      var dist = Math.abs(sortedValues[i] - target);
      if (dist < bestDist) {
        bestDist = dist;
        best = i;
      }
    }
    return best;
  }

  /* --- level 3: what the astronomer gets -------------------------------
   *
   * Three honesty devices sit on this panel. The colour-scale ends are
   * printed, because two runs with different scales look alike and are not.
   * The primary beam's half-power ring is drawn, because sources outside it
   * are dimmed by the instrument rather than by the sky. And the clean
   * comparison re-runs the identical request with the interference removed
   * and the seed pinned, so the difference between the two images is the
   * interference and nothing else.
   */

  // Which result the image panel is showing: the run itself, or its cached
  // interference-free twin. A/B state lives on the history entry, so
  // sliding the run strip shows each run's own comparison.
  function shownEntry() {
    return state.history[state.historyIndex] || null;
  }

  function shownImageResult() {
    var entry = shownEntry();
    if (entry && entry.showClean && entry.clean) { return entry.clean; }
    return state.result;
  }

  function renderImage() {
    var canvas = $("image-canvas");
    var result = shownImageResult();
    if (!result) { return; }
    var image = result.image;
    var skySources = result.sky_sources || [];
    var inField = skySources.filter(function (source) { return source.in_field; });
    var outside = skySources.filter(function (source) { return !source.in_field; });

    drawHeatmap(canvas, {
      values: image.values,
      vmin: image.vmin_jy,
      vmax: image.vmax_jy,
      xLow: image.l[0],
      xHigh: image.l[image.l.length - 1],
      yLow: image.m[0],
      yHigh: image.m[image.m.length - 1],
      xLabel: "l (direction cosine)",
      yLabel: "m (direction cosine)",
      barLabel: "Jy",
      xFormat: function (v) { return v.toFixed(3); },
      yFormat: function (v) { return v.toFixed(3); },
      barFormat: function (v) { return v.toFixed(2); },
      marker: { x: image.peak.l, y: image.peak.m },
      truthMarkers: inField.map(function (source) {
        return { x: source.l, y: source.m, label: source.name };
      }),
      // The image axes are direction cosines, so an angular radius enters
      // as its sine. At these angles the difference is below 1e-4, but the
      // sine is what the axis actually measures.
      circles: beamCircle(image),
      readCell: function (row, col) {
        return "l " + image.l[col].toFixed(4) + "\nm " + image.m[row].toFixed(4)
          + "\n" + image.values[row][col].toFixed(3) + " Jy";
      }
    });

    // One row per source that landed in the field: what went in, what came
    // back out, and where it sits. Numbers right-aligned, so the catalog
    // and recovered columns can be read against each other.
    // The beam-response column only exists when the run had a primary beam;
    // an empty column of dashes would say nothing.
    var withBeam = skySources.some(function (source) {
      return source.beam_response !== null && source.beam_response !== undefined;
    });
    var headers = ["source", "catalog Jy", "recovered Jy", "position"];
    if (withBeam) { headers.splice(3, 0, "beam response"); }
    var rows = inField.map(function (source) {
      var col = nearestIndex(image.l, source.l);
      var row = nearestIndex(image.m, source.m);
      var line = [
        source.name,
        source.flux_jy.toFixed(2),
        image.values[row][col].toFixed(2),
        "l " + source.l.toFixed(4) + ", m " + source.m.toFixed(4)
      ];
      if (withBeam) {
        line.splice(3, 0, source.beam_response === null
          || source.beam_response === undefined
          ? "—"
          : source.beam_response.toFixed(2));
      }
      return line;
    });
    fillMetricTable($("image-recovered"), headers, rows);

    var peakLine = "brightest pixel: " + image.peak.value_jy.toFixed(3) + " Jy at l "
      + image.peak.l.toFixed(4) + ", m " + image.peak.m.toFixed(4);
    $("image-sub").textContent = peakLine;

    $("image-scale").textContent = "scale " + image.vmin_jy.toFixed(2) + " – "
      + image.vmax_jy.toFixed(2) + " Jy";

    var legend = $("image-beam-legend");
    var beam = image.beam;
    if (beam && beam.half_power_rad !== null && beam.half_power_rad !== undefined) {
      legend.textContent = "dashed circle = primary beam half-power, "
        + deg(beam.half_power_rad).toFixed(2) + "° radius at band centre";
      legend.hidden = false;
    } else {
      legend.hidden = true;
    }

    renderAbState();

    var outsideNote = $("image-outside");
    if (outside.length) {
      $("image-outside-text").textContent = "outside the imaged field: "
        + outside.map(function (source) { return source.name; }).join(", ");
      outsideNote.hidden = false;
    } else {
      outsideNote.hidden = true;
    }

    $("image-sidelobe-note").hidden = skySources.length < 1;
  }

  function beamCircle(image) {
    var beam = image.beam;
    if (!beam || beam.half_power_rad === null || beam.half_power_rad === undefined) {
      return [];
    }
    return [{ x: 0, y: 0, r: Math.sin(beam.half_power_rad), colour: TRUTH_MARKER }];
  }

  function renderAbState() {
    var entry = shownEntry();
    var button = $("image-ab-toggle");
    var label = $("image-ab-state");
    var comparable = Boolean(entry) && entry.result.sources.length > 0;
    button.disabled = !comparable || state.abRunning;
    button.title = comparable
      ? "Re-run this observation with the interference removed and the same seed"
      : "This run has no interference to remove";
    setPressed(button, Boolean(entry && entry.showClean));
    label.textContent = entry && entry.showClean
      ? "showing: clean, no interference"
      : "showing: contaminated";
    label.className = entry && entry.showClean ? "badge badge-amber" : "badge";
  }

  function showAbNote(text) {
    var note = $("image-ab-note");
    if (!text) {
      note.hidden = true;
      return;
    }
    $("image-ab-note-text").textContent = text;
    note.hidden = false;
  }

  /* The clean twin of a run: the same request with the interference list
     emptied and the same explicit seed, so noise, gains and the sky are
     bit-for-bit what the contaminated run had and the only difference in
     the image is the interference. It is computed once per run and cached
     on that run's history entry; flipping back and forth costs nothing. */
  function toggleClean() {
    var entry = shownEntry();
    if (!entry || state.abRunning) { return; }
    if (entry.showClean) {
      entry.showClean = false;
      showAbNote("");
      renderImage();
      return;
    }
    if (entry.clean) {
      entry.showClean = true;
      showAbNote("");
      renderImage();
      return;
    }

    var payload = JSON.parse(JSON.stringify(entry.request));
    payload.rfi_sources = [];
    // The seed is what makes the two runs comparable, so it is asserted
    // rather than assumed: a request that somehow arrived without one is
    // pinned to the seed the contaminated run reported using.
    if (payload.sim.seed === null || payload.sim.seed === undefined
        || !isFinite(payload.sim.seed)) {
      payload.sim.seed = entry.result.observation.seed;
    }

    state.abRunning = true;
    renderAbState();
    showAbNote("Running the same observation without interference — same seed, same"
      + " everything else.");
    request("/api/simulate?pol=" + entry.result.observation.waterfall_pol, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).then(function (clean) {
      entry.clean = clean;
      entry.showClean = true;
      showAbNote("");
    }).catch(function (error) {
      showNotice("error", "Could not run the clean comparison: " + error.message);
      showAbNote("");
    }).then(function () {
      state.abRunning = false;
      renderImage();
    });
  }

  function renderUv() {
    var svg = $("uv-plot");
    var result = state.result;
    if (!result) { return; }
    var rect = svg.getBoundingClientRect();
    var width = Math.max(120, rect.width);
    var height = Math.max(100, rect.height);
    svg.setAttribute("viewBox", "0 0 " + width + " " + height);
    while (svg.firstChild) { svg.removeChild(svg.firstChild); }

    var margin = { left: 46, right: 12, top: 8, bottom: 30 };
    var plotW = width - margin.left - margin.right;
    var plotH = height - margin.top - margin.bottom;
    var limit = Math.max(1, result.uv.max_lambda) * 1.1;
    var half = Math.min(plotW, plotH) / 2;
    var cx = margin.left + plotW / 2;
    var cy = margin.top + plotH / 2;

    [0.5, 1].forEach(function (fraction) {
      svg.appendChild(svgNode("circle", {
        cx: cx, cy: cy, r: half * fraction, "class": "uv-grid", fill: "none"
      }));
    });
    svg.appendChild(svgNode("line", {
      x1: cx - half, y1: cy, x2: cx + half, y2: cy, "class": "uv-axis"
    }));
    svg.appendChild(svgNode("line", {
      x1: cx, y1: cy - half, x2: cx, y2: cy + half, "class": "uv-axis"
    }));

    var points = [];
    result.uv.u.forEach(function (u, index) {
      var v = result.uv.v[index];
      points.push([u, v]);
      points.push([-u, -v]);
    });

    points.forEach(function (point) {
      svg.appendChild(svgNode("circle", {
        cx: cx + point[0] / limit * half,
        cy: cy - point[1] / limit * half,
        r: 1.7,
        "class": "uv-point"
      }));
    });

    [[cx + half, cy + 14, "end", limit.toFixed(0)],
     [cx, cy + 14, "middle", "0"],
     [cx - 6, cy - half + 4, "end", limit.toFixed(0)]].forEach(function (entry) {
      var text = svgNode("text", {
        x: entry[0], y: entry[1], "class": "uv-tick-label", "text-anchor": entry[2]
      });
      text.textContent = entry[3];
      svg.appendChild(text);
    });

    var uLabel = svgNode("text", {
      x: cx, y: height - 6, "class": "uv-tick-label", "text-anchor": "middle"
    });
    uLabel.textContent = "u (wavelengths)";
    svg.appendChild(uLabel);
    var vLabel = svgNode("text", {
      x: 12, y: cy, "class": "uv-tick-label", "text-anchor": "middle",
      transform: "rotate(-90 12 " + cy + ")"
    });
    vLabel.textContent = "v (wavelengths)";
    svg.appendChild(vLabel);

    svg.uvSpec = { cx: cx, cy: cy, half: half, limit: limit, points: points };
    $("uv-sub").textContent =
      points.length + " samples · longest "
      + result.uv.max_lambda.toFixed(0) + " λ";
  }

  function renderMaskToggles() {
    var host = $("mask-toggles");
    while (host.firstChild) { host.removeChild(host.firstChild); }
    if (!state.result || !state.result.sources.length) {
      host.appendChild(el("span", "checkline", "none in this run"));
      return;
    }
    state.result.sources.forEach(function (source, index) {
      var button = el("button", "mask-toggle");
      button.type = "button";
      button.setAttribute("aria-pressed", String(state.maskVisible[index]));
      button.appendChild(el("span", "mask-chip"));
      button.appendChild(document.createTextNode(
        source.name + " " + (source.occupancy * 100).toFixed(1) + "%"
      ));
      button.title = source.name + " occupies "
        + (source.occupancy * 100).toFixed(2) + "% of the time-frequency cells";
      button.addEventListener("click", function () {
        state.maskVisible[index] = !state.maskVisible[index];
        button.setAttribute("aria-pressed", String(state.maskVisible[index]));
        renderWaterfall();
      });
      host.appendChild(button);
    });

    /* The display grid is coarser than the mask: a cell is drawn flagged
       if any voltage sample inside it was, so the red extent is an upper
       bound on the occupancy printed on the chip. Say so rather than let
       a 1% source that paints half the picture look like a contradiction. */
    var pooled = state.result.waterfall.time_samples_per_cell;
    if (pooled > 1) {
      host.appendChild(el(
        "span",
        "checkline",
        "mask pooled over " + pooled
          + " samples/pixel — displayed extent exceeds occupancy"
      ));
    }
  }

  function renderAntennaSelect() {
    var select = $("waterfall-antenna");
    while (select.firstChild) { select.removeChild(select.firstChild); }
    if (!state.result) { return; }
    state.result.waterfall.antennas.forEach(function (plane, index) {
      var option = el("option", null, String(index));
      option.value = index;
      select.appendChild(option);
    });
    select.value = state.waterfallAntenna;
  }

  function renderResults() {
    if (!state.result) { return; }
    renderAntennaSelect();
    renderMaskToggles();
    renderAllAntennasControl();
    renderThumbnails();
    renderFlagger();
    renderWaterfall();
    renderVisibilities();
    renderImage();
    renderUv();
  }

  /* ------------------------------------------------------- 9. flagging */

  /* A classical flagger, run on the server against the same voltages the
   * featured waterfall shows, and scored against the same ground truth an
   * excision algorithm would be scored against. Two things make this
   * pedagogy rather than decoration:
   *
   *   - the outcome is painted per cell, split into caught / missed / false
   *     alarm, so a number like "recall 0.45" has a picture attached; and
   *   - the flagger decides on its own grid, coarser in time than the
   *     display, so the overlay is placed by frequency and time (see
   *     drawHeatmap's coordOverlays) and the grid is printed. A method that
   *     decides on accumulated power is not comparing like with like
   *     against one that decides on pre-detection voltages, and the panel
   *     says so instead of hiding it.
   *
   * Nothing is computed here: masks, overlays and scores all come back from
   * the server. Results are cached on the run's history entry and keyed by
   * the antenna and receptor they were computed for, so featuring another
   * antenna puts the control back rather than showing a stale overlay.
   */

  function flaggerSchema() {
    return (state.defaults && state.defaults.flaggers) || null;
  }

  function activeFlag() {
    var entry = shownEntry();
    if (!entry || !entry.flag) { return null; }
    if (entry.flag.antenna !== state.waterfallAntenna) { return null; }
    if (entry.flag.pol !== state.waterfallPol) { return null; }
    return entry.flag;
  }

  function renderFlaggerControls() {
    var schema = flaggerSchema();
    var panel = $("flagger-controls");
    if (!schema || !schema.methods.length) {
      panel.hidden = true;
      return;
    }
    panel.hidden = false;
    var host = $("flagger-methods");
    while (host.firstChild) { host.removeChild(host.firstChild); }

    schema.methods.forEach(function (method) {
      var id = "flagger-method-" + method.value;
      var wrap = el("div", "flagger-method");
      var line = el("label", "checkline flagger-method-name");
      var box = el("input");
      box.type = "checkbox";
      box.id = id;
      box.value = method.value;
      box.checked = state.flagMethods.indexOf(method.value) >= 0;
      box.addEventListener("change", function () {
        var at = state.flagMethods.indexOf(method.value);
        if (box.checked && at < 0) {
          state.flagMethods.push(method.value);
        } else if (!box.checked && at >= 0) {
          state.flagMethods.splice(at, 1);
        }
        syncFlaggerCap();
      });
      line.appendChild(box);
      line.appendChild(document.createTextNode(" " + method.label));
      wrap.appendChild(line);
      wrap.appendChild(el("p", "field-hint", method.summary));
      wrap.appendChild(el("p", "field-hint flagger-method-grid",
        "decides on: " + method.grid));
      host.appendChild(wrap);
    });
    syncFlaggerCap();
  }

  // At the cap the unticked boxes go dead rather than silently dropping a
  // choice, so the limit is visible before the request is refused.
  function syncFlaggerCap() {
    var schema = flaggerSchema();
    if (!schema) { return; }
    var full = state.flagMethods.length >= schema.max_methods;
    Array.prototype.forEach.call(
      $("flagger-methods").querySelectorAll("input[type=checkbox]"),
      function (box) { box.disabled = full && !box.checked; }
    );
    $("flagger-run").disabled = state.flagRunning || state.flagMethods.length === 0;
  }

  function setFlaggerStatus(kind, text) {
    var node = $("flagger-status");
    node.className = "status-cell status-" + kind;
    node.textContent = text;
  }

  function runFlagger() {
    var entry = shownEntry();
    if (!entry || state.flagRunning || !state.flagMethods.length) { return; }
    state.flagRunning = true;
    state.flagError = false;
    syncFlaggerCap();
    setFlaggerStatus("warn", "flagging…");

    var antenna = state.waterfallAntenna;
    var pol = state.waterfallPol;
    request("/api/flag", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        request: entry.request,
        methods: state.flagMethods.slice(),
        antenna: antenna,
        pol: pol
      })
    }).then(function (response) {
      entry.flag = { antenna: antenna, pol: pol, response: response, active: 0 };
      response.warnings.forEach(function (message) { showNotice("note", message); });
      setFlaggerStatus("ok", "done   " + fmt(response.wall_time_s, 2) + " s");
    }).catch(function (error) {
      // A refusal (a 422 on the method list or the request) is reported in
      // full through the page's own notice area, and left standing on the
      // control's status cell until something about the run changes.
      state.flagError = true;
      showNotice("error", error.message);
      setFlaggerStatus("error", "failed   " + shorten(error.message, 60));
    }).then(function () {
      state.flagRunning = false;
      syncFlaggerCap();
      renderFlagger();
      renderWaterfall();
    });
  }

  function renderFlaggerPills(flag) {
    var host = $("flagger-pills");
    while (host.firstChild) { host.removeChild(host.firstChild); }
    var methods = flag ? flag.response.methods : [];
    // One method needs no chooser; two do, and only ever one is painted.
    if (methods.length < 2) { return; }
    methods.forEach(function (method, index) {
      var pill = el("button", "mask-toggle");
      pill.type = "button";
      pill.setAttribute("aria-pressed", String(index === flag.active));
      pill.appendChild(document.createTextNode(method.label));
      pill.addEventListener("click", function () {
        flag.active = index;
        renderFlaggerPills(flag);
        renderWaterfall();
      });
      host.appendChild(pill);
    });
  }

  function renderFlaggerLegend(flag) {
    var host = $("flagger-legend");
    while (host.firstChild) { host.removeChild(host.firstChild); }
    if (!flag) { return; }
    [["caught", "caught — interference the method flagged"],
     ["missed", "missed — interference it left in"],
     ["false_alarm", "false alarm — clean data it threw away"]].forEach(function (pair) {
      var item = el("span", "legend-item");
      var swatch = el("span", "legend-swatch");
      swatch.style.background = "rgba(" + FLAG_COLOURS[pair[0]] + ", " + FLAG_ALPHA + ")";
      item.appendChild(swatch);
      item.appendChild(document.createTextNode(pair[1]));
      host.appendChild(item);
    });
  }

  function renderFlagger() {
    var schema = flaggerSchema();
    if (!schema || !schema.methods.length) { return; }
    var flag = activeFlag();
    renderFlaggerPills(flag);
    renderFlaggerLegend(flag);
    $("flagger-note").hidden = !flag;

    if (!flag) {
      $("flagger-grid-note").textContent = "";
      fillMetricTable($("flagger-metrics"), [], []);
      if (!state.flagRunning && !state.flagError) { setFlaggerStatus("info", "idle"); }
      return;
    }

    var grid = flag.response.grid;
    $("flagger-grid-note").textContent = "flagger grid: " + grid.chan_bins
      + " ch × " + grid.n_accumulations + " accumulations of " + grid.m
      + " samples (" + (grid.accumulation_s * 1000).toFixed(1) + " ms) · antenna "
      + flag.antenna + ", pol " + flag.pol;

    // One column per method, so the head-to-head is one glance -- including
    // the grid each one decided on, which is the honest caveat on the rest
    // of the column.
    var methods = flag.response.methods;
    var headers = ["metric"].concat(methods.map(function (method) {
      return method.label;
    }));
    var rows = [
      ["precision", function (s) { return score(s.precision); }],
      ["recall", function (s) { return score(s.recall); }],
      ["F1", function (s) { return score(s.f1); }],
      ["false-positive rate", function (s) { return score(s.false_positive_rate); }],
      ["truth occupancy", function (s) {
        return s.truth_occupancy === null || s.truth_occupancy === undefined
          ? "—"
          : (s.truth_occupancy * 100).toFixed(2) + "%";
      }]
    ].map(function (row) {
      return [row[0]].concat(methods.map(function (method) {
        return row[1](method.scores);
      }));
    });
    rows.push(["decides on"].concat(methods.map(function (method) {
      return method.grid;
    })));
    fillMetricTable($("flagger-metrics"), headers, rows);
  }

  /* --------------------------------------- 10. the mock observatory */

  /* A simulated day of drift scanning, and a live horizon chart.
   *
   * The day is a background job on the server: this side posts the setup,
   * polls for progress, and pulls each frame's little image as it lands.
   * Three rules shape everything below.
   *
   * **One colour scale for the whole day.** Frames are drawn against the
   * brightest pixel of the day so far, so a frame going bright means the
   * sky went bright, not that the scale moved. The scale only ever grows,
   * and the caption says what it is.
   *
   * **The timeline is the instrument.** Twenty-four hours of this strip in
   * one band: night shaded from the real solar altitude, a mark where each
   * catalogue source transits, ticks for satellite passes, and a playhead.
   * Clicking it scrubs. Everything else on the tab stays quiet.
   *
   * **Motion is opt-in.** With `prefers-reduced-motion` set, a finished day
   * never starts playing by itself -- but Play still plays, because the
   * preference is about surprise, not about capability.
   */

  var DAY_FPS = 6;
  var DAY_POLL_MS = 700;
  var FRAME_FETCH_LIMIT = 6;   // frame images in flight at once
  var SKY_POLL_MS = 10000;
  var TIMELINE_DEBOUNCE_MS = 250;

  // Named strips the declination box offers as one click each. Cygnus A's
  // is a fixed catalogue declination; the Sun's moves with the date, so it
  // is filled in from the timeline the server sends back.
  var CYG_A_DEC_DEG = 40.734;

  function reduceMotion() {
    return Boolean(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  }

  function pad2(value) { return (value < 10 ? "0" : "") + value; }

  // "2026-08-05T13:05:50.892" -> "13:05". The server speaks ISO UTC
  // throughout and the page never converts to local time: a drift scan is
  // indexed by sidereal time, and mixing in a local clock would invite the
  // reader to compare two different days.
  function utcClock(isot) {
    if (!isot) { return "--:--"; }
    return String(isot).slice(11, 16);
  }

  function fractionClock(fraction) {
    var minutes = Math.round(clamp(fraction, 0, 1) * 1440);
    return pad2(Math.floor(minutes / 60) % 24) + ":" + pad2(minutes % 60);
  }

  function todayUtc() {
    var now = new Date();
    return now.getUTCFullYear() + "-" + pad2(now.getUTCMonth() + 1) + "-" + pad2(now.getUTCDate());
  }

  function obsSite() {
    return state.site || (state.defaults && state.defaults.array) || null;
  }

  // The element set of the setup's satellite source, if it has one, so the
  // timeline can mark that satellite's passes through the strip.
  function setupTleText() {
    var satellite = state.rfiSources.filter(function (source) {
      return source.type === "satellite";
    })[0];
    if (!satellite) { return ""; }
    if (satellite.tle_source === "custom") { return satellite.tle_text || ""; }
    return (state.defaults && state.defaults.sample_tle) || "";
  }

  function resetObservatory() {
    var site = obsSite();
    state.obs = {
      dec_deg: site ? site.latitude_deg : 0,
      date: todayUtc(),
      frames: 96,
      resolution: "coarse",
      carry: false,
      jobId: null,
      total: 0,
      done: 0,
      failed: 0,
      jobState: "idle",
      meta: [],            // per-frame metadata, index-aligned
      images: [],          // per-frame image grids, index-aligned, null until fetched
      inFlight: 0,
      cursor: 0,
      playing: false,
      scaleMax: null,
      scaleSoft: null,
      autoplayed: false,
      fieldOfView: 0.04,
      timeline: null,
      timelineKey: "",
      pollTimer: null,
      playTimer: null,
      timelineTimer: null,
      error: null
    };
  }

  /* --- controls ------------------------------------------------------ */

  function renderDecChips() {
    var host = $("day-dec-chips");
    while (host.firstChild) { host.removeChild(host.firstChild); }
    var site = obsSite();
    var sunDec = state.obs.timeline ? state.obs.timeline.sun.dec_deg : null;
    var chips = [
      ["Zenith strip", site ? site.latitude_deg : null],
      ["Cyg A's strip", CYG_A_DEC_DEG],
      ["Sun's strip today", sunDec]
    ];
    chips.forEach(function (entry) {
      var value = entry[1];
      var label = entry[0];
      if (value !== null && value !== undefined) {
        label += " (" + (value >= 0 ? "+" : "") + value.toFixed(1) + "°)";
      }
      var chip = el("button", "chip", label);
      chip.type = "button";
      if (value === null || value === undefined) {
        chip.disabled = true;
        chip.title = "Build or reload the timeline to learn where the Sun is on this date";
      } else {
        chip.addEventListener("click", function () {
          state.obs.dec_deg = Number(value.toFixed(3));
          $("day-dec").value = state.obs.dec_deg;
          onDayControlsChanged();
        });
      }
      host.appendChild(chip);
    });
  }

  // Two hints that are really one physics statement: the sky drifts through
  // the field in a fixed number of minutes, and a frame cadence coarser
  // than that can miss a source entirely. Said in the place the user
  // controls it, and marked as a warning only when it is actually true.
  function renderCadenceHint() {
    var obs = state.obs;
    if (!obs) { return; }
    var hint = $("day-cadence");
    var minutesPerFrame = 1440 / Math.max(1, obs.frames);
    var crossing = obs.timeline ? obs.timeline.field_crossing_minutes : null;
    var text = obs.frames + " frames — one every " + minutesPerFrame.toFixed(0) + " min.";
    hint.classList.remove("field-hint-warn");
    if (crossing) {
      text += " A source crosses this strip in " + crossing.toFixed(0) + " min.";
      if (minutesPerFrame > crossing) {
        text += " Raise the frames to " + Math.min(288, Math.ceil(1440 / crossing))
          + " to be sure of catching one.";
        hint.classList.add("field-hint-warn");
      }
    }
    hint.textContent = text;
  }

  function renderResolutionNote() {
    var note = $("day-resolution-note");
    if (state.obs.resolution === "fine") {
      note.textContent = "Every frame is a full run at " + state.sim.n_chan + " channels and "
        + state.sim.n_blocks + " integrations — minutes, not seconds.";
    } else {
      note.textContent = "Fewer channels and one integration: noisier, same source positions.";
    }
  }

  function renderDayControls() {
    var obs = state.obs;
    if (!obs) { return; }
    $("day-dec").value = obs.dec_deg;
    $("day-date").value = obs.date;
    $("day-frames").value = obs.frames;
    $("day-resolution").value = obs.resolution;
    $("day-carry").checked = obs.carry;
    renderDecChips();
    renderCadenceHint();
    renderResolutionNote();
  }

  function setDayStatus(kind, text) {
    var cell = $("day-status");
    cell.className = "status-cell status-" + kind;
    cell.textContent = text;
  }

  function showDayError(text) {
    state.obs.error = text || null;
    $("day-error").hidden = !text;
    if (text) { $("day-error-text").textContent = text; }
  }

  /* --- the timeline data --------------------------------------------- */

  function timelineKey() {
    var site = obsSite() || {};
    return [state.obs.date, state.obs.dec_deg, site.latitude_deg, site.longitude_deg,
      site.height_m, setupTleText().length].join("|");
  }

  function refreshTimeline() {
    var site = obsSite();
    if (!site || !state.obs) { return; }
    var key = timelineKey();
    if (key === state.obs.timelineKey) { return; }
    state.obs.timelineKey = key;
    var query = "?date=" + encodeURIComponent(state.obs.date)
      + "&dec_deg=" + encodeURIComponent(state.obs.dec_deg)
      + "&latitude_deg=" + encodeURIComponent(site.latitude_deg)
      + "&longitude_deg=" + encodeURIComponent(site.longitude_deg)
      + "&height_m=" + encodeURIComponent(site.height_m)
      + "&tle_text=" + encodeURIComponent(setupTleText());
    request("/api/observatory/timeline" + query).then(function (timeline) {
      state.obs.timeline = timeline;
      renderDecChips();
      renderCadenceHint();
      drawTimeline();
    }).catch(function (error) {
      state.obs.timelineKey = "";
      showDayError("The timeline could not be worked out: " + error.message);
    });
  }

  function scheduleTimeline() {
    window.clearTimeout(state.obs.timelineTimer);
    state.obs.timelineTimer = window.setTimeout(refreshTimeline, TIMELINE_DEBOUNCE_MS);
  }

  function onDayControlsChanged() {
    renderDecChips();
    renderCadenceHint();
    renderResolutionNote();
    scheduleTimeline();
  }

  /* --- building the day ---------------------------------------------- */

  function stopDayTimers() {
    window.clearTimeout(state.obs.pollTimer);
    window.clearInterval(state.obs.playTimer);
    state.obs.pollTimer = null;
    state.obs.playTimer = null;
  }

  function buildDay() {
    var obs = state.obs;
    stopDayTimers();
    obs.playing = false;
    showDayError(null);
    obs.jobId = null;
    obs.meta = [];
    obs.images = [];
    obs.inFlight = 0;
    obs.cursor = 0;
    obs.done = 0;
    obs.failed = 0;
    obs.scaleMax = null;
    obs.scaleSoft = null;
    obs.autoplayed = false;
    obs.total = obs.frames;
    obs.jobState = "building";

    $("day-build").disabled = true;
    $("day-cancel").hidden = false;
    $("day-progress").hidden = false;
    setDayStatus("warn", "Building the day — 0 of " + obs.frames + " frames");
    renderMovie();

    request("/api/observatory/day", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        setup: buildRequest(),
        date: obs.date,
        pointing_dec_deg: obs.dec_deg,
        n_frames: Math.round(obs.frames),
        resolution: obs.resolution,
        carry_setup_sources: Boolean(obs.carry)
      })
    }).then(function (started) {
      obs.jobId = started.id;
      obs.total = started.total;
      pollDay();
    }).catch(function (error) {
      obs.jobState = "failed";
      $("day-build").disabled = false;
      $("day-cancel").hidden = true;
      $("day-progress").hidden = true;
      setDayStatus("error", "could not start");
      showDayError(error.message);
    });
  }

  // A finished day starts playing once there is something to play -- which
  // is a moment after the job reports "done", because the frame images are
  // still arriving. Never when the reader has asked for reduced motion,
  // and never twice for the same day.
  function maybeAutoplay() {
    var obs = state.obs;
    if (obs.autoplayed || obs.jobState !== "done" || obs.failed) { return; }
    if (reduceMotion() || readyFrames() < 2) { return; }
    obs.autoplayed = true;
    setPlaying(true);
  }

  function cancelDay() {
    if (!state.obs.jobId) { return; }
    request("/api/observatory/day/" + state.obs.jobId + "/cancel", { method: "POST" })
      .catch(function () { /* a day that is already gone needs no stopping */ });
    setDayStatus("info", "stopping");
  }

  function pollDay() {
    var obs = state.obs;
    if (!obs.jobId) { return; }
    request("/api/observatory/day/" + obs.jobId).then(function (status) {
      if (obs.jobId !== status.id) { return; }
      obs.total = status.total;
      obs.done = status.done;
      obs.failed = status.failed;
      obs.jobState = status.state;
      obs.meta = status.frames;
      // The scale only ever grows, so a frame drawn early is not redrawn
      // darker later on -- it is redrawn against a scale that has room for
      // the brightest thing the day turned out to hold.
      if (status.scale_max_jy !== null && status.scale_max_jy !== undefined) {
        obs.scaleMax = Math.max(obs.scaleMax || 0, status.scale_max_jy);
      }
      obs.scaleSoft = status.scale_soft_jy || null;
      if (obs.images.length !== obs.total) {
        obs.images = new Array(obs.total);
      }

      $("day-progress-fill").style.width =
        (obs.total ? (100 * obs.done / obs.total) : 0).toFixed(1) + "%";
      fetchPendingFrames();
      renderMovie();
      drawTimeline();

      if (status.state === "building") {
        setDayStatus("warn", "Building the day — " + obs.done + " of " + obs.total + " frames");
        obs.pollTimer = window.setTimeout(pollDay, DAY_POLL_MS);
        return;
      }

      $("day-build").disabled = false;
      $("day-cancel").hidden = true;
      $("day-progress").hidden = true;
      if (status.state === "failed") {
        setDayStatus("error", "the day could not be built");
        showDayError(status.error || "the frame pool stopped unexpectedly");
      } else if (status.state === "cancelled" || status.state === "cancelling") {
        setDayStatus("info", "Stopped — " + obs.done + " of " + obs.total + " frames built");
      } else if (obs.failed) {
        setDayStatus("warn", "Built with " + plural(obs.failed, "failed frame") + " — press play");
      } else {
        setDayStatus("ok", "Built — press play");
      }
      updateMovieControls();
      maybeAutoplay();
    }).catch(function (error) {
      $("day-build").disabled = false;
      $("day-cancel").hidden = true;
      $("day-progress").hidden = true;
      setDayStatus("error", "lost track of this day");
      showDayError(error.message);
    });
  }

  // Frame images are pulled one request each, a few at a time: the status
  // poll is deliberately imageless, so this is where the pictures arrive.
  function fetchPendingFrames() {
    var obs = state.obs;
    for (var index = 0; index < obs.total; index += 1) {
      if (obs.inFlight >= FRAME_FETCH_LIMIT) { return; }
      var meta = obs.meta[index];
      if (!meta || meta.error || obs.images[index] !== undefined) { continue; }
      obs.images[index] = null;             // claimed, not yet arrived
      obs.inFlight += 1;
      fetchFrame(obs.jobId, index);
    }
  }

  function fetchFrame(jobId, index) {
    request("/api/observatory/day/" + jobId + "/frame/" + index).then(function (frame) {
      if (state.obs.jobId !== jobId) { return; }
      state.obs.images[index] = frame.pending ? undefined : frame.image;
      if (frame.field_of_view_rad) { state.obs.fieldOfView = frame.field_of_view_rad; }
      if (index === state.obs.cursor) { renderMovie(); }
      drawTimeline();
    }).catch(function () {
      if (state.obs.jobId === jobId) { state.obs.images[index] = undefined; }
    }).then(function () {
      state.obs.inFlight = Math.max(0, state.obs.inFlight - 1);
      if (state.obs.jobId === jobId) {
        fetchPendingFrames();
        maybeAutoplay();
      }
    });
  }

  /* --- playback ------------------------------------------------------ */

  function readyFrames() {
    return state.obs.images.filter(function (image) { return Boolean(image); }).length;
  }

  function setPlaying(on) {
    var obs = state.obs;
    window.clearInterval(obs.playTimer);
    obs.playing = Boolean(on) && readyFrames() > 1;
    if (obs.playing) {
      obs.playTimer = window.setInterval(function () { stepFrame(1); }, 1000 / DAY_FPS);
    }
    updateMovieControls();
  }

  function stepFrame(step) {
    var obs = state.obs;
    if (!obs.total) { return; }
    // Skip past frames that failed or have not arrived, so playback never
    // stalls on a hole; if nothing is ready, stop rather than spin.
    for (var tried = 0; tried < obs.total; tried += 1) {
      obs.cursor = ((obs.cursor + step) % obs.total + obs.total) % obs.total;
      if (obs.images[obs.cursor]) { renderMovie(); return; }
    }
    setPlaying(false);
  }

  function goToFrame(index) {
    state.obs.cursor = clamp(Math.round(index), 0, Math.max(0, state.obs.total - 1));
    renderMovie();
  }

  function updateMovieControls() {
    var ready = readyFrames();
    var play = $("movie-play");
    play.disabled = ready < 2;
    play.textContent = state.obs.playing ? "Pause" : "Play";
    play.setAttribute("aria-pressed", String(state.obs.playing));
    $("movie-back").disabled = ready < 1;
    $("movie-forward").disabled = ready < 1;
  }

  /* --- drawing the movie --------------------------------------------- */

  /* One fixed brightness mapping for the whole day.
   *
   * A day of drift scanning spans six orders of magnitude: an empty field
   * is a couple of millijanskys of noise, Cygnus A crossing it is fifteen
   * hundred janskys. Linear, and either the source is the only thing you
   * ever see or the noise saturates. So the day is drawn through an
   * arcsinh stretch -- linear near zero, logarithmic far from it, the
   * ordinary astronomical answer -- with its soft point at what a typical
   * frame's brightest pixel actually is.
   *
   * It is still one mapping for the whole day, computed once from the
   * finished frames: a frame that looks brighter *is* brighter. Only the
   * spacing of the greys between the two ends is non-linear, and the
   * caption says so.
   */
  function asinh(value) {
    return Math.log(value + Math.sqrt(value * value + 1));
  }

  function dayStretch(image, soft, vmax) {
    var top = asinh(vmax / soft) || 1;
    return image.map(function (row) {
      return row.map(function (value) {
        return asinh(Math.max(value, 0) / soft) / top;
      });
    });
  }

  function lmAxis(nPix, fieldOfView) {
    var axis = [];
    for (var i = 0; i < nPix; i += 1) {
      axis.push(nPix === 1 ? 0 : -0.5 * fieldOfView + fieldOfView * (i / (nPix - 1)));
    }
    return axis;
  }

  function renderMovie() {
    var obs = state.obs;
    if (!obs) { return; }
    var image = obs.images[obs.cursor];
    var meta = obs.meta[obs.cursor];
    $("movie-empty").hidden = Boolean(image);
    updateMovieControls();
    drawTimeline();

    if (!image) {
      $("movie-counter").textContent = obs.total
        ? "frame " + (obs.cursor + 1) + " of " + obs.total + " — not built yet"
        : "";
      $("movie-meta").textContent = meta && meta.error ? "This frame failed: " + meta.error : "";
      $("movie-scale").textContent = "";
      return;
    }

    var axis = lmAxis(image.length, obs.fieldOfView);
    var vmax = obs.scaleMax || 1;
    // The soft point only exists once a frame has finished; without one
    // the day is drawn linearly, which is right for a day that is one
    // frame long.
    var soft = obs.scaleSoft;
    var stretched = soft ? dayStretch(image, soft, vmax) : image;
    var top = soft ? asinh(vmax / soft) || 1 : 1;
    drawHeatmap($("movie-canvas"), {
      values: stretched,
      vmin: 0,
      vmax: soft ? 1 : vmax,
      xLow: axis[0],
      xHigh: axis[axis.length - 1],
      yLow: axis[0],
      yHigh: axis[axis.length - 1],
      xLabel: "l (direction cosine)",
      yLabel: "m (direction cosine)",
      barLabel: "Jy",
      xFormat: function (v) { return v.toFixed(3); },
      yFormat: function (v) { return v.toFixed(3); },
      // The colour bar is labelled in janskys even though the values it
      // was handed are stretched: its two ends are undone by the inverse
      // of the stretch, so the reader sees the physical scale.
      barFormat: function (v) {
        var jy = soft ? soft * Math.sinh(v * top) : v;
        return jy.toFixed(jy < 10 ? 3 : 0);
      },
      truthMarkers: (meta && meta.sources || []).map(function (source) {
        return { x: source.l, y: source.m, label: source.name };
      }),
      readCell: function (row, col) {
        return "l " + axis[col].toFixed(4) + "\nm " + axis[row].toFixed(4)
          + "\n" + image[row][col].toFixed(3) + " Jy";
      }
    });

    $("movie-scale").textContent = "scale 0 – " + vmax.toFixed(vmax < 10 ? 3 : 0)
      + " Jy" + (soft ? ", arcsinh stretch" : "") + " (fixed for the day)";
    $("movie-counter").textContent = "frame " + (obs.cursor + 1) + " of " + obs.total;

    var parts = [];
    if (meta) {
      parts.push(utcClock(meta.utc) + " UTC");
      parts.push("LST " + fractionClock(((meta.lst_deg % 360) + 360) % 360 / 360));
      parts.push("pointing " + meta.ra_deg.toFixed(2) + "°, " + meta.dec_deg.toFixed(2) + "°");
      if (meta.sources.length) {
        parts.push(meta.sources.map(function (source) {
          return source.name + " in field: " + source.flux_jy.toFixed(0) + " Jy";
        }).join(" · "));
      } else {
        parts.push("empty field");
      }
    }
    $("movie-meta").textContent = parts.join("   ");
  }

  /* --- drawing the timeline ------------------------------------------ */

  // Night is shaded from the real solar altitude rather than from sunrise
  // and sunset alone, so civil, nautical and astronomical twilight appear
  // as the gradient they are. Above the horizon is the page's own day
  // colour; 18 degrees below it is as dark as the band gets.
  function nightColour(altitudeDeg) {
    if (altitudeDeg > 0) { return "#eef1f6"; }
    var depth = clamp(-altitudeDeg / 18, 0, 1);
    var top = [222, 228, 236];
    var bottom = [23, 26, 30];
    return "rgb(" + top.map(function (value, i) {
      return Math.round(value + (bottom[i] - value) * depth);
    }).join(",") + ")";
  }

  function drawTimeline() {
    var svg = $("timeline");
    if (!state.obs) { return; }
    var timeline = state.obs.timeline;
    while (svg.firstChild) { svg.removeChild(svg.firstChild); }
    var width = Math.max(320, Math.round(svg.getBoundingClientRect().width));
    var height = 92;
    svg.setAttribute("viewBox", "0 0 " + width + " " + height);
    if (!timeline) { return; }

    var bandY = 24;
    var bandH = 30;
    var laneY = bandY + bandH;
    var laneH = 16;

    function x(fraction) { return clamp(fraction, 0, 1) * width; }

    // The sky band: one thin rectangle per sample of the solar altitude.
    var altitudes = timeline.sun.altitude_deg;
    var fractions = timeline.sun.sample_fractions;
    for (var i = 0; i < altitudes.length - 1; i += 1) {
      var left = x(fractions[i]);
      var right = x(fractions[i + 1]);
      svg.appendChild(svgNode("rect", {
        x: left, y: bandY, width: Math.max(1, right - left + 0.6), height: bandH,
        fill: nightColour((altitudes[i] + altitudes[i + 1]) / 2)
      }));
    }
    svg.appendChild(svgNode("rect", {
      x: 0.5, y: bandY + 0.5, width: width - 1, height: bandH - 1,
      fill: "none", class: "tl-rule"
    }));

    // Which frames exist: a hairline per built frame along the top of the
    // band, so the band doubles as the progress bar it already is.
    state.obs.meta.forEach(function (meta, index) {
      if (!state.obs.total) { return; }
      var at = x((index + 0.5) / state.obs.total);
      svg.appendChild(svgNode("line", {
        x1: at, y1: bandY + 1, x2: at, y2: bandY + 5,
        class: meta && !meta.error ? "tl-frame-tick" : "tl-frame-tick-pending"
      }));
    });

    // Sunrise and sunset: a small disc on the horizon line of the band.
    function sunGlyph(fraction, rising) {
      var at = x(fraction);
      var group = svgNode("g", {});
      group.appendChild(svgNode("circle", {
        cx: at, cy: bandY + bandH / 2, r: 3.4, fill: "#fdcb6e",
        stroke: "#d4a017", "stroke-width": 0.8
      }));
      group.appendChild(svgNode("line", {
        x1: at - 6, y1: bandY + bandH / 2, x2: at + 6, y2: bandY + bandH / 2,
        stroke: "#d4a017", "stroke-width": 0.8
      }));
      var label = svgNode("text", {
        x: at, y: bandY - 4, class: "tl-label", "text-anchor": "middle"
      });
      label.textContent = (rising ? "sunrise " : "sunset ") + fractionClock(fraction);
      group.appendChild(label);
      svg.appendChild(group);
    }
    timeline.sun.sunrise.forEach(function (fraction) { sunGlyph(fraction, true); });
    timeline.sun.sunset.forEach(function (fraction) { sunGlyph(fraction, false); });

    // Satellite passes, as thin ticks in the lane under the band.
    (timeline.satellite.passes || []).forEach(function (pass) {
      svg.appendChild(svgNode("rect", {
        x: x(pass.start), y: laneY + 1,
        width: Math.max(1.4, x(pass.end) - x(pass.start)), height: laneH - 2,
        fill: "#00cec9", opacity: 0.75
      }));
    });

    // Calibrator transits. A source this strip never sees is drawn faint
    // and dashed rather than left out: "never in this strip" is a fact
    // about the declination the user chose, and hiding it hides the fix.
    var placed = [];
    timeline.sources.forEach(function (source) {
      source.transits.forEach(function (fraction) {
        var at = x(fraction);
        svg.appendChild(svgNode("line", {
          x1: at, y1: bandY, x2: at, y2: laneY + laneH,
          class: source.in_field ? "tl-marker" : "tl-marker-faint"
        }));
        if (!source.in_field) { return; }
        // Labels are dropped rather than overprinted when two transits
        // land within a few pixels of each other.
        var collides = placed.some(function (other) { return Math.abs(other - at) < 34; });
        if (collides) { return; }
        placed.push(at);
        var label = svgNode("text", {
          x: clamp(at, 16, width - 16), y: laneY + laneH + 11,
          class: "tl-marker-label", "text-anchor": "middle"
        });
        label.textContent = source.name;
        svg.appendChild(label);
      });
    });

    // Hour rule.
    for (var hour = 0; hour <= 24; hour += 3) {
      var hx = x(hour / 24);
      svg.appendChild(svgNode("line", {
        x1: hx, y1: laneY + laneH, x2: hx, y2: laneY + laneH + 3, class: "tl-rule"
      }));
      var tick = svgNode("text", {
        x: clamp(hx, 10, width - 10), y: height - 3, class: "tl-label", "text-anchor": "middle"
      });
      tick.textContent = pad2(hour % 24) + ":00";
      svg.appendChild(tick);
    }

    // The playhead, only once there is a day to point into.
    if (state.obs.total) {
      // Kept a few pixels inside the band so the arrowhead at the top of
      // the first and last frames is not half cut off by the edge.
      var head = clamp(x((state.obs.cursor + 0.5) / state.obs.total), 5, width - 5);
      svg.appendChild(svgNode("line", {
        x1: head, y1: bandY - 2, x2: head, y2: laneY + laneH + 3, class: "tl-playhead"
      }));
      svg.appendChild(svgNode("path", {
        d: "M " + (head - 4) + " " + (bandY - 8) + " L " + (head + 4) + " " + (bandY - 8)
          + " L " + head + " " + (bandY - 2) + " Z",
        class: "tl-playhead-head"
      }));
    }

    var missing = timeline.sources.filter(function (source) { return !source.in_field; });
    var legend = ["night shaded from the real solar altitude"];
    if (missing.length) {
      legend.push(missing.map(function (source) { return source.name; }).join(", ")
        + ": never in this strip");
    }
    if (timeline.satellite.configured) {
      var passes = (timeline.satellite.passes || []).length;
      legend.push(timeline.satellite.note
        || (passes === 1 ? "one satellite pass" : passes + " satellite passes")
          + " through the field, in cyan");
    }
    $("timeline-legend").textContent = legend.join(" · ");
  }

  function scrubTimeline(event) {
    if (!state.obs.total) { return; }
    var svg = $("timeline");
    var rect = svg.getBoundingClientRect();
    var fraction = clamp((event.clientX - rect.left) / rect.width, 0, 1);
    setPlaying(false);
    goToFrame(Math.floor(fraction * state.obs.total));
  }

  /* --- the live monitor ---------------------------------------------- */

  var SKY_RADIUS = 150;      // internal units; the SVG scales to its box
  var AIRCRAFT_LABELS = 6;  // callsigns drawn, nearest first; the rest carry a tooltip

  // Zenith at the centre, horizon at the rim, altitude linear in radius --
  // the ordinary "stereographic-looking" horizon chart, which is really an
  // equidistant projection. Azimuth runs North through East, matching the
  // rest of the package.
  function skyPoint(altitudeDeg, azimuthDeg) {
    var radius = SKY_RADIUS * clamp((90 - altitudeDeg) / 90, 0, 1);
    var angle = rad(azimuthDeg);
    return [SKY_RADIUS + radius * Math.sin(angle), SKY_RADIUS - radius * Math.cos(angle)];
  }

  function drawSkyChart() {
    var svg = $("sky-chart");
    while (svg.firstChild) { svg.removeChild(svg.firstChild); }
    var box = 2 * SKY_RADIUS + 34;
    svg.setAttribute("viewBox", "-17 -17 " + box + " " + box);

    [30, 60].forEach(function (altitude) {
      svg.appendChild(svgNode("circle", {
        cx: SKY_RADIUS, cy: SKY_RADIUS,
        r: SKY_RADIUS * (90 - altitude) / 90, class: "sky-ring-dashed"
      }));
      var tick = svgNode("text", {
        x: SKY_RADIUS + 3, y: SKY_RADIUS - SKY_RADIUS * (90 - altitude) / 90 - 2,
        class: "sky-tick"
      });
      tick.textContent = altitude + "°";
      svg.appendChild(tick);
    });
    svg.appendChild(svgNode("circle", {
      cx: SKY_RADIUS, cy: SKY_RADIUS, r: SKY_RADIUS, class: "sky-ring"
    }));

    [["N", 0], ["E", 90], ["S", 180], ["W", 270]].forEach(function (entry) {
      var point = skyPoint(-6, entry[1]);
      var label = svgNode("text", { x: point[0], y: point[1] + 3.5, class: "sky-cardinal" });
      label.textContent = entry[0];
      svg.appendChild(label);
    });

    var sky = state.sky.data;
    if (!sky) { return; }

    function labelled(point, text, className) {
      var label = svgNode("text", {
        x: point[0] + 6, y: point[1] + 3, class: className || "sky-glyph-label"
      });
      label.textContent = text;
      svg.appendChild(label);
    }

    // Aircraft first, so the ephemeris never hides behind traffic: a
    // heading-rotated chevron with its callsign. Only the nearest few are
    // labelled -- a busy corridor puts thirty of them along the rim, and
    // thirty overlapping callsigns say less than none.
    (sky.aircraft || []).forEach(function (aircraft, index) {
      if (aircraft.altitude_deg <= 0) { return; }
      var point = skyPoint(aircraft.altitude_deg, aircraft.azimuth_deg);
      var heading = aircraft.heading_deg === null ? 0 : aircraft.heading_deg;
      var glyph = svgNode("path", {
        d: "M 0 -5 L 4 4 L 0 1.5 L -4 4 Z",
        class: "sky-aircraft",
        transform: "translate(" + point[0].toFixed(2) + "," + point[1].toFixed(2)
          + ") rotate(" + heading.toFixed(1) + ")"
      });
      var title = svgNode("title", {});
      title.textContent = (aircraft.callsign || aircraft.id) + " — "
        + fmt(aircraft.altitude_deg) + "° up, " + fmt(aircraft.range_km) + " km away";
      glyph.appendChild(title);
      svg.appendChild(glyph);
      if (aircraft.callsign && index < AIRCRAFT_LABELS) { labelled(point, aircraft.callsign); }
    });

    (sky.satellites || []).forEach(function (satellite) {
      var point = skyPoint(satellite.altitude_deg, satellite.azimuth_deg);
      svg.appendChild(svgNode("rect", {
        x: point[0] - 2.5, y: point[1] - 2.5, width: 5, height: 5,
        class: "sky-satellite", transform: "rotate(45 " + point[0] + " " + point[1] + ")"
      }));
      labelled(point, satellite.name);
    });

    (sky.sources || []).forEach(function (source) {
      if (source.altitude_deg <= 0) { return; }
      var point = skyPoint(source.altitude_deg, source.azimuth_deg);
      svg.appendChild(svgNode("circle", {
        cx: point[0], cy: point[1], r: 3, class: "sky-source"
      }));
      labelled(point, source.name);
    });

    if (sky.moon.up) {
      var moon = skyPoint(sky.moon.altitude_deg, sky.moon.azimuth_deg);
      svg.appendChild(svgNode("circle", { cx: moon[0], cy: moon[1], r: 4.5, class: "sky-moon" }));
      labelled(moon, "Moon");
    }
    if (sky.sun.up) {
      var sun = skyPoint(sky.sun.altitude_deg, sky.sun.azimuth_deg);
      svg.appendChild(svgNode("circle", { cx: sun[0], cy: sun[1], r: 6, class: "sky-sun" }));
      labelled(sun, "Sun");
    }
  }

  function renderSkyLayers() {
    var host = $("sky-layers");
    while (host.firstChild) { host.removeChild(host.firstChild); }
    var sky = state.sky.data;
    if (!sky) { return; }
    Object.keys(sky.layers).forEach(function (name) {
      var layer = sky.layers[name];
      var cell = el("span", "status-cell status-" + (layer.status === "ok" ? "ok" : "warn"),
        name + " — " + layer.note);
      if (layer.detail) { cell.title = layer.detail; }
      host.appendChild(cell);
    });
  }

  function renderSkyTable() {
    var sky = state.sky.data;
    if (!sky) { return; }
    var rows = [];
    rows.push(["Sun", sky.sun.up ? fmt(sky.sun.altitude_deg) + "°" : "below horizon",
      sky.sun.up ? fmt(sky.sun.azimuth_deg) + "°" : "—", fmt(sky.sun.flux_jy, 0) + " Jy"]);
    rows.push(["Moon", sky.moon.up ? fmt(sky.moon.altitude_deg) + "°" : "below horizon",
      sky.moon.up ? fmt(sky.moon.azimuth_deg) + "°" : "—", "—"]);
    (sky.sources || []).forEach(function (source) {
      rows.push([source.name,
        source.up ? fmt(source.altitude_deg) + "°" : "below horizon",
        source.up ? fmt(source.azimuth_deg) + "°" : "—",
        fmt(source.flux_jy, 0) + " Jy"]);
    });
    fillMetricTable($("sky-table"), ["object", "altitude", "azimuth", "L-band flux"], rows);
  }

  function renderSkyNow() {
    drawSkyChart();
    renderSkyLayers();
    renderSkyTable();
    var sky = state.sky.data;
    if (!sky) { return; }
    $("sky-updated").textContent = "last updated " + utcClock(sky.utc) + " UTC · LST "
      + fractionClock(sky.lst_deg / 360) + " · "
      + (sky.aircraft || []).length + " aircraft, " + (sky.satellites || []).length + " satellites";
  }

  function skyVisible() {
    return state.tab === "observatory" && document.visibilityState !== "hidden";
  }

  function pollSky() {
    if (!skyVisible()) { return; }
    var site = obsSite();
    var query = site
      ? "?latitude_deg=" + encodeURIComponent(site.latitude_deg)
        + "&longitude_deg=" + encodeURIComponent(site.longitude_deg)
        + "&height_m=" + encodeURIComponent(site.height_m)
      : "";
    request("/api/sky/now" + query).then(function (sky) {
      state.sky.data = sky;
      state.sky.error = null;
      renderSkyNow();
    }).catch(function (error) {
      state.sky.error = error.message;
      $("sky-updated").textContent = "the monitor could not be reached: " + error.message;
    });
  }

  // Polling stops the moment the tab or the page goes out of sight: this
  // is a live feed on somebody else's server, and a forgotten browser tab
  // must not keep asking for it.
  function syncSkyPolling() {
    if (skyVisible()) {
      if (!state.sky.timer) {
        state.sky.timer = window.setInterval(pollSky, SKY_POLL_MS);
      }
      pollSky();
      return;
    }
    window.clearInterval(state.sky.timer);
    state.sky.timer = null;
  }

  /* --- binding -------------------------------------------------------- */

  function bindObservatory() {
    $("day-dec").addEventListener("change", function (event) {
      var value = Number(event.target.value);
      state.obs.dec_deg = isFinite(value) ? clamp(value, -90, 90) : state.obs.dec_deg;
      event.target.value = state.obs.dec_deg;
      onDayControlsChanged();
    });
    $("day-date").addEventListener("change", function (event) {
      if (!/^\d{4}-\d{2}-\d{2}$/.test(event.target.value)) {
        event.target.value = state.obs.date;
        return;
      }
      state.obs.date = event.target.value;
      onDayControlsChanged();
    });
    $("day-frames").addEventListener("change", function (event) {
      var value = Math.round(Number(event.target.value));
      state.obs.frames = isFinite(value) ? clamp(value, 1, 288) : state.obs.frames;
      event.target.value = state.obs.frames;
      renderCadenceHint();
    });
    $("day-resolution").addEventListener("change", function (event) {
      state.obs.resolution = event.target.value === "fine" ? "fine" : "coarse";
      renderResolutionNote();
    });
    $("day-carry").addEventListener("change", function (event) {
      state.obs.carry = event.target.checked;
    });

    $("day-build").addEventListener("click", buildDay);
    $("day-cancel").addEventListener("click", cancelDay);

    $("movie-play").addEventListener("click", function () { setPlaying(!state.obs.playing); });
    $("movie-back").addEventListener("click", function () { setPlaying(false); stepFrame(-1); });
    $("movie-forward").addEventListener("click", function () { setPlaying(false); stepFrame(1); });

    $("movie-canvas").addEventListener("keydown", function (event) {
      if (event.key === " " || event.key === "Spacebar") {
        event.preventDefault();
        setPlaying(!state.obs.playing);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        setPlaying(false);
        stepFrame(1);
      } else if (event.key === "ArrowLeft") {
        event.preventDefault();
        setPlaying(false);
        stepFrame(-1);
      }
    });

    var timeline = $("timeline");
    timeline.addEventListener("click", scrubTimeline);
    timeline.addEventListener("keydown", function (event) {
      var step = event.key === "ArrowRight" ? 1 : (event.key === "ArrowLeft" ? -1 : 0);
      if (!step) { return; }
      event.preventDefault();
      setPlaying(false);
      stepFrame(step);
    });

    document.addEventListener("visibilitychange", syncSkyPolling);
    bindTooltip($("movie-canvas"));
  }

  /* ------------------------------------------- 11. tabs and navigation */

  /* Three tabs, Setup, Results and Mock Observatory, with the hash as the
   * single source of
   * truth: a reload or a bookmark lands where it left off, and the back
   * button walks the tabs. The topbar's run controls sit outside both
   * panels, so a run can be started from either.
   *
   * Canvases measure themselves when they draw, and a hidden panel measures
   * zero, so switching to Results redraws it rather than showing whatever
   * size it had when it was last visible.
   */
  var TABS = ["setup", "results", "observatory"];

  function tabFromHash() {
    var name = String(window.location.hash || "").replace("#", "");
    return TABS.indexOf(name) >= 0 ? name : "setup";
  }

  function showTab(name) {
    state.tab = TABS.indexOf(name) >= 0 ? name : "setup";
    TABS.forEach(function (candidate) {
      var button = $("tab-" + candidate);
      var view = $("view-" + candidate);
      var on = candidate === state.tab;
      button.classList.toggle("is-active", on);
      button.setAttribute("aria-selected", String(on));
      button.tabIndex = on ? 0 : -1;
      view.hidden = !on;
    });
    // The Setup sub-links are a table of contents for the Setup panel only.
    $("subnav").hidden = state.tab !== "setup";
    if (state.tab === "results") { renderResults(); }
    if (state.tab === "observatory") {
      renderDayControls();
      refreshTimeline();
      drawTimeline();
      renderMovie();
      renderSkyNow();
    }
    // The monitor only polls while it is the tab on screen; leaving stops
    // it, coming back starts it and asks once immediately.
    syncSkyPolling();
  }

  function goToTab(name) {
    if (window.location.hash === "#" + name) {
      showTab(name);
      return;
    }
    window.location.hash = "#" + name;   // the hashchange handler does the rest
  }

  function bindTabs() {
    TABS.forEach(function (name, index) {
      var button = $("tab-" + name);
      button.addEventListener("click", function () { goToTab(name); });
      button.addEventListener("keydown", function (event) {
        var step = event.key === "ArrowRight" ? 1 : (event.key === "ArrowLeft" ? -1 : 0);
        if (!step) { return; }
        event.preventDefault();
        var next = TABS[(index + step + TABS.length) % TABS.length];
        goToTab(next);
        $("tab-" + next).focus();
      });
    });
    window.addEventListener("hashchange", function () { showTab(tabFromHash()); });
    showTab(tabFromHash());
  }

  /* The Setup sub-links are a table of contents for one long panel:
     clicking one scrolls to its section, and the section you are actually
     looking at wears the active pill. The pill follows the *topmost*
     section still inside the band just under the header, which is the one a
     reader thinks of as "where I am". */
  function bindNav() {
    var links = Array.prototype.slice.call(document.querySelectorAll(".navlink-sub"));
    var sections = links.map(function (link) {
      return $(link.getAttribute("data-target"));
    });

    function setActive(link) {
      links.forEach(function (other) {
        if (other === link) {
          other.classList.add("is-active");
          other.setAttribute("aria-current", "true");
        } else {
          other.classList.remove("is-active");
          other.removeAttribute("aria-current");
        }
      });
    }

    links.forEach(function (link, index) {
      link.addEventListener("click", function (event) {
        var section = sections[index];
        if (!section) { return; }
        event.preventDefault();
        section.scrollIntoView({ behavior: "smooth", block: "start" });
        setActive(link);   // immediate; the observer confirms it on arrival
      });
    });

    setActive(links[0]);
    if (!window.IntersectionObserver) { return; }

    var onScreen = sections.map(function () { return false; });
    var observer = new window.IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        var index = sections.indexOf(entry.target);
        if (index >= 0) { onScreen[index] = entry.isIntersecting; }
      });
      for (var i = 0; i < onScreen.length; i += 1) {
        if (onScreen[i]) { setActive(links[i]); return; }
      }
      // Nothing in the band (only possible mid-fling): leave the pill be.
    }, { rootMargin: "-104px 0px -55% 0px", threshold: 0 });

    sections.forEach(function (section) {
      if (section) { observer.observe(section); }
    });
  }

  /* ------------------------------------------------------- 11. tooltips */

  function bindTooltip(canvas) {
    var tooltip = $("tooltip");
    canvas.addEventListener("mousemove", function (event) {
      var meta = canvas.plotSpec;
      if (!meta || !meta.spec.readCell) { tooltip.hidden = true; return; }
      var rect = canvas.getBoundingClientRect();
      var x = event.clientX - rect.left;
      var y = event.clientY - rect.top;
      var plot = meta.plot;
      if (x < plot.x || x > plot.x + plot.w || y < plot.y || y > plot.y + plot.h) {
        tooltip.hidden = true;
        return;
      }
      var col = clamp(Math.floor((x - plot.x) / plot.w * meta.cols), 0, meta.cols - 1);
      var rowFromTop = clamp(Math.floor((y - plot.y) / plot.h * meta.rows), 0, meta.rows - 1);
      var row = meta.rows - 1 - rowFromTop;
      tooltip.textContent = meta.spec.readCell(row, col);
      tooltip.hidden = false;
      tooltip.style.left = (event.clientX + 14) + "px";
      tooltip.style.top = (event.clientY + 14) + "px";
    });
    canvas.addEventListener("mouseleave", function () { $("tooltip").hidden = true; });
  }

  function bindUvTooltip() {
    var svg = $("uv-plot");
    var tooltip = $("tooltip");
    svg.addEventListener("mousemove", function (event) {
      var meta = svg.uvSpec;
      if (!meta) { return; }
      var rect = svg.getBoundingClientRect();
      var x = (event.clientX - rect.left) / rect.width * (svg.viewBox.baseVal.width || rect.width);
      var y = (event.clientY - rect.top) / rect.height
        * (svg.viewBox.baseVal.height || rect.height);
      var best = null;
      var bestDistance = 100;
      meta.points.forEach(function (point) {
        var px = meta.cx + point[0] / meta.limit * meta.half;
        var py = meta.cy - point[1] / meta.limit * meta.half;
        var distance = (px - x) * (px - x) + (py - y) * (py - y);
        if (distance < bestDistance) { bestDistance = distance; best = point; }
      });
      if (!best) { tooltip.hidden = true; return; }
      tooltip.textContent = "u " + best[0].toFixed(1) + " λ\nv "
        + best[1].toFixed(1) + " λ\nlength "
        + Math.sqrt(best[0] * best[0] + best[1] * best[1]).toFixed(1) + " λ";
      tooltip.hidden = false;
      tooltip.style.left = (event.clientX + 14) + "px";
      tooltip.style.top = (event.clientY + 14) + "px";
    });
    svg.addEventListener("mouseleave", function () { $("tooltip").hidden = true; });
  }

  /* ----------------------------------------------------- 12. wiring */

  function renderSiteMeta() {
    var site = state.site || defaultSite();
    var meta = $("site-meta");
    while (meta.firstChild) { meta.removeChild(meta.firstChild); }
    [["Layout", site.name || "custom"],
     ["Site latitude, longitude",
      site.latitude_deg.toFixed(3) + "°, " + site.longitude_deg.toFixed(3) + "°"],
     ["Height above sea level", site.height_m.toFixed(0) + " m"]].forEach(function (pair) {
      meta.appendChild(el("dt", null, pair[0]));
      meta.appendChild(el("dd", null, pair[1]));
    });
  }

  // Where the run points, in words, above the sky sources. The zenith of
  // the site at the start of the recording -- so it moves when another
  // layout, on another site, is loaded.
  function renderPointingHint() {
    var pointing = state.pointing;
    if (!pointing) { return; }
    $("pointing-hint").textContent =
      "The telescope points at RA " + fmt(pointing.ra_deg, 3) + "°, Dec "
      + fmt(pointing.dec_deg, 3) + "° — the zenith at the start of the recording, which"
      + " the simulator tracks for the whole run. Keep sources within ±"
      + fmt(pointing.field_half_width_deg, 1) + "° of it: that is the imaged field of view.";
  }

  function refreshPointing() {
    var site = state.site || defaultSite();
    var query = "?latitude_deg=" + encodeURIComponent(site.latitude_deg)
      + "&longitude_deg=" + encodeURIComponent(site.longitude_deg)
      + "&height_m=" + encodeURIComponent(site.height_m);
    return request("/api/pointing" + query).then(function (pointing) {
      state.pointing = pointing;
      renderPointingHint();
      renderSkyCards();
    }).catch(function (error) {
      showNotice("error", "Could not work out where the telescope points: " + error.message);
    });
  }

  function bindControls() {
    var presetSelect = $("preset");
    PRESETS.forEach(function (preset) {
      var option = el("option", null, preset.label);
      option.value = preset.id;
      presetSelect.appendChild(option);
    });
    $("load-preset").addEventListener("click", function () {
      applyPreset(presetSelect.value);
    });

    $("add-antenna").addEventListener("click", addAntennaAtRandom);

    var arraySelect = $("array-choice");
    $("load-array").addEventListener("click", function () {
      loadArray(arraySelect.value);
    });

    // The interference tick boxes bind themselves as they are drawn
    // (see `renderRfiKinds`), since the grid is rebuilt on every change.

    $("add-sky").addEventListener("click", function () {
      if (state.skySources.length >= state.defaults.limits.max_sky_sources) {
        showNotice("error", "That is the most sky sources one run takes ("
          + state.defaults.limits.max_sky_sources + ").");
        return;
      }
      state.skySources.push(newSkySource());
      renderSkyCards();
    });

    $("add-line").addEventListener("click", function () {
      if (state.spectralLines.length >= state.defaults.limits.max_spectral_lines) {
        showNotice("error", "That is the most spectral lines one run takes ("
          + state.defaults.limits.max_spectral_lines + ").");
        return;
      }
      state.spectralLines.push(newSpectralLine());
      renderLineCards();
    });

    runButtons().forEach(function (button) {
      button.addEventListener("click", run);
    });

    // Featuring another antenna invalidates nothing on the server, but it
    // does invalidate the flagger overlay: that was scored on one antenna's
    // voltages and means nothing on another's. `activeFlag` keys on the
    // antenna, so the panel puts the control back by itself.
    // The picker is the typed route to the same thing a tile click does.
    $("waterfall-antenna").addEventListener("change", function (event) {
      featureAntenna(Number(event.target.value));
    });

    $("all-antennas-toggle").addEventListener("click", function () {
      state.allAntennas = !state.allAntennas;
      renderAllAntennasControl();
      renderThumbnails();
      if (!state.allAntennas) { renderWaterfall(); }
    });

    // The master ground-truth switch. The per-source chips inside "More"
    // choose *which* layers this paints; with the switch off, none of them
    // paint at all.
    $("truth-toggle").addEventListener("click", function (event) {
      state.truthVisible = !state.truthVisible;
      setPressed(event.currentTarget, state.truthVisible);
      renderWaterfall();
    });

    $("vis-truth-toggle").addEventListener("click", function (event) {
      state.visTruth = !state.visTruth;
      setPressed(event.currentTarget, state.visTruth);
      renderVisibilities();
    });

    $("vis-baseline").addEventListener("change", function (event) {
      state.visBaseline = Number(event.target.value);
      renderVisSpectrum();
    });

    // A canvas inside a closed fold measures zero, so the spectra are drawn
    // when the fold opens rather than behind it.
    $("vis-more").addEventListener("toggle", function () {
      if ($("vis-more").open) { renderVisSpectrum(); }
    });

    $("flagger-run").addEventListener("click", runFlagger);

    $("image-ab-toggle").addEventListener("click", toggleClean);

    // Which receptor the waterfall shows is a server-side reduction (see
    // simulate.py's `pol` query param), not a client-side redraw, so
    // changing it re-runs the observation rather than just repainting.
    $("waterfall-pol").addEventListener("change", function (event) {
      state.waterfallPol = Number(event.target.value);
      run();
    });

    $("hatch-toggle").addEventListener("change", function (event) {
      state.hatch = event.target.checked;
      renderWaterfall();
    });

    $("runs-banner-latest").addEventListener("click", function () {
      showRun(state.history.length - 1);
    });

    bindTabs();
    bindNav();

    bindTooltip($("waterfall-canvas"));
    bindTooltip($("vis-canvas"));
    bindTooltip($("image-canvas"));
    bindUvTooltip();

    var pending = null;
    window.addEventListener("resize", function () {
      if (pending) { cancelAnimationFrame(pending); }
      pending = requestAnimationFrame(function () {
        pending = null;
        renderResults();
        if (state.tab === "observatory") {
          renderMovie();
          drawTimeline();
        }
      });
    });
  }

  function boot() {
    request("/api/defaults").then(function (defaults) {
      state.defaults = defaults;
      resetToDefaults();
      // Before `bindControls`, which routes to whichever tab the hash
      // names -- possibly this one, whose renderers need `state.obs`.
      resetObservatory();

      renderSiteMeta();
      renderPointingHint();
      bindSitePlan();
      bindControls();
      renderAllForms();
      renderMaskToggles();
      renderFlaggerControls();

      renderDayControls();
      bindObservatory();
      drawSkyChart();
      if (state.tab === "observatory") {
        refreshTimeline();
        syncSkyPolling();
      }

      // The layout catalogue is a convenience, not a prerequisite: a
      // server offering none still runs, with an empty picker.
      request("/api/arrays").then(function (arrays) {
        state.arrays = arrays;
        renderArrayChoices();
      }).catch(function () {
        state.arrays = [];
        renderArrayChoices();
      });

      run();
    }).catch(function (error) {
      showNotice("error", "Could not reach the simulator: " + error.message);
    });
  }

  boot();
}());
