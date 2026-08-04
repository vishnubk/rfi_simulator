/* Interference simulator console -- vanilla JS, no framework, no network
 * beyond this server's own two endpoints.
 *
 * Shape of the file:
 *   1. constants and small helpers          6. the run
 *   2. state                                7. console displays
 *   3. request wrapper                      8. tooltips
 *   4. site plan                            9. wiring and boot
 *   5. source and observation forms
 *
 * Rule of the house: the browser never computes physics. Masks, images,
 * occupancy and warnings all arrive from the server as ground truth; this
 * file only arranges pixels.
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

  var MASK_RGB = "228, 87, 75";
  var MASK_ALPHA = 0.45;
  var SKY_MARKER = "#3e8fb0";
  var AXIS_STROKE = "#39404b";
  var GRID_STROKE = "#262c35";
  var AXIS_TEXT = "#9aa1ac";
  var PLOT_FONT = '11px "SF Mono", "Cascadia Mono", ui-monospace, monospace';

  var VIEW = 400;          // site-plan internal coordinate box, px
  var SHEET_PAD = 30;      // margin inside it, px
  var NICE_STEPS = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000];

  function $(id) { return document.getElementById(id); }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) { node.className = className; }
    if (text !== undefined && text !== null) { node.textContent = text; }
    return node;
  }

  function clamp(value, low, high) {
    return Math.min(high, Math.max(low, value));
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
    skySources: [],
    rfiSources: [],
    spectralLines: [],
    sim: {},
    realism: {},
    result: null,
    running: false,
    waterfallAntenna: 0,
    waterfallPol: 0,
    maskVisible: [],
    hatch: false,
    notices: [],
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
      n + " antennas · " + baselines + " baselines · longest "
      + fmt(longestBaseline(), 1) + " m";
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
      renderSitePlan();
    });

    $("clear-array").addEventListener("click", function () {
      state.antennas = [];
      renderSitePlan();
    });
  }

  /* ----------------------------------------------------------- 5. forms */

  // One field descriptor -> one labelled control. `values` is the object
  // the control writes back into, in API units; the descriptor's `factor`
  // is what the person sees divided by.
  function buildField(descriptor, values, onChange) {
    var wrap = el("div", "field" + (descriptor.kind === "text"
      && descriptor.multiline ? " field-wide" : ""));
    var id = "f" + Math.random().toString(36).slice(2, 9);
    var label = el("label", "field-label");
    label.htmlFor = id;
    label.textContent = descriptor.label;
    if (descriptor.unit) {
      label.appendChild(document.createTextNode(" "));
      label.appendChild(el("span", "field-unit", "(" + descriptor.unit + ")"));
    }
    wrap.appendChild(label);

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

    wrap.appendChild(input);
    return wrap;
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
      var grid = el("div", "field-grid");
      state.defaults.sky_source.fields.forEach(function (descriptor) {
        grid.appendChild(buildField(descriptor, source));
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
        grid.appendChild(buildField(descriptor, source));
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

  function buildRfiExtras(source) {
    var row = el("div", "card-extras");

    var coupling = extraToggle("coupling scatter", Boolean(source.coupling));
    var couplingSigma = extraNumber(
      source.coupling ? source.coupling.sigma_db : 3.0, 0.5, 0, 60,
      "Per-antenna lognormal coupling scatter, dB"
    );
    couplingSigma.hidden = !source.coupling;
    function syncCoupling() {
      source.coupling = coupling.box.checked
        ? { type: "lognormal", sigma_db: parseFloat(couplingSigma.value) || 0, seed: Math.round(state.sim.seed) }
        : null;
      couplingSigma.hidden = !coupling.box.checked;
    }
    coupling.box.addEventListener("change", syncCoupling);
    couplingSigma.addEventListener("change", syncCoupling);
    row.appendChild(coupling.label);
    row.appendChild(couplingSigma);

    var polarized = extraToggle("polarized", Boolean(source.polarization));
    var polAngle = extraNumber(
      source.polarization ? source.polarization.angle_deg : 45.0, 1, -360, 360,
      "Linear polarization angle, degrees, first receptor towards the second"
    );
    polAngle.hidden = !source.polarization;
    function syncPolarization() {
      source.polarization = polarized.box.checked
        ? { type: "linear", angle_deg: parseFloat(polAngle.value) || 0 }
        : null;
      polAngle.hidden = !polarized.box.checked;
    }
    polarized.box.addEventListener("change", syncPolarization);
    polAngle.addEventListener("change", syncPolarization);
    row.appendChild(polarized.label);
    row.appendChild(polAngle);

    if (source.type === "tower" || source.type === "comb") {
      var envelope = extraToggle("clocked on/off", Boolean(source.envelope));
      var envPeriod = extraNumber(
        source.envelope ? source.envelope.period_s * 1000 : 20.0, 1, 0.1, 1e5, "Envelope period, ms"
      );
      var envDuty = extraNumber(
        source.envelope ? source.envelope.duty : 0.5, 0.05, 0, 1, "Envelope duty (on-fraction)"
      );
      envPeriod.hidden = envDuty.hidden = !source.envelope;
      function syncEnvelope() {
        source.envelope = envelope.box.checked
          ? {
            type: "periodic",
            period_s: (parseFloat(envPeriod.value) || 20.0) / 1000,
            duty: parseFloat(envDuty.value) || 0.5
          }
          : null;
        envPeriod.hidden = envDuty.hidden = !envelope.box.checked;
      }
      envelope.box.addEventListener("change", syncEnvelope);
      envPeriod.addEventListener("change", syncEnvelope);
      envDuty.addEventListener("change", syncEnvelope);
      row.appendChild(envelope.label);
      row.appendChild(envPeriod);
      row.appendChild(envDuty);
    }

    if (source.type === "impulsive") {
      var periodic = typeof source.arrival === "object" && source.arrival !== null;
      var arrival = extraToggle("periodic arrivals (not Poisson)", periodic);
      var arrRate = extraNumber(
        periodic ? source.arrival.rate_hz : 100.0, 10, 0, 1e5, "Arrival rate, events/s"
      );
      var arrJitter = extraNumber(
        periodic ? source.arrival.jitter_s * 1000 : 0.5, 0.1, 0, 1e4, "Arrival jitter, ms"
      );
      arrRate.hidden = arrJitter.hidden = !periodic;
      function syncArrival() {
        source.arrival = arrival.box.checked
          ? {
            type: "periodic",
            rate_hz: parseFloat(arrRate.value) || 100.0,
            jitter_s: (parseFloat(arrJitter.value) || 0) / 1000
          }
          : "poisson";
        arrRate.hidden = arrJitter.hidden = !arrival.box.checked;
      }
      arrival.box.addEventListener("change", syncArrival);
      arrRate.addEventListener("change", syncArrival);
      arrJitter.addEventListener("change", syncArrival);
      row.appendChild(arrival.label);
      row.appendChild(arrRate);
      row.appendChild(arrJitter);
    }

    return row;
  }

  function renderRfiCards() {
    var host = $("rfi-cards");
    while (host.firstChild) { host.removeChild(host.firstChild); }
    $("rfi-empty").hidden = state.rfiSources.length > 0;

    state.rfiSources.forEach(function (source, index) {
      var descriptorSet = rfiType(source.type);
      var card = el("div", "card");
      var head = el("div", "card-head");
      var swatch = el("span", "card-swatch");
      swatch.style.background = "rgba(" + MASK_RGB + ", 1)";
      head.appendChild(swatch);
      head.appendChild(el("span", "card-kind", descriptorSet.label));
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
        renderRfiCards();
      });
      head.appendChild(remove);
      card.appendChild(head);

      var body = el("div", "card-body");
      var grid = el("div", "field-grid");
      descriptorSet.fields.forEach(function (descriptor) {
        // The pasted element set only matters when it is going to be used.
        if (descriptor.name === "tle_text" && source.tle_source !== "custom") { return; }
        grid.appendChild(buildField(descriptor, source, function () {
          if (descriptor.name === "tle_source") { renderRfiCards(); }
        }));
      });
      body.appendChild(grid);
      card.appendChild(body);
      card.appendChild(buildRfiExtras(source));
      host.appendChild(card);
    });
  }

  function renderSimFields() {
    var host = $("sim-fields");
    while (host.firstChild) { host.removeChild(host.firstChild); }
    var limits = state.defaults.limits;
    var descriptors = [
      {
        name: "n_chan", label: "Channels", kind: "number", factor: 1,
        min: 4, max: limits.max_n_chan, step: 4, default: state.defaults.sim.n_chan
      },
      {
        name: "n_blocks", label: "Integrations", kind: "number", factor: 1,
        min: 1, max: limits.max_n_blocks, step: 1, default: state.defaults.sim.n_blocks
      },
      {
        name: "center_freq_hz", label: "Band centre", kind: "number", unit: "MHz",
        factor: 1e6, min: 1e6, max: 1e11, step: 0.1,
        default: state.defaults.sim.center_freq_hz
      },
      {
        name: "noise_std", label: "Receiver noise", kind: "number", unit: "√Jy",
        factor: 1, min: 0, max: 1e4, step: 0.1, default: state.defaults.sim.noise_std,
        help: "Its square is the noise power added to every autocorrelation."
      },
      {
        name: "seed", label: "Seed", kind: "number", factor: 1,
        min: 0, max: 2147483647, step: 1, default: state.defaults.sim.seed,
        help: "The same seed gives byte-identical data."
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
    $("sim-summary").textContent =
      fmt(duration * 1000, 1) + " ms of data · "
      + fmt(bandwidth / 1e6, 3) + " MHz wide · "
      + fmt(sim.chan_width_hz / 1e3, 2) + " kHz channels · start "
      + sim.start_time_utc + " UTC";
  }

  // The realism panel's fields have no server-side schema of their own
  // (see simulate.py -- these groups are Optional request fields, on only
  // when their `*_enabled` toggle is checked): the list below is this
  // file's own field descriptors, the same pattern `renderSimFields`
  // already uses for the observation fields. Each entry's `section` opens
  // a new labelled subgroup within the one field-grid.
  var REALISM_FIELDS = [
    { name: "n_pol", label: "Polarizations", kind: "choice", default: "1",
      options: [{ value: "1", label: "1 (single)" }, { value: "2", label: "2 (dual, XX/YY)" }],
      section: "Polarization" },

    { name: "instrument_enabled", label: "Per-antenna gain/bandpass realism", kind: "toggle",
      default: false, section: "Instrument" },
    { name: "gain_scatter_db", label: "Gain scatter", kind: "number", unit: "dB", factor: 1,
      min: 0, max: 10, step: 0.05, default: 0.4 },
    { name: "phase_offsets", label: "Phase offsets", kind: "choice", default: "zero",
      options: [{ value: "zero", label: "zero (calibrated)" }, { value: "uniform", label: "uniform (uncalibrated)" }] },
    { name: "bandpass_ripple_db", label: "Bandpass ripple", kind: "number", unit: "dB", factor: 1,
      min: 0, max: 5, step: 0.01, default: 0.05 },
    { name: "band_slope_db", label: "Band slope", kind: "number", unit: "dB", factor: 1,
      min: 0, max: 10, step: 0.05, default: 0.0 },
    { name: "subband_scatter_db", label: "Subband scatter", kind: "number", unit: "dB", factor: 1,
      min: 0, max: 10, step: 0.05, default: 0.0 },
    { name: "n_subbands", label: "Subbands", kind: "number", factor: 1,
      min: 1, max: 64, step: 1, default: 1 },

    { name: "quantization_enabled", label: "4-bit quantization (int4)", kind: "toggle",
      default: false, section: "Quantization" },
    { name: "quant_target_counts", label: "Target rms", kind: "number", unit: "counts", factor: 1,
      min: 0.1, max: 20, step: 0.01, default: 1.33 },

    { name: "channelizer_enabled", label: "Polyphase filterbank channelizer", kind: "toggle",
      default: false, section: "Channelizer" },
    { name: "n_taps", label: "Taps", kind: "number", factor: 1, min: 1, max: 32, step: 1, default: 4 },
    { name: "window", label: "Window", kind: "choice", default: "hamming",
      options: [{ value: "hann", label: "hann" }, { value: "hamming", label: "hamming" },
        { value: "blackman", label: "blackman" }] },
    { name: "sinc_bandwidth", label: "Sinc bandwidth", kind: "number", factor: 1,
      min: 0.1, max: 8, step: 0.01, default: 1.01 },

    { name: "calibration_enabled", label: "Residual calibration errors", kind: "toggle",
      default: false, section: "Calibration errors" },
    { name: "phase_error_deg_rms", label: "Phase error", kind: "number", unit: "deg", factor: 1,
      min: 0, max: 180, step: 0.5, default: 5.0 },
    // max matches CalibrationErrorParams.delay_error_ns_rms's `le` bound in
    // simulate.py -- keep the two in sync if either changes.
    { name: "delay_error_ns_rms", label: "Delay error", kind: "number", unit: "ns", factor: 1,
      min: 0, max: 10, step: 0.1, default: 0.0 },
    { name: "amplitude_error_db_rms", label: "Amplitude error", kind: "number", unit: "dB", factor: 1,
      min: 0, max: 10, step: 0.05, default: 0.0 },

    { name: "beam_enabled", label: "Primary beam attenuation", kind: "toggle",
      default: false, section: "Primary beam" },
    { name: "beam_type", label: "Beam shape", kind: "choice", default: "gaussian",
      options: [{ value: "gaussian", label: "Gaussian" }, { value: "airy", label: "Airy" }] },
    { name: "dish_diameter_m", label: "Dish diameter", kind: "number", unit: "m", factor: 1,
      min: 0.1, max: 1000, step: 0.1, default: 4.5 }
  ];

  function renderRealismFields() {
    var host = $("realism-fields");
    while (host.firstChild) { host.removeChild(host.firstChild); }
    REALISM_FIELDS.forEach(function (descriptor) {
      if (descriptor.section) {
        host.appendChild(el("div", "subgroup-label", descriptor.section));
      }
      host.appendChild(buildField(descriptor, state.realism));
    });
  }

  function defaultRealism() {
    var values = {};
    REALISM_FIELDS.forEach(function (descriptor) { values[descriptor.name] = descriptor.default; });
    return values;
  }

  function newSkySource() {
    var values = Object.assign({}, state.defaults.sky_source.defaults);
    values.name = "source " + (state.skySources.length + 1);
    return values;
  }

  function newRfiSource(typeName) {
    var descriptorSet = rfiType(typeName);
    var values = Object.assign({ type: typeName }, descriptorSet.defaults);
    values.name = descriptorSet.label.toLowerCase();
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

  /* ------------------------------------------------------------- 6. run */

  function showNotice(kind, text) {
    state.notices.push({ kind: kind, text: text });
    renderNotices();
  }

  function renderNotices() {
    var host = $("notices");
    while (host.firstChild) { host.removeChild(host.firstChild); }
    state.notices.forEach(function (notice) {
      var node = el("div", "notice" + (notice.kind === "error" ? " notice-error" : ""));
      node.appendChild(el("span", "notice-kind",
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
      sky_sources: state.skySources.map(function (source) {
        return { name: source.name, l: source.l, m: source.m, flux_jy: source.flux_jy };
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

  function run() {
    if (state.running) { return; }
    state.running = true;
    state.notices = [];
    renderNotices();
    $("run").disabled = true;
    $("run-status").textContent = "running…";
    $("waterfall-sweep").hidden = state.result === null;

    request("/api/simulate?pol=" + state.waterfallPol, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildRequest())
    }).then(function (result) {
      state.result = result;
      state.maskVisible = result.sources.map(function () { return true; });
      state.waterfallAntenna = clamp(
        state.waterfallAntenna, 0, result.waterfall.antennas.length - 1
      );
      var dualPol = result.observation.n_pol === 2;
      $("waterfall-pol").hidden = !dualPol;
      $("waterfall-pol-label").hidden = !dualPol;
      result.warnings.forEach(function (message) { showNotice("note", message); });
      $("run-status").textContent =
        "done in " + fmt(result.wall_time_s, 2) + " s";
      $("wall-time").textContent = fmt(result.wall_time_s, 2) + " s wall";
      renderResults();
    }).catch(function (error) {
      showNotice("error", error.message);
      $("run-status").textContent = "not run";
    }).then(function () {
      state.running = false;
      $("run").disabled = false;
      $("waterfall-sweep").hidden = true;
    });
  }

  /* -------------------------------------------------------- 7. displays */

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

    // Marker: the image's peak, in the sky-source colour.
    if (spec.marker) {
      var mx = plot.x + (spec.marker.x - spec.xLow) / (spec.xHigh - spec.xLow) * plot.w;
      var my = plot.y + plot.h - (spec.marker.y - spec.yLow) / (spec.yHigh - spec.yLow) * plot.h;
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

  function renderWaterfall() {
    var canvas = $("waterfall-canvas");
    var result = state.result;
    if (!result) { return; }
    var water = result.waterfall;
    var values = water.antennas[state.waterfallAntenna];
    var overlays = result.sources.filter(function (source, index) {
      return state.maskVisible[index];
    }).map(function (source) {
      return { mask: source.mask };
    });

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
    var names = state.result.sources.filter(function (source, index) {
      return state.maskVisible[index] && source.mask[row][col];
    }).map(function (source) { return source.name; });
    return names.length ? "\nflagged: " + names.join(", ") : "";
  }

  function renderImage() {
    var canvas = $("image-canvas");
    var result = state.result;
    if (!result) { return; }
    var image = result.image;

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
      readCell: function (row, col) {
        return "l " + image.l[col].toFixed(4) + "\nm " + image.m[row].toFixed(4)
          + "\n" + image.values[row][col].toFixed(3) + " Jy";
      }
    });

    $("image-sub").textContent =
      "peak " + image.peak.value_jy.toFixed(3) + " Jy at l "
      + image.peak.l.toFixed(4) + ", m " + image.peak.m.toFixed(4);
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
    renderWaterfall();
    renderImage();
    renderUv();
  }

  /* -------------------------------------------------------- 8. tooltips */

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

  /* ------------------------------------------------------ 9. wiring */

  function renderSiteMeta() {
    var array = state.defaults.array;
    var meta = $("site-meta");
    while (meta.firstChild) { meta.removeChild(meta.firstChild); }
    [["site", array.latitude_deg.toFixed(3) + "°, " + array.longitude_deg.toFixed(3) + "°"],
     ["height", array.height_m.toFixed(0) + " m"]].forEach(function (pair) {
      meta.appendChild(el("dt", null, pair[0]));
      meta.appendChild(el("dd", null, pair[1]));
    });
  }

  function bindControls() {
    var typeSelect = $("rfi-type");
    state.defaults.rfi_types.forEach(function (entry) {
      var option = el("option", null, entry.label);
      option.value = entry.type;
      typeSelect.appendChild(option);
    });
    function showTypeSummary() {
      $("rfi-type-summary").textContent = rfiType(typeSelect.value).summary;
    }
    typeSelect.addEventListener("change", showTypeSummary);
    showTypeSummary();

    $("add-rfi").addEventListener("click", function () {
      if (state.rfiSources.length >= state.defaults.limits.max_rfi_sources) {
        showNotice("error", "That is the most interference sources one run takes ("
          + state.defaults.limits.max_rfi_sources + ").");
        return;
      }
      state.rfiSources.push(newRfiSource(typeSelect.value));
      renderRfiCards();
    });

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

    $("run").addEventListener("click", run);

    $("waterfall-antenna").addEventListener("change", function (event) {
      state.waterfallAntenna = Number(event.target.value);
      renderWaterfall();
    });

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

    bindTooltip($("waterfall-canvas"));
    bindTooltip($("image-canvas"));
    bindUvTooltip();

    var pending = null;
    window.addEventListener("resize", function () {
      if (pending) { cancelAnimationFrame(pending); }
      pending = requestAnimationFrame(function () {
        pending = null;
        renderResults();
      });
    });
  }

  function boot() {
    request("/api/defaults").then(function (defaults) {
      state.defaults = defaults;
      state.antennas = defaults.array.antennas.map(function (row) { return row.slice(); });
      state.sim = {
        n_chan: defaults.sim.n_chan,
        n_blocks: defaults.sim.n_blocks,
        center_freq_hz: defaults.sim.center_freq_hz,
        noise_std: defaults.sim.noise_std,
        seed: defaults.sim.seed
      };
      state.skySources = [newSkySource()];
      state.rfiSources = [];
      state.spectralLines = [];
      state.realism = defaultRealism();

      renderSiteMeta();
      bindSitePlan();
      bindControls();
      renderSitePlan();
      renderSkyCards();
      renderRfiCards();
      renderLineCards();
      renderSimFields();
      renderRealismFields();
      renderMaskToggles();
      run();
    }).catch(function (error) {
      showNotice("error", "Could not reach the simulator: " + error.message);
    });
  }

  boot();
}());
