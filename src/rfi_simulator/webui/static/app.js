/* Interference simulator console -- vanilla JS, no framework, no network
 * beyond this server's own two endpoints.
 *
 * Shape of the file:
 *   1. constants and small helpers          6. scenario presets
 *   2. state                                7. the run
 *   3. request wrapper                      8. result displays
 *   4. site plan                            9. tooltips
 *   5. source and observation forms        10. wiring and boot
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
  var MIN_SPACING_M = 5;   // how far apart a dropped antenna tries to land
  var LARGE_ARRAY = 32;    // above this, a loaded layout gets a "slower run" note
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
    $("sim-summary").textContent =
      "This run records " + fmt(duration * 1000, 1) + " ms of data from "
      + fmt((state.sim.center_freq_hz - half) / 1e6, 3) + " to "
      + fmt((state.sim.center_freq_hz + half) / 1e6, 3) + " MHz ("
      + fmt(bandwidth / 1e6, 3) + " MHz in "
      + fmt(sim.chan_width_hz / 1e3, 2) + " kHz channels), starting "
      + sim.start_time_utc + " UTC.";
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
    $("section-run").scrollIntoView({ behavior: "smooth", block: "start" });
    run();
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

  // The Run button appears twice -- once in the sticky header, once beside
  // the results -- so both copies move together.
  function runButtons() { return [$("run"), $("run-main")]; }

  function setRunStatus(text) {
    [$("run-status"), $("run-status-main")].forEach(function (node) {
      node.textContent = text;
    });
  }

  function run() {
    if (state.running) { return; }
    state.running = true;
    state.notices = [];
    renderNotices();
    runButtons().forEach(function (button) { button.disabled = true; });
    setRunStatus("running…");
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
      $("waterfall-pol-group").hidden = result.observation.n_pol !== 2;
      result.warnings.forEach(function (message) { showNotice("note", message); });
      setRunStatus("done in " + fmt(result.wall_time_s, 2) + " s");
      $("wall-time").textContent = fmt(result.wall_time_s, 2) + " s wall";
      renderResults();
    }).catch(function (error) {
      showNotice("error", error.message);
      setRunStatus("not run");
    }).then(function () {
      state.running = false;
      runButtons().forEach(function (button) { button.disabled = false; });
      $("waterfall-sweep").hidden = true;
    });
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

  /* -------------------------------------------------------- 9. tooltips */

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

  /* ----------------------------------------------------- 10. wiring */

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
      resetToDefaults();

      renderSiteMeta();
      renderPointingHint();
      bindSitePlan();
      bindControls();
      renderAllForms();
      renderMaskToggles();

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
