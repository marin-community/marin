# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: A002, E501

"""Small local web UI for roofline reports."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from marin.tools.roofline.hardware import load_hardware_registry
from marin.tools.roofline.types import RooflineReport


def serve_report(report: RooflineReport, address: str) -> None:
    host, port_text = _parse_address(address)
    page = _render_html(report)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path not in {"/", "/index.html"}:
                self.send_response(404)
                self.end_headers()
                return
            encoded = page.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer((host, int(port_text)), Handler)
    print(f"Serving roofline dashboard at http://{host}:{port_text}/")
    server.serve_forever()


def _parse_address(address: str) -> tuple[str, str]:
    if ":" not in address:
        raise ValueError("--serve must use host:port, for example 127.0.0.1:6070")
    host, port = address.rsplit(":", 1)
    return host, port


def _render_html(report: RooflineReport) -> str:
    hardware_registry = {
        name: hardware.to_dict() for name, hardware in sorted(load_hardware_registry().items(), key=lambda item: item[0])
    }
    return _HTML_TEMPLATE.replace("__REPORT_JSON__", _script_json(report.to_dict())).replace(
        "__HARDWARE_JSON__", _script_json(hardware_registry)
    )


def _script_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Marin Roofline Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #1f2933;
      background: #f7f8f3;
    }
    body { margin: 0; }
    main { display: grid; grid-template-columns: minmax(300px, 380px) 1fr; min-height: 100vh; }
    aside { border-right: 1px solid #cfd8dc; padding: 18px; background: #eef3f1; overflow: auto; }
    section { padding: 18px 22px; overflow: auto; }
    h1 { font-size: 22px; margin: 0 0 16px; }
    h2 { font-size: 14px; margin: 22px 0 8px; text-transform: uppercase; letter-spacing: 0; color: #52616b; }
    label { display: block; font-size: 12px; color: #52616b; margin: 10px 0 4px; }
    textarea, input, select { width: 100%; box-sizing: border-box; border: 1px solid #b0bec5; border-radius: 6px; background: white; padding: 8px; }
    textarea { min-height: 150px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
    .control-row { display: grid; grid-template-columns: 1fr 96px; gap: 10px; align-items: end; }
    .totals { display: grid; grid-template-columns: repeat(7, minmax(132px, 1fr)); gap: 10px; margin-bottom: 14px; }
    .metric { border: 1px solid #c8d6d0; border-radius: 6px; padding: 10px; background: #ffffff; }
    .metric span { display: block; font-size: 12px; color: #5c6b73; }
    .metric strong { font-size: 18px; }
    table { border-collapse: collapse; width: 100%; background: white; }
    th, td { border-bottom: 1px solid #d9e2df; padding: 8px; font-size: 13px; text-align: right; vertical-align: middle; }
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
    th { position: sticky; top: 0; background: #e7eeee; z-index: 1; }
    th button { all: unset; cursor: pointer; color: #20323a; font-weight: 700; }
    th button::after { content: attr(data-arrow); color: #607d8b; padding-left: 4px; }
    input[type="range"] { width: 120px; }
    .warning { color: #7a4a00; background: #fff8e1; border: 1px solid #ffe0a3; border-radius: 6px; padding: 8px; margin-bottom: 8px; }
    .toolbar { display: flex; gap: 8px; margin: 10px 0 14px; }
    .toolbar button { width: 100%; }
    button { border: 1px solid #607d8b; border-radius: 6px; background: #34515e; color: white; padding: 8px 10px; cursor: pointer; }
    .muted { color: #697a82; font-size: 12px; margin-top: 6px; }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } aside { border-right: 0; border-bottom: 1px solid #cfd8dc; } .totals { grid-template-columns: 1fr 1fr; } }
  </style>
</head>
<body>
<script id="report" type="application/json">__REPORT_JSON__</script>
<script id="hardware-registry" type="application/json">__HARDWARE_JSON__</script>
<main>
  <aside>
    <h1>Roofline Dashboard</h1>
    <h2>Scenario</h2>
    <div class="control-row">
      <div>
        <label for="hardware-select">Hardware</label>
        <select id="hardware-select"></select>
      </div>
      <div>
        <label for="node-count">Nodes</label>
        <input id="node-count" type="number" min="1" step="1">
      </div>
    </div>
    <label for="critical-path-mode">Critical Path Proxy</label>
    <select id="critical-path-mode">
      <option value="compute_expert">compute + exposed expert all-to-all</option>
      <option value="compute_all_comm">compute + all comm</option>
      <option value="max_compute_comm">max(compute, comm)</option>
      <option value="observed_track_summed">normalized observed</option>
    </select>
    <div class="muted">Default assumes expert exchange is exposed and does not materially overlap compute.</div>
    <label for="comm-fabric">Comm fabric</label>
    <select id="comm-fabric">
      <option value="inter_host">inter-host fabric</option>
      <option value="intra_host">intra-host fabric</option>
    </select>
    <div class="control-row">
      <div>
        <label for="profile-devices">Profile devices</label>
        <input id="profile-devices" type="number" min="1" step="1">
      </div>
      <div>
        <label for="profile-steps">Profile steps</label>
        <input id="profile-steps" type="number" min="1" step="1">
      </div>
    </div>
    <div class="muted">Actual/device time is track-summed profile time / (profile devices x profile steps).</div>
    <h2>Imports</h2>
    <input id="wandb" aria-label="W&B run" readonly>
    <input id="profile" aria-label="Profile path" readonly style="margin-top:8px">
    <h2>Model Spec</h2>
    <textarea id="model"></textarea>
    <h2>Hardware</h2>
    <textarea id="hardware"></textarea>
    <h2>Attribution Rules</h2>
    <textarea id="rules"></textarea>
    <div class="toolbar"><button id="download">Save mapping</button></div>
  </aside>
  <section>
    <div id="warnings"></div>
    <div class="totals" id="totals"></div>
    <table>
      <thead>
        <tr>
          <th><button data-sort="semantic_op" onclick="sortBy('semantic_op')">Semantic op</button></th>
          <th><button data-sort="kind" onclick="sortBy('kind')">Kind</button></th>
          <th><button data-sort="work" onclick="sortBy('work')">Per-device work</button></th>
          <th><button data-sort="ideal" onclick="sortBy('ideal')">Ideal/device ms</button></th>
          <th><button data-sort="efficiency" onclick="sortBy('efficiency')">Efficiency</button></th>
          <th><button data-sort="estimate" onclick="sortBy('estimate')">Modeled/device ms</button></th>
          <th><button data-sort="observed" onclick="sortBy('observed')">Actual/device ms</button></th>
          <th><button data-sort="track_summed" onclick="sortBy('track_summed')">Track-summed ms</button></th>
          <th><button data-sort="basis" onclick="sortBy('basis')">Basis</button></th>
          <th><button data-sort="achieved" onclick="sortBy('achieved')">Achieved</button></th>
        </tr>
      </thead>
      <tbody id="rows"></tbody>
    </table>
  </section>
</main>
<script>
const report = JSON.parse(document.getElementById("report").textContent);
const hardwareRegistry = JSON.parse(document.getElementById("hardware-registry").textContent);
const model = document.getElementById("model");
const hardware = document.getElementById("hardware");
const rules = document.getElementById("rules");
const hardwareSelect = document.getElementById("hardware-select");
const nodeCount = document.getElementById("node-count");
const profileDevices = document.getElementById("profile-devices");
const profileSteps = document.getElementById("profile-steps");
const criticalPathMode = document.getElementById("critical-path-mode");
const commFabric = document.getElementById("comm-fabric");
const state = {
  hardwareName: report.hardware.name,
  nodes: inferredNodeCount(),
  profileDevices: inferredProfileDevices(),
  profileSteps: inferredProfileSteps(),
  commFabric: "inter_host",
  sortKey: "kind",
  sortDir: "asc",
  manualEfficiencies: {},
};

model.value = JSON.stringify(report.model, null, 2);
rules.value = JSON.stringify(report.attribution_rules, null, 2);
document.getElementById("wandb").value = report.imports.wandb_run_url || "";
document.getElementById("profile").value = report.imports.profile_path || "";
document.getElementById("warnings").innerHTML = (report.imports.warnings || []).map(w => `<div class="warning">${escapeHtml(w)}</div>`).join("");
for (const name of Object.keys(hardwareRegistry).sort()) {
  const option = document.createElement("option");
  option.value = name;
  option.textContent = name;
  hardwareSelect.append(option);
}
hardwareSelect.value = state.hardwareName;
nodeCount.value = String(state.nodes);
profileDevices.value = String(state.profileDevices);
profileSteps.value = String(state.profileSteps);
commFabric.value = state.commFabric;

function inferredNodeCount() {
  const meshDevices = Number(report.model?.derived?.mesh_devices || 0);
  const perHost = Number(report.hardware?.devices_per_host || 0);
  if (meshDevices > 0 && perHost > 0) {
    return Math.max(1, Math.ceil(meshDevices / perHost));
  }
  return Math.max(1, Number(report.model?.mesh?.replica_dcn || 1));
}
function inferredProfileDevices() {
  if (report.imports?.profile_devices) return Math.max(1, Number(report.imports.profile_devices));
  return Math.max(1, Number(report.hardware?.devices_per_host || 1));
}
function inferredProfileSteps() {
  if (report.imports?.profile_steps) return Math.max(1, Number(report.imports.profile_steps));
  return 1;
}
function currentHardware() { return hardwareRegistry[state.hardwareName] || report.hardware; }
function secondsToMs(value) { return value == null || Number.isNaN(value) ? "" : (value * 1000).toFixed(3); }
function escapeHtml(value) {
  const replacements = { "&": "&amp;", "<": "&lt;", ">": "&gt;", "\\\"": "&quot;", "'": "&#39;" };
  return String(value).replace(/[&<>"']/g, ch => replacements[ch]);
}
function escapeAttr(value) { return escapeHtml(value).replace(/\\n/g, "&#10;"); }
function aggregateDevices(hw) { return Math.max(1, Number(hw.devices_per_host || 1) * Number(state.nodes || 1)); }
function commBandwidthGbps(row, hw) {
  if (state.commFabric === "intra_host") return Number(hw.intra_host_collective_bandwidth_gbps || 0);
  return Number(hw.inter_host_collective_bandwidth_gbps || 0);
}
function commBandwidthBytesPerSecond(row, hw) { return commBandwidthGbps(row, hw) * 1e9 / 8; }
function commFabricLabel(row) {
  return state.commFabric === "intra_host" ? "intra-host" : "inter-host";
}
function rowIdeal(row, hw) {
  if (row.kind === "compute") {
    const throughput = Number(hw.bf16_peak_tflops_per_device || 0) * 1e12;
    return throughput > 0 ? Number(row.estimated_flops || 0) / throughput : Number(row.ideal_time || 0);
  }
  if (row.kind === "comm") {
    const throughput = commBandwidthBytesPerSecond(row, hw);
    return throughput > 0 ? Number(row.estimated_bytes || 0) / throughput : Number(row.ideal_time || 0);
  }
  return Number(row.ideal_time || 0);
}
function rowEfficiency(row, index, ideal) {
  if (Object.prototype.hasOwnProperty.call(state.manualEfficiencies, index)) {
    return state.manualEfficiencies[index];
  }
  const achieved = profileEfficiency(row, ideal);
  if (achieved != null) {
    return achieved;
  }
  return Number(row.user_efficiency || 1);
}
function profileEfficiency(row, ideal) {
  if (row.observed_comparable_to_model === false) return null;
  const observed = normalizedObserved(row);
  if (observed != null && observed > 0 && ideal > 0) {
    return ideal / observed;
  }
  return null;
}
function rowEstimate(ideal, efficiency) { return ideal / Number(efficiency || 1); }
function rowAchieved(row, ideal) {
  return profileEfficiency(row, ideal);
}
function estimateTitle(row, hw, ideal, efficiency, estimate) {
  const pct = formatEfficiency(efficiency);
  const formula = row.formula ? `\\nmodel: ${row.formula}` : "";
  if (row.kind === "compute") {
    const throughput = Number(hw.bf16_peak_tflops_per_device || 0) * 1e12;
    return `${row.semantic_op}\\nideal = flops / throughput = ${formatSci(row.estimated_flops)} / ${formatSci(throughput)} = ${secondsToMs(ideal)} ms\\nefficiency = ${pct}\\nestimate = ideal / efficiency = ${secondsToMs(estimate)} ms${normalizationNote(row)}${formula}`;
  }
  if (row.kind === "comm") {
    const bandwidth = commBandwidthGbps(row, hw);
    const throughput = commBandwidthBytesPerSecond(row, hw);
    return `${row.semantic_op}\\nideal = bytes / throughput = ${formatSci(row.estimated_bytes)} / (${commFabricLabel(row)} ${bandwidth} Gbps / 8) = ${secondsToMs(ideal)} ms\\nthroughput = ${formatSci(throughput)} B/s\\nefficiency = ${pct}\\nestimate = ideal / efficiency = ${secondsToMs(estimate)} ms${normalizationNote(row)}${formula}`;
  }
  return `${row.semantic_op}\\nestimate = ${secondsToMs(estimate)} ms${formula}`;
}
function formatEfficiency(efficiency) {
  const pct = Number(efficiency || 0) * 100;
  if (pct > 0 && pct < 0.01) return `${pct.toExponential(2)}%`;
  if (pct < 1) return `${pct.toFixed(3)}%`;
  if (pct < 10) return `${pct.toFixed(2)}%`;
  return `${pct.toFixed(1)}%`;
}
function formatSci(value) {
  const numeric = Number(value || 0);
  return numeric === 0 ? "0" : numeric.toExponential(3);
}
function computedRows() {
  const hw = currentHardware();
  return report.rows.map((row, index) => {
    const ideal = rowIdeal(row, hw);
    const efficiency = rowEfficiency(row, index, ideal);
    const estimate = rowEstimate(ideal, efficiency);
    const achieved = rowAchieved(row, ideal);
    return { row, index, ideal, efficiency, estimate, achieved };
  });
}
function observedNormalizationFactor() {
  return Math.max(1, Number(state.profileDevices || 1) * Number(state.profileSteps || 1));
}
function normalizedObserved(row) {
  if (row.profile_observed_time == null) return null;
  return Number(row.profile_observed_time) / observedNormalizationFactor();
}
function normalizationNote(row) {
  const observed = normalizedObserved(row);
  if (observed == null) return "";
  return `\\nactual/device = track-summed / (${state.profileDevices} devices * ${state.profileSteps} steps) = ${secondsToMs(observed)} ms`;
}
function sortedRows() {
  const values = computedRows();
  values.sort((left, right) => {
    const result = compare(sortValue(left, state.sortKey), sortValue(right, state.sortKey));
    return state.sortDir === "asc" ? result : -result;
  });
  return values;
}
function sortValue(item, key) {
  const row = item.row;
  if (key === "ideal") return item.ideal;
  if (key === "work") return rowWork(item.row);
  if (key === "estimate") return item.estimate;
  if (key === "efficiency") return Number(item.efficiency || 0);
  if (key === "observed") return Number(normalizedObserved(row) ?? -1);
  if (key === "track_summed") return Number(row.profile_observed_time ?? -1);
  if (key === "achieved") return Number(item.achieved ?? -1);
  if (key === "basis") return row.observed_time_basis || "";
  return row[key] || "";
}
function compare(left, right) {
  if (typeof left === "number" && typeof right === "number") return left - right;
  return String(left).localeCompare(String(right));
}
function totals(rows) {
  const compute = rows.filter(item => item.row.kind === "compute").reduce((acc, item) => acc + item.estimate, 0);
  const comm = rows.filter(item => item.row.kind === "comm").reduce((acc, item) => acc + item.estimate, 0);
  const flops = rows.reduce((acc, item) => acc + Number(item.row.estimated_flops || 0), 0);
  const bytes = rows.reduce((acc, item) => acc + Number(item.row.estimated_bytes || 0), 0);
  const trackSummedObserved = report.totals.observed_track_summed_time;
  const observed = trackSummedObserved == null ? null : Number(trackSummedObserved) / observedNormalizationFactor();
  return {
    compute,
    comm,
    flops,
    bytes,
    observed,
    trackSummedObserved,
    scenario: compute + comm,
    critical: criticalPathProxy(rows, compute, comm, observed),
  };
}
function criticalPathProxy(rows, compute, comm, observed) {
  if (criticalPathMode.value === "compute_all_comm") return compute + comm;
  if (criticalPathMode.value === "max_compute_comm") return Math.max(compute, comm);
  if (criticalPathMode.value === "observed_track_summed") return observed;
  const expert = rows.find(item => item.row.semantic_op === "expert_all_to_all")?.estimate || 0;
  return compute + expert;
}
function renderTotals() {
  const values = totals(computedRows());
  const cells = [
    ["Compute modeled/device", values.compute],
    ["Comm modeled/device", values.comm],
    ["Exposed path/device", values.critical],
    ["Per-device FLOPs", formatFlops(values.flops)],
    ["Per-device comm bytes", formatBytes(values.bytes)],
    ["Actual/device observed", values.observed],
    ["Track-summed observed", values.trackSummedObserved],
    ["Scenario modeled/device", values.scenario],
  ];
  document.getElementById("totals").innerHTML = cells.map(([label, value]) => {
    const display = typeof value === "string" ? value : `${secondsToMs(value)} ms`;
    return `<div class="metric"><span>${label}</span><strong>${display}</strong></div>`;
  }).join("");
}
function rowWork(row) {
  if (row.kind === "compute") return Number(row.estimated_flops || 0);
  if (row.kind === "comm") return Number(row.estimated_bytes || 0);
  return Math.max(Number(row.estimated_flops || 0), Number(row.estimated_bytes || 0));
}
function formatWork(row) {
  if (row.kind === "compute") return formatFlops(Number(row.estimated_flops || 0));
  if (row.kind === "comm") return formatBytes(Number(row.estimated_bytes || 0));
  const flops = Number(row.estimated_flops || 0);
  const bytes = Number(row.estimated_bytes || 0);
  if (flops > 0) return formatFlops(flops);
  if (bytes > 0) return formatBytes(bytes);
  return "";
}
function formatFlops(value) {
  if (!value) return "";
  if (value >= 1e15) return `${(value / 1e15).toFixed(3)} PF`;
  if (value >= 1e12) return `${(value / 1e12).toFixed(3)} TF`;
  if (value >= 1e9) return `${(value / 1e9).toFixed(3)} GF`;
  return value.toExponential(3);
}
function formatBytes(value) {
  if (!value) return "";
  if (value >= 1e12) return `${(value / 1e12).toFixed(3)} TB`;
  if (value >= 1e9) return `${(value / 1e9).toFixed(3)} GB`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(3)} MB`;
  return `${value.toFixed(0)} B`;
}
function renderRows() {
  const hw = currentHardware();
  document.getElementById("rows").innerHTML = sortedRows().map(({ row, index, ideal, efficiency, estimate, achieved }) => {
    const observed = normalizedObserved(row);
    const observedText = observed == null ? "" : secondsToMs(observed);
    const trackSummedText = row.profile_observed_time == null ? "" : secondsToMs(row.profile_observed_time);
    const achievedText = achieved == null ? "" : formatEfficiency(achieved);
    const title = escapeAttr(estimateTitle(row, hw, ideal, efficiency, estimate));
    const sliderValue = (efficiency * 100).toFixed(3);
    const efficiencyText = formatEfficiency(efficiency);
    return `<tr>
      <td>${escapeHtml(row.semantic_op)}</td><td>${escapeHtml(row.kind)}</td><td title="${escapeAttr(row.formula || "")}">${formatWork(row)}</td><td>${secondsToMs(ideal)}</td>
      <td><input type="range" min="0.001" max="200" step="0.001" value="${sliderValue}" data-index="${index}"><span>${efficiencyText}</span></td>
      <td title="${title}">${secondsToMs(estimate)}</td><td title="track-summed / (profile devices * profile steps)">${observedText}</td><td>${trackSummedText}</td><td>${escapeHtml(row.observed_time_basis)}</td><td>${achievedText}</td>
    </tr>`;
  }).join("");
  document.querySelectorAll("input[type=range]").forEach(slider => {
    slider.addEventListener("input", event => {
      const target = event.target;
      state.manualEfficiencies[Number(target.dataset.index)] = Number(target.value) / 100;
      renderRows();
      renderTotals();
    });
  });
  document.querySelectorAll("th button").forEach(button => {
    button.dataset.arrow = button.dataset.sort === state.sortKey ? (state.sortDir === "asc" ? "▲" : "▼") : "";
  });
}
function renderHardware() {
  hardware.value = JSON.stringify(currentHardware(), null, 2);
}
hardwareSelect.addEventListener("change", event => {
  state.hardwareName = event.target.value;
  renderHardware();
  renderRows();
  renderTotals();
});
nodeCount.addEventListener("input", event => {
  state.nodes = Math.max(1, Number(event.target.value || 1));
  renderRows();
  renderTotals();
});
profileDevices.addEventListener("input", event => {
  state.profileDevices = Math.max(1, Number(event.target.value || 1));
  renderRows();
  renderTotals();
});
profileSteps.addEventListener("input", event => {
  state.profileSteps = Math.max(1, Number(event.target.value || 1));
  renderRows();
  renderTotals();
});
criticalPathMode.addEventListener("change", () => {
  renderTotals();
});
commFabric.addEventListener("change", event => {
  state.commFabric = event.target.value;
  renderRows();
  renderTotals();
});
function sortBy(key) {
  if (state.sortKey === key) {
    state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
  } else {
    state.sortKey = key;
    state.sortDir = "asc";
  }
  renderRows();
}
document.getElementById("download").addEventListener("click", () => {
  const blob = new Blob([rules.value], {type: "application/json"});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "roofline_attribution.json";
  link.click();
});
renderHardware();
renderTotals();
renderRows();
</script>
</body>
</html>"""
