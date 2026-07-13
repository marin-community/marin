import * as d3 from "d3";

import rawData from "./generated/dashboard_data.json";
import { renderMixtureChart } from "./mixture";
import { renderScatter } from "./scatter";
import "./styles.css";
import type {
  DashboardData,
  DashboardState,
  ExplorerTab,
  FitParameter,
  MixtureRow,
  ModelId,
  NikeSwooshDiagnostic,
  PointDatum,
  SortMode,
  SwarmData,
  ViewMode,
} from "./types";

const data = rawData as unknown as DashboardData;
const modelIds = Object.keys(data.models) as ModelId[];
const swarmIds = Object.keys(data.swarms);

function requiredElement<T extends HTMLElement>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Missing ${selector} root`);
  return element;
}

function isModel(value: string | null): value is ModelId {
  return value !== null && modelIds.includes(value as ModelId);
}

function isView(value: string | null): value is ViewMode {
  return value === "prediction" || value === "residual" || value === "standardized";
}

function isSort(value: string | null): value is SortMode {
  return value === "difference" || value === "exposure" || value === "domain";
}

function isTab(value: string | null): value is ExplorerTab {
  return value === "mixtures" || value === "fit";
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatMetric(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

function formatParameter(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude < 1e-3 || magnitude >= 1e4)) return d3.format(".4e")(value);
  return d3.format(".6~g")(value);
}

function formatSigned(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
}

const query = new URLSearchParams(window.location.search);
const initialSwarm = swarmIds.includes(query.get("swarm") ?? "") ? (query.get("swarm") as string) : "300m";
const initialTargetIds = Object.keys(data.swarms[initialSwarm]!.targets);
const initialTarget = initialTargetIds.includes(query.get("target") ?? "")
  ? (query.get("target") as string)
  : initialTargetIds[0]!;
const state: DashboardState = {
  swarm: initialSwarm,
  target: initialTarget,
  model: isModel(query.get("model")) ? (query.get("model") as ModelId) : "separate_heads",
  tab: isTab(query.get("tab")) ? (query.get("tab") as ExplorerTab) : "mixtures",
  view: isView(query.get("view")) ? (query.get("view") as ViewMode) : "prediction",
  selectedId: query.get("selected"),
  baselineId: query.get("baseline") ?? "",
  showFit: query.get("fit") !== "0",
  showHeldout: query.get("heldout") !== "0",
  showNoise: query.get("noise") === "1",
  hideAliases: query.get("aliases") !== "1",
  phaseFamily:
    query.get("phase") === "single_phase" || query.get("phase") === "two_phase"
      ? (query.get("phase") as "single_phase" | "two_phase")
      : "all",
  search: query.get("search") ?? "",
  sort: isSort(query.get("sort")) ? (query.get("sort") as SortMode) : "difference",
  parameterDomain: query.get("parameterDomain") ?? "",
  parameterGroup: query.get("parameterGroup") ?? "",
};

const app = requiredElement<HTMLDivElement>("#app");
app.innerHTML = `
  <header class="masthead">
    <div class="masthead-copy">
      <div class="eyebrow">Marin · surrogate diagnostics across mixture swarms</div>
      <h1>Mixture Fit Observatory</h1>
      <p>Inspect out-of-fold errors, the policies behind them, and every interpretable parameter in the fitted response surface.</p>
    </div>
    <div class="dataset-stamp" id="dataset-stamp"></div>
  </header>
  <main>
    <section class="selection-shell">
      <nav class="selection-rail" aria-label="Fit selection">
        <div class="rail-block"><span class="control-label">01 · Swarm</span><div id="swarm-controls" class="vertical-control"></div></div>
        <div class="rail-block"><span class="control-label">02 · Objective</span><div id="target-controls" class="vertical-control"></div></div>
        <div class="rail-block model-rail"><span class="control-label">03 · Surrogate</span><div id="model-controls" class="vertical-control"></div></div>
      </nav>
      <div class="selection-context">
        <div class="context-topline"><span id="swarm-kicker"></span><span id="phase-stamp"></span></div>
        <h2 id="swarm-title"></h2>
        <p id="swarm-description"></p>
        <div class="model-note" id="model-note"></div>
        <div class="workspace-tabs" role="tablist">
          <button data-action="tab" data-value="mixtures" role="tab">Mixture Explorer</button>
          <button data-action="tab" data-value="fit" role="tab">Fit Explorer</button>
        </div>
      </div>
    </section>

    <section class="metric-strip" id="metric-strip" aria-label="Model diagnostics"></section>

    <div id="mixture-workspace">
      <section class="analysis-grid">
        <article class="panel scatter-panel">
          <div class="panel-heading scatter-heading">
            <div><span class="section-index">A</span><h2>Prediction field</h2><p id="scatter-caption"></p></div>
            <div class="view-controls" id="view-controls"></div>
          </div>
          <div class="filter-row">
            <label><input id="filter-fit" type="checkbox" /> Fit designs</label>
            <label><input id="filter-heldout" type="checkbox" /> Heldout</label>
            <label><input id="filter-noise" type="checkbox" /> Repeat anchors</label>
            <label><input id="filter-alias" type="checkbox" /> Hide shared alias</label>
            <label class="select-label">Policy
              <select id="phase-filter"><option value="all">All</option><option value="single_phase">One phase</option><option value="two_phase">Two phase</option></select>
            </label>
            <label class="search-label"><span class="sr-only">Search checkpoints</span>
              <input id="run-search" list="run-search-options" type="search" placeholder="Find a checkpoint…" />
              <datalist id="run-search-options"></datalist>
            </label>
            <button class="quiet-button" data-action="reset-filters">Reset</button>
          </div>
          <div id="scatter" class="scatter"></div>
          <div class="encoding-note">
            <span><i class="phase-key one-phase"></i> one-phase checkpoint</span>
            <span><i class="phase-key two-phase"></i> two-phase checkpoint</span>
            <span id="noise-encoding"></span>
          </div>
        </article>
        <aside class="panel inspector-panel" id="inspector"></aside>
      </section>

      <section class="panel mixture-panel">
        <div class="panel-heading comparison-heading">
          <div><span class="section-index">B</span><h2>Mixture anatomy</h2><p>Selected mixture in color; comparison policy in gray. Phase bars carry realized epochs.</p></div>
          <div class="comparison-controls">
            <label>Compare against <select id="baseline-select"></select></label>
            <label>Order <select id="sort-select">
              <option value="difference">Largest aggregate change</option>
              <option value="exposure">Highest exposure</option>
              <option value="domain">Domain family</option>
            </select></label>
          </div>
        </div>
        <div id="comparison-summary" class="comparison-summary"></div>
        <div id="mixture-chart" class="mixture-chart"></div>
      </section>
    </div>

    <div id="fit-workspace" hidden>
      <section class="fit-layout">
        <aside class="panel fit-index">
          <div class="fit-index-heading"><span class="section-index">PARAMETER SCOPE</span><h2>Inspect the fitted mechanism</h2></div>
          <label>Metric<input id="parameter-metric" type="text" readonly /></label>
          <label id="parameter-group-label">Group<select id="parameter-group"></select></label>
          <label id="parameter-domain-label">Bucket<select id="parameter-domain"></select></label>
          <div id="fit-caveats" class="fit-caveats"></div>
        </aside>
        <article class="panel parameter-panel">
          <div class="panel-heading">
            <div><span class="section-index">C</span><h2>Fit parameters</h2><p id="parameter-caption"></p></div>
          </div>
          <div id="parameter-table"></div>
        </article>
      </section>
      <section class="panel swoosh-panel" id="swoosh-panel" hidden>
        <div class="panel-heading"><div><span class="section-index">D</span><h2>Nike-swoosh response</h2><p id="swoosh-caption"></p></div></div>
        <div id="swoosh-charts" class="swoosh-charts"></div>
      </section>
      <section class="panel tuning-panel">
        <details><summary>Fitting protocol and tuned hyperparameters</summary><div id="fit-protocol"></div><pre id="fit-tuning"></pre></details>
      </section>
    </div>

    <section class="methodology-band">
      <div><span>Fit semantics</span><strong id="fit-semantics"></strong></div>
      <div><span>Noise scale</span><strong id="noise-unit"></strong></div>
      <div><span>Phase budget</span><strong id="phase-budget"></strong></div>
      <div><span>Parameters</span><strong id="parameter-count"></strong></div>
    </section>
  </main>
  <footer><span id="provenance"></span><span>All state is encoded in the URL.</span></footer>
  <div id="tooltip" class="tooltip" role="status"></div>
`;

const tooltip = requiredElement<HTMLElement>("#tooltip");

function swarm(): SwarmData {
  return data.swarms[state.swarm]!;
}

function rowsById(): Map<string, { row: MixtureRow; index: number }> {
  return new Map(swarm().rows.map((row, index) => [row.id, { row, index }]));
}

function rowsByName(): Map<string, MixtureRow> {
  return new Map(swarm().rows.map((row) => [row.name, row]));
}

function predictionFor(row: MixtureRow, fullFit = false): number | null {
  const entry = rowsById().get(row.id);
  if (!entry) return null;
  const series = swarm().predictions[state.target]![state.model];
  return (fullFit ? series.fullFitPrediction[entry.index] : series.prediction[entry.index]) ?? null;
}

function pointData(): PointDatum[] {
  const current = swarm();
  const noiseScale = Math.max(current.targets[state.target]!.noiseReference.differenceStandardDeviation, 1e-12);
  const series = current.predictions[state.target]![state.model];
  return current.rows.flatMap((row, rowIndex) => {
    const observed = row.observed[state.target];
    const prediction = series.prediction[rowIndex];
    const fullFitPrediction = series.fullFitPrediction[rowIndex];
    if (observed === null || observed === undefined || prediction === null || prediction === undefined || fullFitPrediction === null || fullFitPrediction === undefined) return [];
    const residual = prediction - observed;
    return [{ row, rowIndex, observed, prediction, fullFitPrediction, residual, standardizedResidual: residual / noiseScale }];
  });
}

function visiblePoints(): PointDatum[] {
  const search = state.search.trim().toLowerCase();
  return pointData().filter((point) => {
    if (point.row.split === "fit" && !state.showFit) return false;
    if (point.row.split === "heldout" && !state.showHeldout) return false;
    if (point.row.split === "noise_reference" && !state.showNoise) return false;
    if (point.row.split === "candidate") return false;
    if (state.hideAliases && point.row.isSharedAlias) return false;
    if (state.phaseFamily !== "all" && point.row.phaseFamily !== state.phaseFamily) return false;
    if (search && !`${point.row.name} ${point.row.panel ?? ""} ${point.row.targetDomain ?? ""}`.toLowerCase().includes(search)) return false;
    return true;
  });
}

function chooseDefaultSelection(): string {
  const heldout = pointData().filter((point) => point.row.split === "heldout" && !point.row.isSharedAlias);
  if (heldout.length > 0) {
    return d3.greatest(heldout, (point) => Math.abs(point.standardizedResidual))?.row.id ?? heldout[0]!.row.id;
  }
  return d3.least(pointData(), (point) => point.observed)?.row.id ?? swarm().rows[0]?.id ?? "";
}

function ensureState(): void {
  const current = swarm();
  const targetIds = Object.keys(current.targets);
  if (!targetIds.includes(state.target)) state.target = targetIds[0]!;
  const byId = rowsById();
  if (!state.selectedId || !byId.has(state.selectedId)) state.selectedId = chooseDefaultSelection();
  const baselineIds = current.baselines[state.target]!.map((option) => option.id);
  if (!baselineIds.includes(state.baselineId)) state.baselineId = baselineIds[0] ?? "";
  const detail = current.fits[state.target]![state.model];
  const domains = detail.parameters.filter((parameter) => parameter.scope === "domain" && parameter.domainId).map((parameter) => parameter.domainId as string);
  const groups = detail.parameters.filter((parameter) => parameter.scope === "group" && parameter.groupLabel).map((parameter) => parameter.groupLabel as string);
  if (!domains.includes(state.parameterDomain)) state.parameterDomain = domains[0] ?? "";
  if (!groups.includes(state.parameterGroup)) state.parameterGroup = groups[0] ?? "";
}

function updateUrl(): void {
  const params = new URLSearchParams();
  params.set("swarm", state.swarm);
  params.set("target", state.target);
  params.set("model", state.model);
  params.set("tab", state.tab);
  params.set("view", state.view);
  if (state.selectedId) params.set("selected", state.selectedId);
  if (state.baselineId) params.set("baseline", state.baselineId);
  if (!state.showFit) params.set("fit", "0");
  if (!state.showHeldout) params.set("heldout", "0");
  if (state.showNoise) params.set("noise", "1");
  if (!state.hideAliases) params.set("aliases", "1");
  if (state.phaseFamily !== "all") params.set("phase", state.phaseFamily);
  if (state.search) params.set("search", state.search);
  params.set("sort", state.sort);
  if (state.parameterDomain) params.set("parameterDomain", state.parameterDomain);
  if (state.parameterGroup) params.set("parameterGroup", state.parameterGroup);
  window.history.replaceState(null, "", `${window.location.pathname}?${params.toString()}`);
}

function renderSelection(): void {
  const current = swarm();
  requiredElement<HTMLElement>("#swarm-controls").innerHTML = swarmIds.map((id) =>
    `<button data-action="swarm" data-value="${escapeHtml(id)}" class="${id === state.swarm ? "active" : ""}"><strong>${escapeHtml(data.swarms[id]!.label)}</strong><span>${data.swarms[id]!.dataset.fitDesignCount} rows · ${data.swarms[id]!.domains.length} buckets</span></button>`,
  ).join("");
  requiredElement<HTMLElement>("#target-controls").innerHTML = Object.keys(current.targets).map((id) =>
    `<button data-action="target" data-value="${escapeHtml(id)}" class="${id === state.target ? "active" : ""}">${escapeHtml(current.targets[id]!.label)}</button>`,
  ).join("");
  requiredElement<HTMLElement>("#model-controls").innerHTML = modelIds.map((id) =>
    `<button data-action="model" data-value="${id}" class="${id === state.model ? "active" : ""}"><strong>${escapeHtml(data.models[id]!.label)}</strong></button>`,
  ).join("");
  requiredElement<HTMLElement>("#swarm-kicker").textContent = `${current.dataset.fitDesignCount} fit rows · ${current.domains.length} buckets`;
  requiredElement<HTMLElement>("#phase-stamp").textContent = `${d3.format(".0%")(current.dataset.phaseFractions[0])} / ${d3.format(".0%")(current.dataset.phaseFractions[1])} phases`;
  requiredElement<HTMLElement>("#swarm-title").textContent = current.label;
  requiredElement<HTMLElement>("#swarm-description").textContent = current.description;
  const detail = current.fits[state.target]![state.model];
  const note = requiredElement<HTMLElement>("#model-note");
  note.innerHTML = `<div><span class="control-label">Selected fit</span><strong>${escapeHtml(detail.modelLabel)}</strong></div><p>${escapeHtml(detail.description)}</p><small>${detail.parameterCount} semantic parameters · ${escapeHtml(detail.protocol)}</small>`;
  document.querySelectorAll<HTMLButtonElement>("[data-action='tab']").forEach((button) => {
    const active = button.dataset.value === state.tab;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  const stamp = requiredElement<HTMLElement>("#dataset-stamp");
  stamp.innerHTML = `<span>FIT ROWS</span><strong>${current.dataset.fitDesignCount}</strong><span>HELDOUT</span><strong>${current.dataset.heldoutCount}</strong><span>DOMAINS</span><strong>${current.domains.length}</strong>`;
}

function renderMetricStrip(): void {
  const current = swarm();
  const diagnostics = current.diagnostics[state.target]![state.model];
  const fit = current.fits[state.target]![state.model];
  const cards: Array<[string, number | null, string]> = [
    ["OOF RMSE", diagnostics.fitOof.rmse, `${diagnostics.fitOof.n} fit designs`],
    ["OOF Spearman", diagnostics.fitOof.spearman, "Held-out folds"],
    ["Train RMSE", fit.diagnostics.train.rmse, "Full fit"],
    ["Train Spearman", fit.diagnostics.train.spearman, "Full fit"],
  ];
  if (diagnostics.heldout.n > 0) {
    cards.push(["Heldout RMSE", diagnostics.heldout.rmse, `${diagnostics.heldout.n} checkpoints`]);
    cards.push(["Heldout Spearman", diagnostics.heldout.spearman, "No refitting"]);
  } else {
    cards.push(["Rows", current.dataset.fitDesignCount, "Observed surface"]);
    cards.push(["Parameters", fit.parameterCount, "Semantic records"]);
  }
  requiredElement<HTMLElement>("#metric-strip").innerHTML = cards.map(([label, value, detailText]) =>
    `<div class="metric-card"><span>${label}</span><strong>${formatMetric(value, label.includes("Spearman") ? 3 : label === "Rows" || label === "Parameters" ? 0 : 5)}</strong><small>${escapeHtml(detailText)}</small></div>`,
  ).join("");
}

function renderViewControls(): void {
  const labels: Record<ViewMode, string> = { prediction: "Predicted ↔ observed", residual: "Residual", standardized: "Scale-normalized" };
  requiredElement<HTMLElement>("#view-controls").innerHTML = (Object.keys(labels) as ViewMode[]).map((view) =>
    `<button data-action="view" data-value="${view}" class="${view === state.view ? "active" : ""}">${labels[view]}</button>`,
  ).join("");
}

function renderFilters(): void {
  const current = swarm();
  requiredElement<HTMLInputElement>("#filter-fit").checked = state.showFit;
  requiredElement<HTMLInputElement>("#filter-heldout").checked = state.showHeldout;
  requiredElement<HTMLInputElement>("#filter-noise").checked = state.showNoise;
  requiredElement<HTMLInputElement>("#filter-alias").checked = state.hideAliases;
  requiredElement<HTMLSelectElement>("#phase-filter").value = state.phaseFamily;
  requiredElement<HTMLInputElement>("#run-search").value = state.search;
  requiredElement<HTMLDataListElement>("#run-search-options").innerHTML = current.rows.filter((row) => row.observed[state.target] !== null).map((row) => `<option value="${escapeHtml(row.name)}"></option>`).join("");
}

function splitLabel(row: MixtureRow): string {
  if (row.split === "fit") return "Fit design · OOF";
  if (row.split === "heldout") return "Heldout · full fit";
  if (row.split === "noise_reference") return "Repeat anchor · full fit";
  return "Candidate · full fit";
}

function performanceBlock(row: MixtureRow, label: string): string {
  const current = swarm();
  const observed = row.observed[state.target];
  const predicted = predictionFor(row);
  const residual = observed !== null && observed !== undefined && predicted !== null ? predicted - observed : null;
  const scale = Math.max(current.targets[state.target]!.noiseReference.differenceStandardDeviation, 1e-12);
  return `<div class="performance-block"><span class="performance-label">${escapeHtml(label)}</span><strong>${escapeHtml(row.name)}</strong><div class="performance-grid"><div><span>Observed</span><b>${formatMetric(observed, 6)}</b></div><div><span>Predicted</span><b>${formatMetric(predicted, 6)}</b></div><div><span>Residual</span><b>${formatSigned(residual, 6)}</b></div><div><span>Scale units</span><b>${formatSigned(residual === null ? null : residual / scale, 2)}×</b></div></div></div>`;
}

function renderInspector(): void {
  const byId = rowsById();
  const byName = rowsByName();
  const selected = state.selectedId ? byId.get(state.selectedId)?.row : undefined;
  if (!selected) return;
  const nearest = byId.get(selected.diagnostics.nearestFitId)?.row;
  const paired = selected.pairedRow ? byName.get(selected.pairedRow) : undefined;
  const observed = selected.observed[state.target];
  const predicted = predictionFor(selected);
  const residual = observed !== null && observed !== undefined && predicted !== null ? predicted - observed : null;
  const scale = Math.max(swarm().targets[state.target]!.noiseReference.differenceStandardDeviation, 1e-12);
  requiredElement<HTMLElement>("#inspector").innerHTML = `
    <div class="inspector-topline"><span class="section-index">SELECTED</span><span>${escapeHtml(splitLabel(selected))}</span></div>
    <h2>${escapeHtml(selected.name)}</h2>
    <div class="badge-row"><span class="badge ${selected.phaseFamily === "single_phase" ? "orange" : "blue"}">${escapeHtml(selected.phaseFamily?.replace("_", " ") ?? "unknown phase")}</span><span class="badge">${escapeHtml(selected.panel ?? "unclassified")}</span>${selected.isSharedAlias ? '<span class="badge warning">shared alias</span>' : ""}</div>
    <div class="selected-performance ${residual !== null && residual < 0 ? "danger" : ""}">
      <div><span>Observed</span><strong>${formatMetric(observed, 6)}</strong></div><div><span>${selected.split === "fit" ? "OOF prediction" : "Full-fit prediction"}</span><strong>${formatMetric(predicted, 6)}</strong></div><div><span>Residual</span><strong>${formatSigned(residual, 6)}</strong></div><div><span>Scale units</span><strong>${formatSigned(residual === null ? null : residual / scale, 2)}×</strong></div>
    </div>
    <p class="residual-reading">${residual === null ? "No observed value at this objective." : residual < 0 ? "Optimistic: predicted BPB is lower than observed." : "Pessimistic: observed BPB is lower than predicted."}</p>
    <div class="diagnostic-grid"><div><span>Phase TV</span><strong>${selected.diagnostics.phaseTv.toFixed(3)}</strong></div><div><span>Aggregate TV</span><strong>${selected.diagnostics.aggregateTvToProportional.toFixed(3)}</strong></div><div><span>Aggregate KL</span><strong>${selected.diagnostics.aggregateKlToProportional.toFixed(3)}</strong></div><div><span>Max exposure</span><strong>${selected.diagnostics.maxEpoch.toFixed(2)}e</strong></div><div><span>Support distance</span><strong>${selected.diagnostics.supportDistance.toFixed(3)}</strong></div><div><span>Target bucket</span><strong>${escapeHtml(selected.targetDomain ?? "—")}</strong></div></div>
    <div class="inspector-actions">${nearest && nearest.id !== selected.id ? `<button data-action="select-row" data-value="${escapeHtml(nearest.id)}">Nearest fit: ${escapeHtml(nearest.name)}</button>` : ""}${paired ? `<button data-action="select-row" data-value="${escapeHtml(paired.id)}">Phase counterpart: ${escapeHtml(paired.name)}</button>` : ""}${selected.wandbUrl ? `<a href="${escapeHtml(selected.wandbUrl)}" target="_blank" rel="noreferrer">Open W&B run</a>` : ""}</div>
  `;
}

function renderScatterPanel(): void {
  const points = visiblePoints();
  requiredElement<HTMLElement>("#scatter-caption").textContent = `${points.length} observed checkpoints. Click a point to inspect the policy behind its residual.`;
  renderScatter(requiredElement<HTMLElement>("#scatter"), points, {
    target: swarm().targets[state.target]!,
    view: state.view,
    selectedId: state.selectedId,
    tooltip,
    onSelect: (id) => { state.selectedId = id; renderAll(); },
  });
}

function renderComparison(): void {
  const current = swarm();
  const byId = rowsById();
  const selected = state.selectedId ? byId.get(state.selectedId)?.row : undefined;
  if (!selected) return;
  const baselineSelect = requiredElement<HTMLSelectElement>("#baseline-select");
  const options = current.baselines[state.target]!;
  if (!options.some((option) => option.id === state.baselineId)) state.baselineId = options[0]?.id ?? "";
  baselineSelect.innerHTML = options.map((option) => `<option value="${escapeHtml(option.id)}" ${option.id === state.baselineId ? "selected" : ""}>${escapeHtml(option.label)}</option>`).join("");
  requiredElement<HTMLSelectElement>("#sort-select").value = state.sort;
  const baseline = byId.get(state.baselineId)?.row;
  if (!baseline) return;
  requiredElement<HTMLElement>("#comparison-summary").innerHTML = `${performanceBlock(selected, "Selected mixture")}${performanceBlock(baseline, "Comparison mixture")}`;
  renderMixtureChart(requiredElement<HTMLElement>("#mixture-chart"), { selected, baseline, domains: current.domains, sort: state.sort, tooltip });
}

function parameterRow(parameter: FitParameter): string {
  const interpreted = parameter.transformedValue === null || parameter.transformedValue === undefined
    ? "—"
    : `<strong>${formatParameter(parameter.transformedValue)}</strong><span>${escapeHtml(parameter.transformedLabel ?? "")}${parameter.unit ? ` · ${escapeHtml(parameter.unit)}` : ""}</span>`;
  return `<tr><td><code>${escapeHtml(parameter.symbol)}</code><small>${escapeHtml(parameter.key)}</small></td><td><strong>${formatParameter(parameter.value)}</strong>${parameter.unit && !parameter.transformedLabel ? `<span>${escapeHtml(parameter.unit)}</span>` : ""}</td><td>${interpreted}</td><td>${escapeHtml(parameter.role)}</td></tr>`;
}

function parameterSection(title: string, subtitle: string, parameters: FitParameter[]): string {
  if (parameters.length === 0) return "";
  return `<section class="parameter-section"><div class="parameter-section-title"><h3>${escapeHtml(title)}</h3><span>${escapeHtml(subtitle)}</span></div><div class="parameter-table-wrap"><table><thead><tr><th>Parameter</th><th>Fitted value</th><th>Interpretable value</th><th>Role / interpretation</th></tr></thead><tbody>${parameters.map(parameterRow).join("")}</tbody></table></div></section>`;
}

function renderParameterExplorer(): void {
  const current = swarm();
  const detail = current.fits[state.target]![state.model];
  const globalParameters = detail.parameters.filter((parameter) => parameter.scope === "global");
  const groups = [...new Set(detail.parameters.filter((parameter) => parameter.scope === "group" && parameter.groupLabel).map((parameter) => parameter.groupLabel as string))];
  const domains = [...new Set(detail.parameters.filter((parameter) => parameter.scope === "domain" && parameter.domainId).map((parameter) => parameter.domainId as string))];
  const groupLabel = requiredElement<HTMLElement>("#parameter-group-label");
  const groupSelect = requiredElement<HTMLSelectElement>("#parameter-group");
  groupLabel.hidden = groups.length === 0;
  groupSelect.innerHTML = groups.map((group) => `<option value="${escapeHtml(group)}" ${group === state.parameterGroup ? "selected" : ""}>${escapeHtml(group)}</option>`).join("");
  const domainLabel = requiredElement<HTMLElement>("#parameter-domain-label");
  const domainSelect = requiredElement<HTMLSelectElement>("#parameter-domain");
  domainLabel.hidden = domains.length === 0;
  domainSelect.innerHTML = domains.map((domain) => {
    const metadata = current.domains.find((entry) => entry.id === domain);
    return `<option value="${escapeHtml(domain)}" ${domain === state.parameterDomain ? "selected" : ""}>${escapeHtml(metadata?.label ?? domain)}</option>`;
  }).join("");
  requiredElement<HTMLInputElement>("#parameter-metric").value = current.targets[state.target]!.label;
  const groupParameters = detail.parameters.filter((parameter) => parameter.scope === "group" && parameter.groupLabel === state.parameterGroup);
  const domainParameters = detail.parameters.filter((parameter) => parameter.scope === "domain" && parameter.domainId === state.parameterDomain);
  const domainMetadata = current.domains.find((entry) => entry.id === state.parameterDomain);
  requiredElement<HTMLElement>("#parameter-caption").textContent = `${detail.parameterCount} semantic parameter records. Global terms are always shown; select a group or bucket for local terms.`;
  requiredElement<HTMLElement>("#parameter-table").innerHTML = [
    parameterSection("Global mechanism", "Shared across the fitted response surface", globalParameters),
    parameterSection(state.parameterGroup || "Group terms", "Shared within a semantic GRP family", groupParameters),
    parameterSection(domainMetadata?.label ?? (state.parameterDomain || "Bucket terms"), domainMetadata ? `${domainMetadata.group} · proportional weight ${d3.format(".3%")(domainMetadata.proportionalWeight)}` : "Per-bucket response", domainParameters),
  ].join("");
  requiredElement<HTMLElement>("#fit-caveats").innerHTML = detail.caveats.length > 0
    ? `<span class="control-label">Interpretation caveats</span>${detail.caveats.map((caveat) => `<p>${escapeHtml(caveat)}</p>`).join("")}`
    : '<span class="fit-clean">No model-specific interpretation caveat.</span>';
  requiredElement<HTMLElement>("#fit-protocol").textContent = detail.protocol;
  requiredElement<HTMLElement>("#fit-tuning").textContent = JSON.stringify(detail.tuning, null, 2);
}

function drawSwooshChart(container: HTMLElement, diagnostic: NikeSwooshDiagnostic, variant: "sliceFit" | "overallFit"): void {
  container.replaceChildren();
  const fit = diagnostic[variant];
  const width = 560;
  const height = 330;
  const margin = { top: 48, right: 24, bottom: 58, left: 72 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const allY = [...diagnostic.observed.y, ...fit.prediction];
  const yExtent = d3.extent(allY) as [number, number];
  const yPadding = Math.max((yExtent[1] - yExtent[0]) * 0.08, 0.005);
  const x = d3.scaleLinear().domain([0, 1]).range([0, innerWidth]);
  const y = d3.scaleLinear().domain([yExtent[0] - yPadding, yExtent[1] + yPadding]).nice().range([innerHeight, 0]);
  const svg = d3.select(container).append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img").attr("aria-label", fit.label);
  const plot = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
  plot.append("g").attr("class", "grid-lines").call(d3.axisLeft(y).ticks(6).tickSize(-innerWidth).tickFormat(() => "")).call((group) => group.select(".domain").remove());
  plot.append("g").attr("class", "axis").attr("transform", `translate(0,${innerHeight})`).call(d3.axisBottom(x).ticks(5).tickFormat(d3.format(".0%")));
  plot.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(6).tickFormat(d3.format(".3f")));
  const pairs = diagnostic.grid.map((value, index) => [value, fit.prediction[index]] as [number, number]);
  plot.append("path").datum(pairs).attr("fill", "none").attr("stroke", variant === "sliceFit" ? "#dc6b36" : "#1f7d72").attr("stroke-width", 3).attr("d", d3.line<[number, number]>().x((point) => x(point[0])).y((point) => y(point[1])));
  plot.selectAll("circle.observed").data(diagnostic.observed.x.map((value, index) => [value, diagnostic.observed.y[index]] as [number, number])).join("circle").attr("class", "observed").attr("cx", (point) => x(point[0])).attr("cy", (point) => y(point[1])).attr("r", 4).attr("fill", "#fffdf7").attr("stroke", "#173f5f").attr("stroke-width", 1.5);
  plot.append("circle").attr("cx", x(fit.minimumX)).attr("cy", y(fit.minimumY)).attr("r", 6).attr("fill", "#d7a72d").attr("stroke", "#173f5f");
  svg.append("text").attr("class", "swoosh-title").attr("x", margin.left).attr("y", 24).text(fit.label);
  svg.append("text").attr("class", "swoosh-minimum").attr("x", margin.left).attr("y", 42).text(`Predicted minimum: p1=${d3.format(".1%")(fit.minimumX)}, BPB=${fit.minimumY.toFixed(4)}`);
  svg.append("text").attr("class", "axis-label").attr("x", margin.left + innerWidth / 2).attr("y", height - 12).attr("text-anchor", "middle").text(diagnostic.xLabel);
  svg.append("text").attr("class", "axis-label").attr("transform", "rotate(-90)").attr("x", -(margin.top + innerHeight / 2)).attr("y", 18).attr("text-anchor", "middle").text("BPB");
}

function renderNikeSwoosh(): void {
  const diagnostic = swarm().nikeSwoosh[state.target]?.[state.model];
  const panel = requiredElement<HTMLElement>("#swoosh-panel");
  panel.hidden = diagnostic === undefined;
  if (!diagnostic) return;
  requiredElement<HTMLElement>("#swoosh-caption").textContent = `${diagnostic.sliceDefinition} Compare a fit trained only on that subset with the full-surface fit evaluated on the same slice.`;
  const charts = requiredElement<HTMLElement>("#swoosh-charts");
  charts.innerHTML = '<div class="swoosh-chart"></div><div class="swoosh-chart"></div>';
  const chartElements = charts.querySelectorAll<HTMLElement>(".swoosh-chart");
  drawSwooshChart(chartElements[0]!, diagnostic, "sliceFit");
  drawSwooshChart(chartElements[1]!, diagnostic, "overallFit");
}

function renderMethodology(): void {
  const current = swarm();
  const target = current.targets[state.target]!;
  const noise = target.noiseReference;
  requiredElement<HTMLElement>("#fit-semantics").textContent = current.dataset.fitProtocol;
  requiredElement<HTMLElement>("#noise-unit").textContent = `${noise.differenceStandardDeviation.toFixed(5)} BPB · ${target.noiseLabel ?? `n=${noise.n}`}`;
  requiredElement<HTMLElement>("#phase-budget").textContent = `${d3.format(".0%")(current.dataset.phaseFractions[0])} / ${d3.format(".0%")(current.dataset.phaseFractions[1])}`;
  requiredElement<HTMLElement>("#parameter-count").textContent = `${current.fits[state.target]![state.model].parameterCount} semantic records`;
  requiredElement<HTMLElement>("#provenance").textContent = `Generated ${new Date(data.generatedAt).toLocaleString()} · ${state.swarm} · ${current.dataset.oofSeeds.length} OOF seed(s)`;
  requiredElement<HTMLElement>("#noise-encoding").textContent = target.noiseLabel ?? "Color: absolute error in repeat-difference SDs";
}

function renderWorkspace(): void {
  requiredElement<HTMLElement>("#mixture-workspace").hidden = state.tab !== "mixtures";
  requiredElement<HTMLElement>("#fit-workspace").hidden = state.tab !== "fit";
  if (state.tab === "mixtures") {
    renderViewControls();
    renderFilters();
    renderInspector();
    renderScatterPanel();
    renderComparison();
  } else {
    renderParameterExplorer();
    renderNikeSwoosh();
  }
}

function renderAll(): void {
  ensureState();
  renderSelection();
  renderMetricStrip();
  renderWorkspace();
  renderMethodology();
  updateUrl();
}

app.addEventListener("click", (event) => {
  const button = (event.target as HTMLElement).closest<HTMLElement>("[data-action]");
  if (!button) return;
  const action = button.dataset.action;
  const value = button.dataset.value ?? "";
  if (action === "swarm" && swarmIds.includes(value)) {
    state.swarm = value;
    state.target = Object.keys(data.swarms[value]!.targets)[0]!;
    state.selectedId = null;
    state.baselineId = "";
    state.search = "";
  } else if (action === "target" && Object.keys(swarm().targets).includes(value)) {
    state.target = value;
    state.baselineId = "";
  } else if (action === "model" && isModel(value)) {
    state.model = value;
  } else if (action === "tab" && isTab(value)) {
    state.tab = value;
  } else if (action === "view" && isView(value)) {
    state.view = value;
  } else if (action === "select-row" && rowsById().has(value)) {
    state.selectedId = value;
  } else if (action === "reset-filters") {
    state.showFit = true;
    state.showHeldout = true;
    state.showNoise = false;
    state.hideAliases = true;
    state.phaseFamily = "all";
    state.search = "";
  }
  renderAll();
});

app.addEventListener("change", (event) => {
  const element = event.target as HTMLInputElement | HTMLSelectElement;
  if (element.id === "filter-fit") state.showFit = (element as HTMLInputElement).checked;
  else if (element.id === "filter-heldout") state.showHeldout = (element as HTMLInputElement).checked;
  else if (element.id === "filter-noise") state.showNoise = (element as HTMLInputElement).checked;
  else if (element.id === "filter-alias") state.hideAliases = (element as HTMLInputElement).checked;
  else if (element.id === "phase-filter") state.phaseFamily = element.value as DashboardState["phaseFamily"];
  else if (element.id === "baseline-select") state.baselineId = element.value;
  else if (element.id === "sort-select" && isSort(element.value)) state.sort = element.value;
  else if (element.id === "parameter-domain") state.parameterDomain = element.value;
  else if (element.id === "parameter-group") state.parameterGroup = element.value;
  else if (element.id === "run-search") {
    state.search = element.value;
    const exact = rowsByName().get(element.value);
    if (exact) state.selectedId = exact.id;
  }
  renderAll();
});

let searchTimer = 0;
app.addEventListener("input", (event) => {
  const element = event.target as HTMLInputElement;
  if (element.id !== "run-search") return;
  window.clearTimeout(searchTimer);
  searchTimer = window.setTimeout(() => {
    state.search = element.value;
    const exact = rowsByName().get(element.value);
    if (exact) state.selectedId = exact.id;
    renderAll();
  }, 180);
});

let resizeTimer = 0;
window.addEventListener("resize", () => {
  window.clearTimeout(resizeTimer);
  resizeTimer = window.setTimeout(() => {
    if (state.tab === "mixtures") renderScatterPanel();
    else renderNikeSwoosh();
  }, 120);
});

renderAll();
