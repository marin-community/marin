import * as d3 from "d3";

import rawData from "./generated/dashboard_data.json";
import { renderMath } from "./math";
import { renderMixtureChart } from "./mixture";
import { modelForm } from "./modelForm";
import { renderScatter } from "./scatter";
import "./styles.css";
import type {
  DashboardData,
  DashboardState,
  ExplorerTab,
  FitDetail,
  FitParameter,
  MixtureRow,
  ModelId,
  NikeSwooshDiagnostic,
  PolicyClass,
  PolicyFilter,
  PointDatum,
  SortMode,
  SwarmData,
  ViewMode,
} from "./types";

const data = rawData as unknown as DashboardData;
const hiddenModelIds = new Set<ModelId>([
  "bucket_family_power_separate_heads_family_onset",
  "bucket_family_weibull_shared_onset",
  "bucket_family_weibull_family_replay",
]);
const modelIds = (Object.keys(data.models) as ModelId[]).filter((id) => !hiddenModelIds.has(id));
const modelFamilyIds = [...new Set(modelIds.map((id) => data.models[id]!.familyId))];
const swarmIds = Object.keys(data.swarms);

function requiredElement<T extends HTMLElement>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Missing ${selector} root`);
  return element;
}

function isModel(value: string | null): value is ModelId {
  return value !== null && modelIds.includes(value as ModelId);
}

function modelsInFamily(familyId: string): ModelId[] {
  return modelIds.filter((id) => data.models[id]!.familyId === familyId);
}

function isView(value: string | null): value is ViewMode {
  return value === "prediction" || value === "residual" || value === "standardized" || value === "swoosh";
}

function isSort(value: string | null): value is SortMode {
  return value === "difference" || value === "exposure" || value === "domain";
}

function isTab(value: string | null): value is ExplorerTab {
  return value === "mixtures" || value === "fit";
}

function isPolicyClass(value: string | null): value is PolicyClass {
  return value === "single_phase" || value === "two_phase";
}

function isPolicyFilter(value: string | null): value is PolicyFilter {
  return value === "in_policy" || value === "off_policy" || value === "all";
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
  policyClass: isPolicyClass(query.get("policy")) ? (query.get("policy") as PolicyClass) : "two_phase",
  policyFilter: isPolicyFilter(query.get("points")) ? (query.get("points") as PolicyFilter) : "in_policy",
  model: isModel(query.get("model")) ? (query.get("model") as ModelId) : "separate_heads",
  tab: isTab(query.get("tab")) ? (query.get("tab") as ExplorerTab) : "mixtures",
  view: isView(query.get("view")) ? (query.get("view") as ViewMode) : "prediction",
  selectedId: query.get("selected"),
  baselineId: query.get("baseline") ?? "",
  showFit: query.get("fit") !== "0",
  showHeldout: query.get("heldout") !== "0",
  showNoise: query.get("noise") === "1",
  hideAliases: query.get("aliases") === "0",
  search: query.get("search") ?? "",
  sort: isSort(query.get("sort")) ? (query.get("sort") as SortMode) : "difference",
  parameterDomain: query.get("parameterDomain") ?? "",
  parameterGroup: query.get("parameterGroup") ?? "",
};
const MODEL_DOCK_STORAGE_KEY = "mixture-observatory:model-dock-collapsed";
const MODEL_DOCK_POSITION_STORAGE_KEY = "mixture-observatory:model-dock-position";
const MODEL_DOCK_EDGE_MARGIN = 10;
interface ModelDockPosition {
  left: number;
  top: number;
}

interface ModelDockDrag {
  pointerId: number;
  offsetX: number;
  offsetY: number;
}

function storedModelDockPosition(): ModelDockPosition | null {
  const raw = window.localStorage.getItem(MODEL_DOCK_POSITION_STORAGE_KEY);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as Partial<ModelDockPosition>;
    if (Number.isFinite(parsed.left) && Number.isFinite(parsed.top)) {
      return { left: parsed.left as number, top: parsed.top as number };
    }
  } catch {
    // Discard malformed browser state and retain the default dock position.
  }
  window.localStorage.removeItem(MODEL_DOCK_POSITION_STORAGE_KEY);
  return null;
}

let modelDockCollapsed = window.localStorage.getItem(MODEL_DOCK_STORAGE_KEY) === "1";
let modelDockPosition = storedModelDockPosition();
let modelDockDrag: ModelDockDrag | null = null;

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
        <div class="rail-block"><span class="control-label">03 · Policy class</span><div id="policy-controls" class="vertical-control"></div></div>
      </nav>
      <div class="selection-context">
        <div class="context-topline"><span id="swarm-kicker"></span><span id="phase-stamp"></span></div>
        <h2 id="swarm-title"></h2>
        <p id="swarm-description"></p>
        <div class="model-form" id="model-form"></div>
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
            <div><span class="section-index">A</span><h2 id="scatter-title">Prediction field</h2><p id="scatter-caption"></p></div>
            <div class="view-controls" id="view-controls"></div>
          </div>
          <div class="filter-row" id="scatter-filters">
            <label><input id="filter-fit" type="checkbox" /> Fit designs</label>
            <label><input id="filter-heldout" type="checkbox" /> Full-fit points</label>
            <label><input id="filter-noise" type="checkbox" /> Repeat anchors</label>
            <label><input id="filter-alias" type="checkbox" /> Hide shared alias</label>
            <label class="select-label">Points
              <select id="policy-filter"><option value="in_policy">In-policy</option><option value="off_policy">Off-policy</option><option value="all">All</option></select>
            </label>
            <label class="search-label"><span class="sr-only">Search checkpoints</span>
              <input id="run-search" list="run-search-options" type="search" placeholder="Find a checkpoint…" />
              <datalist id="run-search-options"></datalist>
            </label>
            <button class="quiet-button" data-action="reset-filters">Reset</button>
          </div>
          <div id="scatter" class="scatter"></div>
          <div class="encoding-note" id="scatter-encoding">
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
  <aside id="model-dock" class="model-dock" aria-label="Surrogate model selection">
    <header class="model-dock-heading" title="Drag to reposition">
      <div><span class="control-label">Surrogate model · drag to reposition</span><strong id="model-dock-current"></strong></div>
      <button class="model-dock-toggle" data-action="toggle-model-dock" type="button"></button>
    </header>
    <div class="model-dock-body"><div id="model-controls" class="model-family-grid"></div></div>
  </aside>
  <div id="tooltip" class="tooltip" role="status"></div>
`;

const tooltip = requiredElement<HTMLElement>("#tooltip");

function clampModelDockPosition(position: ModelDockPosition): ModelDockPosition {
  const dock = requiredElement<HTMLElement>("#model-dock");
  const maxLeft = Math.max(MODEL_DOCK_EDGE_MARGIN, window.innerWidth - dock.offsetWidth - MODEL_DOCK_EDGE_MARGIN);
  const maxTop = Math.max(MODEL_DOCK_EDGE_MARGIN, window.innerHeight - dock.offsetHeight - MODEL_DOCK_EDGE_MARGIN);
  return {
    left: Math.min(Math.max(position.left, MODEL_DOCK_EDGE_MARGIN), maxLeft),
    top: Math.min(Math.max(position.top, MODEL_DOCK_EDGE_MARGIN), maxTop),
  };
}

function applyModelDockPosition(): void {
  if (!modelDockPosition) return;
  const dock = requiredElement<HTMLElement>("#model-dock");
  modelDockPosition = clampModelDockPosition(modelDockPosition);
  dock.style.left = `${modelDockPosition.left}px`;
  dock.style.top = `${modelDockPosition.top}px`;
  dock.style.right = "auto";
  dock.style.bottom = "auto";
}

function persistModelDockPosition(): void {
  if (!modelDockPosition) return;
  window.localStorage.setItem(MODEL_DOCK_POSITION_STORAGE_KEY, JSON.stringify(modelDockPosition));
}

function setupModelDockDragging(): void {
  const dock = requiredElement<HTMLElement>("#model-dock");
  const handle = requiredElement<HTMLElement>(".model-dock-heading");
  handle.addEventListener("pointerdown", (event) => {
    if ((event.target as HTMLElement).closest("button")) return;
    const rect = dock.getBoundingClientRect();
    modelDockPosition = { left: rect.left, top: rect.top };
    applyModelDockPosition();
    modelDockDrag = {
      pointerId: event.pointerId,
      offsetX: event.clientX - rect.left,
      offsetY: event.clientY - rect.top,
    };
    handle.setPointerCapture(event.pointerId);
    dock.classList.add("dragging");
    event.preventDefault();
  });
  handle.addEventListener("pointermove", (event) => {
    if (!modelDockDrag || event.pointerId !== modelDockDrag.pointerId) return;
    modelDockPosition = {
      left: event.clientX - modelDockDrag.offsetX,
      top: event.clientY - modelDockDrag.offsetY,
    };
    applyModelDockPosition();
  });
  const finishDrag = (event: PointerEvent): void => {
    if (!modelDockDrag || event.pointerId !== modelDockDrag.pointerId) return;
    modelDockDrag = null;
    dock.classList.remove("dragging");
    if (handle.hasPointerCapture(event.pointerId)) handle.releasePointerCapture(event.pointerId);
    persistModelDockPosition();
  };
  handle.addEventListener("pointerup", finishDrag);
  handle.addEventListener("pointercancel", finishDrag);
}

function swarm(): SwarmData {
  return data.swarms[state.swarm]!;
}

function availablePolicyClasses(): PolicyClass[] {
  return swarm().dataset.policyClasses;
}

function currentFit(): FitDetail {
  const fit = swarm().fits[state.target]?.[state.policyClass]?.[state.model];
  if (!fit) throw new Error(`Missing ${state.swarm}/${state.target}/${state.policyClass}/${state.model} fit`);
  return fit;
}

function currentPredictions() {
  const predictions = swarm().predictions[state.target]?.[state.policyClass]?.[state.model];
  if (!predictions) {
    throw new Error(`Missing ${state.swarm}/${state.target}/${state.policyClass}/${state.model} predictions`);
  }
  return predictions;
}

function currentDiagnostics() {
  const diagnostics = swarm().diagnostics[state.target]?.[state.policyClass]?.[state.model];
  if (!diagnostics) {
    throw new Error(`Missing ${state.swarm}/${state.target}/${state.policyClass}/${state.model} diagnostics`);
  }
  return diagnostics;
}

function swooshDiagnostic(): NikeSwooshDiagnostic | undefined {
  return swarm().nikeSwoosh[state.target]?.[state.policyClass]?.[state.model];
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
  const series = currentPredictions();
  return (fullFit ? series.fullFitPrediction[entry.index] : series.prediction[entry.index]) ?? null;
}

function pointData(): PointDatum[] {
  const current = swarm();
  const noiseScale = Math.max(current.targets[state.target]!.noiseReference.differenceStandardDeviation, 1e-12);
  const series = currentPredictions();
  return current.rows.flatMap((row, rowIndex) => {
    const observed = row.observed[state.target];
    const prediction = series.prediction[rowIndex];
    const fullFitPrediction = series.fullFitPrediction[rowIndex];
    if (observed === null || observed === undefined || prediction === null || prediction === undefined || fullFitPrediction === null || fullFitPrediction === undefined) return [];
    const residual = prediction - observed;
    return [{
      row,
      rowIndex,
      displaySplit: effectiveSplit(row),
      inPolicy: isInPolicy(row),
      observed,
      prediction,
      fullFitPrediction,
      residual,
      standardizedResidual: residual / noiseScale,
    }];
  });
}

function effectiveSplit(row: MixtureRow): PointDatum["displaySplit"] {
  if (row.fitPolicies.includes(state.policyClass)) return "fit";
  if (!isInPolicy(row)) return "off_policy";
  if (row.split === "noise_reference" || row.split === "candidate") return row.split;
  return "heldout";
}

function isInPolicy(row: MixtureRow): boolean {
  return row.policyClasses.includes(state.policyClass);
}

function visiblePoints(): PointDatum[] {
  const search = state.search.trim().toLowerCase();
  return pointData().filter((point) => {
    const split = effectiveSplit(point.row);
    const inPolicy = isInPolicy(point.row);
    if (split === "fit" && !state.showFit) return false;
    if ((split === "heldout" || split === "off_policy") && !state.showHeldout) return false;
    if (split === "noise_reference" && !state.showNoise) return false;
    if (split === "candidate") return false;
    if (state.hideAliases && point.row.isSharedAlias) return false;
    if (state.policyFilter === "in_policy" && !inPolicy) return false;
    if (state.policyFilter === "off_policy" && inPolicy) return false;
    if (search && !`${point.row.name} ${point.row.panel ?? ""} ${point.row.targetDomain ?? ""}`.toLowerCase().includes(search)) return false;
    return true;
  });
}

function chooseDefaultSelection(): string {
  const allPoints = pointData().filter((point) => !point.row.isSharedAlias);
  const filtered = allPoints.filter((point) => {
    if (state.policyFilter === "in_policy") return isInPolicy(point.row);
    if (state.policyFilter === "off_policy") return !isInPolicy(point.row);
    return true;
  });
  const relevant = filtered.length > 0 ? filtered : allPoints;
  const projections = relevant.filter((point) => {
    const split = effectiveSplit(point.row);
    return split === "heldout" || split === "off_policy";
  });
  if (projections.length > 0) {
    return d3.greatest(projections, (point) => Math.abs(point.standardizedResidual))?.row.id ?? projections[0]!.row.id;
  }
  return d3.least(relevant, (point) => point.observed)?.row.id ?? swarm().rows[0]?.id ?? "";
}

function ensureState(): void {
  const current = swarm();
  const targetIds = Object.keys(current.targets);
  if (!targetIds.includes(state.target)) state.target = targetIds[0]!;
  const policies = availablePolicyClasses();
  if (!policies.includes(state.policyClass)) state.policyClass = policies.includes("two_phase") ? "two_phase" : policies[0]!;
  if (state.view === "swoosh" && !swooshDiagnostic()) state.view = "prediction";
  const byId = rowsById();
  const selected = state.selectedId ? byId.get(state.selectedId)?.row : undefined;
  const selectedMatchesFilter = selected
    && (state.policyFilter === "all"
      || (state.policyFilter === "in_policy" && isInPolicy(selected))
      || (state.policyFilter === "off_policy" && !isInPolicy(selected)));
  if (!selectedMatchesFilter) state.selectedId = chooseDefaultSelection();
  const baselineIds = current.baselines[state.target]!.map((option) => option.id);
  if (!baselineIds.includes(state.baselineId)) state.baselineId = baselineIds[0] ?? "";
  const detail = currentFit();
  const domains = detail.parameters.filter((parameter) => parameter.scope === "domain" && parameter.domainId).map((parameter) => parameter.domainId as string);
  const groups = detail.parameters.filter((parameter) => parameter.scope === "group" && parameter.groupLabel).map((parameter) => parameter.groupLabel as string);
  if (!domains.includes(state.parameterDomain)) state.parameterDomain = domains[0] ?? "";
  if (!groups.includes(state.parameterGroup)) state.parameterGroup = groups[0] ?? "";
}

function updateUrl(): void {
  const params = new URLSearchParams();
  params.set("swarm", state.swarm);
  params.set("target", state.target);
  params.set("policy", state.policyClass);
  if (state.policyFilter !== "in_policy") params.set("points", state.policyFilter);
  params.set("model", state.model);
  params.set("tab", state.tab);
  params.set("view", state.view);
  if (state.selectedId) params.set("selected", state.selectedId);
  if (state.baselineId) params.set("baseline", state.baselineId);
  if (!state.showFit) params.set("fit", "0");
  if (!state.showHeldout) params.set("heldout", "0");
  if (state.showNoise) params.set("noise", "1");
  if (state.hideAliases) params.set("aliases", "0");
  if (state.search) params.set("search", state.search);
  params.set("sort", state.sort);
  if (state.parameterDomain) params.set("parameterDomain", state.parameterDomain);
  if (state.parameterGroup) params.set("parameterGroup", state.parameterGroup);
  window.history.replaceState(null, "", `${window.location.pathname}?${params.toString()}`);
}

function renderModelForm(detail: FitDetail): void {
  const form = modelForm(state.model, detail, state.swarm, state.policyClass);
  const element = requiredElement<HTMLElement>("#model-form");
  const chips = [
    `<span><small>model size</small><strong>${detail.parameterCount} parameters</strong></span>`,
    ...form.chips.map((chip) => `<span><small>${escapeHtml(chip.label)}</small><strong>${renderMath(chip.tex, false)}</strong></span>`),
  ];
  element.innerHTML = `
    <header class="model-form-heading">
      <div>
        <span class="control-label">Selected surrogate · functional form</span>
        <h3>${escapeHtml(detail.modelLabel)}</h3>
        <p>${escapeHtml(detail.description)}</p>
      </div>
      <div class="model-form-chips">${chips.join("")}</div>
    </header>
    <section class="model-form-topline">
      <span class="formula-index">Prediction</span>
      <div class="formula-primary">${renderMath(form.topLevelTex, true)}</div>
      <p>${escapeHtml(form.topLevelExplanation)}</p>
    </section>
    <div class="model-form-layers">
      ${form.layers.map((layer) => `
        <article class="formula-layer">
          <span class="formula-index">${escapeHtml(layer.label)}</span>
          <h4>${escapeHtml(layer.title)}</h4>
          <div class="formula-detail">${renderMath(layer.tex, true)}</div>
          <p>${escapeHtml(layer.explanation)}</p>
        </article>
      `).join("")}
    </div>
    <footer>${escapeHtml(detail.protocol)}</footer>
  `;
}

function renderSelection(): void {
  const current = swarm();
  requiredElement<HTMLElement>("#swarm-controls").innerHTML = swarmIds.map((id) =>
    `<button data-action="swarm" data-value="${escapeHtml(id)}" class="${id === state.swarm ? "active" : ""}"><strong>${escapeHtml(data.swarms[id]!.label)}</strong><span>${data.swarms[id]!.dataset.fitDesignCount} rows · ${data.swarms[id]!.domains.length} buckets</span></button>`,
  ).join("");
  requiredElement<HTMLElement>("#target-controls").innerHTML = Object.keys(current.targets).map((id) =>
    `<button data-action="target" data-value="${escapeHtml(id)}" class="${id === state.target ? "active" : ""}">${escapeHtml(current.targets[id]!.label)}</button>`,
  ).join("");
  requiredElement<HTMLElement>("#policy-controls").innerHTML = availablePolicyClasses().map((policy) => {
    const label = policy === "single_phase" ? "One phase" : "Two phase";
    const descriptor = policy === "single_phase" ? "Tied schedule" : "Independent phase weights";
    const count = current.dataset.policyFitCounts[policy] ?? 0;
    return `<button data-action="policy" data-value="${policy}" class="${policy === state.policyClass ? "active" : ""}"><strong>${label}</strong><span>${descriptor} · ${count} fit rows</span></button>`;
  }).join("");
  const selectedFamily = data.models[state.model]!.familyId;
  requiredElement<HTMLElement>("#model-controls").innerHTML = modelFamilyIds.map((familyId) => {
    const first = modelsInFamily(familyId)[0]!;
    const variants = modelsInFamily(familyId);
    return `<section class="model-family-row ${familyId === selectedFamily ? "active" : ""}">
      <header>
        <strong>${escapeHtml(data.models[first]!.familyLabel)}</strong>
        <span>${variants.length} variant${variants.length === 1 ? "" : "s"}</span>
      </header>
      <div class="model-variant-buttons">
        ${variants.map((id) => `<button data-action="model" data-value="${id}" class="${id === state.model ? "active" : ""}">${escapeHtml(data.models[id]!.variantLabel)}</button>`).join("")}
      </div>
    </section>`;
  }).join("");
  const modelDock = requiredElement<HTMLElement>("#model-dock");
  modelDock.classList.toggle("collapsed", modelDockCollapsed);
  applyModelDockPosition();
  requiredElement<HTMLElement>("#model-dock-current").textContent = data.models[state.model]!.label;
  const modelDockToggle = requiredElement<HTMLButtonElement>(".model-dock-toggle");
  modelDockToggle.textContent = modelDockCollapsed ? "Expand" : "Minimize";
  modelDockToggle.setAttribute("aria-expanded", String(!modelDockCollapsed));
  const fitCount = current.dataset.policyFitCounts[state.policyClass] ?? 0;
  requiredElement<HTMLElement>("#swarm-kicker").textContent = `${fitCount} ${state.policyClass.replace("_", "-")} fit rows · ${current.domains.length} buckets`;
  requiredElement<HTMLElement>("#phase-stamp").textContent = state.policyClass === "single_phase"
    ? "phase-tied policy"
    : `${d3.format(".0%")(current.dataset.phaseFractions[0])} / ${d3.format(".0%")(current.dataset.phaseFractions[1])} phases`;
  requiredElement<HTMLElement>("#swarm-title").textContent = current.label;
  requiredElement<HTMLElement>("#swarm-description").textContent = current.description;
  const detail = currentFit();
  renderModelForm(detail);
  document.querySelectorAll<HTMLButtonElement>("[data-action='tab']").forEach((button) => {
    const active = button.dataset.value === state.tab;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  const stamp = requiredElement<HTMLElement>("#dataset-stamp");
  const heldoutCount = current.rows.filter((row) => isInPolicy(row) && effectiveSplit(row) === "heldout" && row.observed[state.target] !== null).length;
  stamp.innerHTML = `<span>FIT ROWS</span><strong>${fitCount}</strong><span>HELDOUT</span><strong>${heldoutCount}</strong><span>DOMAINS</span><strong>${current.domains.length}</strong>`;
}

function renderMetricStrip(): void {
  const current = swarm();
  const diagnostics = currentDiagnostics();
  const fit = currentFit();
  const foldCount = current.dataset.oofSeeds.length * 5;
  const cards: Array<{
    label: string;
    value: number | null;
    detail: string;
    decimals: number;
    definition: string;
  }> = [
    {
      label: "OOF RMSE",
      value: diagnostics.fitOof.rmse,
      detail: `${diagnostics.fitOof.n} fit designs`,
      decimals: 5,
      definition: "Root mean squared error of the out-of-fold predictions.",
    },
    {
      label: "OOF Spearman",
      value: diagnostics.fitOof.spearman,
      detail: "Held-out predictions",
      decimals: 3,
      definition: "Spearman rank correlation between observed and out-of-fold predicted BPB.",
    },
    {
      label: "OOF Regret@1",
      value: diagnostics.fitOof.foldMeanRegretAt1,
      detail: `Mean over ${foldCount} held-out folds`,
      decimals: 5,
      definition: "Within each held-out fold, observed BPB of the predicted winner minus the best observed BPB; then averaged across folds.",
    },
    {
      label: "OOF tail optimism",
      value: diagnostics.fitOof.lowerTailOptimism,
      detail: `Best predicted ${diagnostics.fitOof.lowerTailCount}/${diagnostics.fitOof.n}`,
      decimals: 5,
      definition: "Mean positive observed-minus-predicted BPB in the best predicted 15% of rows (at least five). Positive values mean the surrogate is overoptimistic.",
    },
    {
      label: "OOF low-tail RMSE",
      value: diagnostics.fitOof.lowTailRmse,
      detail: "Same predicted lower tail",
      decimals: 5,
      definition: "RMSE in the best predicted 15% of rows (at least five), the region used for mixture selection.",
    },
    {
      label: "Train RMSE",
      value: fit.diagnostics.train.rmse,
      detail: "Full fit",
      decimals: 5,
      definition: "In-sample RMSE after refitting the selected surrogate on the complete policy-matched panel.",
    },
    {
      label: "Train Spearman",
      value: fit.diagnostics.train.spearman,
      detail: "Full fit",
      decimals: 3,
      definition: "In-sample Spearman correlation after refitting on the complete policy-matched panel.",
    },
  ];
  if (diagnostics.heldout.n > 0) {
    cards.push(
      {
        label: "Heldout RMSE",
        value: diagnostics.heldout.rmse,
        detail: `${diagnostics.heldout.n} checkpoints`,
        decimals: 5,
        definition: "RMSE on independent checkpoints projected by the full fit without refitting.",
      },
      {
        label: "Heldout Spearman",
        value: diagnostics.heldout.spearman,
        detail: "No refitting",
        decimals: 3,
        definition: "Spearman correlation on independent checkpoints projected by the full fit.",
      },
      {
        label: "Heldout Regret@1",
        value: diagnostics.heldout.regretAt1,
        detail: "One heldout selection set",
        decimals: 5,
        definition: "Observed BPB of the predicted heldout winner minus the best observed BPB in the heldout set.",
      },
      {
        label: "Heldout tail optimism",
        value: diagnostics.heldout.lowerTailOptimism,
        detail: `Best predicted ${diagnostics.heldout.lowerTailCount}/${diagnostics.heldout.n}`,
        decimals: 5,
        definition: "Mean positive observed-minus-predicted BPB in the best predicted 15% of heldout checkpoints.",
      },
      {
        label: "Heldout low-tail RMSE",
        value: diagnostics.heldout.lowTailRmse,
        detail: "Same predicted lower tail",
        decimals: 5,
        definition: "RMSE in the best predicted 15% of heldout checkpoints.",
      },
    );
  } else {
    cards.push(
      {
        label: "Rows",
        value: current.dataset.policyFitCounts[state.policyClass] ?? 0,
        detail: `${state.policyClass.replace("_", "-")} fit panel`,
        decimals: 0,
        definition: "Number of policy-matched observations used to fit this surrogate.",
      },
      {
        label: "Parameters",
        value: fit.parameterCount,
        detail: "Semantic records",
        decimals: 0,
        definition: "Number of fitted parameter records exposed in the Fit Explorer.",
      },
    );
  }
  requiredElement<HTMLElement>("#metric-strip").innerHTML = cards.map((card) =>
    `<div class="metric-card" title="${escapeHtml(card.definition)}"><span>${escapeHtml(card.label)}</span><strong>${formatMetric(card.value, card.decimals)}</strong><small>${escapeHtml(card.detail)}</small></div>`,
  ).join("");
}

function renderViewControls(): void {
  const views: Array<[ViewMode, string]> = [
    ["prediction", "Predicted ↔ observed"],
    ["residual", "Residual"],
    ["standardized", "Scale-normalized"],
  ];
  if (swooshDiagnostic()) views.push(["swoosh", "Nike swoosh"]);
  requiredElement<HTMLElement>("#view-controls").innerHTML = views.map(([view, label]) =>
    `<button data-action="view" data-value="${view}" class="${view === state.view ? "active" : ""}">${label}</button>`,
  ).join("");
}

function renderFilters(): void {
  const current = swarm();
  requiredElement<HTMLInputElement>("#filter-fit").checked = state.showFit;
  requiredElement<HTMLInputElement>("#filter-heldout").checked = state.showHeldout;
  requiredElement<HTMLInputElement>("#filter-noise").checked = state.showNoise;
  requiredElement<HTMLInputElement>("#filter-alias").checked = state.hideAliases;
  requiredElement<HTMLSelectElement>("#policy-filter").value = state.policyFilter;
  requiredElement<HTMLInputElement>("#run-search").value = state.search;
  requiredElement<HTMLDataListElement>("#run-search-options").innerHTML = current.rows.filter((row) => row.observed[state.target] !== null).map((row) => `<option value="${escapeHtml(row.name)}"></option>`).join("");
}

function splitLabel(row: MixtureRow): string {
  const split = effectiveSplit(row);
  if (split === "fit") return "Fit design · OOF";
  if (split === "heldout") return "In-policy heldout · full fit";
  if (split === "off_policy") return "Off-policy projection · full fit";
  if (split === "noise_reference") return "Repeat anchor · full fit";
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
  const selectedSplit = effectiveSplit(selected);
  requiredElement<HTMLElement>("#inspector").innerHTML = `
    <div class="inspector-topline"><span class="section-index">SELECTED</span><span>${escapeHtml(splitLabel(selected))}</span></div>
    <h2>${escapeHtml(selected.name)}</h2>
    <div class="badge-row"><span class="badge ${selected.phaseFamily === "single_phase" ? "orange" : "blue"}">${escapeHtml(selected.phaseFamily?.replace("_", " ") ?? "unknown phase")}</span><span class="badge ${isInPolicy(selected) ? "" : "warning"}">${isInPolicy(selected) ? "in-policy" : "off-policy"}</span><span class="badge">${escapeHtml(selected.panel ?? "unclassified")}</span>${selected.isSharedAlias ? '<span class="badge warning">shared alias</span>' : ""}</div>
    <div class="selected-performance ${residual !== null && residual < 0 ? "danger" : ""}">
      <div><span>Observed</span><strong>${formatMetric(observed, 6)}</strong></div><div><span>${selectedSplit === "fit" ? "OOF prediction" : "Full-fit projection"}</span><strong>${formatMetric(predicted, 6)}</strong></div><div><span>Residual</span><strong>${formatSigned(residual, 6)}</strong></div><div><span>Scale units</span><strong>${formatSigned(residual === null ? null : residual / scale, 2)}×</strong></div>
    </div>
    <p class="residual-reading">${residual === null ? "No observed value at this objective." : residual < 0 ? "Optimistic: predicted BPB is lower than observed." : "Pessimistic: observed BPB is lower than predicted."}</p>
    <div class="diagnostic-grid"><div><span>Phase TV</span><strong>${selected.diagnostics.phaseTv.toFixed(3)}</strong></div><div><span>Aggregate TV</span><strong>${selected.diagnostics.aggregateTvToProportional.toFixed(3)}</strong></div><div><span>Aggregate KL</span><strong>${selected.diagnostics.aggregateKlToProportional.toFixed(3)}</strong></div><div><span>Max exposure</span><strong>${selected.diagnostics.maxEpoch.toFixed(2)}e</strong></div><div><span>Support distance</span><strong>${selected.diagnostics.supportDistance.toFixed(3)}</strong></div><div><span>Target bucket</span><strong>${escapeHtml(selected.targetDomain ?? "—")}</strong></div></div>
    <div class="inspector-actions">${nearest && nearest.id !== selected.id ? `<button data-action="select-row" data-value="${escapeHtml(nearest.id)}">Nearest fit: ${escapeHtml(nearest.name)}</button>` : ""}${paired ? `<button data-action="select-row" data-value="${escapeHtml(paired.id)}">Phase counterpart: ${escapeHtml(paired.name)}</button>` : ""}${selected.wandbUrl ? `<a href="${escapeHtml(selected.wandbUrl)}" target="_blank" rel="noreferrer">Open W&B run</a>` : ""}</div>
  `;
}

function renderScatterPanel(): void {
  const scatter = requiredElement<HTMLElement>("#scatter");
  const filters = requiredElement<HTMLElement>("#scatter-filters");
  const encoding = requiredElement<HTMLElement>("#scatter-encoding");
  const diagnostic = swooshDiagnostic();
  if (state.view === "swoosh" && diagnostic) {
    requiredElement<HTMLElement>("#scatter-title").textContent = "Nike-swoosh response";
    requiredElement<HTMLElement>("#scatter-caption").textContent = `${diagnostic.sliceDefinition} The left curve is refit only on this slice; the right curve evaluates the full-surface fit on the identical slice.`;
    filters.hidden = true;
    encoding.hidden = true;
    scatter.classList.add("swoosh-mode");
    scatter.innerHTML = '<div class="swoosh-charts swoosh-charts-inline"><div class="swoosh-chart"></div><div class="swoosh-chart"></div></div><p class="swoosh-interaction-note">Observed checkpoints are shared across both views. Select any point to inspect its exact two-phase policy below.</p>';
    const chartElements = scatter.querySelectorAll<HTMLElement>(".swoosh-chart");
    const options: SwooshChartOptions = {
      selectedId: state.selectedId,
      tooltip,
      onSelect: (id) => { state.selectedId = id; renderAll(); },
    };
    drawSwooshChart(chartElements[0]!, diagnostic, "sliceFit", options);
    drawSwooshChart(chartElements[1]!, diagnostic, "overallFit", options);
    return;
  }

  const points = visiblePoints();
  requiredElement<HTMLElement>("#scatter-title").textContent = "Prediction field";
  const pointScope = state.policyFilter === "in_policy"
    ? `${state.policyClass.replace("_", "-")} policies`
    : state.policyFilter === "off_policy"
      ? `off-policy projections from the ${state.policyClass.replace("_", "-")} fit`
      : "in-policy and off-policy checkpoints";
  requiredElement<HTMLElement>("#scatter-caption").textContent = `${points.length} ${pointScope}. Click a point to inspect the policy behind its residual.`;
  filters.hidden = false;
  encoding.hidden = false;
  scatter.classList.remove("swoosh-mode");
  renderScatter(scatter, points, {
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
  const detail = currentFit();
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

interface SwooshChartOptions {
  selectedId: string | null;
  tooltip: HTMLElement;
  onSelect: (id: string) => void;
}

interface SwooshPoint {
  x: number;
  y: number;
  rowId: string;
}

function showSwooshTooltip(
  event: MouseEvent,
  point: SwooshPoint,
  prediction: number,
  diagnostic: NikeSwooshDiagnostic,
  tooltipElement: HTMLElement,
): void {
  const row = rowsById().get(point.rowId)?.row;
  tooltipElement.innerHTML = `
    <div class="tooltip-kicker">Observed slice checkpoint</div>
    <strong>${escapeHtml(row?.name ?? point.rowId)}</strong>
    <dl>
      <dt>${escapeHtml(diagnostic.xLabel)}</dt><dd>${d3.format(".2%")(point.x)}</dd>
      <dt>Observed BPB</dt><dd>${point.y.toFixed(6)}</dd>
      <dt>Curve prediction</dt><dd>${prediction.toFixed(6)}</dd>
      <dt>Residual</dt><dd>${formatSigned(prediction - point.y, 6)}</dd>
    </dl>`;
  tooltipElement.classList.add("visible");
  tooltipElement.style.left = `${event.clientX + 16}px`;
  tooltipElement.style.top = `${event.clientY + 16}px`;
}

function drawSwooshChart(
  container: HTMLElement,
  diagnostic: NikeSwooshDiagnostic,
  variant: "sliceFit" | "overallFit",
  options: SwooshChartOptions,
): void {
  container.replaceChildren();
  const fit = diagnostic[variant];
  const width = 560;
  const height = 330;
  const margin = { top: 48, right: 24, bottom: 58, left: 72 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const allY = [
    ...diagnostic.observed.y,
    ...diagnostic.sliceFit.prediction,
    ...diagnostic.overallFit.prediction,
  ];
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
  const observed = diagnostic.observed.x.map((value, index): SwooshPoint => ({
    x: value,
    y: diagnostic.observed.y[index]!,
    rowId: diagnostic.observed.rowIds[index]!,
  }));
  const nearestPrediction = (value: number): number => {
    const index = d3.leastIndex(diagnostic.grid, (candidate) => Math.abs(candidate - value));
    if (index === undefined || index < 0) throw new Error("Nike-swoosh diagnostic has an empty prediction grid");
    return fit.prediction[index]!;
  };
  plot
    .selectAll("circle.observed")
    .data(observed)
    .join("circle")
    .attr("class", "observed")
    .attr("cx", (point) => x(point.x))
    .attr("cy", (point) => y(point.y))
    .attr("r", (point) => (point.rowId === options.selectedId ? 6 : 4))
    .attr("fill", (point) => (point.rowId === options.selectedId ? "#f1bb31" : "#fffdf7"))
    .attr("stroke", "#173f5f")
    .attr("stroke-width", (point) => (point.rowId === options.selectedId ? 2.5 : 1.5))
    .attr("tabindex", 0)
    .attr("role", "button")
    .attr("aria-label", (point) => `${rowsById().get(point.rowId)?.row.name ?? point.rowId}; ${diagnostic.xLabel} ${point.x}; observed BPB ${point.y}`)
    .style("cursor", "pointer")
    .on("mouseenter", (event, point) => showSwooshTooltip(event as MouseEvent, point, nearestPrediction(point.x), diagnostic, options.tooltip))
    .on("mousemove", (event, point) => showSwooshTooltip(event as MouseEvent, point, nearestPrediction(point.x), diagnostic, options.tooltip))
    .on("mouseleave", () => options.tooltip.classList.remove("visible"))
    .on("click", (_event, point) => options.onSelect(point.rowId))
    .on("keydown", (event, point) => {
      if ((event as KeyboardEvent).key === "Enter" || (event as KeyboardEvent).key === " ") {
        event.preventDefault();
        options.onSelect(point.rowId);
      }
    });
  plot.append("circle").attr("cx", x(fit.minimumX)).attr("cy", y(fit.minimumY)).attr("r", 6).attr("fill", "#d7a72d").attr("stroke", "#173f5f");
  svg.append("text").attr("class", "swoosh-title").attr("x", margin.left).attr("y", 24).text(fit.label);
  svg.append("text").attr("class", "swoosh-minimum").attr("x", margin.left).attr("y", 42).text(`Predicted minimum: p1=${d3.format(".1%")(fit.minimumX)}, BPB=${fit.minimumY.toFixed(4)}`);
  svg.append("text").attr("class", "axis-label").attr("x", margin.left + innerWidth / 2).attr("y", height - 12).attr("text-anchor", "middle").text(diagnostic.xLabel);
  svg.append("text").attr("class", "axis-label").attr("transform", "rotate(-90)").attr("x", -(margin.top + innerHeight / 2)).attr("y", 18).attr("text-anchor", "middle").text(diagnostic.yLabel);
}

function renderNikeSwoosh(): void {
  const diagnostic = swooshDiagnostic();
  const panel = requiredElement<HTMLElement>("#swoosh-panel");
  panel.hidden = diagnostic === undefined;
  if (!diagnostic) return;
  requiredElement<HTMLElement>("#swoosh-caption").textContent = `${diagnostic.sliceDefinition} Compare a fit trained only on that subset with the full-surface fit evaluated on the same slice.`;
  const charts = requiredElement<HTMLElement>("#swoosh-charts");
  charts.innerHTML = '<div class="swoosh-chart"></div><div class="swoosh-chart"></div>';
  const chartElements = charts.querySelectorAll<HTMLElement>(".swoosh-chart");
  const options: SwooshChartOptions = {
    selectedId: state.selectedId,
    tooltip,
    onSelect: (id) => { state.selectedId = id; renderAll(); },
  };
  drawSwooshChart(chartElements[0]!, diagnostic, "sliceFit", options);
  drawSwooshChart(chartElements[1]!, diagnostic, "overallFit", options);
}

function renderMethodology(): void {
  const current = swarm();
  const target = current.targets[state.target]!;
  const noise = target.noiseReference;
  requiredElement<HTMLElement>("#fit-semantics").textContent = currentFit().protocol;
  requiredElement<HTMLElement>("#noise-unit").textContent = `${noise.differenceStandardDeviation.toFixed(5)} BPB · ${target.noiseLabel ?? `n=${noise.n}`}`;
  requiredElement<HTMLElement>("#phase-budget").textContent = state.policyClass === "single_phase"
    ? "tied weights across both phases"
    : `${d3.format(".0%")(current.dataset.phaseFractions[0])} / ${d3.format(".0%")(current.dataset.phaseFractions[1])}`;
  requiredElement<HTMLElement>("#parameter-count").textContent = `${currentFit().parameterCount} semantic records`;
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
    const policies = data.swarms[value]!.dataset.policyClasses;
    if (!policies.includes(state.policyClass)) state.policyClass = policies.includes("two_phase") ? "two_phase" : policies[0]!;
    state.selectedId = null;
    state.baselineId = "";
    state.search = "";
  } else if (action === "target" && Object.keys(swarm().targets).includes(value)) {
    state.target = value;
    state.baselineId = "";
  } else if (action === "policy" && isPolicyClass(value) && availablePolicyClasses().includes(value)) {
    state.policyClass = value;
    state.selectedId = null;
    state.baselineId = "";
  } else if (action === "model" && isModel(value)) {
    state.model = value;
  } else if (action === "toggle-model-dock") {
    modelDockCollapsed = !modelDockCollapsed;
    window.localStorage.setItem(MODEL_DOCK_STORAGE_KEY, modelDockCollapsed ? "1" : "0");
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
    state.policyFilter = "in_policy";
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
  else if (element.id === "policy-filter" && isPolicyFilter(element.value)) state.policyFilter = element.value;
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
    applyModelDockPosition();
    persistModelDockPosition();
    if (state.tab === "mixtures") renderScatterPanel();
    else renderNikeSwoosh();
  }, 120);
});

setupModelDockDragging();
renderAll();
