import * as d3 from "d3";

import type { MixtureRow, PhasePopulationMetadata, TargetMetadata } from "./types";

export interface PhasePopulationPoint {
  row: MixtureRow;
  anchor: MixtureRow;
  observed: number;
  anchorObserved: number;
  delta: number;
  standardizedDelta: number;
}

export interface PhasePopulationSummary {
  count: number;
  betterCount: number;
  strongGainCount: number;
  strongLossCount: number;
  meanDelta: number;
  medianDelta: number;
  bestDelta: number;
}

interface PhasePopulationChartOptions {
  target: TargetMetadata;
  differenceStandardDeviation: number;
  selectedId: string | null;
  tooltip: HTMLElement;
  onSelect: (id: string | null) => void;
}

const ANCHOR_ORDER = ["uncheatable_frontier", "table9_frontier"];
const INK = "#183247";
const MUTED = "#72818c";
const GRID = "#d8d3c8";
const NOISE_BAND = "#ebe7dc";

function anchorKey(metadata: PhasePopulationMetadata): string {
  return `${metadata.panelId}::${metadata.anchorId}::${metadata.seedBlock}`;
}

function directionKey(metadata: PhasePopulationMetadata): string {
  return `${metadata.panelId}::${metadata.anchorId}::${metadata.contrastFamily}::${metadata.directionId}`;
}

export function phasePopulationPanelLabel(panelId: string): string {
  if (panelId === "delphi-3e18-frontier-random-phase-population") return "Isotropic frontier population";
  if (panelId === "delphi-3e18-aggressive-phase-asymmetry") return "Aggressive asymmetry panel";
  return panelId.replaceAll("_", " ").replaceAll("-", " ");
}

export function phasePopulationFamilyLabel(family: string): string {
  if (family === "random_isotropic") return "Isotropic direction";
  if (family === "balanced_partition") return "Balanced antithetic partition";
  if (family === "handcrafted_late_quality") return "Handcrafted late-quality";
  if (family === "dolmino_late_continuum") return "Dolmino-late continuum";
  if (family === "center_control") return "Tied control";
  return family.replaceAll("_", " ");
}

export function phasePopulationAnchorLabel(anchorId: string): string {
  if (anchorId === "uncheatable_frontier") return "Uncheatable frontier anchor";
  if (anchorId === "table9_frontier") return "Table-9 frontier anchor";
  return anchorId.replaceAll("_", " ");
}

export function summarizePhasePopulation(
  points: readonly PhasePopulationPoint[],
  strongEffectThreshold = 1.96,
): PhasePopulationSummary {
  if (points.length === 0) {
    throw new Error("Cannot summarize an empty phase population");
  }
  return {
    count: points.length,
    betterCount: points.filter((point) => point.delta < 0).length,
    strongGainCount: points.filter((point) => point.standardizedDelta < -strongEffectThreshold).length,
    strongLossCount: points.filter((point) => point.standardizedDelta > strongEffectThreshold).length,
    meanDelta: d3.mean(points, (point) => point.delta) ?? 0,
    medianDelta: d3.median(points, (point) => point.delta) ?? 0,
    bestDelta: d3.min(points, (point) => point.delta) ?? 0,
  };
}

export function phasePopulationPoints(
  rows: readonly MixtureRow[],
  targetId: string,
  differenceStandardDeviation: number,
): PhasePopulationPoint[] {
  const populationRows = rows.filter((row) => row.phasePopulation);
  const controls = new Map<string, MixtureRow>();
  for (const row of populationRows) {
    const metadata = row.phasePopulation!;
    if (metadata.contrastFamily !== "center_control") continue;
    const key = anchorKey(metadata);
    if (controls.has(key)) throw new Error(`Duplicate phase-population control ${key}`);
    controls.set(key, row);
  }

  const scale = Math.max(differenceStandardDeviation, 1e-12);
  return populationRows.flatMap((row) => {
    const metadata = row.phasePopulation!;
    if (metadata.contrastFamily === "center_control") return [];
    const anchor = controls.get(anchorKey(metadata));
    if (!anchor) throw new Error(`Missing seed-matched tied control for ${row.name}`);
    const observed = row.observed[targetId];
    const anchorObserved = anchor.observed[targetId];
    if (observed === null || observed === undefined || anchorObserved === null || anchorObserved === undefined) {
      return [];
    }
    const delta = observed - anchorObserved;
    return [{ row, anchor, observed, anchorObserved, delta, standardizedDelta: delta / scale }];
  });
}

function deterministicJitter(candidateId: string, width: number): number {
  let hash = 2166136261;
  for (const character of candidateId) {
    hash ^= character.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return ((((hash >>> 0) % 1001) / 1000) - 0.5) * width;
}

function markerType(family: string): d3.SymbolType {
  if (family === "balanced_partition") return d3.symbolDiamond;
  if (family === "handcrafted_late_quality") return d3.symbolSquare;
  if (family === "dolmino_late_continuum") return d3.symbolTriangle;
  if (family === "random_isotropic") return d3.symbolCircle;
  return d3.symbolCross;
}

function optionalPercent(value: number | null): string {
  return value === null ? "—" : d3.format(".0%")(value);
}

function optionalNumber(value: number | null, format: string): string {
  return value === null ? "—" : d3.format(format)(value);
}

function showTooltip(
  event: MouseEvent,
  datum: PhasePopulationPoint,
  target: TargetMetadata,
  tooltip: HTMLElement,
): void {
  const metadata = datum.row.phasePopulation!;
  tooltip.innerHTML = `
    <div class="tooltip-kicker">${phasePopulationAnchorLabel(metadata.anchorId)} · ${phasePopulationFamilyLabel(metadata.contrastFamily)}</div>
    <strong>${datum.row.name}</strong>
    <dl>
      <dt>Panel</dt><dd>${phasePopulationPanelLabel(metadata.panelId)}</dd>
      <dt>Phase TV</dt><dd>${datum.row.diagnostics.phaseTv.toFixed(4)}</dd>
      <dt>Design level</dt><dd>${metadata.radiusFraction === null ? optionalNumber(metadata.targetPhaseTv, ".2f") : optionalPercent(metadata.radiusFraction)}</dd>
      <dt>Two-phase policy</dt><dd>${datum.observed.toFixed(6)} ${target.label}</dd>
      <dt>Tied control</dt><dd>${datum.anchorObserved.toFixed(6)}</dd>
      <dt>Two-phase − tied</dt><dd>${d3.format("+.6f")(datum.delta)} BPB</dd>
      <dt>Noise units</dt><dd>${d3.format("+.2f")(datum.standardizedDelta)}×</dd>
      <dt>Direction</dt><dd>${metadata.directionLabel}</dd>
      <dt>Sign</dt><dd>${metadata.sign || "—"}</dd>
      <dt>Seed block</dt><dd>${metadata.seedBlock}</dd>
    </dl>`;
  tooltip.classList.add("visible");
  tooltip.style.left = `${event.clientX + 16}px`;
  tooltip.style.top = `${event.clientY + 16}px`;
}

function hideTooltip(tooltip: HTMLElement): void {
  tooltip.classList.remove("visible");
}

export function renderPhasePopulationChart(
  container: HTMLElement,
  points: PhasePopulationPoint[],
  options: PhasePopulationChartOptions,
): void {
  container.replaceChildren();
  if (points.length === 0) {
    container.innerHTML = '<div class="empty-state">No phase-population policies match the selected filters.</div>';
    return;
  }

  const anchorIds = ANCHOR_ORDER.filter((anchorId) =>
    points.some((point) => point.row.phasePopulation?.anchorId === anchorId),
  );
  const width = Math.max(container.clientWidth, 980);
  const height = 570;
  const margin = { top: 78, right: 34, bottom: 76, left: 84 };
  const facetGap = 74;
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const facetWidth = (innerWidth - facetGap * (anchorIds.length - 1)) / anchorIds.length;
  const noiseScale = Math.max(options.differenceStandardDeviation, 1e-12);
  const maximumMagnitude = Math.max(noiseScale * 1.25, d3.max(points, (point) => Math.abs(point.delta)) ?? 0);
  const preliminaryY = d3.scaleLinear().domain([-maximumMagnitude * 1.08, maximumMagnitude * 1.08]).nice(8);
  const preliminaryDomain = preliminaryY.domain();
  const symmetricLimit = Math.max(Math.abs(preliminaryDomain[0]!), Math.abs(preliminaryDomain[1]!));
  const y = d3.scaleLinear().domain([-symmetricLimit, symmetricLimit]).range([innerHeight, 0]);
  const maximumTv = Math.max(0.1, d3.max(points, (point) => point.row.diagnostics.phaseTv) ?? 0);
  const colorLimit = Math.max(
    noiseScale,
    d3.quantile(points.map((point) => Math.abs(point.delta)).sort(d3.ascending), 0.95) ?? noiseScale,
  );
  const color = d3.scaleDiverging<string>(
    [-colorLimit, 0, colorLimit],
    (value) => d3.interpolateRdYlGn(1 - value),
  );

  const svg = d3
    .select(container)
    .append("svg")
    .attr("class", "population-svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("role", "img")
    .attr("aria-label", `${options.target.label} paired BPB effects for ${points.length} phase schedules`)
    .on("click", () => options.onSelect(null));
  const root = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
  const allPointSelections: Array<d3.Selection<SVGPathElement, PhasePopulationPoint, SVGGElement, unknown>> = [];

  for (const [facetIndex, anchorId] of anchorIds.entries()) {
    const offset = facetIndex * (facetWidth + facetGap);
    const facet = root.append("g").attr("transform", `translate(${offset},0)`);
    const facetPoints = points.filter((point) => point.row.phasePopulation?.anchorId === anchorId);
    const x = d3.scaleLinear().domain([0, maximumTv * 1.04]).nice(6).range([0, facetWidth]);
    const median = d3.median(facetPoints, (point) => point.delta) ?? 0;
    const best = d3.min(facetPoints, (point) => point.delta) ?? 0;

    facet
      .append("rect")
      .attr("x", 0)
      .attr("y", y(noiseScale))
      .attr("width", facetWidth)
      .attr("height", y(-noiseScale) - y(noiseScale))
      .attr("fill", NOISE_BAND)
      .attr("opacity", 0.72);
    facet
      .append("g")
      .attr("class", "grid-lines")
      .call(d3.axisLeft(y).ticks(8).tickSize(-facetWidth).tickFormat(() => ""))
      .call((group) => group.select(".domain").remove())
      .call((group) => group.selectAll("line").attr("stroke", GRID).attr("stroke-dasharray", "2,4"));
    facet
      .append("line")
      .attr("x1", 0)
      .attr("x2", facetWidth)
      .attr("y1", y(0))
      .attr("y2", y(0))
      .attr("stroke", INK)
      .attr("stroke-width", 2);
    facet
      .append("text")
      .attr("class", "population-tied-label")
      .attr("x", facetWidth - 4)
      .attr("y", y(0) - 8)
      .attr("text-anchor", "end")
      .text("tied control · 0 BPB");
    facet
      .append("g")
      .attr("class", "axis")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x).ticks(6).tickFormat(d3.format(".2f")));
    if (facetIndex === 0) {
      facet.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(8).tickFormat(d3.format("+.3f")));
      facet
        .append("text")
        .attr("class", "axis-label")
        .attr("transform", "rotate(-90)")
        .attr("x", -innerHeight / 2)
        .attr("y", -64)
        .attr("text-anchor", "middle")
        .text("Two-phase − seed-matched tied control (BPB)");
    }
    facet
      .append("text")
      .attr("class", "population-facet-title")
      .attr("x", facetWidth / 2)
      .attr("y", -42)
      .attr("text-anchor", "middle")
      .text(phasePopulationAnchorLabel(anchorId));
    facet
      .append("text")
      .attr("class", "population-facet-summary")
      .attr("x", facetWidth / 2)
      .attr("y", -20)
      .attr("text-anchor", "middle")
      .text(`${facetPoints.length} policies · median ${d3.format("+.4f")(median)} · best ${d3.format("+.4f")(best)}`);
    facet
      .append("text")
      .attr("class", "axis-label")
      .attr("x", facetWidth / 2)
      .attr("y", innerHeight + 54)
      .attr("text-anchor", "middle")
      .text("Phase total variation");

    const pointSelection = facet
      .append("g")
      .attr("class", "population-point-layer")
      .selectAll<SVGPathElement, PhasePopulationPoint>("path")
      .data(facetPoints, (point) => point.row.id)
      .join("path")
      .attr("class", "population-point")
      .attr("d", (point) =>
        d3.symbol()
          .type(markerType(point.row.phasePopulation!.contrastFamily))
          .size(point.row.id === options.selectedId ? 120 : 72)(),
      )
      .attr(
        "transform",
        (point) =>
          `translate(${x(point.row.diagnostics.phaseTv) + deterministicJitter(point.row.phasePopulation!.candidateId, 12)},${y(point.delta)})`,
      )
      .attr("fill", (point) => color(point.delta))
      .attr("fill-opacity", 0.84)
      .attr("stroke", (point) => (point.row.id === options.selectedId ? INK : "#fffdf7"))
      .attr("stroke-width", (point) => (point.row.id === options.selectedId ? 2.8 : 1.1))
      .attr("tabindex", 0)
      .attr("role", "button")
      .attr("aria-label", (point) => `${point.row.name}; two-phase minus tied ${point.delta} BPB`)
      .style("cursor", "pointer")
      .on("mouseenter", (event, point) => {
        focusDirection(point);
        showTooltip(event as MouseEvent, point, options.target, options.tooltip);
      })
      .on("mousemove", (event, point) => showTooltip(event as MouseEvent, point, options.target, options.tooltip))
      .on("mouseleave", () => {
        hideTooltip(options.tooltip);
        focusDirection(points.find((point) => point.row.id === options.selectedId));
      })
      .on("focus", (_event, point) => focusDirection(point))
      .on("blur", () => focusDirection(points.find((point) => point.row.id === options.selectedId)))
      .on("click", (event, point) => {
        (event as MouseEvent).stopPropagation();
        options.onSelect(point.row.id);
      })
      .on("keydown", (event, point) => {
        if ((event as KeyboardEvent).key === "Enter" || (event as KeyboardEvent).key === " ") {
          event.preventDefault();
          options.onSelect(point.row.id);
        }
      });
    allPointSelections.push(pointSelection);
  }

  function focusDirection(focused: PhasePopulationPoint | undefined): void {
    for (const selection of allPointSelections) {
      selection
        .attr("opacity", (point) => {
          if (!focused) return 1;
          return directionKey(point.row.phasePopulation!) === directionKey(focused.row.phasePopulation!) ? 1 : 0.22;
        })
        .attr("stroke-width", (point) => (point.row.id === focused?.row.id ? 2.8 : 1.1));
    }
  }

  focusDirection(points.find((point) => point.row.id === options.selectedId));
}
