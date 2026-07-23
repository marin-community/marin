import * as d3 from "d3";

import type { MixtureRow, TargetMetadata } from "./types";

export interface PhasePopulationPoint {
  row: MixtureRow;
  anchor: MixtureRow;
  observed: number;
  anchorObserved: number;
  delta: number;
  standardizedDelta: number;
}

interface PhasePopulationChartOptions {
  target: TargetMetadata;
  selectedId: string | null;
  tooltip: HTMLElement;
  onSelect: (id: string) => void;
}

const ANCHOR_ORDER = ["uncheatable_frontier", "table9_frontier"];
const RADII = [0.25, 0.5, 0.75];
const INK = "#183247";
const MUTED = "#72818c";
const GRID = "#d8d3c8";
const PAPER = "#fffdf7";
const NOISE_BAND = "#ebe7dc";

function anchorKey(anchorId: string, seedBlock: number): string {
  return `${anchorId}::${seedBlock}`;
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
    const key = anchorKey(metadata.anchorId, metadata.seedBlock);
    if (controls.has(key)) throw new Error(`Duplicate phase-population control ${key}`);
    controls.set(key, row);
  }

  const scale = Math.max(differenceStandardDeviation, 1e-12);
  return populationRows.flatMap((row) => {
    const metadata = row.phasePopulation!;
    if (metadata.contrastFamily !== "random_isotropic") return [];
    const anchor = controls.get(anchorKey(metadata.anchorId, metadata.seedBlock));
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

function anchorLabel(anchorId: string): string {
  if (anchorId === "uncheatable_frontier") return "Uncheatable frontier anchor";
  if (anchorId === "table9_frontier") return "Table-9 frontier anchor";
  return anchorId.replaceAll("_", " ");
}

function deterministicJitter(directionId: string, radius: number, width: number): number {
  let hash = 2166136261;
  for (const character of `${directionId}:${radius}`) {
    hash ^= character.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return ((((hash >>> 0) % 1001) / 1000) - 0.5) * width;
}

function showTooltip(
  event: MouseEvent,
  datum: PhasePopulationPoint,
  target: TargetMetadata,
  tooltip: HTMLElement,
): void {
  const metadata = datum.row.phasePopulation!;
  tooltip.innerHTML = `
    <div class="tooltip-kicker">${anchorLabel(metadata.anchorId)} · ${metadata.directionLabel}</div>
    <strong>${datum.row.name}</strong>
    <dl>
      <dt>Radius</dt><dd>${d3.format(".0%")(metadata.radiusFraction)} of feasible limit</dd>
      <dt>Random policy</dt><dd>${datum.observed.toFixed(6)} ${target.label}</dd>
      <dt>Tied control</dt><dd>${datum.anchorObserved.toFixed(6)}</dd>
      <dt>Random − tied</dt><dd>${d3.format("+.6f")(datum.delta)} BPB</dd>
      <dt>Noise units</dt><dd>${d3.format("+.2f")(datum.standardizedDelta)}×</dd>
      <dt>Phase TV</dt><dd>${datum.row.diagnostics.phaseTv.toFixed(4)}</dd>
      <dt>Phase information</dt><dd>${metadata.phaseInformationKl.toExponential(3)} nats</dd>
      <dt>Seed block</dt><dd>${metadata.seedBlock}</dd>
    </dl>`;
  tooltip.classList.add("visible");
  tooltip.style.left = `${event.clientX + 16}px`;
  tooltip.style.top = `${event.clientY + 16}px`;
}

function hideTooltip(tooltip: HTMLElement): void {
  tooltip.classList.remove("visible");
}

function summary(values: readonly number[]): {
  lower: number;
  q1: number;
  median: number;
  q3: number;
  upper: number;
  mean: number;
} {
  const sorted = [...values].sort(d3.ascending);
  return {
    lower: d3.quantileSorted(sorted, 0.1) ?? 0,
    q1: d3.quantileSorted(sorted, 0.25) ?? 0,
    median: d3.quantileSorted(sorted, 0.5) ?? 0,
    q3: d3.quantileSorted(sorted, 0.75) ?? 0,
    upper: d3.quantileSorted(sorted, 0.9) ?? 0,
    mean: d3.mean(sorted) ?? 0,
  };
}

export function renderPhasePopulationChart(
  container: HTMLElement,
  points: PhasePopulationPoint[],
  options: PhasePopulationChartOptions,
): void {
  container.replaceChildren();
  if (points.length === 0) {
    container.innerHTML = '<div class="empty-state">This swarm has no phase-population panel for the selected objective.</div>';
    return;
  }

  const anchorIds = ANCHOR_ORDER.filter((anchorId) =>
    points.some((point) => point.row.phasePopulation?.anchorId === anchorId),
  );
  const width = Math.max(container.clientWidth, 940);
  const height = 540;
  const margin = { top: 66, right: 34, bottom: 72, left: 76 };
  const facetGap = 74;
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const facetWidth = (innerWidth - facetGap * (anchorIds.length - 1)) / anchorIds.length;
  const maximumMagnitude = d3.max(points, (point) => Math.abs(point.standardizedDelta)) ?? 1;
  const yLimit = Math.max(2.2, Math.ceil(maximumMagnitude * 2) / 2 + 0.25);
  const y = d3.scaleLinear().domain([-yLimit, yLimit]).range([innerHeight, 0]);
  const colorLimit = Math.max(1.5, d3.quantile(points.map((point) => Math.abs(point.standardizedDelta)).sort(d3.ascending), 0.95) ?? 1.5);
  const color = d3.scaleDiverging<string>([-colorLimit, 0, colorLimit], (value) => d3.interpolateRdYlGn(1 - value));

  const svg = d3
    .select(container)
    .append("svg")
    .attr("class", "population-svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("role", "img")
    .attr("aria-label", `${options.target.label} effects for ${points.length} aggregate-matched phase schedules`);
  const root = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
  const allPointSelections: Array<d3.Selection<SVGCircleElement, PhasePopulationPoint, SVGGElement, unknown>> = [];

  for (const [facetIndex, anchorId] of anchorIds.entries()) {
    const offset = facetIndex * (facetWidth + facetGap);
    const facet = root.append("g").attr("transform", `translate(${offset},0)`);
    const facetPoints = points.filter((point) => point.row.phasePopulation?.anchorId === anchorId);
    const x = d3.scalePoint<number>().domain(RADII).range([facetWidth * 0.16, facetWidth * 0.84]);

    facet
      .append("rect")
      .attr("x", 0)
      .attr("y", y(1))
      .attr("width", facetWidth)
      .attr("height", y(-1) - y(1))
      .attr("fill", NOISE_BAND)
      .attr("opacity", 0.72);
    facet
      .append("g")
      .attr("class", "grid-lines")
      .call(d3.axisLeft(y).ticks(7).tickSize(-facetWidth).tickFormat(() => ""))
      .call((group) => group.select(".domain").remove())
      .call((group) => group.selectAll("line").attr("stroke", GRID).attr("stroke-dasharray", "2,4"));
    for (const value of [-1, 0, 1]) {
      facet
        .append("line")
        .attr("x1", 0)
        .attr("x2", facetWidth)
        .attr("y1", y(value))
        .attr("y2", y(value))
        .attr("stroke", value === 0 ? INK : MUTED)
        .attr("stroke-width", value === 0 ? 1.6 : 1)
        .attr("stroke-dasharray", value === 0 ? null : "3,4");
    }
    facet
      .append("g")
      .attr("class", "axis")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x).tickFormat((value) => d3.format(".0%")(value)));
    if (facetIndex === 0) {
      facet.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(7).tickFormat(d3.format("+.1f")));
      facet
        .append("text")
        .attr("class", "axis-label")
        .attr("transform", "rotate(-90)")
        .attr("x", -innerHeight / 2)
        .attr("y", -56)
        .attr("text-anchor", "middle")
        .text("Random − seed-matched tied control / difference SD");
    }
    facet
      .append("text")
      .attr("class", "population-facet-title")
      .attr("x", facetWidth / 2)
      .attr("y", -32)
      .attr("text-anchor", "middle")
      .text(anchorLabel(anchorId));
    facet
      .append("text")
      .attr("class", "axis-label")
      .attr("x", facetWidth / 2)
      .attr("y", innerHeight + 52)
      .attr("text-anchor", "middle")
      .text("Fraction of feasible phase-contrast radius");

    for (const radius of RADII) {
      const radiusPoints = facetPoints.filter((point) => point.row.phasePopulation?.radiusFraction === radius);
      const center = x(radius);
      if (center === undefined || radiusPoints.length === 0) continue;
      const distribution = summary(radiusPoints.map((point) => point.standardizedDelta));
      const boxWidth = Math.min(52, facetWidth * 0.12);
      facet
        .append("line")
        .attr("x1", center)
        .attr("x2", center)
        .attr("y1", y(distribution.lower))
        .attr("y2", y(distribution.upper))
        .attr("stroke", MUTED);
      facet
        .append("rect")
        .attr("x", center - boxWidth / 2)
        .attr("y", y(distribution.q3))
        .attr("width", boxWidth)
        .attr("height", Math.max(1, y(distribution.q1) - y(distribution.q3)))
        .attr("fill", PAPER)
        .attr("fill-opacity", 0.7)
        .attr("stroke", MUTED);
      facet
        .append("line")
        .attr("x1", center - boxWidth / 2)
        .attr("x2", center + boxWidth / 2)
        .attr("y1", y(distribution.median))
        .attr("y2", y(distribution.median))
        .attr("stroke", INK)
        .attr("stroke-width", 1.5);
      facet
        .append("path")
        .attr("d", d3.symbol().type(d3.symbolDiamond).size(74)())
        .attr("transform", `translate(${center},${y(distribution.mean)})`)
        .attr("fill", "#d8a72c")
        .attr("stroke", INK)
        .attr("stroke-width", 1.4);
      const fractionBetter = d3.mean(radiusPoints, (point) => Number(point.delta < 0)) ?? 0;
      facet
        .append("text")
        .attr("class", "population-summary-label")
        .attr("x", center)
        .attr("y", y(distribution.mean) - 11)
        .attr("text-anchor", "middle")
        .text(`${d3.format(".0%")(fractionBetter)} better`);

      const pointSelection = facet
        .append("g")
        .attr("class", "population-point-layer")
        .selectAll<SVGCircleElement, PhasePopulationPoint>("circle")
        .data(radiusPoints, (point) => point.row.id)
        .join("circle")
        .attr("class", "population-point")
        .attr("cx", (point) => center + deterministicJitter(point.row.phasePopulation!.directionId, radius, boxWidth * 1.5))
        .attr("cy", (point) => y(point.standardizedDelta))
        .attr("r", (point) => (point.row.id === options.selectedId ? 6.5 : 4.6))
        .attr("fill", (point) => color(point.standardizedDelta))
        .attr("fill-opacity", 0.82)
        .attr("stroke", (point) => (point.row.id === options.selectedId ? INK : "#fffdf7"))
        .attr("stroke-width", (point) => (point.row.id === options.selectedId ? 2.8 : 1.1))
        .attr("tabindex", 0)
        .attr("role", "button")
        .attr("aria-label", (point) => `${point.row.name}; random minus tied ${point.delta}`)
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
        .on("click", (_event, point) => options.onSelect(point.row.id))
        .on("keydown", (event, point) => {
          if ((event as KeyboardEvent).key === "Enter" || (event as KeyboardEvent).key === " ") {
            event.preventDefault();
            options.onSelect(point.row.id);
          }
        });
      allPointSelections.push(pointSelection);
    }
  }

  function focusDirection(focused: PhasePopulationPoint | undefined): void {
    for (const selection of allPointSelections) {
      selection
        .attr("opacity", (point) => {
          if (!focused) return 1;
          const focusedMetadata = focused.row.phasePopulation!;
          const metadata = point.row.phasePopulation!;
          return metadata.directionId === focusedMetadata.directionId ? 1 : 0.38;
        })
        .attr("stroke-width", (point) => (point.row.id === focused?.row.id ? 2.8 : 1.1));
    }
  }

  focusDirection(points.find((point) => point.row.id === options.selectedId));
}
