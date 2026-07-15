import * as d3 from "d3";

import type { PointDatum, TargetMetadata, ViewMode } from "./types";

interface ScatterOptions {
  target: TargetMetadata;
  view: ViewMode;
  selectedId: string | null;
  tooltip: HTMLElement;
  onSelect: (id: string) => void;
}

const INK = "#17324a";
const MUTED = "#75828a";
const GRID = "#d9d6ca";
const ONE_PHASE = "#e46f38";
const TWO_PHASE = "#153d55";

function pointSymbol(datum: PointDatum): d3.SymbolType {
  if (datum.displaySplit === "fit") return d3.symbolCircle;
  if (datum.displaySplit === "noise_reference") return d3.symbolTriangle;
  if (datum.displaySplit === "off_policy") return d3.symbolSquare;
  return d3.symbolDiamond;
}

function pointColor(datum: PointDatum, maxStandardizedError: number): string {
  const scaled = Math.min(Math.abs(datum.standardizedResidual) / maxStandardizedError, 1);
  return d3.interpolateRdYlGn(1 - scaled);
}

function labelForSplit(split: PointDatum["displaySplit"], inPolicy: boolean): string {
  if (split === "fit") return "Fit design · grouped OOF";
  if (split === "noise_reference") return "Proportional repeat · full fit";
  if (split === "off_policy") return "Off-policy projection · full fit";
  return inPolicy ? "In-policy heldout · full fit" : "Full-fit projection";
}

function showTooltip(event: MouseEvent, datum: PointDatum, tooltip: HTMLElement): void {
  const sign = datum.residual >= 0 ? "+" : "";
  tooltip.innerHTML = `
    <div class="tooltip-kicker">${labelForSplit(datum.displaySplit, datum.inPolicy)}</div>
    <strong>${datum.row.name}</strong>
    <dl>
      <dt>Observed</dt><dd>${datum.observed.toFixed(6)}</dd>
      <dt>Predicted</dt><dd>${datum.prediction.toFixed(6)}</dd>
      <dt>Residual</dt><dd>${sign}${datum.residual.toFixed(6)}</dd>
      <dt>Noise units</dt><dd>${datum.standardizedResidual.toFixed(2)}×</dd>
      <dt>Support distance</dt><dd>${datum.row.diagnostics.supportDistance.toFixed(3)} TV</dd>
    </dl>`;
  tooltip.classList.add("visible");
  tooltip.style.left = `${event.clientX + 16}px`;
  tooltip.style.top = `${event.clientY + 16}px`;
}

function hideTooltip(tooltip: HTMLElement): void {
  tooltip.classList.remove("visible");
}

function paddedExtent(values: number[], fraction = 0.06): [number, number] {
  const extent = d3.extent(values) as [number, number];
  const span = Math.max(extent[1] - extent[0], Math.abs(extent[0]) * 0.02, 1e-4);
  return [extent[0] - span * fraction, extent[1] + span * fraction];
}

export function renderScatter(
  container: HTMLElement,
  points: PointDatum[],
  options: ScatterOptions,
): void {
  container.replaceChildren();
  if (points.length === 0) {
    container.innerHTML = '<div class="empty-state">No observed checkpoints match these filters.</div>';
    return;
  }

  const width = Math.max(container.clientWidth, 560);
  const height = Math.max(480, Math.min(600, width * 0.64));
  const margin = { top: 28, right: 30, bottom: 64, left: 78 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const maxStandardizedError = Math.max(
    2,
    d3.quantile(points.map((point) => Math.abs(point.standardizedResidual)).sort(d3.ascending), 0.95) ?? 2,
  );

  const x = d3
    .scaleLinear()
    .domain(paddedExtent(points.map((point) => point.observed)))
    .nice()
    .range([0, innerWidth]);
  const yValues = points.map((point) => {
    if (options.view === "prediction") return point.prediction;
    if (options.view === "residual") return point.residual;
    return point.standardizedResidual;
  });
  let yDomain = paddedExtent(yValues);
  if (options.view !== "prediction") {
    const magnitude = Math.max(Math.abs(yDomain[0]), Math.abs(yDomain[1]));
    yDomain = [-magnitude, magnitude];
  }
  const y = d3.scaleLinear().domain(yDomain).nice().range([innerHeight, 0]);

  const svg = d3
    .select(container)
    .append("svg")
    .attr("class", "scatter-svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("role", "img")
    .attr(
      "aria-label",
      `${options.target.label}: ${options.view} plot with ${points.length} checkpoints`,
    );
  const plot = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  plot
    .append("g")
    .attr("class", "grid-lines")
    .call(d3.axisLeft(y).ticks(7).tickSize(-innerWidth).tickFormat(() => ""))
    .call((group) => group.select(".domain").remove())
    .call((group) => group.selectAll("line").attr("stroke", GRID).attr("stroke-dasharray", "2,4"));

  const xAxis = d3.axisBottom(x).ticks(7).tickFormat(d3.format(".4~f"));
  const yAxis = d3.axisLeft(y).ticks(7).tickFormat(d3.format(options.view === "standardized" ? ".1f" : ".4~f"));
  plot
    .append("g")
    .attr("class", "axis")
    .attr("transform", `translate(0,${innerHeight})`)
    .call(xAxis);
  plot.append("g").attr("class", "axis").call(yAxis);

  if (options.view === "prediction") {
    const domain = x.domain();
    const low = domain[0] ?? 0;
    const high = domain[1] ?? low;
    plot
      .append("line")
      .attr("x1", x(low))
      .attr("x2", x(high))
      .attr("y1", y(low))
      .attr("y2", y(high))
      .attr("stroke", MUTED)
      .attr("stroke-width", 1.4)
      .attr("stroke-dasharray", "7,5");
  } else {
    plot
      .append("line")
      .attr("x1", 0)
      .attr("x2", innerWidth)
      .attr("y1", y(0))
      .attr("y2", y(0))
      .attr("stroke", MUTED)
      .attr("stroke-width", 1.4)
      .attr("stroke-dasharray", "7,5");
  }

  plot
    .append("text")
    .attr("class", "axis-label")
    .attr("x", innerWidth / 2)
    .attr("y", innerHeight + 50)
    .attr("text-anchor", "middle")
    .text(`Observed ${options.target.label}`);
  const yLabel =
    options.view === "prediction"
      ? `Predicted ${options.target.label}`
      : options.view === "residual"
        ? "Prediction residual (predicted − observed)"
        : "Residual / proportional difference SD";
  plot
    .append("text")
    .attr("class", "axis-label")
    .attr("transform", "rotate(-90)")
    .attr("x", -innerHeight / 2)
    .attr("y", -58)
    .attr("text-anchor", "middle")
    .text(yLabel);

  const pointLayer = plot.append("g").attr("class", "point-layer");
  const symbol = d3.symbol<PointDatum>().size((datum) => (datum.displaySplit === "noise_reference" ? 76 : 68));
  const point = pointLayer
    .selectAll<SVGPathElement, PointDatum>("path")
    .data(points, (datum) => datum.row.id)
    .join("path")
    .attr("d", (datum) => symbol.type(pointSymbol(datum))(datum) ?? "")
    .attr("transform", (datum) => {
      const yValue =
        options.view === "prediction"
          ? datum.prediction
          : options.view === "residual"
            ? datum.residual
            : datum.standardizedResidual;
      return `translate(${x(datum.observed)},${y(yValue)})`;
    })
    .attr("fill", (datum) => pointColor(datum, maxStandardizedError))
    .attr("fill-opacity", (datum) => (datum.displaySplit === "fit" ? 0.78 : 0.94))
    .attr("stroke", (datum) => (datum.row.phaseFamily === "single_phase" ? ONE_PHASE : TWO_PHASE))
    .attr("stroke-width", (datum) => (datum.row.id === options.selectedId ? 3 : 1.6))
    .attr("tabindex", 0)
    .attr("role", "button")
    .attr("aria-label", (datum) => `${datum.row.name}; observed ${datum.observed}; predicted ${datum.prediction}`)
    .style("cursor", "pointer")
    .on("mouseenter", (event, datum) => showTooltip(event as MouseEvent, datum, options.tooltip))
    .on("mousemove", (event, datum) => showTooltip(event as MouseEvent, datum, options.tooltip))
    .on("mouseleave", () => hideTooltip(options.tooltip))
    .on("click", (_event, datum) => options.onSelect(datum.row.id))
    .on("keydown", (event, datum) => {
      if ((event as KeyboardEvent).key === "Enter" || (event as KeyboardEvent).key === " ") {
        event.preventDefault();
        options.onSelect(datum.row.id);
      }
    });

  const selected = points.find((datum) => datum.row.id === options.selectedId);
  if (selected) {
    const selectedY =
      options.view === "prediction"
        ? selected.prediction
        : options.view === "residual"
          ? selected.residual
          : selected.standardizedResidual;
    pointLayer
      .insert("circle", ":first-child")
      .attr("cx", x(selected.observed))
      .attr("cy", y(selectedY))
      .attr("r", 12)
      .attr("fill", "none")
      .attr("stroke", INK)
      .attr("stroke-width", 2)
      .attr("stroke-dasharray", "3,3");
  }

  const legend = svg
    .append("g")
    .attr("class", "scatter-legend")
    .attr("transform", `translate(${margin.left + 6},${margin.top + 5})`);
  const legendItems: Array<[string, d3.SymbolType]> = [
    ["Fit · OOF", d3.symbolCircle],
    ["Heldout", d3.symbolDiamond],
    ["Off-policy", d3.symbolSquare],
    ["Repeat", d3.symbolTriangle],
  ];
  legendItems.forEach(([label, type], index) => {
    const group = legend.append("g").attr("transform", `translate(${index * 112},0)`);
    group
      .append("path")
      .attr("d", d3.symbol().type(type).size(54)() ?? "")
      .attr("transform", "translate(7,7)")
      .attr("fill", "#d6cda9")
      .attr("stroke", INK);
    group.append("text").attr("x", 19).attr("y", 11).text(label);
  });

  point.raise();
}
