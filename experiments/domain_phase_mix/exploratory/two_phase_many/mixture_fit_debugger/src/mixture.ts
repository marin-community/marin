import * as d3 from "d3";

import type { DomainMetadata, MixtureRow, SortMode } from "./types";

interface MixtureChartOptions {
  selected: MixtureRow;
  baseline: MixtureRow;
  domains: DomainMetadata[];
  sort: SortMode;
  tooltip: HTMLElement;
}

interface DomainDatum {
  domain: DomainMetadata;
  index: number;
  selected: [number, number, number, number];
  baseline: [number, number, number, number];
  phaseEpochs: [number, number];
  baselinePhaseEpochs: [number, number];
}

const COLUMN_COLORS = ["#d96735", "#1f7d72", "#d8a72c", "#173f5f"];
const BASELINE_COLOR = "#d6d2c5";
const INK = "#183247";
const GRID = "#ddd9cd";

function columnValue(values: [number, number, number, number], column: number): number {
  return values[column] ?? 0;
}

function chartRows(options: MixtureChartOptions): DomainDatum[] {
  const rows = options.domains.map((domain, index) => ({
    domain,
    index,
    selected: [
      options.selected.phase0[index] ?? 0,
      options.selected.phase1[index] ?? 0,
      options.selected.aggregate[index] ?? 0,
      options.selected.totalEpochs[index] ?? 0,
    ] as [number, number, number, number],
    baseline: [
      options.baseline.phase0[index] ?? 0,
      options.baseline.phase1[index] ?? 0,
      options.baseline.aggregate[index] ?? 0,
      options.baseline.totalEpochs[index] ?? 0,
    ] as [number, number, number, number],
    phaseEpochs: [
      options.selected.phase0Epochs[index] ?? 0,
      options.selected.phase1Epochs[index] ?? 0,
    ] as [number, number],
    baselinePhaseEpochs: [
      options.baseline.phase0Epochs[index] ?? 0,
      options.baseline.phase1Epochs[index] ?? 0,
    ] as [number, number],
  }));
  if (options.sort === "difference") {
    rows.sort(
      (left, right) =>
        Math.abs(right.selected[2] - right.baseline[2]) -
        Math.abs(left.selected[2] - left.baseline[2]),
    );
  } else if (options.sort === "phase_difference") {
    rows.sort((left, right) => {
      const leftContrastChange = Math.abs(
        (left.selected[1] - left.selected[0]) - (left.baseline[1] - left.baseline[0]),
      );
      const rightContrastChange = Math.abs(
        (right.selected[1] - right.selected[0]) - (right.baseline[1] - right.baseline[0]),
      );
      return rightContrastChange - leftContrastChange;
    });
  } else if (options.sort === "exposure") {
    rows.sort((left, right) => right.selected[3] - left.selected[3]);
  } else {
    rows.sort((left, right) =>
      `${left.domain.group}/${left.domain.label}`.localeCompare(
        `${right.domain.group}/${right.domain.label}`,
      ),
    );
  }
  return rows;
}

function valueLabel(column: number, datum: DomainDatum): string {
  if (column === 0) return `${d3.format(".2%")(datum.selected[0])} · ${d3.format(".2f")(datum.phaseEpochs[0])}e`;
  if (column === 1) return `${d3.format(".2%")(datum.selected[1])} · ${d3.format(".2f")(datum.phaseEpochs[1])}e`;
  if (column === 2) return d3.format(".2%")(datum.selected[2]);
  return `${d3.format(".2f")(datum.selected[3])}e`;
}

function showTooltip(
  event: MouseEvent,
  datum: DomainDatum,
  column: number,
  selectedName: string,
  baselineName: string,
  tooltip: HTMLElement,
): void {
  const columnLabels = ["Phase 0 weight", "Phase 1 weight", "Aggregate weight", "Total exposure"];
  const selected = columnValue(datum.selected, column);
  const baseline = columnValue(datum.baseline, column);
  const selectedValue = column === 3 ? `${selected.toFixed(4)} epochs` : d3.format(".4%")(selected);
  const baselineValue = column === 3 ? `${baseline.toFixed(4)} epochs` : d3.format(".4%")(baseline);
  tooltip.innerHTML = `
    <div class="tooltip-kicker">${columnLabels[column]}</div>
    <strong>${datum.domain.label}</strong>
    <dl>
      <dt>${selectedName}</dt><dd>${selectedValue}</dd>
      <dt>${baselineName}</dt><dd>${baselineValue}</dd>
      <dt>Proportional</dt><dd>${d3.format(".4%")(datum.domain.proportionalWeight)}</dd>
      <dt>Corpus size</dt><dd>${d3.format(".3s")(datum.domain.tokenCount)} tokens</dd>
    </dl>`;
  tooltip.classList.add("visible");
  tooltip.style.left = `${event.clientX + 16}px`;
  tooltip.style.top = `${event.clientY + 16}px`;
}

function hideTooltip(tooltip: HTMLElement): void {
  tooltip.classList.remove("visible");
}

export function renderMixtureChart(container: HTMLElement, options: MixtureChartOptions): void {
  container.replaceChildren();
  const rows = chartRows(options);
  const rowHeight = 27;
  const headerHeight = 72;
  const bottomMargin = 34;
  const labelWidth = 242;
  const panelWidth = 250;
  const panelGap = 26;
  const width = labelWidth + panelWidth * 4 + panelGap * 3 + 44;
  const height = headerHeight + rows.length * rowHeight + bottomMargin;
  const panelOffsets = Array.from({ length: 4 }, (_, index) => labelWidth + index * (panelWidth + panelGap));
  const columnLabels = ["Phase 0 weight", "Phase 1 weight", "Aggregate weight", "Aggregate exposure"];

  const svg = d3
    .select(container)
    .append("svg")
    .attr("class", "mixture-svg")
    .attr("width", width)
    .attr("height", height)
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("role", "img")
    .attr(
      "aria-label",
      `Mixture comparison between ${options.selected.name} and ${options.baseline.name}`,
    );

  const scales = Array.from({ length: 4 }, (_, column) => {
    const maximum =
      d3.max(rows, (row) =>
        Math.max(columnValue(row.selected, column), columnValue(row.baseline, column)),
      ) ?? 1;
    return d3.scaleLinear().domain([0, Math.max(maximum * 1.2, column === 3 ? 0.25 : 0.005)]).nice().range([0, panelWidth]);
  });

  columnLabels.forEach((label, column) => {
    const offset = panelOffsets[column] ?? 0;
    svg
      .append("text")
      .attr("class", "mixture-column-title")
      .attr("x", offset)
      .attr("y", 24)
      .text(label);
    svg
      .append("g")
      .attr("class", "axis mixture-axis")
      .attr("transform", `translate(${offset},${headerHeight - 18})`)
      .call(
        d3
          .axisTop(scales[column] as d3.ScaleLinear<number, number>)
          .ticks(4)
          .tickFormat(column === 3 ? d3.format(".1f") : d3.format(".0%")),
      )
      .call((group) => group.selectAll("line").attr("stroke", GRID))
      .call((group) => group.select(".domain").attr("stroke", GRID));
  });

  let previousGroup = "";
  rows.forEach((datum, rowIndex) => {
    const y = headerHeight + rowIndex * rowHeight;
    if (datum.domain.group !== previousGroup) {
      svg
        .append("line")
        .attr("x1", 0)
        .attr("x2", width - 20)
        .attr("y1", y - 5)
        .attr("y2", y - 5)
        .attr("stroke", rowIndex === 0 ? "transparent" : GRID)
        .attr("stroke-width", 1);
      previousGroup = datum.domain.group;
    }
    svg
      .append("text")
      .attr("class", "domain-label")
      .attr("x", labelWidth - 14)
      .attr("y", y + 14)
      .attr("text-anchor", "end")
      .text(datum.domain.label)
      .append("title")
      .text(`${datum.domain.group} · ${datum.domain.id}`);

    for (let column = 0; column < 4; column += 1) {
      const offset = panelOffsets[column] ?? 0;
      const scale = scales[column] as d3.ScaleLinear<number, number>;
      svg
        .append("rect")
        .attr("x", offset)
        .attr("y", y + 1)
        .attr("width", scale(columnValue(datum.baseline, column)))
        .attr("height", 20)
        .attr("rx", 2)
        .attr("fill", BASELINE_COLOR)
        .attr("opacity", 0.78)
        .on("mouseenter", (event) =>
          showTooltip(
            event as MouseEvent,
            datum,
            column,
            options.selected.name,
            options.baseline.name,
            options.tooltip,
          ),
        )
        .on("mousemove", (event) =>
          showTooltip(
            event as MouseEvent,
            datum,
            column,
            options.selected.name,
            options.baseline.name,
            options.tooltip,
          ),
        )
        .on("mouseleave", () => hideTooltip(options.tooltip));
      svg
        .append("rect")
        .attr("x", offset)
        .attr("y", y + 5)
        .attr("width", scale(columnValue(datum.selected, column)))
        .attr("height", 12)
        .attr("rx", 2)
        .attr("fill", COLUMN_COLORS[column] ?? INK)
        .on("mouseenter", (event) =>
          showTooltip(
            event as MouseEvent,
            datum,
            column,
            options.selected.name,
            options.baseline.name,
            options.tooltip,
          ),
        )
        .on("mousemove", (event) =>
          showTooltip(
            event as MouseEvent,
            datum,
            column,
            options.selected.name,
            options.baseline.name,
            options.tooltip,
          ),
        )
        .on("mouseleave", () => hideTooltip(options.tooltip));
      const labelX = Math.min(
        offset + scale(columnValue(datum.selected, column)) + 5,
        offset + panelWidth - 2,
      );
      svg
        .append("text")
        .attr("class", "bar-value")
        .attr("x", labelX)
        .attr("y", y + 15)
        .attr("text-anchor", labelX >= offset + panelWidth - 3 ? "end" : "start")
        .text(valueLabel(column, datum));
    }
  });

  const legend = svg
    .append("g")
    .attr("class", "mixture-legend")
    .attr("transform", `translate(${labelWidth},${height - 15})`);
  legend.append("rect").attr("x", 0).attr("y", -9).attr("width", 18).attr("height", 8).attr("fill", INK);
  legend.append("text").attr("x", 24).attr("y", 0).text(options.selected.name);
  legend
    .append("rect")
    .attr("x", 260)
    .attr("y", -11)
    .attr("width", 18)
    .attr("height", 12)
    .attr("fill", BASELINE_COLOR);
  legend.append("text").attr("x", 284).attr("y", 0).text(options.baseline.name);
}
