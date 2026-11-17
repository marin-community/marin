// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
import type * as d3 from "d3";

export interface ErrorBarConfig<T> {
  getCx: (d: T) => number;
  getYMin: (d: T) => number;
  getYMax: (d: T) => number;
  capWidth: number;
  stroke: string;
  strokeWidth?: number;
}

export function renderErrorBars<T>(
  group: d3.Selection<SVGGElement, unknown, null, undefined>,
  data: T[],
  config: ErrorBarConfig<T>,
): d3.Selection<SVGGElement, T, SVGGElement, unknown> {
  const selection = group.selectAll<SVGGElement, T>(".error-bar").data(data);
  const sw = config.strokeWidth ?? 1;

  const entered = selection.enter().append("g").attr("class", "error-bar");
  entered
    .append("line")
    .attr("class", "eb-stem")
    .attr("stroke", config.stroke)
    .attr("stroke-width", sw);
  entered
    .append("line")
    .attr("class", "eb-cap-top")
    .attr("stroke", config.stroke)
    .attr("stroke-width", sw);
  entered
    .append("line")
    .attr("class", "eb-cap-bot")
    .attr("stroke", config.stroke)
    .attr("stroke-width", sw);

  selection.exit().remove();

  const merged = entered.merge(selection);

  merged
    .select(".eb-stem")
    .attr("x1", (d) => config.getCx(d))
    .attr("x2", (d) => config.getCx(d))
    .attr("y1", (d) => config.getYMax(d))
    .attr("y2", (d) => config.getYMin(d))
    .attr("stroke", config.stroke);

  merged
    .select(".eb-cap-top")
    .attr("x1", (d) => config.getCx(d) - config.capWidth / 2)
    .attr("x2", (d) => config.getCx(d) + config.capWidth / 2)
    .attr("y1", (d) => config.getYMax(d))
    .attr("y2", (d) => config.getYMax(d))
    .attr("stroke", config.stroke);

  merged
    .select(".eb-cap-bot")
    .attr("x1", (d) => config.getCx(d) - config.capWidth / 2)
    .attr("x2", (d) => config.getCx(d) + config.capWidth / 2)
    .attr("y1", (d) => config.getYMin(d))
    .attr("y2", (d) => config.getYMin(d))
    .attr("stroke", config.stroke);

  return merged;
}

export function updateErrorBarsOnZoom<T>(
  group: d3.Selection<SVGGElement, unknown, null, undefined>,
  config: ErrorBarConfig<T>,
): void {
  const bars = group.selectAll<SVGGElement, T>(".error-bar");

  bars
    .select(".eb-stem")
    .attr("x1", (d) => config.getCx(d))
    .attr("x2", (d) => config.getCx(d))
    .attr("y1", (d) => config.getYMax(d))
    .attr("y2", (d) => config.getYMin(d));

  bars
    .select(".eb-cap-top")
    .attr("x1", (d) => config.getCx(d) - config.capWidth / 2)
    .attr("x2", (d) => config.getCx(d) + config.capWidth / 2)
    .attr("y1", (d) => config.getYMax(d))
    .attr("y2", (d) => config.getYMax(d));

  bars
    .select(".eb-cap-bot")
    .attr("x1", (d) => config.getCx(d) - config.capWidth / 2)
    .attr("x2", (d) => config.getCx(d) + config.capWidth / 2)
    .attr("y1", (d) => config.getYMin(d))
    .attr("y2", (d) => config.getYMin(d));
}
