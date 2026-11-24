// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
import type * as d3 from "d3";

import type { ContinuousScale } from "../types";

export interface HorizontalBarConfig<T> {
  getY: (d: T) => string;
  getX: (d: T) => number;
  getColor: (d: T) => string;
  stroke?: string;
  strokeWidth?: number;
  rx?: number;
  opacity?: number;
  keyFn?: (d: T) => string;
}

export function renderHorizontalBars<T>(
  group: d3.Selection<SVGGElement, unknown, null, undefined>,
  data: T[],
  yScale: d3.ScaleBand<string>,
  xScale: ContinuousScale,
  config: HorizontalBarConfig<T>,
): d3.Selection<SVGRectElement, T, SVGGElement, unknown> {
  return group
    .selectAll<SVGRectElement, T>(".bar")
    .data(data, config.keyFn)
    .join("rect")
    .attr("class", "bar")
    .attr("x", 0)
    .attr("y", (d) => yScale(config.getY(d)) || 0)
    .attr("width", (d) => Math.max(0, xScale(config.getX(d))))
    .attr("height", yScale.bandwidth())
    .attr("fill", (d) => config.getColor(d))
    .attr("stroke", config.stroke ?? "none")
    .attr("stroke-width", config.strokeWidth ?? 0)
    .attr("rx", config.rx ?? 2)
    .attr("opacity", config.opacity ?? 0.85)
    .attr("cursor", "pointer");
}

export function updateHorizontalBarsOnZoom<T>(
  group: d3.Selection<SVGGElement, unknown, null, undefined>,
  xScale: ContinuousScale,
  getX: (d: T) => number,
): void {
  group
    .selectAll<SVGRectElement, T>(".bar")
    .attr("width", (d) => Math.max(0, xScale(getX(d))));
}

export function updateHorizontalBarsForDisplay<T>(
  group: d3.Selection<SVGGElement, unknown, null, undefined>,
  config: HorizontalBarConfig<T>,
): void {
  group
    .selectAll<SVGRectElement, T>(".bar")
    .attr("fill", (d) => config.getColor(d))
    .attr("stroke", config.stroke ?? "none")
    .attr("stroke-width", config.strokeWidth ?? 0);
}
