// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
import type * as d3 from "d3";

import { lttbDownsample } from "../downsample";
import type { ContinuousScale } from "../types";

const DEFAULT_MAX_POINTS = 2000;

export interface PointConfig<T> {
  getCx: (d: T) => number;
  getCy: (d: T) => number;
  getColor: (d: T) => string;
  getRadius?: (d: T) => number;
  stroke?: string;
  strokeWidth?: number;
  keyFn?: (d: T) => string;
  getX?: (d: T) => number;
  getY?: (d: T) => number;
  maxPoints?: number;
}

export function renderPoints<T>(
  group: d3.Selection<SVGGElement, unknown, null, undefined>,
  data: T[],
  config: PointConfig<T>,
  xScale?: ContinuousScale,
  yScale?: ContinuousScale,
): d3.Selection<SVGCircleElement, T, SVGGElement, unknown> {
  const cx =
    config.getX && xScale ? (d: T) => xScale(config.getX!(d)) : config.getCx;
  const cy =
    config.getY && yScale ? (d: T) => yScale(config.getY!(d)) : config.getCy;

  let renderData = data;
  if (config.getX && config.getY) {
    const max = config.maxPoints ?? DEFAULT_MAX_POINTS;
    if (data.length > max) {
      renderData = lttbDownsample(data, max, config.getX, config.getY);
    }
  }

  return group
    .selectAll<SVGCircleElement, T>(".point")
    .data(renderData, config.keyFn)
    .join("circle")
    .attr("class", "point")
    .attr("cx", cx)
    .attr("cy", cy)
    .attr("r", (d) => config.getRadius?.(d) ?? 4)
    .attr("fill", (d) => config.getColor(d))
    .attr("stroke", config.stroke ?? "none")
    .attr("stroke-width", config.strokeWidth ?? 0)
    .attr("cursor", "pointer");
}

export function updatePointsOnZoom<T>(
  group: d3.Selection<SVGGElement, unknown, null, undefined>,
  getCx: (d: T) => number,
  getCy: (d: T) => number,
): void {
  group
    .selectAll<SVGCircleElement, T>(".point")
    .attr("cx", (d) => getCx(d))
    .attr("cy", (d) => getCy(d));
}
