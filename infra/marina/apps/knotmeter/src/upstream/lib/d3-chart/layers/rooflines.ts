// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
import * as d3 from "d3";

import type { ContinuousScale } from "../types";

export interface RooflineConfig {
  getColor: (key: string) => string;
  getOpacity?: (key: string) => number;
  isVisible?: (key: string) => boolean;
  strokeWidth?: number;
  strokeDasharray?: string;
  getDasharray?: (key: string) => string | null;
  curve?: d3.CurveFactory;
}

function dasharrayFor(config: RooflineConfig, key: string): string | null {
  return config.getDasharray
    ? config.getDasharray(key)
    : (config.strokeDasharray ?? null);
}

interface RooflineEntry<T> {
  key: string;
  points: T[];
}

export function renderRooflines<T extends { x: number; y: number }>(
  zoomGroup: d3.Selection<SVGGElement, unknown, null, undefined>,
  rooflines: Record<string, T[]>,
  xScale: ContinuousScale,
  yScale: ContinuousScale,
  config: RooflineConfig,
): void {
  const {
    getColor,
    getOpacity,
    isVisible,
    strokeWidth = 2,
    strokeDasharray,
    getDasharray,
  } = config;
  const lineGenerator = d3
    .line<T>()
    .x((d) => xScale(d.x))
    .y((d) => yScale(d.y))
    .curve(config.curve ?? d3.curveMonotoneX);

  const entries: RooflineEntry<T>[] = Object.entries(rooflines)
    .filter(
      ([key, points]) => points.length >= 2 && (!isVisible || isVisible(key)),
    )
    .map(([key, points]) => ({ key, points }));

  const selection = zoomGroup
    .selectAll<SVGPathElement, RooflineEntry<T>>(".roofline-path")
    .data(entries, (d) => d.key);

  const entered = selection
    .enter()
    .append("path")
    .attr("class", (d) => `roofline-path roofline-${d.key}`)
    .attr("fill", "none");

  selection.exit().remove();

  const merged = entered.merge(selection);
  merged
    .attr("class", (d) => `roofline-path roofline-${d.key}`)
    .attr("stroke", (d) => getColor(d.key))
    .attr("stroke-width", strokeWidth)
    .attr("d", (d) => lineGenerator(d.points) ?? "");

  if (getDasharray) {
    merged.attr("stroke-dasharray", (d) => getDasharray(d.key));
  } else if (strokeDasharray) {
    merged.attr("stroke-dasharray", strokeDasharray);
  }

  merged.each(function (d) {
    const el = d3.select(this);
    const opacity = getOpacity?.(d.key);
    if (opacity !== undefined) {
      el.style("opacity", opacity);
    }
  });
}

export function updateRooflinesForDisplay(
  zoomGroup: d3.Selection<SVGGElement, unknown, null, undefined>,
  config: RooflineConfig,
): void {
  const { getColor, getOpacity, strokeWidth = 2 } = config;
  zoomGroup
    .selectAll<SVGPathElement, RooflineEntry<{ x: number; y: number }>>(
      ".roofline-path",
    )
    .attr("stroke", (d) => getColor(d.key))
    .attr("stroke-width", strokeWidth)
    .attr("stroke-dasharray", (d) => dasharrayFor(config, d.key))
    .each(function (d) {
      const opacity = getOpacity?.(d.key);
      if (opacity !== undefined) d3.select(this).style("opacity", opacity);
    });
}

export function updateRooflinesOnZoom<T extends { x: number; y: number }>(
  zoomGroup: d3.Selection<SVGGElement, unknown, null, undefined>,
  rooflines: Record<string, T[]>,
  newXScale: ContinuousScale,
  newYScale: ContinuousScale,
  curve: d3.CurveFactory = d3.curveMonotoneX,
): void {
  const lineGenerator = d3
    .line<T>()
    .x((d) => newXScale(d.x))
    .y((d) => newYScale(d.y))
    .curve(curve);

  Object.entries(rooflines).forEach(([key, points]) => {
    if (points.length < 2) return;
    const selection = zoomGroup.select<SVGPathElement>(
      `.${cssEscapeToken(`roofline-${key}`)}`,
    );
    if (!selection.empty()) {
      selection.attr("d", lineGenerator(points) as string);
    }
  });
}

function cssEscapeToken(token: string): string {
  if (typeof CSS !== "undefined" && typeof CSS.escape === "function")
    return CSS.escape(token);
  return token.replaceAll(/[^a-zA-Z0-9_-]/gu, (c) => `\\${c}`);
}
