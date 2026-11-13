// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
import * as d3 from "d3";

import {
  HIT_AREA_RADIUS,
  type ShapeKey,
  getShapeConfig,
  getShapeKeyForPrecision,
  applyNormalState,
  normalStateAttrs,
} from "@/lib/charts/chart-rendering";

import type { ContinuousScale } from "../types";
import {
  setAttrIfChanged,
  setStyleIfChanged,
  setTextIfChanged,
} from "../chart-update";
import { CHART_TYPE, px } from "../typography";

export interface ScatterPointConfig<T> {
  getColor: (d: T) => string;
  getOpacity?: (d: T) => number;
  getPointerEvents?: (d: T) => string;
  hideLabels?: boolean;
  getLabelText?: (d: T) => string;
  foreground?: string;
  dataAttrs?: Record<string, (d: T) => string>;
  selectedPrecisions?: readonly string[];
  getShapeKey?: (d: T) => ShapeKey;
}

const resolveShapeKey = (
  precision: string,
  selectedPrecisions?: readonly string[],
): ShapeKey =>
  selectedPrecisions && selectedPrecisions.length > 0
    ? getShapeKeyForPrecision(precision, selectedPrecisions)
    : "circle";

export function syncPointShape(
  g: d3.Selection<SVGGElement, unknown, null, undefined>,
  shapeKey: ShapeKey,
  fill: string,
): void {
  const targetType = getShapeConfig(shapeKey).type;
  const existing = g.select<SVGElement>(".visible-shape").node();
  const currentType = existing?.tagName.toLowerCase();
  if (!existing || currentType !== targetType) {
    g.select(".visible-shape").remove();
    const shape = g
      .append(targetType)
      .attr("class", "visible-shape")
      .attr("data-shape-key", shapeKey)
      .attr("fill", fill)
      .attr("stroke", "none")
      .attr("cursor", "pointer") as d3.Selection<
      SVGCircleElement | SVGRectElement | SVGPathElement,
      unknown,
      null,
      undefined
    >;
    applyNormalState(shape, shapeKey);
  } else {
    setAttrIfChanged(existing, "fill", fill);
    setAttrIfChanged(existing, "data-shape-key", shapeKey);
    for (const [name, value] of normalStateAttrs(shapeKey)) {
      setAttrIfChanged(existing, name, value);
    }
  }
}

export function renderScatterPoints<
  T extends { precision: string; x: number; y: number },
>(
  zoomGroup: d3.Selection<SVGGElement, unknown, null, undefined>,
  data: T[],
  xScale: ContinuousScale,
  yScale: ContinuousScale,
  config: ScatterPointConfig<T>,
  keyFn?: (d: T) => string,
): d3.Selection<SVGGElement, T, SVGGElement, unknown> {
  const selection = zoomGroup
    .selectAll<SVGGElement, T>(".dot-group")
    .data(data, keyFn);
  const positionFn = (d: T) => `translate(${xScale(d.x)},${yScale(d.y)})`;

  const entered = selection.enter().append("g").attr("class", "dot-group");

  entered
    .append("circle")
    .attr("r", HIT_AREA_RADIUS)
    .attr("fill", "transparent")
    .attr("cursor", "pointer");

  selection.exit().remove();

  const points = entered.merge(selection);

  points.attr("transform", positionFn);

  if (config.getOpacity) {
    points.style("opacity", config.getOpacity);
  }
  if (config.getPointerEvents) {
    points.style("pointer-events", config.getPointerEvents);
  }
  if (config.dataAttrs) {
    for (const [attr, fn] of Object.entries(config.dataAttrs)) {
      points.attr(`data-${attr}`, fn);
    }
  }

  points.each(function (d) {
    const g = d3.select(this);
    const shapeKey = config.getShapeKey
      ? config.getShapeKey(d)
      : resolveShapeKey(d.precision, config.selectedPrecisions);
    syncPointShape(g, shapeKey, config.getColor(d));
  });

  if (config.getLabelText && config.foreground) {
    const labelGetter = config.getLabelText;
    const foreground = config.foreground;
    points.each(function (d) {
      syncPointLabel(this, labelGetter(d), foreground, config.hideLabels);
    });
  } else {
    points.selectAll(".point-label").remove();
  }

  return points;
}

export function syncPointLabel(
  group: SVGGElement,
  text: string,
  foreground: string,
  hideLabels: boolean | undefined,
): void {
  const lines = text.split("\n");
  const firstDy = `${-(0.8 + (lines.length - 1) * 1.1)}em`;
  const label = d3
    .select(group)
    .selectAll<SVGTextElement, boolean>(".point-label")
    .data([true])
    .join("text");
  const element = label.node();
  if (!element) return;
  setAttrIfChanged(element, "class", "point-label");
  setAttrIfChanged(element, "text-anchor", "middle");
  setAttrIfChanged(element, "fill", foreground);
  setAttrIfChanged(element, "font-size", px(CHART_TYPE.dataLabel));
  setAttrIfChanged(element, "font-weight", "700");
  setAttrIfChanged(element, "pointer-events", "none");
  setStyleIfChanged(element, "display", hideLabels ? "none" : "");
  setStyleIfChanged(element, "opacity", hideLabels ? "0" : "1");
  label
    .selectAll<SVGTSpanElement, string>("tspan")
    .data(lines)
    .join("tspan")
    .each(function (line, index) {
      setAttrIfChanged(this, "x", "0");
      setAttrIfChanged(this, "dy", index === 0 ? firstDy : "1.1em");
      setTextIfChanged(this, line);
    });
}

export function updateScatterPointsForDisplay<
  T extends { precision: string; x: number; y: number },
>(
  zoomGroup: d3.Selection<SVGGElement, unknown, null, undefined>,
  config: ScatterPointConfig<T>,
): void {
  const points = zoomGroup.selectAll<SVGGElement, T>(".dot-group");
  points.each(function (d) {
    const point = d3.select(this);
    if (config.getOpacity)
      setStyleIfChanged(this, "opacity", String(config.getOpacity(d)));
    if (config.getPointerEvents) {
      setStyleIfChanged(this, "pointer-events", config.getPointerEvents(d));
    }
    const shapeKey = config.getShapeKey
      ? config.getShapeKey(d)
      : resolveShapeKey(d.precision, config.selectedPrecisions);
    syncPointShape(
      point as unknown as d3.Selection<SVGGElement, unknown, null, undefined>,
      shapeKey,
      config.getColor(d),
    );
  });
  points.selectAll<SVGTextElement, unknown>(".point-label").each(function () {
    if (config.foreground) setAttrIfChanged(this, "fill", config.foreground);
    else this.removeAttribute("fill");
    setStyleIfChanged(this, "display", config.hideLabels ? "none" : "");
    setStyleIfChanged(this, "opacity", config.hideLabels ? "0" : "1");
  });
}

export interface TooltipContainerGeometry {
  bounds: DOMRect;
  width: number;
  height: number;
}

interface TooltipGeometry {
  width: number;
  height: number;
  sizeStyle: string;
  className: string | null;
}

let containerGeometryCache = new WeakMap<
  HTMLElement,
  TooltipContainerGeometry
>();
let tooltipGeometryCache = new WeakMap<HTMLDivElement, TooltipGeometry>();
let geometryResetPending = false;

function scheduleGeometryReset(): void {
  if (geometryResetPending) return;
  geometryResetPending = true;
  const reset = () => {
    containerGeometryCache = new WeakMap();
    tooltipGeometryCache = new WeakMap();
    geometryResetPending = false;
  };
  if (typeof requestAnimationFrame === "function") requestAnimationFrame(reset);
  else setTimeout(reset, 0);
}

export function invalidateTooltipGeometry(
  tooltip: HTMLDivElement | null,
): void {
  if (tooltip) tooltipGeometryCache.delete(tooltip);
}

function tooltipSizeStyleSignature(tooltip: HTMLDivElement): string {
  const { style } = tooltip;
  return [
    style.width,
    style.height,
    style.minWidth,
    style.minHeight,
    style.maxWidth,
    style.maxHeight,
    style.padding,
    style.borderWidth,
    style.boxSizing,
    style.font,
    style.fontSize,
    style.fontWeight,
    style.lineHeight,
    style.letterSpacing,
    style.whiteSpace,
  ].join("|");
}

function getTooltipGeometry(tooltip: HTMLDivElement): TooltipGeometry {
  const sizeStyle = tooltipSizeStyleSignature(tooltip);
  const className = tooltip.getAttribute("class");
  const cached = tooltipGeometryCache.get(tooltip);
  if (
    cached &&
    cached.sizeStyle === sizeStyle &&
    cached.className === className
  )
    return cached;
  const bounds = tooltip.getBoundingClientRect();
  const geometry = {
    width: bounds.width || tooltip.offsetWidth,
    height: bounds.height || tooltip.offsetHeight,
    sizeStyle,
    className,
  };
  tooltipGeometryCache.set(tooltip, geometry);
  scheduleGeometryReset();
  return geometry;
}

export function getTooltipContainerGeometry(
  container: HTMLElement,
): TooltipContainerGeometry {
  const cached = containerGeometryCache.get(container);
  if (cached) return cached;

  const geometry = {
    bounds: container.getBoundingClientRect(),
    width: container.clientWidth,
    height: container.clientHeight,
  };
  containerGeometryCache.set(container, geometry);
  scheduleGeometryReset();
  return geometry;
}

export function computeTooltipPosition(
  mx: number,
  my: number,
  tooltip:
    | d3.Selection<HTMLDivElement | null, unknown, null, undefined>
    | d3.Selection<HTMLDivElement, unknown, null, undefined>,
  container: HTMLElement,
  offset = 10,
  containerGeometry?: TooltipContainerGeometry,
): { left: number; top: number } {
  const node = tooltip.node();
  if (!node) return { left: mx + offset, top: my + offset };

  if (node.style.display !== "block") {
    node.style.display = "block";
    invalidateTooltipGeometry(node);
  }

  const tooltipGeometry = getTooltipGeometry(node);
  const tw = tooltipGeometry.width;
  const th = tooltipGeometry.height;
  const geometry = containerGeometry ?? getTooltipContainerGeometry(container);
  const rect = geometry.bounds;
  const cw = geometry.width;
  const ch = geometry.height;
  const EDGE_PAD = 4;

  let left = mx + offset + tw <= cw ? mx + offset : mx - offset - tw;
  left = Math.max(EDGE_PAD, Math.min(cw - tw - EDGE_PAD, left));

  let top = my + offset + th <= ch ? my + offset : my - offset - th;
  top = Math.max(EDGE_PAD, Math.min(ch - th - EDGE_PAD, top));

  const viewportWidth =
    document.documentElement.clientWidth || window.innerWidth;
  const viewportHeight =
    document.documentElement.clientHeight || window.innerHeight;
  left += rect.left;
  top += rect.top;
  left = Math.max(EDGE_PAD, Math.min(viewportWidth - tw - EDGE_PAD, left));
  top = Math.max(EDGE_PAD, Math.min(viewportHeight - th - EDGE_PAD, top));

  return { left, top };
}

export function updateScatterPointsOnZoom(
  zoomGroup: d3.Selection<SVGGElement, unknown, null, undefined>,
  newXScale: ContinuousScale,
  newYScale: ContinuousScale,
  className = ".dot-group",
): void {
  zoomGroup
    .selectAll<SVGGElement, { x: number; y: number }>(className)
    .attr("transform", (d) => `translate(${newXScale(d.x)},${newYScale(d.y)})`);
}
