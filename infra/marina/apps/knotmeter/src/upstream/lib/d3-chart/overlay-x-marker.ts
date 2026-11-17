// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
import * as d3 from "d3";
import { invalidateTooltipGeometry } from "@/lib/d3-chart/layers/scatter-points";
import {
  installTouchInputTracking,
  isTouchCompatMouseEvent,
} from "@/lib/d3-chart/touch-input";

export interface OverlayTooltipHandle<T> {
  isPinned: () => boolean;
  pinTooltip: (data: T, isOverlay: boolean) => void;
}

export interface OverlayRulerLifecycle<T> {
  show: (data: T, marker: SVGGElement) => void;
  hide: () => void;
}

export interface OverlayXMarkerOptions<T> {
  markerSelector: string;
  normalPath: string;
  hoverPath: string;
  normalStrokeWidth?: number;
  hoverStrokeWidth?: number;
  tooltip: d3.Selection<HTMLDivElement, unknown, null, undefined>;
  handle: OverlayTooltipHandle<T> | null;
  content: (data: T, pinned: boolean) => string;
  position: (event: PointerEvent | MouseEvent) => { left: number; top: number };
  rulers?: OverlayRulerLifecycle<T>;
  onClick?: (data: T) => void;
}

export function xMarkerPath(size: number, armScale = 1): string {
  const arm = size * armScale;
  return `M ${-arm} ${-arm} L ${arm} ${arm} M ${-arm} ${arm} L ${arm} ${-arm}`;
}

export function overlayMarkerPosition(
  marker: Element,
): { x: number; y: number } | null {
  const transform = marker.getAttribute("transform") ?? "";
  const match = transform.match(/translate\((?<x>[^, ]+)[, ]+(?<y>[^) ]+)\)/u);
  const x = Number(match?.groups?.x);
  const y = Number(match?.groups?.y);
  return Number.isFinite(x) && Number.isFinite(y) ? { x, y } : null;
}

export function attachOverlayXMarkerHandlers<T>(
  points: d3.Selection<SVGGElement, T, SVGGElement, unknown>,
  options: OverlayXMarkerOptions<T>,
): void {
  const normalStrokeWidth = options.normalStrokeWidth ?? 2.5;
  const hoverStrokeWidth = options.hoverStrokeWidth ?? 3.5;
  const positionTooltip = (event: PointerEvent | MouseEvent) => {
    const position = options.position(event);
    options.tooltip
      .style("left", `${position.left}px`)
      .style("top", `${position.top}px`);
  };

  installTouchInputTracking();

  points
    .on("mouseenter", function (event, data) {
      if (options.handle?.isPinned()) return;
      if (isTouchCompatMouseEvent(event)) return;
      d3.select(this)
        .select(options.markerSelector)
        .attr("d", options.hoverPath)
        .attr("stroke-width", hoverStrokeWidth);
      options.tooltip
        .style("opacity", 1)
        .style("display", "block")
        .style("pointer-events", "none")
        .html(options.content(data, false));
      invalidateTooltipGeometry(options.tooltip.node());
      options.rulers?.show(data, this);
    })
    .on("mousemove", (event) => {
      if (options.handle?.isPinned()) return;
      if (isTouchCompatMouseEvent(event)) return;
      positionTooltip(event);
    })
    .on("mouseleave", function () {
      if (options.handle?.isPinned()) return;
      d3.select(this)
        .select(options.markerSelector)
        .attr("d", options.normalPath)
        .attr("stroke-width", normalStrokeWidth);
      options.tooltip.style("opacity", 0).style("display", "none");
      options.rulers?.hide();
    })
    .on("click", function (event, data) {
      event.stopPropagation();
      options.tooltip
        .html(options.content(data, true))
        .style("opacity", 1)
        .style("display", "block")
        .style("pointer-events", "auto");
      invalidateTooltipGeometry(options.tooltip.node());
      positionTooltip(event);
      options.rulers?.show(data, this);
      options.handle?.pinTooltip(data, true);
      options.onClick?.(data);
    });
}
