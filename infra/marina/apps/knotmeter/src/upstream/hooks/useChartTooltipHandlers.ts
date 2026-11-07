// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
import * as d3 from "d3";
import { useCallback } from "react";
import {
  computeTooltipPosition,
  getTooltipContainerGeometry,
  invalidateTooltipGeometry,
} from "@/lib/d3-chart/layers/scatter-points";
import {
  installTouchInputTracking,
  isTouchCompatMouseEvent,
} from "@/lib/d3-chart/touch-input";
import { useStickyTooltip } from "./useStickyTooltip";

import { overlayMarkerPosition } from "@/lib/d3-chart/overlay-x-marker";
export type RulerType = "vertical" | "horizontal" | "crosshair" | "none";

export interface ChartTooltipConfig<TData> {
  rulerType: RulerType;

  generateTooltipContent: (data: TData, isPinned: boolean) => string;

  getRulerX?: (
    data: TData,
    xScale:
      | d3.ScaleBand<string>
      | d3.ScaleLinear<number, number, never>
      | d3.ScaleLogarithmic<number, number, never>,
  ) => number;

  getRulerY?: (
    data: TData,
    yScale:
      | d3.ScaleBand<string>
      | d3.ScaleLinear<number, number, never>
      | d3.ScaleLogarithmic<number, number, never>,
  ) => number;

  onHoverStart?: (
    selection: d3.Selection<any, TData, any, any>,
    data: TData,
  ) => void;

  onHoverEnd?: (
    selection: d3.Selection<any, TData, any, any>,
    data: TData,
  ) => void;

  onPointClick?: (data: TData) => void;
}

export interface ChartTooltipHandlers<TData> {
  pinnedPoint: TData | null;
  pinnedPointIsOverlay: boolean;
  pinTooltip: (data: TData, isOverlay?: boolean) => void;
  dismissTooltip: (clearPinnedPoint?: boolean) => void;
  isPinned: () => boolean;
  hideTooltipElements: (
    tooltipRef: React.RefObject<HTMLDivElement | null>,
    svgRef: React.RefObject<SVGSVGElement | null>,
  ) => void;

  createRulers: (
    group: d3.Selection<SVGGElement, unknown, null, undefined>,
    rulerType: RulerType,
    width: number,
    height: number,
    foregroundColor: string,
  ) => {
    rulerGroup: d3.Selection<SVGGElement, unknown, null, undefined>;
    verticalRuler?: d3.Selection<SVGLineElement, unknown, null, undefined>;
    horizontalRuler?: d3.Selection<SVGLineElement, unknown, null, undefined>;
  };

  attachHandlers: (
    selection: d3.Selection<any, TData, any, any>,
    config: ChartTooltipConfig<TData>,
    containerElement: HTMLDivElement,
    tooltipElement: d3.Selection<any, unknown, any, any>,
    rulers: {
      rulerGroup: d3.Selection<SVGGElement, unknown, null, undefined>;
      verticalRuler: d3.Selection<SVGLineElement, unknown, null, undefined>;
      horizontalRuler?: d3.Selection<SVGLineElement, unknown, null, undefined>;
    },
    xScale:
      | d3.ScaleBand<string>
      | d3.ScaleLinear<number, number, never>
      | d3.ScaleLogarithmic<number, number, never>,
    yScale?:
      | d3.ScaleBand<string>
      | d3.ScaleLinear<number, number, never>
      | d3.ScaleLogarithmic<number, number, never>,
    svgRef?: React.RefObject<SVGSVGElement | null>,
    zoomAxes?: "x" | "y" | "both",
  ) => void;
}

export function useChartTooltipHandlers<TData>(): ChartTooltipHandlers<TData> {
  const {
    pinnedPoint,
    pinnedPointIsOverlay,
    pinTooltip,
    dismissTooltip,
    isPinned,
    hideTooltipElements,
  } = useStickyTooltip<TData>();

  const createRulers = useCallback(
    (
      group: d3.Selection<SVGGElement, unknown, null, undefined>,
      rulerType: RulerType,
      width: number,
      height: number,
      foregroundColor: string,
    ) => {
      let rulerGroup = group.select<SVGGElement>(".ruler-group");
      if (rulerGroup.empty()) {
        rulerGroup = group
          .append("g")
          .attr("class", "ruler-group")
          .style("pointer-events", "none");
      }
      rulerGroup.style("display", "none");

      let verticalRuler:
        | d3.Selection<SVGLineElement, unknown, null, undefined>
        | undefined;

      if (rulerType !== "horizontal" && rulerType !== "none") {
        let vRuler = rulerGroup.select<SVGLineElement>(".vertical-ruler");
        if (vRuler.empty()) {
          vRuler = rulerGroup
            .append("line")
            .attr("class", "vertical-ruler")
            .attr("stroke-width", 1)
            .attr("stroke-dasharray", "4 4")
            .attr("opacity", 0.8);
        }
        vRuler.attr("stroke", foregroundColor).attr("y1", 0).attr("y2", height);
        verticalRuler = vRuler;
      }

      let horizontalRuler:
        | d3.Selection<SVGLineElement, unknown, null, undefined>
        | undefined;

      if (rulerType === "crosshair" || rulerType === "horizontal") {
        let hRuler = rulerGroup.select<SVGLineElement>(".horizontal-ruler");
        if (hRuler.empty()) {
          hRuler = rulerGroup
            .append("line")
            .attr("class", "horizontal-ruler")
            .attr("stroke-width", 1)
            .attr("stroke-dasharray", "4 4")
            .attr("opacity", 0.8);
        }
        hRuler.attr("stroke", foregroundColor).attr("x1", 0).attr("x2", width);
        horizontalRuler = hRuler;
      }

      return { rulerGroup, verticalRuler, horizontalRuler };
    },
    [],
  );

  const attachHandlers = useCallback(
    (
      selection: d3.Selection<any, TData, any, any>,
      config: ChartTooltipConfig<TData>,
      containerElement: HTMLDivElement,
      tooltipElement: d3.Selection<any, unknown, any, any>,
      rulers: {
        rulerGroup: d3.Selection<SVGGElement, unknown, null, undefined>;
        verticalRuler?: d3.Selection<SVGLineElement, unknown, null, undefined>;
        horizontalRuler?: d3.Selection<
          SVGLineElement,
          unknown,
          null,
          undefined
        >;
      },
      xScale:
        | d3.ScaleBand<string>
        | d3.ScaleLinear<number, number, never>
        | d3.ScaleLogarithmic<number, number, never>,
      yScale?:
        | d3.ScaleBand<string>
        | d3.ScaleLinear<number, number, never>
        | d3.ScaleLogarithmic<number, number, never>,
      svgRef?: React.RefObject<SVGSVGElement | null>,
      zoomAxes: "x" | "y" | "both" = "both",
    ) => {
      const { rulerGroup, verticalRuler, horizontalRuler } = rulers;
      const effectiveZoomAxes = zoomAxes;

      const getZoomedScales = () => {
        let curX = xScale;
        let curY = yScale;
        if (svgRef?.current) {
          const transform = d3.zoomTransform(svgRef.current);
          if (
            (effectiveZoomAxes === "x" || effectiveZoomAxes === "both") &&
            "invert" in xScale
          ) {
            curX = transform.rescaleX(xScale as any);
          }
          if (
            (effectiveZoomAxes === "y" || effectiveZoomAxes === "both") &&
            yScale &&
            "invert" in yScale
          ) {
            curY = transform.rescaleY(yScale as any);
          }
        }
        return { curX, curY };
      };

      installTouchInputTracking();

      selection
        .on("mouseenter", function (event, d) {
          if (isPinned()) return;
          if (isTouchCompatMouseEvent(event)) return;

          if (config.onHoverStart) {
            config.onHoverStart(d3.select(this), d);
          } else {
            d3.select(this).attr("opacity", 0.8);
          }

          tooltipElement
            .style("opacity", 1)
            .style("display", "block")
            .style("pointer-events", "none")
            .html(config.generateTooltipContent(d, false));
          invalidateTooltipGeometry(tooltipElement.node());

          const markerPosition =
            this instanceof Element ? overlayMarkerPosition(this) : null;
          const { curX: currentXScale, curY: currentYScale } =
            getZoomedScales();

          if (verticalRuler || horizontalRuler) {
            rulerGroup.style("display", "block");
          }
          if (verticalRuler && config.getRulerX) {
            const x = markerPosition?.x ?? config.getRulerX(d, currentXScale);
            verticalRuler.attr("x1", x).attr("x2", x);
          }

          if (horizontalRuler && currentYScale && config.getRulerY) {
            const y = markerPosition?.y ?? config.getRulerY(d, currentYScale);
            horizontalRuler.attr("y1", y).attr("y2", y);
          }
        })
        .on("mousemove", (event) => {
          if (isPinned()) return;
          if (isTouchCompatMouseEvent(event)) return;

          const geometry = getTooltipContainerGeometry(containerElement);
          const mx = event.clientX - geometry.bounds.left;
          const my = event.clientY - geometry.bounds.top;
          const pos = computeTooltipPosition(
            mx,
            my,
            tooltipElement,
            containerElement,
            10,
            geometry,
          );
          tooltipElement
            .style("left", `${pos.left}px`)
            .style("top", `${pos.top}px`);
        })
        .on("mouseleave", function (_event, d) {
          if (isPinned()) return;

          if (config.onHoverEnd) {
            config.onHoverEnd(d3.select(this), d);
          } else {
            d3.select(this).attr("opacity", 1);
          }

          tooltipElement.style("opacity", 0).style("display", "none");
          rulerGroup.style("display", "none");
        })
        .on("click", function (event, d) {
          event.stopPropagation();

          const geometry = getTooltipContainerGeometry(containerElement);
          const mx = event.clientX - geometry.bounds.left;
          const my = event.clientY - geometry.bounds.top;
          tooltipElement.html(config.generateTooltipContent(d, true));
          invalidateTooltipGeometry(tooltipElement.node());
          const pos = computeTooltipPosition(
            mx,
            my,
            tooltipElement,
            containerElement,
            10,
            geometry,
          );
          tooltipElement
            .style("left", `${pos.left}px`)
            .style("top", `${pos.top}px`)
            .style("opacity", 1)
            .style("display", "block")
            .style("pointer-events", "auto");

          const markerPosition =
            this instanceof Element ? overlayMarkerPosition(this) : null;
          const { curX: currentXScale, curY: currentYScale } =
            getZoomedScales();
          if (verticalRuler || horizontalRuler) {
            rulerGroup.style("display", "block");
          }
          if (verticalRuler && config.getRulerX) {
            const x = markerPosition?.x ?? config.getRulerX(d, currentXScale);
            verticalRuler.attr("x1", x).attr("x2", x);
          }
          if (horizontalRuler && currentYScale && config.getRulerY) {
            const y = markerPosition?.y ?? config.getRulerY(d, currentYScale);
            horizontalRuler.attr("y1", y).attr("y2", y);
          }

          pinTooltip(d);
          config.onPointClick?.(d);
        });
    },
    [isPinned, pinTooltip],
  );

  return {
    pinnedPoint,
    pinnedPointIsOverlay,
    pinTooltip,
    dismissTooltip,
    isPinned,
    hideTooltipElements,
    createRulers,
    attachHandlers,
  };
}
