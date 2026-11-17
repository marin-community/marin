// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.

import React, { useImperativeHandle, useRef } from "react";
import * as d3 from "d3";

import { D3ChartWrapper } from "@/components/ui/d3-chart-wrapper";
import { useChartTooltipHandlers } from "@/hooks/useChartTooltipHandlers";
import { useChartZoom } from "@/hooks/useChartZoom";
import { useResponsiveChartDimensions } from "@/hooks/useResponsiveChartDimensions";

import type { ContinuousScale } from "../types";

import { isBandScale, type BuiltScale } from "./scale-builders";
import type { D3ChartHandle, D3ChartProps } from "./types";
import { useD3ChartRenderer } from "./useD3ChartRenderer";

function D3ChartInner<T>(
  {
    chartId,
    data,
    height = 600,
    margin,
    watermark,
    testId,
    grabCursor = true,
    instructions,
    clipContent,
    dataIdentity,
    metricIdentity,
    displayIdentity,
    xScale: xScaleConfig,
    yScale: yScaleConfig,
    xAxis,
    yAxis,
    layers,
    zoom: zoomConfig,
    tooltip: tooltipConfig,
    transitionDuration,
    legendElement,
    noDataOverlay,
    caption,
    scrollablePlot,
    onRender,
    onDisplayUpdate,
  }: D3ChartProps<T>,
  ref: React.ForwardedRef<D3ChartHandle>,
) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const tooltipRef = useRef<HTMLDivElement | null>(null);
  const scalesRef = useRef<{ xScale: BuiltScale; yScale: BuiltScale } | null>(
    null,
  );

  const { dimensions, setContainerRef } = useResponsiveChartDimensions({
    height,
  });

  const { zoomTransformRef, setupZoom } = useChartZoom({
    resetEventName:
      zoomConfig?.resetEventName ?? `d3chart_zoom_reset_${chartId}`,
    defaultZoomK: zoomConfig?.defaultZoomK,
    scaleExtent: zoomConfig?.scaleExtent ?? [1, 20],
    svgRef,
    onReset: zoomConfig?.onReset,
  });

  const {
    pinnedPoint,
    pinnedPointIsOverlay,
    pinTooltip,
    dismissTooltip,
    isPinned,
    hideTooltipElements,
    createRulers,
    attachHandlers,
  } = useChartTooltipHandlers<T>();

  useImperativeHandle(
    ref,
    () => ({
      dismissTooltip,
      hideTooltip: () => {
        hideTooltipElements(tooltipRef, svgRef);
      },
      getPinnedPoint: () => pinnedPoint,
      getPinnedPointIsOverlay: () => pinnedPointIsOverlay,
      isPinned,
      pinTooltip: pinTooltip as (point: unknown, isOverlay?: boolean) => void,
      getSvgElement: () => svgRef.current,
      getTooltipElement: () => tooltipRef.current,
      getScales: () => scalesRef.current,
      refreshDataPositions: () => {
        const svg = svgRef.current;
        const scales = scalesRef.current;
        if (!svg || !scales) return;
        if (isBandScale(scales.xScale) || isBandScale(scales.yScale)) return;
        const transform = d3.zoomTransform(svg);
        const curX = transform.rescaleX(scales.xScale as ContinuousScale);
        const curY = transform.rescaleY(scales.yScale as ContinuousScale);
        d3.select(svg)
          .selectAll<SVGGElement, { x: number; y: number }>(".dot-group")
          .attr("transform", (d) => `translate(${curX(d.x)},${curY(d.y)})`);
      },
    }),
    [
      dismissTooltip,
      hideTooltipElements,
      pinnedPoint,
      pinnedPointIsOverlay,
      isPinned,
      pinTooltip,
    ],
  );

  useD3ChartRenderer(
    {
      chartId,
      data,
      height,
      margin,
      watermark,
      testId,
      grabCursor,
      instructions,
      clipContent,
      dataIdentity,
      metricIdentity,
      displayIdentity,
      xScale: xScaleConfig,
      yScale: yScaleConfig,
      xAxis,
      yAxis,
      layers,
      zoom: zoomConfig,
      tooltip: tooltipConfig,
      transitionDuration,
      legendElement,
      noDataOverlay,
      onRender,
      onDisplayUpdate,
    },
    {
      svgRef,
      tooltipRef,
      dimensions,
      scalesRef,
      setupZoom,
      zoomTransformRef,
      isPinned,
      pinTooltip,
      dismissTooltip,
      hideTooltipElements,
      createRulers,
      attachHandlers,
    },
  );

  return (
    <D3ChartWrapper
      chartId={chartId}
      svgRef={svgRef}
      tooltipRef={tooltipRef}
      setContainerRef={setContainerRef}
      dimensions={dimensions}
      pinnedPoint={pinnedPoint}
      isPinned={isPinned}
      dismissTooltip={dismissTooltip}
      hideTooltipElements={hideTooltipElements}
      instructions={instructions}
      zoomEnabled={zoomConfig?.enabled ?? false}
      testId={testId}
      grabCursor={grabCursor}
      legendElement={legendElement}
      noDataOverlay={noDataOverlay}
      caption={caption}
      scrollablePlot={scrollablePlot}
    />
  );
}

export const D3Chart = React.memo(React.forwardRef(D3ChartInner)) as <T>(
  props: D3ChartProps<T> & { ref?: React.Ref<D3ChartHandle> },
) => React.ReactElement | null;
