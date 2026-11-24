// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
import type * as d3 from "d3";

import type { ChartLayout, ChartMargin, ContinuousScale } from "../types";
import type { AnyScale } from "../chart-update";
import type { BuiltScale } from "./scale-builders";
import type { BarConfig } from "../layers/bars";
import type { HorizontalBarConfig } from "../layers/horizontal-bars";
import type { PointConfig } from "../layers/points";
import type { ErrorBarConfig } from "../layers/error-bars";
import type { LineConfig } from "../layers/lines";
import type { RooflineConfig } from "../layers/rooflines";
import type { BarLabelConfig } from "../layers/bar-labels";
import type { ScatterPointConfig } from "../layers/scatter-points";
import type { RadarConfig } from "../layers/radar";

export type ScaleConfig =
  | { type: "band"; domain: string[]; padding?: number }
  | { type: "linear"; domain: [number, number]; nice?: boolean }
  | { type: "log"; domain: [number, number]; nice?: boolean }
  | { type: "time"; domain: [Date, Date]; nice?: boolean };

interface LayerBase {
  key?: string;
}

export interface BarLayerConfig<T> extends LayerBase {
  type: "bar";
  data: T[];
  config: BarConfig<T>;
}

export interface HorizontalBarLayerConfig<T> extends LayerBase {
  type: "horizontalBar";
  data: T[];
  config: HorizontalBarConfig<T>;
}

export interface PointLayerConfig<T> extends LayerBase {
  type: "point";
  data: T[];
  config: PointConfig<T>;
}

export interface ErrorBarLayerConfig<T> extends LayerBase {
  type: "errorBar";
  data: T[];
  config: ErrorBarConfig<T>;
}

export interface LineLayerConfig extends LayerBase {
  type: "line";
  lines: Record<string, { x: number; y: number }[]>;
  config: LineConfig;
}

export interface RooflineLayerConfig extends LayerBase {
  type: "roofline";
  rooflines: Record<string, { x: number; y: number }[]>;
  config: RooflineConfig;
}

export interface BarLabelLayerConfig<T> extends LayerBase {
  type: "barLabel";
  data: T[];
  config: BarLabelConfig<T>;
}

export interface ScatterLayerConfig<
  T extends { precision: string; x: number; y: number },
> extends LayerBase {
  type: "scatter";
  data: T[];
  config: ScatterPointConfig<T>;
  keyFn?: (d: T) => string;
}

export interface RadarLayerConfig<T> extends LayerBase {
  type: "radar";
  data: T[];
  config: RadarConfig<T>;
}

export interface CustomLayerConfig extends LayerBase {
  type: "custom";
  render:
    | ((
        zoomGroup: d3.Selection<SVGGElement, unknown, null, undefined>,
        ctx: RenderContext,
      ) => d3.Selection<any, any, any, any> | void)
    | null;
  displayIdentity?: string;
  onDisplayUpdate?:
    | ((
        zoomGroup: d3.Selection<SVGGElement, unknown, null, undefined>,
        ctx: RenderContext,
      ) => void)
    | null;
  onZoom?:
    | ((
        zoomGroup: d3.Selection<SVGGElement, unknown, null, undefined>,
        ctx: ZoomContext,
      ) => void)
    | null;
}

export type LayerConfig<T = any> =
  | BarLayerConfig<T>
  | HorizontalBarLayerConfig<T>
  | PointLayerConfig<T>
  | ErrorBarLayerConfig<T>
  | LineLayerConfig
  | RooflineLayerConfig
  | BarLabelLayerConfig<T>
  | ScatterLayerConfig<T & { precision: string; x: number; y: number }>
  | RadarLayerConfig<T>
  | CustomLayerConfig;

export interface AxisConfig {
  label?: string;
  tickFormat?: (d: d3.AxisDomain) => string;
  tickCount?: number;
  tickValues?: (number | Date)[] | ((scale: AnyScale) => (number | Date)[]);
  grid?: boolean;
  customize?: (
    axisGroup: d3.Selection<SVGGElement, unknown, null, undefined>,
  ) => void;
}

export interface ZoomConfig {
  enabled: boolean;
  axes?: "x" | "y" | "both";
  scaleExtent?: [number, number];
  resetEventName?: string;
  defaultZoomK?: number;
  constrain?: (
    transform: d3.ZoomTransform,
    extent: [[number, number], [number, number]],
    translateExtent: [[number, number], [number, number]],
  ) => d3.ZoomTransform;
  rescaleX?: (
    xScale: ContinuousScale,
    transform: d3.ZoomTransform,
  ) => ContinuousScale;
  rescaleY?: (
    yScale: ContinuousScale,
    transform: d3.ZoomTransform,
  ) => ContinuousScale;
  customTransformStorage?: (transform: d3.ZoomTransform) => d3.ZoomTransform;
  onReset?: () => void;
  onZoom?: (
    event: d3.D3ZoomEvent<SVGSVGElement, unknown>,
    ctx: ZoomContext,
  ) => void;
}

export interface TooltipConfig<T = any> {
  rulerType: "vertical" | "horizontal" | "crosshair" | "none";
  content: (d: T, isPinned: boolean) => string;
  getRulerX?: (
    d: T,
    xScale:
      | d3.ScaleBand<string>
      | d3.ScaleLinear<number, number>
      | d3.ScaleLogarithmic<number, number>,
  ) => number;
  getRulerY?: (
    d: T,
    yScale:
      | d3.ScaleLinear<number, number>
      | d3.ScaleLogarithmic<number, number>,
  ) => number;
  onHoverStart?: (sel: d3.Selection<any, T, any, any>, d: T) => void;
  onHoverEnd?: (sel: d3.Selection<any, T, any, any>, d: T) => void;
  onPointClick?: (d: T) => void;
  attachToLayer?: number;
  proximityHover?: boolean;
  getDataX?: (d: T) => number;
}

export interface RenderContext {
  layout: ChartLayout;
  tooltipElement: HTMLDivElement;
  xScale: AnyScale | d3.ScaleTime<number, number>;
  yScale: AnyScale | d3.ScaleTime<number, number>;
  renderedXScale?: AnyScale | d3.ScaleTime<number, number>;
  renderedYScale?: AnyScale | d3.ScaleTime<number, number>;
  width: number;
  height: number;
  transitionDuration?: number;
}

export interface ZoomContext extends RenderContext {
  newXScale: AnyScale | d3.ScaleTime<number, number>;
  newYScale: AnyScale | d3.ScaleTime<number, number>;
  transform: d3.ZoomTransform;
}

export interface D3ChartHandle {
  dismissTooltip: (clearPinnedPoint?: boolean) => void;
  hideTooltip: () => void;
  getPinnedPoint: () => unknown;
  getPinnedPointIsOverlay: () => boolean;
  isPinned: () => boolean;
  pinTooltip: (point: unknown, isOverlay?: boolean) => void;
  getSvgElement: () => SVGSVGElement | null;
  getTooltipElement: () => HTMLDivElement | null;
  getScales: () => { xScale: BuiltScale; yScale: BuiltScale } | null;
  refreshDataPositions: () => void;
}

export interface D3ChartProps<T = any> {
  chartId: string;
  data: T[];
  height?: number;
  margin?: ChartMargin;
  watermark?: "logo" | "unofficial" | "none";
  testId?: string;
  grabCursor?: boolean;
  instructions?: string;
  clipContent?: boolean;
  dataIdentity?: string;
  metricIdentity?: string;
  displayIdentity?: string;

  xScale?: ScaleConfig;
  yScale?: ScaleConfig;
  xAxis?: AxisConfig;
  yAxis?: AxisConfig;

  layers: LayerConfig<T>[];

  zoom?: ZoomConfig;
  tooltip?: TooltipConfig<T>;

  transitionDuration?: number;
  legendElement?: React.ReactNode;
  noDataOverlay?: React.ReactNode;
  caption?: React.ReactNode;
  scrollablePlot?: { minWidth: number; label: string; enabled: boolean };

  onRender?: (ctx: RenderContext) => void;
  onDisplayUpdate?: (ctx: RenderContext) => void;
}
