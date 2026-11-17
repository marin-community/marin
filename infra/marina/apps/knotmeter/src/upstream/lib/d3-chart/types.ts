// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
import type * as d3 from "d3";

export interface ChartLayout {
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  g: d3.Selection<SVGGElement, unknown, null, undefined>;
  zoomGroup: d3.Selection<SVGGElement, unknown, null, undefined>;
  xAxisGroup: d3.Selection<SVGGElement, unknown, null, undefined>;
  yAxisGroup: d3.Selection<SVGGElement, unknown, null, undefined>;
  gridGroup: d3.Selection<SVGGElement, unknown, null, undefined>;
  defs: d3.Selection<SVGDefsElement, unknown, null, undefined>;
  width: number;
  height: number;
  margin: ChartMargin;
}

export interface ChartMargin {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

export interface ChartSetupConfig {
  chartId: string;
  containerWidth: number;
  containerHeight: number;
  margin: ChartMargin;
  watermark: "logo" | "unofficial" | "none";
  xLabel?: string;
  yLabel?: string;
  clipContent?: boolean;
  hideAxes?: boolean;
}

export type ContinuousScale =
  | d3.ScaleLinear<number, number>
  | d3.ScaleLogarithmic<number, number>;
