// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
import * as d3 from "d3";

import type { ChartLayout, ChartSetupConfig } from "./types";
import { CHART_TYPE, px } from "./typography";
import {
  createLogoWatermark,
  createUnofficialWatermark,
  positionUnofficialWatermarkImage,
} from "./watermark";

export function setupChartStructure(
  svgElement: SVGSVGElement,
  config: ChartSetupConfig,
): ChartLayout {
  const {
    chartId,
    containerWidth,
    containerHeight,
    margin,
    watermark,
    clipContent = true,
    hideAxes = false,
  } = config;
  const width = containerWidth - margin.left - margin.right;
  const height = containerHeight - margin.top - margin.bottom;

  const svg = d3.select(svgElement);
  svg.attr("width", containerWidth).attr("height", containerHeight);

  const existingG = svg.select<SVGGElement>(".chart-root");
  const isNew = existingG.empty();

  if (isNew) {
    svg.selectAll("*").remove();

    const defs = svg.append("defs");

    if (watermark === "logo") {
      createLogoWatermark(
        svg,
        defs,
        containerWidth,
        containerHeight,
        width,
        height,
        margin,
        chartId,
      );
    } else if (watermark === "unofficial") {
      createUnofficialWatermark(svg, defs, width, height, margin, chartId);
    }

    const g = svg
      .append("g")
      .attr("class", "chart-root")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const gridGroup = g.append("g").attr("class", "grid");
    if (!hideAxes) {
      gridGroup
        .append("line")
        .attr("class", "border-right")
        .attr("x1", width)
        .attr("x2", width)
        .attr("y1", 0)
        .attr("y2", height);
      gridGroup
        .append("line")
        .attr("class", "border-top")
        .attr("x1", 0)
        .attr("x2", width)
        .attr("y1", 0)
        .attr("y2", 0);
    }

    const xAxisGroup = g
      .append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${height})`);
    const yAxisGroup = g.append("g").attr("class", "y-axis");

    let zoomGroup: d3.Selection<SVGGElement, unknown, null, undefined>;
    if (clipContent) {
      defs
        .append("clipPath")
        .attr("id", `clip-${config.chartId}`)
        .append("rect")
        .attr("width", width)
        .attr("height", height);

      zoomGroup = g
        .append("g")
        .attr("class", "zoom-group")
        .attr("clip-path", `url(#clip-${config.chartId})`);
    } else {
      zoomGroup = g.append("g").attr("class", "zoom-group");
    }

    if (config.yLabel) {
      svg
        .append("text")
        .attr("class", "y-axis-label")
        .attr("transform", "rotate(-90)")
        .attr("x", -(margin.top + height / 2))
        .attr("y", 14)
        .attr("text-anchor", "middle")
        .attr("font-size", px(CHART_TYPE.axisLabel))
        .text(config.yLabel);
    }

    if (config.xLabel) {
      svg
        .append("text")
        .attr("class", "x-axis-label")
        .attr("x", margin.left + width / 2)
        .attr("y", containerHeight - 10)
        .attr("text-anchor", "middle")
        .attr("font-size", px(CHART_TYPE.axisLabel))
        .text(config.xLabel);
    }

    return {
      svg,
      g,
      zoomGroup,
      xAxisGroup,
      yAxisGroup,
      gridGroup,
      defs,
      width,
      height,
      margin,
    };
  }

  const g = existingG;
  g.attr("transform", `translate(${margin.left},${margin.top})`);

  const defs = svg.select<SVGDefsElement>("defs");

  const gridGroup = g.select<SVGGElement>(".grid");
  if (!config.hideAxes) {
    gridGroup
      .select(".border-right")
      .attr("x1", width)
      .attr("x2", width)
      .attr("y2", height);
    gridGroup.select(".border-top").attr("x2", width);
  }

  const xAxisGroup = g
    .select<SVGGElement>(".x-axis")
    .attr("transform", `translate(0,${height})`);
  const yAxisGroup = g.select<SVGGElement>(".y-axis");

  if (clipContent) {
    defs
      .select(`#clip-${config.chartId} rect`)
      .attr("width", width)
      .attr("height", height);
  }

  const zoomGroup = g.select<SVGGElement>(".zoom-group");

  const logoPatternId = `logo-pattern-${chartId}`;
  const unofficialPatternId = `unofficial-pattern-${chartId}`;
  const hasLogo = !defs.select(`#${logoPatternId}`).empty();
  const hasUnofficial = !defs.select(`#${unofficialPatternId}`).empty();
  const hasUnofficialImage = !svg.select(".unofficial-watermark-image").empty();
  const needsSwitch =
    (watermark === "unofficial" && (!hasUnofficial || !hasUnofficialImage)) ||
    (watermark === "logo" && !hasLogo) ||
    (watermark === "none" && (hasLogo || hasUnofficial || hasUnofficialImage));

  if (needsSwitch) {
    svg.select(".watermark-rect").remove();
    svg.select(".unofficial-watermark-image").remove();
    defs.select(`#${logoPatternId}`).remove();
    defs.select(`#${unofficialPatternId}`).remove();
    if (watermark === "logo") {
      createLogoWatermark(
        svg,
        defs,
        containerWidth,
        containerHeight,
        width,
        height,
        margin,
        chartId,
      );
    } else if (watermark === "unofficial") {
      createUnofficialWatermark(svg, defs, width, height, margin, chartId);
    }
  } else {
    svg
      .select(".watermark-rect")
      .attr("x", margin.left)
      .attr("y", margin.top)
      .attr("width", width)
      .attr("height", height);
    if (watermark === "logo") {
      const logoSize = Math.min(width, height) * 0.6;
      const pattern = defs.select(`#${logoPatternId}`);
      if (!pattern.empty()) {
        pattern.attr("width", containerWidth).attr("height", containerHeight);
        pattern
          .select("image")
          .attr("width", logoSize)
          .attr("height", logoSize)
          .attr("x", margin.left + (width - logoSize) / 2)
          .attr("y", margin.top + (height - logoSize) / 2);
      }
    } else if (watermark === "unofficial") {
      positionUnofficialWatermarkImage(svg, width, height, margin);
    }
  }

  const yLabel = svg.select(".y-axis-label");
  if (!yLabel.empty()) {
    yLabel.attr("x", -(margin.top + height / 2)).text(config.yLabel ?? "");
  }

  const xLabelEl = svg.select(".x-axis-label");
  if (!xLabelEl.empty()) {
    xLabelEl
      .attr("x", margin.left + width / 2)
      .attr("y", containerHeight - 10)
      .text(config.xLabel ?? "");
  }

  return {
    svg,
    g,
    zoomGroup,
    xAxisGroup,
    yAxisGroup,
    gridGroup,
    defs,
    width,
    height,
    margin,
  };
}
