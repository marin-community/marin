// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
import * as d3 from "d3";

import { CHART_TYPE, px } from "../typography";

export interface RadarConfig<T = any> {
  axes: { label: string; unit?: string }[];
  getValue: (d: T, axisIndex: number) => number | null;
  getRawValue?: (d: T, axisIndex: number) => number | null;
  getColor: (d: T) => string;
  getLabel?: (d: T) => string;
  levels?: number;
  labelMargin?: number;
  keyFn?: (d: T) => string;
}

export interface RadarDot<T> {
  item: T;
  axisIndex: number;
  cx: number;
  cy: number;
  color: string;
  key: string;
}

export function renderRadar<T>(
  group: d3.Selection<SVGGElement, unknown, null, undefined>,
  data: T[],
  width: number,
  height: number,
  config: RadarConfig<T>,
): d3.Selection<SVGCircleElement, RadarDot<T>, SVGGElement, unknown> {
  const {
    axes,
    getValue,
    getColor,
    levels = 5,
    labelMargin = 30,
    keyFn,
  } = config;
  const numAxes = axes.length;
  const angleSlice = (Math.PI * 2) / numAxes;

  const size = Math.min(width, height);
  const radius = (size - labelMargin * 2) / 2;
  const centerX = width / 2;
  const centerY = height / 2;

  let radarG = group.select<SVGGElement>(".radar-center");
  if (radarG.empty()) {
    radarG = group.append("g").attr("class", "radar-center");
  }
  radarG.attr("transform", `translate(${centerX},${centerY})`);

  if (radius <= 0) {
    return radarG
      .selectAll<SVGCircleElement, RadarDot<T>>(".radar-dot")
      .data([] as RadarDot<T>[]);
  }

  const cosAngles = new Float64Array(numAxes);
  const sinAngles = new Float64Array(numAxes);
  for (let i = 0; i < numAxes; i++) {
    const angle = angleSlice * i - Math.PI / 2;
    cosAngles[i] = Math.cos(angle);
    sinAngles[i] = Math.sin(angle);
  }

  const rPerLevel = radius / levels;

  const gridData = d3.range(1, levels + 1);
  radarG
    .selectAll<SVGCircleElement, number>(".radar-grid")
    .data(gridData, (d) => d)
    .join("circle")
    .attr("class", "radar-grid")
    .attr("cx", 0)
    .attr("cy", 0)
    .attr("r", (d) => rPerLevel * d)
    .attr("fill", "none")
    .attr("stroke", "var(--border-alt)")
    .attr("stroke-width", 0.5)
    .attr("stroke-dasharray", (d) => (d < levels ? "2,3" : "none"));

  radarG
    .selectAll<SVGTextElement, number>(".radar-grid-label")
    .data(gridData, (d) => d)
    .join("text")
    .attr("class", "radar-grid-label")
    .attr("x", 4)
    .attr("y", (d) => -rPerLevel * d + 4)
    .attr("font-size", px(CHART_TYPE.micro))
    .attr("fill", "var(--border)")
    .attr("opacity", 0.6)
    .text((d) => `${Math.round((d / levels) * 100)}%`);

  radarG
    .selectAll<SVGLineElement, number>(".radar-spoke")
    .data(d3.range(numAxes), (d) => d)
    .join("line")
    .attr("class", "radar-spoke")
    .attr("x1", 0)
    .attr("y1", 0)
    .attr("x2", (_, i) => cosAngles[i] * radius)
    .attr("y2", (_, i) => sinAngles[i] * radius)
    .attr("stroke", "var(--border)")
    .attr("stroke-width", 0.8)
    .attr("opacity", 0.4);

  const labelRadius = radius + 20;
  const axisData = axes.map((a, i) => ({ ...a, index: i }));
  radarG
    .selectAll<SVGTextElement, (typeof axisData)[0]>(".radar-axis-label")
    .data(axisData, (d) => d.label)
    .join("text")
    .attr("class", "radar-axis-label")
    .attr("x", (d) => cosAngles[d.index] * labelRadius)
    .attr("y", (d) => sinAngles[d.index] * labelRadius)
    .attr("text-anchor", (d) => {
      const cos = cosAngles[d.index];
      if (Math.abs(cos) < 0.1) return "middle";
      return cos > 0 ? "start" : "end";
    })
    .attr("dominant-baseline", "central")
    .attr("font-size", px(CHART_TYPE.dataLabel))
    .attr("font-weight", "500")
    .attr("fill", "var(--foreground)")
    .text((d) => d.label);

  const lineGen = d3
    .line<[number, number]>()
    .x((d) => d[0])
    .y((d) => d[1]);

  radarG
    .selectAll<SVGPathElement, T>(".radar-polygon")
    .data(data, keyFn ?? ((_, i) => i))
    .join("path")
    .attr("class", "radar-polygon")
    .attr("d", (d) => {
      const points: [number, number][] = [];
      for (let i = 0; i < numAxes; i++) {
        const val = getValue(d, i);
        const r = val === null ? 0 : val * radius;
        points.push([cosAngles[i] * r, sinAngles[i] * r]);
      }
      points.push(points[0]);
      return lineGen(points);
    })
    .attr("fill", (d) => getColor(d))
    .attr("fill-opacity", 0.08)
    .attr("stroke", (d) => getColor(d))
    .attr("stroke-width", 1.5)
    .attr("stroke-opacity", 0.7);

  const dots: RadarDot<T>[] = [];
  for (let di = 0; di < data.length; di++) {
    const item = data[di];
    const color = getColor(item);
    const itemKey = keyFn ? keyFn(item) : String(di);
    for (let i = 0; i < numAxes; i++) {
      const val = getValue(item, i);
      if (val === null) continue;
      const r = val * radius;
      dots.push({
        item,
        axisIndex: i,
        cx: cosAngles[i] * r,
        cy: sinAngles[i] * r,
        color,
        key: `${itemKey}-${i}`,
      });
    }
  }

  return radarG
    .selectAll<SVGCircleElement, RadarDot<T>>(".radar-dot")
    .data(dots, (d) => d.key)
    .join("circle")
    .attr("class", "radar-dot")
    .attr("cx", (d) => d.cx)
    .attr("cy", (d) => d.cy)
    .attr("r", 3.5)
    .attr("fill", (d) => d.color)
    .attr("stroke", "white")
    .attr("stroke-width", 1)
    .style("cursor", "pointer");
}
