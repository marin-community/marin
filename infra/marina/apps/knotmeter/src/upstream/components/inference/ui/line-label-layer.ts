// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
import * as d3 from "d3";

import { pointNearestX } from "@/components/inference/ui/line-label-anchor";
import {
  setAttrIfChanged,
  setStyleIfChanged,
} from "@/lib/d3-chart/chart-update";
import { plotClipSize } from "@/lib/d3-chart/plot-bounds";
import { CHART_TYPE, px } from "@/lib/d3-chart/typography";

export interface CartesianPoint {
  x: number;
  y: number;
}

export interface LineLabelSeries<TPoint extends CartesianPoint> {
  key: string;
  seriesId: string;
  label: string;
  color: string;
  points: readonly TPoint[];
}

export interface LineLabelPlacement {
  key: string;
  seriesId: string;
  label: string;
  color: string;
  x: number;
  y: number;
  visible: boolean;
}

export interface RectBounds {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

export function rectsOverlap(a: RectBounds, b: RectBounds): boolean {
  return !(
    a.right < b.left ||
    a.left > b.right ||
    a.bottom < b.top ||
    a.top > b.bottom
  );
}

export function rectsStrictlyOverlap(a: RectBounds, b: RectBounds): boolean {
  return (
    a.right > b.left && a.left < b.right && a.bottom > b.top && a.top < b.bottom
  );
}

export function firstNonCollidingRect(
  candidates: readonly RectBounds[],
  placed: readonly RectBounds[],
): number | null {
  for (let i = 0; i < candidates.length; i++) {
    if (!placed.some((other) => rectsOverlap(candidates[i], other))) return i;
  }
  return null;
}

export function horizontalShiftIntoBounds(
  rect: RectBounds,
  bounds: RectBounds,
): number | null {
  if (rect.right - rect.left > bounds.right - bounds.left) return null;
  if (rect.left < bounds.left) return bounds.left - rect.left;
  if (rect.right > bounds.right) return bounds.right - rect.right;
  return 0;
}

export function verticalShiftIntoBounds(
  rect: RectBounds,
  bounds: RectBounds,
): number | null {
  if (rect.bottom - rect.top > bounds.bottom - bounds.top) return null;
  if (rect.top < bounds.top) return bounds.top - rect.top;
  if (rect.bottom > bounds.bottom) return bounds.bottom - rect.bottom;
  return 0;
}

export function fitsVertically(rect: RectBounds, bounds: RectBounds): boolean {
  return rect.top >= bounds.top && rect.bottom <= bounds.bottom;
}

export function pillShiftIntoBounds(
  rect: RectBounds,
  bounds: RectBounds,
): CartesianPoint {
  return {
    x: horizontalShiftIntoBounds(rect, bounds) ?? bounds.left - rect.left,
    y: verticalShiftIntoBounds(rect, bounds) ?? bounds.top - rect.top,
  };
}

function pillLocalBox(node: SVGGElement): RectBounds | null {
  const background = node.querySelector<SVGRectElement>(".ll-bg");
  if (!background) return null;
  const x = Number(background.getAttribute("x"));
  const y = Number(background.getAttribute("y"));
  const width = Number(background.getAttribute("width"));
  const height = Number(background.getAttribute("height"));
  if (![x, y, width, height].every((value) => Number.isFinite(value)))
    return null;
  if (!(width > 0) || !(height > 0)) return null;
  return { left: x, top: y, right: x + width, bottom: y + height };
}

function pillBoxAt(local: RectBounds, tx: number, ty: number): RectBounds {
  return {
    left: tx + local.left,
    right: tx + local.right,
    top: ty + local.top,
    bottom: ty + local.bottom,
  };
}

interface PillLayoutItem {
  node: SVGGElement;
  label: LineLabelPlacement;
}

function layoutPills(
  items: readonly PillLayoutItem[],
  offsetX: number,
  offsetY: number,
  bounds: RectBounds | null,
  obstacles: readonly RectBounds[],
): void {
  const placed: RectBounds[] = [...obstacles];

  for (const { node, label } of items) {
    const tx0 = label.x + offsetX;
    const ty0 = label.y + offsetY;
    const local = bounds ? pillLocalBox(node) : null;
    if (!bounds || !local || !label.visible) {
      node.setAttribute("transform", `translate(${tx0},${ty0})`);
      continue;
    }

    const mirrorX = 2 * label.x - tx0 - local.left - local.right;
    const mirrorY = 2 * label.y - ty0 - local.top - local.bottom;
    const candidates: [number, number][] = [
      [tx0, ty0],
      [tx0, mirrorY],
      [mirrorX, ty0],
      [mirrorX, mirrorY],
    ];
    const rowHeight = local.bottom - local.top + 4;
    for (let row = 1; row <= 3; row++) {
      for (const cx of [tx0, mirrorX]) {
        candidates.push(
          [cx, ty0 - row * rowHeight],
          [cx, ty0 + row * rowHeight],
        );
      }
    }

    const clamped = candidates.map(([cx, cy]) => {
      const shift = pillShiftIntoBounds(pillBoxAt(local, cx, cy), bounds);
      const tx = cx + shift.x;
      const ty = cy + shift.y;
      return { tx, ty, box: pillBoxAt(local, tx, ty) };
    });
    const chosen =
      clamped.find(
        (candidate) =>
          !placed.some((other) => rectsStrictlyOverlap(candidate.box, other)),
      ) ?? clamped[0];

    node.setAttribute("transform", `translate(${chosen.tx},${chosen.ty})`);
    placed.push(chosen.box);
  }
}

export function plotAreaBounds(
  zoomGroup: d3.Selection<SVGGElement, unknown, null, undefined>,
): RectBounds | null {
  const node = zoomGroup.node();
  const size = node ? plotClipSize(node) : null;
  return size
    ? { left: 0, top: 0, right: size.width, bottom: size.height }
    : null;
}

export interface PlacedBox {
  x: number;
  y: number;
  halfW: number;
}

export function parallelismLabelBoxes(root: SVGGElement | null): PlacedBox[] {
  const boxes: PlacedBox[] = [];
  if (!root) return boxes;
  for (const node of root.querySelectorAll<SVGGElement>(".parallelism-label")) {
    if (node.style.opacity === "0") continue;
    const match = /translate\((?<tx>[^,]+),(?<ty>[^)]+)\)/u.exec(
      node.getAttribute("transform") ?? "",
    );
    if (!match?.groups) continue;
    const x = Number(match.groups.tx);
    const y = Number(match.groups.ty);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const width = Number(
      node.querySelector(".pl-bg")?.getAttribute("width") ?? 0,
    );
    boxes.push({ x, y, halfW: (Number.isFinite(width) ? width : 0) / 2 });
  }
  return boxes;
}

function pillObstacles(
  zoomGroup: d3.Selection<SVGGElement, unknown, null, undefined>,
  selector = ".line-label, .parallelism-label",
): RectBounds[] {
  const boxes: RectBounds[] = [];
  zoomGroup.selectAll<SVGGElement, unknown>(selector).each(function () {
    if (this.style.opacity === "0") return;
    const match = /translate\((?<tx>[^,]+),(?<ty>[^)]+)\)/u.exec(
      this.getAttribute("transform") ?? "",
    );
    const background = this.querySelector<SVGRectElement>(".ll-bg, .pl-bg");
    if (!match?.groups || !background) return;
    const tx = Number(match.groups.tx);
    const ty = Number(match.groups.ty);
    const x = Number(background.getAttribute("x"));
    const y = Number(background.getAttribute("y"));
    const width = Number(background.getAttribute("width"));
    const height = Number(background.getAttribute("height"));
    if (![tx, ty, x, y, width, height].every((value) => Number.isFinite(value)))
      return;
    boxes.push({
      left: tx + x,
      top: ty + y,
      right: tx + x + width,
      bottom: ty + y + height,
    });
  });
  return boxes;
}

const ANCHOR_SLOTS = [0.25, 0.5, 0.75, 1] as const;

function lineCandidates<TPoint extends CartesianPoint>(
  points: readonly TPoint[],
  seriesIndex: number,
): TPoint[] {
  const last = points.length - 1;
  if (last <= 0) return [points[0]];
  const at = (fraction: number) =>
    points[Math.max(1, Math.min(last, Math.round(fraction * last)))];
  const candidates: TPoint[] = [];
  for (let slot = 0; slot < ANCHOR_SLOTS.length; slot++) {
    const point = at(ANCHOR_SLOTS[(slot + seriesIndex) % ANCHOR_SLOTS.length]);
    if (!candidates.includes(point)) candidates.push(point);
  }
  return candidates;
}

export function placeLineLabels<TPoint extends CartesianPoint>(
  series: readonly LineLabelSeries<TPoint>[],
  xScale: (value: number) => number,
  yScale: (value: number) => number,
  options: {
    collisionWidth: number;
    collisionHeight?: number;
    anchors?: Map<string, number>;
    pinAnchors?: boolean;
    obstacles?: readonly PlacedBox[];
  },
): LineLabelPlacement[] {
  const collisionHeight = options.collisionHeight ?? CHART_TYPE.lineLabel + 8;
  const placed: PlacedBox[] = [...(options.obstacles ?? [])];
  const result: LineLabelPlacement[] = [];
  const sorted = [...series].toSorted(
    (a, b) => yScale(a.points[0]?.y ?? 0) - yScale(b.points[0]?.y ?? 0),
  );

  const labelHalfWidth = options.collisionWidth / 2;
  const collides = (x: number, y: number) =>
    placed.some(
      (other) =>
        Math.abs(other.y - y) < collisionHeight &&
        Math.abs(other.x - x) < other.halfW + labelHalfWidth,
    );
  const collisionCost = (x: number, y: number) =>
    placed.reduce((cost, other) => {
      const dx = other.halfW + labelHalfWidth - Math.abs(other.x - x);
      const dy = collisionHeight - Math.abs(other.y - y);
      return dx > 0 && dy > 0 ? cost + dx * dy : cost;
    }, 0);

  const emit = (entry: LineLabelSeries<TPoint>, point: TPoint) => {
    const x = xScale(point.x);
    const y = yScale(point.y);
    placed.push({ x, y, halfW: labelHalfWidth });
    result.push({
      key: entry.key,
      seriesId: entry.seriesId,
      label: entry.label,
      color: entry.color,
      x,
      y,
      visible: true,
    });
  };

  const crowded: { entry: LineLabelSeries<TPoint>; candidates: TPoint[] }[] =
    [];

  for (const [seriesIndex, entry] of sorted.entries()) {
    if (entry.points.length === 0) continue;
    const candidates = lineCandidates(entry.points, seriesIndex);

    if (options.pinAnchors && options.anchors) {
      let anchorX = options.anchors.get(entry.key);
      if (anchorX === undefined) {
        const candidate = candidates.find(
          (point) => !collides(xScale(point.x), yScale(point.y)),
        );
        anchorX = (candidate ?? candidates.at(-1)!).x;
        options.anchors.set(entry.key, anchorX);
      }
      const point = pointNearestX(entry.points, anchorX);
      const x = xScale(point.x);
      const y = yScale(point.y);
      placed.push({ x, y, halfW: labelHalfWidth });
      result.push({
        key: entry.key,
        seriesId: entry.seriesId,
        label: entry.label,
        color: entry.color,
        x,
        y,
        visible: true,
      });
      continue;
    }

    const candidate = candidates.find(
      (point) => !collides(xScale(point.x), yScale(point.y)),
    );
    if (candidate) {
      emit(entry, candidate);
      continue;
    }

    crowded.push({ entry, candidates });
  }

  for (const { entry, candidates } of crowded) {
    emit(
      entry,
      candidates.reduce((best, point) =>
        collisionCost(xScale(point.x), yScale(point.y)) <
        collisionCost(xScale(best.x), yScale(best.y))
          ? point
          : best,
      ),
    );
  }

  return result;
}

export interface LineLabelIconSpec {
  href: string;
  width: number;
  height: number;
}

const ICON_TEXT_GAP = 4;

function iconSpace(icon: LineLabelIconSpec | undefined): number {
  return icon ? icon.width + ICON_TEXT_GAP : 0;
}

export function renderLineLabels(
  group: d3.Selection<SVGGElement, unknown, null, undefined>,
  labels: readonly LineLabelPlacement[],
  options: {
    seriesAttribute: string;
    opacity?: number;
    offsetX?: number;
    offsetY?: number;
    iconFor?: (label: LineLabelPlacement) => LineLabelIconSpec | undefined;
    configureText?: (
      text: d3.Selection<SVGTextElement, LineLabelPlacement, null, undefined>,
      label: LineLabelPlacement,
    ) => void;
    configureGroup?: (
      labelGroup: d3.Selection<
        SVGGElement,
        LineLabelPlacement,
        null,
        undefined
      >,
      label: LineLabelPlacement,
    ) => void;
    bounds?: RectBounds | null;
  },
): void {
  const opacity = options.opacity ?? 1;
  const offsetX = options.offsetX ?? 8;
  const offsetY = options.offsetY ?? -14;
  const bounds =
    options.bounds === undefined ? plotAreaBounds(group) : options.bounds;
  const selection = group
    .selectAll<SVGGElement, LineLabelPlacement>(".line-label")
    .data(labels, (label) => label.key)
    .join(
      (enter) => {
        const labelGroup = enter
          .append("g")
          .attr("class", "line-label")
          .style("pointer-events", "none");
        labelGroup
          .append("rect")
          .attr("class", "ll-bg")
          .attr("rx", 4)
          .attr("ry", 4);
        labelGroup
          .append("text")
          .attr("class", "ll-text")
          .attr("text-anchor", "start")
          .attr("dominant-baseline", "central")
          .attr("fill", "white")
          .attr("font-size", px(CHART_TYPE.lineLabel))
          .attr("font-weight", "600");
        return labelGroup;
      },
      (update) => update,
      (exit) => exit.remove(),
    )
    .attr("data-line-key", (label) => label.key)
    .attr(options.seriesAttribute, (label) => label.seriesId)
    .attr(
      "transform",
      (label) => `translate(${label.x + offsetX},${label.y + offsetY})`,
    )
    .style("transition", "opacity 150ms ease")
    .style("opacity", (label) => (label.visible ? opacity : 0));

  selection.each(function (label) {
    const labelGroup = d3.select<SVGGElement, LineLabelPlacement>(this);
    options.configureGroup?.(labelGroup, label);
    const text = labelGroup.select<SVGTextElement>(".ll-text");
    const space = iconSpace(options.iconFor?.(label));
    if (space > 0) text.attr("x", space);
    else text.attr("x", null);
    if (options.configureText) options.configureText(text, label);
    else text.text(label.label);
  });

  const measured: {
    node: SVGGElement;
    label: LineLabelPlacement;
    bbox: DOMRect;
  }[] = [];
  selection.each(function (label) {
    const text = this.querySelector<SVGTextElement>(".ll-text");
    if (text) measured.push({ node: this, label, bbox: text.getBBox() });
  });
  for (const { node, label, bbox } of measured) {
    const labelGroup = d3.select(node);
    const icon = options.iconFor?.(label);
    const space = iconSpace(icon);
    labelGroup
      .select(".ll-bg")
      .attr("x", bbox.x - space - 5)
      .attr("y", bbox.y - 3)
      .attr("width", bbox.width + space + 10)
      .attr("height", bbox.height + 6)
      .attr("fill", label.color);

    labelGroup
      .selectAll<SVGImageElement, LineLabelIconSpec>(".ll-logo")
      .data(icon ? [icon] : [])
      .join("image")
      .attr("class", "ll-logo")
      .attr("href", (d) => d.href)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .attr("x", bbox.x - space)
      .attr("y", (d) => bbox.y + bbox.height / 2 - d.height / 2)
      .attr("width", (d) => d.width)
      .attr("height", (d) => d.height);
  }

  layoutPills(
    measured.map(({ node, label }) => ({ node, label })),
    offsetX,
    offsetY,
    bounds,
    bounds ? pillObstacles(group, ".parallelism-label") : [],
  );
}

export function updateRenderedLineLabels(
  group: d3.Selection<SVGGElement, unknown, null, undefined>,
  labels: readonly LineLabelPlacement[],
  options: {
    opacity?: number;
    offsetX?: number;
    offsetY?: number;
    bounds?: RectBounds | null;
  } = {},
): void {
  const byKey = new Map(labels.map((label) => [label.key, label]));
  const opacity = options.opacity ?? 1;
  const offsetX = options.offsetX ?? 8;
  const offsetY = options.offsetY ?? -14;
  const bounds =
    options.bounds === undefined ? plotAreaBounds(group) : options.bounds;
  const items: PillLayoutItem[] = [];
  group.selectAll<SVGGElement, unknown>(".line-label").each(function () {
    const element = d3.select(this);
    const label = byKey.get(element.attr("data-line-key"));
    if (!label) {
      element.style("opacity", 0);
      return;
    }
    element.style("opacity", label.visible ? opacity : 0);
    items.push({ node: this, label });
  });
  const order = new Map(labels.map((label, index) => [label.key, index]));
  items.sort(
    (a, b) => (order.get(a.label.key) ?? 0) - (order.get(b.label.key) ?? 0),
  );
  layoutPills(
    items,
    offsetX,
    offsetY,
    bounds,
    bounds ? pillObstacles(group, ".parallelism-label") : [],
  );
}

export const PLACEMENT_HIDDEN_ATTR = "data-placement-hidden";

export interface PointLabelPlacementOptions {
  bounds?: RectBounds | null;
  avoidCollisions?: boolean;
}

export function placePointLabels(
  zoomGroup: d3.Selection<SVGGElement, unknown, null, undefined>,
  options: PointLabelPlacementOptions = {},
): void {
  const bounds =
    options.bounds === undefined ? plotAreaBounds(zoomGroup) : options.bounds;
  const avoidCollisions = options.avoidCollisions ?? true;
  interface LabelInfo {
    element: SVGTextElement;
    tspans: SVGTSpanElement[];
    centerX: number;
    centerY: number;
    width: number;
    lineCount: number;
    defaultFirstY: number;
    placementHidden: boolean;
  }
  const pending: Omit<LabelInfo, "width">[] = [];
  const ascent = 9;
  const descent = 3;
  const lineHeight = 11;
  const padding = 2;

  zoomGroup.selectAll<SVGGElement, unknown>(".dot-group").each(function () {
    const label = this.querySelector<SVGTextElement>(".point-label");
    if (!label) return;
    const placementHidden = label.hasAttribute(PLACEMENT_HIDDEN_ATTR);
    const release = () => {
      if (!placementHidden) return;
      label.removeAttribute(PLACEMENT_HIDDEN_ATTR);
      label.style.opacity = "1";
    };
    if (
      label.style.display === "none" ||
      label.style.visibility === "hidden" ||
      (!placementHidden && label.style.opacity === "0") ||
      this.style.opacity === "0"
    ) {
      release();
      return;
    }
    const tspans = [...label.querySelectorAll<SVGTSpanElement>("tspan")];
    const transform = this.getAttribute("transform") ?? "";
    const match = transform.match(/translate\((?<tx>[^,]+),(?<ty>[^)]+)\)/u);
    if (tspans.length === 0 || !match) {
      release();
      return;
    }
    const lineCount = tspans.length;
    pending.push({
      element: label,
      tspans,
      centerX: Number.parseFloat(match[1]),
      centerY: Number.parseFloat(match[2]),
      lineCount,
      defaultFirstY: -(8 + (lineCount - 1) * lineHeight),
      placementHidden,
    });
  });

  const labels: LabelInfo[] = pending.map((label) => ({
    ...label,
    width: label.element.getBBox().width,
  }));
  labels.sort((a, b) => a.centerX - b.centerX);
  const placed: RectBounds[] = avoidCollisions ? pillObstacles(zoomGroup) : [];

  for (const label of labels) {
    const blockHeight = (label.lineCount - 1) * lineHeight + ascent + descent;
    const firstBaselines = [
      label.defaultFirstY,
      14,
      label.defaultFirstY - blockHeight - 2,
      14 + blockHeight + 2,
    ];
    const candidates: { firstY: number; dx: number; box: RectBounds }[] = [];
    for (const firstY of firstBaselines) {
      const box: RectBounds = {
        left: label.centerX - label.width / 2 - padding,
        right: label.centerX + label.width / 2 + padding,
        top: label.centerY + firstY - ascent - padding,
        bottom:
          label.centerY +
          firstY +
          (label.lineCount - 1) * lineHeight +
          descent +
          padding,
      };
      let dx = 0;
      if (bounds) {
        if (!fitsVertically(box, bounds)) continue;
        const shift = horizontalShiftIntoBounds(box, bounds);
        if (shift === null) continue;
        dx = shift;
      }
      candidates.push({
        firstY,
        dx,
        box: {
          left: box.left + dx,
          right: box.right + dx,
          top: box.top,
          bottom: box.bottom,
        },
      });
    }
    const index = avoidCollisions
      ? firstNonCollidingRect(
          candidates.map((candidate) => candidate.box),
          placed,
        )
      : candidates.length > 0
        ? 0
        : null;
    if (index === null) {
      setAttrIfChanged(label.tspans[0], "dy", `${label.defaultFirstY}px`);
      for (const tspan of label.tspans) setAttrIfChanged(tspan, "x", "0");
      setStyleIfChanged(label.element, "opacity", "0");
      if (!label.placementHidden)
        label.element.setAttribute(PLACEMENT_HIDDEN_ATTR, "");
      continue;
    }
    const chosen = candidates[index];
    setAttrIfChanged(label.tspans[0], "dy", `${chosen.firstY}px`);
    for (const tspan of label.tspans)
      setAttrIfChanged(tspan, "x", String(chosen.dx));
    if (label.placementHidden)
      label.element.removeAttribute(PLACEMENT_HIDDEN_ATTR);
    setStyleIfChanged(label.element, "opacity", "1");
    if (avoidCollisions) placed.push(chosen.box);
  }
}

export function avoidPointLabelCollisions(
  zoomGroup: d3.Selection<SVGGElement, unknown, null, undefined>,
  bounds?: RectBounds | null,
): void {
  placePointLabels(zoomGroup, { bounds, avoidCollisions: true });
}

export function keepPointLabelsInPlot(
  zoomGroup: d3.Selection<SVGGElement, unknown, null, undefined>,
  bounds?: RectBounds | null,
): void {
  placePointLabels(zoomGroup, { bounds, avoidCollisions: false });
}
