// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
import type * as d3 from "d3";

type GroupSelection = d3.Selection<SVGGElement, unknown, null, undefined>;

export interface PerfRulerEndInput {
  py: number;
  rawY: number;
}

export interface PerfRulerGeometry {
  x: number;
  y1: number;
  y2: number;
  ratio: number;
  ratioLabel: string;
}

export function formatPerfRatio(ratio: number): string {
  if (!Number.isFinite(ratio) || ratio <= 0) return "";
  if (ratio >= 100) return `${Math.round(ratio)}x`;
  if (ratio >= 10) return `${ratio.toFixed(1)}x`;
  return `${ratio.toFixed(2)}x`;
}

export interface PerfRulerPathLike {
  getTotalLength: () => number;
  getPointAtLength: (length: number) => { x: number; y: number };
}

export function intersectPathAtX(
  path: PerfRulerPathLike,
  x: number,
  opts?: { tolerance?: number; maxIterations?: number },
): { x: number; y: number } | null {
  if (!Number.isFinite(x)) return null;
  const total = path.getTotalLength();
  if (!Number.isFinite(total) || total <= 0) return null;

  const start = path.getPointAtLength(0);
  const end = path.getPointAtLength(total);
  const ascending = end.x >= start.x;
  const minX = Math.min(start.x, end.x);
  const maxX = Math.max(start.x, end.x);
  const slack = 0.5;
  if (x < minX - slack || x > maxX + slack) return null;

  const tolerance = opts?.tolerance ?? 0.25;
  const maxIterations = opts?.maxIterations ?? 48;
  let lo = 0;
  let hi = total;
  let best = Math.abs(start.x - x) <= Math.abs(end.x - x) ? start : end;
  for (let i = 0; i < maxIterations; i++) {
    const mid = (lo + hi) / 2;
    const p = path.getPointAtLength(mid);
    if (Math.abs(p.x - x) < Math.abs(best.x - x)) best = p;
    if (Math.abs(p.x - x) <= tolerance) return { x: p.x, y: p.y };
    if (ascending === p.x < x) lo = mid;
    else hi = mid;
  }
  return { x: best.x, y: best.y };
}

export function computeIsoXRulerGeometry(
  x: number,
  a: PerfRulerEndInput,
  b: PerfRulerEndInput,
): PerfRulerGeometry | null {
  const inputs = [x, a.py, a.rawY, b.py, b.rawY];
  if (!inputs.every((value) => Number.isFinite(value))) return null;
  if (a.rawY <= 0 || b.rawY <= 0) return null;

  const hi = Math.max(a.rawY, b.rawY);
  const lo = Math.min(a.rawY, b.rawY);
  const ratio = hi / lo;

  return {
    x,
    y1: Math.min(a.py, b.py),
    y2: Math.max(a.py, b.py),
    ratio,
    ratioLabel: formatPerfRatio(ratio),
  };
}

export function isPerfRulerCurveVisible(
  opacity: string | null | undefined,
): boolean {
  if (opacity === null || opacity === undefined) return true;
  const trimmed = opacity.trim();
  if (trimmed === "") return true;
  const value = Number(trimmed);
  return Number.isNaN(value) ? true : value > 0;
}

export interface PerfRulerXRange {
  min: number;
  max: number;
}

export function pathXExtent(path: PerfRulerPathLike): PerfRulerXRange | null {
  const total = path.getTotalLength();
  if (!Number.isFinite(total) || total <= 0) return null;
  const start = path.getPointAtLength(0);
  const end = path.getPointAtLength(total);
  if (!Number.isFinite(start.x) || !Number.isFinite(end.x)) return null;
  return { min: Math.min(start.x, end.x), max: Math.max(start.x, end.x) };
}

export function clampIsoX(
  x: number,
  a: PerfRulerXRange,
  b?: PerfRulerXRange | null,
): number | null {
  if (!Number.isFinite(x)) return null;
  const min = b ? Math.max(a.min, b.min) : a.min;
  const max = b ? Math.min(a.max, b.max) : a.max;
  if (min > max) return null;
  return Math.min(Math.max(x, min), max);
}

export interface PerfRulerCurveClick {
  curve: string;
  isoX: number;
}

export interface PerfRulerMeasurement {
  id: number;
  curveA: string;
  curveB: string;
  isoX: number;
}

export interface PerfRulerState {
  rulers: PerfRulerMeasurement[];
  draft: PerfRulerCurveClick | null;
  nextId: number;
}

export const MAX_PERF_RULERS = 8;

export const EMPTY_PERF_RULER_STATE: PerfRulerState = {
  rulers: [],
  draft: null,
  nextId: 1,
};

export function nextPerfRulerState(
  prev: PerfRulerState,
  click: PerfRulerCurveClick,
  clampToOverlap?: (
    curveA: string,
    curveB: string,
    isoX: number,
  ) => number | null,
): PerfRulerState {
  if (!Number.isFinite(click.isoX)) return prev;
  if (!prev.draft)
    return { ...prev, draft: { curve: click.curve, isoX: click.isoX } };
  if (click.curve === prev.draft.curve) {
    return prev.draft.isoX === click.isoX
      ? prev
      : { ...prev, draft: { curve: prev.draft.curve, isoX: click.isoX } };
  }
  let isoX = prev.draft.isoX;
  if (clampToOverlap) {
    const clamped = clampToOverlap(prev.draft.curve, click.curve, isoX);
    if (clamped === null || !Number.isFinite(clamped)) return prev;
    isoX = clamped;
  }
  const measurement: PerfRulerMeasurement = {
    id: prev.nextId,
    curveA: prev.draft.curve,
    curveB: click.curve,
    isoX,
  };
  const rulers = [...prev.rulers, measurement];
  while (rulers.length > MAX_PERF_RULERS) rulers.shift();
  return { rulers, draft: null, nextId: prev.nextId + 1 };
}

export function movePerfRulerIsoX(
  prev: PerfRulerState,
  id: number,
  isoX: number,
): PerfRulerState {
  if (!Number.isFinite(isoX)) return prev;
  const index = prev.rulers.findIndex((ruler) => ruler.id === id);
  if (index === -1 || prev.rulers[index].isoX === isoX) return prev;
  const rulers = [...prev.rulers];
  rulers[index] = { ...rulers[index], isoX };
  return { ...prev, rulers };
}

export function deletePerfRuler(
  prev: PerfRulerState,
  id: number,
): PerfRulerState {
  const rulers = prev.rulers.filter((ruler) => ruler.id !== id);
  return rulers.length === prev.rulers.length ? prev : { ...prev, rulers };
}

export function clearPerfRulers(prev: PerfRulerState): PerfRulerState {
  if (prev.rulers.length === 0 && prev.draft === null) return prev;
  return { rulers: [], draft: null, nextId: prev.nextId };
}

export function prunePerfRulers(
  prev: PerfRulerState,
  curveExists: (curve: string) => boolean,
): PerfRulerState {
  const rulers = prev.rulers.filter(
    (ruler) => curveExists(ruler.curveA) && curveExists(ruler.curveB),
  );
  const draft = prev.draft && curveExists(prev.draft.curve) ? prev.draft : null;
  if (rulers.length === prev.rulers.length && draft === prev.draft) return prev;
  return { ...prev, rulers, draft };
}

const PERF_RULER_URL_RULER_SEPARATOR = ";";
const PERF_RULER_URL_FIELD_SEPARATOR = "|";
const PERF_RULER_CURVE_ID = /^(?:overlay-)?roofline-(?!path$)[\w%.~-]+$/u;

export function serializePerfRulers(state: PerfRulerState): string {
  return state.rulers
    .map((ruler) =>
      [Number(ruler.isoX.toPrecision(4)), ruler.curveA, ruler.curveB].join(
        PERF_RULER_URL_FIELD_SEPARATOR,
      ),
    )
    .join(PERF_RULER_URL_RULER_SEPARATOR);
}

export function parsePerfRulers(
  raw: string | null | undefined,
): PerfRulerState {
  if (!raw) return EMPTY_PERF_RULER_STATE;
  const parsed: Omit<PerfRulerMeasurement, "id">[] = [];
  for (const entry of raw.split(PERF_RULER_URL_RULER_SEPARATOR)) {
    const fields = entry.split(PERF_RULER_URL_FIELD_SEPARATOR);
    if (fields.length !== 3) continue;
    const [isoXField, curveA, curveB] = fields;
    if (isoXField.trim() === "" || curveA === curveB) continue;
    if (!PERF_RULER_CURVE_ID.test(curveA) || !PERF_RULER_CURVE_ID.test(curveB))
      continue;
    const isoX = Number(isoXField);
    if (!Number.isFinite(isoX)) continue;
    parsed.push({ curveA, curveB, isoX });
  }
  if (parsed.length === 0) return EMPTY_PERF_RULER_STATE;
  const kept = parsed.slice(-MAX_PERF_RULERS);
  return {
    rulers: kept.map((ruler, index) => ({ id: index + 1, ...ruler })),
    draft: null,
    nextId: kept.length + 1,
  };
}

export function perfRulerCurveSet(state: PerfRulerState): Set<string> {
  const curves = new Set<string>();
  for (const ruler of state.rulers) {
    curves.add(ruler.curveA);
    curves.add(ruler.curveB);
  }
  if (state.draft) curves.add(state.draft.curve);
  return curves;
}

export interface PerfRulerRenderOptions {
  color: string;
  halo?: string;
  capHalfWidth?: number;
  labelFontSize?: number;
  onDelete?: (id: number) => void;
}

export const DEFAULT_LABEL_FONT_SIZE = 32;
const LABEL_OFFSET_X = 46;
const LABEL_OFFSET_Y = 46;
const ARROW_TIP_GAP = 5;
const ARROW_HEAD_LENGTH = 9;
const ARROW_HEAD_HALF_WIDTH = 4.5;
const DELETE_GAP = 18;
const LABEL_COLLISION_NUDGE_TRIES = 4;

export interface PerfRulerLabelLayoutOptions {
  chartWidth?: number;
  chartHeight?: number;
  fontSize?: number;
}

export interface PerfRulerLabelLayout {
  labelX: number;
  labelY: number;
  textAnchor: "start" | "end";
  side: 1 | -1;
  arrowPath: string;
  arrowHeadPath: string;
  deleteX: number;
  deleteY: number;
}

function estimateLabelWidth(label: string, fontSize: number): number {
  return label.length * fontSize * 0.6;
}

function buildLabelLayout(
  geometry: Pick<PerfRulerGeometry, "x" | "y1" | "y2" | "ratioLabel">,
  side: 1 | -1,
  labelY: number,
  fontSize: number,
): PerfRulerLabelLayout {
  const midY = (geometry.y1 + geometry.y2) / 2;
  const labelX = geometry.x + side * LABEL_OFFSET_X;
  const above = labelY < midY;
  const startX = labelX + side * 4;
  const startY = labelY + (above ? 1 : -1) * (fontSize / 2 + 6);
  const tipX = geometry.x + side * ARROW_TIP_GAP;
  const backX = tipX + side * ARROW_HEAD_LENGTH;
  const arrowPath = `M ${startX} ${startY} Q ${startX} ${midY} ${backX} ${midY}`;
  const arrowHeadPath =
    `M ${tipX} ${midY} ` +
    `L ${backX} ${midY - ARROW_HEAD_HALF_WIDTH} ` +
    `L ${backX} ${midY + ARROW_HEAD_HALF_WIDTH} Z`;
  const estimatedWidth = estimateLabelWidth(geometry.ratioLabel, fontSize);
  return {
    labelX,
    labelY,
    textAnchor: side === 1 ? "start" : "end",
    side,
    arrowPath,
    arrowHeadPath,
    deleteX: labelX + side * (estimatedWidth + DELETE_GAP),
    deleteY: labelY,
  };
}

export function computePerfRulerLabelLayout(
  geometry: Pick<PerfRulerGeometry, "x" | "y1" | "y2" | "ratioLabel">,
  opts?: PerfRulerLabelLayoutOptions,
): PerfRulerLabelLayout {
  const fontSize = opts?.fontSize ?? DEFAULT_LABEL_FONT_SIZE;
  const midY = (geometry.y1 + geometry.y2) / 2;
  const estimatedWidth = estimateLabelWidth(geometry.ratioLabel, fontSize);
  const requiredRoom = LABEL_OFFSET_X + estimatedWidth + 8;

  let side: 1 | -1 = 1;
  if (opts?.chartWidth !== undefined) {
    const roomRight = opts.chartWidth - geometry.x;
    if (roomRight < requiredRoom && geometry.x > roomRight) side = -1;
  }

  let labelY = midY - LABEL_OFFSET_Y;
  if (labelY - fontSize / 2 < 4) {
    labelY = midY + LABEL_OFFSET_Y;
    if (
      opts?.chartHeight !== undefined &&
      labelY + fontSize / 2 > opts.chartHeight - 4
    ) {
      labelY = Math.max(fontSize / 2 + 4, midY - LABEL_OFFSET_Y);
    }
  }

  return buildLabelLayout(geometry, side, labelY, fontSize);
}

function clampLabelY(
  y: number,
  fontSize: number,
  chartHeight?: number,
): number {
  const min = fontSize / 2 + 4;
  const max =
    chartHeight === undefined
      ? Number.POSITIVE_INFINITY
      : chartHeight - 4 - fontSize / 2;
  return Math.max(Math.min(y, max), min);
}

function labelSpan(
  layout: PerfRulerLabelLayout,
  label: string,
  fontSize: number,
): { x0: number; x1: number } {
  const width = estimateLabelWidth(label, fontSize);
  return layout.textAnchor === "start"
    ? { x0: layout.labelX, x1: layout.labelX + width }
    : { x0: layout.labelX - width, x1: layout.labelX };
}

export function computePerfRulerLabelLayouts(
  geometries: (Pick<
    PerfRulerGeometry,
    "x" | "y1" | "y2" | "ratioLabel"
  > | null)[],
  opts?: PerfRulerLabelLayoutOptions,
): (PerfRulerLabelLayout | null)[] {
  const fontSize = opts?.fontSize ?? DEFAULT_LABEL_FONT_SIZE;
  const placed: { x0: number; x1: number; y: number }[] = [];
  return geometries.map((geometry) => {
    if (!geometry) return null;
    let layout = computePerfRulerLabelLayout(geometry, opts);
    const midY = (geometry.y1 + geometry.y2) / 2;
    const overlapsPlaced = (candidate: PerfRulerLabelLayout): boolean => {
      const span = labelSpan(candidate, geometry.ratioLabel, fontSize);
      return placed.some(
        (box) =>
          span.x0 < box.x1 &&
          box.x0 < span.x1 &&
          Math.abs(candidate.labelY - box.y) < fontSize + 8,
      );
    };
    for (
      let attempt = 0;
      attempt < LABEL_COLLISION_NUDGE_TRIES && overlapsPlaced(layout);
      attempt++
    ) {
      const direction = layout.labelY < midY ? -1 : 1;
      let nudgedY = layout.labelY + direction * (fontSize + 10);
      if (direction === -1 && nudgedY - fontSize / 2 < 4)
        nudgedY = midY + LABEL_OFFSET_Y;
      nudgedY = clampLabelY(nudgedY, fontSize, opts?.chartHeight);
      if (nudgedY === layout.labelY) break;
      layout = buildLabelLayout(geometry, layout.side, nudgedY, fontSize);
    }
    const span = labelSpan(layout, geometry.ratioLabel, fontSize);
    placed.push({ x0: span.x0, x1: span.x1, y: layout.labelY });
    return layout;
  });
}

export interface PerfRulerRenderEntry {
  id: number;
  geometry: PerfRulerGeometry;
  layout: PerfRulerLabelLayout;
}

const DELETE_HIDE_DELAY_MS = 250;
const hideTimers = new WeakMap<Element, ReturnType<typeof setTimeout>>();

function setPerfRulerHover(node: Element, hovered: boolean): void {
  const deleteButton = node.querySelector<SVGGElement>(".pr-delete");
  if (deleteButton) deleteButton.style.display = hovered ? "" : "none";
  const line = node.querySelector<SVGLineElement>(".pr-line");
  line?.setAttribute("stroke-width", hovered ? "3" : "2");
}

export function renderPerfRulers(
  group: GroupSelection,
  entries: PerfRulerRenderEntry[],
  opts: PerfRulerRenderOptions,
): void {
  const selection = group
    .selectAll<SVGGElement, PerfRulerRenderEntry>(".perf-ruler")
    .data(entries, (entry) => String(entry.id));

  selection.exit().remove();

  const entered = selection.enter().append("g").attr("class", "perf-ruler");
  entered
    .append("line")
    .attr("class", "pr-line")
    .attr("stroke-width", 2)
    .attr("stroke-dasharray", "5 4");
  entered
    .append("line")
    .attr("class", "pr-cap pr-cap-top")
    .attr("stroke-width", 2);
  entered
    .append("line")
    .attr("class", "pr-cap pr-cap-bottom")
    .attr("stroke-width", 2);
  entered
    .append("path")
    .attr("class", "pr-arrow")
    .attr("fill", "none")
    .attr("stroke-width", 2.5)
    .attr("stroke-linecap", "round");
  entered.append("path").attr("class", "pr-arrow-head").attr("stroke", "none");
  entered
    .append("text")
    .attr("class", "pr-text pr-text-ratio")
    .attr("dominant-baseline", "central")
    .attr("font-weight", "800")
    .attr("paint-order", "stroke")
    .attr("stroke-linejoin", "round")
    .style("pointer-events", "auto")
    .style("cursor", "default");
  entered
    .append("line")
    .attr("class", "pr-drag")
    .attr("stroke", "transparent")
    .attr("stroke-width", 16)
    .style("pointer-events", "stroke")
    .style("cursor", "ew-resize");
  const enteredDelete = entered
    .append("g")
    .attr("class", "pr-delete no-export")
    .style("display", "none")
    .style("pointer-events", "all")
    .style("cursor", "pointer");
  enteredDelete.append("circle").attr("class", "pr-delete-bg").attr("r", 10);
  enteredDelete
    .append("text")
    .attr("class", "pr-delete-x")
    .attr("text-anchor", "middle")
    .attr("dominant-baseline", "central")
    .attr("font-size", "12px")
    .attr("font-weight", "700")
    .text("×");

  const merged = entered.merge(selection).style("pointer-events", "none");

  const capHalfWidth = opts.capHalfWidth ?? 6;
  const fontSize = opts.labelFontSize ?? DEFAULT_LABEL_FONT_SIZE;

  merged
    .select<SVGLineElement>(".pr-line")
    .attr("x1", (entry) => entry.geometry.x)
    .attr("x2", (entry) => entry.geometry.x)
    .attr("y1", (entry) => entry.geometry.y1)
    .attr("y2", (entry) => entry.geometry.y2)
    .attr("stroke", opts.color);
  merged
    .select<SVGLineElement>(".pr-cap-top")
    .attr("x1", (entry) => entry.geometry.x - capHalfWidth)
    .attr("x2", (entry) => entry.geometry.x + capHalfWidth)
    .attr("y1", (entry) => entry.geometry.y1)
    .attr("y2", (entry) => entry.geometry.y1)
    .attr("stroke", opts.color);
  merged
    .select<SVGLineElement>(".pr-cap-bottom")
    .attr("x1", (entry) => entry.geometry.x - capHalfWidth)
    .attr("x2", (entry) => entry.geometry.x + capHalfWidth)
    .attr("y1", (entry) => entry.geometry.y2)
    .attr("y2", (entry) => entry.geometry.y2)
    .attr("stroke", opts.color);
  merged
    .select<SVGLineElement>(".pr-drag")
    .attr("x1", (entry) => entry.geometry.x)
    .attr("x2", (entry) => entry.geometry.x)
    .attr("y1", (entry) => entry.geometry.y1)
    .attr("y2", (entry) => entry.geometry.y2);

  merged
    .select<SVGPathElement>(".pr-arrow")
    .attr("d", (entry) => entry.layout.arrowPath)
    .attr("stroke", opts.color);
  merged
    .select<SVGPathElement>(".pr-arrow-head")
    .attr("d", (entry) => entry.layout.arrowHeadPath)
    .attr("fill", opts.color);
  merged
    .select<SVGTextElement>(".pr-text-ratio")
    .attr("x", (entry) => entry.layout.labelX)
    .attr("y", (entry) => entry.layout.labelY)
    .attr("text-anchor", (entry) => entry.layout.textAnchor)
    .attr("font-size", `${fontSize}px`)
    .attr("fill", opts.color)
    .attr("stroke", opts.halo ?? "var(--background)")
    .attr("stroke-width", 5)
    .text((entry) => entry.geometry.ratioLabel);

  merged
    .select<SVGGElement>(".pr-delete")
    .attr(
      "transform",
      (entry) => `translate(${entry.layout.deleteX}, ${entry.layout.deleteY})`,
    )
    .on("click", (event: MouseEvent, entry: PerfRulerRenderEntry) => {
      event.stopPropagation();
      opts.onDelete?.(entry.id);
    });
  merged.select<SVGCircleElement>(".pr-delete-bg").attr("fill", opts.color);
  merged
    .select<SVGTextElement>(".pr-delete-x")
    .attr("fill", opts.halo ?? "var(--background)");

  merged
    .on("mouseenter", (event: MouseEvent) => {
      const node = event.currentTarget as Element;
      const timer = hideTimers.get(node);
      if (timer !== undefined) {
        clearTimeout(timer);
        hideTimers.delete(node);
      }
      setPerfRulerHover(node, true);
    })
    .on("mouseleave", (event: MouseEvent) => {
      const node = event.currentTarget as Element;
      const timer = setTimeout(() => {
        hideTimers.delete(node);
        setPerfRulerHover(node, false);
      }, DELETE_HIDE_DELAY_MS);
      hideTimers.set(node, timer);
    });
}
