// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce ScatterGraph.tsx's ruler interaction.
import { useLayoutEffect, useRef, useState } from "react";
import * as d3 from "d3";
import {
  clampIsoX,
  computeIsoXRulerGeometry,
  computePerfRulerLabelLayouts,
  deletePerfRuler,
  intersectPathAtX,
  movePerfRulerIsoX,
  nextPerfRulerState,
  parsePerfRulers,
  pathXExtent,
  perfRulerCurveSet,
  renderPerfRulers,
  serializePerfRulers,
  type PerfRulerState,
  type PerfRulerRenderEntry,
} from "@/lib/d3-chart/layers/perf-ruler";
import type {
  CustomLayerConfig,
  RenderContext,
  ZoomContext,
} from "@/lib/d3-chart/D3Chart/types";
import type { ContinuousScale } from "@/lib/d3-chart/types";
import { setParameters } from "./navigation";

type Group = d3.Selection<SVGGElement, unknown, null, undefined>;
interface Context {
  group: Group;
  x: ContinuousScale;
  y: ContinuousScale;
  width: number;
  height: number;
}

export function usePerfRulers(
  enabled: boolean,
  raw: string | null,
  scopes: Record<string, string>,
) {
  const [state, setState] = useState<PerfRulerState>(() =>
    parsePerfRulers(raw),
  );
  const [message, setMessage] = useState("");
  const stateRef = useRef(state);
  stateRef.current = state;
  const context = useRef<Context | null>(null);
  const enabledRef = useRef(enabled);
  enabledRef.current = enabled;
  const scopesRef = useRef(scopes);
  scopesRef.current = scopes;
  const drawRef = useRef<() => void>(() => {});
  const lastUrl = useRef(raw);
  useLayoutEffect(() => {
    if (lastUrl.current !== raw) {
      lastUrl.current = raw;
      setState(parsePerfRulers(raw));
    }
  }, [raw]);
  function commit(next: PerfRulerState) {
    stateRef.current = next;
    setState(next);
    const serialized = serializePerfRulers(next);
    lastUrl.current = serialized || null;
    setParameters({ rulers: serialized });
  }
  function clampToOverlap(
    curveA: string,
    curveB: string,
    isoX: number,
  ): number | null {
    const ctx = context.current;
    if (!ctx) return null;
    if (scopesRef.current[curveA] !== scopesRef.current[curveB]) return null;
    const a = ctx.group.select<SVGPathElement>(`.${CSS.escape(curveA)}`).node();
    const b = ctx.group.select<SVGPathElement>(`.${CSS.escape(curveB)}`).node();
    if (!a || !b) return null;
    const extentA = pathXExtent(a);
    const extentB = pathXExtent(b);
    if (!extentA || !extentB) return null;
    const value = clampIsoX(ctx.x(isoX), extentA, extentB);
    return value === null ? null : ctx.x.invert(value);
  }
  function selectCurve(curve: string, isoX: number) {
    const previous = stateRef.current;
    const next = nextPerfRulerState(previous, { curve, isoX }, clampToOverlap);
    setMessage(
      next === previous
        ? "Choose curves for the same model revision, prepared workload and precision with an overlapping measured range."
        : next.draft
          ? "Select a second comparable curve or point."
          : "Drag the ruler to compare another operating point.",
    );
    commit(next);
  }
  function remove(id: number) {
    commit(deletePerfRuler(stateRef.current, id));
  }
  function draw() {
    const ctx = context.current;
    if (!ctx) return;
    const { group, x, y, width, height } = ctx;
    const mode = enabledRef.current;
    const paths: { curve: string; path: string }[] = [];
    group.selectAll<SVGPathElement, unknown>(".line-path").each(function () {
      const key = [...this.classList]
        .find((c) => c.startsWith("line-") && c !== "line-path")
        ?.slice(5);
      if (!key) return;
      const curve = `roofline-${key}`;
      this.classList.add(curve);
      const path = this.getAttribute("d");
      if (path && this.getTotalLength() > 0) paths.push({ curve, path });
    });
    const selected = perfRulerCurveSet(stateRef.current);
    group
      .selectAll<SVGPathElement, { curve: string; path: string }>(
        ".perf-ruler-hit",
      )
      .data(mode ? paths : [], (d) => d.curve)
      .join("path")
      .attr("class", "perf-ruler-hit")
      .attr("d", (d) => d.path)
      .attr("fill", "none")
      .attr("stroke", "var(--mark)")
      .attr("stroke-opacity", (d) => (selected.has(d.curve) ? 0.2 : 0))
      .attr("stroke-width", 14)
      .style("pointer-events", "stroke")
      .style("cursor", "crosshair")
      .on("click", (event: MouseEvent, d) => {
        event.stopPropagation();
        const [pixelX] = d3.pointer(event, group.node());
        selectCurve(d.curve, x.invert(pixelX));
      });
    const rulers = mode ? stateRef.current.rulers : [];
    const geometries = rulers.map((ruler) => {
      if (scopesRef.current[ruler.curveA] !== scopesRef.current[ruler.curveB])
        return null;
      const pixelX = x(ruler.isoX);
      const ends = [ruler.curveA, ruler.curveB].map((curve) => {
        const node = group
          .select<SVGPathElement>(`.${CSS.escape(curve)}`)
          .node();
        const hit = node ? intersectPathAtX(node, pixelX) : null;
        return hit ? { py: hit.y, rawY: y.invert(hit.y) } : null;
      });
      return ends[0] && ends[1]
        ? computeIsoXRulerGeometry(pixelX, ends[0], ends[1])
        : null;
    });
    const layouts = computePerfRulerLabelLayouts(geometries, {
      chartWidth: width,
      chartHeight: height,
    });
    const entries: PerfRulerRenderEntry[] = rulers.flatMap((ruler, i) =>
      geometries[i] && layouts[i]
        ? [{ id: ruler.id, geometry: geometries[i]!, layout: layouts[i]! }]
        : [],
    );
    renderPerfRulers(group, entries, {
      color: "var(--mark)",
      halo: "var(--paper)",
      onDelete: remove,
    });
    group
      .selectAll<SVGLineElement, PerfRulerRenderEntry>(".perf-ruler .pr-drag")
      .call(
        d3
          .drag<SVGLineElement, PerfRulerRenderEntry>()
          .on("drag", (event, entry) => {
            const ruler = stateRef.current.rulers.find(
              (r) => r.id === entry.id,
            );
            if (!ruler) return;
            const value = clampToOverlap(
              ruler.curveA,
              ruler.curveB,
              x.invert(event.x),
            );
            if (value === null) return;
            stateRef.current = movePerfRulerIsoX(
              stateRef.current,
              entry.id,
              value,
            );
            drawRef.current();
          })
          .on("end", () => commit(stateRef.current)),
      );
  }
  drawRef.current = draw;
  useLayoutEffect(() => {
    drawRef.current();
  }, [state, enabled]);
  function render(group: Group, ctx: RenderContext) {
    context.current = {
      group,
      x: ctx.xScale as ContinuousScale,
      y: ctx.yScale as ContinuousScale,
      width: ctx.width,
      height: ctx.height,
    };
    drawRef.current();
  }
  const layer: CustomLayerConfig = {
    type: "custom",
    key: "perf-rulers",
    render,
    onDisplayUpdate: render,
    onZoom(group: Group, ctx: ZoomContext) {
      context.current = {
        group,
        x: ctx.newXScale as ContinuousScale,
        y: ctx.newYScale as ContinuousScale,
        width: ctx.width,
        height: ctx.height,
      };
      drawRef.current();
    },
  };
  return {
    state,
    message,
    layer,
    raise() {
      context.current?.group.selectAll(".perf-ruler").raise();
    },
    selectPoint(series: string, x: number) {
      selectCurve(`roofline-${series}`, x);
    },
    remove,
    clear() {
      commit({ rulers: [], draft: null, nextId: 1 });
      setMessage("");
    },
  };
}
