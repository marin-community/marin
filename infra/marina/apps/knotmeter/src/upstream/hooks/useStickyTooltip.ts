// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
import * as d3 from "d3";
import { useCallback, useRef, useState } from "react";

import { type ShapeKey, applyNormalState } from "@/lib/charts/chart-rendering";

const VALID_SHAPE_KEYS: ReadonlySet<ShapeKey> = new Set([
  "circle",
  "square",
  "triangle",
  "diamond",
]);

const readShapeKey = (el: SVGElement): ShapeKey => {
  const k = el.dataset.shapeKey;
  return k && VALID_SHAPE_KEYS.has(k as ShapeKey) ? (k as ShapeKey) : "circle";
};

export const useStickyTooltip = <T = any>() => {
  const [pinnedPoint, setPinnedPoint] = useState<T | null>(null);
  const [pinnedPointIsOverlay, setPinnedPointIsOverlay] = useState(false);

  const pinnedPointRef = useRef<T | null>(null);

  const pinTooltip = useCallback((point: T, isOverlay = false) => {
    pinnedPointRef.current = point;
    setPinnedPoint(point);
    setPinnedPointIsOverlay(isOverlay);
  }, []);

  const dismissTooltip = useCallback((deferred = false) => {
    pinnedPointRef.current = null;

    if (deferred) {
      requestAnimationFrame(() => {
        setPinnedPoint(null);
        setPinnedPointIsOverlay(false);
      });
    } else {
      setPinnedPoint(null);
      setPinnedPointIsOverlay(false);
    }
  }, []);

  const isPinned = useCallback(() => pinnedPointRef.current !== null, []);

  const hideTooltipElements = useCallback(
    (
      tooltipRef: React.RefObject<HTMLDivElement | null>,
      svgRef: React.RefObject<SVGSVGElement | null>,
    ) => {
      if (tooltipRef.current) {
        tooltipRef.current.style.opacity = "0";
        tooltipRef.current.style.display = "none";
        tooltipRef.current.style.pointerEvents = "none";
      }

      if (svgRef.current) {
        const svg = d3.select(svgRef.current);
        svg.select(".ruler-group").style("display", "none");
        svg.select(".cursor-line").style("opacity", "0");

        svg.selectAll(".dot-group").each(function () {
          const group = d3.select(this);
          const shape = group.select<SVGElement>(".visible-shape");
          const node = shape.node();
          if (node) {
            applyNormalState(shape as any, readShapeKey(node));
          }
        });
      }
    },
    [],
  );

  return {
    pinnedPoint,
    pinnedPointIsOverlay,

    pinnedPointRef,

    pinTooltip,
    dismissTooltip,
    isPinned,
    hideTooltipElements,
  };
};
