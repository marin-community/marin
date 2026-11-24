// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
import { useCallback, useEffect, useMemo, useRef } from "react";
import * as d3 from "d3";

export interface UseChartZoomOptions {
  resetEventName: string;

  defaultZoomK?: number;

  scaleExtent?: [number, number];

  onReset?: () => void;

  svgRef: React.RefObject<SVGSVGElement | null>;
}

export interface SetupZoomOptions {
  translateExtent?: [[number, number], [number, number]];

  extent?: [[number, number], [number, number]];

  constrain?: (
    transform: d3.ZoomTransform,
    extent: [[number, number], [number, number]],
    translateExtent: [[number, number], [number, number]],
  ) => d3.ZoomTransform;

  onZoom?: (event: d3.D3ZoomEvent<SVGSVGElement, unknown>) => void;

  customTransformStorage?: (transform: d3.ZoomTransform) => d3.ZoomTransform;
}

export interface UseChartZoomResult {
  zoomRef: React.MutableRefObject<d3.ZoomBehavior<
    SVGSVGElement,
    unknown
  > | null>;

  zoomTransformRef: React.MutableRefObject<d3.ZoomTransform>;

  defaultZoomTransform: d3.ZoomTransform;

  setupZoom: (
    svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
    width: number,
    height: number,
    options?: SetupZoomOptions,
  ) => d3.ZoomBehavior<SVGSVGElement, unknown>;
}

export function zoomEventFilter(event: {
  type: string;
  shiftKey?: boolean;
  ctrlKey?: boolean;
  button?: number;
  touches?: ArrayLike<unknown>;
}): boolean {
  if (event.type.startsWith("touch")) return (event.touches?.length ?? 0) >= 2;
  if (event.type === "wheel") return Boolean(event.shiftKey) && !event.ctrlKey;
  return !event.ctrlKey && !event.button;
}

export function useChartZoom(options: UseChartZoomOptions): UseChartZoomResult {
  const {
    resetEventName,
    defaultZoomK = 1,
    scaleExtent = [1, 100],
    onReset,
    svgRef,
  } = options;

  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const defaultZoomTransform = useMemo(
    () => d3.zoomIdentity.scale(defaultZoomK),
    [defaultZoomK],
  );
  const zoomTransformRef = useRef<d3.ZoomTransform>(defaultZoomTransform);
  const scaleExtentRef = useRef(scaleExtent);
  scaleExtentRef.current = scaleExtent;

  useEffect(() => {
    const handleResetZoom = () => {
      if (svgRef.current && zoomRef.current) {
        zoomTransformRef.current = defaultZoomTransform;

        const transition = d3
          .select(svgRef.current)
          .transition("zoom")
          .duration(750)
          .call(zoomRef.current.transform as any, defaultZoomTransform);

        if (onReset) {
          transition.on("end", () => {
            onReset();
          });
        }
      }
    };

    window.addEventListener(resetEventName, handleResetZoom);
    return () => window.removeEventListener(resetEventName, handleResetZoom);
  }, [resetEventName, defaultZoomTransform, onReset, svgRef]);

  const setupZoom = useCallback(
    (
      svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
      width: number,
      height: number,
      setupOptions?: SetupZoomOptions,
    ): d3.ZoomBehavior<SVGSVGElement, unknown> => {
      const {
        translateExtent = [
          [0, -Infinity],
          [width, Infinity],
        ],
        extent = [
          [0, 0],
          [width, height],
        ],
        constrain,
        onZoom,
        customTransformStorage,
      } = setupOptions || {};

      const zoom = d3
        .zoom<SVGSVGElement, unknown>()
        .filter(zoomEventFilter)
        .wheelDelta((event: WheelEvent) => {
          const delta = event.deltaY || event.deltaX;
          return (
            -delta *
            (event.deltaMode === 1 ? 0.05 : event.deltaMode ? 1 : 0.002)
          );
        })
        .scaleExtent(scaleExtentRef.current)
        .extent(extent)
        .translateExtent(translateExtent);

      if (constrain) {
        zoom.constrain(constrain);
      }

      if (onZoom) {
        zoom.on("zoom", onZoom);
      }

      zoom.on("zoom.store", (event) => {
        zoomTransformRef.current = customTransformStorage
          ? customTransformStorage(event.transform)
          : event.transform;
      });

      svg.call(zoom as any);

      zoomRef.current = zoom;

      const stored = zoomTransformRef.current;
      const nodeTransform = d3.zoomTransform(svg.node()!);
      const isIdentity = (t: d3.ZoomTransform) =>
        t.k === 1 && t.x === 0 && t.y === 0;
      if (!isIdentity(stored) || !isIdentity(nodeTransform)) {
        svg.call(zoom.transform as any, stored);
      }

      svg.on("dblclick.zoom", () => {
        svg
          .transition("zoom")
          .duration(750)
          .call(zoom.transform as any, defaultZoomTransform);
        zoomTransformRef.current = defaultZoomTransform;
      });

      return zoom;
    },
    [defaultZoomTransform],
  );

  return {
    zoomRef,
    zoomTransformRef,
    defaultZoomTransform,
    setupZoom,
  };
}
