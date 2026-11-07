// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
import { useCallback, useEffect, useRef, useState } from "react";

export interface UseResponsiveChartDimensionsOptions {
  height?: number;
}

export interface UseResponsiveChartDimensionsResult {
  dimensions: { width: number; height: number };
  containerRef: React.RefObject<HTMLDivElement | null>;
  setContainerRef: (element: HTMLDivElement | null) => void;
}

export function useResponsiveChartDimensions(
  options: UseResponsiveChartDimensionsOptions = {},
): UseResponsiveChartDimensionsResult {
  const { height = 600 } = options;

  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height });
  const resizeObserverRef = useRef<ResizeObserver | null>(null);

  const updateDimensions = useCallback((width: number, h: number) => {
    setDimensions((prev) =>
      prev.width === width && prev.height === h ? prev : { width, height: h },
    );
  }, []);

  const setContainerRef = useCallback(
    (element: HTMLDivElement | null) => {
      if (resizeObserverRef.current) {
        resizeObserverRef.current.disconnect();
        resizeObserverRef.current = null;
      }

      containerRef.current = element;

      if (element) {
        const initialWidth = element.getBoundingClientRect().width;
        updateDimensions(initialWidth, height);

        resizeObserverRef.current = new ResizeObserver((entries) => {
          if (entries[0]) {
            const { width: observedWidth } = entries[0].contentRect;
            updateDimensions(observedWidth, height);
          }
        });

        resizeObserverRef.current.observe(element);
      }
    },
    [height, updateDimensions],
  );

  useEffect(
    () => () => {
      if (resizeObserverRef.current) {
        resizeObserverRef.current.disconnect();
      }
    },
    [],
  );

  useEffect(() => {
    setDimensions((prev) =>
      prev.height === height ? prev : { ...prev, height },
    );
  }, [height]);

  return {
    dimensions,
    containerRef,
    setContainerRef,
  };
}
