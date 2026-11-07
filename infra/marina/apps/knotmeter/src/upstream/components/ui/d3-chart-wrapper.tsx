// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.

import React, { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import {
  useIsCoarsePointer,
  useIsMobileViewport,
  useMediaQuery,
} from "@/hooks/useMediaQuery";
import type { Locale } from "@/lib/i18n/i18n";
import { useLocale } from "@/lib/i18n/use-locale";

const DEFAULT_CHART_INSTRUCTIONS = {
  en: "Shift+Scroll to zoom • Drag to pan • Double-click to reset • Click a point to pin tooltip",
  zh: "按住 Shift 滚动以缩放 · 拖动以平移 · 双击以重置 · 点击数据点固定提示框",
} as const;

const SHEET_STRINGS = {
  en: { title: "Point details", close: "Close point details" },
  zh: { title: "数据点详情", close: "关闭数据点详情" },
} as const;

const SHEET_SWIPE_DISMISS_PX = 64;

const TOUCH_CHART_INSTRUCTIONS = {
  en: "Pinch to zoom • Two-finger drag to pan • Double-tap to reset • Tap a point to pin tooltip",
  zh: "双指捏合以缩放 · 双指拖动以平移 · 双击以重置 · 点按数据点固定提示框",
} as const;

export const TOUCH_PRIMARY_QUERY = "(hover: none) and (pointer: coarse)";

export function resolveChartInstructions(
  instructions: string | undefined,
  locale: Locale,
  useTouchHint: boolean,
): string {
  const resolved = instructions ?? DEFAULT_CHART_INSTRUCTIONS[locale];
  return useTouchHint && resolved ? TOUCH_CHART_INSTRUCTIONS[locale] : resolved;
}

function PortalTooltip({
  chartId,
  tooltipRef,
  pinned,
  onSheetDismiss,
}: {
  chartId: string;
  tooltipRef: React.RefObject<HTMLDivElement | null>;
  pinned: boolean;
  onSheetDismiss: () => void;
}) {
  const locale = useLocale();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  const isMobile = useIsMobileViewport();
  const isTouch = useIsCoarsePointer();
  const sheet = pinned && isMobile && isTouch;

  const onSheetDismissRef = useRef(onSheetDismiss);
  onSheetDismissRef.current = onSheetDismiss;

  useEffect(() => {
    const el = tooltipRef.current;
    if (!sheet || !el) return;
    let startY: number | null = null;
    const onStart = (event: TouchEvent) => {
      startY =
        el.scrollTop <= 0 && event.touches.length === 1
          ? event.touches[0]!.clientY
          : null;
    };
    const onEnd = (event: TouchEvent) => {
      if (startY === null) return;
      const originY = startY;
      startY = null;
      const endY = event.changedTouches[0]?.clientY ?? originY;
      if (endY - originY > SHEET_SWIPE_DISMISS_PX) onSheetDismissRef.current();
    };
    el.addEventListener("touchstart", onStart, { passive: true });
    el.addEventListener("touchend", onEnd, { passive: true });
    return () => {
      el.removeEventListener("touchstart", onStart);
      el.removeEventListener("touchend", onEnd);
    };
  }, [sheet, tooltipRef]);

  useEffect(() => {
    if (!sheet) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onSheetDismissRef.current();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [sheet]);

  const node = (
    <div
      ref={tooltipRef}
      data-chart-tooltip={chartId}
      data-sheet={sheet ? "true" : undefined}
      role={sheet ? "dialog" : undefined}
      aria-modal={sheet ? false : undefined}
      aria-label={sheet ? SHEET_STRINGS[locale].title : undefined}
      style={{
        position: "fixed",
        left: 0,
        top: 0,
        opacity: pinned ? 1 : 0,
        pointerEvents: pinned ? "auto" : "none",
        display: pinned ? "block" : "none",
        zIndex: 9999,
      }}
    />
  );
  if (!mounted || typeof document === "undefined") return node;
  return createPortal(
    <>
      {sheet && (
        <button
          type="button"
          data-testid="chart-sheet-backdrop"
          aria-label={SHEET_STRINGS[locale].close}
          onClick={(event) => {
            event.stopPropagation();
            onSheetDismissRef.current();
          }}
          className="fixed inset-0 z-[9998] cursor-default bg-black/45 backdrop-blur-[1px] animate-in fade-in-0 duration-200 md:hidden"
        />
      )}
      {node}
    </>,
    document.body,
  );
}

export interface D3ChartWrapperProps {
  chartId: string;
  svgRef: React.RefObject<SVGSVGElement | null>;
  tooltipRef: React.RefObject<HTMLDivElement | null>;
  setContainerRef: (el: HTMLDivElement | null) => void;
  dimensions: { width: number; height: number };
  pinnedPoint: unknown | null;
  isPinned: () => boolean;
  dismissTooltip: () => void;
  hideTooltipElements: (
    tooltipRef: React.RefObject<HTMLDivElement | null>,
    svgRef: React.RefObject<SVGSVGElement | null>,
  ) => void;
  legendElement: React.ReactNode;
  noDataOverlay?: React.ReactNode;
  caption?: React.ReactNode;
  instructions?: string;
  zoomEnabled?: boolean;
  testId?: string;
  grabCursor?: boolean;
  scrollablePlot?: { minWidth: number; label: string; enabled: boolean };
}

export function D3ChartWrapper({
  chartId,
  svgRef,
  tooltipRef,
  setContainerRef,
  dimensions,
  pinnedPoint,
  isPinned,
  dismissTooltip,
  hideTooltipElements,
  legendElement,
  noDataOverlay,
  caption,
  instructions,
  zoomEnabled = false,
  testId,
  grabCursor = true,
  scrollablePlot,
}: D3ChartWrapperProps) {
  const locale = useLocale();
  const isTouchPrimary = useMediaQuery(TOUCH_PRIMARY_QUERY);
  const resolvedInstructions = resolveChartInstructions(
    instructions,
    locale,
    zoomEnabled && isTouchPrimary,
  );

  const plot = (
    <div
      className="flex flex-col lg:flex-row w-full"
      style={scrollablePlot ? { minWidth: scrollablePlot.minWidth } : undefined}
    >
      <div ref={setContainerRef} className="relative flex-1 min-w-0">
        <div className="relative">
          <svg
            ref={svgRef}
            data-testid="d3-chart-svg"
            width="100%"
            height={dimensions.height}
            style={{
              cursor: grabCursor ? "grab" : undefined,
              touchAction: "pan-y",
            }}
            onMouseDown={
              grabCursor
                ? (e) => {
                    (e.currentTarget as SVGSVGElement).style.cursor =
                      "grabbing";
                  }
                : undefined
            }
            onMouseUp={
              grabCursor
                ? (e) => {
                    (e.currentTarget as SVGSVGElement).style.cursor = "grab";
                  }
                : undefined
            }
            onClick={() => {
              if (isPinned()) {
                dismissTooltip();
                hideTooltipElements(tooltipRef, svgRef);
              }
            }}
          />
          <PortalTooltip
            chartId={chartId}
            tooltipRef={tooltipRef}
            pinned={Boolean(pinnedPoint)}
            onSheetDismiss={() => {
              dismissTooltip();
              hideTooltipElements(tooltipRef, svgRef);
            }}
          />
          {noDataOverlay}
        </div>
        {resolvedInstructions && (
          <p className="no-export text-xs text-muted-foreground text-center mt-2">
            {resolvedInstructions}
          </p>
        )}
        <div className="overflow-hidden max-h-0">
          <div id={`${chartId}-export`} className="p-4"></div>
        </div>
      </div>
      {legendElement && (
        <div
          data-slot="chart-legend-wrapper"
          className="w-full lg:w-auto lg:shrink-0 lg:self-start relative mt-3 lg:mt-0 lg:has-[.sidebar-legend]:w-fit lg:has-[.sidebar-legend]:min-w-48 lg:has-[.sidebar-legend]:max-w-96"
        >
          {legendElement}
        </div>
      )}
    </div>
  );

  return (
    <div id={chartId} data-testid={testId}>
      {caption && <figcaption>{caption}</figcaption>}
      {scrollablePlot ? (
        <>
          {scrollablePlot.enabled && (
            <p className="no-export text-xs text-muted-foreground mb-2">
              {scrollablePlot.label}
            </p>
          )}
          <div
            data-chart-scroll
            className={
              scrollablePlot.enabled
                ? "overflow-x-auto rounded-sm focus-visible:outline-2 focus-visible:outline-ring"
                : undefined
            }
            tabIndex={scrollablePlot.enabled ? 0 : undefined}
            role={scrollablePlot.enabled ? "region" : undefined}
            aria-label={
              scrollablePlot.enabled ? scrollablePlot.label : undefined
            }
            onKeyDown={(event) => {
              if (!scrollablePlot.enabled) return;
              if (event.target !== event.currentTarget) return;
              if (event.key !== "ArrowLeft" && event.key !== "ArrowRight")
                return;
              event.preventDefault();
              event.currentTarget.scrollBy({
                left:
                  (event.key === "ArrowRight" ? 1 : -1) *
                  event.currentTarget.clientWidth *
                  0.8,
              });
            }}
          >
            {plot}
          </div>
        </>
      ) : (
        plot
      )}
    </div>
  );
}
