// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.

import { useCallback, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "@/navigation";

import {
  type RequestRecord,
  type RequestTimeline,
} from "@/hooks/api/use-request-timeline";
import {
  SegmentedToggle,
  type SegmentedToggleOption,
} from "@/components/ui/segmented-toggle";

import { useLocale } from "@/lib/i18n/use-locale";
import type { Locale } from "@/lib/i18n/i18n";

import { sliceTimelineByPhase } from "./phase-slice";
import { TimelineBars } from "./timeline-bars";
import { formatDuration } from "./timeline-format";
import {
  CHART_WIDTH,
  HEADER_HEIGHT,
  LABEL_WIDTH,
  PADDING_RIGHT,
  ROW_GAP,
  ROW_HEIGHT,
  ROW_SPAN,
  TIMELINE_BODY_MAX_HEIGHT,
  timelineSvgHeight,
  visibleTimelineRowRange,
} from "./timeline-layout";
import {
  buildRequestTimelineRows,
  computeStableRowIndex,
  conversationHref,
  requestIdleStats,
  type RequestTimelineRow,
  type RowMode,
} from "./timeline-rows";
import type { SortedRequestTimes } from "./timeline-cursor-stats";
import {
  consumeTimelineViewSnapshot,
  saveTimelineViewSnapshot,
  type PhaseFilter,
} from "./timeline-view-snapshot";
import {
  CursorPopover,
  TimelineTooltip,
  type CursorState,
  type TooltipData,
} from "./timeline-tooltips";

export {
  buildRequestTimelineRows,
  computeStableRowIndex,
  conversationReplayKey,
  conversationHref,
  datasetConvId,
  requestIdleStats,
  splitTimelineCid,
  subagentIdOf,
} from "./timeline-rows";
export type { RequestIdleStats, RequestTimelineRow } from "./timeline-rows";
export { parseTimelineViewSnapshot } from "./timeline-view-snapshot";
export type { TimelineViewSnapshot } from "./timeline-view-snapshot";

export function shouldHandleTimelineBarClick({
  altKey = false,
  button,
  ctrlKey = false,
  metaKey = false,
  shiftKey = false,
}: {
  altKey?: boolean;
  button: number;
  ctrlKey?: boolean;
  metaKey?: boolean;
  shiftKey?: boolean;
}): boolean {
  return button === 0 && !altKey && !ctrlKey && !metaKey && !shiftKey;
}

const ROW_MODE_OPTIONS: Record<"en" | "zh", SegmentedToggleOption<RowMode>[]> =
  {
    en: [
      {
        value: "conversation",
        label: "By conversation",
        testId: "timeline-mode-conversation",
      },
      { value: "worker", label: "By worker", testId: "timeline-mode-worker" },
    ],
    zh: [
      {
        value: "conversation",
        label: "按对话",
        testId: "timeline-mode-conversation",
      },
      { value: "worker", label: "按 worker", testId: "timeline-mode-worker" },
    ],
  };

export function timelinePhaseLabels(locale: Locale): {
  profiling: string;
  warmup: string;
} {
  return locale === "zh"
    ? { profiling: "profiling", warmup: "warmup" }
    : { profiling: "Profiling", warmup: "Warmup" };
}

const phaseOptions = (locale: Locale): SegmentedToggleOption<PhaseFilter>[] => {
  const labels = timelinePhaseLabels(locale);
  return [
    {
      value: "profiling",
      label: labels.profiling,
      testId: "timeline-phase-profiling",
    },
    { value: "warmup", label: labels.warmup, testId: "timeline-phase-warmup" },
  ];
};

const TIMELINE_STRINGS = {
  en: {
    empty: "No requests in the current filter.",
    rowMode: "Row mode",
    phaseFilter: "Phase filter",
    requests: (n: number) => `${n} request${n === 1 ? "" : "s"}`,
    conversations: (n: number) => `${n} conversations`,
    workers: (n: number) => `${n} workers`,
    span: "span",
    idle: "idle",
    idleTitle:
      "Time between the first request start and final request end with no requests in flight",
    resetZoom: "reset zoom",
    conversation: "Conversation",
    worker: "worker",
    collapse: "Collapse streams",
    expand: "Expand streams",
    aux: "aux",
    chartAria: "Request execution timeline",
    hint: "shift+scroll to zoom · drag to pan · double-click to reset",
  },
  zh: {
    empty: "当前筛选条件下没有请求。",
    rowMode: "时间线行模式",
    phaseFilter: "阶段筛选",
    requests: (n: number) => `${n} 个请求`,
    conversations: (n: number) => `${n} 个对话`,
    workers: (n: number) => `${n} 个 worker`,
    span: "跨度",
    idle: "空闲",
    idleTitle: "从第一个请求开始到最后一个请求结束之间，没有在途请求的总时长",
    resetZoom: "重置缩放",
    conversation: "对话",
    worker: "worker",
    collapse: "折叠请求流",
    expand: "展开请求流",
    aux: "辅助流",
    chartAria: "请求执行时间线",
    hint: "Shift+滚轮缩放 · 拖动平移 · 双击重置",
  },
} as const;

const PLOT_WIDTH = CHART_WIDTH - PADDING_RIGHT;

export function RequestTimelineView({
  data,
  datasetSlug,
  pointId,
}: {
  data: RequestTimeline;
  datasetSlug?: string | null;
  pointId: string;
}) {
  const router = useRouter();
  const locale = useLocale();
  const t = TIMELINE_STRINGS[locale];
  const [rowMode, setRowMode] = useState<RowMode>("conversation");
  const [phaseFilter, setPhaseFilter] = useState<PhaseFilter>("profiling");
  const [tooltip, setTooltip] = useState<TooltipData | null>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewportHeight, setViewportHeight] = useState(
    TIMELINE_BODY_MAX_HEIGHT,
  );

  const scrollRef = useRef<HTMLDivElement>(null);
  const liveStateRef = useRef<{
    viewStart: number;
    viewEnd: number | null;
    rowMode: RowMode;
    phaseFilter: PhaseFilter;
    expandedSubagents: ReadonlySet<string>;
  }>({
    viewStart: 0,
    viewEnd: null,
    rowMode: "conversation",
    phaseFilter: "profiling",
    expandedSubagents: new Set(),
  });

  const openConversation = useCallback(
    (req: RequestRecord) => {
      if (!datasetSlug) return;
      if (scrollRef.current) {
        const live = liveStateRef.current;
        saveTimelineViewSnapshot(pointId, {
          viewStart: live.viewStart,
          viewEnd: live.viewEnd,
          rowMode: live.rowMode,
          phaseFilter: live.phaseFilter,
          expanded: [...live.expandedSubagents],
          scrollTop: scrollRef.current.scrollTop,
          scrollLeft: scrollRef.current.scrollLeft,
        });
      }

      router.push(conversationHref(datasetSlug, req, locale));
    },
    [datasetSlug, router, pointId, locale],
  );
  const [expandedSubagents, setExpandedSubagents] = useState<
    ReadonlySet<string>
  >(() => new Set());
  const toggleSubagent = useCallback((key: string) => {
    setExpandedSubagents((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);
  const dragRef = useRef<{ startX: number; vs: number; ve: number } | null>(
    null,
  );

  const didPan = useRef(false);

  const hasWarmup = useMemo(
    () => data.requests.some((r) => r.phase !== "profiling"),
    [data.requests],
  );

  const filtered = useMemo(
    () =>
      sliceTimelineByPhase(data, hasWarmup ? phaseFilter : "profiling")
        .requests,
    [data, phaseFilter, hasWarmup],
  );
  const stableRowIndex = useMemo(
    () => computeStableRowIndex(data.requests, rowMode),
    [data.requests, rowMode],
  );
  const rows = useMemo(
    () =>
      buildRequestTimelineRows(
        filtered,
        rowMode,
        expandedSubagents,
        stableRowIndex,
        locale,
      ),
    [filtered, rowMode, expandedSubagents, stableRowIndex, locale],
  );
  const idleStats = useMemo(() => requestIdleStats(filtered), [filtered]);

  const sortedTimes = useMemo<SortedRequestTimes>(() => {
    const credits = filtered.map((r) => r.credit).toSorted((a, b) => a - b);
    const starts = filtered.map((r) => r.start).toSorted((a, b) => a - b);
    const ends = filtered.map((r) => r.end).toSorted((a, b) => a - b);
    return { credits, starts, ends };
  }, [filtered]);

  const [cursor, setCursor] = useState<CursorState | null>(null);

  const { dataStart, dataEnd } = useMemo(() => {
    if (filtered.length === 0) return { dataStart: 0, dataEnd: 1 };
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    for (const r of filtered) {
      if (r.credit < min) min = r.credit;
      if (r.end > max) max = r.end;
    }
    return { dataStart: min, dataEnd: max };
  }, [filtered]);
  const totalNs = Math.max(dataEnd - dataStart, 1);

  const [viewStart, setViewStart] = useState(0);
  const [viewEnd, setViewEnd] = useState<number | null>(null);
  const vStart = viewStart;
  const vEnd = viewEnd ?? totalNs;
  const visibleDur = Math.max(vEnd - vStart, 1);
  const isZoomed = viewEnd !== null;

  liveStateRef.current = {
    viewStart,
    viewEnd,
    rowMode,
    phaseFilter,
    expandedSubagents,
  };

  useLayoutEffect(() => {
    const snapshot = consumeTimelineViewSnapshot(pointId);
    if (!snapshot) return;
    setRowMode(snapshot.rowMode);
    setPhaseFilter(snapshot.phaseFilter);
    setExpandedSubagents(new Set(snapshot.expanded));
    setViewStart(snapshot.viewStart);
    setViewEnd(snapshot.viewEnd);
    const target = { top: snapshot.scrollTop, left: snapshot.scrollLeft };
    requestAnimationFrame(() => {
      const el = scrollRef.current;
      if (!el) return;
      el.scrollTop = target.top;
      el.scrollLeft = target.left;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pointId]);

  const svgHeight = timelineSvgHeight(rows.length);
  const visibleRange = useMemo(
    () => visibleTimelineRowRange(rows.length, scrollTop, viewportHeight),
    [rows.length, scrollTop, viewportHeight],
  );
  const visibleRows = rows.slice(visibleRange.start, visibleRange.end);

  useLayoutEffect(() => {
    const element = scrollRef.current;
    if (!element) return;
    const updateHeight = () => setViewportHeight(element.clientHeight);
    updateHeight();
    const observer = new ResizeObserver(updateHeight);
    observer.observe(element);
    return () => observer.disconnect();
  }, [rows.length]);

  const zoomSvgRef = useRef<SVGSVGElement | null>(null);
  const handleWheel = useCallback(
    (e: WheelEvent) => {
      if (!e.shiftKey) return;
      e.preventDefault();
      const rect = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseRatio = Math.max(0, Math.min(1, mouseX / PLOT_WIDTH));
      const curStart = vStart;
      const curEnd = vEnd;
      const curDur = curEnd - curStart;
      const delta = e.deltaY || e.deltaX;
      const factor = delta > 0 ? 1.2 : 1 / 1.2;
      const newDur = Math.min(
        Math.max(curDur * factor, totalNs * 0.001),
        totalNs,
      );
      const pivot = curStart + mouseRatio * curDur;
      let newStart = pivot - mouseRatio * newDur;
      let newEnd = pivot + (1 - mouseRatio) * newDur;
      if (newStart < 0) {
        newEnd -= newStart;
        newStart = 0;
      }
      if (newEnd > totalNs) {
        newStart -= newEnd - totalNs;
        newEnd = totalNs;
        if (newStart < 0) newStart = 0;
      }
      if (newEnd - newStart >= totalNs * 0.99) {
        setViewStart(0);
        setViewEnd(null);
      } else {
        setViewStart(newStart);
        setViewEnd(newEnd);
      }
    },
    [vStart, vEnd, totalNs],
  );

  useLayoutEffect(() => {
    const svg = zoomSvgRef.current;
    if (!svg) return;
    svg.addEventListener("wheel", handleWheel, { passive: false });
    return () => svg.removeEventListener("wheel", handleWheel);
  }, [handleWheel]);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (e.button !== 0) return;
      didPan.current = false;
      dragRef.current = { startX: e.clientX, vs: vStart, ve: vEnd };
    },
    [vStart, vEnd],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (dragRef.current) {
        const dx = e.clientX - dragRef.current.startX;
        if (Math.abs(dx) > 4) didPan.current = true;
        const nsPerPx = visibleDur / PLOT_WIDTH;
        const delta = -dx * nsPerPx;
        let ns = dragRef.current.vs + delta;
        let ne = dragRef.current.ve + delta;
        const dur = ne - ns;
        if (ns < 0) {
          ns = 0;
          ne = dur;
        }
        if (ne > totalNs) {
          ne = totalNs;
          ns = totalNs - dur;
          if (ns < 0) ns = 0;
        }
        setViewStart(ns);
        setViewEnd(ne);
        setTooltip(null);
        setCursor(null);
        return;
      }
      const rect = e.currentTarget.getBoundingClientRect();
      const xPx = Math.max(0, Math.min(PLOT_WIDTH, e.clientX - rect.left));
      const nsPerPx = visibleDur / PLOT_WIDTH;
      const tNs = vStart + xPx * nsPerPx;
      setCursor({ xPx, tNs, clientX: e.clientX, clientY: e.clientY });
    },
    [visibleDur, totalNs, vStart],
  );

  const handleMouseUp = useCallback(() => {
    dragRef.current = null;
  }, []);

  const handleMouseLeave = useCallback(() => {
    dragRef.current = null;
    setCursor(null);
  }, []);

  const resetZoom = useCallback(() => {
    setViewStart(0);
    setViewEnd(null);
  }, []);

  const handleBarHover = useCallback(
    (e: React.MouseEvent, row: RequestTimelineRow, req: RequestRecord) => {
      setTooltip({ x: e.clientX, y: e.clientY, row, req });
    },
    [],
  );
  const handleBarLeave = useCallback(() => setTooltip(null), []);
  const handleBarClick = useCallback(
    (e: React.MouseEvent, req: RequestRecord) => {
      if (didPan.current) {
        e.preventDefault();
        return;
      }
      if (!shouldHandleTimelineBarClick(e)) return;
      e.preventDefault();
      openConversation(req);
    },
    [openConversation],
  );

  if (rows.length === 0) {
    return (
      <div className="rounded-lg border border-border/40 bg-card/40 p-4 text-sm text-muted-foreground">
        {t.empty}
      </div>
    );
  }

  const totalRequests = filtered.length;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <SegmentedToggle
          value={rowMode}
          options={ROW_MODE_OPTIONS[locale]}
          onValueChange={(mode) => {
            setRowMode(mode);
          }}
          ariaLabel={t.rowMode}
          testId="timeline-row-mode"
          buttonClassName="px-2.5 py-1 text-xs"
        />
        {hasWarmup && (
          <SegmentedToggle
            value={phaseFilter}
            options={phaseOptions(locale)}
            onValueChange={(phase) => {
              setPhaseFilter(phase);
              resetZoom();
              setScrollTop(0);
              if (scrollRef.current) scrollRef.current.scrollTop = 0;
            }}
            ariaLabel={t.phaseFilter}
            testId="timeline-phase-filter"
            buttonClassName="px-2.5 py-1 text-xs"
          />
        )}
        <span className="ml-auto text-xs text-muted-foreground">
          {t.requests(totalRequests)} ·{" "}
          {rowMode === "conversation"
            ? t.conversations(rows.length)
            : t.workers(rows.length)}{" "}
          · {t.span} {formatDuration((dataEnd - dataStart) / 1e6)} ·{" "}
          <span data-testid="timeline-total-idle-time" title={t.idleTitle}>
            {t.idle} {formatDuration(idleStats.idleNs / 1e6)}
            {idleStats.spanNs > 0
              ? ` (${((idleStats.idleNs / idleStats.spanNs) * 100).toFixed(1)}%)`
              : ""}
          </span>
          {isZoomed && (
            <>
              {" · "}
              <button
                type="button"
                onClick={() => {
                  resetZoom();
                }}
                className="text-foreground hover:underline"
              >
                {t.resetZoom}
              </button>
            </>
          )}
        </span>
      </div>

      <div className="rounded-md border border-border/60 bg-card overflow-hidden">
        <div
          ref={scrollRef}
          data-testid="request-timeline-scroll"
          className="overflow-auto"
          style={{ maxHeight: TIMELINE_BODY_MAX_HEIGHT }}
          onScroll={(event) => setScrollTop(event.currentTarget.scrollTop)}
        >
          <div className="flex w-max">
            <div
              className="sticky left-0 z-10 flex-shrink-0 border-r border-border/60 bg-card"
              style={{
                width: `min(${LABEL_WIDTH}px, 48vw)`,
                height: svgHeight,
              }}
            >
              <div
                className="sticky top-0 z-20 border-b border-border/60 bg-card flex items-end px-2 pb-1"
                style={{ height: HEADER_HEIGHT }}
              >
                <span className="text-[9px] font-mono font-bold uppercase tracking-eyebrow text-muted-foreground">
                  {rowMode === "conversation" ? t.conversation : t.worker}
                </span>
              </div>
              {visibleRows.map((row, visibleIndex) => {
                const rowIndex = visibleRange.start + visibleIndex;
                const isSubagentRow = row.kind === "subagent";
                const isChildRow = row.kind === "stream" || row.kind === "aux";
                const isExpandable =
                  isSubagentRow && (row.streamCount ?? 1) > 1;
                const isExpanded =
                  isExpandable && expandedSubagents.has(row.key);
                return (
                  <div
                    key={row.key}
                    data-timeline-row-kind={row.kind}
                    className="flex items-center gap-1 overflow-hidden pr-2"
                    style={{
                      position: "absolute",
                      top: HEADER_HEIGHT + rowIndex * ROW_SPAN,
                      left: 0,
                      right: 0,
                      height: ROW_HEIGHT + ROW_GAP,
                      paddingLeft: 4 + row.depth * 10,
                    }}
                  >
                    {isExpandable ? (
                      <button
                        type="button"
                        onClick={() => {
                          toggleSubagent(row.key);
                        }}
                        className="size-3.5 flex items-center justify-center text-muted-foreground hover:text-foreground shrink-0"
                        aria-label={isExpanded ? t.collapse : t.expand}
                        title={isExpanded ? t.collapse : t.expand}
                      >
                        <span className="text-3xs leading-none">
                          {isExpanded ? "▾" : "▸"}
                        </span>
                      </button>
                    ) : (
                      <span className="size-3.5 shrink-0" />
                    )}
                    <span
                      className="inline-block w-1 h-3 rounded-sm flex-shrink-0"
                      style={{
                        backgroundColor: row.color,
                        opacity: isChildRow ? 0.4 : isSubagentRow ? 0.55 : 1,
                      }}
                    />
                    <span
                      className="text-3xs font-mono truncate"
                      style={{
                        color: row.color,
                        opacity: isChildRow ? 0.7 : isSubagentRow ? 0.85 : 1,
                      }}
                    >
                      {row.label}
                      {isExpandable && (
                        <span className="text-muted-foreground ml-1">
                          ×{row.streamCount}
                        </span>
                      )}
                      {isSubagentRow && (row.auxCount ?? 0) > 0 && (
                        <span className="text-muted-foreground ml-1">
                          +{row.auxCount} {t.aux}
                        </span>
                      )}
                    </span>
                    <span className="text-[9px] font-mono text-muted-foreground ml-auto shrink-0">
                      {row.requests.length > 0 ? row.requests.length : "—"}
                    </span>
                  </div>
                );
              })}
            </div>

            <div className="flex-shrink-0" style={{ height: svgHeight }}>
              <svg
                ref={zoomSvgRef}
                data-testid="request-timeline-svg"
                role="img"
                aria-label={t.chartAria}
                width={CHART_WIDTH}
                height={viewportHeight}
                className="sticky top-0 block bg-card"
                style={{ cursor: isZoomed ? "grab" : "crosshair" }}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseLeave}
                onDoubleClick={resetZoom}
              >
                <TimelineBars
                  rows={visibleRows}
                  firstRowIndex={visibleRange.start}
                  scrollTop={scrollTop}
                  viewportHeight={viewportHeight}
                  expandedSubagents={expandedSubagents}
                  dataStart={dataStart}
                  vStart={vStart}
                  vEnd={vEnd}
                  datasetSlug={datasetSlug}
                  locale={locale}
                  onBarHover={handleBarHover}
                  onBarLeave={handleBarLeave}
                  onBarClick={handleBarClick}
                />

                {cursor && (
                  <line
                    x1={cursor.xPx}
                    x2={cursor.xPx}
                    y1={0}
                    y2={viewportHeight}
                    stroke="currentColor"
                    strokeWidth={1}
                    opacity={0.45}
                    pointerEvents="none"
                  />
                )}
              </svg>
            </div>
          </div>
        </div>
      </div>

      <div className="flex items-center px-1 text-2xs text-muted-foreground">
        <span className="ml-auto opacity-70">{t.hint}</span>
      </div>

      {cursor && !tooltip && (
        <CursorPopover
          cursor={cursor}
          dataStart={dataStart}
          times={sortedTimes}
        />
      )}

      {tooltip && (
        <TimelineTooltip data={tooltip} linkable={Boolean(datasetSlug)} />
      )}
    </div>
  );
}
