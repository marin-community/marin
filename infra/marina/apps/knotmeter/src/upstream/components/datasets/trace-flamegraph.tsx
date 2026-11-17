// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";

import type { ConversationStructure } from "@/hooks/api/use-datasets";

import { useLocale } from "@/lib/i18n/use-locale";
import { compact, formatShare } from "./format";
import {
  buildRowOverlaps,
  buildVisibleRows,
  computeBraceLayout,
  formatElapsedTime,
  MAX_LANES,
  OVERLAP_COLORS,
  resolveDeepLinkTarget,
  type VisibleRow,
} from "./trace-flamegraph-model";

export {
  findRequestOverlapGroups,
  formatElapsedTime,
  resolveDeepLinkTarget,
} from "./trace-flamegraph-model";
export type {
  DeepLinkHighlight,
  DeepLinkTarget,
  RequestOverlapGroup,
  TimedRequest,
} from "./trace-flamegraph-model";

const SEG = {
  cached: "#10b981",
  uncached: "#f59e0b",
  output: "#8b5cf6",
} as const;

const STRINGS = {
  en: {
    cachedPrefix: "Cached prefix",
    uncachedInput: "Uncached input",
    output: "Output",
    parallelBracket: "Bracketed rows ran in parallel",
    expandAll: "Expand all",
    collapseAll: "Collapse all",
    denseParallel: (max: number, overflow: number) =>
      `Dense parallel region — bracket lanes capped at ${max}; ${overflow} further overlapping ${overflow === 1 ? "group is" : "groups are"} folded into the last lane.`,
    cachedPct: "Cached %",
    fromStart: "From start",
    parallelWith: (n: number) =>
      `Ran in parallel with ${n} other request${n === 1 ? "" : "s"}`,
  },
  zh: {
    cachedPrefix: "缓存前缀",
    uncachedInput: "未缓存输入",
    output: "输出",
    parallelBracket: "括号内行并行运行",
    expandAll: "全部展开",
    collapseAll: "全部折叠",
    denseParallel: (max: number, overflow: number) =>
      `密集并行区域——括号通道上限 ${max}；另有 ${overflow} 个重叠组折叠到最后通道。`,
    cachedPct: "缓存 %",
    fromStart: "距开始",
    parallelWith: (n: number) => `与另外 ${n} 个请求并行运行`,
  },
} as const;

const LANE_W = 14;

interface TooltipState {
  x: number;
  y: number;
  row: VisibleRow;
}

export function TraceFlamegraph({
  structure,
  highlightTurn,
  highlightRawIndex,
  highlightInnerIndex,
  highlightAgentId,
}: {
  structure: ConversationStructure;
  highlightTurn?: number | null;
  highlightRawIndex?: number | null;
  highlightInnerIndex?: number | null;
  highlightAgentId?: string | null;
}) {
  const nodes = structure.nodes;
  const locale = useLocale();
  const t = STRINGS[locale];

  const LEGEND = [
    { key: "cached", label: t.cachedPrefix, color: SEG.cached },
    { key: "uncached", label: t.uncachedInput, color: SEG.uncached },
    { key: "output", label: t.output, color: SEG.output },
  ] as const;

  const target = useMemo(
    () =>
      resolveDeepLinkTarget(nodes, {
        turn: highlightTurn,
        raw: highlightRawIndex,
        inner: highlightInnerIndex,
        agent: highlightAgentId,
      }),
    [
      nodes,
      highlightTurn,
      highlightRawIndex,
      highlightInnerIndex,
      highlightAgentId,
    ],
  );

  const [expanded, setExpanded] = useState<Set<number>>(() =>
    typeof target?.expandGroup === "number"
      ? new Set([target.expandGroup])
      : new Set(),
  );
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const [highlightKey, setHighlightKey] = useState<string | null>(
    target?.rowKey ?? null,
  );

  useEffect(() => {
    if (!target) return;
    if (typeof target.expandGroup === "number") {
      const gi = target.expandGroup;
      setExpanded((prev) => (prev.has(gi) ? prev : new Set(prev).add(gi)));
    }
    setHighlightKey(target.rowKey);
    const raf = requestAnimationFrame(() => {
      scrollRef.current
        ?.querySelector<HTMLElement>(`[data-rowkey="${target.rowKey}"]`)
        ?.scrollIntoView({ block: "center", behavior: "smooth" });
    });
    return () => {
      cancelAnimationFrame(raf);
    };
  }, [target]);

  const groupIndexes = useMemo(() => {
    const out: number[] = [];
    nodes.forEach((node, i) => {
      if (node.kind === "subagent") out.push(i);
    });
    return out;
  }, [nodes]);

  const toggle = useCallback((i: number) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });
  }, []);

  const expandAll = useCallback(
    () => setExpanded(new Set(groupIndexes)),
    [groupIndexes],
  );
  const collapseAll = useCallback(() => setExpanded(new Set()), []);

  const overlapsByRow = useMemo(() => buildRowOverlaps(nodes), [nodes]);

  const rows = useMemo(
    () => buildVisibleRows(nodes, expanded, overlapsByRow, locale),
    [nodes, expanded, overlapsByRow, locale],
  );

  const maxTotal = useMemo(
    () => Math.max(1, ...rows.filter((r) => !r.isGroup).map((r) => r.total)),
    [rows],
  );
  const maxGroupTotal = useMemo(
    () => Math.max(1, ...rows.filter((r) => r.isGroup).map((r) => r.total)),
    [rows],
  );

  const braces = useMemo(() => computeBraceLayout(rows), [rows]);

  const onMove = (e: React.MouseEvent, row: VisibleRow) => {
    setTooltip({ x: e.clientX, y: e.clientY, row });
  };

  const onFocus = (e: React.FocusEvent<HTMLDivElement>, row: VisibleRow) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setTooltip({
      x: rect.left + rect.width / 2,
      y: rect.top + rect.height / 2,
      row,
    });
  };

  const ariaValueText = (row: VisibleRow) =>
    locale === "zh"
      ? `${t.cachedPrefix}：${compact(row.cached)}；${t.uncachedInput}：${compact(row.uncached)}；${t.output}：${compact(row.output)}`
      : `${t.cachedPrefix}: ${compact(row.cached)}; ${t.uncachedInput}: ${compact(row.uncached)}; ${t.output}: ${compact(row.output)}`;

  return (
    <div className="relative">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3 text-xs">
          {LEGEND.map((l) => (
            <span key={l.key} className="inline-flex items-center gap-1.5">
              <span
                className="inline-block size-3 rounded-sm"
                style={{ backgroundColor: l.color }}
              />
              <span className="text-muted-foreground">{l.label}</span>
            </span>
          ))}
          <span className="inline-flex items-center gap-1.5">
            <span
              className="inline-block h-4 w-2 rounded-l-sm border-y-2 border-l-2"
              style={{ borderColor: OVERLAP_COLORS[0] }}
            />
            <span className="text-muted-foreground">{t.parallelBracket}</span>
          </span>
        </div>
        {groupIndexes.length > 0 && (
          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => {
                expandAll();
              }}
              className="rounded-md border border-border/40 px-2 py-1 text-xs hover:bg-accent"
            >
              {t.expandAll}
            </button>
            <button
              type="button"
              onClick={() => {
                collapseAll();
              }}
              className="rounded-md border border-border/40 px-2 py-1 text-xs hover:bg-accent"
            >
              {t.collapseAll}
            </button>
          </div>
        )}
      </div>

      {braces.overflowLanes > 0 && (
        <p className="mb-2 text-2xs text-muted-foreground">
          {t.denseParallel(MAX_LANES, braces.overflowLanes)}
        </p>
      )}

      <div
        ref={scrollRef}
        className="max-h-[520px] overflow-auto rounded-md border border-border/40 bg-muted/10 p-2"
        data-testid="flamegraph-scroll"
      >
        <div className="flex min-w-[36rem] flex-col gap-0">
          {rows.map((row, idx) => {
            const denom = row.isGroup ? maxGroupTotal : maxTotal;
            const widthPct = Math.min(
              100,
              Math.max(0.5, (row.total / denom) * 100),
            );
            const cw = row.total > 0 ? (row.cached / row.total) * 100 : 0;
            const uw = row.total > 0 ? (row.uncached / row.total) * 100 : 0;
            const ow = row.total > 0 ? (row.output / row.total) * 100 : 0;
            const isHighlighted = row.key === highlightKey;
            const segs = braces.rowSegs[idx]!;
            return (
              <div
                key={row.key}
                data-rowkey={row.key}
                data-selected={isHighlighted || undefined}
                className={`flex items-stretch rounded-sm transition-colors duration-700 ${
                  isHighlighted ? "bg-primary/20 ring-2 ring-primary" : "ring-0"
                }`}
              >
                {braces.laneCount > 0 && (
                  <div
                    className="relative shrink-0 self-stretch"
                    style={{ width: braces.laneCount * LANE_W }}
                  >
                    {segs.map(({ lane, seg }) => {
                      const top = seg.role === "first" ? "50%" : "0";
                      const bottom = seg.role === "last" ? "50%" : "0";
                      return (
                        <div
                          key={`${lane}-${seg.groupId}`}
                          className="absolute top-0 bottom-0"
                          style={{ left: lane * LANE_W, width: LANE_W }}
                          {...(seg.isMember
                            ? {
                                "data-testid": `flamegraph-overlap-${row.key}`,
                                "data-overlap-group": seg.groupId,
                              }
                            : {})}
                          title={
                            seg.isMember
                              ? `${t.parallelWith(seg.peerCount)} (+${formatElapsedTime(seg.startS)}–${formatElapsedTime(seg.endS)})`
                              : undefined
                          }
                        >
                          <div
                            className="absolute"
                            style={{
                              left: 5,
                              width: 2,
                              top,
                              bottom,
                              backgroundColor: seg.color,
                              opacity: seg.isMember ? 0.95 : 0.3,
                              borderTopLeftRadius: seg.role === "first" ? 3 : 0,
                              borderBottomLeftRadius:
                                seg.role === "last" ? 3 : 0,
                            }}
                          />
                          {seg.isMember && (
                            <div
                              className="absolute"
                              style={{
                                left: 5,
                                top: "50%",
                                height: 2,
                                width: LANE_W - 7,
                                transform: "translateY(-1px)",
                                backgroundColor: seg.color,
                              }}
                            />
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}

                <div
                  className="flex flex-1 items-center gap-2 py-0.5"
                  style={{ paddingLeft: row.indent * 20 }}
                >
                  <div className="flex w-52 shrink-0 items-center overflow-hidden">
                    {row.isGroup ? (
                      <button
                        type="button"
                        onClick={() => {
                          if (row.groupIndex !== undefined)
                            toggle(row.groupIndex);
                        }}
                        className="flex items-center gap-1 truncate text-left text-xs font-medium text-foreground hover:text-primary"
                      >
                        <span className="inline-block w-3 text-muted-foreground">
                          {row.isExpanded ? "▾" : "▸"}
                        </span>
                        <span className="truncate">{row.label}</span>
                      </button>
                    ) : (
                      <span className="truncate pl-4 text-xs text-foreground">
                        {row.label}
                      </span>
                    )}
                  </div>

                  <div
                    className="w-36 shrink-0 text-2xs tabular-nums text-muted-foreground"
                    data-testid={`flamegraph-time-${row.key}`}
                  >
                    {row.timeLabel ?? "—"}
                  </div>

                  <div
                    data-testid={`flamegraph-bar-${row.key}`}
                    className="relative flex h-5 flex-1 items-center"
                    role="meter"
                    tabIndex={0}
                    aria-label={row.label}
                    aria-valuemin={0}
                    aria-valuemax={denom}
                    aria-valuenow={row.total}
                    aria-valuetext={ariaValueText(row)}
                    aria-describedby={
                      tooltip?.row.key === row.key
                        ? "flamegraph-tooltip"
                        : undefined
                    }
                    onMouseMove={(e) => onMove(e, row)}
                    onMouseLeave={(e) => {
                      if (
                        e.currentTarget.ownerDocument.activeElement !==
                        e.currentTarget
                      ) {
                        setTooltip(null);
                      }
                    }}
                    onFocus={(e) => onFocus(e, row)}
                    onBlur={() => setTooltip(null)}
                  >
                    <div
                      className={`flex overflow-hidden rounded-sm ${
                        row.isGroup ? "h-2.5 opacity-80" : "h-5"
                      }`}
                      style={{ width: `${widthPct}%` }}
                    >
                      <div
                        style={{ width: `${cw}%`, backgroundColor: SEG.cached }}
                      />
                      <div
                        style={{
                          width: `${uw}%`,
                          backgroundColor: SEG.uncached,
                        }}
                      />
                      <div
                        style={{ width: `${ow}%`, backgroundColor: SEG.output }}
                      />
                    </div>
                  </div>

                  <div className="w-16 shrink-0 text-right text-2xs tabular-nums text-muted-foreground">
                    {compact(row.total)}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {tooltip &&
        mounted &&
        createPortal(
          <div
            id="flamegraph-tooltip"
            role="tooltip"
            data-testid="flamegraph-tooltip"
            className="pointer-events-none fixed z-50 rounded-md border border-border bg-popover px-2.5 py-1.5 text-xs shadow-md"
            style={{ left: tooltip.x + 12, top: tooltip.y + 12 }}
          >
            <div className="mb-1 font-medium text-foreground">
              {tooltip.row.label}
              {tooltip.row.sublabel ? (
                <span className="ml-1 font-normal text-muted-foreground">
                  {tooltip.row.sublabel}
                </span>
              ) : null}
            </div>
            <div className="grid grid-cols-[auto_auto] gap-x-3 gap-y-0.5 text-muted-foreground">
              <span style={{ color: SEG.cached }}>{t.cachedPrefix}</span>
              <span className="text-right tabular-nums text-foreground">
                {compact(tooltip.row.cached)}
              </span>
              <span style={{ color: SEG.uncached }}>{t.uncachedInput}</span>
              <span className="text-right tabular-nums text-foreground">
                {compact(tooltip.row.uncached)}
              </span>
              <span style={{ color: SEG.output }}>{t.output}</span>
              <span className="text-right tabular-nums text-foreground">
                {compact(tooltip.row.output)}
              </span>
              <span>{t.cachedPct}</span>
              <span className="text-right tabular-nums text-foreground">
                {formatShare(
                  tooltip.row.cached,
                  tooltip.row.cached + tooltip.row.uncached,
                )}
              </span>
              <span>{t.fromStart}</span>
              <span className="text-right tabular-nums text-foreground">
                {tooltip.row.timeLabel ?? "—"}
              </span>
            </div>
          </div>,
          document.body,
        )}
    </div>
  );
}
