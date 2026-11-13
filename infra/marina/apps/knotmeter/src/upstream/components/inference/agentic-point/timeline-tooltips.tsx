// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.

import type { RequestRecord } from "@/hooks/api/use-request-timeline";

import { formatDuration, formatTickLabel } from "./timeline-format";
import {
  cursorStatsAt,
  type SortedRequestTimes,
} from "./timeline-cursor-stats";
import {
  requestSourceLabel,
  shortenWid,
  type RequestTimelineRow,
} from "./timeline-rows";
import { useLocale } from "@/lib/i18n/use-locale";

const TOOLTIP_STRINGS = {
  en: {
    cancelled: "cancelled",
    total: "Total",
    queue: "Queue wait",
    phase: "Phase",
    depth: "Agent depth",
    worker: "worker",
    started: "Started at",
    link: "Click to view this conversation in the dataset →",
    inflight: "In flight",
    running: "running",
    waiting: "waiting",
    completed: "Completed",
    relative: (value: string) => `relative to t₀ (${value}s wall-clock)`,
  },
  zh: {
    cancelled: "已取消",
    total: "总时长",
    queue: "排队等待",
    phase: "阶段",
    depth: "Agent 深度",
    worker: "worker",
    started: "开始时间",
    link: "点击可在数据集中查看此对话 →",
    inflight: "在途",
    running: "运行中",
    waiting: "等待中",
    completed: "已完成",
    relative: (value: string) => `相对于 t₀（挂钟时间 ${value}s）`,
  },
} as const;

export interface TooltipData {
  x: number;
  y: number;
  row: RequestTimelineRow;
  req: RequestRecord;
}

export function TimelineTooltip({
  data,
  linkable,
}: {
  data: TooltipData;
  linkable?: boolean;
}) {
  const locale = useLocale();
  const t = TOOLTIP_STRINGS[locale];
  const { row, req } = data;
  const totalMs = (req.end - req.start) / 1e6;
  const queueMs = (req.start - req.credit) / 1e6;
  return (
    <div
      data-testid="request-timeline-tooltip"
      className="fixed z-50 pointer-events-none rounded-md border border-border bg-card p-2.5 shadow-lg text-2xs"
      style={{ left: data.x + 12, top: data.y - 10, maxWidth: 280 }}
    >
      <div className="flex items-center gap-2 font-medium text-foreground">
        <span
          className="inline-block w-2 h-2 rounded-sm"
          style={{ backgroundColor: row.color }}
        />
        <span className="truncate">{row.label}</span>
        <span className="text-muted-foreground">
          · {requestSourceLabel(req, locale)}
        </span>
        {req.cancelled && (
          <span className="text-destructive">· {t.cancelled}</span>
        )}
      </div>
      <div className="mt-1.5 grid grid-cols-2 gap-x-3 gap-y-0.5 text-muted-foreground">
        <span>{t.total}</span>
        <span className="text-foreground text-right tabular-nums">
          {formatDuration(totalMs)}
        </span>
        <span>{t.queue}</span>
        <span className="text-foreground text-right tabular-nums">
          {queueMs > 0.5 ? formatDuration(queueMs) : "—"}
        </span>
        {req.ttftMs !== null && (
          <>
            <span>TTFT</span>
            <span className="text-foreground text-right tabular-nums">
              {formatDuration(req.ttftMs)}
            </span>
          </>
        )}
        {req.isl !== null && (
          <>
            <span>ISL</span>
            <span className="text-foreground text-right tabular-nums">
              {req.isl.toLocaleString()}
            </span>
          </>
        )}
        {req.osl !== null && (
          <>
            <span>OSL</span>
            <span className="text-foreground text-right tabular-nums">
              {req.osl.toLocaleString()}
            </span>
          </>
        )}
        <span>{t.phase}</span>
        <span className="text-foreground text-right">{req.phase}</span>
        {req.ad > 0 && (
          <>
            <span>{t.depth}</span>
            <span className="text-foreground text-right tabular-nums">
              {req.ad}
            </span>
          </>
        )}
        <span>{t.worker}</span>
        <span className="text-foreground text-right truncate">
          {shortenWid(req.wid)}
        </span>
      </div>
      <div className="mt-1.5 pt-1 border-t border-border/40 text-3xs text-muted-foreground">
        {t.started} {formatTickLabel(req.start)}
      </div>
      {linkable && (
        <div className="mt-1 text-3xs font-medium text-primary">{t.link}</div>
      )}
    </div>
  );
}

export interface CursorState {
  xPx: number;
  tNs: number;
  clientX: number;
  clientY: number;
}

export function CursorPopover({
  cursor,
  dataStart,
  times,
}: {
  cursor: CursorState;
  dataStart: number;
  times: SortedRequestTimes;
}) {
  const locale = useLocale();
  const copy = TOOLTIP_STRINGS[locale];
  const t = cursor.tNs;
  const { running, waiting, completed, inflight } = cursorStatsAt(times, t);
  const tSec = t / 1e9;
  const wantLeft = cursor.clientX + 14;
  const left =
    typeof window === "undefined" || wantLeft + 220 < window.innerWidth
      ? wantLeft
      : cursor.clientX - 220;
  return (
    <div
      className="fixed z-40 pointer-events-none rounded-md border border-border bg-card/95 backdrop-blur p-2 shadow-lg text-2xs font-mono"
      style={{ left, top: cursor.clientY - 60, minWidth: 180 }}
    >
      <div className="flex justify-between gap-3 text-foreground">
        <span className="text-muted-foreground">t =</span>
        <span className="tabular-nums">
          {tSec < 60 ? `${tSec.toFixed(3)} s` : `${(tSec / 60).toFixed(3)} m`}
        </span>
      </div>
      <div className="mt-1 pt-1 border-t border-border/40 grid grid-cols-2 gap-x-3 gap-y-0.5 text-muted-foreground">
        <span>{copy.inflight}</span>
        <span className="text-foreground text-right tabular-nums">
          {inflight}
        </span>
        <span className="pl-3 text-3xs">{copy.running}</span>
        <span className="text-foreground text-right tabular-nums">
          {running}
        </span>
        <span className="pl-3 text-3xs">{copy.waiting}</span>
        <span className="text-foreground text-right tabular-nums">
          {waiting}
        </span>
        <span>{copy.completed}</span>
        <span className="text-foreground text-right tabular-nums">
          {completed}
        </span>
      </div>
      <div className="mt-1 pt-1 border-t border-border/40 text-[9px] text-muted-foreground">
        {copy.relative((dataStart / 1e9).toFixed(0))}
      </div>
    </div>
  );
}
