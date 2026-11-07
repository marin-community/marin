// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.

import { memo } from "react";

import type { RequestRecord } from "@/hooks/api/use-request-timeline";
import type { Locale } from "@/lib/i18n/i18n";

import {
  CHART_WIDTH,
  HEADER_HEIGHT,
  PADDING_RIGHT,
  ROW_GAP,
  ROW_HEIGHT,
  ROW_SPAN,
} from "./timeline-layout";
import { formatTickLabel } from "./timeline-format";
import { conversationHref, type RequestTimelineRow } from "./timeline-rows";

const PHASE_COLORS: Record<string, string> = {
  profiling: "#22c55e",
  warmup: "#94a3b8",
  unknown: "#64748b",
};

const NICE_TICK_MS = [
  100, 250, 500, 1000, 2000, 5000, 10_000, 30_000, 60_000, 120_000, 300_000,
  600_000, 1_800_000,
];

export interface TimelineBarsProps {
  rows: RequestTimelineRow[];
  firstRowIndex: number;
  scrollTop: number;
  viewportHeight: number;
  expandedSubagents: ReadonlySet<string>;
  dataStart: number;
  vStart: number;
  vEnd: number;
  datasetSlug?: string | null;
  locale: Locale;
  onBarHover: (
    e: React.MouseEvent,
    row: RequestTimelineRow,
    req: RequestRecord,
  ) => void;
  onBarLeave: () => void;
  onBarClick: (e: React.MouseEvent, req: RequestRecord) => void;
}

export const TimelineBars = memo(
  ({
    rows,
    firstRowIndex,
    scrollTop,
    viewportHeight,
    expandedSubagents,
    dataStart,
    vStart,
    vEnd,
    datasetSlug,
    locale,
    onBarHover,
    onBarLeave,
    onBarClick,
  }: TimelineBarsProps) => {
    const visibleDur = Math.max(vEnd - vStart, 1);
    const scale = (CHART_WIDTH - PADDING_RIGHT) / visibleDur;
    const xOf = (ns: number) => (ns - dataStart - vStart) * scale;

    const targetMs = visibleDur / 1e6 / 8;
    const tickMs = NICE_TICK_MS.find((n) => n >= targetMs) ?? targetMs;
    const tickNs = tickMs * 1e6;
    const ticks: number[] = [];
    const tickStart = Math.floor(vStart / tickNs) * tickNs;
    for (let t = tickStart; t <= vEnd + tickNs; t += tickNs) {
      if (t >= vStart && t <= vEnd) ticks.push(t);
    }

    return (
      <>
        <defs>
          <clipPath id="timeline-body-clip">
            <rect
              x={0}
              y={HEADER_HEIGHT}
              width={CHART_WIDTH}
              height={Math.max(0, viewportHeight - HEADER_HEIGHT)}
            />
          </clipPath>
        </defs>
        <line
          x1={0}
          y1={HEADER_HEIGHT}
          x2={CHART_WIDTH}
          y2={HEADER_HEIGHT}
          stroke="currentColor"
          opacity={0.15}
        />

        {ticks.map((t) => {
          const x = (t - vStart) * scale;
          return (
            <g key={t}>
              <line
                x1={x}
                y1={HEADER_HEIGHT}
                x2={x}
                y2={viewportHeight}
                stroke="currentColor"
                opacity={0.08}
                strokeDasharray="2 4"
              />
              <text
                x={x + 2}
                y={HEADER_HEIGHT - 6}
                fill="currentColor"
                opacity={0.55}
                fontSize={9}
                fontFamily="ui-monospace, SFMono-Regular, monospace"
              >
                {formatTickLabel(t)}
              </text>
            </g>
          );
        })}

        <g clipPath="url(#timeline-body-clip)">
          {rows.map((row, idx) => (
            <line
              key={`sep-${row.key}`}
              x1={0}
              y1={HEADER_HEIGHT + (firstRowIndex + idx) * ROW_SPAN - scrollTop}
              x2={CHART_WIDTH}
              y2={HEADER_HEIGHT + (firstRowIndex + idx) * ROW_SPAN - scrollTop}
              stroke="currentColor"
              opacity={0.04}
            />
          ))}

          {rows.map((row, rowIdx) => {
            const yTop =
              HEADER_HEIGHT +
              (firstRowIndex + rowIdx) * (ROW_HEIGHT + ROW_GAP) -
              scrollTop +
              2;
            const barH = ROW_HEIGHT - 4;
            if (
              row.kind === "subagent" &&
              (row.streamCount ?? 1) > 1 &&
              expandedSubagents.has(row.key)
            ) {
              return null;
            }
            return row.requests.map((req) => {
              const xCredit = xOf(req.credit);
              const xStart = xOf(req.start);
              const xEnd = xOf(req.end);
              if (xEnd < -2 || xCredit > CHART_WIDTH + 2) return null;
              const runW = Math.max(xEnd - xStart, 1);
              const queueW = Math.max(xStart - xCredit, 0);
              const phaseColor =
                PHASE_COLORS[req.phase] ?? PHASE_COLORS.unknown!;
              const barKey = `${req.cid}-${req.ti}-${req.start}`;
              const barChildren = (
                <>
                  {queueW >= 1 && (
                    <rect
                      x={xCredit}
                      y={yTop + barH / 2 - 1}
                      width={queueW}
                      height={2}
                      fill={row.color}
                      opacity={0.35}
                    />
                  )}
                  <rect
                    x={xStart}
                    y={yTop}
                    width={runW}
                    height={barH}
                    rx={2}
                    fill={row.color}
                    opacity={
                      req.cancelled
                        ? 0.35
                        : row.kind === "stream" || row.kind === "aux"
                          ? 0.5
                          : row.kind === "subagent"
                            ? 0.6
                            : 0.85
                    }
                  />
                  <rect
                    x={xStart}
                    y={yTop + barH - 2}
                    width={runW}
                    height={2}
                    rx={1}
                    fill={phaseColor}
                    opacity={0.85}
                  />
                  {req.cancelled && runW > 6 && (
                    <line
                      x1={xStart + 1}
                      y1={yTop + 1}
                      x2={xStart + runW - 1}
                      y2={yTop + barH - 1}
                      stroke="currentColor"
                      strokeWidth={0.7}
                      opacity={0.6}
                    />
                  )}
                </>
              );
              if (!datasetSlug) {
                return (
                  <g
                    key={barKey}
                    onMouseMove={(e) => onBarHover(e, row, req)}
                    onMouseLeave={onBarLeave}
                  >
                    {barChildren}
                  </g>
                );
              }
              return (
                <a
                  key={barKey}
                  href={conversationHref(datasetSlug, req, locale)}
                  aria-label={`Open corpus request ${req.srcOuter ?? req.ti}${req.srcInner === undefined ? "" : ` / child ${req.srcInner}`}`}
                  data-corpus-id={req.srcTrace}
                  data-source-outer={req.srcOuter}
                  data-source-inner={req.srcInner}
                  data-start-ns={req.start}
                  data-end-ns={req.end}
                  data-credit-ns={req.credit}
                  onMouseMove={(e) => onBarHover(e, row, req)}
                  onMouseLeave={onBarLeave}
                  onClick={(e) => onBarClick(e, req)}
                  onDragStart={(e) => e.preventDefault()}
                  style={{ cursor: "pointer" }}
                >
                  {barChildren}
                </a>
              );
            });
          })}
        </g>
      </>
    );
  },
);
