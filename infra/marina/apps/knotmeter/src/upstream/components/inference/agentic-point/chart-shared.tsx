// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.

import { useLocale } from "@/lib/i18n/use-locale";

import {
  LEGEND_BASELINE_OFFSET,
  LEGEND_FONT_SIZE,
  LEGEND_ROW_HEIGHT,
  LEGEND_SWATCH_INSET,
  LEGEND_SWATCH_WIDTH,
  LEGEND_TEXT_OFFSET,
  type LegendLayout,
} from "./chart-legend";

export const CHART_PAD = { top: 12, right: 16, bottom: 56, left: 60 } as const;

export const CHART_SIZES = {
  inline: { width: 720, height: 260 },
  expanded: { width: 1300, height: 520 },
} as const;

export const PERCENTILE_COLORS = {
  mean: "#ef4444",
  p50: "#3b82f6",
  p75: "#22c55e",
  p90: "#f59e0b",
  p95: "#ef4444",
  p99: "#a855f7",
} as const;

export const fmtCount = (n: number): string =>
  n >= 10000
    ? new Intl.NumberFormat("en-US").format(Math.round(n))
    : String(Math.round(n));

export const fmtSeconds = (s: number): string => {
  if (s < 60) return `${Math.round(s)}s`;
  const m = Math.floor(s / 60);
  const rem = Math.round(s % 60);
  return `${m}m ${rem}s`;
};

export function ChartEmpty({
  height = 260,
  message,
}: {
  height?: number;
  message?: string;
}) {
  const locale = useLocale();
  return (
    <div
      className="grid place-items-center text-xs text-muted-foreground"
      style={{ height }}
    >
      {message ?? (locale === "zh" ? "暂无数据" : "No data")}
    </div>
  );
}

export function ChartSkeleton() {
  return <div className="h-[260px] rounded-md bg-muted/30 animate-pulse" />;
}

export type LegendSwatch = "line" | "dashed-line" | "area";

export interface ChartLegendEntry {
  label: string;
  color: string;
  swatch?: LegendSwatch;
  strokeWidth?: number;
  opacity?: number;
}

export function ChartLegend({
  entries,
  layout,
  left,
  baselineY,
}: {
  entries: readonly ChartLegendEntry[];
  layout: LegendLayout;
  left: number;
  baselineY: number;
}) {
  const topRowBaseline = baselineY - (layout.rows - 1) * LEGEND_ROW_HEIGHT;
  return (
    <>
      {entries.map((entry, i) => {
        const placed = layout.items[i];
        if (!placed) return null;
        const x = left + placed.x;
        const y = topRowBaseline + placed.row * LEGEND_ROW_HEIGHT;
        const swatch = entry.swatch ?? "line";
        return (
          <g key={`legend-${i}`}>
            {swatch === "area" ? (
              <rect
                x={x + LEGEND_SWATCH_INSET}
                y={y - 9}
                width={LEGEND_SWATCH_WIDTH}
                height={8}
                fill={entry.color}
                opacity={entry.opacity ?? 1}
              />
            ) : (
              <line
                x1={x + LEGEND_SWATCH_INSET}
                x2={x + LEGEND_SWATCH_INSET + LEGEND_SWATCH_WIDTH}
                y1={y - 4}
                y2={y - 4}
                stroke={entry.color}
                strokeWidth={entry.strokeWidth ?? 2}
                strokeDasharray={swatch === "dashed-line" ? "5 3" : undefined}
                opacity={entry.opacity ?? 1}
              />
            )}
            <text
              x={x + LEGEND_TEXT_OFFSET}
              y={y}
              fontSize={LEGEND_FONT_SIZE}
              fill="currentColor"
              opacity={0.9}
            >
              {entry.label}
            </text>
          </g>
        );
      })}
    </>
  );
}

export { LEGEND_BASELINE_OFFSET };
