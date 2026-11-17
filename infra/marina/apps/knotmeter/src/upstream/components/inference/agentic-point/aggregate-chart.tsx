// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.

import { useMemo } from "react";

import { ChartHover, type HoverItem } from "./chart-hover";
import { ChartEmpty, PERCENTILE_COLORS } from "./chart-shared";
import { useLocale } from "@/lib/i18n/use-locale";

export type PercentileKey = "mean" | "p50" | "p75" | "p90" | "p95" | "p99";

interface PercentileLine {
  key: PercentileKey;
  label: string;
  color: string;
}

const PERCENTILE_LINES: PercentileLine[] = [
  { key: "mean", label: "Mean", color: PERCENTILE_COLORS.mean },
  { key: "p50", label: "P50", color: PERCENTILE_COLORS.p50 },
  { key: "p75", label: "P75", color: PERCENTILE_COLORS.p75 },
  { key: "p90", label: "P90", color: PERCENTILE_COLORS.p90 },
  { key: "p95", label: "P95", color: PERCENTILE_COLORS.p95 },
  { key: "p99", label: "P99", color: PERCENTILE_COLORS.p99 },
];

const PAD = { top: 16, right: 16, bottom: 90, left: 64 };

export interface AggregatePoint {
  label: string;
  values: Partial<Record<PercentileKey, number>>;
  id?: number;
}

export function AggregateChart({
  points,
  unit,
  yMax,
  yFmt,
  width = 720,
  height = 320,
}: {
  points: readonly AggregatePoint[];
  unit: string;
  yMax?: number;
  yFmt?: (v: number) => string;
  width?: number;
  height?: number;
}) {
  const locale = useLocale();
  const percentileLines = useMemo(
    () =>
      PERCENTILE_LINES.map((line) =>
        line.key === "mean" && locale === "zh"
          ? { ...line, label: "平均值" }
          : line,
      ),
    [locale],
  );
  const W = width;
  const H = height;
  const fmt = (v: number) =>
    yFmt
      ? yFmt(v)
      : v >= 10000
        ? new Intl.NumberFormat("en-US").format(Math.round(v))
        : v.toFixed(v < 10 ? 2 : 0);

  const computed = useMemo(() => {
    if (points.length === 0) return null;
    let yMaxComputed = 0;
    for (const p of points) {
      for (const line of percentileLines) {
        const v = p.values[line.key];
        if (typeof v === "number" && Number.isFinite(v) && v > yMaxComputed)
          yMaxComputed = v;
      }
    }
    const yTop = yMax ?? (yMaxComputed === 0 ? 1 : yMaxComputed * 1.05);
    const innerW = W - PAD.left - PAD.right;
    const innerH = H - PAD.top - PAD.bottom;
    return { yTop, innerW, innerH };
  }, [points, W, H, yMax, percentileLines]);

  if (!computed) {
    return <ChartEmpty height={H} />;
  }
  const { yTop, innerW, innerH } = computed;

  const xOf = (i: number) =>
    points.length === 1
      ? PAD.left + innerW / 2
      : PAD.left + (i / (points.length - 1)) * innerW;
  const yOf = (v: number) => PAD.top + (1 - v / yTop) * innerH;

  const yTicks = Array.from({ length: 5 }, (_, i) => (yTop * i) / 4);

  const resolve = (fraction: number) => {
    const idx = Math.round(fraction * (points.length - 1));
    const p = points[Math.max(0, Math.min(points.length - 1, idx))];
    if (!p) return null;
    const items: HoverItem[] = [];
    for (const line of percentileLines) {
      const v = p.values[line.key];
      if (typeof v !== "number" || !Number.isFinite(v)) continue;
      items.push({ color: line.color, label: line.label, value: fmt(v) });
    }
    return { items, title: p.label };
  };

  return (
    <div className="w-full">
      <div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
        {percentileLines.map((line) => (
          <div key={line.key} className="flex items-center gap-1.5">
            <span
              className="inline-block w-3 h-0.5"
              style={{ backgroundColor: line.color }}
            />
            <span className="text-muted-foreground">{line.label}</span>
          </div>
        ))}
        <span className="ml-auto text-muted-foreground">
          {locale === "zh"
            ? `${points.length} 个配置 · 单位：${unit}`
            : `${points.length} configs · units: ${unit}`}
        </span>
      </div>
      <ChartHover pad={PAD} width={W} height={H} resolve={resolve}>
        {yTicks.map((v, i) => {
          const y = yOf(v);
          return (
            <g key={`y${i}`}>
              <line
                x1={PAD.left}
                x2={PAD.left + innerW}
                y1={y}
                y2={y}
                stroke="currentColor"
                opacity={0.08}
              />
              <text
                x={PAD.left - 8}
                y={y + 3}
                fontSize={10}
                fill="currentColor"
                opacity={0.55}
                textAnchor="end"
              >
                {fmt(v)}
              </text>
            </g>
          );
        })}

        {points.map((p, i) => {
          const x = xOf(i);
          return (
            <g key={`x${i}`}>
              <line
                x1={x}
                x2={x}
                y1={PAD.top + innerH}
                y2={PAD.top + innerH + 4}
                stroke="currentColor"
                opacity={0.4}
              />
              <text
                x={x}
                y={PAD.top + innerH + 8}
                fontSize={10}
                fill="currentColor"
                opacity={0.7}
                textAnchor="end"
                transform={`rotate(-30 ${x} ${PAD.top + innerH + 8})`}
              >
                {p.label}
              </text>
            </g>
          );
        })}

        <line
          x1={PAD.left}
          x2={PAD.left + innerW}
          y1={PAD.top + innerH}
          y2={PAD.top + innerH}
          stroke="currentColor"
          opacity={0.25}
        />

        {percentileLines.map((line) => {
          const segments: { x1: number; y1: number; x2: number; y2: number }[] =
            [];
          let prev: { x: number; y: number } | null = null;
          for (let i = 0; i < points.length; i++) {
            const v = points[i]!.values[line.key];
            if (typeof v !== "number" || !Number.isFinite(v)) {
              prev = null;
              continue;
            }
            const x = xOf(i);
            const y = yOf(v);
            if (prev) segments.push({ x1: prev.x, y1: prev.y, x2: x, y2: y });
            prev = { x, y };
          }
          return (
            <g key={`hline-${line.key}`} opacity={0.35}>
              {segments.map((s, j) => (
                <line
                  key={`s${j}`}
                  x1={s.x1}
                  y1={s.y1}
                  x2={s.x2}
                  y2={s.y2}
                  stroke={line.color}
                  strokeWidth={1}
                />
              ))}
            </g>
          );
        })}

        {points.map((p, i) => {
          const x = xOf(i);
          const present = percentileLines
            .filter(
              (line) =>
                typeof p.values[line.key] === "number" &&
                Number.isFinite(p.values[line.key]!),
            )
            .map((line) => ({ ...line, value: p.values[line.key]! }));
          if (present.length === 0) return null;
          const pctlOnly = present.filter((p2) => p2.key !== "mean");
          const bandValues = pctlOnly.length > 0 ? pctlOnly : present;
          const bandYs = bandValues.map((b) => yOf(b.value));
          const yLo = Math.min(...bandYs);
          const yHi = Math.max(...bandYs);
          return (
            <g key={`bar-${i}`}>
              <line
                x1={x}
                x2={x}
                y1={yLo}
                y2={yHi}
                stroke="currentColor"
                strokeWidth={1}
                opacity={0.35}
              />
              {present.map((b) => {
                const ty = yOf(b.value);
                if (b.key === "mean") {
                  const s = 4;
                  return (
                    <polygon
                      key={`m-${b.key}`}
                      points={`${x},${ty - s} ${x + s},${ty} ${x},${ty + s} ${x - s},${ty}`}
                      fill={b.color}
                      stroke={b.color}
                    />
                  );
                }
                return (
                  <line
                    key={`tk-${b.key}`}
                    x1={x - 6}
                    x2={x + 6}
                    y1={ty}
                    y2={ty}
                    stroke={b.color}
                    strokeWidth={2.5}
                  />
                );
              })}
            </g>
          );
        })}
      </ChartHover>
    </div>
  );
}
