import { useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { WandbChart, WandbRunSeries } from "../api";
import { useWandb } from "../hooks/useWandb";
import { formatRelative, useContainerSize } from "./chartUtils";

// One color per run series; the panel plots the original run and its
// resume, so two entries cover it (cycles if more runs are configured).
const SERIES_COLORS = [
  "#f59e0b", // amber-500
  "#10b981", // emerald-500
];

// Cumulative-token axis formatter: 33500000000 → "34B", 1.5e12 → "1.5T".
function formatTokens(value: number): string {
  if (value >= 1e12) return `${(value / 1e12).toFixed(1)}T`;
  if (value >= 1e9) return `${(value / 1e9).toFixed(0)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(0)}M`;
  return Math.round(value).toString();
}

// A run's terminal state is worth surfacing (the first hero run crashed
// at step ~15k and was resumed), but "running" is the boring default.
function seriesName(run: string, state: string): string {
  return state === "running" ? run : `${run} (${state})`;
}

// The first warmup points of a loss curve sit an order of magnitude above
// the rest, and a naive [min, max] y-domain squashes everything
// interesting onto the x-axis. Cap the domain at the 98th percentile of
// all plotted values instead — the few points above it are clipped by
// allowDataOverflow on the YAxis, which is how the eye reads a zoomed
// W&B chart anyway.
function clippedYDomain(series: WandbRunSeries[]): [number, number] {
  const ys = series
    .flatMap((s) => s.points.map((p) => p.y))
    .sort((a, b) => a - b);
  if (ys.length === 0) return [0, 1];
  const min = ys[0];
  const max = ys[Math.min(ys.length - 1, Math.floor(ys.length * 0.98))];
  const pad = (max - min) * 0.05 || Math.abs(max) * 0.05 || 1;
  return [min - pad, max + pad];
}

function ChartCard({ chart }: { chart: WandbChart }) {
  const { ref, size } = useContainerSize<HTMLDivElement>();
  const hasData = chart.series.some((s) => s.points.length > 0);
  const yDomain = useMemo(() => clippedYDomain(chart.series), [chart.series]);
  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
          {chart.title}
        </h4>
        <span className="text-xs text-slate-600">vs training tokens</span>
      </div>
      <div ref={ref} className="h-72 w-full">
        {hasData && size ? (
          <LineChart
            width={size.width}
            height={size.height}
            margin={{ top: 4, right: 8, bottom: 4, left: 12 }}
          >
            <CartesianGrid stroke="#1e293b" strokeDasharray="2 4" />
            <XAxis
              dataKey="x"
              type="number"
              domain={["dataMin", "dataMax"]}
              tickFormatter={formatTokens}
              stroke="#475569"
              fontSize={11}
            />
            <YAxis
              width={58}
              type="number"
              domain={yDomain}
              allowDataOverflow
              tickFormatter={(v: number) => v.toFixed(2)}
              stroke="#475569"
              fontSize={11}
            />
            <Tooltip
              contentStyle={{
                background: "#0f172a",
                border: "1px solid #1e293b",
                borderRadius: 4,
                fontSize: 12,
              }}
              labelFormatter={(value) => `${formatTokens(value as number)} tokens`}
              formatter={(value) => (value as number).toFixed(4)}
            />
            <Legend
              verticalAlign="bottom"
              height={20}
              iconType="plainline"
              wrapperStyle={{ fontSize: 11, color: "#94a3b8" }}
            />
            {chart.series.map((s, i) => (
              <Line
                key={s.run}
                data={s.points}
                dataKey="y"
                name={seriesName(s.run, s.state)}
                stroke={SERIES_COLORS[i % SERIES_COLORS.length]}
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        ) : (
          <div className="flex h-full items-center justify-center text-center text-sm text-slate-500">
            no history returned for this metric yet
          </div>
        )}
      </div>
    </div>
  );
}

export function WandbPanel() {
  const { data, isLoading, error } = useWandb();

  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
          Training
        </h3>
        {data && (
          <span className="text-xs text-slate-500">
            updated {formatRelative(data.fetchedAt)}
          </span>
        )}
      </div>
      <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-4">
        {isLoading && <div className="text-slate-400">loading…</div>}
        {error && (
          <div className="text-rose-400">failed to load: {(error as Error).message}</div>
        )}
        {data?.error && <div className="text-sm text-rose-400">{data.error}</div>}
        {data && !data.error && (
          <>
            <div className="mb-4 flex flex-wrap items-baseline gap-x-2 text-sm">
              <span className="text-slate-200">{data.reportTitle}</span>
              <a
                href={data.reportUrl}
                target="_blank"
                rel="noreferrer"
                className="text-xs text-slate-500 hover:text-emerald-300"
              >
                wandb report ↗
              </a>
            </div>
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              {data.charts.map((chart) => (
                <ChartCard key={chart.key} chart={chart} />
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
