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
import type { WandbChart, WandbPoint, WandbRunSeries } from "../api";
import { useWandb } from "../hooks/useWandb";
import { formatRelative, useContainerSize } from "./chartUtils";

// One color per run series, in report runset order (cycles if the report
// ever pins more runs than entries).
const SERIES_COLORS = [
  "#f59e0b", // amber-500
  "#10b981", // emerald-500
  "#06b6d4", // cyan-500
  "#8b5cf6", // violet-500
];

// W&B-style readability for dense, noisy series (train loss): draw a
// debiased EMA as the main line and keep the raw trace faint behind it.
// Sparse series (evals log every few hours) are plotted as-is — smoothing
// a handful of points only distorts them.
const SMOOTHING_ALPHA = 0.92;
const SMOOTH_MIN_POINTS = 50;

// Cumulative-token axis formatter: 33500000000 → "34B", 1.5e12 → "1.5T".
function formatTokens(value: number): string {
  if (value >= 1e12) return `${(value / 1e12).toFixed(1)}T`;
  if (value >= 1e9) return `${(value / 1e9).toFixed(0)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(0)}M`;
  return Math.round(value).toString();
}

// A run's terminal state is worth surfacing (hero runs crash and get
// resumed), but "running" is the boring default. Run names are experiment
// ids pushing 80 chars; middle-truncate to keep the legend one line and
// the distinguishing suffix (resume15k_v2_10T) visible.
function seriesName(run: string, state: string): string {
  const truncated = run.length > 44 ? `${run.slice(0, 21)}…${run.slice(-22)}` : run;
  return state === "running" ? truncated : `${truncated} (${state})`;
}

function emaSmooth(points: WandbPoint[]): WandbPoint[] {
  let ema = 0;
  return points.map((p, i) => {
    ema = ema * SMOOTHING_ALPHA + p.y * (1 - SMOOTHING_ALPHA);
    return { x: p.x, y: ema / (1 - Math.pow(SMOOTHING_ALPHA, i + 1)) };
  });
}

// Warmup loss sits an order of magnitude above the rest of the curve, and
// a naive [min, max] y-domain squashes everything interesting onto the
// x-axis. Compute the domain from each series with its first 10% of
// points dropped (that's where warmup lives), capped at the 98th
// percentile of what remains; points above are clipped by
// allowDataOverflow on the YAxis — the same effect as the y-axis cap the
// W&B report's own panels use. Domain ends snap to 0.05 so ticks land on
// clean values.
function clippedYDomain(series: WandbRunSeries[]): [number, number] {
  const ys = series
    .flatMap((s) => s.points.slice(Math.ceil(s.points.length * 0.1)).map((p) => p.y))
    .sort((a, b) => a - b);
  if (ys.length === 0) return [0, 1];
  const min = ys[0];
  const max = ys[Math.min(ys.length - 1, Math.floor(ys.length * 0.98))];
  const pad = (max - min) * 0.05 || Math.abs(max) * 0.05 || 1;
  return [Math.floor((min - pad) * 20) / 20, Math.ceil((max + pad) * 20) / 20];
}

function ChartCard({ chart }: { chart: WandbChart }) {
  const { ref, size } = useContainerSize<HTMLDivElement>();
  const hasData = chart.series.some((s) => s.points.length > 0);
  const yDomain = useMemo(() => clippedYDomain(chart.series), [chart.series]);
  const smoothed = useMemo(
    () =>
      chart.series.map((s) =>
        s.points.length >= SMOOTH_MIN_POINTS ? emaSmooth(s.points) : null,
      ),
    [chart.series],
  );
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
              tick={{ fill: "#cbd5e1", fontSize: 11 }}
            />
            <YAxis
              width={58}
              type="number"
              domain={yDomain}
              allowDataOverflow
              tickFormatter={(v: number) => v.toFixed(2)}
              stroke="#475569"
              tick={{ fill: "#cbd5e1", fontSize: 11 }}
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
            {chart.series.map((s, i) => {
              const color = SERIES_COLORS[i % SERIES_COLORS.length];
              const smooth = smoothed[i];
              return smooth ? (
                // Faint raw trace behind the smoothed line; only the
                // smoothed line participates in legend and tooltip.
                [
                  <Line
                    key={`${s.run}-raw`}
                    data={s.points}
                    dataKey="y"
                    legendType="none"
                    tooltipType="none"
                    stroke={color}
                    strokeWidth={1}
                    strokeOpacity={0.25}
                    dot={false}
                    isAnimationActive={false}
                  />,
                  <Line
                    key={s.run}
                    data={smooth}
                    dataKey="y"
                    name={seriesName(s.run, s.state)}
                    stroke={color}
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />,
                ]
              ) : (
                <Line
                  key={s.run}
                  data={s.points}
                  dataKey="y"
                  name={seriesName(s.run, s.state)}
                  stroke={color}
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              );
            })}
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
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2 xl:grid-cols-3">
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
