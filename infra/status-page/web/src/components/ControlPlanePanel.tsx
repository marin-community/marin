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
import type { ServiceHealthSeries, ServiceHealthSummarySample } from "../api";
import { useControlPlaneHealth } from "../hooks/useControlPlaneHealth";
import {
  displayedSpanLabel,
  formatClock,
  formatDuration,
  useContainerSize,
} from "./chartUtils";

const SERIES_COLORS: Record<string, string> = {
  prod_iris: "#10b981",
  prod_finelog: "#06b6d4",
  dev_iris: "#8b5cf6",
  dev_finelog: "#f59e0b",
};

const METRICS = ["p50", "max"] as const;

const METRIC_STYLES: Record<
  (typeof METRICS)[number],
  { strokeDasharray?: string; strokeWidth: number }
> = {
  p50: { strokeWidth: 2.3 },
  max: { strokeDasharray: "6 4", strokeWidth: 2 },
};

function lineKey(seriesId: string, metric: (typeof METRICS)[number]): string {
  return `${seriesId}_${metric}`;
}

function useChartData(samples: ServiceHealthSummarySample[], series: ServiceHealthSeries[]) {
  return useMemo(() => {
    const rows = samples.map((sample) => {
      const row: Record<string, number | null> = { t: sample.t };
      for (const s of series) {
        const stats = sample.stats[s.id];
        for (const metric of METRICS) {
          row[lineKey(s.id, metric)] = stats?.[metric] ?? null;
        }
      }
      return row;
    });
    return rows;
  }, [samples, series]);
}

export function ControlPlanePanel() {
  const { data, isLoading, error } = useControlPlaneHealth();
  const samples = data?.summarySamples ?? [];
  const series = data?.series ?? [];
  const latestErrors = useMemo(
    () => (data?.latest ?? []).filter((snapshot) => !snapshot.reachable || snapshot.error),
    [data?.latest],
  );
  const chartRows = useChartData(samples, series);
  const { ref: chartRef, size: chartSize } = useContainerSize<HTMLDivElement>();
  const aggregationLabel = data?.aggregationWindowMs
    ? `${formatDuration(data.aggregationWindowMs)} rolling`
    : "rolling";
  const pointLabel = data?.summaryPointIntervalMs
    ? `${formatDuration(data.summaryPointIntervalMs)} points`
    : "points";

  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
          /health Latency
        </h3>
        <span className="text-xs text-slate-500">
          {samples.length > 1
            ? `${displayedSpanLabel(samples)} · ${pointLabel} · ${aggregationLabel} · p50/max`
            : "history warming up"}
        </span>
      </div>
      <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-4">
        {isLoading && <div className="text-slate-400">loading…</div>}
        {error && (
          <div className="text-rose-400">failed to load: {(error as Error).message}</div>
        )}
        {data && (
          <>
            {latestErrors.length > 0 && (
              <div className="mb-3 space-y-1 text-xs text-rose-400">
                {latestErrors.map((snapshot) => (
                  <div key={snapshot.id}>
                    {snapshot.name}: {snapshot.error ?? "unreachable"}
                  </div>
                ))}
              </div>
            )}

            <div ref={chartRef} className="h-56 w-full">
              {samples.length > 1 && chartSize && series.length > 0 ? (
                <LineChart
                  width={chartSize.width}
                  height={chartSize.height}
                  data={chartRows}
                  margin={{ top: 4, right: 8, bottom: 4, left: 12 }}
                >
                  <CartesianGrid stroke="#1e293b" strokeDasharray="2 4" />
                  <XAxis
                    dataKey="t"
                    type="number"
                    domain={["dataMin", "dataMax"]}
                    tickFormatter={formatClock}
                    stroke="#475569"
                    fontSize={11}
                  />
                  <YAxis
                    width={58}
                    stroke="#475569"
                    fontSize={11}
                    tickFormatter={(value) => `${value}ms`}
                    label={{
                      value: "Latency (ms)",
                      angle: -90,
                      position: "insideLeft",
                      style: { fill: "#64748b", fontSize: 11 },
                    }}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "#0f172a",
                      border: "1px solid #1e293b",
                      borderRadius: 4,
                      fontSize: 12,
                    }}
                    formatter={(value, name) => [
                      typeof value === "number" ? `${value}ms` : "down",
                      name,
                    ]}
                    labelFormatter={(value) => new Date(value as number).toLocaleString()}
                  />
                  <Legend
                    verticalAlign="bottom"
                    height={20}
                    iconType="plainline"
                    wrapperStyle={{ fontSize: 11, color: "#94a3b8" }}
                  />
                  {series.flatMap((s) =>
                    METRICS.map((metric) => (
                      <Line
                        key={lineKey(s.id, metric)}
                        type="linear"
                        dataKey={lineKey(s.id, metric)}
                        name={`${s.name} ${metric}`}
                        stroke={SERIES_COLORS[s.id] ?? "#94a3b8"}
                        strokeWidth={METRIC_STYLES[metric].strokeWidth}
                        strokeDasharray={METRIC_STYLES[metric].strokeDasharray}
                        dot={false}
                        isAnimationActive={false}
                      />
                    )),
                  )}
                </LineChart>
              ) : (
                <div className="flex h-full items-center justify-center text-sm text-slate-500">
                  history warming up — probes run every 30s, chart shows rolling
                  p50/max once we have two points
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
