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
import type { ProvisioningHistorySample } from "../api";
import { useProvisioningHistory } from "../hooks/useProvisioningHistory";
import { displayedSpanLabel, formatClock, useContainerSize } from "./chartUtils";

// dataKey for the fleet-average line; distinct from any region name.
const FLEET_KEY = "__fleet__";
const FLEET_COLOR = "#e2e8f0"; // slate-200 — brighter than the region lines
const FALLBACK_REGION_COLOR = "#64748b"; // slate-500, for a region not in the shared map

function pct(ratio: number | null | undefined): number | null {
  return ratio === null || ratio === undefined ? null : Math.round(ratio * 1000) / 10;
}

// Flatten `{t, fleet, regions:{...}}` into rows recharts consumes via a
// dataKey per line, and return the ordered region list (alphabetical, matching
// the worker chart's coloring). Values are percentages so the 0–100% axis and
// tooltip read naturally.
function useChartData(samples: ProvisioningHistorySample[]) {
  return useMemo(() => {
    const seen = new Set<string>();
    for (const s of samples) {
      for (const region of Object.keys(s.regions)) seen.add(region);
    }
    const regions = [...seen].sort();
    const rows = samples.map((s) => {
      const row: Record<string, number | null> = { t: s.t, [FLEET_KEY]: pct(s.fleet) };
      for (const region of regions) row[region] = pct(s.regions[region]);
      return row;
    });
    return { rows, regions };
  }, [samples]);
}

function RatioTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: { name: string; value: number | null; color: string; dataKey: string }[];
  label?: number;
}) {
  if (!active || !payload?.length) return null;
  const rows = payload.filter((p) => p.value !== null && p.value !== undefined);
  return (
    <div className="rounded border border-slate-800 bg-slate-950 px-2 py-1 text-xs">
      <div className="mb-0.5 text-slate-400">{new Date(label as number).toLocaleString()}</div>
      {rows.map((p) => (
        <div key={p.dataKey} style={{ color: p.color }}>
          {p.name}: {p.value}%
        </div>
      ))}
    </div>
  );
}

// colorByRegion is shared with the sibling worker chart (built off the union
// of both charts' regions) so a region reads the same color in both.
export function ProvisioningHistoryChart({
  colorByRegion,
}: {
  colorByRegion: Map<string, string>;
}) {
  const { data, isLoading, error } = useProvisioningHistory();
  const samples = data?.samples ?? [];
  const { ref: chartRef, size: chartSize } = useContainerSize<HTMLDivElement>();
  const { rows, regions } = useChartData(samples);

  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
          Provisioning success
        </h4>
        <span className="text-xs text-slate-600">
          {samples.length > 1 ? displayedSpanLabel(samples) : "no cycles yet"}
        </span>
      </div>
      <div ref={chartRef} className="h-56 w-full">
        {isLoading && <div className="text-sm text-slate-500">loading…</div>}
        {error && (
          <div className="text-sm text-rose-400">failed to load: {(error as Error).message}</div>
        )}
        {data?.error && <div className="text-sm text-rose-400">{data.error}</div>}
        {!isLoading && !error && !data?.error &&
          (samples.length > 1 && chartSize ? (
            <LineChart
              width={chartSize.width}
              height={chartSize.height}
              data={rows}
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
                domain={[0, 100]}
                tickFormatter={(v) => `${v}%`}
                stroke="#475569"
                fontSize={11}
                label={{
                  value: "Success",
                  angle: -90,
                  position: "insideLeft",
                  style: { fill: "#64748b", fontSize: 11 },
                }}
              />
              <Tooltip content={<RatioTooltip />} />
              <Legend
                verticalAlign="bottom"
                height={20}
                iconType="plainline"
                wrapperStyle={{ fontSize: 11, color: "#94a3b8" }}
              />
              {regions.map((region) => (
                <Line
                  key={region}
                  type="linear"
                  dataKey={region}
                  name={region}
                  stroke={colorByRegion.get(region) ?? FALLBACK_REGION_COLOR}
                  strokeWidth={1.5}
                  dot={false}
                  isAnimationActive={false}
                />
              ))}
              <Line
                type="linear"
                dataKey={FLEET_KEY}
                name="average"
                stroke={FLEET_COLOR}
                strokeWidth={2.5}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          ) : (
            <div className="flex h-full items-center justify-center text-center text-sm text-slate-500">
              waiting for provisioning cycles — the canary rolls one up every 15min
            </div>
          ))}
      </div>
    </div>
  );
}
