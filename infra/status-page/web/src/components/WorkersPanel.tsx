import { useCallback, useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { WorkerResourceTotals, WorkerSample } from "../api";
import { useWorkers } from "../hooks/useWorkers";
import { useWorkersHistory } from "../hooks/useWorkersHistory";

// Palette for per-region chart lines. Picked for good contrast on the
// dark background; cycles if there are more regions than entries.
const REGION_COLORS = [
  "#10b981", // emerald-500
  "#06b6d4", // cyan-500
  "#8b5cf6", // violet-500
  "#f59e0b", // amber-500
  "#ec4899", // pink-500
  "#f43f5e", // rose-500
  "#14b8a6", // teal-500
  "#3b82f6", // blue-500
];

function formatClock(ms: number): string {
  const d = new Date(ms);
  const hh = d.getHours().toString().padStart(2, "0");
  const mm = d.getMinutes().toString().padStart(2, "0");
  return `${hh}:${mm}`;
}

// Turn total_cpu_millicores into a human-readable CPU-core count.
// 1000 millicores = 1 full core. k-suffix for anything above ~10k.
function formatCores(millicores: number): string {
  const cores = millicores / 1000;
  if (cores >= 10000) return `${Math.round(cores / 1000)}k`;
  if (cores >= 1000) return `${(cores / 1000).toFixed(1)}k`;
  return Math.round(cores).toString();
}

// Binary (IEC) byte formatter — KiB / MiB / GiB / TiB. Most Iris workers
// report total_memory_bytes in the hundreds of GiB per node, so the
// aggregate is almost always in the TiB range.
function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KiB", "MiB", "GiB", "TiB", "PiB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 100 ? Math.round(value) : value.toFixed(1)} ${units[unit]}`;
}

// useContainerSize + explicit LineChart pixel dimensions sidesteps
// recharts' ResponsiveContainer, which fires a "width(-1) / height(-1)"
// warning under React StrictMode because its internal effect runs
// before the container's first layout has committed. We own the
// ResizeObserver lifecycle ourselves and only mount the chart once
// we have a non-zero size.
//
// The ref here is a CALLBACK REF, not a RefObject, because the div it
// attaches to lives inside a data-loading conditional. A useRef-based
// approach would run its effect once on first render with ref.current
// still null (div not yet rendered), then never re-run — the observer
// would never get set up. A callback ref stores the element in state,
// which triggers the effect as soon as React mounts the element.
// Rendered inline in the header row next to the big `healthy` count.
// CPU and memory are the currently-unallocated ("free") amounts across
// healthy workers; `chips` is the raw total of TPU chips across healthy
// workers regardless of commitment (iris schedules at whole-VM grain,
// so "available chips" collapses to a tiny and misleading number on a
// busy cluster).
function ResourceLine({ resources }: { resources: WorkerResourceTotals }) {
  return (
    <>
      <span className="text-slate-600">·</span>
      <span className="text-sm text-slate-400">
        <span className="font-mono text-slate-200">
          {formatCores(resources.cpuAvailableMillicores)}
        </span>{" "}
        CPU
      </span>
      <span className="text-slate-600">·</span>
      <span className="text-sm text-slate-400">
        <span className="font-mono text-slate-200">
          {formatBytes(resources.memoryAvailableBytes)}
        </span>{" "}
        memory
      </span>
      <span className="text-slate-600">·</span>
      <span className="text-sm text-slate-400">
        <span className="font-mono text-slate-200">{resources.chipsTotal}</span>{" "}
        chips
      </span>
    </>
  );
}

function useContainerSize<T extends HTMLElement>() {
  const [node, setNode] = useState<T | null>(null);
  const [size, setSize] = useState<{ width: number; height: number } | null>(null);

  const ref = useCallback((el: T | null) => setNode(el), []);

  useEffect(() => {
    if (!node) return;
    const obs = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          setSize({ width, height });
        }
      }
    });
    obs.observe(node);
    return () => obs.disconnect();
  }, [node]);

  return { ref, size };
}

// Flatten `{t, regions: {us-west4: 232, ...}}` samples into
// `{t, "us-west4": 232, ...}` rows that recharts can consume via
// `dataKey={region}` on each <Line>. Also returns the ordered list of
// region names (most-populated first according to the current snapshot,
// with any extra regions that only appear in history appended).
function useChartData(samples: WorkerSample[], currentOrder: string[]) {
  return useMemo(() => {
    const seen = new Set<string>();
    for (const s of samples) {
      for (const r of Object.keys(s.regions)) {
        seen.add(r);
      }
    }
    const ordered: string[] = [];
    for (const r of currentOrder) {
      if (seen.has(r)) {
        ordered.push(r);
        seen.delete(r);
      }
    }
    for (const r of Array.from(seen).sort()) {
      ordered.push(r);
    }
    const rows = samples.map((s) => {
      const row: Record<string, number> = { t: s.t };
      for (const r of ordered) {
        if (s.regions[r] !== undefined) {
          row[r] = s.regions[r];
        }
      }
      return row;
    });
    return { rows, regions: ordered };
  }, [samples, currentOrder]);
}

export function WorkersPanel() {
  const { data, isLoading, error } = useWorkers();
  const history = useWorkersHistory();
  const samples = history.data?.samples ?? [];
  const { ref: chartRef, size: chartSize } = useContainerSize<HTMLDivElement>();
  const currentOrder = useMemo(
    () => (data?.byRegion ?? []).map((r) => r.region),
    [data?.byRegion],
  );
  const { rows: chartRows, regions: chartRegions } = useChartData(samples, currentOrder);

  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
          Workers
        </h3>
        <span className="text-xs text-slate-500">
          {samples.length > 1
            ? `${samples.length} samples · last 24h`
            : "history warming up"}
        </span>
      </div>
      <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-4">
        {isLoading && <div className="text-slate-400">loading…</div>}
        {error && (
          <div className="text-rose-400">failed to load: {(error as Error).message}</div>
        )}
        {data?.error && <div className="text-sm text-rose-400">{data.error}</div>}
        {data && !data.error && (
          <>
            <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
              <span className="text-4xl font-bold text-emerald-300">{data.healthy}</span>
              <ResourceLine resources={data.resources} />
            </div>
            <div ref={chartRef} className="mt-4 h-56 w-full">
              {samples.length > 1 && chartSize && chartRegions.length > 0 ? (
                <LineChart
                  width={chartSize.width}
                  height={chartSize.height}
                  data={chartRows}
                  margin={{ top: 4, right: 8, bottom: 4, left: -16 }}
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
                  <YAxis stroke="#475569" fontSize={11} />
                  <Tooltip
                    contentStyle={{
                      background: "#0f172a",
                      border: "1px solid #1e293b",
                      borderRadius: 4,
                      fontSize: 12,
                    }}
                    labelFormatter={(value) =>
                      new Date(value as number).toLocaleString()
                    }
                  />
                  <Legend
                    verticalAlign="bottom"
                    height={20}
                    iconType="plainline"
                    wrapperStyle={{ fontSize: 11, color: "#94a3b8" }}
                  />
                  {chartRegions.map((region, i) => (
                    <Line
                      key={region}
                      type="monotone"
                      dataKey={region}
                      name={region}
                      stroke={REGION_COLORS[i % REGION_COLORS.length]}
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                      connectNulls
                    />
                  ))}
                </LineChart>
              ) : (
                <div className="flex h-full items-center justify-center text-sm text-slate-500">
                  history warming up — samples collected every 30s, chart appears once
                  we have two points
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
