import type { JobStateCount } from "../api";
import { useJobs } from "../hooks/useJobs";

// Emerald = success, rose = broken, amber = in-flight, slate = cancelled
// / terminal non-issue. Unknown states fall through as slate so new enum
// values don't crash rendering.
const STATE_COLORS: Record<string, string> = {
  succeeded: "bg-emerald-500",
  running: "bg-amber-400",
  building: "bg-amber-500",
  pending: "bg-slate-500",
  killed: "bg-slate-600",
  failed: "bg-rose-500",
  worker_failed: "bg-rose-600",
  unschedulable: "bg-rose-400",
  unspecified: "bg-slate-700",
};

function stateColor(name: string): string {
  return STATE_COLORS[name] ?? "bg-slate-600";
}

function StateRow({ row, total }: { row: JobStateCount; total: number }) {
  const pct = total > 0 ? (row.count / total) * 100 : 0;
  return (
    <div className="flex items-center gap-3 text-sm">
      <span className="w-28 shrink-0 text-slate-400">{row.name}</span>
      <div className="relative h-4 flex-1 overflow-hidden rounded bg-slate-800">
        <div
          className={`h-full ${stateColor(row.name)}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="w-12 shrink-0 text-right font-mono text-xs text-slate-300">
        {row.count}
      </span>
      <span className="w-10 shrink-0 text-right text-xs text-slate-500">
        {pct.toFixed(0)}%
      </span>
    </div>
  );
}

export function JobsPanel() {
  const { data, isLoading, error } = useJobs();

  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
          Jobs
        </h3>
        <span className="text-xs text-slate-500">submitted in the last 24h</span>
      </div>
      <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-4">
        {isLoading && <div className="text-slate-400">loading…</div>}
        {error && (
          <div className="text-rose-400">failed to load: {(error as Error).message}</div>
        )}
        {data?.error && <div className="text-sm text-rose-400">{data.error}</div>}
        {data && !data.error && (
          <>
            <div className="flex items-baseline gap-3">
              <span className="text-4xl font-bold text-slate-100">{data.total}</span>
              <span className="text-slate-400">total</span>
            </div>
            {data.byState.length === 0 ? (
              <div className="mt-4 text-sm text-slate-500">no jobs in the last 24h</div>
            ) : (
              <div className="mt-4 space-y-2">
                {data.byState.map((row) => (
                  <StateRow key={row.state} row={row} total={data.total} />
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
