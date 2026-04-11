import { useIris } from "../hooks/useIris";
import { JobsPanel } from "./JobsPanel";
import { WorkersPanel } from "./WorkersPanel";

function formatRelative(iso: string): string {
  const delta = Date.now() - Date.parse(iso);
  if (!Number.isFinite(delta)) return iso;
  const seconds = Math.round(delta / 1000);
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.round(seconds / 60);
  return `${minutes}m ago`;
}

export function IrisPanel() {
  const { data, isLoading, error, dataUpdatedAt } = useIris();

  return (
    <section>
      <div className="mb-3 flex items-baseline justify-between">
        <h2 className="text-xl font-semibold text-slate-200">Iris</h2>
        <span className="text-xs text-slate-500">
          {dataUpdatedAt ? `updated ${formatRelative(new Date(dataUpdatedAt).toISOString())}` : ""}
        </span>
      </div>
      <div className="space-y-4">
        {isLoading && <div className="text-slate-400">loading…</div>}
        {error && (
          <div className="text-rose-400">failed to load: {(error as Error).message}</div>
        )}
        {data && (
          <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-4">
            <div className="flex flex-wrap items-center gap-3">
              <span
                className={`h-3 w-3 rounded-full ${data.reachable ? "bg-emerald-500" : "bg-rose-500"}`}
              />
              <span className="text-lg font-semibold">{data.cluster}</span>
              <span className="text-slate-400">
                {data.reachable ? "reachable" : "unreachable"}
              </span>
              {data.latencyMs !== null && (
                <span className="text-slate-500">· {data.latencyMs}ms</span>
              )}
              {data.controllerUrl && (
                <span className="font-mono text-xs text-slate-500">· {data.controllerUrl}</span>
              )}
            </div>
            {data.error && <div className="mt-2 text-sm text-rose-400">{data.error}</div>}
          </div>
        )}
        <WorkersPanel />
        <JobsPanel />
      </div>
    </section>
  );
}
