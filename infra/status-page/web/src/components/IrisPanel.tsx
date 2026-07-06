import { useIris } from "../hooks/useIris";
import { formatDuration, formatRelative } from "./chartUtils";
import { ControlPlanePanel } from "./ControlPlanePanel";
import { JobsPanel } from "./JobsPanel";
import { WandbPanel } from "./WandbPanel";
import { WorkersPanel } from "./WorkersPanel";

function percentileTitle(spanMs: number, count: number): string {
  if (spanMs > 0) return `over last ${formatDuration(spanMs)}, n=${count}`;
  return `n=${count}`;
}

export function IrisPanel() {
  const { data, isLoading, error, dataUpdatedAt } = useIris();

  return (
    <section>
      <div className="mb-3 flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h2 className="text-xl font-semibold text-slate-200">Iris</h2>
        {data && (
          <>
            <span
              className={`h-3 w-3 shrink-0 translate-y-0.5 rounded-full ${data.reachable ? "bg-emerald-500" : "bg-rose-500"}`}
            />
            <span className="text-sm text-slate-300">{data.cluster}</span>
            <span className="text-sm text-slate-400">
              {data.reachable ? "reachable" : "unreachable"}
            </span>
            {data.latencyMs !== null && (
              <span
                className={`text-sm ${data.latencyMs > 20 ? "text-rose-400" : "text-slate-500"}`}
              >
                · {data.latencyMs}ms
              </span>
            )}
            {data.pingPercentiles && (
              <span
                className="text-xs text-slate-500"
                title={percentileTitle(data.pingSpanMs, data.pingSampleCount)}
              >
                · p50 {data.pingPercentiles.p50}ms · p90 {data.pingPercentiles.p90}ms · p99{" "}
                {data.pingPercentiles.p99}ms
              </span>
            )}
            {data.controllerUrl && (
              <span className="font-mono text-xs text-slate-500">· {data.controllerUrl}</span>
            )}
            <a
              href="https://iris.oa.dev"
              target="_blank"
              rel="noreferrer"
              className="text-xs text-slate-500 hover:text-emerald-300"
            >
              · iris.oa.dev ↗
            </a>
          </>
        )}
        <span className="ml-auto text-xs text-slate-500">
          {dataUpdatedAt ? `updated ${formatRelative(new Date(dataUpdatedAt).toISOString())}` : ""}
        </span>
      </div>
      <div className="space-y-4">
        {isLoading && <div className="text-slate-400">loading…</div>}
        {error && (
          <div className="text-rose-400">failed to load: {(error as Error).message}</div>
        )}
        {data?.error && <div className="text-sm text-rose-400">{data.error}</div>}
        <WorkersPanel />
        <WandbPanel />
        <ControlPlanePanel />
        <JobsPanel />
      </div>
    </section>
  );
}
