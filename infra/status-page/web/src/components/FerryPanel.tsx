import type { CSSProperties } from "react";
import { useFerry } from "../hooks/useFerry";
import type { FerryRun, FerryWorkflowStatus } from "../api";

// Diagonal gray/red stripe marks cancelled runs — they count as failures
// for success-rate math but carry a distinct cause worth surfacing.
const CANCELLED_STRIPE: CSSProperties = {
  backgroundImage:
    "repeating-linear-gradient(45deg, #64748b 0, #64748b 3px, #f43f5e 3px, #f43f5e 6px)",
};

function runAppearance(run: FerryRun): { className: string; style?: CSSProperties } {
  if (run.status !== "completed") return { className: "bg-amber-400" };
  switch (run.conclusion) {
    case "success":
      return { className: "bg-emerald-500" };
    case "failure":
      return { className: "bg-rose-500" };
    case "cancelled":
      return { className: "", style: CANCELLED_STRIPE };
    default:
      return { className: "bg-slate-600" };
  }
}

// Flag runs whose wall time is at least SLOW_RUN_STDDEV_THRESHOLD standard
// deviations longer than the mean of the preceding successful runs. Uses up
// to SLOW_RUN_MAX samples, but requires at least SLOW_RUN_MIN so the baseline
// doesn't collapse to noise on new/sparse workflows. Successful only —
// failures/cancels/timeouts have unrepresentative wall times (early exits,
// hangs) and would poison the baseline. history[0] is the most recent run,
// so "prior" means higher indices.
const SLOW_RUN_MIN = 3;
const SLOW_RUN_MAX = 7;
const SLOW_RUN_STDDEV_THRESHOLD = 1;

interface SlowRunBaseline {
  threshold: number;
  sampleSize: number;
}

function slowRunBaseline(history: FerryRun[], index: number): SlowRunBaseline | null {
  const priorDurations: number[] = [];
  for (let j = index + 1; j < history.length && priorDurations.length < SLOW_RUN_MAX; j++) {
    const prev = history[j];
    if (prev.conclusion === "success" && prev.durationSeconds !== null) {
      priorDurations.push(prev.durationSeconds);
    }
  }
  if (priorDurations.length < SLOW_RUN_MIN) return null;
  const mean = priorDurations.reduce((a, b) => a + b, 0) / priorDurations.length;
  const variance =
    priorDurations.reduce((s, x) => s + (x - mean) ** 2, 0) / priorDurations.length;
  const stddev = Math.sqrt(variance);
  // σ=0 (all prior durations identical) still yields a valid threshold at
  // the mean — anything strictly slower than a perfectly stable baseline is
  // genuinely anomalous.
  return {
    threshold: mean + SLOW_RUN_STDDEV_THRESHOLD * stddev,
    sampleSize: priorDurations.length,
  };
}

function isSlowRun(history: FerryRun[], index: number): boolean {
  const run = history[index];
  if (run.durationSeconds === null) return false;
  const baseline = slowRunBaseline(history, index);
  return baseline !== null && run.durationSeconds > baseline.threshold;
}

function formatDuration(seconds: number | null): string {
  if (seconds === null) return "—";
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remSec = seconds % 60;
  if (minutes < 60) return `${minutes}m ${remSec}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}

function formatRelative(iso: string): string {
  const delta = Date.now() - Date.parse(iso);
  if (!Number.isFinite(delta)) return iso;
  const seconds = Math.round(delta / 1000);
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 48) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  return `${days}d ago`;
}

function WorkflowCard({ wf }: { wf: FerryWorkflowStatus }) {
  const latest = wf.latest;
  const successRate = wf.successRate;
  // Match the denominator the server uses for `successRate`
  // (githubActions.ts:computeSuccessRate) — completed runs only, not
  // the raw history length which can include queued/in-progress runs.
  const completedCount = wf.history.filter(
    (r) => r.status === "completed" && r.conclusion !== null,
  ).length;
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-4">
      <div className="flex items-baseline justify-between gap-4">
        <h3 className="text-lg font-semibold text-slate-100">{wf.name}</h3>
        <span className="text-xs text-slate-500">{wf.file}</span>
      </div>

      {wf.error ? (
        <div className="mt-3 text-sm text-rose-400">{wf.error}</div>
      ) : (
        <>
          <div className="mt-3 flex flex-wrap items-center gap-3 text-sm">
            {latest ? (
              <a
                href={latest.url}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-2 text-slate-200 hover:text-emerald-300"
              >
                {(() => {
                  const a = runAppearance(latest);
                  return <span className={`h-3 w-3 rounded-full ${a.className}`} style={a.style} />;
                })()}
                <span className="font-mono text-xs">{latest.shaShort}</span>
                <span className="text-slate-400">{formatRelative(latest.startedAt)}</span>
                <span className="text-slate-500">({formatDuration(latest.durationSeconds)})</span>
              </a>
            ) : (
              <span className="text-slate-400">no runs yet</span>
            )}
            <span className="ml-auto text-slate-400">
              {successRate === null
                ? "—"
                : `${Math.round(successRate * 100)}% success over ${completedCount}`}
            </span>
          </div>

          {/* Single-row strip — dots and inter-dot gap shrink on mobile
              so all 30 fit on a ~340px phone content area without
              wrapping to a second row. */}
          <div className="mt-3 flex gap-px sm:gap-1">
            {wf.history.map((run, i) => {
              const a = runAppearance(run);
              const slow = isSlowRun(wf.history, i);
              const baseline = slow ? slowRunBaseline(wf.history, i) : null;
              return (
                <a
                  key={run.id}
                  href={run.url}
                  target="_blank"
                  rel="noreferrer"
                  className={`group relative h-5 w-2 rounded-sm sm:w-2.5 ${a.className} hover:ring-2 hover:ring-slate-400`}
                  style={a.style}
                >
                  {slow && (
                    <span
                      aria-label="slow run"
                      className="pointer-events-none absolute -right-0.5 -top-1 font-bold leading-none text-amber-300"
                      style={{ fontSize: "10px", textShadow: "0 0 2px #0f172a, 0 0 2px #0f172a" }}
                    >
                      !
                    </span>
                  )}
                  <div className="pointer-events-none absolute bottom-full left-1/2 z-10 mb-2 hidden -translate-x-1/2 whitespace-nowrap rounded border border-slate-700 bg-slate-950/95 px-2 py-1 text-xs text-slate-200 shadow-lg group-hover:block">
                    <div className="font-mono text-slate-300">{run.shaShort}</div>
                    <div className="text-slate-400">
                      {run.conclusion ?? run.status} · {formatRelative(run.startedAt)}
                    </div>
                    <div>wall time: {formatDuration(run.durationSeconds)}</div>
                    {slow && baseline !== null && (
                      <div className="text-amber-300">
                        slow · prior {baseline.sampleSize} successful runs mean+1σ ≈{" "}
                        {formatDuration(Math.round(baseline.threshold))}
                      </div>
                    )}
                  </div>
                </a>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

export function FerryPanel() {
  const { data, isLoading, error, dataUpdatedAt } = useFerry();

  return (
    <section>
      <div className="mb-3 flex items-baseline justify-between">
        <h2 className="text-xl font-semibold text-slate-200">Ferries</h2>
        <span className="text-xs text-slate-500">
          {dataUpdatedAt ? `updated ${formatRelative(new Date(dataUpdatedAt).toISOString())}` : ""}
        </span>
      </div>
      {isLoading && <div className="text-slate-400">loading…</div>}
      {error && <div className="text-rose-400">failed to load: {(error as Error).message}</div>}
      {data && (
        <div className="grid gap-3 md:grid-cols-2">
          {data.workflows.map((wf) => (
            <WorkflowCard key={wf.file} wf={wf} />
          ))}
        </div>
      )}
    </section>
  );
}
