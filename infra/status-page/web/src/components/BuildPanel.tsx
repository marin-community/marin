import type { CommitState, CommitStatus } from "../api";
import { useBuilds } from "../hooks/useBuilds";

// Mirrors the status dot in GitHub's commits view: green = all checks
// passed, rose = something failed/errored, amber = still running,
// slate = no checks configured for this commit.
function stateColor(state: CommitState): string {
  switch (state) {
    case "SUCCESS":
      return "bg-emerald-500";
    case "FAILURE":
      return "bg-rose-500";
    case "ERROR":
      return "bg-rose-600";
    case "PENDING":
    case "EXPECTED":
      return "bg-amber-400";
    case "NONE":
    default:
      return "bg-slate-600";
  }
}

function stateLabel(state: CommitState): string {
  switch (state) {
    case "SUCCESS":
      return "success";
    case "FAILURE":
      return "failure";
    case "ERROR":
      return "error";
    case "PENDING":
      return "pending";
    case "EXPECTED":
      return "expected";
    case "NONE":
      return "no checks";
  }
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

function LatestLine({ commit }: { commit: CommitStatus }) {
  return (
    <a
      href={commit.url}
      target="_blank"
      rel="noreferrer"
      className="inline-flex items-baseline gap-2 text-slate-200 hover:text-emerald-300"
    >
      <span className={`h-3 w-3 shrink-0 translate-y-0.5 rounded-full ${stateColor(commit.state)}`} />
      <span className="font-mono text-xs">{commit.shortOid}</span>
      <span className="truncate">{commit.headline}</span>
      <span className="text-slate-500">· {formatRelative(commit.committedAt)}</span>
    </a>
  );
}

export function BuildPanel() {
  const { data, isLoading, error, dataUpdatedAt } = useBuilds();
  const latest = data?.commits?.[0];
  const successRate = data?.successRate ?? null;
  const finalized = (data?.commits ?? []).filter(
    (c) => c.state === "SUCCESS" || c.state === "FAILURE" || c.state === "ERROR",
  ).length;

  return (
    <section>
      <div className="mb-3 flex items-baseline justify-between">
        <h2 className="text-xl font-semibold text-slate-200">GitHub Build</h2>
        <span className="text-xs text-slate-500">
          {dataUpdatedAt ? `updated ${formatRelative(new Date(dataUpdatedAt).toISOString())}` : ""}
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
            <div className="flex flex-wrap items-center gap-3 text-sm">
              {latest ? (
                <LatestLine commit={latest} />
              ) : (
                <span className="text-slate-400">no commits</span>
              )}
              <span className="ml-auto shrink-0 text-slate-400">
                {successRate === null
                  ? "—"
                  : `${Math.round(successRate * 100)}% success over ${finalized}`}
              </span>
            </div>

            {/* Thin dots (w-1.5) so 100 commits fit a sparkline-style
                strip without dominating the card. Each dot links to its
                commit page on GitHub. */}
            <div className="mt-3 flex flex-wrap gap-[3px]">
              {data.commits.map((c) => (
                <a
                  key={c.oid}
                  href={c.url}
                  target="_blank"
                  rel="noreferrer"
                  title={`${c.shortOid} · ${stateLabel(c.state)} · ${formatRelative(c.committedAt)}\n${c.headline}`}
                  className={`h-5 w-1.5 rounded-sm ${stateColor(c.state)} hover:ring-2 hover:ring-slate-400`}
                />
              ))}
            </div>
          </>
        )}
      </div>
    </section>
  );
}
