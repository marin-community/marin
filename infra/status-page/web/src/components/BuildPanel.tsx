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

// Simple SVG crown rendered as an absolutely-positioned overlay on top
// of the avatar. Gold peaks with red and blue "jewels" at the base.
// The SVG lives inside a <div> wrapper because Tailwind v4 emits the
// newer standalone CSS `translate` / `rotate` properties which can be
// flaky on SVG roots in some browsers — applying them to a regular
// block wrapper is much more reliable.
// Tilted slightly to the left so it reads as jauntily worn rather than
// perfectly symmetric.
// Simple SVG crown rendered as an absolutely-positioned overlay on top
// of the avatar. Gold peaks with red and blue "jewels" at the base.
// The SVG lives inside a <div> wrapper because Tailwind v4 emits the
// newer standalone CSS `translate` / `rotate` properties which can be
// flaky on SVG roots in some browsers — applying them to a regular
// block wrapper is much more reliable.
// `origin-bottom` pins the rotation origin to the bottom edge of the
// crown (where it meets the head), so the base stays centered over the
// avatar and only the peaks tip to the left. That reads as "worn at an
// angle" rather than "off-center".
function Crown() {
  return (
    <div
      aria-hidden="true"
      className="pointer-events-none absolute -top-5 left-1/2 h-6 w-8 origin-bottom -translate-x-1/2 -rotate-12 scale-[0.8] drop-shadow"
    >
      <svg viewBox="0 0 24 18" className="h-full w-full">
        <path
          d="M2 15 L2 7 L7 11 L12 3 L17 11 L22 7 L22 15 Z"
          fill="#fbbf24"
          stroke="#92400e"
          strokeWidth="1"
          strokeLinejoin="round"
        />
        <rect x="2" y="14" width="20" height="2" fill="#b45309" />
        <circle cx="12" cy="3" r="1" fill="#ef4444" />
        <circle cx="2" cy="7" r="0.9" fill="#3b82f6" />
        <circle cx="22" cy="7" r="0.9" fill="#3b82f6" />
      </svg>
    </div>
  );
}

// Poop emoji shown when the latest commit's checks failed or errored.
// Rendered as text so it picks up the OS / browser's native emoji font.
// Same `origin-bottom -rotate-12` as the Crown so both decorations
// share the "pivoting from the base" tilt.
function FailureMark() {
  return (
    <span
      aria-hidden="true"
      className="pointer-events-none absolute -top-5 left-1/2 origin-bottom -translate-x-1/2 -rotate-12 text-2xl leading-none drop-shadow"
    >
      💩
    </span>
  );
}

// Pick the decoration for the avatar based on the latest commit's
// rollup state. Only SUCCESS (crown) and FAILURE/ERROR (poop) are
// decorated — everything else (pending, no checks, unknown) shows a
// plain avatar.
function AvatarDecoration({ state }: { state: CommitState }) {
  switch (state) {
    case "SUCCESS":
      return <Crown />;
    case "FAILURE":
    case "ERROR":
      return <FailureMark />;
    default:
      return null;
  }
}

// Static avatar beside the commit line with a state-driven decoration
// on top (crown / loading bar / poop).
function CommitAvatar({ commit }: { commit: CommitStatus }) {
  if (!commit.authorAvatarUrl) {
    return null;
  }
  return (
    <span className="relative inline-block">
      <img
        src={commit.authorAvatarUrl}
        alt={commit.author}
        className="h-6 w-6 rounded-full ring-1 ring-slate-700"
      />
      <AvatarDecoration state={commit.state} />
    </span>
  );
}

function LatestLine({ commit }: { commit: CommitStatus }) {
  return (
    <a
      href={commit.url}
      target="_blank"
      rel="noreferrer"
      className="inline-flex items-center gap-2 text-slate-200 hover:text-emerald-300"
    >
      <CommitAvatar commit={commit} />
      <span className={`h-3 w-3 shrink-0 rounded-full ${stateColor(commit.state)}`} />
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

            {/* Each dot uses flex-1 so the strip stretches to fill the
                full card width, giving equal space to every commit
                regardless of how wide the card is. */}
            <div className="mt-3 flex gap-[3px]">
              {data.commits.map((c) => (
                <a
                  key={c.oid}
                  href={c.url}
                  target="_blank"
                  rel="noreferrer"
                  title={`${c.shortOid} · ${stateLabel(c.state)} · ${formatRelative(c.committedAt)}\n${c.headline}`}
                  className={`h-5 flex-1 rounded-sm ${stateColor(c.state)} hover:ring-2 hover:ring-slate-400`}
                />
              ))}
            </div>
          </>
        )}
      </div>
    </section>
  );
}
