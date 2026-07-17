import type { ReactNode } from "react";
import type {
  NightlyCell,
  NightlyDurationState,
  NightlyLane,
  NightlyResponse,
  NightlyRun,
} from "../api";
import { useNightlies } from "../hooks/useNightlies";
import { formatRelative } from "./chartUtils";

const GROUP_LABELS = { marin: "Marin", forks: "Forks" } as const;
const SUBGROUP_LABELS = {
  training: "Training",
  data: "Data",
  cluster: "Cluster",
  evaluation: "Evaluation",
  rl: "RL",
  inference: "Inference",
} as const;

type IconKind =
  | "success"
  | "failure"
  | "cancelled"
  | "running"
  | "missing"
  | "clock"
  | "quiet"
  | "unknown";

function StatusIcon({ kind }: { kind: IconKind }) {
  const common = {
    className: `nightly-status-icon nightly-status-${kind}`,
    viewBox: "0 0 20 20",
    "aria-hidden": true,
  } as const;
  switch (kind) {
    case "success":
      return (
        <svg {...common}>
          <circle cx="10" cy="10" r="8" />
          <path d="m6 10 2.5 2.5L14.5 7" />
        </svg>
      );
    case "failure":
      return (
        <svg {...common}>
          <circle cx="10" cy="10" r="8" />
          <path d="m7 7 6 6m0-6-6 6" />
        </svg>
      );
    case "cancelled":
      return (
        <svg {...common}>
          <circle cx="10" cy="10" r="8" />
          <path d="M6.5 13.5 13.5 6.5" />
        </svg>
      );
    case "running":
      return (
        <svg {...common}>
          <circle cx="10" cy="10" r="7" />
          <path d="M10 5v5l3 2" />
        </svg>
      );
    case "missing":
      return (
        <svg {...common}>
          <rect x="3" y="3" width="14" height="14" rx="3" />
          <path d="M10 6.5v4.5m0 2.5v.1" />
        </svg>
      );
    case "clock":
      return (
        <svg {...common}>
          <circle cx="10" cy="10" r="7" />
          <path d="M10 6v4l2.5 1.5" />
        </svg>
      );
    case "quiet":
      return (
        <svg {...common}>
          <path d="M5 10h10" />
        </svg>
      );
    case "unknown":
      return (
        <svg {...common}>
          <circle cx="10" cy="10" r="8" />
          <path d="M7.8 7.5a2.4 2.4 0 0 1 4.6 1c0 1.7-2.4 1.8-2.4 3.3m0 2v.1" />
        </svg>
      );
  }
}

function compactDuration(seconds: number | null): string {
  if (seconds === null) return "—";
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h${String(minutes % 60).padStart(2, "0")}`;
}

function spokenDuration(seconds: number | null): string {
  if (seconds === null) return "duration unavailable";
  if (seconds < 60) return `${seconds} ${seconds === 1 ? "second" : "seconds"}`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} ${minutes === 1 ? "minute" : "minutes"}`;
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  const hourText = `${hours} ${hours === 1 ? "hour" : "hours"}`;
  if (remainingMinutes === 0) return hourText;
  return `${hourText} ${remainingMinutes} ${remainingMinutes === 1 ? "minute" : "minutes"}`;
}

function durationDescription(state: NightlyDurationState): string {
  switch (state) {
    case "too-short":
      return "suspiciously short";
    case "normal":
      return "within expected range";
    case "slow":
      return "slow";
    case "very-slow":
      return "very slow";
    case "baseline-pending":
      return "baseline pending";
    case "not-applicable":
      return "duration unavailable";
  }
}

function runStatus(run: NightlyRun): { icon: IconKind; label: string; tone: string } {
  if (run.status !== "completed") {
    return { icon: "running", label: run.status.replaceAll("_", " "), tone: "text-sky-300" };
  }
  switch (run.conclusion) {
    case "success":
      return { icon: "success", label: "GitHub success", tone: "text-emerald-300" };
    case "failure":
      return { icon: "failure", label: "GitHub failure", tone: "text-rose-300" };
    case "cancelled":
    case "timed_out":
      return {
        icon: "cancelled",
        label: `GitHub ${run.conclusion.replaceAll("_", " ")}`,
        tone: "text-orange-300",
      };
    default:
      return {
        icon: "unknown",
        label: `GitHub ${run.conclusion ?? run.status}`,
        tone: "text-slate-300",
      };
  }
}

function emptyStatus(cell: NightlyCell): { icon: IconKind; label: string; tone: string } {
  switch (cell.state) {
    case "not-scheduled":
      return { icon: "quiet", label: "not scheduled", tone: "text-slate-700" };
    case "not-introduced":
      return { icon: "quiet", label: "not introduced", tone: "text-slate-600" };
    case "retired":
      return { icon: "quiet", label: "retired", tone: "text-slate-600" };
    case "not-yet-due":
      return { icon: "clock", label: "not yet due", tone: "text-slate-500" };
    case "missing":
      return { icon: "missing", label: "scheduled run missing", tone: "text-rose-300" };
    case "unavailable":
      return { icon: "unknown", label: "GitHub data unavailable", tone: "text-slate-300" };
    case "run":
      throw new Error("run cells use runStatus");
  }
}

function durationClass(state: NightlyDurationState): string {
  switch (state) {
    case "too-short":
      return "nightly-duration-short";
    case "slow":
      return "nightly-duration-slow";
    case "very-slow":
      return "nightly-duration-very-slow";
    case "baseline-pending":
      return "nightly-duration-pending";
    case "normal":
    case "not-applicable":
      return "nightly-duration-normal";
  }
}

function expectedRange(lane: NightlyLane): string {
  if (!lane.expectedDuration) return "baseline pending";
  return `${compactDuration(lane.expectedDuration.minSeconds)}–${compactDuration(lane.expectedDuration.maxSeconds)}`;
}

function spokenExpectedRange(lane: NightlyLane): string {
  if (!lane.expectedDuration) return "baseline pending";
  return `${spokenDuration(lane.expectedDuration.minSeconds)} to ${spokenDuration(lane.expectedDuration.maxSeconds)}`;
}

function statusClass(cell: NightlyCell): string {
  if (cell.state === "missing") return "nightly-status-problem";
  if (!cell.run || cell.run.status !== "completed") return "";
  if (cell.run.conclusion === "failure") return "nightly-status-problem";
  if (cell.run.conclusion === "cancelled" || cell.run.conclusion === "timed_out") {
    return "nightly-status-warning";
  }
  return "";
}

function CellFrame({
  cell,
  lane,
  children,
}: {
  cell: NightlyCell;
  lane: NightlyLane;
  children: ReactNode;
}) {
  const suspicious = cell.run?.conclusion === "success" && cell.durationState === "too-short";
  return (
    <div
      className={`nightly-cell-frame ${durationClass(cell.durationState)} ${statusClass(cell)} ${suspicious ? "nightly-suspicious" : ""}`}
    >
      {cell.run?.recovered && <span className="nightly-recovered" aria-hidden="true" />}
      {children}
      <span className="sr-only">Expected duration: {spokenExpectedRange(lane)}.</span>
    </div>
  );
}

function NightlyDataCell({
  cell,
  lane,
  dateHeaderId,
  boundary,
}: {
  cell: NightlyCell;
  lane: NightlyLane;
  dateHeaderId: string;
  boundary?: "group" | "subgroup";
}) {
  const headerIds = `${dateHeaderId} group-${lane.group} subgroup-${lane.subgroup} lane-${lane.id}`;
  if (!cell.run) {
    const status = emptyStatus(cell);
    const description = `${lane.label}, ${cell.date}: ${status.label}`;
    return (
      <td
        headers={headerIds}
        className={`nightly-data-cell ${boundary ? `nightly-${boundary}-start` : ""}`}
      >
        <CellFrame cell={cell} lane={lane}>
          <div className={`nightly-cell-content ${status.tone}`}>
            <StatusIcon kind={status.icon} />
            <span className="nightly-duration" aria-hidden="true">—</span>
            <span className="sr-only">{description}.</span>
          </div>
        </CellFrame>
      </td>
    );
  }

  const status = runStatus(cell.run);
  const duration = compactDuration(cell.run.durationSeconds);
  const recovery = cell.run.recovered ? "; failed then passed" : "";
  const description = `${lane.label}, ${cell.date}: ${status.label}; ${spokenDuration(cell.run.durationSeconds)}; ${durationDescription(cell.durationState)}${recovery}; expected ${spokenExpectedRange(lane)}`;
  return (
    <td
      headers={headerIds}
      className={`nightly-data-cell ${boundary ? `nightly-${boundary}-start` : ""}`}
    >
      <CellFrame cell={cell} lane={lane}>
        <details className="nightly-details">
          <summary className={`nightly-cell-content ${status.tone}`} aria-label={description}>
            <StatusIcon kind={status.icon} />
            <span className="nightly-duration">{duration}</span>
          </summary>
          <div className="nightly-detail-card">
            <div className="font-semibold text-slate-100">{lane.label}</div>
            <div>{status.label}</div>
            <div>
              {duration} · {durationDescription(cell.durationState)} · expected {expectedRange(lane)}
            </div>
            <div>{lane.scheduleLabel}</div>
            <div className="font-mono text-[10px] text-slate-400">{cell.run.shaShort}</div>
            {cell.run.recovered && <div className="text-rose-300">failed → passed</div>}
            {cell.run.attemptHistoryError && (
              <div className="text-slate-400">attempt history unavailable</div>
            )}
            {(cell.collidingRunUrls?.length ?? 0) > 0 && (
              <div className="text-amber-300">
                <div>multiple scheduled runs; latest shown</div>
                {cell.collidingRunUrls?.map((url, index) => (
                  <a
                    key={url}
                    href={url}
                    target="_blank"
                    rel="noreferrer"
                    className="block text-sky-300 underline-offset-2 hover:underline"
                  >
                    Earlier scheduled run {index + 1}
                  </a>
                ))}
              </div>
            )}
            {(lane.expectedDuration?.evidenceUrls?.length ?? 0) > 0 && (
              <div>
                Range evidence:{" "}
                {lane.expectedDuration?.evidenceUrls?.map((url, index) => (
                  <span key={url}>
                    {index > 0 && ", "}
                    <a
                      href={url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-sky-300 underline-offset-2 hover:underline"
                    >
                      {index + 1}
                    </a>
                  </span>
                ))}
              </div>
            )}
            <a
              href={cell.run.url}
              target="_blank"
              rel="noreferrer"
              className="mt-1 inline-block font-medium text-sky-300 underline-offset-2 hover:underline"
            >
              Open GitHub run
            </a>
          </div>
        </details>
      </CellFrame>
    </td>
  );
}

interface HeaderSpan {
  key: string;
  label: string;
  span: number;
  startsGroup: boolean;
}

function consecutiveHeaders(
  lanes: NightlyLane[],
  keyFor: (lane: NightlyLane) => string,
  labelFor: (lane: NightlyLane) => string,
): HeaderSpan[] {
  const headers: HeaderSpan[] = [];
  for (const lane of lanes) {
    const key = keyFor(lane);
    const last = headers.at(-1);
    if (last?.key === key) {
      last.span += 1;
    } else {
      headers.push({
        key,
        label: labelFor(lane),
        span: 1,
        startsGroup: headers.length > 0,
      });
    }
  }
  return headers;
}

function laneBoundary(
  lanes: NightlyLane[],
  index: number,
): "group" | "subgroup" | undefined {
  const lane = lanes[index];
  const previous = lanes[index - 1];
  if (!previous) return undefined;
  if (previous.group !== lane.group) return "group";
  if (previous.subgroup !== lane.subgroup) return "subgroup";
  return undefined;
}

function Matrix({ data }: { data: NightlyResponse }) {
  const groups = consecutiveHeaders(
    data.lanes,
    (lane) => lane.group,
    (lane) => GROUP_LABELS[lane.group],
  );
  const subgroups = consecutiveHeaders(
    data.lanes,
    (lane) => `${lane.group}-${lane.subgroup}`,
    (lane) => SUBGROUP_LABELS[lane.subgroup],
  );

  return (
    <div className="nightly-table-shell">
      <table className="nightly-table">
        <caption className="sr-only">
          Seven UTC days of scheduled regression status and duration by lane
        </caption>
        <colgroup>
          <col className="nightly-date-column" />
          {data.lanes.map((lane) => (
            <col key={lane.id} />
          ))}
        </colgroup>
        <thead>
          <tr>
            <th rowSpan={3} scope="col" className="nightly-date-heading">
              UTC
            </th>
            {groups.map((group) => (
              <th
                key={group.key}
                id={`group-${group.key}`}
                scope="colgroup"
                colSpan={group.span}
                className={`nightly-group-heading ${group.startsGroup ? "nightly-group-start" : ""}`}
              >
                {group.label}
              </th>
            ))}
          </tr>
          <tr>
            {subgroups.map((group) => (
              <th
                key={group.key}
                id={`subgroup-${group.key.split("-").at(-1)}`}
                scope="colgroup"
                colSpan={group.span}
                className={`nightly-subgroup-heading ${group.startsGroup ? "nightly-subgroup-start" : ""}`}
              >
                {group.label}
              </th>
            ))}
          </tr>
          <tr>
            {data.lanes.map((lane, index) => {
              const boundary = laneBoundary(data.lanes, index);
              return (
                <th
                  key={lane.id}
                  id={`lane-${lane.id}`}
                  scope="col"
                  title={`${lane.label} · ${lane.scheduleLabel}`}
                  className={`nightly-lane-heading ${boundary ? `nightly-${boundary}-start` : ""}`}
                >
                  {lane.shortLabel}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {data.rows.map((row) => {
            const dateHeaderId = `date-${row.date}`;
            return (
              <tr key={row.date}>
                <th id={dateHeaderId} scope="row" className="nightly-date-heading">
                  {row.label}
                </th>
                {row.cells.map((cell, index) => {
                  const lane = data.lanes[index];
                  return (
                    <NightlyDataCell
                      key={lane.id}
                      cell={cell}
                      lane={lane}
                      dateHeaderId={dateHeaderId}
                      boundary={laneBoundary(data.lanes, index)}
                    />
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Legend() {
  return (
    <ul aria-label="Duration legend" className="nightly-legend">
      <li><span className="nightly-legend-swatch nightly-duration-short nightly-suspicious" />short success</li>
      <li><span className="nightly-legend-swatch nightly-duration-slow" />slow</li>
      <li><span className="nightly-legend-swatch nightly-duration-very-slow" />very slow</li>
      <li><span className="nightly-legend-swatch nightly-duration-pending" />pending</li>
    </ul>
  );
}

export function NightlyPanel() {
  const { data, isLoading, error, dataUpdatedAt } = useNightlies();

  return (
    <section aria-labelledby="nightly-heading">
      <div className="mb-2 flex flex-wrap items-end justify-between gap-2">
        <div className="flex items-baseline gap-3">
          <h2 id="nightly-heading" className="text-xl font-semibold text-slate-200">
            Nightly regressions
          </h2>
          {data && (
            <span className="text-sm font-semibold text-slate-300">
              Today: {data.today.healthy}/{data.today.due} healthy
            </span>
          )}
        </div>
        <div className="flex items-center gap-3 text-xs text-slate-400">
          <Legend />
          {dataUpdatedAt > 0 && (
            <span>updated {formatRelative(new Date(dataUpdatedAt).toISOString())}</span>
          )}
        </div>
      </div>
      {isLoading && <div className="text-slate-400">loading…</div>}
      {error && <div className="text-rose-400">failed to load: {(error as Error).message}</div>}
      {data && <Matrix data={data} />}
    </section>
  );
}
