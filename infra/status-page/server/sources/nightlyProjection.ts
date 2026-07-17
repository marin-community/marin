import {
  workflowSourceUrl,
  type NightlyLaneConfig,
  type ExpectedDuration,
} from "./nightlyConfig.js";

export type NightlyCellState =
  | "not-scheduled"
  | "not-introduced"
  | "retired"
  | "not-yet-due"
  | "missing"
  | "unavailable"
  | "run";

export type NightlyDurationState =
  | "not-applicable"
  | "baseline-pending"
  | "too-short"
  | "normal"
  | "slow"
  | "very-slow";

export interface NightlyRun {
  id: number;
  status: string;
  conclusion: string | null;
  sha: string;
  createdAt: string;
  runStartedAt: string | null;
  updatedAt: string;
  url: string;
  runAttempt: number;
  event: string;
  headBranch: string | null;
  actor: string;
}

export interface NightlyAttempt {
  attempt: number;
  status: string;
  conclusion: string | null;
  runStartedAt: string | null;
  updatedAt: string;
  url: string;
}

export interface NightlyLaneSnapshot {
  laneId: string;
  fetchedAt: string;
  runs: NightlyRun[];
  attemptsByRunId?: Record<string, NightlyAttempt[]>;
  attemptErrorsByRunId?: Record<string, string>;
  error?: string;
}

export interface NightlyLaneView {
  id: string;
  label: string;
  shortLabel: string;
  group: NightlyLaneConfig["group"];
  subgroup: NightlyLaneConfig["subgroup"];
  repository: string;
  workflowFile: string;
  workflowUrl: string;
  scheduleLabel: string;
  overdueGraceMinutes: number;
  overdueGraceProvenance: string;
  expectedDuration?: ExpectedDuration;
}

export interface NightlyRunView extends NightlyRun {
  shaShort: string;
  durationSeconds: number | null;
  recovered: boolean;
  priorAttempts: NightlyAttempt[];
  attemptHistoryError?: string;
}

export interface NightlyCell {
  laneId: string;
  date: string;
  expectedAt: string | null;
  state: NightlyCellState;
  due: boolean;
  healthy: boolean;
  durationState: NightlyDurationState;
  run?: NightlyRunView;
  sourceFetchedAt?: string;
  sourceError?: string;
  collidingRunUrls?: string[];
}

export interface NightlyRow {
  date: string;
  label: string;
  cells: NightlyCell[];
}

export interface NightlyResponse {
  generatedAt: string;
  lanes: NightlyLaneView[];
  rows: NightlyRow[];
  today: { healthy: number; due: number };
}

const DAY_MS = 24 * 60 * 60 * 1000;
const RECOVERY_CONCLUSIONS = new Set(["failure", "timed_out", "startup_failure"]);

function utcDayStart(date: Date): Date {
  return new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()));
}

function addUtcDays(date: Date, days: number): Date {
  return new Date(date.getTime() + days * DAY_MS);
}

function dateKey(date: Date): string {
  return date.toISOString().slice(0, 10);
}

function expectedAt(lane: NightlyLaneConfig, day: Date): Date | null {
  if (!lane.schedule.weekdays.includes(day.getUTCDay() as 0 | 1 | 2 | 3 | 4 | 5 | 6)) {
    return null;
  }
  return new Date(
    Date.UTC(
      day.getUTCFullYear(),
      day.getUTCMonth(),
      day.getUTCDate(),
      lane.schedule.hour,
      lane.schedule.minute,
    ),
  );
}

function nextExpectedAt(lane: NightlyLaneConfig, expected: Date): Date {
  for (let offset = 1; offset <= 7; offset += 1) {
    const candidate = expectedAt(lane, addUtcDays(utcDayStart(expected), offset));
    if (candidate) return candidate;
  }
  throw new Error(`${lane.id}: schedule has no next occurrence`);
}

function durationState(
  durationSeconds: number | null,
  range: ExpectedDuration | undefined,
  completed: boolean,
): NightlyDurationState {
  if (durationSeconds === null) return "not-applicable";
  if (!range) return "baseline-pending";
  if (completed && durationSeconds < range.minSeconds) return "too-short";
  if (durationSeconds <= range.maxSeconds) return "normal";
  if (durationSeconds > range.maxSeconds * 1.5) return "very-slow";
  return "slow";
}

function runDuration(run: NightlyRun, now: Date): number | null {
  if (!run.runStartedAt) return null;
  const start = Date.parse(run.runStartedAt);
  const end = run.status === "completed" ? Date.parse(run.updatedAt) : now.getTime();
  if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
  return Math.max(0, Math.round((end - start) / 1000));
}

function isSuccessfulRun(run: NightlyRun): boolean {
  return run.status === "completed" && run.conclusion === "success";
}

function scheduleLabel(lane: NightlyLaneConfig): string {
  const time = `${String(lane.schedule.hour).padStart(2, "0")}:${String(lane.schedule.minute).padStart(2, "0")} UTC`;
  if (lane.schedule.weekdays.length === 7) return `Daily ${time}`;
  const weekdays = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  return `${lane.schedule.weekdays.map((day) => weekdays[day]).join(", ")} ${time}`;
}

function rowLabel(day: Date): string {
  return new Intl.DateTimeFormat("en", {
    timeZone: "UTC",
    weekday: "short",
    month: "short",
    day: "numeric",
  }).format(day);
}

function runCell(
  lane: NightlyLaneConfig,
  date: string,
  expected: Date,
  snapshot: NightlyLaneSnapshot,
  candidates: NightlyRun[],
  now: Date,
): NightlyCell {
  const sorted = [...candidates].sort(
    (left, right) => Date.parse(right.createdAt) - Date.parse(left.createdAt),
  );
  const run = sorted[0];
  const durationSeconds = runDuration(run, now);
  const runDurationState = durationState(
    durationSeconds,
    lane.expectedDuration,
    run.status === "completed",
  );
  const priorAttempts = snapshot.attemptsByRunId?.[String(run.id)] ?? [];
  const recovered =
    isSuccessfulRun(run) &&
    priorAttempts.some((attempt) =>
      attempt.conclusion ? RECOVERY_CONCLUSIONS.has(attempt.conclusion) : false,
    );
  const healthy = isSuccessfulRun(run) && runDurationState !== "too-short";

  return {
    laneId: lane.id,
    date,
    expectedAt: expected.toISOString(),
    state: "run",
    due: true,
    healthy,
    durationState: runDurationState,
    run: {
      ...run,
      shaShort: run.sha.slice(0, 7),
      durationSeconds,
      recovered,
      priorAttempts,
      attemptHistoryError: snapshot.attemptErrorsByRunId?.[String(run.id)],
    },
    sourceFetchedAt: snapshot.fetchedAt,
    collidingRunUrls: sorted.slice(1).map((candidate) => candidate.url),
  };
}

function emptyCell(
  lane: NightlyLaneConfig,
  date: string,
  expected: Date | null,
  snapshot: NightlyLaneSnapshot,
  now: Date,
): NightlyCell {
  const base = {
    laneId: lane.id,
    date,
    expectedAt: expected?.toISOString() ?? null,
    healthy: false,
    durationState: "not-applicable" as const,
    sourceFetchedAt: snapshot.fetchedAt,
  };
  if (!expected) return { ...base, state: "not-scheduled", due: false };
  if (lane.activeFrom && date < lane.activeFrom) {
    return { ...base, state: "not-introduced", due: false };
  }
  if (lane.activeUntil && date > lane.activeUntil) {
    return { ...base, state: "retired", due: false };
  }

  const dueAt = expected.getTime() + lane.overdueGraceMinutes * 60 * 1000;
  if (now.getTime() < dueAt) return { ...base, state: "not-yet-due", due: false };
  if (snapshot.error) {
    return {
      ...base,
      state: "unavailable",
      due: true,
      sourceError: snapshot.error,
    };
  }
  return { ...base, state: "missing", due: true };
}

export function projectNightlies(
  lanes: readonly NightlyLaneConfig[],
  snapshots: readonly NightlyLaneSnapshot[],
  now: Date,
): NightlyResponse {
  const snapshotByLane = new Map(snapshots.map((snapshot) => [snapshot.laneId, snapshot]));
  const todayStart = utcDayStart(now);
  const rows: NightlyRow[] = [];

  for (let offset = 0; offset < 7; offset += 1) {
    const day = addUtcDays(todayStart, -offset);
    const date = dateKey(day);
    const cells = lanes.map((lane): NightlyCell => {
      const snapshot = snapshotByLane.get(lane.id) ?? {
        laneId: lane.id,
        fetchedAt: now.toISOString(),
        runs: [],
        error: "No source snapshot",
      };
      const expected = expectedAt(lane, day);
      if (!expected || (lane.activeFrom && date < lane.activeFrom) || (lane.activeUntil && date > lane.activeUntil)) {
        return emptyCell(lane, date, expected, snapshot, now);
      }

      const nextExpected = nextExpectedAt(lane, expected);
      const candidates = snapshot.runs.filter((run) => {
        const created = Date.parse(run.createdAt);
        return created >= expected.getTime() && created < nextExpected.getTime();
      });
      if (candidates.length > 0) {
        return runCell(lane, date, expected, snapshot, candidates, now);
      }
      return emptyCell(lane, date, expected, snapshot, now);
    });
    rows.push({ date, label: rowLabel(day), cells });
  }

  const todayCells = rows[0]?.cells ?? [];
  return {
    generatedAt: now.toISOString(),
    lanes: lanes.map((lane) => ({
      id: lane.id,
      label: lane.label,
      shortLabel: lane.shortLabel,
      group: lane.group,
      subgroup: lane.subgroup,
      repository: lane.repository,
      workflowFile: lane.workflowFile,
      workflowUrl: workflowSourceUrl(lane),
      scheduleLabel: scheduleLabel(lane),
      overdueGraceMinutes: lane.overdueGraceMinutes,
      overdueGraceProvenance: lane.overdueGraceProvenance,
      ...(lane.expectedDuration ? { expectedDuration: lane.expectedDuration } : {}),
    })),
    rows,
    today: {
      healthy: todayCells.filter((cell) => cell.due && cell.healthy).length,
      due: todayCells.filter((cell) => cell.due).length,
    },
  };
}

export function selectedReruns(response: NightlyResponse): Map<string, Set<number>> {
  const selected = new Map<string, Set<number>>();
  for (const row of response.rows) {
    for (const cell of row.cells) {
      if (!cell.run || cell.run.runAttempt <= 1) continue;
      const runIds = selected.get(cell.laneId) ?? new Set<number>();
      runIds.add(cell.run.id);
      selected.set(cell.laneId, runIds);
    }
  }
  return selected;
}
