import type { NightlyLaneConfig } from "./sources/nightlyConfig.js";
import {
  projectNightlies,
  selectedReruns,
  type NightlyAttempt,
  type NightlyLaneSnapshot,
  type NightlyResponse,
} from "./sources/nightlyProjection.js";

const GITHUB_FETCH_CONCURRENCY = 4;

export interface NightlyAttemptResult {
  attempt?: NightlyAttempt;
  error?: string;
}

export type NightlyLaneLoader = (
  lane: NightlyLaneConfig,
  now: Date,
) => Promise<NightlyLaneSnapshot>;
export type NightlyAttemptLoader = (
  lane: NightlyLaneConfig,
  runId: number,
  attemptNumber: number,
) => Promise<NightlyAttemptResult>;

async function mapWithConcurrency<T, R>(
  values: readonly T[],
  concurrency: number,
  mapper: (value: T) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(values.length);
  let nextIndex = 0;
  const workers = Array.from({ length: Math.min(concurrency, values.length) }, async () => {
    while (nextIndex < values.length) {
      const index = nextIndex;
      nextIndex += 1;
      results[index] = await mapper(values[index]);
    }
  });
  await Promise.all(workers);
  return results;
}

export async function nightliesResponse(
  lanes: readonly NightlyLaneConfig[],
  now: Date,
  loadLane: NightlyLaneLoader,
  loadAttempt: NightlyAttemptLoader,
): Promise<NightlyResponse> {
  const snapshots = await mapWithConcurrency(lanes, GITHUB_FETCH_CONCURRENCY, (lane) =>
    loadLane(lane, now),
  );
  const reruns = selectedReruns(projectNightlies(lanes, snapshots, now));
  const enriched = snapshots.map((snapshot) => ({
    ...snapshot,
    attemptsByRunId: { ...snapshot.attemptsByRunId },
    attemptErrorsByRunId: { ...snapshot.attemptErrorsByRunId },
  }));
  const enrichedByLane = new Map(enriched.map((snapshot) => [snapshot.laneId, snapshot]));
  const attemptTasks = lanes.flatMap((lane) => {
    const runIds = reruns.get(lane.id) ?? new Set<number>();
    const snapshot = enrichedByLane.get(lane.id);
    if (!snapshot) return [];
    return [...runIds].flatMap((runId) => {
      const run = snapshot.runs.find((candidate) => candidate.id === runId);
      if (!run) return [];
      return Array.from({ length: run.runAttempt - 1 }, (_, index) => ({
        lane,
        runId,
        attemptNumber: index + 1,
      }));
    });
  });

  const attemptResults = await mapWithConcurrency(
    attemptTasks,
    GITHUB_FETCH_CONCURRENCY,
    async (task) => ({
      ...task,
      result: await loadAttempt(task.lane, task.runId, task.attemptNumber),
    }),
  );
  for (const { lane, runId, result } of attemptResults) {
    const snapshot = enrichedByLane.get(lane.id);
    if (!snapshot) continue;
    const key = String(runId);
    if (result.attempt) {
      snapshot.attemptsByRunId[key] = [
        ...(snapshot.attemptsByRunId[key] ?? []),
        result.attempt,
      ];
    }
    if (result.error) {
      const prior = snapshot.attemptErrorsByRunId[key];
      snapshot.attemptErrorsByRunId[key] = prior ? `${prior}; ${result.error}` : result.error;
    }
  }

  return projectNightlies(lanes, enriched, now);
}
