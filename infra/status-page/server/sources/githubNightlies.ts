import { githubAuthHeaders } from "./github.js";
import type { NightlyLaneConfig } from "./nightlyConfig.js";
import type {
  NightlyAttempt,
  NightlyLaneSnapshot,
  NightlyRun,
} from "./nightlyProjection.js";

interface GhRun {
  id: number;
  status: string;
  conclusion: string | null;
  head_sha: string;
  created_at: string;
  run_started_at: string | null;
  updated_at: string;
  html_url: string;
  run_attempt?: number;
  event: string;
  head_branch: string | null;
  actor: { login: string } | null;
}

interface GhRunsResponse {
  workflow_runs?: GhRun[];
}

export interface AttemptFetchResult {
  attempt?: NightlyAttempt;
  error?: string;
}

const DAY_MS = 24 * 60 * 60 * 1000;

function queryStart(lane: NightlyLaneConfig, now: Date): Date {
  const today = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate());
  const cadenceDays = lane.schedule.weekdays.length === 7 ? 1 : 7;
  return new Date(today - (6 + cadenceDays) * DAY_MS);
}

function toNightlyRun(run: GhRun): NightlyRun {
  return {
    id: run.id,
    status: run.status,
    conclusion: run.conclusion,
    sha: run.head_sha,
    createdAt: run.created_at,
    runStartedAt: run.run_started_at,
    updatedAt: run.updated_at,
    url: run.html_url,
    runAttempt: run.run_attempt ?? 1,
    event: run.event,
    headBranch: run.head_branch,
    actor: run.actor?.login ?? "unknown",
  };
}

function errorMessage(prefix: string, status: number, body: string): string {
  return `${prefix} ${status}: ${body.slice(0, 200)}`;
}

export async function fetchNightlyLane(
  lane: NightlyLaneConfig,
  now: Date,
  request: typeof fetch = fetch,
): Promise<NightlyLaneSnapshot> {
  const fetchedAt = now.toISOString();
  const params = new URLSearchParams({
    branch: lane.branch,
    event: "schedule",
    per_page: "30",
    created: `>=${queryStart(lane, now).toISOString()}`,
  });
  const url =
    `https://api.github.com/repos/${lane.repository}/actions/workflows/` +
    `${encodeURIComponent(lane.workflowFile)}/runs?${params.toString()}`;

  try {
    const response = await request(url, { headers: githubAuthHeaders() });
    if (!response.ok) {
      const body = await response.text().catch(() => "");
      return {
        laneId: lane.id,
        fetchedAt,
        runs: [],
        error: errorMessage("GitHub API", response.status, body),
      };
    }
    const payload = (await response.json()) as GhRunsResponse;
    return {
      laneId: lane.id,
      fetchedAt,
      runs: (payload.workflow_runs ?? []).map(toNightlyRun),
    };
  } catch (error) {
    return {
      laneId: lane.id,
      fetchedAt,
      runs: [],
      error: `GitHub API fetch failed: ${(error as Error).message}`,
    };
  }
}

export async function fetchNightlyAttempt(
  lane: NightlyLaneConfig,
  runId: number,
  attemptNumber: number,
  request: typeof fetch = fetch,
): Promise<AttemptFetchResult> {
  const url =
    `https://api.github.com/repos/${lane.repository}/actions/runs/${runId}/attempts/` +
    String(attemptNumber);
  try {
    const response = await request(url, { headers: githubAuthHeaders() });
    if (!response.ok) {
      const body = await response.text().catch(() => "");
      return { error: errorMessage("GitHub attempt API", response.status, body) };
    }
    const attempt = (await response.json()) as GhRun;
    return {
      attempt: {
        attempt: attemptNumber,
        status: attempt.status,
        conclusion: attempt.conclusion,
        runStartedAt: attempt.run_started_at,
        updatedAt: attempt.updated_at,
        url: attempt.html_url,
      },
    };
  } catch (error) {
    return { error: `GitHub attempt fetch failed: ${(error as Error).message}` };
  }
}
