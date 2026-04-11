// GitHub Actions workflow-run status for the Ferry panel (canary/smoke
// workflows). The Build panel uses a different source
// (server/sources/githubCommits.ts) because it needs per-commit rollup
// state, not per-workflow run history.
//
// The repo is public, so GITHUB_TOKEN is only used to lift the rate limit
// from 60/hr (unauth, per egress IP) to 5000/hr (authenticated). It grants
// no extra access.

import { githubAuthHeaders, REPO } from "./github.js";

export const FERRY_WORKFLOWS = [
  { name: "Canary ferry", file: "marin-canary-ferry.yaml" },
  { name: "Datakit smoke", file: "marin-datakit-smoke.yaml" },
] as const;

export type WorkflowConfig = (typeof FERRY_WORKFLOWS)[number];

export interface FerryRun {
  id: number;
  conclusion: string | null; // "success" | "failure" | "cancelled" | null when running
  status: string; // "completed" | "in_progress" | "queued"
  sha: string;
  shaShort: string;
  startedAt: string;
  durationSeconds: number | null;
  url: string;
  event: string;
  actor: string;
}

export interface FerryWorkflowStatus {
  name: string;
  file: string;
  latest: FerryRun | null;
  history: FerryRun[];
  successRate: number | null; // [0, 1] over completed runs in the window; null if no completed runs
  fetchedAt: string;
  error?: string;
}

// GitHub's API response shape for the subset of fields we read. Keeping
// this narrow and hand-typed avoids pulling in @octokit just for types.
interface GhRun {
  id: number;
  conclusion: string | null;
  status: string;
  head_sha: string;
  run_started_at: string;
  updated_at: string;
  html_url: string;
  event: string;
  actor: { login: string } | null;
}

interface GhRunsResponse {
  workflow_runs: GhRun[];
}

function toFerryRun(run: GhRun): FerryRun {
  const startedMs = Date.parse(run.run_started_at);
  const updatedMs = Date.parse(run.updated_at);
  const durationSeconds =
    run.status === "completed" && Number.isFinite(startedMs) && Number.isFinite(updatedMs)
      ? Math.max(0, Math.round((updatedMs - startedMs) / 1000))
      : null;
  return {
    id: run.id,
    conclusion: run.conclusion,
    status: run.status,
    sha: run.head_sha,
    shaShort: run.head_sha.slice(0, 7),
    startedAt: run.run_started_at,
    durationSeconds,
    url: run.html_url,
    event: run.event,
    actor: run.actor?.login ?? "unknown",
  };
}

function computeSuccessRate(runs: FerryRun[]): number | null {
  const completed = runs.filter((r) => r.status === "completed" && r.conclusion !== null);
  if (completed.length === 0) return null;
  const successes = completed.filter((r) => r.conclusion === "success").length;
  return successes / completed.length;
}

export async function fetchWorkflowStatus(
  workflow: WorkflowConfig,
  historyWindow: number,
): Promise<FerryWorkflowStatus> {
  const url =
    `https://api.github.com/repos/${REPO}/actions/workflows/${workflow.file}` +
    `/runs?per_page=${historyWindow}&branch=main`;

  const res = await fetch(url, { headers: githubAuthHeaders() });
  const fetchedAt = new Date().toISOString();
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    return {
      name: workflow.name,
      file: workflow.file,
      latest: null,
      history: [],
      successRate: null,
      fetchedAt,
      error: `GitHub API ${res.status}: ${body.slice(0, 200)}`,
    };
  }

  const data = (await res.json()) as GhRunsResponse;
  const history = (data.workflow_runs ?? []).map(toFerryRun);
  return {
    name: workflow.name,
    file: workflow.file,
    latest: history[0] ?? null,
    history,
    successRate: computeSuccessRate(history),
    fetchedAt,
  };
}
