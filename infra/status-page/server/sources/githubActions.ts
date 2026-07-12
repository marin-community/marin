// GitHub Actions workflow-run status for the Ferry panel (canary/smoke
// workflows). The Build panel uses a different source
// (server/sources/githubCommits.ts) because it needs per-commit rollup
// state, not per-workflow run history.
//
// The repo is public, so GITHUB_TOKEN is only used to lift the rate limit
// from 60/hr (unauth, per egress IP) to 5000/hr (authenticated). It grants
// no extra access.

import { githubAuthHeaders, REPO } from "./github.js";

// A ferry is one card on the dashboard. Most map to a single workflow file;
// the datakit ferry runs three tiers (tier1/2/3) we surface as three strips
// under one card. `label` is the per-tier caption (null for single-tier
// ferries, where the file subtitle identifies the workflow).
export interface FerryTier {
  label: string | null;
  file: string;
}

export interface FerryGroupConfig {
  name: string;
  tiers: FerryTier[];
}

export const FERRY_GROUPS: FerryGroupConfig[] = [
  { name: "Canary ferry", tiers: [{ label: null, file: "marin-canary-ferry.yaml" }] },
  { name: "CW ferry", tiers: [{ label: null, file: "marin-canary-ferry-coreweave.yaml" }] },
  {
    name: "Datakit ferry",
    tiers: [
      { label: "tier1", file: "marin-canary-datakit-tier1.yaml" },
      { label: "tier2", file: "marin-canary-datakit-tier2.yaml" },
      { label: "tier3", file: "marin-canary-datakit-tier3.yaml" },
    ],
  },
];

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

export interface FerryTierStatus {
  label: string | null;
  file: string;
  latest: FerryRun | null;
  history: FerryRun[];
  successRate: number | null; // [0, 1] over completed runs in the strip; null if no completed runs
  fetchedAt: string;
  error?: string;
}

export interface FerryGroupStatus {
  name: string;
  tiers: FerryTierStatus[];
}

// GitHub's API response shape for the subset of fields we read. Keeping
// this narrow and hand-typed avoids pulling in @octokit just for types.
interface GhRun {
  id: number;
  conclusion: string | null;
  status: string;
  head_sha: string;
  created_at: string;
  run_started_at: string | null;
  updated_at: string;
  html_url: string;
  event: string;
  actor: { login: string } | null;
}

interface GhRunsResponse {
  workflow_runs: GhRun[];
}

function toFerryRun(run: GhRun): FerryRun {
  const startedAt = run.run_started_at ?? run.created_at;
  const startedMs = run.run_started_at === null ? Number.NaN : Date.parse(run.run_started_at);
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
    startedAt,
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

export async function fetchTierStatus(
  tier: FerryTier,
  runLimit: number,
): Promise<FerryTierStatus> {
  const fetchedAt = new Date().toISOString();
  // Fetch a fixed number of most-recent runs rather than a time window: the
  // datakit tiers run on different cadences (tier1/2 daily, tier3 weekly), so
  // a shared time window leaves the weekly strip nearly empty. A per-tier run
  // count gives every strip the same number of bars regardless of cadence.
  const params = new URLSearchParams({
    per_page: String(runLimit),
    branch: "main",
  });
  const url =
    `https://api.github.com/repos/${REPO}/actions/workflows/${tier.file}` +
    `/runs?${params.toString()}`;

  // Every failure path returns a snapshot with `error` set instead of
  // throwing, so callers that aggregate multiple tiers with Promise.all
  // can surface one-tier failures in the UI without turning /api/ferry
  // into a 500.
  try {
    const res = await fetch(url, { headers: githubAuthHeaders() });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      return {
        label: tier.label,
        file: tier.file,
        latest: null,
        history: [],
        successRate: null,
        fetchedAt,
        error: `GitHub API ${res.status}: ${body.slice(0, 200)}`,
      };
    }

    const data = (await res.json()) as GhRunsResponse;
    const history = (data.workflow_runs ?? []).slice(0, runLimit).map(toFerryRun);
    return {
      label: tier.label,
      file: tier.file,
      latest: history[0] ?? null,
      history,
      successRate: computeSuccessRate(history),
      fetchedAt,
    };
  } catch (err) {
    return {
      label: tier.label,
      file: tier.file,
      latest: null,
      history: [],
      successRate: null,
      fetchedAt,
      error: `GitHub API fetch failed: ${(err as Error).message}`,
    };
  }
}
