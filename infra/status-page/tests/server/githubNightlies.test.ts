import assert from "node:assert/strict";
import test from "node:test";
import { fetchNightlyAttempt, fetchNightlyLane } from "../../server/sources/githubNightlies.js";
import type { NightlyLaneConfig } from "../../server/sources/nightlyConfig.js";

const LANE: NightlyLaneConfig = {
  id: "fork",
  label: "Fork nightly",
  shortLabel: "Fork",
  group: "forks",
  subgroup: "inference",
  repository: "marin-community/example-fork",
  workflowFile: "marin nightly.yaml",
  branch: "main",
  schedule: { weekdays: [0, 1, 2, 3, 4, 5, 6], hour: 10, minute: 0 },
  activeFrom: "2026-07-15",
  overdueGraceMinutes: 240,
  overdueGraceProvenance: "test policy",
};

const GH_RUN = {
  id: 42,
  status: "completed",
  conclusion: "success",
  head_sha: "abcdef1234567890",
  created_at: "2026-07-17T11:00:00.000Z",
  run_started_at: "2026-07-17T11:01:00.000Z",
  updated_at: "2026-07-17T11:10:00.000Z",
  html_url: "https://github.com/marin-community/example-fork/actions/runs/42",
  run_attempt: 2,
  event: "schedule",
  head_branch: "main",
  actor: { login: "github-actions" },
};

test("lane fetch uses repository-qualified bounded scheduled-run query", async () => {
  let requestedUrl = "";
  const request: typeof fetch = async (input) => {
    requestedUrl = String(input);
    return Response.json({ workflow_runs: [GH_RUN] });
  };

  const snapshot = await fetchNightlyLane(
    LANE,
    new Date("2026-07-17T15:00:00.000Z"),
    request,
  );
  const url = new URL(requestedUrl);

  assert.equal(
    url.pathname,
    "/repos/marin-community/example-fork/actions/workflows/marin%20nightly.yaml/runs",
  );
  assert.equal(url.searchParams.get("event"), "schedule");
  assert.equal(url.searchParams.get("branch"), "main");
  assert.equal(url.searchParams.get("per_page"), "30");
  assert.equal(url.searchParams.get("created"), ">=2026-07-10T00:00:00.000Z");
  assert.equal(snapshot.runs[0].runAttempt, 2);
  assert.equal(snapshot.runs[0].createdAt, GH_RUN.created_at);
});

test("lane fetch returns a lane-local error snapshot", async () => {
  const request: typeof fetch = async () => new Response("rate limited", { status: 403 });
  const snapshot = await fetchNightlyLane(
    LANE,
    new Date("2026-07-17T15:00:00.000Z"),
    request,
  );

  assert.deepEqual(snapshot.runs, []);
  assert.match(snapshot.error ?? "", /^GitHub API 403: rate limited/);
});

test("attempt fetch uses the documented numbered endpoint", async () => {
  let requestedUrl = "";
  const request: typeof fetch = async (input) => {
    requestedUrl = String(input);
    return Response.json({ ...GH_RUN, conclusion: "failure", run_attempt: 1 });
  };

  const result = await fetchNightlyAttempt(LANE, 42, 1, request);

  assert.equal(
    requestedUrl,
    "https://api.github.com/repos/marin-community/example-fork/actions/runs/42/attempts/1",
  );
  assert.equal(result.attempt?.attempt, 1);
  assert.equal(result.attempt?.conclusion, "failure");
});
