import assert from "node:assert/strict";
import test from "node:test";
import type { NightlyLaneConfig } from "../../server/sources/nightlyConfig.js";
import {
  projectNightlies,
  selectedReruns,
  type NightlyLaneSnapshot,
  type NightlyRun,
} from "../../server/sources/nightlyProjection.js";

const DAILY = [0, 1, 2, 3, 4, 5, 6] as const;

function lane(overrides: Partial<NightlyLaneConfig> = {}): NightlyLaneConfig {
  return {
    id: "lane",
    label: "Lane",
    shortLabel: "Lane",
    group: "marin",
    subgroup: "training",
    repository: "marin-community/marin",
    workflowFile: "nightly.yaml",
    branch: "main",
    schedule: { weekdays: DAILY, hour: 10, minute: 0 },
    overdueGraceMinutes: 240,
    overdueGraceProvenance: "test policy",
    ...overrides,
  };
}

function snapshot(laneId: string, runs: NightlyRun[] = []): NightlyLaneSnapshot {
  return { laneId, fetchedAt: "2026-07-17T15:00:00.000Z", runs };
}

function run(overrides: Partial<NightlyRun> = {}): NightlyRun {
  return {
    id: 1,
    status: "completed",
    conclusion: "success",
    sha: "1234567890abcdef",
    createdAt: "2026-07-17T11:00:00.000Z",
    runStartedAt: "2026-07-17T11:00:00.000Z",
    updatedAt: "2026-07-17T11:10:00.000Z",
    url: "https://github.com/marin-community/marin/actions/runs/1",
    runAttempt: 1,
    event: "schedule",
    headBranch: "main",
    actor: "github-actions",
    ...overrides,
  };
}

test("weekly lane shows one missing occurrence and six quiet non-occurrences", () => {
  const weekly = lane({
    id: "weekly",
    schedule: { weekdays: [1], hour: 1, minute: 0 },
    overdueGraceMinutes: 480,
  });
  const response = projectNightlies(
    [weekly],
    [snapshot(weekly.id)],
    new Date("2026-07-17T15:00:00.000Z"),
  );

  assert.deepEqual(
    response.rows.map((row) => [row.date, row.cells[0].state]),
    [
      ["2026-07-17", "not-scheduled"],
      ["2026-07-16", "not-scheduled"],
      ["2026-07-15", "not-scheduled"],
      ["2026-07-14", "not-scheduled"],
      ["2026-07-13", "missing"],
      ["2026-07-12", "not-scheduled"],
      ["2026-07-11", "not-scheduled"],
    ],
  );
  assert.deepEqual(response.today, { healthy: 0, due: 0 });
});

test("per-lane lifecycle and grace distinguish not introduced from not yet due", () => {
  const newLane = lane({
    id: "new-lane",
    activeFrom: "2026-07-17",
    schedule: { weekdays: DAILY, hour: 7, minute: 30 },
    overdueGraceMinutes: 300,
  });
  const response = projectNightlies(
    [newLane],
    [snapshot(newLane.id)],
    new Date("2026-07-17T08:00:00.000Z"),
  );

  assert.equal(response.rows[0].cells[0].state, "not-yet-due");
  assert.equal(response.rows[0].cells[0].due, false);
  assert.equal(response.rows[1].cells[0].state, "not-introduced");
});

test("too-short success stays green from GitHub but is excluded from health", () => {
  const bounded = lane({
    id: "bounded",
    expectedDuration: { minSeconds: 360, maxSeconds: 900, provenance: "reviewed" },
  });
  const pending = lane({
    id: "pending",
    workflowFile: "pending.yaml",
  });
  const response = projectNightlies(
    [bounded, pending],
    [
      snapshot(bounded.id, [
        run({
          id: 10,
          runStartedAt: "2026-07-17T11:00:00.000Z",
          updatedAt: "2026-07-17T11:01:11.000Z",
        }),
      ]),
      snapshot(pending.id, [run({ id: 11 })]),
    ],
    new Date("2026-07-17T15:00:00.000Z"),
  );

  assert.equal(response.rows[0].cells[0].run?.conclusion, "success");
  assert.equal(response.rows[0].cells[0].durationState, "too-short");
  assert.equal(response.rows[0].cells[0].healthy, false);
  assert.equal(response.rows[0].cells[1].durationState, "baseline-pending");
  assert.equal(response.rows[0].cells[1].healthy, true);
  assert.deepEqual(response.today, { healthy: 1, due: 2 });
});

test("active run cannot be too short but becomes slow after exceeding its range", () => {
  const bounded = lane({
    expectedDuration: { minSeconds: 360, maxSeconds: 900, provenance: "reviewed" },
  });
  const early = projectNightlies(
    [bounded],
    [
      snapshot(bounded.id, [
        run({ status: "in_progress", conclusion: null, updatedAt: "2026-07-17T11:00:00.000Z" }),
      ]),
    ],
    new Date("2026-07-17T11:01:00.000Z"),
  );
  const late = projectNightlies(
    [bounded],
    [
      snapshot(bounded.id, [
        run({ status: "in_progress", conclusion: null, updatedAt: "2026-07-17T11:00:00.000Z" }),
      ]),
    ],
    new Date("2026-07-17T11:20:00.000Z"),
  );

  assert.equal(early.rows[0].cells[0].durationState, "normal");
  assert.equal(early.rows[0].cells[0].healthy, false);
  assert.equal(late.rows[0].cells[0].durationState, "slow");
});

test("latest colliding scheduled run wins and preserves failed-then-passed history", () => {
  const nightly = lane({
    expectedDuration: { minSeconds: 300, maxSeconds: 1200, provenance: "reviewed" },
  });
  const earlier = run({ id: 20, createdAt: "2026-07-17T10:30:00.000Z" });
  const recovered = run({
    id: 21,
    createdAt: "2026-07-17T11:00:00.000Z",
    runAttempt: 2,
  });
  const source: NightlyLaneSnapshot = {
    ...snapshot(nightly.id, [earlier, recovered]),
    attemptsByRunId: {
      "21": [
        {
          attempt: 1,
          status: "completed",
          conclusion: "failure",
          runStartedAt: "2026-07-17T10:35:00.000Z",
          updatedAt: "2026-07-17T10:40:00.000Z",
          url: recovered.url,
        },
      ],
    },
  };
  const response = projectNightlies(
    [nightly],
    [source],
    new Date("2026-07-17T15:00:00.000Z"),
  );
  const cell = response.rows[0].cells[0];

  assert.equal(cell.run?.id, 21);
  assert.equal(cell.run?.recovered, true);
  assert.deepEqual(cell.collidingRunUrls, [earlier.url]);
  assert.deepEqual([...selectedReruns(response).get(nightly.id) ?? []], [21]);
});

test("due source failure is unavailable while pre-grace failure remains not yet due", () => {
  const nightly = lane();
  const failedSource: NightlyLaneSnapshot = {
    ...snapshot(nightly.id),
    error: "GitHub API unavailable",
  };

  const beforeGrace = projectNightlies(
    [nightly],
    [failedSource],
    new Date("2026-07-17T12:00:00.000Z"),
  );
  const afterGrace = projectNightlies(
    [nightly],
    [failedSource],
    new Date("2026-07-17T15:00:00.000Z"),
  );

  assert.equal(beforeGrace.rows[0].cells[0].state, "not-yet-due");
  assert.equal(afterGrace.rows[0].cells[0].state, "unavailable");
  assert.deepEqual(afterGrace.today, { healthy: 0, due: 1 });
});
