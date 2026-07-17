import assert from "node:assert/strict";
import test from "node:test";
import { nightliesResponse } from "../../server/nightlies.js";
import type { NightlyLaneConfig } from "../../server/sources/nightlyConfig.js";
import type { NightlyLaneSnapshot } from "../../server/sources/nightlyProjection.js";

const LANE: NightlyLaneConfig = {
  id: "nightly",
  label: "Nightly",
  shortLabel: "Nightly",
  group: "marin",
  subgroup: "training",
  repository: "marin-community/marin",
  workflowFile: "nightly.yaml",
  branch: "main",
  schedule: { weekdays: [0, 1, 2, 3, 4, 5, 6], hour: 10, minute: 0 },
  overdueGraceMinutes: 240,
  overdueGraceProvenance: "test policy",
  expectedDuration: { minSeconds: 300, maxSeconds: 1200, provenance: "test evidence" },
};

test("response enrichment preserves final result when attempt history is unavailable", async () => {
  const source: NightlyLaneSnapshot = {
    laneId: LANE.id,
    fetchedAt: "2026-07-17T15:00:00.000Z",
    runs: [
      {
        id: 42,
        status: "completed",
        conclusion: "success",
        sha: "abcdef1234567890",
        createdAt: "2026-07-17T11:00:00.000Z",
        runStartedAt: "2026-07-17T14:00:00.000Z",
        updatedAt: "2026-07-17T14:10:00.000Z",
        url: "https://github.com/marin-community/marin/actions/runs/42",
        runAttempt: 2,
        event: "schedule",
        headBranch: "main",
        actor: "github-actions",
      },
    ],
  };

  const response = await nightliesResponse(
    [LANE],
    new Date("2026-07-17T15:00:00.000Z"),
    async () => source,
    async () => ({ error: "attempt API unavailable" }),
  );
  const cell = response.rows[0].cells[0];

  assert.equal(cell.run?.conclusion, "success");
  assert.equal(cell.run?.durationSeconds, 600);
  assert.equal(cell.run?.attemptHistoryError, "attempt API unavailable");
  assert.equal(cell.run?.recovered, false);
  assert.deepEqual(response.today, { healthy: 1, due: 1 });
});
