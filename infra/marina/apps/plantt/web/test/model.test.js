import assert from "node:assert/strict";
import test from "node:test";

import LZString from "lz-string";

import {
  capacitySegments,
  dateBounds,
  decodeImport,
  resolvePlan,
} from "../src/model.js";

const { compressToEncodedURIComponent } = LZString;

const plan = {
  title: "Test plan",
  capacity: [{ name: "Cluster", chip: "H100", chips: 8, from: "2026-09-01", color: "#456" }],
  workstreams: [
    {
      name: "Build",
      tasks: [
        { name: "Train", start: ["date", "2026-09-02"], end: ["days", 4], cluster: "Cluster", chips: 6 },
        {
          name: "Eval",
          start: ["after", "Train", ["days", 2]],
          end: ["days", 3],
          cluster: "Cluster",
          chips: 4,
          deps: ["Train"],
        },
      ],
    },
  ],
};

test("resolvePlan schedules named and lagged starts", () => {
  const resolved = resolvePlan(plan);

  assert.equal(resolved.tasks.get("Train").end.toISOString().slice(0, 10), "2026-09-06");
  assert.equal(resolved.tasks.get("Eval").start.toISOString().slice(0, 10), "2026-09-08");
  assert.equal(resolved.tasks.get("Eval").end.toISOString().slice(0, 10), "2026-09-11");
});

test("resolvePlan rejects scheduling cycles", () => {
  const cyclic = structuredClone(plan);
  cyclic.workstreams[0].tasks[0].start = "Eval";

  assert.throws(() => resolvePlan(cyclic), /Scheduling cycle/);
});

test("capacitySegments reports overlapping demand above capacity", () => {
  const overlapping = structuredClone(plan);
  overlapping.workstreams[0].tasks[1].start = ["date", "2026-09-04"];
  const resolved = resolvePlan(overlapping);
  const segments = capacitySegments(overlapping.capacity[0], resolved, dateBounds(overlapping, resolved));

  assert.equal(segments.some((segment) => segment.used === 10 && segment.available === 8 && segment.over), true);
});

test("decodeImport reads legacy compressed URLs without retaining chart data in the new URL", () => {
  const encoded = compressToEncodedURIComponent(JSON.stringify({ uuid: "legacy", d: plan }));
  const imported = decodeImport(`https://openathena.ai/plantt/?${encoded}#${encoded}`);

  assert.deepEqual(imported, plan);
});

test("resolvePlan rejects dependencies on absent items", () => {
  const invalid = structuredClone(plan);
  invalid.workstreams[0].tasks[1].deps = ["Missing"];

  assert.throws(() => resolvePlan(invalid), /unknown item 'Missing'/);
});
