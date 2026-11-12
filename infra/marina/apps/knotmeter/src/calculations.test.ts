import { describe, expect, it } from "vitest";
import { globalParetoFrontier } from "@/lib/charts/pareto-frontier";
import {
  computeIsoXRulerGeometry,
  parsePerfRulers,
  serializePerfRulers,
  nextPerfRulerState,
} from "@/lib/d3-chart/layers/perf-ruler";
import { sliceTimelineByPhase } from "@/components/inference/agentic-point/phase-slice";
import { buildRequestTimelineRows } from "@/components/inference/agentic-point/timeline-rows";
import { inflightUniqueTokens } from "@/components/inference/agentic-point/time-series-math";
import { metricValue, seriesKey, comparisonScope } from "./metrics";
import { type Run, topologyLabel } from "./types";
import {
  interpolateAtServicePoint,
  matchedHistory,
  improvementWaterfall,
} from "./history-math";
import {
  comparisonRuns,
  hierarchicalTimeline,
  fixtureRun,
} from "../tests/frontend-fixtures";

describe("benchmark data boundaries", () => {
  it("groups calibration repeats across dates without mixing baselines or GPU denominators", () => {
    const baseline = fixtureRun(1, "2026-10-05", "GB200", 8, 20, 30);
    const points: Run[] = [1, 2, 6].flatMap((tp) =>
      Array.from({ length: tp === 2 ? 5 : 4 }, (_, i) => ({
        ...baseline,
        key: `calibration-${tp}-${i}`,
        date: `2026-11-${10 + (i % 3)}`,
        source: {
          kind: "private-calibration" as const,
          commit: "source",
          files: {},
          job_id: `/test/job-${tp}-${i}`,
          attempt_uid: `attempt-${tp}-${i}`,
          artifact_uri: `s3://test/${tp}-${i}`,
        },
        measurement_seconds: 900,
        topology: {
          tp,
          dp: tp === 6 ? 2 : 16 / tp,
          ep: tp === 6 ? 12 : 16,
          pp: 1,
        },
        aggregate: { ...baseline.aggregate, tp: tp === 6 ? 12 : 16 },
        calibration: {
          curve_fingerprint: `matched-controls-${tp}`,
          profiling_sent: 3,
          profiling_cancelled: 1,
          profiling_errors: 0,
          caveat: "15-minute calibration",
        },
      })),
    );
    const groups = new Map<string, Run[]>();
    for (const point of points) {
      const key = seriesKey(point);
      groups.set(key, [...(groups.get(key) || []), point]);
    }
    expect([...groups.values()].map((p) => p.length)).toEqual([4, 5, 4]);
    expect(new Set(points.map((p) => p.key)).size).toBe(13);
    expect(topologyLabel(points.at(-1)!)).toBe("TP6 / DP2 / EP12");
    expect(comparisonScope(points[0])).not.toBe(comparisonScope(baseline));
    const histories = matchedHistory(
      [baseline, ...points],
      20,
      "output",
      "total",
    );
    expect(histories).toHaveLength(4);
    for (const history of histories)
      for (const point of history.points)
        expect(new Set(point.sources.map(comparisonScope)).size).toBe(1);
  });
  it("preserves the first profiling request after a large wall-clock origin", () => {
    const timeline = hierarchicalTimeline();
    timeline.requests = timeline.requests.slice(0, 2).map((r, i) => ({
      ...r,
      phase: i ? "profiling" : "warmup",
      start: i ? 51 : 0,
      credit: i ? 50 : 0,
      end: i ? 1000 : 40,
    }));
    const profiling = sliceTimelineByPhase(timeline, "profiling");
    expect(profiling.requests).toHaveLength(1);
    expect(profiling.requests[0].start).toBe(0);
    expect(profiling.requests[0].credit).toBe(-1);
    expect(profiling.requests[0].end).toBe(949);
  });
  it("keeps repeated replays, streams and auxiliary work in separate lanes", () => {
    const requests = hierarchicalTimeline().requests;
    const collapsed = buildRequestTimelineRows(
      requests,
      "conversation",
      new Set(),
    );
    const subagent = collapsed.find((r) => r.kind === "subagent")!;
    const expanded = buildRequestTimelineRows(
      requests,
      "conversation",
      new Set([subagent.key]),
    );
    expect(expanded.filter((r) => r.kind === "parent")).toHaveLength(2);
    expect(expanded.filter((r) => r.kind === "stream")).toHaveLength(2);
    expect(expanded.filter((r) => r.kind === "aux")).toHaveLength(2);
    expect(
      buildRequestTimelineRows(requests, "worker", new Set()),
    ).toHaveLength(2);
  });
  it("retains shared-prefix occupancy until the last overlapping replay expires", () => {
    const values = inflightUniqueTokens([
      { cid: "a", start: 0, end: 10e9, isl: 100 },
      { cid: "a", start: 2e9, end: 12e9, isl: 80 },
      { cid: "b", start: 4e9, end: 6e9, isl: 50 },
    ]);
    expect(values.find((p) => p.t === 6)?.value).toBe(100);
    expect(values.find((p) => p.t === 10)?.value).toBe(80);
    expect(values.at(-1)?.value).toBe(0);
  });
  it("uses the measured throughput frontier and refuses to extrapolate history", () => {
    const runs = comparisonRuns();
    const first = runs.filter(
      (r) => r.source.kind === "github-actions" && r.source.run_id === 1001,
    );
    expect(
      interpolateAtServicePoint(first, 20, "output", "total")!.value,
    ).toBeCloseTo(30, 10);
    expect(
      interpolateAtServicePoint(first, 15, "output", "total")!.value,
    ).toBeCloseTo(35, 10);
    expect(interpolateAtServicePoint(first, 31, "output", "total")).toBeNull();
    expect(
      interpolateAtServicePoint(first.slice(0, 1), 20, "output", "total"),
    ).toBeNull();
    const history = matchedHistory(
      runs.filter((r) => r.hardware === "H100"),
      20,
      "output",
      "total",
    )[0];
    expect(history.points.map((p) => p.value)).toEqual([30, 45, 33]);
    const waterfall = improvementWaterfall(history.points);
    expect(waterfall.map((s) => s.kind)).toEqual(["baseline", "gain", "best"]);
    expect(waterfall[1].delta).toBe(15);
    expect(waterfall.at(-1)!.to).toBe(45);
    const points = [
      { x: 10, y: 40 },
      { x: 20, y: 30 },
      { x: 30, y: 20 },
      { x: 15, y: 10 },
    ];
    expect(globalParetoFrontier(points, true, true)).toEqual(
      points.slice(0, 3),
    );
  });
  it("keeps workload changes out of matched history while allowing runtime improvements", () => {
    const runs = comparisonRuns()
      .filter((r) => r.hardware === "H100")
      .slice(0, 6);
    const changed = runs.map((r) =>
      r.date === "2026-10-12" ? { ...r, workload_key: "different-subset" } : r,
    );
    const split = matchedHistory(changed, 20, "output", "total");
    expect(split).toHaveLength(2);
    expect(split.map((s) => s.points.length)).toEqual([1, 1]);
    const improvedRuntime = runs.map((r) =>
      r.date === "2026-10-12" ? { ...r, runtime: "new-runtime" } : r,
    );
    expect(
      matchedHistory(improvedRuntime, 20, "output", "total")[0].points.map(
        (p) => p.value,
      ),
    ).toEqual([30, 45]);
  });
  it("preserves uncached token accounting and ruler ratios across shared URLs", () => {
    const run = fixtureRun(1, "2026-10-05", "H100", 8, 20, 100);
    expect(metricValue(run, "input", "uncached")).toBe(225);
    expect(metricValue(run, "total", "uncached")).toBe(325);
    expect(metricValue(run, "output", "uncached")).toBe(100);
    delete run.metrics.cache.theoretical_cache_hit_rate;
    expect(metricValue(run, "input", "uncached")).toBeNull();
    const geometry = computeIsoXRulerGeometry(
      150,
      { py: 100, rawY: 60 },
      { py: 200, rawY: 30 },
    )!;
    expect(geometry.ratio).toBe(2);
    const draft = nextPerfRulerState(parsePerfRulers(null), {
      curve: "roofline-a",
      isoX: 20,
    });
    const ruler = nextPerfRulerState(draft, { curve: "roofline-b", isoX: 25 });
    expect(parsePerfRulers(serializePerfRulers(ruler)).rulers[0]).toMatchObject(
      { isoX: 20, curveA: "roofline-a", curveB: "roofline-b" },
    );
  });
});
