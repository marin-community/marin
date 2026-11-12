// SPDX-License-Identifier: GPL-3.0-only
// Throughput interpolation and record gains adapt InferenceX 315128ce.
import {
  hermiteInterpolate,
  monotoneSlopes,
  paretoFrontUpperLeft,
} from "@/components/calculator/interpolation";
import {
  axisValue,
  metricValue,
  workloadIdentity,
  comparisonScope,
  type Metric,
  type TokenBasis,
} from "./metrics";
import {
  type Run,
  submissionIdentity,
  measurementLabel,
  topologyLabel,
} from "./types";

export interface HistoryPoint {
  date: string;
  value: number;
  sources: Run[];
  operatingPoint: number;
}
export interface HistorySeries {
  key: string;
  label: string;
  points: HistoryPoint[];
}
export interface ImprovementStep {
  kind: "baseline" | "gain" | "best";
  date: string;
  from: number;
  to: number;
  delta: number;
  sources: Run[];
}

export function interpolateAtServicePoint(
  runs: Run[],
  target: number,
  metric: Metric,
  basis: TokenBasis,
): { value: number; sources: Run[] } | null {
  const points = runs.map((run) => ({
    run,
    x: axisValue(run, "interactivity", "p90"),
    frontier: metricValue(run, "total", basis),
    value: metricValue(run, metric, basis),
  }));
  if (!points.length || points.some((p) => p.x === null || p.frontier === null))
    return null;
  const front = paretoFrontUpperLeft(
    points,
    (p) => p.x!,
    (p) => p.frontier!,
  ).toSorted((a, b) => a.x! - b.x!);
  if (!front.length || target < front[0].x! || target > front.at(-1)!.x!)
    return null;
  if (front.some((p) => p.value === null)) return null;
  const xs = front.map((p) => p.x!);
  const ys = front.map((p) => p.value!);
  const value =
    front.length === 1
      ? ys[0]
      : Math.max(0, hermiteInterpolate(xs, ys, monotoneSlopes(xs, ys), target));
  return { value, sources: front.map((p) => p.run) };
}

export function matchedHistory(
  runs: Run[],
  target: number,
  metric: Metric,
  basis: TokenBasis,
): HistorySeries[] {
  const groups = new Map<string, Run[]>();
  for (const run of runs) {
    const key = JSON.stringify([
      run.model,
      run.hardware,
      run.framework,
      run.precision,
      run.model_revision,
      run.dataset,
      run.dataset_revision,
      workloadIdentity(run),
      comparisonScope(run),
      run.calibration?.curve_fingerprint,
    ]);
    const group = groups.get(key) || [];
    group.push(run);
    groups.set(key, group);
  }
  return [...groups].map(([key, group]) => {
    const byDate = new Map<string, Run[]>();
    for (const run of group) {
      const values = byDate.get(run.date) || [];
      values.push(run);
      byDate.set(run.date, values);
    }
    const points: HistoryPoint[] = [];
    for (const [date, values] of byDate) {
      // InferenceX uses the latest submission's complete curve within each UTC date.
      const latest = [...values].sort(
        (a, b) =>
          b.source_created_at.localeCompare(a.source_created_at) ||
          submissionIdentity(b).localeCompare(submissionIdentity(a)),
      )[0];
      const curve = values.filter(
        (r) =>
          latest.calibration ||
          submissionIdentity(r) === submissionIdentity(latest),
      );
      const interpolated = interpolateAtServicePoint(
        curve,
        target,
        metric,
        basis,
      );
      if (interpolated)
        points.push({ date, ...interpolated, operatingPoint: target });
    }
    const example = group[0];
    return {
      key,
      label: `${example.hardware} · ${example.framework} · ${example.precision} · ${measurementLabel(example)}${example.calibration ? ` · ${topologyLabel(example)}` : ""} · model ${example.model_revision.slice(0, 8)} · corpus ${example.dataset_revision.slice(0, 8)} · workload ${example.workload_key?.slice(0, 8) || "unknown (reimport)"}`,
      points: points.sort((a, b) => a.date.localeCompare(b.date)),
    };
  });
}

// Recoveries to a previous record are not new gains. No synthetic extensions are used.
export function improvementWaterfall(
  points: readonly HistoryPoint[],
): ImprovementStep[] {
  const ordered = points
    .filter((p) => Number.isFinite(p.value) && p.value > 0)
    .toSorted((a, b) => a.date.localeCompare(b.date));
  if (!ordered.length) return [];
  const step = (
    point: HistoryPoint,
    kind: ImprovementStep["kind"],
    from: number,
  ): ImprovementStep => ({
    kind,
    date: point.date,
    from,
    to: point.value,
    delta: point.value - from,
    sources: point.sources,
  });
  let best = ordered[0];
  const steps = [step(best, "baseline", 0)];
  for (const point of ordered.slice(1)) {
    if (point.value - best.value <= Math.max(1, Math.abs(best.value)) * 1e-9)
      continue;
    steps.push(step(point, "gain", best.value));
    best = point;
  }
  steps.push(step(best, "best", 0));
  return steps;
}
