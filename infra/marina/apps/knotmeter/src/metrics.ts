// SPDX-License-Identifier: GPL-3.0-only
// Token accounting adapts InferenceX 315128ce benchmark-transform.ts.
import { type Run, measurementSeconds, submissionIdentity } from "./types";

export const METRICS = {
  output: "Output throughput / GPU",
  input: "Input throughput / GPU",
  total: "Total throughput / GPU",
};
export const AXES = {
  ttft: "TTFT (seconds)",
  interactivity: "Interactivity (tokens/s/user)",
  e2e: "E2E latency (seconds)",
  normalized: "E2E-normalized interactivity (tokens/s/user)",
};
export type Metric = keyof typeof METRICS;
export type Axis = keyof typeof AXES;
export type TokenBasis = "total" | "uncached";

export function metricValue(
  run: Run,
  metric: Metric,
  basis: TokenBasis,
): number | null {
  const rates = run.metrics.throughput.per_gpu;
  const value = rates[`${metric}_tput_tps`];
  if (basis === "total" || metric === "output")
    return Number.isFinite(value) ? value : null;
  const rate = run.metrics.cache?.theoretical_cache_hit_rate;
  if (typeof rate !== "number" || rate < 0 || rate > 1 || !rates.input_tput_tps)
    return null;
  return Math.max(0, value - rates.input_tput_tps * rate);
}

export function axisValue(
  run: Run,
  axis: Axis,
  percentile: string,
): number | null {
  const latency = run.metrics.latency;
  const value =
    axis === "interactivity"
      ? 1 / latency.full_response_itl?.[percentile]
      : axis === "normalized"
        ? latency.e2e_norm_intvty?.[percentile]
        : latency[axis === "e2e" ? "e2el" : "ttft"]?.[percentile];
  return Number.isFinite(value) && value > 0 ? value : null;
}

export function workloadIdentity(run: Run): string {
  // Legacy imports cannot establish a matched workload; reimport their saved bundle.
  return run.workload_key || `unknown:${run.key}`;
}

export function seriesKey(run: Run): string {
  if (run.calibration)
    return `calibration_${run.calibration.curve_fingerprint}`;
  const a = run.aggregate;
  const submission =
    run.source.kind === "github-actions"
      ? [run.source.run_id, run.source.run_attempt]
      : [submissionIdentity(run)];
  return [
    "s",
    ...submission,
    run.hardware,
    run.precision,
    run.framework,
    a.tp,
    a.ep,
    a.pp,
    a.dcp_size,
    a.pcp_size,
    a.dp_attention,
    a.kv_offloading,
    run.model_revision,
    run.dataset_revision,
    workloadIdentity(run),
  ]
    .join("_")
    .replace(/[^\w-]/g, "_");
}

export function comparisonScope(run: Run): string {
  return JSON.stringify([
    run.model,
    run.model_revision,
    run.dataset,
    run.dataset_revision,
    run.precision,
    workloadIdentity(run),
    run.source.kind,
    measurementSeconds(run),
  ]);
}
