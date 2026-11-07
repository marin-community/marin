// SPDX-License-Identifier: GPL-3.0-only
// Offline adapter for the pinned InferenceX reducer; see THIRD_PARTY.md.
import { readFileSync } from "node:fs";
import {
  CHART_METRIC_KEYS,
  computeChartSeriesFromMetricPhases,
} from "./upstream/compute-chart-series";
import {
  extractServerMetricSamples,
  percentilesOf,
} from "./upstream/aggregate-stats";

if (process.argv[2] === "--keys") {
  process.stdout.write(
    JSON.stringify([
      ...CHART_METRIC_KEYS,
      "vllm:gpu_prefix_cache_hits",
      "vllm:gpu_prefix_cache_queries",
    ]),
  );
} else {
  const payload = JSON.parse(readFileSync(0, "utf8"));
  const reduced = computeChartSeriesFromMetricPhases(
    payload.metrics,
    payload.warmup_metrics,
    { framework: process.argv[2] },
  );
  const samples = extractServerMetricSamples(
    JSON.stringify({ metrics: payload.metrics }),
  );
  process.stdout.write(
    JSON.stringify({
      ...reduced,
      aggregates: {
        isl: percentilesOf(payload.profile.isl),
        osl: percentilesOf(payload.profile.osl),
        kvCacheUtil: percentilesOf(samples.kvCacheUtil),
        prefixCacheHitRate: percentilesOf(samples.prefixCacheHitRate),
      },
    }),
  );
}
