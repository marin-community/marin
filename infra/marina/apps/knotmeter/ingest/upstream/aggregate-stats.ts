// SPDX-License-Identifier: GPL-3.0-only
// InferenceX 315128ce agentic-shared.ts and agentic-aggregates.ts calculations.
export interface MetricPercentiles {
  mean: number;
  p50: number;
  p75: number;
  p90: number;
  p95: number;
  p99: number;
  n: number;
}

export function quantile(sortedAsc: number[], q: number): number {
  if (sortedAsc.length === 0) return Number.NaN;
  if (sortedAsc.length === 1) return sortedAsc[0]!;
  const pos = (sortedAsc.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sortedAsc[lo]!;
  return sortedAsc[lo]! + (sortedAsc[hi]! - sortedAsc[lo]!) * (pos - lo);
}

export function meanOf(xs: number[]): number {
  if (xs.length === 0) return Number.NaN;
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

export function percentilesOf(samples: number[]): MetricPercentiles | null {
  const clean = samples.filter((v) => Number.isFinite(v));
  if (clean.length === 0) return null;
  const sorted = [...clean].toSorted((a, b) => a - b);
  return {
    mean: meanOf(sorted),
    p50: quantile(sorted, 0.5),
    p75: quantile(sorted, 0.75),
    p90: quantile(sorted, 0.9),
    p95: quantile(sorted, 0.95),
    p99: quantile(sorted, 0.99),
    n: sorted.length,
  };
}

interface TimeSlice {
  start_ns?: number;
  end_ns?: number;
  avg?: number;
  rate?: number;
  count?: number;
  sum?: number;
}
interface Series {
  labels?: Record<string, string>;
  timeslices?: TimeSlice[];
}
interface MetricMeta {
  series?: Series[];
}
interface MetricsJson {
  metrics?: Record<string, MetricMeta>;
}

/**
 * Aggregate a per-timeslice field across all series of a metric, indexed by
 * the timeslice's `start_ns`. vllm reports one series per engine on
 * multi-engine DP/PP deployments, so we sum (or average) across engines to
 * get the cluster-wide value at each timeslice.
 *
 * `field` selects which numeric field on a timeslice to read (`avg` for
 * gauges, `rate` for counter deltas). `combine` controls cross-engine math:
 * 'sum' for running/waiting/throughput counters where the cluster total is
 * the sum; 'avg' for KV cache utilization, which is bounded [0, 1] per
 * engine and should be averaged across engines for the cluster view.
 */
function aggregateSeriesByStart(
  metricSeries: readonly Series[] | undefined,
  field: "avg" | "rate",
  combine: "sum" | "avg",
): Map<number, number> {
  const sums = new Map<number, number>();
  const counts = new Map<number, number>();
  for (const s of metricSeries ?? []) {
    for (const ts of s.timeslices ?? []) {
      if (typeof ts.start_ns !== "number") continue;
      const v = ts[field];
      if (typeof v !== "number" || !Number.isFinite(v)) continue;
      sums.set(ts.start_ns, (sums.get(ts.start_ns) ?? 0) + v);
      counts.set(ts.start_ns, (counts.get(ts.start_ns) ?? 0) + 1);
    }
  }
  if (combine === "sum") return sums;
  const out = new Map<number, number>();
  for (const [t, s] of sums) out.set(t, s / (counts.get(t) ?? 1));
  return out;
}

/** First metric whose series array is non-empty; supports vllm/sglang fallback. */
function pickFirstNonEmpty(
  metrics: Record<string, MetricMeta>,
  ...names: string[]
): Series[] | undefined {
  for (const name of names) {
    const s = metrics[name]?.series;
    if (s && s.length > 0) return s;
  }
  return undefined;
}

/**
 * Parse the server_metrics_json → time-series arrays for KV cache util and
 * prefix cache hit rate (per-interval, computed from the prometheus
 * counters).
 *
 * Aggregates across all engine series so multi-engine DP/PP deployments are
 * counted correctly.
 */
export function extractServerMetricSamples(json: string): {
  kvCacheUtil: number[];
  prefixCacheHitRate: number[];
} {
  const parsed = JSON.parse(json) as MetricsJson;
  const metrics = parsed.metrics ?? {};

  // KV cache util — per-engine gauge in [0, 1]. Average across engines so the
  // value stays a percentage; summing would give meaningless 0..N.
  const kvSeriesAll = pickFirstNonEmpty(
    metrics,
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
    "sglang:token_usage",
    "atom:kv_cache_usage_ratio",
    "trtllm_kv_cache_utilization",
  );
  const kvCacheUtil = [
    ...aggregateSeriesByStart(kvSeriesAll, "avg", "avg").values(),
  ];

  // Prefix cache hit rate per interval = Σhits.rate / Σqueries.rate across
  // all engines. Sum first, then divide. SGLang names: cached_tokens / prompt_tokens.
  const hitsAll = pickFirstNonEmpty(
    metrics,
    "vllm:prefix_cache_hits",
    "vllm:gpu_prefix_cache_hits",
    "sglang:cached_tokens",
    "atom:prefix_cache_cached_tokens",
    "trtllm_prompt_cached_tokens",
    "trtllm_prompt_cached_tokens_total",
  );
  const queriesAll = pickFirstNonEmpty(
    metrics,
    "vllm:prefix_cache_queries",
    "vllm:gpu_prefix_cache_queries",
    "vllm:prompt_tokens",
    "sglang:prompt_tokens",
    "atom:prefix_cache_full_tokens",
    "trtllm_prompt_tokens",
    "trtllm_prompt_tokens_total",
  );
  const hitsByT = aggregateSeriesByStart(hitsAll, "rate", "sum");
  const qByT = aggregateSeriesByStart(queriesAll, "rate", "sum");
  const prefixCacheHitRate: number[] = [];
  for (const [t, h] of hitsByT) {
    const q = qByT.get(t);
    if (q !== undefined && q > 0) prefixCacheHitRate.push(h / q);
  }
  if (prefixCacheHitRate.length === 0) {
    const directRate = aggregateSeriesByStart(
      pickFirstNonEmpty(
        metrics,
        "trtllm_kv_cache_hit_rate",
        "atom:prefix_cache_hit_ratio",
      ),
      "avg",
      "avg",
    );
    prefixCacheHitRate.push(...directRate.values());
  }

  return { kvCacheUtil, prefixCacheHitRate };
}
