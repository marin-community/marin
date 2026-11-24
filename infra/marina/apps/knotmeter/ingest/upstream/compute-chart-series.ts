// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
import {
  selectServerMetricsAdapter,
  type MetricSource,
  type ServerMetricsContext,
} from "./server-metrics-adapters";

export const CHART_SERIES_VERSION = 19;

export interface TimeSeriesPoint {
  t: number;
  value: number;
}

export interface QueueDepthPoint {
  t: number;
  running: number;
  waiting: number;
  total: number;
}

export interface ChartSeries {
  version: number;
  startNs: number;
  endNs: number;
  durationS: number;
  timeslicesCount: number;
  kvCacheUsage: TimeSeriesPoint[];
  prefixCacheHitRate: TimeSeriesPoint[];
  queueDepth: QueueDepthPoint[];
  promptTokensBySource: Record<string, TimeSeriesPoint[]>;
  prefillTps: TimeSeriesPoint[];
  decodeTps: TimeSeriesPoint[];
  prefixCacheHitsTps: TimeSeriesPoint[];
  hostKvCacheUsage: TimeSeriesPoint[];
  kvCacheUsageByEngine: { engineLabel: string; points: TimeSeriesPoint[] }[];
  metricSources: MetricSourceSeries[];
}

export interface MetricSourceSeries {
  source: MetricSource;
  kvCacheUsage: TimeSeriesPoint[];
  prefixCacheHitRate: TimeSeriesPoint[];
  queueDepth: QueueDepthPoint[];
  promptTokensBySource: Record<string, TimeSeriesPoint[]>;
  promptTps: TimeSeriesPoint[];
  generationTps: TimeSeriesPoint[];
  prefixCacheHitsTps: TimeSeriesPoint[];
  hostKvCacheUsage: TimeSeriesPoint[];
  kvCacheUsageByEngine: { engineLabel: string; points: TimeSeriesPoint[] }[];
}

interface RawSlice {
  start_ns?: number;
  end_ns?: number;
  avg?: number;
  rate?: number;
  sum?: number;
}

type RawSliceField = "avg" | "rate" | "sumRate";

interface RawSeries {
  endpoint_url?: string;
  labels?: Record<string, string>;
  timeslices?: RawSlice[];
}

export interface RawMetric {
  series?: RawSeries[];
}

export type MetricsMap = Record<string, RawMetric>;

export const CHART_METRIC_KEYS = new Set([
  "vllm:kv_cache_usage_perc",
  "vllm:gpu_cache_usage_perc",
  "vllm:prefix_cache_hits",
  "vllm:prefix_cache_queries",
  "vllm:num_requests_running",
  "vllm:num_requests_waiting",
  "vllm:prompt_tokens",
  "vllm:generation_tokens",
  "vllm:prompt_tokens_by_source",
  "sglang:token_usage",
  "sglang:cached_tokens",
  "sglang:prompt_tokens",
  "sglang:generation_tokens",
  "sglang:num_running_reqs",
  "sglang:num_queue_reqs",
  "sglang:realtime_tokens",
  "sglang:hicache_host_used_tokens",
  "sglang:hicache_host_total_tokens",
  "atom:kv_cache_usage_ratio",
  "atom:prefix_cache_cached_tokens",
  "atom:prefix_cache_full_tokens",
  "atom:prefix_cache_hit_ratio",
  "atom:requests_running",
  "atom:requests_waiting",
  "atom:prompt_tokens",
  "atom:generation_tokens",
  "dynamo_component_gpu_cache_usage_percent",
  "trtllm_kv_cache_utilization",
  "trtllm_kv_cache_host_utilization",
  "trtllm_kv_cache_hit_rate",
  "trtllm_prompt_cached_tokens",
  "trtllm_prompt_cached_tokens_total",
  "trtllm_prompt_tokens",
  "trtllm_prompt_tokens_total",
  "trtllm_prefill_batch_tokens",
  "trtllm_generation_tokens",
  "trtllm_generation_tokens_total",
  "trtllm_num_requests_running",
  "trtllm_num_requests_waiting",
]);

function mergePhaseMetrics(
  profiling: MetricsMap,
  warmup: MetricsMap,
): MetricsMap {
  if (Object.keys(warmup).length === 0) return profiling;
  if (Object.keys(profiling).length === 0) return warmup;
  const out: MetricsMap = {};
  for (const name of new Set([
    ...Object.keys(profiling),
    ...Object.keys(warmup),
  ])) {
    out[name] = {
      series: [
        ...(profiling[name]?.series ?? []),
        ...(warmup[name]?.series ?? []),
      ],
    };
  }
  return out;
}

export function computeChartSeriesFromMetricPhases(
  profiling: MetricsMap,
  warmup: MetricsMap,
  context: ServerMetricsContext = {},
): ChartSeries {
  return buildSeriesFromMetrics(mergePhaseMetrics(profiling, warmup), context);
}

const INTRA_ENGINE_SHARD_LABELS = new Set([
  "tp_rank",
  "pp_rank",
  "ep_rank",
  "moe_ep_rank",
]);

function seriesIdentityKey(s: RawSeries): string {
  const labels = s.labels ?? {};
  const names = Object.keys(labels)
    .filter((name) => !INTRA_ENGINE_SHARD_LABELS.has(name))
    .toSorted();
  if (names.length === 0) return `@${s.endpoint_url ?? ""}`;
  return names.map((name) => `${name}\u0001${labels[name]}`).join("\u0002");
}

const ENGINE_ROLE_BY_NATIVE_LABEL: Record<string, string> = {
  prefill: "prefill",
  decode: "decode",
  backend: "decode",
};

function labelOrNull(
  labels: Record<string, string>,
  ...names: string[]
): string | null {
  for (const name of names) {
    const value = labels[name]?.trim();
    if (value) return value;
  }
  return null;
}

function engineRankLabel(labels: Record<string, string>): string | null {
  return labelOrNull(labels, "engine", "engine_idx", "dp_rank");
}

function engineRoleLabel(labels: Record<string, string>): string | null {
  for (const name of ["engine_type", "dynamo_component"]) {
    const native = labelOrNull(labels, name);
    const role = native ? ENGINE_ROLE_BY_NATIVE_LABEL[native] : undefined;
    if (role) return role;
  }
  return null;
}

function engineRankSortKey(rank: string | null): number {
  return rank !== null && /^\d+$/u.test(rank)
    ? Number(rank)
    : Number.POSITIVE_INFINITY;
}

const ROLE_SORT_ORDER: Record<string, number> = { prefill: 0, decode: 1 };
function engineRoleSortKey(role: string | null): number {
  return role === null ? 2 : (ROLE_SORT_ORDER[role] ?? 2);
}

function endpointHostPort(endpointUrl: string): string | null {
  const hostPort = /^\w+:\/\/(?<hostPort>[^/]+)/u.exec(endpointUrl)?.groups?.[
    "hostPort"
  ];
  return hostPort ?? (endpointUrl || null);
}

function engineDiscriminator(
  labels: Record<string, string>,
  endpointUrl: string,
  preferEndpoint = false,
): string | null {
  const worker = preferEndpoint ? null : labelOrNull(labels, "worker_id");
  if (worker) return worker.length > 4 ? worker.slice(-4) : worker;
  return endpointHostPort(endpointUrl);
}

const MIRROR_MEAN_TOLERANCE = 0.02;

const MIRROR_RATE_RELATIVE_TOLERANCE = 0.05;

function looksMirrored(
  means: readonly number[],
  field: RawSliceField,
): boolean {
  const spread = Math.max(...means) - Math.min(...means);
  if (field === "avg") return spread <= MIRROR_MEAN_TOLERANCE;
  const scale = Math.max(...means.map((m) => Math.abs(m)));
  if (scale === 0) return true;
  return spread <= MIRROR_RATE_RELATIVE_TOLERANCE * scale;
}

interface LogicalEngine {
  engineLabel: string;
  points: TimeSeriesPoint[];
}

type ScrapeMap = Map<number, { sum: number; count: number }>;

interface EngineGroup {
  labels: Record<string, string>;
  byEndpoint: Map<string, ScrapeMap>;
}

interface ResolvedEngine {
  rank: string | null;
  role: string | null;
  discriminator: string | null;
  workerKey: string | null;
  order: number;
  points: TimeSeriesPoint[];
}

function scrapesToPoints(
  scrapes: ScrapeMap,
  tOf: (ns: number) => number,
): TimeSeriesPoint[] {
  return [...scrapes.entries()]
    .toSorted((a, b) => a[0] - b[0])
    .map(([startNs, { sum, count }]) => ({
      t: tOf(startNs),
      value: sum / count,
    }));
}

function meanOf(scrapes: ScrapeMap): number {
  let total = 0;
  for (const { sum, count } of scrapes.values()) total += sum / count;
  return scrapes.size === 0 ? 0 : total / scrapes.size;
}

function spanOf(scrapes: ScrapeMap): number {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (const ns of scrapes.keys()) {
    if (ns < lo) lo = ns;
    if (ns > hi) hi = ns;
  }
  return hi >= lo ? hi - lo : 0;
}

function resolveComponents(
  series: readonly RawSeries[] | undefined,
  tOf: (ns: number) => number,
  field: RawSliceField = "avg",
): ResolvedEngine[] {
  const groups = new Map<string, EngineGroup>();
  for (const s of series ?? []) {
    const key = seriesIdentityKey(s);
    let group = groups.get(key);
    if (!group) {
      group = { labels: s.labels ?? {}, byEndpoint: new Map() };
      groups.set(key, group);
    }
    const endpoint = s.endpoint_url ?? "";
    let scrapes = group.byEndpoint.get(endpoint);
    if (!scrapes) {
      scrapes = new Map();
      group.byEndpoint.set(endpoint, scrapes);
    }
    for (const ts of s.timeslices ?? []) {
      if (typeof ts.start_ns !== "number" || !Number.isFinite(ts.start_ns))
        continue;
      const durationS =
        typeof ts.start_ns === "number" && typeof ts.end_ns === "number"
          ? (ts.end_ns - ts.start_ns) / 1e9
          : 0;
      const value =
        field === "sumRate"
          ? typeof ts.sum === "number" && durationS > 0
            ? ts.sum / durationS
            : undefined
          : ts[field];
      if (typeof value !== "number" || !Number.isFinite(value)) continue;
      const at = scrapes.get(ts.start_ns);
      if (at) {
        at.sum += value;
        at.count++;
      } else {
        scrapes.set(ts.start_ns, { sum: value, count: 1 });
      }
    }
  }

  const resolved: ResolvedEngine[] = [];
  for (const group of groups.values()) {
    const endpoints = [...group.byEndpoint]
      .filter(([, scrapes]) => scrapes.size > 0)
      .toSorted((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
    if (endpoints.length === 0) continue;

    const rank = engineRankLabel(group.labels);
    const role = engineRoleLabel(group.labels);
    const workerId = labelOrNull(group.labels, "worker_id");
    const push = (
      endpointUrl: string,
      scrapes: ScrapeMap,
      preferEndpoint: boolean,
    ) => {
      resolved.push({
        rank,
        role,
        discriminator: engineDiscriminator(
          group.labels,
          endpointUrl,
          preferEndpoint,
        ),
        workerKey: workerId ?? (endpointUrl || null),
        order: resolved.length,
        points: scrapesToPoints(scrapes, tOf),
      });
    };

    if (endpoints.length === 1) {
      push(endpoints[0]![0], endpoints[0]![1], false);
      continue;
    }

    const means = endpoints.map(([, scrapes]) => meanOf(scrapes));
    const mirrored = looksMirrored(means, field);
    if (!mirrored) {
      for (const [endpointUrl, scrapes] of endpoints)
        push(endpointUrl, scrapes, true);
      continue;
    }
    const best = endpoints.reduce((a, b) => {
      const sa = spanOf(a[1]);
      const sb = spanOf(b[1]);
      if (sb !== sa) return sb > sa ? b : a;
      if (b[1].size !== a[1].size) return b[1].size > a[1].size ? b : a;
      return a;
    });
    push(best[0], best[1], false);
  }

  return resolved.toSorted(
    (a, b) =>
      engineRoleSortKey(a.role) - engineRoleSortKey(b.role) ||
      engineRankSortKey(a.rank) - engineRankSortKey(b.rank) ||
      (a.discriminator ?? "").localeCompare(b.discriminator ?? "") ||
      a.order - b.order,
  );
}

function resolveLogicalEngines(
  series: readonly RawSeries[] | undefined,
  tOf: (ns: number) => number,
): LogicalEngine[] {
  const ordered = resolveComponents(series, tOf, "avg");

  const roles = new Set(ordered.map((e) => e.role).filter((r) => r !== null));
  const showRole = roles.size > 1;
  const withBase = ordered.map((engine, idx) => {
    const named =
      showRole && engine.role
        ? engine.rank === null
          ? engine.role
          : `${engine.role} ${engine.rank}`
        : engine.rank;
    return { ...engine, base: named ?? engine.discriminator ?? `#${idx}` };
  });

  const baseCounts = new Map<string, number>();
  for (const engine of withBase) {
    baseCounts.set(engine.base, (baseCounts.get(engine.base) ?? 0) + 1);
  }
  const used = new Set<string>();
  return withBase.map((engine) => {
    let label = engine.base;
    if ((baseCounts.get(engine.base) ?? 0) > 1 && engine.discriminator) {
      label = `${engine.base} (${engine.discriminator})`;
    }
    let candidate = label;
    for (let n = 2; used.has(candidate); n++) candidate = `${label} #${n}`;
    used.add(candidate);
    return { engineLabel: candidate, points: engine.points };
  });
}

function averageAcrossEngines(
  engines: readonly { points: TimeSeriesPoint[] }[],
): TimeSeriesPoint[] {
  const active = engines.filter((engine) => engine.points.length > 0);
  if (active.length === 0) return [];
  if (active.length === 1) return active[0]!.points;

  const timeline = [
    ...new Set(active.flatMap((e) => e.points.map((p) => p.t))),
  ].toSorted((a, b) => a - b);
  const cursors: number[] = Array.from({ length: active.length }, () => -1);
  const lastT = active.map((engine) => engine.points.at(-1)!.t);
  const carryLimit = active.map((engine) => carryLimitSeconds(engine.points));
  const out: TimeSeriesPoint[] = [];
  for (const t of timeline) {
    let sum = 0;
    let n = 0;
    for (const [i, engine] of active.entries()) {
      const points = engine.points;
      let cursor = cursors[i]!;
      while (cursor + 1 < points.length && points[cursor + 1]!.t <= t) cursor++;
      cursors[i] = cursor;
      if (cursor < 0 || t > lastT[i]!) continue;
      if (t - points[cursor]!.t > carryLimit[i]!) continue;
      sum += points[cursor]!.value;
      n++;
    }
    if (n > 0) out.push({ t, value: sum / n });
  }
  return out;
}

function carryLimitSeconds(points: readonly TimeSeriesPoint[]): number {
  if (points.length < 2) return Number.POSITIVE_INFINITY;
  const gaps: number[] = [];
  for (let i = 1; i < points.length; i++) {
    const gap = points[i]!.t - points[i - 1]!.t;
    if (gap > 0) gaps.push(gap);
  }
  if (gaps.length === 0) return Number.POSITIVE_INFINITY;
  gaps.sort((a, b) => a - b);
  return 5 * gaps[gaps.length >> 1]!;
}

const MAX_TICK_NS = 300 * 1e9;
const MAX_GRID_TICKS = 500_000;
const GRID_EPSILON = 1e-6;

function canonicalTickNs(metrics: MetricsMap): number | null {
  let best: readonly RawSlice[] | null = null;
  for (const metricMeta of Object.values(metrics)) {
    for (const s of metricMeta?.series ?? []) {
      const ts = s.timeslices ?? [];
      if (ts.length > (best?.length ?? 0)) best = ts;
    }
  }
  if (!best || best.length < 2) return null;
  const gaps: number[] = [];
  for (let i = 1; i < best.length; i++) {
    const a = best[i - 1]!.start_ns;
    const b = best[i]!.start_ns;
    if (typeof a !== "number" || typeof b !== "number") continue;
    const gap = b - a;
    if (gap > 0 && gap <= MAX_TICK_NS) gaps.push(gap);
  }
  if (gaps.length === 0) return null;
  gaps.sort((a, b) => a - b);
  const median = gaps[gaps.length >> 1]!;
  return median > 0 ? median : null;
}

interface StepCursor {
  points: readonly TimeSeriesPoint[];
  carryLimit: number;
  cursor: number;
}

function stepValueAt(state: StepCursor, t: number): number | null {
  const { points } = state;
  while (state.cursor + 1 < points.length && points[state.cursor + 1]!.t <= t)
    state.cursor++;
  if (state.cursor < 0 || t > points.at(-1)!.t) return null;
  if (t - points[state.cursor]!.t > state.carryLimit) return null;
  return points[state.cursor]!.value;
}

function sumOntoGrid(
  components: readonly (readonly TimeSeriesPoint[])[],
  tickS: number | null,
): TimeSeriesPoint[] {
  const active = components.filter((points) => points.length > 0);
  if (active.length === 0) return [];
  const first = Math.min(...active.map((points) => points[0]!.t));
  const last = Math.max(...active.map((points) => points.at(-1)!.t));
  if (!Number.isFinite(first) || !Number.isFinite(last)) return [];

  let timeline: number[];
  if (tickS && tickS > 0) {
    const firstTick = Math.ceil(first / tickS - GRID_EPSILON);
    const lastTick = Math.floor(last / tickS + GRID_EPSILON);
    const ticks = lastTick - firstTick + 1;
    if (ticks < 1 || ticks > MAX_GRID_TICKS) return [];
    timeline = Array.from({ length: ticks }, (_, i) => (firstTick + i) * tickS);
  } else {
    timeline = [
      ...new Set(active.flatMap((points) => points.map((p) => p.t))),
    ].toSorted((a, b) => a - b);
  }

  const states: StepCursor[] = active.map((points) => ({
    points,
    carryLimit: carryLimitSeconds(points),
    cursor: -1,
  }));
  const out: TimeSeriesPoint[] = [];
  for (const t of timeline) {
    let sum = 0;
    let n = 0;
    for (const state of states) {
      const value = stepValueAt(state, t);
      if (value === null) continue;
      sum += value;
      n++;
    }
    if (n > 0) out.push({ t, value: sum });
  }
  return out;
}

function summedSeries(
  series: readonly RawSeries[] | undefined,
  tOf: (ns: number) => number,
  field: RawSliceField,
  tickS: number | null,
): TimeSeriesPoint[] {
  const components = resolveComponents(series, tOf, field).map((c) => c.points);
  return sumOntoGrid(components, tickS);
}

function byT(points: readonly TimeSeriesPoint[]): Map<number, number> {
  return new Map(points.map((p) => [p.t, p.value]));
}

function buildSeriesFromMetrics(
  metrics: MetricsMap,
  context: ServerMetricsContext,
  includeMetricSources = true,
  originStartNs?: number,
): ChartSeries {
  let startNs = Number.POSITIVE_INFINITY;
  let endNs = 0;
  let timeslicesCount = 0;
  for (const metricMeta of Object.values(metrics)) {
    for (const s of metricMeta?.series ?? []) {
      const ts = s.timeslices ?? [];
      if (ts.length === 0) continue;
      timeslicesCount = Math.max(timeslicesCount, ts.length);
      const first = ts[0]!;
      const last = ts.at(-1)!;
      if (typeof first.start_ns === "number" && first.start_ns < startNs)
        startNs = first.start_ns;
      if (typeof last.end_ns === "number" && last.end_ns > endNs)
        endNs = last.end_ns;
    }
  }
  if (!Number.isFinite(startNs)) startNs = 0;
  const tOf = (ns: number) => (ns - (originStartNs ?? startNs)) / 1e9;
  const tickNs = canonicalTickNs(metrics);
  const tickS = tickNs === null ? null : tickNs / 1e9;

  const pickSeries = (...names: string[]): readonly RawSeries[] | undefined => {
    for (const name of names) {
      const s = metrics[name]?.series;
      if (s && s.length > 0) return s;
    }
    return undefined;
  };

  const nativeTrtKv = metrics["trtllm_kv_cache_utilization"]?.series ?? [];
  const dynamoTrtKv =
    selectServerMetricsAdapter(context).id === "trtllm"
      ? (metrics["dynamo_component_gpu_cache_usage_percent"]?.series ?? [])
      : [];
  const rankedEndpoints = new Set(dynamoTrtKv.map((s) => s.endpoint_url));
  const trtKv = [
    ...dynamoTrtKv.map((s) => ({
      ...s,
      labels: { ...s.labels, metric_endpoint: s.endpoint_url ?? "" },
    })),
    ...nativeTrtKv.filter((s) => !rankedEndpoints.has(s.endpoint_url)),
  ];
  const kvSeries =
    pickSeries(
      "vllm:kv_cache_usage_perc",
      "vllm:gpu_cache_usage_perc",
      "sglang:token_usage",
      "atom:kv_cache_usage_ratio",
    ) ?? trtKv;
  const engines = resolveLogicalEngines(kvSeries, tOf);
  const kvCacheUsage: TimeSeriesPoint[] = averageAcrossEngines(engines);
  const kvCacheUsageByEngine = engines.length > 1 ? engines : [];

  const hitsSeries = pickSeries(
    "vllm:prefix_cache_hits",
    "sglang:cached_tokens",
    "atom:prefix_cache_cached_tokens",
    "trtllm_prompt_cached_tokens",
    "trtllm_prompt_cached_tokens_total",
  );
  const qsSeries = pickSeries(
    "vllm:prefix_cache_queries",
    "vllm:prompt_tokens",
    "sglang:prompt_tokens",
    "atom:prefix_cache_full_tokens",
    "trtllm_prompt_tokens",
    "trtllm_prompt_tokens_total",
  );
  const hitsOnGrid = summedSeries(hitsSeries, tOf, "rate", tickS);
  const qsOnGrid = summedSeries(qsSeries, tOf, "rate", tickS);
  const qsByT = byT(qsOnGrid);
  const prefixCacheHitRate: TimeSeriesPoint[] = [];
  for (const { t, value: h } of hitsOnGrid) {
    const q = qsByT.get(t);
    if (q !== undefined && q > 0) prefixCacheHitRate.push({ t, value: h / q });
  }
  if (prefixCacheHitRate.length === 0) {
    prefixCacheHitRate.push(
      ...averageAcrossEngines(
        resolveLogicalEngines(
          pickSeries("trtllm_kv_cache_hit_rate", "atom:prefix_cache_hit_ratio"),
          tOf,
        ),
      ),
    );
  }

  const runSeries = pickSeries(
    "vllm:num_requests_running",
    "sglang:num_running_reqs",
    "atom:requests_running",
    "trtllm_num_requests_running",
  );
  const waitSeries = pickSeries(
    "vllm:num_requests_waiting",
    "sglang:num_queue_reqs",
    "atom:requests_waiting",
    "trtllm_num_requests_waiting",
  );
  const runOnGrid = summedSeries(runSeries, tOf, "avg", tickS);
  const waitByT = byT(summedSeries(waitSeries, tOf, "avg", tickS));
  const runByT = byT(runOnGrid);
  const queueDepth: QueueDepthPoint[] = [];
  const allTimes = new Set<number>([...runByT.keys(), ...waitByT.keys()]);
  for (const t of [...allTimes].toSorted((a, b) => a - b)) {
    const running = runByT.get(t) ?? 0;
    const waiting = waitByT.get(t) ?? 0;
    queueDepth.push({ t, running, waiting, total: running + waiting });
  }

  const counterRate = (...names: string[]): TimeSeriesPoint[] =>
    summedSeries(pickSeries(...names), tOf, "rate", tickS);
  const promptCounterTps = counterRate(
    "vllm:prompt_tokens",
    "sglang:prompt_tokens",
    "atom:prompt_tokens",
    "trtllm_prompt_tokens",
    "trtllm_prompt_tokens_total",
  );
  const decodeTps = counterRate(
    "vllm:generation_tokens",
    "sglang:generation_tokens",
    "atom:generation_tokens",
    "trtllm_generation_tokens",
    "trtllm_generation_tokens_total",
  );
  const prefixCacheHitsTps = counterRate(
    "vllm:prefix_cache_hits",
    "sglang:cached_tokens",
    "atom:prefix_cache_cached_tokens",
    "trtllm_prompt_cached_tokens",
    "trtllm_prompt_cached_tokens_total",
  );
  const trtllmComputedPromptTps = summedSeries(
    metrics["trtllm_prefill_batch_tokens"]?.series,
    tOf,
    "sumRate",
    tickS,
  );
  const prefillTps =
    promptCounterTps.length > 0
      ? promptCounterTps
      : sumOntoGrid([trtllmComputedPromptTps, prefixCacheHitsTps], tickS);
  if (prefixCacheHitRate.length === 0 && (!qsSeries || qsSeries.length === 0)) {
    const prefillByT = byT(prefillTps);
    for (const { t, value: cached } of prefixCacheHitsTps) {
      const prompt = prefillByT.get(t);
      if (prompt !== undefined && prompt > 0) {
        prefixCacheHitRate.push({ t, value: cached / prompt });
      }
    }
  }

  const hostTotalByT = byT(
    summedSeries(
      metrics["sglang:hicache_host_total_tokens"]?.series,
      tOf,
      "avg",
      tickS,
    ),
  );
  const hostKvCacheUsage: TimeSeriesPoint[] = [];
  for (const { t, value: used } of summedSeries(
    metrics["sglang:hicache_host_used_tokens"]?.series,
    tOf,
    "avg",
    tickS,
  )) {
    const total = hostTotalByT.get(t);
    if (total !== undefined && total > 0) {
      hostKvCacheUsage.push({ t, value: used / total });
    }
  }
  if (hostKvCacheUsage.length === 0) {
    hostKvCacheUsage.push(
      ...averageAcrossEngines(
        resolveLogicalEngines(
          metrics["trtllm_kv_cache_host_utilization"]?.series,
          tOf,
        ),
      ),
    );
  }

  const promptBySrc = new Map<string, RawSeries[]>();
  const addSeriesRates = (label: string, series: RawSeries): void => {
    const existing = promptBySrc.get(label);
    if (existing) existing.push(series);
    else promptBySrc.set(label, [series]);
  };
  for (const series of metrics["vllm:prompt_tokens_by_source"]?.series ?? []) {
    const labels = series.labels ?? {};
    const source =
      labels["source"] ??
      labels["reason"] ??
      labels["kind"] ??
      JSON.stringify(labels);
    addSeriesRates(source, series);
  }
  if (promptBySrc.size === 0) {
    for (const series of metrics["sglang:realtime_tokens"]?.series ?? []) {
      const labels = series.labels ?? {};
      const mode = labels["mode"] ?? "unknown";
      if (mode !== "prefill_compute") continue;
      addSeriesRates("compute (miss)", series);
    }
    for (const series of metrics["sglang:cached_tokens"]?.series ?? []) {
      const labels = series.labels ?? {};
      const src = labels["cache_source"] ?? "cache hit";
      const label =
        src === "device"
          ? "cache hit (HBM)"
          : src === "host"
            ? "cache hit (CPU offload)"
            : src === "total"
              ? "cache hit"
              : `cache hit (${src})`;
      addSeriesRates(label, series);
    }
  }
  const promptTokensBySource: Record<string, TimeSeriesPoint[]> = {};
  for (const [source, seriesForSource] of promptBySrc) {
    const arr = summedSeries(seriesForSource, tOf, "rate", tickS).filter(
      (p) => p.value > 0,
    );
    if (arr.length > 0) promptTokensBySource[source] = arr;
  }
  if (Object.keys(promptTokensBySource).length === 0) {
    const cached = prefixCacheHitsTps.filter((point) => point.value > 0);
    const cachedByT = byT(prefixCacheHitsTps);
    const computed =
      trtllmComputedPromptTps.length > 0
        ? trtllmComputedPromptTps.filter((point) => point.value > 0)
        : (metrics["atom:prefix_cache_full_tokens"]
            ? counterRate("atom:prefix_cache_full_tokens")
            : promptCounterTps
          )
            .map(({ t, value: prompt }) => ({
              t,
              value: Math.max(0, prompt - (cachedByT.get(t) ?? 0)),
            }))
            .filter((point) => point.value > 0);
    if (cached.length > 0) promptTokensBySource["cache hit (HBM)"] = cached;
    if (computed.length > 0) promptTokensBySource["compute (miss)"] = computed;
  }

  const metricSources: MetricSourceSeries[] = [];
  const adapter = selectServerMetricsAdapter(context);
  if (includeMetricSources && context.disagg && adapter.id !== "generic") {
    const endpointRoles = new Map<string, string>();
    for (const metric of Object.values(metrics)) {
      for (const series of metric.series ?? []) {
        const endpointUrl = series.endpoint_url;
        const role =
          series.labels?.["disaggregation_mode"] ??
          series.labels?.["dynamo_component"];
        if (endpointUrl && role) endpointRoles.set(endpointUrl, role);
      }
    }
    const grouped = new Map<
      string,
      { source: MetricSource; metrics: MetricsMap }
    >();
    for (const [metricName, metric] of Object.entries(metrics)) {
      for (const series of metric.series ?? []) {
        const roleHint = series.endpoint_url
          ? endpointRoles.get(series.endpoint_url)
          : undefined;
        const identifiedSeries =
          roleHint && !series.labels?.["disaggregation_mode"]
            ? {
                ...series,
                labels: { ...series.labels, disaggregation_mode: roleHint },
              }
            : series;
        const source = adapter.identifySource(identifiedSeries);
        let group = grouped.get(source.id);
        if (!group) {
          group = { source, metrics: {} };
          grouped.set(source.id, group);
        }
        group.metrics[metricName] ??= { series: [] };
        const groupedMetric = group.metrics[metricName];
        groupedMetric.series!.push(series);
      }
    }
    for (const { source, metrics: sourceMetrics } of grouped.values()) {
      const sourceSeries = buildSeriesFromMetrics(
        sourceMetrics,
        context,
        false,
        originStartNs ?? startNs,
      );
      metricSources.push({
        source,
        kvCacheUsage: sourceSeries.kvCacheUsage,
        prefixCacheHitRate: sourceSeries.prefixCacheHitRate,
        queueDepth: sourceSeries.queueDepth,
        promptTokensBySource: sourceSeries.promptTokensBySource,
        promptTps: sourceSeries.prefillTps,
        generationTps: sourceSeries.decodeTps,
        prefixCacheHitsTps: sourceSeries.prefixCacheHitsTps,
        hostKvCacheUsage: sourceSeries.hostKvCacheUsage,
        kvCacheUsageByEngine: sourceSeries.kvCacheUsageByEngine,
      });
    }
    const roleOrder: Record<MetricSource["role"], number> = {
      router: 0,
      prefill: 1,
      decode: 2,
      combined: 3,
      unknown: 4,
    };
    metricSources.sort(
      (a, b) =>
        roleOrder[a.source.role] - roleOrder[b.source.role] ||
        (a.source.endpointUrl ?? "").localeCompare(
          b.source.endpointUrl ?? "",
        ) ||
        a.source.id.localeCompare(b.source.id),
    );
  }
  return {
    version: CHART_SERIES_VERSION,
    startNs,
    endNs,
    durationS: endNs > startNs ? (endNs - startNs) / 1e9 : 0,
    timeslicesCount,
    kvCacheUsage,
    prefixCacheHitRate,
    queueDepth,
    promptTokensBySource,
    prefillTps,
    decodeTps,
    prefixCacheHitsTps,
    hostKvCacheUsage,
    kvCacheUsageByEngine,
    metricSources,
  };
}
