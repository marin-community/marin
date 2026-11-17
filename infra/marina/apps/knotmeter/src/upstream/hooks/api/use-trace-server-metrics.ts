// SPDX-License-Identifier: GPL-3.0-only
// InferenceX 315128ce wire contracts.
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
export interface PointMeta {
  id: number;
  hardware: string;
  framework: string;
  model: string;
  precision: string;
  spec_method: string;
  disagg: boolean;
  is_multinode: boolean;
  prefill_tp: number;
  prefill_ep: number;
  prefill_dp_attention: boolean;
  prefill_num_workers: number;
  decode_tp: number;
  decode_ep: number;
  decode_dp_attention: boolean;
  decode_num_workers: number;
  num_prefill_gpu: number;
  num_decode_gpu: number;
  recipe_fingerprint: string | null;
  conc: number;
  offload_mode: string | null;
  kv_offloading: string | null;
  kv_offload_backend: string | null;
  kv_offload_backend_version: string | null;
  kv_p2p_transfer: string | null;
  router_name: string | null;
  router_version: string | null;
  isl: number | null;
  osl: number | null;
  benchmark_type: string;
  date: string;
  run_url: string | null;
  server_gpu_cache_hit_rate: number | null;
  server_cpu_cache_hit_rate: number | null;
}

export type MetricSourceRole =
  | "router"
  | "prefill"
  | "decode"
  | "combined"
  | "unknown";

export interface MetricSource {
  id: string;
  adapter: string;
  role: MetricSourceRole;
  endpointUrl: string | null;
  nativeRole: string | null;
  workerId: string | null;
  dpRank: string | null;
  engine: string | null;
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

export interface MetricSourceDescriptor {
  source: MetricSource;
}

export interface TraceServerMetrics {
  meta: PointMeta;
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
  kvCachePoolTokens: number | null;
  metricSources: MetricSourceDescriptor[];
}
