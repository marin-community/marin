// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
export type MetricSourceRole =
  | "router"
  | "prefill"
  | "decode"
  | "combined"
  | "unknown";

export interface RawMetricSourceSeries {
  endpoint_url?: string;
  labels?: Record<string, string>;
}

export interface ServerMetricsContext {
  framework?: string | null;
  disagg?: boolean;
}

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

interface ServerMetricsAdapter {
  id: string;
  matches: (context: ServerMetricsContext) => boolean;
  identifySource: (series: RawMetricSourceSeries) => MetricSource;
}

function stableId(
  adapter: string,
  parts: (string | null | undefined)[],
): string {
  return [adapter, ...parts.map((part) => part ?? "")].join("|");
}

const dynamoAdapter: ServerMetricsAdapter = {
  id: "dynamo",
  matches: ({ framework }) => framework?.startsWith("dynamo-") ?? false,
  identifySource(series) {
    const labels = series.labels ?? {};
    const nativeRole = labels["dynamo_component"] ?? null;
    const role: MetricSourceRole =
      nativeRole === "prefill"
        ? "prefill"
        : nativeRole === "backend"
          ? "decode"
          : nativeRole === "frontend" || nativeRole === "router"
            ? "router"
            : "unknown";
    const endpointUrl =
      series.endpoint_url ?? labels["dynamo_endpoint"] ?? null;
    const workerId = labels["worker_id"] ?? null;
    return {
      id: stableId("dynamo", [role, workerId ?? endpointUrl]),
      adapter: "dynamo",
      role,
      endpointUrl,
      nativeRole,
      workerId,
      dpRank: null,
      engine: null,
    };
  },
};

const trtllmAdapter: ServerMetricsAdapter = {
  id: "trtllm",
  matches: ({ framework }) => framework?.toLowerCase().includes("trt") ?? false,
  identifySource(series) {
    const labels = series.labels ?? {};
    const nativeRole =
      labels["disaggregation_mode"] ?? labels["dynamo_component"] ?? null;
    const role: MetricSourceRole =
      nativeRole === "prefill"
        ? "prefill"
        : nativeRole === "decode" || nativeRole === "backend"
          ? "decode"
          : nativeRole === "aggregated"
            ? "combined"
            : "unknown";
    const endpointUrl = series.endpoint_url ?? null;
    const workerId = labels["worker_id"] ?? null;
    return {
      id: stableId("trtllm", [role, endpointUrl ?? workerId]),
      adapter: "trtllm",
      role,
      endpointUrl,
      nativeRole,
      workerId,
      dpRank: null,
      engine: null,
    };
  },
};

const genericAdapter: ServerMetricsAdapter = {
  id: "generic",
  matches: () => true,
  identifySource(series) {
    const labels = series.labels ?? {};
    const endpointUrl = series.endpoint_url ?? null;
    const workerId = labels["worker_id"] ?? null;
    const dpRank = labels["dp_rank"] ?? null;
    const engine = labels["engine"] ?? labels["engine_idx"] ?? null;
    return {
      id: stableId("generic", [workerId ?? endpointUrl]),
      adapter: "generic",
      role:
        endpointUrl || workerId || dpRank || engine ? "unknown" : "combined",
      endpointUrl,
      nativeRole: null,
      workerId,
      dpRank: null,
      engine: null,
    };
  },
};

const ADAPTERS: readonly ServerMetricsAdapter[] = [
  trtllmAdapter,
  dynamoAdapter,
  genericAdapter,
];

export function selectServerMetricsAdapter(
  context: ServerMetricsContext,
): ServerMetricsAdapter {
  return ADAPTERS.find((adapter) => adapter.matches(context)) ?? genericAdapter;
}
