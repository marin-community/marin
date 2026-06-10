// Lightweight reachability and latency probes for control-plane services.
//
// Each series is keyed by environment + service so prod/dev Iris and
// prod/dev finelog never share a metric name in the API or chart. A
// single dashboard instance only probes the active environment.

import { InstancesClient } from "@google-cloud/compute";

const GCP_PROJECT = process.env.GCP_PROJECT ?? "hai-gcp-models";
const GCP_ZONE = process.env.CONTROLLER_ZONE ?? "us-central1-a";
const CACHE_TTL_MS = 60_000;
const HEALTH_TIMEOUT_MS = 5_000;
const LATENCY_SUMMARY_WINDOW_MS = 5 * 60 * 1000;
const SUMMARY_POINT_INTERVAL_MS = 60_000;

export type Environment = "prod" | "dev";
type ServiceKind = "iris" | "finelog";

export interface ServiceHealthSeries {
  id: string;
  environment: Environment;
  service: ServiceKind;
  name: string;
}

interface GceTarget {
  project: string;
  zone: string;
  port: number;
  filter: string;
}

interface ServiceHealthConfig extends ServiceHealthSeries {
  urlEnvVar: string;
  legacyUrlEnvVar?: string;
  target: GceTarget;
}

export interface ServiceHealthSnapshot extends ServiceHealthSeries {
  reachable: boolean;
  latencyMs: number | null;
  url: string | null;
  fetchedAt: string;
  error?: string;
}

export interface ServiceHealthHistorySample {
  t: number;
  latencies: Record<string, number | null>;
  ok: Record<string, boolean>;
}

export interface ServiceLatencyStats {
  p50: number;
  max: number;
}

export interface ServiceHealthSummarySample {
  t: number;
  stats: Record<string, ServiceLatencyStats | null>;
  sampleCounts: Record<string, number>;
}

export interface ServiceHealthResponse {
  environment: Environment;
  series: ServiceHealthSeries[];
  latest: ServiceHealthSnapshot[];
  samples: ServiceHealthHistorySample[];
  summarySamples: ServiceHealthSummarySample[];
  aggregationWindowMs: number;
  summaryPointIntervalMs: number;
  windowMs: number;
  fetchedAt: string;
}

export const SERVICE_HEALTH_CONFIGS: readonly ServiceHealthConfig[] = [
  {
    id: "prod_iris",
    environment: "prod",
    service: "iris",
    name: "Prod Iris",
    urlEnvVar: "PROD_IRIS_URL",
    legacyUrlEnvVar: "CONTROLLER_URL",
    target: {
      project: GCP_PROJECT,
      zone: GCP_ZONE,
      port: 10000,
      filter: "labels.iris-marin-controller=true AND status=RUNNING",
    },
  },
  {
    id: "prod_finelog",
    environment: "prod",
    service: "finelog",
    name: "Prod finelog",
    urlEnvVar: "PROD_FINELOG_URL",
    legacyUrlEnvVar: "FINELOG_URL",
    target: {
      project: GCP_PROJECT,
      zone: GCP_ZONE,
      port: 10001,
      filter: "labels.finelog-name=finelog-marin AND status=RUNNING",
    },
  },
  {
    id: "dev_iris",
    environment: "dev",
    service: "iris",
    name: "Dev Iris",
    urlEnvVar: "DEV_IRIS_URL",
    target: {
      project: GCP_PROJECT,
      zone: GCP_ZONE,
      port: 10000,
      filter: "labels.iris-marin-dev-controller=true AND status=RUNNING",
    },
  },
  {
    id: "dev_finelog",
    environment: "dev",
    service: "finelog",
    name: "Dev finelog",
    urlEnvVar: "DEV_FINELOG_URL",
    target: {
      project: GCP_PROJECT,
      zone: GCP_ZONE,
      port: 10001,
      filter: "labels.finelog-name=finelog-marin-dev AND status=RUNNING",
    },
  },
] as const;

function activeEnvironment(): Environment {
  const explicit = process.env.CONTROL_PLANE_ENV;
  if (explicit === "prod" || explicit === "dev") return explicit;

  const cluster = (process.env.CLUSTER_NAME ?? "marin").toLowerCase();
  return cluster.includes("dev") ? "dev" : "prod";
}

export const ACTIVE_SERVICE_ENVIRONMENT = activeEnvironment();
const ACTIVE_SERVICE_HEALTH_CONFIGS = SERVICE_HEALTH_CONFIGS.filter(
  (config) => config.environment === ACTIVE_SERVICE_ENVIRONMENT,
);

let client: InstancesClient | null = null;
const urlCache = new Map<string, { url: string; expiresAt: number }>();

function getClient(): InstancesClient {
  if (!client) {
    client = new InstancesClient();
  }
  return client;
}

function configuredUrl(config: ServiceHealthConfig): string {
  return process.env[config.urlEnvVar] ?? process.env[config.legacyUrlEnvVar ?? ""] ?? "";
}

async function discoverGceUrl(config: ServiceHealthConfig): Promise<string> {
  const override = configuredUrl(config);
  if (override) return override;

  const now = Date.now();
  const cached = urlCache.get(config.id);
  if (cached && cached.expiresAt > now) {
    return cached.url;
  }

  const [instances] = await Promise.race([
    getClient().list({
      project: config.target.project,
      zone: config.target.zone,
      filter: config.target.filter,
    }),
    new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error("GCE discovery timed out after 15s")), 15_000),
    ),
  ]);

  if (!instances || instances.length === 0) {
    throw new Error(`No VM found (${config.target.filter})`);
  }

  const instance = instances[0];
  const ip = instance.networkInterfaces?.find((iface) => iface.networkIP)?.networkIP;
  if (!ip) {
    throw new Error(`VM ${instance.name ?? "<unknown>"} has no internal IP`);
  }

  const url = `http://${ip}:${config.target.port}`;
  urlCache.set(config.id, { url, expiresAt: now + CACHE_TTL_MS });
  return url;
}

function metadata(config: ServiceHealthConfig): ServiceHealthSeries {
  return {
    id: config.id,
    environment: config.environment,
    service: config.service,
    name: config.name,
  };
}

async function probe(config: ServiceHealthConfig): Promise<ServiceHealthSnapshot> {
  const fetchedAt = new Date().toISOString();
  let base: string;
  try {
    base = await discoverGceUrl(config);
  } catch (err) {
    return {
      ...metadata(config),
      reachable: false,
      latencyMs: null,
      url: null,
      fetchedAt,
      error: `discovery failed: ${(err as Error).message}`,
    };
  }

  const start = performance.now();
  try {
    const res = await fetch(`${base.replace(/\/$/, "")}/health`, {
      signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS),
    });
    const latencyMs = Math.round(performance.now() - start);
    return {
      ...metadata(config),
      reachable: res.ok,
      latencyMs,
      url: base,
      fetchedAt,
      error: res.ok ? undefined : `/health returned ${res.status}`,
    };
  } catch (err) {
    return {
      ...metadata(config),
      reachable: false,
      latencyMs: null,
      url: base,
      fetchedAt,
      error: `fetch failed: ${(err as Error).message}`,
    };
  }
}

export function serviceHealthSeries(): ServiceHealthSeries[] {
  return ACTIVE_SERVICE_HEALTH_CONFIGS.map(metadata);
}

export async function serviceHealthSnapshot(): Promise<ServiceHealthSnapshot[]> {
  return Promise.all(ACTIVE_SERVICE_HEALTH_CONFIGS.map(probe));
}

export function serviceHealthSample(snapshots: ServiceHealthSnapshot[]): ServiceHealthHistorySample {
  const latencies: Record<string, number | null> = {};
  const ok: Record<string, boolean> = {};
  let t = Date.now();
  for (const snapshot of snapshots) {
    latencies[snapshot.id] = snapshot.latencyMs;
    ok[snapshot.id] = snapshot.reachable;
    t = Math.max(t, Date.parse(snapshot.fetchedAt));
  }
  return { t, latencies, ok };
}

function percentile(sorted: number[], p: number): number {
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

function latencyStats(values: number[]): ServiceLatencyStats | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return {
    p50: Math.round(percentile(sorted, 50)),
    max: sorted[sorted.length - 1],
  };
}

export function serviceHealthSummarySamples(
  samples: ServiceHealthHistorySample[],
  aggregationWindowMs = LATENCY_SUMMARY_WINDOW_MS,
  pointIntervalMs = SUMMARY_POINT_INTERVAL_MS,
): ServiceHealthSummarySample[] {
  const series = serviceHealthSeries();
  const summaries = samples.map((sample, i) => {
    const windowStart = sample.t - aggregationWindowMs;
    const stats: Record<string, ServiceLatencyStats | null> = {};
    const sampleCounts: Record<string, number> = {};

    for (const s of series) {
      const values: number[] = [];
      for (let j = i; j >= 0; j -= 1) {
        const candidate = samples[j];
        if (candidate.t < windowStart) break;
        const latency = candidate.latencies[s.id];
        if (latency !== null && latency !== undefined) {
          values.push(latency);
        }
      }
      stats[s.id] = latencyStats(values);
      sampleCounts[s.id] = values.length;
    }

    return { t: sample.t, stats, sampleCounts };
  });

  if (pointIntervalMs <= 0) return summaries;

  const bucketed: ServiceHealthSummarySample[] = [];
  let currentBucket: number | null = null;
  let latestInBucket: ServiceHealthSummarySample | null = null;

  for (const summary of summaries) {
    const bucket = Math.floor(summary.t / pointIntervalMs);
    if (currentBucket !== null && bucket !== currentBucket && latestInBucket) {
      bucketed.push(latestInBucket);
    }
    currentBucket = bucket;
    latestInBucket = summary;
  }

  if (latestInBucket) {
    bucketed.push(latestInBucket);
  }

  return bucketed;
}

export function serviceHealthResponse(
  latest: ServiceHealthSnapshot[],
  samples: ServiceHealthHistorySample[],
  windowMs: number,
): ServiceHealthResponse {
  return {
    environment: ACTIVE_SERVICE_ENVIRONMENT,
    series: serviceHealthSeries(),
    latest,
    samples,
    summarySamples: serviceHealthSummarySamples(samples),
    aggregationWindowMs: LATENCY_SUMMARY_WINDOW_MS,
    summaryPointIntervalMs: SUMMARY_POINT_INTERVAL_MS,
    windowMs,
    fetchedAt: new Date().toISOString(),
  };
}
