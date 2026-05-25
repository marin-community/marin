// Typed fetch wrappers for the server's JSON endpoints.
//
// The types here intentionally mirror the server-side shapes defined in
// server/sources/*.ts. Keeping them duplicated (rather than importing
// across the server/web boundary) avoids tangling tsconfigs and lets the
// web bundle stay independent of node types.

export interface FerryRun {
  id: number;
  conclusion: string | null;
  status: string;
  sha: string;
  shaShort: string;
  startedAt: string;
  durationSeconds: number | null;
  url: string;
  event: string;
  actor: string;
}

export interface FerryWorkflowStatus {
  name: string;
  file: string;
  latest: FerryRun | null;
  history: FerryRun[];
  successRate: number | null;
  fetchedAt: string;
  error?: string;
}

export interface FerryResponse {
  windowDays: number;
  workflows: FerryWorkflowStatus[];
}

// Per-commit aggregate check-run status for the last N commits on main.
// Shape mirrors GitHub's GraphQL `statusCheckRollup.state` — NONE means
// the commit had no checks configured.
export type CommitState =
  | "SUCCESS"
  | "FAILURE"
  | "ERROR"
  | "PENDING"
  | "EXPECTED"
  | "NONE";

export interface CommitStatus {
  oid: string;
  shortOid: string;
  headline: string;
  committedAt: string;
  author: string;
  authorAvatarUrl: string | null;
  url: string;
  state: CommitState;
}

export interface BuildsResponse {
  commits: CommitStatus[];
  successRate: number | null;
  fetchedAt: string;
  error?: string;
}

export interface PingPercentiles {
  p50: number;
  p90: number;
  p99: number;
}

export interface IrisStatus {
  cluster: string;
  reachable: boolean;
  latencyMs: number | null;
  pingPercentiles: PingPercentiles | null;
  pingSampleCount: number;
  pingSpanMs: number;
  pingWindowMs: number;
  controllerUrl: string | null;
  fetchedAt: string;
  error?: string;
  raw?: unknown;
}

export type ServiceEnvironment = "prod" | "dev";
export type ControlPlaneService = "iris" | "finelog";

export interface ServiceHealthSeries {
  id: string;
  environment: ServiceEnvironment;
  service: ControlPlaneService;
  name: string;
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
  environment: ServiceEnvironment;
  series: ServiceHealthSeries[];
  latest: ServiceHealthSnapshot[];
  samples: ServiceHealthHistorySample[];
  summarySamples: ServiceHealthSummarySample[];
  aggregationWindowMs: number;
  summaryPointIntervalMs: number;
  windowMs: number;
  fetchedAt: string;
}

export interface WorkerResourceTotals {
  cpuTotalMillicores: number;
  memoryTotalBytes: number;
  chipsTotal: number;
}

export interface WorkerRegionCount {
  region: string;
  healthy: number;
}

export interface WorkersSnapshot {
  healthy: number;
  resources: WorkerResourceTotals;
  byRegion: WorkerRegionCount[];
  fetchedAt: string;
  error?: string;
}

export interface WorkerSample {
  t: number; // epoch millis
  regions: Record<string, number>;
}

export interface WorkersHistoryResponse {
  samples: WorkerSample[];
}

export interface JobStateCount {
  state: number;
  name: string;
  count: number;
}

export interface JobBucket {
  total: number;
  byState: JobStateCount[];
}

export interface JobsSnapshot {
  inflight: JobBucket;
  last24h: JobBucket;
  windowMs: number;
  fetchedAt: string;
  error?: string;
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`${path} returned ${res.status}`);
  }
  return (await res.json()) as T;
}

export const fetchFerry = () => getJson<FerryResponse>("/api/ferry");
export const fetchBuilds = () => getJson<BuildsResponse>("/api/builds");
export const fetchIris = () => getJson<IrisStatus>("/api/iris");
export const fetchControlPlaneHealth = () =>
  getJson<ServiceHealthResponse>("/api/control-plane/health");
export const fetchWorkers = () => getJson<WorkersSnapshot>("/api/workers");
export const fetchWorkersHistory = () => getJson<WorkersHistoryResponse>("/api/workers/history");
export const fetchJobs = () => getJson<JobsSnapshot>("/api/jobs");
