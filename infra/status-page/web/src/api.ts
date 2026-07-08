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

// One strip within a ferry card. `label` is the tier caption (e.g. "tier1");
// null for single-workflow ferries, where `file` is shown as the subtitle.
export interface FerryTierStatus {
  label: string | null;
  file: string;
  latest: FerryRun | null;
  history: FerryRun[];
  successRate: number | null;
  fetchedAt: string;
  error?: string;
}

export interface FerryGroupStatus {
  name: string;
  tiers: FerryTierStatus[];
}

export interface FerryResponse {
  runLimit: number;
  groups: FerryGroupStatus[];
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
  windowMs: number;
  fetchedAt: string;
  error?: string;
}

// Per-cycle create-success ratio over the trailing 24h: a fleet average plus a
// per-region rollup (region omitted for a cycle with zero resolved attempts).
// `fleet` is null for a cycle that resolved zero attempts. Ratios are 0..1.
// Mirrors server/sources/clusterHistory.ts.
export interface ProvisioningHistorySample {
  t: number; // epoch millis
  fleet: number | null;
  regions: Record<string, number>;
}

export interface ProvisioningHistoryResponse {
  samples: ProvisioningHistorySample[];
  windowMs: number;
  fetchedAt: string;
  error?: string;
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

// Synthetic-canary probes, read from finelog's `infra.canary.metrics`. Mirrors
// server/sources/probes.ts.
export interface ProbeCheck {
  probe: string;
  up: boolean;
  latencyMs: number | null;
  collectedAt: string;
}

export interface ProvisionPool {
  resourceType: string;
  scaleGroup: string;
  zone: string;
  ready: number;
  stockout: number;
  error: number;
  preempted: number;
  outcomes: number;
  successRatio: number | null;
  latencyP50Seconds: number | null;
  latencyP95Seconds: number | null;
}

export interface ProvisionFleet {
  ready: number;
  stockout: number;
  error: number;
  preempted: number;
  outcomes: number;
  successRatio: number | null;
  poolsPlacing: number;
  poolsStockoutDead: number;
  latencyP50Seconds: number | null;
  latencyP95Seconds: number | null;
}

export interface ProvisioningSnapshot {
  windowHours: number | null;
  collectedAt: string | null;
  fleet: ProvisionFleet | null;
  pools: ProvisionPool[];
}

export interface ProbesSnapshot {
  checks: ProbeCheck[];
  provisioning: ProvisioningSnapshot;
  fetchedAt: string;
  error?: string;
}

// W&B training charts for the MoE hero run. Mirrors server/sources/wandb.ts.
export interface WandbPoint {
  x: number; // cumulative training tokens
  y: number;
}

export interface WandbRunSeries {
  run: string;
  state: string;
  points: WandbPoint[];
}

export interface WandbChart {
  key: string;
  title: string;
  series: WandbRunSeries[];
}

export interface WandbSnapshot {
  reportTitle: string;
  reportUrl: string;
  charts: WandbChart[];
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
export const fetchProvisioningHistory = () =>
  getJson<ProvisioningHistoryResponse>("/api/provisioning/history");
export const fetchJobs = () => getJson<JobsSnapshot>("/api/jobs");
export const fetchProbes = () => getJson<ProbesSnapshot>("/api/probes");
export const fetchWandb = () => getJson<WandbSnapshot>("/api/wandb");
