export interface RequestRecord {
  id: string;
  conversation_id: string;
  corpus_id: string;
  source_outer_idx: number;
  source_inner_idx: number | null;
  source_kind: string;
  turn: number;
  phase: string;
  replay_index: number | null;
  worker_id: string;
  agent_depth: number;
  root_correlation_id: string | null;
  parent_correlation_id: string | null;
  correlation_id: string | null;
  credit_ns: number;
  start_ns: number;
  ack_ns: number | null;
  end_ns: number;
  ttft_ms: number | null;
  tpot_ms: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  cancelled: boolean;
  error: string | null;
}
export interface Run {
  key: string;
  date: string;
  source_created_at: string;
  source:
    | {
        kind: "github-actions";
        repository: string;
        run_id: number;
        run_attempt: number;
        head_sha: string;
      }
    | {
        kind: "private-calibration";
        commit: string;
        files: Record<string, string>;
        job_id: string;
        attempt_uid: string;
        artifact_uri: string;
      };
  measurement_seconds?: number;
  topology?: { tp: number; dp: number; ep: number; pp: number };
  calibration?: {
    curve_fingerprint: string;
    profiling_sent: number;
    profiling_cancelled: number;
    profiling_errors: number;
    caveat: string;
  };
  model: string;
  hardware: string;
  physical_gpus: number;
  hosts: number;
  concurrency: number;
  precision: string;
  scenario: string;
  runtime: string;
  recipe: string;
  model_revision: string;
  dataset: string;
  dataset_revision: string;
  dataset_sessions: number;
  workload_key?: string;
  request_count: number;
  profiling_requests: number;
  origin_ns: string;
  metrics: {
    throughput: {
      per_gpu: {
        output_tput_tps: number;
        input_tput_tps: number;
        total_tput_tps: number;
      };
      duration_seconds: number;
    };
    cache: { theoretical_cache_hit_rate?: number };
    tokens: Record<string, Record<string, number>>;
    latency: Record<string, Record<string, number>>;
  };
  framework: string;
  aggregate: Record<string, unknown> & {
    server_metrics?: {
      present: boolean;
      cache: Record<string, number | null>;
      tokens: {
        prompt_total: number | null;
        generation_total: number | null;
        requests_completed: number | null;
        prompt_by_source: {
          computed: number | null;
          gpu_cache_hit: number | null;
          cpu_or_external_cache_hit: number | null;
        };
      };
    };
  };
  diagnostic_aggregates?: Record<
    "isl" | "osl" | "kvCacheUtil" | "prefixCacheHitRate",
    Record<string, number> | null
  >;
  configuration: Record<string, unknown>;
  manifest: unknown;
  validation: unknown;
  source_files: Record<string, string>;
  content_sha256: string;
}
export async function api<T>(path: string): Promise<T> {
  const response = await fetch(`api/${path}`);
  if (!response.ok)
    throw new Error(
      `${response.status}: ${(await response.json()).detail || response.statusText}`,
    );
  return response.json() as Promise<T>;
}
export function modelName(run: Run) {
  return run.model.split("/").at(-1)!;
}
export function sourceUrl(run: Run) {
  if (run.source.kind === "private-calibration")
    return `https://filer.marina.oa.dev/#${encodeURIComponent(run.source.artifact_uri)}`;
  return `https://github.com/${run.source.repository}/actions/runs/${run.source.run_id}/attempts/${run.source.run_attempt}`;
}
export function sourceLabel(run: Run) {
  return run.source.kind === "private-calibration"
    ? `Iris ${run.source.attempt_uid}`
    : `#${run.source.run_id}/${run.source.run_attempt}`;
}
export function sourceCommit(run: Run) {
  return run.source.kind === "private-calibration"
    ? run.source.commit
    : run.source.head_sha;
}
export function submissionIdentity(run: Run) {
  return run.source.kind === "private-calibration"
    ? `${run.source.job_id}/${run.source.attempt_uid}`
    : `${run.source.repository}/${run.source.run_id}/${run.source.run_attempt}`;
}
export function measurementSeconds(run: Run) {
  // GitHub collector imports require a full hour of native coverage.
  return run.measurement_seconds ?? 3600;
}
export function measurementLabel(run: Run) {
  return `${measurementSeconds(run) / 60}-minute ${run.source.kind === "private-calibration" ? "calibration" : "baseline"}`;
}
export function topologyLabel(run: Run) {
  return run.topology
    ? `TP${run.topology.tp} / DP${run.topology.dp} / EP${run.topology.ep}`
    : "Parallelism not reported";
}
export function seriesLabel(run: Run) {
  return run.calibration
    ? `${topologyLabel(run)} · ${run.physical_gpus} ${run.hardware} · ${measurementLabel(run)}`
    : `${run.hardware} · ${run.date} · ${sourceLabel(run)} · ${measurementLabel(run)}`;
}
