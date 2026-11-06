export interface RequestRecord {
  id: string;
  conversation_id: string;
  corpus_id: string;
  source_outer_idx: number;
  source_kind: string;
  turn: number;
  phase: string;
  start_ns: number;
  end_ns: number;
  ttft_ms: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  error: string | null;
}
export interface Run {
  key: string;
  date: string;
  source_created_at: string;
  source: {
    repository: string;
    run_id: number;
    run_attempt: number;
    head_sha: string;
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
  request_count: number;
  profiling_requests: number;
  origin_ns: string;
  metrics: {
    throughput: {
      per_gpu: { output_tput_tps: number };
      duration_seconds: number;
    };
    latency: Record<string, Record<string, number>>;
  };
  framework: string;
  aggregate: unknown;
  configuration: Record<string, unknown>;
  manifest: unknown;
  validation: unknown;
  source_files: Record<string, string>;
  content_sha256: string;
}
export interface CorpusTurn {
  kind: string;
  turnIndex?: number;
  rawIndex?: number;
  innerIndex?: number;
  model?: string;
  startS?: number;
  endS?: number;
  in: number;
  out: number;
  label?: string;
  children?: CorpusTurn[];
}
export interface Corpus {
  dataset: string;
  revision: string;
  conversation_id: string;
  structure: { nodes: CorpusTurn[]; totals: Record<string, number> };
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
  return `https://github.com/${run.source.repository}/actions/runs/${run.source.run_id}/attempts/${run.source.run_attempt}`;
}
