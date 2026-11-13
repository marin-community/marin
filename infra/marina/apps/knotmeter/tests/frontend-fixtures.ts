import type { Run } from "../src/types";
import type {
  RequestTimeline,
  RequestRecord,
} from "../src/upstream/hooks/api/use-request-timeline";

// Synthetic UI measurements. These fixtures are never loaded into a preview database.
export function fixtureRun(
  id: number,
  date: string,
  hardware: string,
  concurrency: number,
  interactivity: number,
  output: number,
): Run {
  return {
    key: `ui-fixture-${id}-c${concurrency}`,
    date,
    source_created_at: `${date}T12:00:00Z`,
    source: {
      kind: "github-actions",
      repository: "example/ui-test-fixtures",
      run_id: id,
      run_attempt: 1,
      head_sha: "ui-fixture-source",
    },
    model: "UI TEST FIXTURE — synthetic model",
    hardware,
    physical_gpus: 8,
    topology: { tp: 1, dp: 8, ep: 8, pp: 1 },
    hosts: 1,
    concurrency,
    precision: "bf16",
    scenario: "AgentX",
    runtime: "ui-fixture-runtime",
    recipe: "ui-fixture-recipe",
    model_revision: "ui-fixture-model-revision",
    dataset: "example/ui-test-corpus",
    dataset_revision: "ui-fixture-corpus-revision",
    dataset_sessions: 2,
    workload_key: "ui-fixture-workload",
    request_count: 6,
    profiling_requests: 6,
    origin_ns: "1793157479513724276",
    metrics: {
      throughput: {
        per_gpu: {
          input_tput_tps: output * 9,
          output_tput_tps: output,
          total_tput_tps: output * 10,
        },
        duration_seconds: 3600,
      },
      cache: { theoretical_cache_hit_rate: 0.75 },
      tokens: {
        input: { mean: 1000, p50: 1000, p75: 1500, p90: 1800, p95: 1900 },
        output_actual: { mean: 100, p50: 100, p75: 150, p90: 180, p95: 190 },
      },
      latency: {
        ttft: {
          p50: 1 / interactivity,
          p75: 1 / interactivity,
          p90: 1 / interactivity,
          p95: 1 / interactivity,
        },
        full_response_itl: {
          p50: 1 / interactivity,
          p75: 1 / interactivity,
          p90: 1 / interactivity,
          p95: 1 / interactivity,
        },
        e2el: {
          p50: 100 / interactivity,
          p75: 100 / interactivity,
          p90: 100 / interactivity,
          p95: 100 / interactivity,
        },
        e2e_norm_intvty: {
          p50: interactivity / 2,
          p75: interactivity / 2,
          p90: interactivity / 2,
          p95: interactivity / 2,
        },
      },
    },
    framework: "vllm",
    aggregate: {
      tp: 8,
      ep: 1,
      pp: 1,
      kv_offloading: false,
      server_metrics: {
        present: true,
        cache: {
          gpu_cache_hit_rate: 0.6,
          cpu_cache_hit_rate: 0.2,
          external_cache_hit_rate: 0.2,
        },
        tokens: {
          prompt_total: 1000,
          generation_total: 100,
          requests_completed: 6,
          prompt_by_source: {
            computed: 200,
            gpu_cache_hit: 600,
            cpu_or_external_cache_hit: 200,
          },
        },
      },
    },
    configuration: { fixture: true },
    manifest: { fixture: true },
    validation: { fixture: true },
    source_files: { fixture: "ui-test" },
    content_sha256: "ui-test-fixture",
  };
}

export function comparisonRuns(): Run[] {
  const runs: Run[] = [];
  for (const [id, date, hw, factor] of [
    [1001, "2026-10-05", "H100", 1],
    [1002, "2026-10-05", "GB200", 2],
    [1003, "2026-10-12", "H100", 1.5],
    [1004, "2026-10-19", "H100", 1.1],
  ] as const)
    for (const [i, x, y] of [
      [8, 10, 40],
      [32, 20, 30],
      [64, 30, 20],
    ])
      runs.push(fixtureRun(id, date, hw, i, x, y * factor));
  return runs;
}

export function hierarchicalTimeline(): RequestTimeline {
  const base: RequestRecord = {
    cid: "fixture-conversation",
    ri: 0,
    ti: 0,
    srcTrace: "fixture-conversation",
    srcOuter: 0,
    wid: "worker-1",
    ad: 0,
    phase: "profiling",
    credit: 0,
    start: 10,
    ack: 12,
    end: 1000,
    ttftMs: 1,
    tpotMs: 1,
    isl: 100,
    osl: 10,
    cancelled: false,
  };
  return {
    version: 6,
    startNs: 1793157479513724200,
    endNs: 1793157479513726200,
    durationS: 0.000002,
    requests: [
      base,
      {
        ...base,
        cid: "fixture-conversation::sa:child:s0",
        ad: 1,
        start: 100,
        end: 700,
        srcOuter: 1,
        srcInner: 0,
      },
      {
        ...base,
        cid: "fixture-conversation::sa:child:s1",
        ad: 1,
        start: 150,
        end: 750,
        srcOuter: 1,
        srcInner: 1,
      },
      {
        ...base,
        cid: "fixture-conversation::sa:child:aux:tool",
        ad: 1,
        start: 200,
        end: 800,
        srcOuter: 1,
        srcInner: 2,
      },
      { ...base, cid: "fixture-conversation::aux:tool", start: 300, end: 900 },
      { ...base, ri: 1, start: 500, end: 1500, wid: "worker-2" },
    ],
  };
}
