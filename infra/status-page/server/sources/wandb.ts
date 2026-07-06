// W&B training charts, backed by the public W&B GraphQL API.
//
// Surfaces the headline series from the public report
// "67B-A2B MoE on 10T tokens": train cross-entropy loss and Paloma macro
// loss, both against cumulative training tokens, overlaying the two runs
// the report plots (the original muon run, which crashed around step 15k,
// and the run that resumed from its checkpoint).
//
// marin-community is a public W&B entity, so the GraphQL endpoint answers
// anonymously — no API key to provision or rotate. Each run costs one
// request: `sampledHistory(specs: [...])` returns every chart's
// downsampled series in a single response.

const WANDB_GRAPHQL_URL = "https://api.wandb.ai/graphql";
const ENTITY = "marin-community";
const PROJECT = "marin_moe";
const REPORT_TITLE = "67B-A2B MoE on 10T tokens";
const REPORT_URL =
  "https://wandb.ai/marin-community/marin_moe/reports/67B-A2B-MoE-on-10T-tokens--VmlldzoxNzM1OTMxMQ";

// Cumulative-token counter logged on every step (train and eval alike),
// so it works as a shared x-axis across metrics and across the resume
// boundary, where `_step` would restart.
const X_KEY = "throughput/total_tokens";
const SAMPLES_PER_SERIES = 256;
const FETCH_TIMEOUT_MS = 15_000;

interface RunConfig {
  id: string; // W&B run name (Levanter sets it to the experiment name)
  label: string; // short legend label
}

const RUNS: RunConfig[] = [
  {
    id: "moe_67b_a2b_d2560_ep1_rep16_bs4096_seq8192_sw2k_v4_2048_muon_10T",
    label: "muon_10T",
  },
  {
    id: "moe_67b_a2b_d2560_ep1_rep8_bs8192_seq8192_sw2k_v4_2048_muon_resume15k_v2_10T",
    label: "resume15k_v2",
  },
];

interface ChartConfig {
  key: string;
  title: string;
  metric: string;
}

// Add a chart by appending here — the endpoint, sampling, and frontend
// grid all derive from this list.
const CHARTS: ChartConfig[] = [
  { key: "train-loss", title: "Train cross-entropy loss", metric: "train/cross_entropy_loss" },
  { key: "paloma-macro-loss", title: "Paloma macro loss", metric: "eval/paloma/macro_loss" },
];

export interface WandbPoint {
  x: number; // cumulative training tokens
  y: number;
}

export interface WandbRunSeries {
  run: string;
  state: string; // W&B run state: running / finished / crashed / ...
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

const RUN_HISTORY_QUERY = `
query RunSampledHistory($entity: String!, $project: String!, $run: String!, $specs: [JSONString!]!) {
  project(entityName: $entity, name: $project) {
    run(name: $run) {
      state
      sampledHistory(specs: $specs)
    }
  }
}`;

type HistoryRow = Record<string, number | null | undefined>;

interface RunHistoryResponse {
  data?: {
    project?: {
      run?: {
        state?: string;
        sampledHistory?: HistoryRow[][];
      } | null;
    } | null;
  };
  errors?: { message: string }[];
}

// Keep rows where both coordinates are finite numbers; sampled eval rows
// can carry nulls for steps where the metric wasn't logged.
function toPoints(rows: HistoryRow[], metric: string): WandbPoint[] {
  const points: WandbPoint[] = [];
  for (const row of rows) {
    const x = row[X_KEY];
    const y = row[metric];
    if (typeof x === "number" && Number.isFinite(x) && typeof y === "number" && Number.isFinite(y)) {
      points.push({ x, y });
    }
  }
  points.sort((a, b) => a.x - b.x);
  return points;
}

async function fetchRunSeries(
  run: RunConfig,
): Promise<{ state: string; byChart: WandbPoint[][] }> {
  const specs = CHARTS.map((chart) =>
    JSON.stringify({ keys: [X_KEY, chart.metric], samples: SAMPLES_PER_SERIES }),
  );
  const ac = new AbortController();
  const timer = setTimeout(
    () => ac.abort(new Error(`wandb query timed out after ${FETCH_TIMEOUT_MS}ms`)),
    FETCH_TIMEOUT_MS,
  );
  try {
    const res = await fetch(WANDB_GRAPHQL_URL, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        query: RUN_HISTORY_QUERY,
        variables: { entity: ENTITY, project: PROJECT, run: run.id, specs },
      }),
      signal: ac.signal,
    });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`wandb graphql ${res.status}: ${body.slice(0, 300)}`);
    }
    const payload = (await res.json()) as RunHistoryResponse;
    if (payload.errors?.length) {
      throw new Error(`wandb graphql: ${payload.errors[0].message}`);
    }
    const runData = payload.data?.project?.run;
    if (!runData) {
      throw new Error(`wandb run ${run.id} not found in ${ENTITY}/${PROJECT}`);
    }
    const histories = runData.sampledHistory ?? [];
    return {
      state: runData.state ?? "unknown",
      byChart: CHARTS.map((chart, i) => toPoints(histories[i] ?? [], chart.metric)),
    };
  } finally {
    clearTimeout(timer);
  }
}

export async function wandbSnapshot(): Promise<WandbSnapshot> {
  const fetchedAt = new Date().toISOString();
  try {
    const perRun = await Promise.all(RUNS.map((run) => fetchRunSeries(run)));
    const charts = CHARTS.map((chart, i) => ({
      key: chart.key,
      title: chart.title,
      series: RUNS.map((run, j) => ({
        run: run.label,
        state: perRun[j].state,
        points: perRun[j].byChart[i],
      })),
    }));
    return { reportTitle: REPORT_TITLE, reportUrl: REPORT_URL, charts, fetchedAt };
  } catch (err) {
    return {
      reportTitle: REPORT_TITLE,
      reportUrl: REPORT_URL,
      charts: [],
      fetchedAt,
      error: (err as Error).message,
    };
  }
}
