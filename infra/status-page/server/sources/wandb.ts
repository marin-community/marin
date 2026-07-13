// W&B training charts, backed by the public W&B GraphQL API.
//
// Surfaces the headline series of the public report "67B-A2B MoE on 10T
// tokens": train cross-entropy loss and Paloma macro loss, both against
// cumulative training tokens. The runs are NOT hardcoded — each snapshot
// first reads the report's spec and charts whatever runs its runset pins,
// so edits to the report (e.g. adding a new resume run) show up here
// without a code change. Which metrics are charted stays fixed in CHARTS.
//
// marin-community is a public W&B entity, so both the report spec and the
// run history are readable anonymously — no API key to provision or
// rotate. Each run costs one request: `sampledHistory(specs: [...])`
// returns every chart's downsampled series in a single response.

const WANDB_GRAPHQL_URL = "https://api.wandb.ai/graphql";
const ENTITY = "marin-community";
const PROJECT = "marin_moe";
const REPORT_URL =
  "https://wandb.ai/marin-community/marin_moe/reports/67B-A2B-MoE-on-10T-tokens--VmlldzoxNzM1OTMxMQ";
// Base64 view id — the trailing token of the report URL (padding restored).
const REPORT_VIEW_ID = "VmlldzoxNzM1OTMxMQ==";

// Cumulative-token counter logged on every step (train and eval alike),
// so it works as a shared x-axis across metrics and across the resume
// boundary, where `_step` would restart. Tokens (not steps) also keep the
// hero runs comparable: they use different batch sizes, so a step means a
// different amount of training in each.
const X_KEY = "throughput/total_tokens";
const SAMPLES_PER_SERIES = 800;
const FETCH_TIMEOUT_MS = 15_000;

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
  { key: "mfu", title: "MFU (%)", metric: "throughput/mfu" },
];

export interface WandbPoint {
  x: number; // cumulative training tokens
  y: number;
}

export interface WandbRunSeries {
  run: string; // full W&B run name; the frontend truncates for display
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

async function wandbGraphql<T>(query: string, variables: Record<string, unknown>): Promise<T> {
  const ac = new AbortController();
  const timer = setTimeout(
    () => ac.abort(new Error(`wandb query timed out after ${FETCH_TIMEOUT_MS}ms`)),
    FETCH_TIMEOUT_MS,
  );
  try {
    const res = await fetch(WANDB_GRAPHQL_URL, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query, variables }),
      signal: ac.signal,
    });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`wandb graphql ${res.status}: ${body.slice(0, 300)}`);
    }
    const payload = (await res.json()) as { data?: T; errors?: { message: string }[] };
    if (payload.errors?.length) {
      throw new Error(`wandb graphql: ${payload.errors[0].message}`);
    }
    if (!payload.data) {
      throw new Error("wandb graphql: empty response");
    }
    return payload.data;
  } finally {
    clearTimeout(timer);
  }
}

const REPORT_QUERY = `
query Report($id: ID!) {
  view(id: $id) {
    displayName
    spec
  }
}`;

// The slice of a report spec we consume: the first panel-grid block's
// first runset, whose `selections.tree` lists the pinned run names.
interface ReportSpecBlock {
  type?: string;
  metadata?: {
    runSets?: {
      selections?: { tree?: string[] };
    }[];
  };
}

async function fetchReportRuns(): Promise<{ title: string; runs: string[] }> {
  const data = await wandbGraphql<{
    view?: { displayName?: string; spec?: string } | null;
  }>(REPORT_QUERY, { id: REPORT_VIEW_ID });
  if (!data.view?.spec) {
    throw new Error(`wandb report ${REPORT_VIEW_ID} not found`);
  }
  const spec = JSON.parse(data.view.spec) as { blocks?: ReportSpecBlock[] };
  const grid = (spec.blocks ?? []).find((b) => b.type === "panel-grid");
  const runs = grid?.metadata?.runSets?.[0]?.selections?.tree ?? [];
  if (runs.length === 0) {
    throw new Error("wandb report pins no runs (runset selections empty)");
  }
  return { title: data.view.displayName ?? "wandb report", runs };
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

async function fetchRunSeries(run: string): Promise<{ state: string; byChart: WandbPoint[][] }> {
  const specs = CHARTS.map((chart) =>
    JSON.stringify({ keys: [X_KEY, chart.metric], samples: SAMPLES_PER_SERIES }),
  );
  const data = await wandbGraphql<{
    project?: { run?: { state?: string; sampledHistory?: HistoryRow[][] } | null } | null;
  }>(RUN_HISTORY_QUERY, { entity: ENTITY, project: PROJECT, run, specs });
  const runData = data.project?.run;
  if (!runData) {
    throw new Error(`wandb run ${run} not found in ${ENTITY}/${PROJECT}`);
  }
  const histories = runData.sampledHistory ?? [];
  return {
    state: runData.state ?? "unknown",
    byChart: CHARTS.map((chart, i) => toPoints(histories[i] ?? [], chart.metric)),
  };
}

export async function wandbSnapshot(): Promise<WandbSnapshot> {
  const fetchedAt = new Date().toISOString();
  try {
    const { title, runs } = await fetchReportRuns();
    const perRun = await Promise.all(runs.map((run) => fetchRunSeries(run)));
    const charts = CHARTS.map((chart, i) => ({
      key: chart.key,
      title: chart.title,
      series: runs.map((run, j) => ({
        run,
        state: perRun[j].state,
        points: perRun[j].byChart[i],
      })),
    }));
    return { reportTitle: title, reportUrl: REPORT_URL, charts, fetchedAt };
  } catch (err) {
    return {
      reportTitle: "67B-A2B MoE on 10T tokens",
      reportUrl: REPORT_URL,
      charts: [],
      fetchedAt,
      error: (err as Error).message,
    };
  }
}
