// Shapes returned by the evaldash server (src/server.py). Kept in sync with the
// dict shapes that RecordStore, cluster.py, and samples.py produce.

export const RUN_STATUS = {
  SUCCEEDED: 'succeeded',
  FAILED: 'failed',
  ARTIFACT_FAILED: 'artifact_failed',
  INFRA_FAILED: 'infra_failed',
} as const

export type RunStatus = (typeof RUN_STATUS)[keyof typeof RUN_STATUS]

export interface RunRow {
  run_id: string
  group_id: string | null
  created_at: string
  version: string | null
  user_name: string | null
  model_name: string | null
  model_location: string | null
  eval_name: string | null
  mechanism: string | null
  backend: string | null
  platform: string | null
  accelerator: string | null
  region: string | null
  status: string
  results_path: string | null
  git_sha: string | null
  image_digest: string | null
  error: string | null
  tasks: string[]
  jobs: Record<string, string>
}

// What a cell's interval covers. `identified` includes the contribution of items the run attempted
// but never graded; `sampling_only` means the run reports no attempted count, so the interval holds
// only if it graded everything it started -- which the record does not establish.
export const INTERVAL_KIND = {
  IDENTIFIED: 'identified',
  SAMPLING_ONLY: 'sampling_only',
} as const

export type IntervalKind = (typeof INTERVAL_KIND)[keyof typeof INTERVAL_KIND]

// Properties of a measurement worth showing beside it. `marin.evaluation.eval_stats.ResultFlag` is
// the full set; these are the ones the panel annotates a cell with.
export const RESULT_FLAG = {
  NO_ANSWERS: 'no_answers',
} as const

// A flag says what was observed, not what to conclude: unextractable answers score zero like wrong
// ones do, so the cell is marked and explained rather than withheld or corrected.
//
// Only flags nothing else on the page already shows belong here. `attrition` has its own per-cause
// table and a coverage figure on the cell, and `capped` is a declared setting rather than a finding;
// repeating either as a warning says the same thing twice and dulls the ones that matter.
export const FLAG_NOTES: Record<string, string> = {
  no_answers: 'no graded item yielded an extractable answer — suspect the grader, not only the model',
  degenerate_stderr: 'the harness recorded a standard error of exactly zero, which no real sample supports',
}

// One benchmark result: the rate over graded items, the interval around it, how much of the
// attempted item set was graded, and the run it came from. `low` is the ranking key -- `value` is a
// complete-case rate and is biased upward when a run lost items.
export interface PanelCell {
  value: number
  low: number
  high: number
  interval_kind: IntervalKind
  metric: string
  metric_kind: string
  n_scored: number
  n_attempted: number | null
  coverage: number | null
  errors: Record<string, number>
  item_cap: number | null
  flags: string[]
  run_id: string
  created_at: string
  version: string | null
  git_sha: string
  eval_runtime: string
}

// Why a (model, benchmark) has no cell: a rejected run (failed status, coverage below the gate, the
// wrong cohort) or a run that produced no metric at all.
export interface MissingCell {
  reason: string
  run_id: string
  status: string
  created_at: string
}

// A cross-benchmark aggregate always travels with the protocol that defines it: which benchmarks,
// which per-benchmark metric, and what it did about the ones a model never ran.
export interface PanelAggregate {
  value: number
  low: number
  high: number
  interval_kind: IntervalKind
  covered: number
  total: number
  panel: string[]
  missing_policy: string
  metrics: string[]
  /** Distinct harness versions the averaged cells came from; more than one is worth seeing. */
  runtimes: string[]
}

export interface PanelRow {
  model: string
  archived: boolean
  cells: Record<string, PanelCell>
  missing: Record<string, MissingCell>
  aggregate: PanelAggregate | null
  covered: number
}

export interface PanelRequest {
  min_coverage: number
  cohort: string
  cohort_version: string | null
  completeness: string
  filters: Record<string, string>
  model_query: string | null
  statuses: string[]
}

export interface Panel {
  benchmarks: string[]
  panel: string[]
  rows: PanelRow[]
  request: PanelRequest
}

// One model's gap to a benchmark's leader: the 95% interval for the difference, and whether it
// clears zero. Overlapping model intervals do not settle an ordering; this does.
export interface Difference {
  low: number
  high: number
  separated: boolean
}

export interface ComparisonRow {
  benchmark: string
  shared: boolean
  leader: string
  cells: Record<string, PanelCell>
  differences: Record<string, Difference>
}

// Head-to-head between named models (/api/compare): per-benchmark cells and gaps, plus each model's
// aggregate over the shared benchmarks (null when it is missing one of them).
export interface Comparison {
  models: string[]
  benchmarks: string[]
  shared: string[]
  rows: ComparisonRow[]
  aggregates: Record<string, PanelAggregate | null>
}

export interface EvalSuite {
  suite: string
  evals: string[]
}

export interface Meta {
  models: string[]
  evals: string[]
  suites: EvalSuite[]
  archived_models: string[]
  users: string[]
  statuses: string[]
  versions: string[]
  // Run properties a panel can be filtered on, each with the values actually present.
  facets: Record<string, string[]>
  current_user: string | null
  store: string
}

export interface RecordParseFailure {
  path: string
  error: string
}

export interface PrefixProbe {
  prefix: string
  last_probe_time: string | null
  last_success_time: string | null
  record_count: number | null
  error: string | null
  parse_failures: RecordParseFailure[]
}

export interface StoreInfo {
  backend: string
  instance: string | null
  database: string | null
  record_count: number
  catalog_generation: number | null
  snapshot_updated_at: string | null
}

export interface Status {
  store: StoreInfo
  ingest: {
    interval_seconds: number
    revalidate_after_seconds: number | null
    last_pass_time: string | null
    prefixes: PrefixProbe[]
  }
}

export interface EvalTask {
  name: string
  num_fewshot: number | null
}

// The canonical record.json shape (records.EvalRunRecord). `headline` is not stored on the record --
// the run-detail endpoint computes it (the rolled-up primary metric) and attaches it to the response.
export interface EvalRecord {
  run_id: string
  group_id: string
  version: string | null
  description: string | null
  created_at: string
  user: string
  model: { name: string; location: string; backend: string }
  eval: { name: string; mechanism: string; tasks: EvalTask[] }
  hardware: { platform: string; accelerator: string; region_or_cluster: string }
  status: string
  error: string | null
  results_path: string
  metrics: Record<string, Record<string, number>>
  jobs: Record<string, string>
  log_tails: Record<string, string[]>
  provenance: { git_sha: string; eval_runtime: string; launch_host: string }
  timing: { started_at: string; finished_at: string | null } | null
  serving: {
    tensor_parallel_size: number | null
    data_parallel_size: number | null
    max_model_len: number | null
    max_gen_tokens: number | null
    extra: Record<string, string>
  } | null
  headline: PanelCell | null
}

// --- Live Iris/finelog protobuf JSON (cluster.py) ---

export interface ProtoTimestamp {
  epoch_ms?: string | number
}

export interface JobAttempt {
  attempt_id: number
  state: string
  worker_id: string
  exit_code: number
  error: string
  started_at?: ProtoTimestamp
  finished_at?: ProtoTimestamp
  is_worker_failure: boolean
  attempt_uid: string
}

export interface JobTask {
  task_id: string
  state: string
  worker_id: string
  exit_code: number
  error: string
  started_at?: ProtoTimestamp
  finished_at?: ProtoTimestamp
  current_attempt_id: number
  attempts: JobAttempt[]
}

export interface JobInfo {
  state: string
  error: string
  exit_code: number
  started_at?: ProtoTimestamp
  finished_at?: ProtoTimestamp
  name: string
  status_message: string
}

export interface JobRole {
  role: string
  job_path: string
  reachable: boolean
  error: string | null
  job: JobInfo | null
  tasks: JobTask[]
}

export interface JobsResponse {
  roles: JobRole[]
}

export interface LogEntry {
  timestamp?: ProtoTimestamp
  source: string
  data: string
  attempt_id: number
  level: string
  key: string
  seq: string | number
}

export interface LogsResponse {
  reachable: boolean
  error: string | null
  source: string
  role: string
  entries: LogEntry[]
}

// --- Per-sample browser (samples.py, mirroring marin.evaluation.archive.EvalSample) ---

export interface SampleTasksResponse {
  available: boolean
  error: string | null
  tasks: { task: string; files: number }[]
}

export type SampleKind = 'multiple_choice' | 'generation' | 'agentic'

export interface ChatMessage {
  role: string
  content: string
}

export interface SampleChoice {
  label: string
  text: string
  loglikelihood: number | null
  is_greedy: boolean | null
}

// How one prediction was scored (marin.evaluation.archive.Grading). `method` names the grader
// (`lm-eval:<metric>`, `harbor:<verifier>`, `judge:<model>`); `detail` is the grader's raw output
// as a JSON string, the escape hatch for anything the typed fields do not carry.
export interface SampleGrading {
  method: string
  metric: string | null
  filter: string | null
  score: number | null
  passed: boolean | null
  detail: string
}

// One evaluated question: the prompt, the model's answer, the gold answer, and its scores.
// `prompt_text` and `prompt_messages` are mutually exclusive; `choices`/`model_choice`/
// `target_choice` are set for `multiple_choice` samples, `output`/`extracted` for `generation`
// samples, and `trajectory_uri` for `agentic` samples. The one unbounded payload, the agentic
// trajectory, is referenced by URI and lazy-loaded on demand.
export interface SampleRow {
  task: string
  doc_id: string
  kind: SampleKind
  prompt_text: string | null
  prompt_messages: ChatMessage[] | null
  choices: SampleChoice[] | null
  model_choice: number | null
  target_choice: number | null
  output: string | null
  extracted: string | null
  target_text: string | null
  trajectory_uri: string | null
  grading: SampleGrading | null
  metrics: Record<string, number>
  correct: boolean | null
  doc: string
}

// One sample-referenced artifact (a trajectory) resolved to text by the server's
// artifact endpoint. `available` is false with a `reason` when the object is out of tree,
// missing, unreadable, or over the size cap — mirroring the logs endpoint's degradation.
export interface ArtifactResponse {
  available: boolean
  reason: string | null
  uri: string
  media_type: string
  size: number | null
  truncated: boolean
  text: string | null
}

// Agent Trajectory Interchange Format (ATIF): one agentic run's steps. A step is either a user
// turn (the task/observation feed) or an agent turn carrying its message, tool calls, the resulting
// observation, and per-step token metrics. Fields are optional because ATIF minor versions add them.
export interface TrajectoryToolCall {
  tool_call_id?: string
  function_name: string
  arguments: Record<string, unknown>
}

// One observation entry. Most carry `content` (a tool's stdout/terminal state); some carry no
// content — a sub-agent delegation records a `subagent_trajectory_ref` instead.
export interface TrajectoryObservationResult {
  source_call_id?: string
  content?: string
  subagent_trajectory_ref?: string
}

export interface TrajectoryObservation {
  results?: TrajectoryObservationResult[]
}

export interface TrajectoryStep {
  step_id: number
  timestamp?: string
  source: string
  model_name?: string
  message?: string
  tool_calls?: TrajectoryToolCall[]
  observation?: TrajectoryObservation | null
  metrics?: Record<string, number>
}

export interface TrajectoryAgent {
  name: string
  version?: string
  model_name?: string
}

export interface Trajectory {
  schema_version?: string
  session_id?: string
  agent?: TrajectoryAgent
  steps: TrajectoryStep[]
  final_metrics?: Record<string, number>
}

// Unpaginated per-task outcome counts; `ungraded` is its own bucket, apart from `incorrect`.
export interface SampleCounts {
  all: number
  correct: number
  incorrect: number
  ungraded: number
}

export interface SamplesResponse {
  available: boolean
  error: string | null
  task: string
  primary_metric: string | null
  metric_columns: string[]
  /** Extraction filters this task was scored under; empty only when its rows carry no filter. */
  extraction_filters: string[]
  /** The filter the returned page was drawn from. */
  extraction_filter: string | null
  total: number
  offset: number
  limit: number
  counts?: SampleCounts
  rows: SampleRow[]
}

// --- Score-over-time + groups ---

// A point on a (model, benchmark) score-over-time series: one run's cell plus its terminal status.
export interface HistoryPoint extends PanelCell {
  status: string
}

export interface HistoryResponse {
  model: string
  task: string
  points: HistoryPoint[]
}

export interface GroupSibling {
  run_id: string
  eval_name: string | null
  model_name: string | null
  status: string
  created_at: string | null
}

export interface GroupResponse {
  group_id: string | null
  siblings: GroupSibling[]
}

// --- Model detail (/api/models/{model}) ---

// A model runs each benchmark many times; a cohort is one version's launch. Newest cohort first.
export interface ModelCohort {
  version: string | null
  created_at: string
  n_evals: number
  n_succeeded: number
  group_id: string | null
}

export interface ModelRun {
  run_id: string
  eval_name: string
  status: string
  created_at: string | null
  version: string | null
  headline: PanelCell | null
  /** Why the run produced no headline; null when it did. */
  gap_reason: string | null
}

// Everything the model view needs in one call: the cohort list for the version selector, every
// succeeded run's score per benchmark for the sparklines, and every run for the run list. The page
// derives each cohort's per-benchmark cells from `runs`, so no precomputed cell map is sent.
export interface ModelDetail {
  model: string
  location: string | null
  backend: string | null
  user: string | null
  current_version: string | null
  cohorts: ModelCohort[]
  history: Record<string, HistoryPoint[]>
  runs: ModelRun[]
}

// --- Agentic failure review (POST /api/runs/{run_id}/samples/review) ---

export interface ReviewCategory {
  label: string
  count: number
  doc_ids: string[]
}

export interface ReviewSummary {
  categories: ReviewCategory[]
  narrative: string
}

// `available` is false with a `reason` when the reviewer is not configured (no API key/SDK) or
// there are no samples — mirroring the logs/artifact endpoints rather than erroring.
export interface ReviewResponse {
  available: boolean
  reason: string | null
  model: string | null
  task: string
  filter: string
  n_reviewed: number
  summary: ReviewSummary | null
}

// One eval within a launch (a serve group), with its headline score.
export interface GroupMember {
  run_id: string
  eval_name: string
  status: string
  created_at: string
  headline: PanelCell | null
}

// A launch: all evals run against one model by one serve group, newest first (/api/groups).
export interface LaunchGroup {
  group_id: string
  model_name: string
  version: string | null
  description: string | null
  user_name: string
  accelerator: string | null
  created_at: string
  status: RunStatus | 'mixed'
  n_evals: number
  n_succeeded: number
  evals: GroupMember[]
}
