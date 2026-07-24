// Shapes returned by the evaldash server (src/server.py). Kept in sync with the
// dict shapes that RecordStore, cluster.py, and samples.py produce.

export interface RunRow {
  run_id: string
  group_id: string | null
  created_at: string
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

// A matrix cell is the latest succeeded score (value + paired stderr) for a (model, task),
// or -- when no run ever succeeded there -- the latest run's failure status. value is null
// in the failure case; run_id always links to a real run.
export interface MatrixCell {
  status: string
  value: number | null
  stderr: number | null
  metric: string | null
  run_id: string
  created_at: string
}

export interface MatrixRow {
  model: string
  version: string | null
  archived: boolean
  cells: Record<string, MatrixCell>
}

export interface LeaderboardEntry {
  model: string
  version: string | null
  archived: boolean
  score: number | null
  stderr: number | null
  covered: number
  total: number
}

export interface Matrix {
  tasks: string[]
  rows: MatrixRow[]
  leaderboard: LeaderboardEntry[]
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
  current_user: string | null
  store: string
}

export interface PrefixProbe {
  prefix: string
  last_probe_time: string | null
  last_success_time: string | null
  record_count: number | null
  error: string | null
}

export interface StoreInfo {
  backend: string
  instance: string | null
  database: string | null
}

export interface Status {
  store: StoreInfo
  ingest: {
    interval_seconds: number
    last_pass_time: string | null
    prefixes: PrefixProbe[]
  }
}

export interface EvalTask {
  name: string
  num_fewshot: number | null
}

// The canonical record.json shape (records.EvalRunRecord).
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
  provenance: { git_sha: string; eval_image: string; launch_host: string }
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

// --- Per-sample browser (samples.py, mirroring marin.evaluation.samples.EvalSample) ---

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

// How one prediction was scored (marin.evaluation.samples.Grading). `method` names the grader
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
// samples, and `trajectory_uri` for `agentic` samples. The two unbounded payloads (the agentic
// trajectory, a prediction's raw exchange) are referenced by URI and lazy-loaded on demand.
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
  exchange_uri: string | null
  grading: SampleGrading | null
  metrics: Record<string, number>
  correct: boolean | null
  doc: string
}

// One sample-referenced artifact (a trajectory, an exchange) resolved to text by the server's
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

export interface SamplesResponse {
  available: boolean
  error: string | null
  task: string
  primary_metric: string | null
  metric_columns: string[]
  total: number
  offset: number
  limit: number
  counts?: { all: number; correct: number; incorrect: number }
  rows: SampleRow[]
}

// --- Score-over-time + groups ---

export interface HistoryPoint {
  run_id: string
  created_at: string | null
  value: number
  stderr: number | null
  metric: string
  status: string
  git_sha: string
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

// One eval within a launch (a serve group), with its headline score.
export interface GroupMember {
  run_id: string
  eval_name: string
  status: string
  created_at: string
  value: number | null
  metric: string | null
  stderr: number | null
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
  status: 'succeeded' | 'failed' | 'infra_failed' | 'mixed'
  n_evals: number
  n_succeeded: number
  evals: GroupMember[]
}
