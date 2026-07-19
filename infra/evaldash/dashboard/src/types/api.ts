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
  cells: Record<string, MatrixCell>
}

export interface LeaderboardEntry {
  model: string
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

export interface Meta {
  models: string[]
  evals: string[]
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
  provenance: { git_sha: string; evalchemy_image: string; launch_host: string }
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

export type SampleKind = 'multiple_choice' | 'generation'

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

// One evaluated question: the prompt, the model's answer, the gold answer, and its scores.
// `prompt_text` and `prompt_messages` are mutually exclusive; `choices`/`model_choice`/
// `target_choice` are set for `multiple_choice` samples, `output`/`extracted` for `generation`
// samples.
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
  metrics: Record<string, number>
  correct: boolean | null
  doc: string
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
