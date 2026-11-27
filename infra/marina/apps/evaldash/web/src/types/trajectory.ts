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
  /** String outcome facts a step carries where a metric would be numeric: stop reason, verification. */
  annotations?: Record<string, string>
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
