export interface Overview {
  case_counts: Record<string, number>
  active_investigation: { case_id: string; title: string; started_at: string; loom_session_url: string | null } | null
  last_poll_at: string | null
}

export interface CaseRow {
  id: string
  trigger: string
  state: string
  priority: number
  title: string
  receiver: string
  group_key: string
  outcome: string | null
  summary: string | null
  opened_at: string
  updated_at: string
  signal_count: number
  firing_count: number
  clusters: string[]
  loom_session_url: string | null
}

export interface Signal {
  id: string
  fingerprint: string
  signal_generation: number
  state: string
  alert_name: string
  severity: string
  cluster: string | null
  namespace: string | null
  object_kind: string | null
  object_name: string | null
  summary: string
  labels: Record<string, string>
  annotations: Record<string, string>
  values: Record<string, unknown>
  generator_url: string | null
}

export interface ChatBlock {
  turn: number
  seq: number
  kind: string
  payload: Record<string, unknown>
  created_at: string
}

export interface CaseDetail {
  case: CaseRow & { loom_session_id: string | null; agent_session_state: string | null; question: string | null }
  signals: Signal[]
  turns: Array<{ id: string; kind: string; state: string; requested_by: string; error: string | null }>
  events: Array<{ id: number; event_type: string; actor: string; created_at: string }>
  chat: { blocks: ChatBlock[]; live_turn: number | null }
  chat_error?: string
}
