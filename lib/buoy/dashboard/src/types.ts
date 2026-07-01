export interface RunRow {
  id: string
  name: string
  user: string | null
  state: string
  created_at: string
}

export interface HistoryMeta {
  rows: number
  parts: number
  columns: string[]
  last_step: number | null
}

export interface Profile {
  artifact_name: string
  logdir: string
  size_bytes: number
}

export interface Manifest {
  entity: string
  project: string
  run_id: string
  display_name: string
  state: string
  url: string
  user: string | null
  created_at: string
  notes: string | null
  tags: string[]
  history: HistoryMeta
  profile: Profile | null
}

// Columnar series from /api/metrics: { key: { x: number[], y: number[] } }.
export type MetricSeries = Record<string, { x: number[]; y: number[] }>

export type TabId = 'summary' | 'charts' | 'profile'
