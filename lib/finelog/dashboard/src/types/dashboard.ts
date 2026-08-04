export type DashboardPanelKind = 'table' | 'stat' | 'timeseries'
export type DashboardPanelWidth = 'half' | 'full'

export interface DashboardVariable {
  name: string
  label: string
  default: string
}

export interface DashboardPanel {
  id: string
  title: string
  kind: DashboardPanelKind
  description: string
  sql: string
  width: DashboardPanelWidth
}

export interface DashboardDefinition {
  version: 1
  id: string
  title: string
  description: string
  variables: DashboardVariable[]
  panels: DashboardPanel[]
}

export interface SavedDashboard {
  definition: DashboardDefinition
  createdAtMs: number
  updatedAtMs: number
}

export interface DashboardListResponse {
  dashboards: SavedDashboard[]
}
