export type DashboardPanelKind = 'table' | 'stat' | 'timeseries'
export type DashboardPanelWidth = 'half' | 'full'

export interface DashboardVariable {
  name: string
  label: string
  default: string
  /**
   * SQL listing the values this variable can take, one per row, read from the
   * first column. It is a dashboard query like any other — same macros, same
   * endpoint — so a list can answer "which jobs ran in the selected window"
   * rather than being a fixed enum. It may not reference other variables, which
   * would make the order they load in significant. Without it the variable is a
   * free-text field.
   */
  optionsSql?: string
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
