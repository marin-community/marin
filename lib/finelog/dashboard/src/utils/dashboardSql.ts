export interface DashboardMacroValues {
  fromMs: number
  toMs: number
  intervalMs: number
  variables: Record<string, string>
}

const MACRO_PATTERN = /\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}/g
const UNEXPANDED_MACRO_PATTERN = /\{\{[^}]*\}\}/

function integerLiteral(value: number, name: string): string {
  if (!Number.isSafeInteger(value)) throw new Error(`${name} must be a safe integer`)
  return String(value)
}

export function sqlStringLiteral(value: string): string {
  if (value.includes('\0')) throw new Error('dashboard variables must not contain NUL bytes')
  return `'${value.replace(/'/g, "''")}'`
}

/** Expand the complete dashboard macro vocabulary into typed SQL literals. */
export function expandDashboardSql(sql: string, values: DashboardMacroValues): string {
  if (values.intervalMs <= 0) throw new Error('interval_ms must be positive')
  const builtins: Record<string, string> = {
    from_ms: integerLiteral(values.fromMs, 'from_ms'),
    to_ms: integerLiteral(values.toMs, 'to_ms'),
    interval_ms: integerLiteral(values.intervalMs, 'interval_ms'),
  }
  const expanded = sql.replace(MACRO_PATTERN, (_match, name: string) => {
    const builtin = builtins[name]
    if (builtin !== undefined) return builtin
    if (!(name in values.variables)) throw new Error(`unknown dashboard variable: ${name}`)
    return sqlStringLiteral(values.variables[name])
  })
  const unknown = expanded.match(UNEXPANDED_MACRO_PATTERN)
  if (unknown) throw new Error(`invalid dashboard macro: ${unknown[0]}`)
  return expanded
}

/** Choose a stable interval that keeps a time-series query near 120 points. */
export function dashboardIntervalMs(fromMs: number, toMs: number): number {
  const target = Math.max(1, (toMs - fromMs) / 120)
  const intervals = [1_000, 5_000, 15_000, 30_000, 60_000, 300_000, 900_000, 3_600_000, 21_600_000]
  return intervals.find((interval) => interval >= target) ?? 86_400_000
}
