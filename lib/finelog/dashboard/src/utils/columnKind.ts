/**
 * Classify a result column so the table, the detail panel, and the chart all
 * render it the same way.
 *
 * A query result carries only Arrow types, which distinguish Int64 from Utf8 but
 * not an epoch-millisecond count from a row count, nor a JSON document from a
 * log line. The extra signal comes from the column name and a sample of its
 * values, so classification stays a pure function of one result set.
 */

/** Plausible epoch-millisecond range: 2001-09-09 to 2100-01-01. */
const EPOCH_MS_MIN = 1_000_000_000_000
const EPOCH_MS_MAX = 4_102_444_800_000

export type ColumnKind = 'timestamp' | 'number' | 'json' | 'text'

/** True for an Arrow type string denoting a numeric column. */
export function isNumericType(arrowType: string | undefined): boolean {
  return /^(Int|Uint|Float|Decimal)/i.test(arrowType ?? '')
}

/** True for an Arrow type string denoting an instant. */
export function isTemporalType(arrowType: string | undefined): boolean {
  return /^(Timestamp|Date)/i.test(arrowType ?? '')
}

/**
 * An integer column holding epoch milliseconds. Requires both a `_ms` name
 * suffix and sampled values inside a plausible epoch range, so a `duration_ms`
 * or a row count is never mistaken for an instant.
 */
function isEpochMillisColumn(name: string, arrowType: string | undefined, sample: unknown[]): boolean {
  if (!isNumericType(arrowType) || !/_ms$/.test(name)) return false
  const numbers = sample.filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
  if (numbers.length === 0) return false
  return numbers.every((v) => v >= EPOCH_MS_MIN && v <= EPOCH_MS_MAX)
}

/** True when every sampled string parses as a JSON object or array. */
function isJsonColumn(sample: unknown[]): boolean {
  const strings = sample.filter((v): v is string => typeof v === 'string' && v.length > 0)
  if (strings.length === 0) return false
  return strings.every((s) => {
    const head = s.trimStart()[0]
    if (head !== '{' && head !== '[') return false
    try {
      JSON.parse(s)
      return true
    } catch {
      return false
    }
  })
}

/**
 * Classify `name` from its Arrow type and up to `limit` non-null values drawn
 * from `rows`.
 */
export function classifyColumn(
  name: string,
  arrowType: string | undefined,
  rows: Record<string, unknown>[],
  limit = 20,
): ColumnKind {
  const sample: unknown[] = []
  for (const row of rows) {
    const value = row[name]
    if (value !== null && value !== undefined) sample.push(value)
    if (sample.length >= limit) break
  }
  if (isTemporalType(arrowType) || isEpochMillisColumn(name, arrowType, sample)) return 'timestamp'
  if (isNumericType(arrowType)) return 'number'
  if (isJsonColumn(sample)) return 'json'
  return 'text'
}

/** Classify every column of a result set. */
export function classifyColumns(
  columns: string[],
  types: Record<string, string>,
  rows: Record<string, unknown>[],
): Record<string, ColumnKind> {
  return Object.fromEntries(columns.map((c) => [c, classifyColumn(c, types[c], rows)]))
}
