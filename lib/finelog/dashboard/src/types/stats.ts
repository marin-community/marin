/** TypeScript shapes mirroring finelog.stats.* messages over Connect-JSON.
 *
 * Manually maintained — only the fields the dashboard reads. Proto enums
 * serialize as their declared name strings (e.g. "COLUMN_TYPE_STRING").
 */

export type ColumnType =
  | 'COLUMN_TYPE_UNKNOWN'
  | 'COLUMN_TYPE_STRING'
  | 'COLUMN_TYPE_INT64'
  | 'COLUMN_TYPE_FLOAT64'
  | 'COLUMN_TYPE_BOOL'
  | 'COLUMN_TYPE_TIMESTAMP_MS'
  | 'COLUMN_TYPE_BYTES'
  | 'COLUMN_TYPE_INT32'

export interface ProtoColumn {
  name: string
  type?: ColumnType
  nullable?: boolean
  index?: ProtoColumnIndex
}

export interface ProtoColumnIndex {
  trigram?: boolean
  exactValues?: string[]
  valueCounts?: boolean
}

export interface ProtoCoveringProjection {
  name: string
  predicateColumn: string
  predicateValues?: string[]
  columns?: string[]
}

export interface ProtoSchema {
  columns?: ProtoColumn[]
  keyColumn?: string
  projections?: ProtoCoveringProjection[]
}

export function shortColumnType(t: ColumnType | undefined): string {
  if (!t) return ''
  return t.replace('COLUMN_TYPE_', '').toLowerCase()
}
