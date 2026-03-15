/**
 * Composable and helpers for the generic query API.
 *
 * Wraps the ExecuteQuery / ExecuteRawQuery RPCs with typed interfaces that
 * mirror the protobuf query DSL from cluster.proto. Row data comes back as
 * JSON-encoded arrays; parseRows() converts them to keyed objects.
 */
import { controllerRpcCall } from './useRpc'
import type { ColumnMeta, QueryResponse, RawQueryResponse } from '@/types/rpc'

// Re-export the response types for convenience so callers can import from
// either this module or types/rpc.
export type { ColumnMeta, QueryResponse, RawQueryResponse }

// ---------------------------------------------------------------------------
// Query DSL types (mirrors the protobuf Query message tree)
// ---------------------------------------------------------------------------

export type AggregateFunc =
  | 'AGG_NONE'
  | 'AGG_COUNT'
  | 'AGG_SUM'
  | 'AGG_AVG'
  | 'AGG_MIN'
  | 'AGG_MAX'
  | 'AGG_COUNT_STAR'

export type ComparisonOp =
  | 'CMP_EQ'
  | 'CMP_NE'
  | 'CMP_LT'
  | 'CMP_LE'
  | 'CMP_GT'
  | 'CMP_GE'

export type LogicalOp = 'LOGICAL_AND' | 'LOGICAL_OR'

export type NullOp = 'NULL_IS_NULL' | 'NULL_IS_NOT_NULL'

export type JoinKind = 'JOIN_INNER' | 'JOIN_LEFT'

export type SortDir = 'SORT_ASC' | 'SORT_DESC'

export interface QueryColumn {
  name: string
  table?: string
  func?: AggregateFunc
  alias?: string
}

export interface QueryTable {
  name: string
  alias?: string
}

export interface QueryValue {
  stringValue?: string
  intValue?: string
  floatValue?: number
  boolValue?: boolean
}

export interface ComparisonFilter {
  column: string
  table?: string
  op: ComparisonOp
  value: QueryValue
}

export interface LogicalFilter {
  op: LogicalOp
  operands: QueryFilter[]
}

export interface NotFilter {
  operand: QueryFilter
}

export interface InFilter {
  column: string
  table?: string
  values: QueryValue[]
}

export interface LikeFilter {
  column: string
  table?: string
  pattern: string
}

export interface NullCheckFilter {
  column: string
  table?: string
  op: NullOp
}

export interface BetweenFilter {
  column: string
  table?: string
  low: QueryValue
  high: QueryValue
}

export interface QueryFilter {
  comparison?: ComparisonFilter
  logical?: LogicalFilter
  not?: NotFilter
  in?: InFilter
  like?: LikeFilter
  nullCheck?: NullCheckFilter
  between?: BetweenFilter
}

export interface QueryJoin {
  table: QueryTable
  kind?: JoinKind
  leftColumn: string
  leftTable?: string
  rightColumn: string
  rightTable?: string
}

export interface QueryGroupBy {
  columns: QueryColumn[]
}

export interface QueryOrderBy {
  column: string
  table?: string
  direction?: SortDir
}

export interface Query {
  from: QueryTable
  columns?: QueryColumn[]
  where?: QueryFilter
  joins?: QueryJoin[]
  groupBy?: QueryGroupBy
  orderBy?: QueryOrderBy[]
  limit?: number
  offset?: number
}

export interface QueryRequest {
  query: Query
}

export interface RawQueryRequest {
  sql: string
}

// ---------------------------------------------------------------------------
// RPC wrappers
// ---------------------------------------------------------------------------

/** Execute a structured query against the controller database. */
export function executeQuery(request: QueryRequest): Promise<QueryResponse> {
  return controllerRpcCall<QueryResponse>('ExecuteQuery', request as unknown as Record<string, unknown>)
}

/** Execute a raw SQL query (admin-only). */
export function executeRawQuery(request: RawQueryRequest): Promise<RawQueryResponse> {
  return controllerRpcCall<RawQueryResponse>('ExecuteRawQuery', request as unknown as Record<string, unknown>)
}

// ---------------------------------------------------------------------------
// Row parsing
// ---------------------------------------------------------------------------

/**
 * Parse JSON-encoded row arrays into keyed objects using column metadata.
 *
 * Each element of `rows` is a JSON array of scalar values aligned with
 * `columns`. This function zips them into Record<string, unknown> objects
 * keyed by column name, which is much more ergonomic for template rendering.
 *
 * Example:
 *   columns = [{ name: "job_id", type: "text" }, { name: "state", type: "integer" }]
 *   rows = ['["abc", 3]']
 *   => [{ job_id: "abc", state: 3 }]
 */
export function parseRows(columns: ColumnMeta[], rows: string[]): Record<string, unknown>[] {
  const names = columns.map((c) => c.name)
  return rows.map((row) => {
    const values = JSON.parse(row) as unknown[]
    const record: Record<string, unknown> = {}
    for (let i = 0; i < names.length; i++) {
      record[names[i]] = values[i]
    }
    return record
  })
}
