/**
 * Composable and helpers for the raw query API.
 *
 * Wraps the ExecuteRawQuery RPC. Row data comes back as JSON-encoded arrays;
 * parseRows() converts them to keyed objects.
 */
import { controllerRpcCall } from './useRpc'
import type { ColumnMeta, RawQueryResponse } from '@/types/rpc'

export type { ColumnMeta, RawQueryResponse }

export interface RawQueryRequest {
  sql: string
}

/** Execute a raw SQL query (admin-only). */
export function executeRawQuery(request: RawQueryRequest): Promise<RawQueryResponse> {
  return controllerRpcCall<RawQueryResponse>('ExecuteRawQuery', request as unknown as Record<string, unknown>)
}

/**
 * Parse JSON-encoded row arrays into keyed objects using column metadata.
 *
 * Each element of `rows` is a JSON array of scalar values aligned with
 * `columns`. This function zips them into Record<string, unknown> objects
 * keyed by column name, which is much more ergonomic for template rendering.
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
