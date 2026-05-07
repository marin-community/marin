/**
 * Decode Arrow IPC payloads returned by finelog.stats.StatsService.Query.
 *
 * Connect-JSON encodes `bytes` fields as base64. We decode to a Uint8Array,
 * hand it to apache-arrow's `tableFromIPC`, and convert to plain JS rows so
 * the rest of the dashboard can render without arrow-aware code.
 */
import { tableFromIPC, type Table } from 'apache-arrow'

export interface ArrowResult {
  columns: string[]
  rows: Record<string, unknown>[]
}

function base64ToUint8(b64: string): Uint8Array {
  const bin = atob(b64)
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out
}

export function decodeArrowIpc(arrowIpc: string | undefined | null): ArrowResult {
  if (!arrowIpc) return { columns: [], rows: [] }
  const bytes = base64ToUint8(arrowIpc)
  const table: Table = tableFromIPC(bytes)
  const columns = table.schema.fields.map((f) => f.name)
  const rows: Record<string, unknown>[] = []
  for (let i = 0; i < table.numRows; i++) {
    const row: Record<string, unknown> = {}
    for (const name of columns) {
      const col = table.getChild(name)
      const v = col?.get(i)
      row[name] = normalize(v)
    }
    rows.push(row)
  }
  return { columns, rows }
}

function normalize(v: unknown): unknown {
  if (v === null || v === undefined) return null
  if (typeof v === 'bigint') return Number(v)
  if (v instanceof Uint8Array) return `<bytes ${v.byteLength}>`
  if (v instanceof Date) return v.toISOString()
  return v
}
