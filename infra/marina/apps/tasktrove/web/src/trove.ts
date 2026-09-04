// The dataset, read from where it is published.
//
// Hugging Face's datasets-server answers a page of rows for a dataset by
// offset, and a task's `task_binary` cell comes back whole, base64-encoded.
// The manifest in `corpus.ts` says which rows are which source, so a page of
// one source is a page of the dataset at that source's offset.

import { decodeBase64, entries, gunzip, text, type Entry } from './tar'

const SERVER = 'https://datasets-server.huggingface.co'
const DATASET = 'open-thoughts/TaskTrove'
/** The most rows one request answers. */
export const PAGE = 100

export type Row = {
  row: number
  path: string
}

export type Task = {
  row: number
  path: string
  /** The gzipped tar, as stored. */
  size: number
  files: File[]
}

export type File = {
  path: string
  size: number
  directory: boolean
  /** The contents as text, or nothing when the file is not UTF-8. */
  text: string | undefined
}

type Cell = { row_idx: number; row: { path: string; task_binary: string }; truncated_cells: string[] }
type Page = { rows: Cell[]; num_rows_total: number; error?: string }

function query(endpoint: string, params: Record<string, string | number>): string {
  const search = new URLSearchParams({ dataset: DATASET, config: 'default', split: 'train' })
  for (const [key, value] of Object.entries(params)) search.set(key, String(value))
  return `${SERVER}/${endpoint}?${search}`
}

async function page(endpoint: string, params: Record<string, string | number>): Promise<Page> {
  const response = await fetch(query(endpoint, params))
  const body = (await response.json()) as Page
  if (!response.ok || body.error) throw new Error(body.error ?? `${endpoint}: ${response.status}`)
  return body
}

/** The paths of `length` rows from `offset`. */
export async function rows(offset: number, length: number): Promise<Row[]> {
  const found = await page('rows', { offset, length: Math.min(length, PAGE) })
  return found.rows.map((cell) => ({ row: cell.row_idx, path: cell.row.path }))
}

/**
 * Every row whose path contains `needle`, across the whole dataset.
 *
 * The server builds an index the first time it is asked and answers with an
 * error until it is done; the caller shows that error as what it is.
 */
export async function search(needle: string, offset: number): Promise<{ rows: Row[]; total: number }> {
  const found = await page('search', { query: needle, offset, length: PAGE })
  return {
    rows: found.rows.map((cell) => ({ row: cell.row_idx, path: cell.row.path })),
    total: found.num_rows_total,
  }
}

function file(entry: Entry): File {
  return {
    path: entry.path,
    size: entry.size,
    directory: entry.directory,
    text: entry.directory ? '' : text(entry.bytes),
  }
}

/** One task, opened: its files, out of the tar the row holds. */
export async function task(row: number): Promise<Task> {
  const found = await page('rows', { offset: row, length: 1 })
  const cell = found.rows[0]
  if (!cell || cell.row_idx !== row) throw new Error(`no row ${row}`)
  if (cell.truncated_cells.includes('task_binary')) {
    throw new Error(`row ${row} is too large for the dataset server to answer whole`)
  }
  const packed = decodeBase64(cell.row.task_binary)
  const archive = await gunzip(packed)
  return {
    row,
    path: cell.row.path,
    size: packed.length,
    files: entries(archive)
      .map(file)
      .sort((a, b) => a.path.localeCompare(b.path)),
  }
}
