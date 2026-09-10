import { asyncBufferFromUrl, parquetReadObjects, rowIndex, type AsyncBuffer, type ParquetRow } from 'hyparquet'

const DATA = '/tasktrove/data'
export const parquetUrl = `${DATA}/tasks.parquet`

export type Counts = Record<string, number>

export type Dockerfile = {
  base_image: string
  tasks: number
  converters: Counts
  sources: Counts
}

export type Manifest = {
  tasktrove: { hf_id: string; revision: string }
  verify_tool_ref: string
  input_tasks: number
  clean_tasks: number
  by_status: Counts
  by_mode: Counts
  by_tag: Counts
  by_source: Record<string, Counts>
  source_verdicts: Record<string, { verdict: 'keep' | 'drop'; family: string; reason: string }>
  by_converter: Record<string, Counts>
  dockerfiles: Record<string, Dockerfile>
}

export type CatalogTask = {
  row: number
  source: string
  path: string
  family: string
  converter: string
  mode: string
  dockerfile_id: string
  environment: string
  language: string
  tags: string[]
  has_solution: boolean
}

export type Corpus = { manifest: Manifest; tasks: CatalogTask[] }

const METADATA_COLUMNS = [
  'source',
  'path',
  'family',
  'converter',
  'mode',
  'dockerfile_id',
  'language',
  'tags',
  'has_solution',
]

async function json<T>(path: string): Promise<T> {
  const response = await fetch(`${DATA}/${path}`)
  if (!response.ok) throw new Error(`${path}: ${response.status}`)
  return response.json() as Promise<T>
}

let parquetFile: Promise<AsyncBuffer> | undefined

function file(): Promise<AsyncBuffer> {
  parquetFile ??= asyncBufferFromUrl({ url: parquetUrl })
  return parquetFile
}

let loading: Promise<Corpus> | undefined

export function corpus(): Promise<Corpus> {
  loading ??= Promise.all([
    json<Manifest>('manifest.json'),
    file().then((parquet) =>
      parquetReadObjects({ file: parquet, columns: METADATA_COLUMNS, includeRowIndex: true })
    ),
  ]).then(([manifest, rows]) => ({
    manifest,
    tasks: rows.map((value: ParquetRow) => {
      const position = value[rowIndex]
      if (position === undefined) throw new Error('The Parquet reader did not return row positions.')
      return {
        row: position,
        source: value.source,
        path: value.path,
        family: value.family,
        converter: value.converter,
        mode: value.mode,
        dockerfile_id: value.dockerfile_id,
        environment: manifest.dockerfiles[value.dockerfile_id]?.base_image ?? value.dockerfile_id,
        language: value.language ?? '',
        tags: value.tags ?? [],
        has_solution: value.has_solution,
      }
    }),
  }))
  return loading
}

export async function archive(task: CatalogTask): Promise<Uint8Array> {
  const rows = await parquetReadObjects({
    file: await file(),
    columns: ['task_binary'],
    rowStart: task.row,
    rowEnd: task.row + 1,
    utf8: false,
  })
  const binary = rows[0]?.task_binary
  if (binary instanceof Uint8Array) return binary
  if (binary instanceof ArrayBuffer) return new Uint8Array(binary)
  if (ArrayBuffer.isView(binary)) return new Uint8Array(binary.buffer, binary.byteOffset, binary.byteLength)
  throw new Error(`Row ${task.row} has no task_binary value.`)
}

export function count(value: number): string {
  return value.toLocaleString('en-US')
}

export function shortRef(value: string): string {
  return value.length > 12 ? value.slice(0, 12) : value
}
