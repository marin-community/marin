import {
  asyncBufferFromUrl,
  parquetMetadataAsync,
  parquetReadObjects,
  rowIndex,
  type AsyncBuffer,
  type FileMetaData,
  type ParquetRow,
} from 'hyparquet'

const DATA = '/tasktrove/data'
export const parquetUrl = `${DATA}/tasks/part-00000.parquet`

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

export type ParquetTask = {
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

let manifestLoading: Promise<Manifest> | undefined

export function manifest(): Promise<Manifest> {
  manifestLoading ??= json<Manifest>('manifest.json')
  return manifestLoading
}

let parquetFile: Promise<AsyncBuffer> | undefined

function file(): Promise<AsyncBuffer> {
  parquetFile ??= asyncBufferFromUrl({ url: parquetUrl })
  return parquetFile
}

let metadataLoading: Promise<FileMetaData> | undefined

async function metadata(): Promise<FileMetaData> {
  metadataLoading ??= file().then(parquetMetadataAsync)
  return metadataLoading
}

export async function rowCount(): Promise<number> {
  return Number((await metadata()).num_rows)
}

function asTask(value: ParquetRow, dataset: Manifest): ParquetTask {
  const position = value[rowIndex]
  if (position === undefined) throw new Error('The Parquet reader did not return a row position.')
  return {
    row: position,
    source: value.source,
    path: value.path,
    family: value.family,
    converter: value.converter,
    mode: value.mode,
    dockerfile_id: value.dockerfile_id,
    environment: dataset.dockerfiles[value.dockerfile_id]?.base_image ?? value.dockerfile_id,
    language: value.language ?? '',
    tags: value.tags ?? [],
    has_solution: value.has_solution,
  }
}

export async function tasks(rowStart: number, rowEnd: number): Promise<ParquetTask[]> {
  const [parquet, dataset, parquetMetadata] = await Promise.all([file(), manifest(), metadata()])
  const rows = await parquetReadObjects({
    file: parquet,
    metadata: parquetMetadata,
    columns: METADATA_COLUMNS,
    rowStart,
    rowEnd,
    includeRowIndex: true,
    useOffsetIndex: true,
  })
  return rows.map((value) => asTask(value, dataset))
}

export async function task(row: number): Promise<ParquetTask> {
  const found = await tasks(row, row + 1)
  if (!found[0]) throw new Error(`No Parquet row ${row}.`)
  return found[0]
}

export async function archive(row: number): Promise<Uint8Array> {
  const rows = await parquetReadObjects({
    file: await file(),
    metadata: await metadata(),
    columns: ['task_binary'],
    rowStart: row,
    rowEnd: row + 1,
    utf8: false,
    useOffsetIndex: true,
  })
  const binary = rows[0]?.task_binary
  if (binary instanceof Uint8Array) return binary
  if (binary instanceof ArrayBuffer) return new Uint8Array(binary)
  if (ArrayBuffer.isView(binary)) return new Uint8Array(binary.buffer, binary.byteOffset, binary.byteLength)
  throw new Error(`Row ${row} has no task_binary value.`)
}

export function count(value: number): string {
  return value.toLocaleString('en-US')
}

export function shortRef(value: string): string {
  return value.length > 12 ? value.slice(0, 12) : value
}
