import {
  asyncBufferFromUrl,
  parquetMetadataAsync,
  parquetReadObjects,
  rowIndex,
  type AsyncBuffer,
  type FileMetaData,
  type ParquetQueryFilter,
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

export type SourceDetails = {
  converters: Counts
  modes: Counts
  languages: Counts
  dockerfiles: Counts
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
  source_details?: Record<string, SourceDetails>
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

export type TaskFilters = {
  source: string
  converter: string
  mode: string
  tag: string
  environment: string
  query: string
}

export type ConverterPolicy = {
  transformation: string
  graders: string[]
}

export const converterPolicies: Record<string, ConverterPolicy> = {
  agent_calendar: { transformation: 'Extract hidden calendar state and deterministic constraints.', graders: ['script'] },
  all_puzzles: { transformation: 'Separate the hidden answer and declare exact or symbolic comparison.', graders: ['exact', 'math'] },
  code_contests: { transformation: 'Move hidden cases into a standard stdin/stdout test bundle.', graders: ['stdio'] },
  codeforces: { transformation: 'Keep rows with enough hidden cases and standardize execution.', graders: ['stdio'] },
  judge_rubric: { transformation: 'Move the hidden rubric into a one-answer judge specification.', graders: ['judge'] },
  nemotron_competitive: { transformation: 'Package the 50 hidden cases as a stdin/stdout test suite.', graders: ['stdio'] },
  nemotron_if_structured: { transformation: 'Declare the schema or deterministic calendar checker.', graders: ['json-schema', 'script'] },
  nemotron_ifeval: { transformation: 'Translate declared constraints into deterministic IFEval checks.', graders: ['ifeval'] },
  nemotron_math: { transformation: 'Move the gold out of the prompt and normalize typed math comparison.', graders: ['math'] },
  nemotron_mcqa: { transformation: 'Normalize choices and keep only an unambiguous option-letter gold.', graders: ['mcq'] },
  nemotron_multichallenge: { transformation: 'Preserve the rubric and its explicit positive or negated polarity.', graders: ['judge'] },
  nemotron_openqa: { transformation: 'Separate the reference answer from the prompt and judge rubric.', graders: ['judge'] },
  nemotron_reasoning: { transformation: 'Package the hidden grid, program tests, or library scorer.', graders: ['exact', 'script', 'reasoning-gym'] },
  nemotron_structured_outputs: { transformation: 'Translate the declared JSON, YAML, TOML, XML, or CSV structure.', graders: ['json-schema', 'xml-elements', 'csv-columns'] },
  nl2bash: { transformation: 'Run the oracle command and compare its sandbox effects.', graders: ['script'] },
  prompt_injection: { transformation: 'Hide the injected action and evaluate the next action deterministically.', graders: ['script'] },
  python_unit_tests: { transformation: 'Package self-contained tests and the expected solution path.', graders: ['pytest'] },
  swe_patched: { transformation: 'Preserve the repository, patch, and trusted FAIL_TO_PASS tests.', graders: ['pytest', 'script'] },
  swe_trusted_paths: { transformation: 'Strip the oracle patch and retain trusted repository test paths.', graders: ['pytest'] },
  taco: { transformation: 'Keep sufficiently tested rows and package hidden stdin/stdout cases.', graders: ['stdio'] },
}

const sourceGraderOverrides: Record<string, string[]> = {
  'laion__nemotron-gym-arc-agi-python-inductive-v2': ['script'],
  'laion__nemotron-gym-arc-agi-transductive-v3': ['exact'],
  'laion__nemotron-gym-reasoning-gym-v2': ['reasoning-gym'],
  'laion__nemotron-gym-instruction-following-calendar-v3': ['script'],
  'laion__nemotron-gym-instruction-following-structured-v3': ['json-schema'],
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

export async function tasks(rowStart: number, rowEnd: number, filters?: TaskFilters): Promise<ParquetTask[]> {
  const [parquet, dataset, parquetMetadata] = await Promise.all([file(), manifest(), metadata()])
  const predicates: ParquetQueryFilter[] = []
  if (filters?.source) predicates.push({ source: { $eq: filters.source } })
  if (filters?.converter) predicates.push({ converter: { $eq: filters.converter } })
  if (filters?.mode) predicates.push({ mode: { $eq: filters.mode } })
  if (filters?.tag) predicates.push({ tags: { $in: [filters.tag] } })
  if (filters?.environment) {
    const ids = Object.entries(dataset.dockerfiles)
      .filter(([, value]) => value.base_image === filters.environment)
      .map(([id]) => id)
    predicates.push({ dockerfile_id: { $in: ids } })
  }
  const filter = predicates.length > 1 ? { $and: predicates } : predicates[0]
  const rows = await parquetReadObjects({
    file: parquet,
    metadata: parquetMetadata,
    columns: METADATA_COLUMNS,
    filter,
    rowStart,
    rowEnd,
    includeRowIndex: true,
    useOffsetIndex: true,
  })
  const query = filters?.query.trim().toLocaleLowerCase()
  return rows
    .map((value) => asTask(value, dataset))
    .filter((value) => !query || value.path.toLocaleLowerCase().includes(query))
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

function subsetWithTotal(entries: [string, number][], total: number): string[] | undefined {
  for (let mask = 1; mask < 1 << entries.length; mask += 1) {
    let sum = 0
    const names: string[] = []
    for (let bit = 0; bit < entries.length; bit += 1) {
      if ((mask & (1 << bit)) === 0) continue
      sum += entries[bit][1]
      names.push(entries[bit][0])
    }
    if (sum === total) return names
  }
  return undefined
}

function inferredSourceConverters(dataset: Manifest): Record<string, string[]> {
  const found: Record<string, Set<string>> = {}
  for (const environment of Object.values(dataset.dockerfiles)) {
    let sources = Object.entries(environment.sources)
    const converters = Object.entries(environment.converters).sort((a, b) => a[1] - b[1])
    for (const [converter, tasks] of converters) {
      const names = converters.length === 1 ? sources.map(([source]) => source) : subsetWithTotal(sources, tasks)
      if (!names) continue
      for (const source of names) (found[source] ??= new Set()).add(converter)
      sources = sources.filter(([source]) => !names.includes(source))
    }
  }
  return Object.fromEntries(Object.entries(found).map(([source, converters]) => [source, [...converters].sort()]))
}

export function sourceDetails(dataset: Manifest): Record<string, SourceDetails> {
  if (dataset.source_details) return dataset.source_details
  const converters = inferredSourceConverters(dataset)
  return Object.fromEntries(
    Object.entries(converters).map(([source, names]) => [
      source,
      {
        converters: Object.fromEntries(names.map((name) => [name, dataset.by_source[source]?.converted ?? 0])),
        modes: Object.fromEntries(
          (sourceGraderOverrides[source] ?? names.flatMap((name) => converterPolicies[name]?.graders ?? [])).map((mode) => [
            mode,
            dataset.by_source[source]?.converted ?? 0,
          ]),
        ),
        languages: {},
        dockerfiles: {},
      },
    ]),
  )
}

export function count(value: number): string {
  return value.toLocaleString('en-US')
}

export function shortRef(value: string): string {
  return value.length > 12 ? value.slice(0, 12) : value
}
