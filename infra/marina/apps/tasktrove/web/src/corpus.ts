const DATA = '/tasktrove/data'
const API = '/tasktrove/api'

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

export type TaskPage = {
  rows: ParquetTask[]
  total: number
  offset: number
  limit: number
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

export async function rowCount(): Promise<number> {
  return (await manifest()).clean_tasks
}

async function api<T>(path: string): Promise<T> {
  const response = await fetch(`${API}/${path}`)
  if (!response.ok) throw new Error(`${path}: ${response.status}`)
  return response.json() as Promise<T>
}

export async function tasks(offset: number, limit: number, filters: TaskFilters): Promise<TaskPage> {
  const parameters = new URLSearchParams({ offset: String(offset), limit: String(limit) })
  for (const [name, value] of Object.entries(filters)) if (value.trim()) parameters.set(name, value.trim())
  return api<TaskPage>(`tasks?${parameters}`)
}

export async function task(row: number): Promise<ParquetTask> {
  return api<ParquetTask>(`tasks/${row}`)
}

export async function archive(row: number): Promise<Uint8Array> {
  const response = await fetch(`${API}/tasks/${row}/archive`)
  if (!response.ok) throw new Error(`tasks/${row}/archive: ${response.status}`)
  return new Uint8Array(await response.arrayBuffer())
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
