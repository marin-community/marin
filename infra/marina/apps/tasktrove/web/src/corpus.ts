// What this site carries of its own: the manifest of the dataset's files, the
// audit's review of each source, and the audit's label for each sampled task.
// The three JSON files come from the app's data directory, which the kernel
// serves; the page reads and holds them once.

/** The kernel serves this app's data directory here (`.data/tasktrove` locally, the bucket in production). */
const CORPUS = '/tasktrove/data'

export type Source = {
  source: string
  file: string
  size: number
  rows: number
  groups: number
  group_rows: number
  largest_group_bytes: number
  /** The row of this source's first task in the dataset. */
  offset: number
}

export type Review = {
  source: string
  template_description: string
  shellsim_verdict: 'yes' | 'partial' | 'no'
  cheapest_unlock: string
  unlock_notes: string
  quality_notes: string
}

export type Label = {
  id: string
  source: string
  path: string
  row: number
  summary: string
  task_kind: string
  agent_needs: string[]
  verifier_needs: string[]
  verifier_mechanism: string
  shellsim_now: 'yes' | 'partial' | 'no'
  shellsim_with: string
  verifier_portable: string
  interesting: number
  well_defined: number
  hack_risk: number
  hack_vector: string
  defects: string
}

export type Corpus = {
  sources: Source[]
  reviews: Map<string, Review>
  labels: Label[]
  /** Labels by dataset row, which is how a task page finds its own. */
  labelled: Map<number, Label>
  total: number
}

async function json<T>(path: string): Promise<T> {
  const response = await fetch(path)
  if (!response.ok) throw new Error(`${path}: ${response.status}`)
  return response.json() as Promise<T>
}

let loading: Promise<Corpus> | undefined

export function corpus(): Promise<Corpus> {
  loading ??= (async () => {
    const [manifest, reviews, labels] = await Promise.all([
      json<Omit<Source, 'offset'>[]>(`${CORPUS}/files.json`),
      json<Review[]>(`${CORPUS}/sources.json`),
      json<Label[]>(`${CORPUS}/labels.json`),
    ])
    let offset = 0
    const sources = manifest.map((entry) => {
      const source = { ...entry, offset }
      offset += entry.rows
      return source
    })
    return {
      sources,
      reviews: new Map(reviews.map((review) => [review.source, review])),
      labels,
      labelled: new Map(labels.map((label) => [label.row, label])),
      total: offset,
    }
  })()
  return loading
}

/** The source a dataset row belongs to. */
export function sourceOf(sources: Source[], row: number): Source | undefined {
  return sources.find((source) => row >= source.offset && row < source.offset + source.rows)
}

const UNITS = ['B', 'KB', 'MB', 'GB']

export function bytes(n: number): string {
  let unit = 0
  let value = n
  while (value >= 1000 && unit < UNITS.length - 1) {
    value /= 1000
    unit++
  }
  return `${value < 10 && unit > 0 ? value.toFixed(1) : Math.round(value)} ${UNITS[unit]}`
}

export function count(n: number): string {
  return n.toLocaleString('en-US')
}
