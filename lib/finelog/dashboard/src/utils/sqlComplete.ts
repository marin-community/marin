/**
 * Identifier completion for the SQL editor.
 *
 * Deliberately not a parser. It reads the token under the caret and the few
 * words before it, which is enough to know whether the writer is naming a table
 * or a column, and ranks the store's own vocabulary against what they have
 * typed. Anything it cannot classify falls back to offering everything, because
 * a wrong guess that hides the column someone wants is worse than a long list.
 */

export type CompletionKind = 'namespace' | 'column' | 'keyword'

export interface Completion {
  /** What the identifier is called; what the list shows. */
  value: string
  /** What accepting it inserts — `value`, quoted when bare SQL would not parse. */
  insert: string
  kind: CompletionKind
  /** Where the value comes from: a column's namespace, or a keyword's group. */
  detail: string
}

/**
 * Identifiers the SQL dialect would otherwise swallow, and so must be inserted
 * quoted.
 *
 * `cluster` is a column of both `telemetry_v1` and `log` and the one most worth
 * grouping by, but `CLUSTER BY` is dialect syntax, so `SELECT name, cluster FROM
 * …` is a parse error while `SELECT cluster, name FROM …` is fine. It is the
 * only collision across the 124 distinct column names registered on the marin
 * hub, so this is a list rather than the dialect's whole keyword set.
 */
const NEEDS_QUOTING = new Set(['cluster'])

/**
 * `name` as it must be written in SQL: bare where that parses, quoted where it
 * would not. A namespace containing a dot (`iris.task`) always needs quoting —
 * bare, it reads as schema-qualified and resolves to nothing.
 */
export function quoteIdentifier(name: string): string {
  const bare = !name.includes('.') && !NEEDS_QUOTING.has(name.toLowerCase())
  return bare ? name : `"${name}"`
}

/** A namespace and the columns it exposes, as the completer needs them. */
export interface NamespaceColumns {
  namespace: string
  columns: { name: string; type: string }[]
}

const KEYWORDS = [
  'SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'LIMIT', 'HAVING', 'JOIN',
  'LEFT JOIN', 'INNER JOIN', 'ON', 'AS', 'AND', 'OR', 'NOT', 'IN', 'LIKE',
  'BETWEEN', 'IS NULL', 'IS NOT NULL', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END',
  'DISTINCT', 'WITH', 'UNION ALL', 'DESC', 'ASC',
]

const FUNCTIONS = [
  'count', 'sum', 'avg', 'min', 'max', 'approx_distinct', 'stddev',
  'date_trunc', 'to_timestamp_millis', 'now', 'extract', 'coalesce',
  'contains', 'prefix', 'regexp_matches', 'starts_with', 'length',
  'lower', 'upper', 'substr', 'split_part', 'cast',
]

/** The word being typed at `caret`, and where it starts. */
export function tokenAt(sql: string, caret: number): { text: string; start: number } {
  let start = caret
  while (start > 0 && /[A-Za-z0-9_."]/.test(sql[start - 1])) start -= 1
  return { text: sql.slice(start, caret), start }
}

/** The last keyword before `start`, uppercased — `FROM`, `SELECT`, `WHERE`, … */
function precedingKeyword(sql: string, start: number): string {
  const words = sql.slice(0, start).trim().split(/[\s(,]+/).filter(Boolean)
  for (let i = words.length - 1; i >= 0; i--) {
    const word = words[i].toUpperCase()
    if (/^[A-Z]+$/.test(word)) return word
  }
  return ''
}

/** Namespaces named in the statement, so their columns rank above the rest. */
function referencedNamespaces(sql: string, known: NamespaceColumns[]): Set<string> {
  const named = new Set<string>()
  for (const [, name] of sql.matchAll(/\b(?:from|join)\s+"?([A-Za-z_][A-Za-z0-9_.]*)"?/gi)) {
    named.add(name)
  }
  return new Set(known.filter((n) => named.has(n.namespace)).map((n) => n.namespace))
}

/**
 * Rank `candidates` against `prefix`: exact prefix matches first, then
 * substring matches, each shortest-first so the closest name is reachable
 * without reading the list.
 */
function rank(candidates: Completion[], prefix: string): Completion[] {
  if (!prefix) return candidates
  const needle = prefix.toLowerCase().replace(/"/g, '')
  const scored: { c: Completion; score: number }[] = []
  for (const c of candidates) {
    const value = c.value.toLowerCase()
    if (value.startsWith(needle)) scored.push({ c, score: 0 })
    else if (value.includes(needle)) scored.push({ c, score: 1 })
  }
  return scored
    .sort((a, b) => a.score - b.score || a.c.value.length - b.c.value.length)
    .map((s) => s.c)
}

/**
 * Completions for the caret position, most relevant first.
 *
 * After `FROM` or `JOIN` only a table can follow, so the list is namespaces
 * alone. Everywhere else columns lead — that is what most of a query is made of
 * — with the columns of namespaces this statement already mentions ahead of the
 * rest, then functions, then keywords.
 */
export function completionsFor(
  sql: string,
  caret: number,
  schema: NamespaceColumns[],
  limit = 12,
): Completion[] {
  const { text, start } = tokenAt(sql, caret)
  const namespaces: Completion[] = schema.map((n) => ({
    value: n.namespace,
    insert: quoteIdentifier(n.namespace),
    kind: 'namespace',
    detail: `${n.columns.length} columns`,
  }))
  if (['FROM', 'JOIN', 'INTO', 'UPDATE'].includes(precedingKeyword(sql, start))) {
    return rank(namespaces, text).slice(0, limit)
  }

  const referenced = referencedNamespaces(sql, schema)
  // One entry per column name — the same name in two namespaces is the same
  // thing to type — but the detail has to say so, or it would credit the column
  // to whichever namespace happened to be listed first.
  const byName = new Map<string, { type: string; namespaces: string[]; relevant: boolean }>()
  for (const ns of schema) {
    const relevant = referenced.size === 0 || referenced.has(ns.namespace)
    for (const col of ns.columns) {
      const entry = byName.get(col.name)
      if (entry) {
        entry.namespaces.push(ns.namespace)
        entry.relevant ||= relevant
      } else {
        byName.set(col.name, { type: col.type, namespaces: [ns.namespace], relevant })
      }
    }
  }
  const columns: Completion[] = []
  const trailing: Completion[] = []
  for (const [name, { type, namespaces: holders, relevant }] of byName) {
    const where = holders.length === 1 ? holders[0] : `${holders.length} namespaces`
    ;(relevant ? columns : trailing).push({
      value: name,
      insert: quoteIdentifier(name),
      kind: 'column',
      detail: `${where} · ${type}`,
    })
  }
  const vocabulary: Completion[] = [
    ...columns,
    ...trailing,
    ...FUNCTIONS.map((f): Completion => ({ value: f, insert: f, kind: 'keyword', detail: 'function' })),
    ...namespaces,
    ...KEYWORDS.map((k): Completion => ({ value: k, insert: k, kind: 'keyword', detail: 'keyword' })),
  ]
  return rank(vocabulary, text).slice(0, limit)
}
