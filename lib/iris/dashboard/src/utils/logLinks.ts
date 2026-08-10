// Detect navigable identifiers inside free-form log text and turn them into
// links. These are deliberately quick regex checks against the rendered line —
// no store lookups — so a stray false positive just yields a dead-but-harmless
// link, never a wrong query.

/** A run of log text, optionally carrying a link target. */
export interface LogSegment {
  text: string
  /** Router target, for identifiers that resolve to a dashboard page. */
  to?: string
  /** Absolute URL, for targets outside the dashboard. */
  href?: string
}

// worker-<slice>-<tpu_index>-<uuid8>. The slice id can itself contain dashes,
// so anchor on the trailing 8-hex uuid that every worker id ends with.
const WORKER_PATTERN = 'worker-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*-[0-9a-f]{8}'

// A task id is a slash path whose final segment is the numeric task index
// (e.g. /alice/train/3), optionally suffixed with :<attempt> (/alice/train/3:1).
// Requires at least one named segment before the index so a bare /3 or a
// file:line like foo.py:42 doesn't match.
const TASK_PATTERN = '(?:/[A-Za-z0-9][A-Za-z0-9._-]*)+/\\d+(?::\\d+)?'

// Anything up to whitespace or a delimiter that cannot appear unescaped in a
// URL. Trailing sentence punctuation is trimmed by `trimTrailingPunctuation`.
const URL_TAIL = '[^\\s<>"\'`]*'
const URL_PATTERN = `https?://${URL_TAIL}`
const GCS_PATTERN = `gs://[a-z0-9][a-z0-9._-]*(?:/${URL_TAIL})?`

// Order matters: a URL and a GCS path both contain runs that look like task
// paths, and JS alternation takes the first alternative that matches at the
// earliest position. Both start before the `/…` a task pattern would latch onto.
const TOKEN_RE = new RegExp(
  `(?<url>${URL_PATTERN})|(?<gcs>${GCS_PATTERN})|(?<worker>${WORKER_PATTERN})|(?<task>${TASK_PATTERN})`,
  'g',
)

// Punctuation that ends a sentence far more often than it ends a URL.
const TRAILING_PUNCTUATION = /[.,;:!?)\]}>'"]+$/

/** Strip sentence punctuation a greedy URL match swallowed (`see http://x/y.` ). */
function trimTrailingPunctuation(match: string): string {
  return match.replace(TRAILING_PUNCTUATION, '')
}

function jobIdOf(taskId: string): string {
  const slash = taskId.lastIndexOf('/')
  return slash > 0 ? taskId.slice(0, slash) : taskId
}

/** Router target for a task id, optionally pinned to a specific attempt. */
export function taskAttemptRoute(taskId: string, attemptId?: string | number): string {
  const base = `/job/${encodeURIComponent(jobIdOf(taskId))}/task/${encodeURIComponent(taskId)}`
  return attemptId !== undefined && attemptId !== '' ? `${base}?attempt=${attemptId}` : base
}

function taskTargetFromMatch(match: string): string {
  const colon = match.lastIndexOf(':')
  if (colon > 0) {
    return taskAttemptRoute(match.slice(0, colon), match.slice(colon + 1))
  }
  return taskAttemptRoute(match)
}

/** Cloud-console browser URL for a `gs://bucket/prefix` path. */
export function gcsConsoleUrl(uri: string): string {
  return `https://console.cloud.google.com/storage/browser/${uri.slice('gs://'.length)}`
}

/**
 * Split a log line into plain-text and linked segments. Worker ids link to the
 * worker page; task/attempt paths link to the task (attempt) page; `http(s)`
 * URLs and `gs://` paths link out of the dashboard. Returns a single plain
 * segment when nothing matches.
 */
export function parseLogLinks(text: string): LogSegment[] {
  const segments: LogSegment[] = []
  let last = 0
  for (const m of text.matchAll(TOKEN_RE)) {
    const start = m.index ?? 0
    const groups = m.groups ?? {}
    // A trimmed URL gives back the punctuation it swallowed, which must stay in
    // the line as plain text rather than vanish into the link.
    const matched = groups.url || groups.gcs ? trimTrailingPunctuation(m[0]) : m[0]
    if (!matched) continue
    if (start > last) segments.push({ text: text.slice(last, start) })
    if (groups.url) {
      segments.push({ text: matched, href: matched })
    } else if (groups.gcs) {
      segments.push({ text: matched, href: gcsConsoleUrl(matched) })
    } else if (groups.worker) {
      segments.push({ text: matched, to: `/worker/${groups.worker}` })
    } else {
      segments.push({ text: matched, to: taskTargetFromMatch(matched) })
    }
    last = start + matched.length
  }
  if (last < text.length) segments.push({ text: text.slice(last) })
  return segments
}
