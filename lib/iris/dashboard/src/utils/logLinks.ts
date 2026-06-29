// Detect navigable iris identifiers inside free-form log text and turn them
// into router links. These are deliberately quick regex checks against the
// rendered line — no store lookups — so a stray false positive just yields a
// dead-but-harmless link, never a wrong query.

/** A run of log text, optionally carrying a router target to link it to. */
export interface LogSegment {
  text: string
  to?: string
}

// worker-<slice>-<tpu_index>-<uuid8>. The slice id can itself contain dashes,
// so anchor on the trailing 8-hex uuid that every worker id ends with.
const WORKER_PATTERN = 'worker-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*-[0-9a-f]{8}'

// A task id is a slash path whose final segment is the numeric task index
// (e.g. /alice/train/3), optionally suffixed with :<attempt> (/alice/train/3:1).
// Requires at least one named segment before the index so a bare /3 or a
// file:line like foo.py:42 doesn't match.
const TASK_PATTERN = '(?:/[A-Za-z0-9][A-Za-z0-9._-]*)+/\\d+(?::\\d+)?'

const TOKEN_RE = new RegExp(`(?<worker>${WORKER_PATTERN})|(?<task>${TASK_PATTERN})`, 'g')

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

/**
 * Split a log line into plain-text and linked segments. Worker ids link to the
 * worker page; task/attempt paths link to the task (attempt) page. Returns a
 * single plain segment when nothing matches.
 */
export function parseLogLinks(text: string): LogSegment[] {
  const segments: LogSegment[] = []
  let last = 0
  for (const m of text.matchAll(TOKEN_RE)) {
    const start = m.index ?? 0
    if (start > last) segments.push({ text: text.slice(last, start) })
    const groups = m.groups ?? {}
    if (groups.worker) {
      segments.push({ text: m[0], to: `/worker/${groups.worker}` })
    } else {
      segments.push({ text: m[0], to: taskTargetFromMatch(m[0]) })
    }
    last = start + m[0].length
  }
  if (last < text.length) segments.push({ text: text.slice(last) })
  return segments
}
