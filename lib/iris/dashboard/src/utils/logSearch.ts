// Client-side search over the log lines already on screen, and detection of the
// lines worth jumping to when hunting for a failure.
//
// Search is deliberately separate from the server-side `substring` filter: a
// filter drops non-matching rows, while a search only marks them, so the
// surrounding context stays visible. Both exist because they answer different
// questions ("show me only these lines" vs "where in these lines is X").

import type { LogSegment } from './logLinks'

/** Half-open `[start, end)` character range of one match within a line. */
export interface MatchRange {
  start: number
  end: number
}

export interface SearchMatcher {
  /** Non-overlapping match ranges within `text`, in ascending order. */
  find(text: string): MatchRange[]
}

/** A query that is blank, unparseable, or compiled and ready to run. */
export type CompiledSearch =
  | { kind: 'empty' }
  | { kind: 'invalid'; message: string }
  | { kind: 'ok'; matcher: SearchMatcher }

// A single pathological line (a serialized tensor, a base64 blob) must not be
// able to generate an unbounded highlight list.
const MAX_MATCHES_PER_LINE = 200

function literalMatcher(query: string, caseSensitive: boolean): SearchMatcher {
  const needle = caseSensitive ? query : query.toLowerCase()
  return {
    find(text: string): MatchRange[] {
      const haystack = caseSensitive ? text : text.toLowerCase()
      const ranges: MatchRange[] = []
      let from = 0
      while (ranges.length < MAX_MATCHES_PER_LINE) {
        const at = haystack.indexOf(needle, from)
        if (at < 0) break
        ranges.push({ start: at, end: at + needle.length })
        from = at + needle.length
      }
      return ranges
    },
  }
}

function regexMatcher(pattern: RegExp): SearchMatcher {
  return {
    find(text: string): MatchRange[] {
      pattern.lastIndex = 0
      const ranges: MatchRange[] = []
      let match: RegExpExecArray | null
      while ((match = pattern.exec(text)) !== null && ranges.length < MAX_MATCHES_PER_LINE) {
        // A zero-width match (`x*`, `^`) never advances lastIndex on its own.
        if (match[0].length === 0) {
          pattern.lastIndex += 1
          continue
        }
        ranges.push({ start: match.index, end: match.index + match[0].length })
      }
      return ranges
    },
  }
}

/** Compile a search query, reporting an unparseable regex rather than throwing. */
export function compileSearch(
  query: string,
  options: { caseSensitive: boolean; useRegex: boolean },
): CompiledSearch {
  if (!query) return { kind: 'empty' }
  if (!options.useRegex) {
    return { kind: 'ok', matcher: literalMatcher(query, options.caseSensitive) }
  }
  try {
    const flags = options.caseSensitive ? 'g' : 'gi'
    return { kind: 'ok', matcher: regexMatcher(new RegExp(query, flags)) }
  } catch (e) {
    return { kind: 'invalid', message: e instanceof Error ? e.message : String(e) }
  }
}

/** A `LogSegment` that also records whether it is part of a search match. */
export interface HighlightedSegment extends LogSegment {
  match?: boolean
}

/**
 * Re-split link `segments` at `ranges`, marking the pieces a search matched.
 *
 * `segments` must tile the original line in order (as `parseLogLinks` returns
 * them) — the ranges are offsets into that line, not into any one segment. A
 * match straddling a link boundary is split at the boundary, so the link stays
 * clickable and both halves stay highlighted.
 */
export function highlightSegments(
  segments: LogSegment[],
  ranges: MatchRange[],
): HighlightedSegment[] {
  if (ranges.length === 0) return segments
  const out: HighlightedSegment[] = []
  let offset = 0
  for (const segment of segments) {
    const segStart = offset
    const segEnd = offset + segment.text.length
    offset = segEnd
    // Cut points inside this segment, in ascending order, from every range edge
    // that falls strictly within it.
    const cuts = new Set<number>()
    for (const range of ranges) {
      if (range.start > segStart && range.start < segEnd) cuts.add(range.start)
      if (range.end > segStart && range.end < segEnd) cuts.add(range.end)
    }
    const bounds = [segStart, ...[...cuts].sort((a, b) => a - b), segEnd]
    for (let i = 0; i < bounds.length - 1; i++) {
      const start = bounds[i]
      const end = bounds[i + 1]
      if (start === end) continue
      const covered = ranges.some((r) => r.start <= start && r.end >= end)
      out.push({
        ...segment,
        text: segment.text.slice(start - segStart, end - segStart),
        ...(covered ? { match: true } : {}),
      })
    }
  }
  return out
}

// Lines that name a failure. Every match is a stop for jump-to-exception, so
// the vocabulary is deliberately narrow: it recognizes the head of a traceback
// or a fatal banner, never the frames beneath one and never prose that merely
// names an exception type. A wider net would land the reader on retry chatter.
const EXCEPTION_PATTERNS = [
  /Traceback \(most recent call last\)/,
  /Fatal Python error/,
  /detected fatal errors/,
  /Segmentation fault/,
  /core dumped/,
  /\bOOMKilled\b/,
  /\bCUDA error\b/,
  /\bNCCL\b.*\berror\b/i,
  /\bRESOURCE_EXHAUSTED\b/,
  /\bDEADLINE_EXCEEDED\b/,
  /\bout of memory\b/i,
  /\bCoscheduled sibling\b/,
  // A bare terminal exception line, anchored so that prose mentioning an error
  // type mid-sentence ("retrying after ConnectionError") does not match.
  /^\s*\w*(?:Error|Exception):/,
]

/** The `source` iris stamps on the failure lines it injects into a task's log. */
const INJECTED_ERROR_SOURCE = 'error'

/** Whether a log entry is a place a failure hunt should stop. */
export function isExceptionEntry(entry: { data?: string; source?: string }): boolean {
  if (entry.source === INJECTED_ERROR_SOURCE) return true
  const data = entry.data ?? ''
  return EXCEPTION_PATTERNS.some((pattern) => pattern.test(data))
}

/**
 * Collapse runs of indices closer together than `gap` to their first element.
 *
 * A traceback trips several exception patterns a few lines apart (its header,
 * then its terminal `ValueError:`). They are one failure, so a jump-to-next
 * walks failures, not lines.
 */
export function groupNearbyIndices(indices: number[], gap: number): number[] {
  const grouped: number[] = []
  for (const index of indices) {
    const previous = grouped[grouped.length - 1]
    if (previous === undefined || index - previous > gap) grouped.push(index)
  }
  return grouped
}
