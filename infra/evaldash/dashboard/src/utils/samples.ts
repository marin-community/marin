// Per-sample outcome and a short "why" hint, derived from the fields the reader already returns.
// Kept here (not persisted) so the list and the viewer render the same summary from one place.
import type { SampleRow } from '@/types/api'

export type SampleOutcome = 'passed' | 'failed' | 'ungraded' | 'errored'

// The correctness filters a sample list offers. 'ungraded' is a real bucket, not a synonym for wrong.
export type SampleFilter = 'all' | 'correct' | 'incorrect' | 'ungraded'

/** The filters to show for a task: 'ungraded' only appears once the task reports unscored samples. */
export function sampleFilters(ungradedCount: number | null | undefined): SampleFilter[] {
  const base: SampleFilter[] = ['all', 'correct', 'incorrect']
  return (ungradedCount ?? 0) > 0 ? [...base, 'ungraded'] : base
}

/** The grader's recorded error, if it emitted one (Harbor writes `{reward, error}` into detail). */
function graderError(row: SampleRow): string | null {
  const detail = row.grading?.detail
  if (!detail) return null
  try {
    const parsed = JSON.parse(detail) as { error?: unknown }
    return typeof parsed.error === 'string' && parsed.error ? parsed.error : null
  } catch {
    return null
  }
}

/**
 * One sample's outcome. A recorded grader error is `errored`; an unscored sample (no primary metric,
 * `correct === null`) is `ungraded`; otherwise pass/fail follows `correct`. `errored` is only claimed
 * when the grader actually reported one — it is never inferred from an empty answer.
 */
export function sampleOutcome(row: SampleRow): SampleOutcome {
  if (graderError(row)) return 'errored'
  if (row.correct === null || row.correct === undefined) return 'ungraded'
  return row.correct ? 'passed' : 'failed'
}

const TONE: Record<SampleOutcome, 'success' | 'danger' | 'warning' | 'muted'> = {
  passed: 'success',
  failed: 'danger',
  errored: 'warning',
  ungraded: 'muted',
}

export function outcomeTone(outcome: SampleOutcome): 'success' | 'danger' | 'warning' | 'muted' {
  return TONE[outcome]
}

const CHIP_CLASS: Record<'success' | 'danger' | 'warning' | 'muted', string> = {
  success: 'bg-status-success-bg text-status-success border-status-success-border',
  danger: 'bg-status-danger-bg text-status-danger border-status-danger-border',
  warning: 'bg-status-warning-bg text-status-warning border-status-warning-border',
  muted: 'bg-surface-sunken text-text-muted border-surface-border',
}

/** Tailwind classes for an outcome chip (bg/text/border), shared by the list and the viewer. */
export function outcomeChipClass(outcome: SampleOutcome): string {
  return CHIP_CLASS[outcomeTone(outcome)]
}

function truncate(text: string, n = 48): string {
  const clean = text.replace(/\s+/g, ' ').trim()
  return clean.length > n ? `${clean.slice(0, n)}…` : clean
}

/**
 * A one-line reason for the outcome, extracted from the normalized fields: the picked-vs-gold choice
 * for MCQ, extracted-vs-target for generation, the reward or recorded error for an agentic trial.
 */
export function sampleHint(row: SampleRow): string {
  const error = graderError(row)
  if (error) return `errored: ${truncate(error)}`

  if (row.kind === 'multiple_choice') {
    const pick = row.model_choice !== null ? (row.choices?.[row.model_choice]?.label ?? '?') : '—'
    const gold = row.target_choice !== null ? (row.choices?.[row.target_choice]?.label ?? '?') : '—'
    if (sampleOutcome(row) === 'ungraded') return `picked ${pick} (ungraded)`
    return row.correct ? `picked ${pick} ✓` : `picked ${pick}, gold ${gold}`
  }

  if (row.kind === 'agentic') {
    const reward = row.grading?.score
    return reward !== null && reward !== undefined ? `reward ${reward}` : 'agentic trial'
  }

  // generation
  const extracted = row.extracted ?? row.output ?? ''
  const target = row.target_text ?? ''
  if (sampleOutcome(row) === 'ungraded') return `“${truncate(extracted)}” (ungraded)`
  return row.correct ? `“${truncate(extracted)}” ✓` : `“${truncate(extracted)}” vs “${truncate(target)}”`
}
