// Per-sample outcome and a short "why" hint, derived from the fields the reader already returns.
// Kept here (not persisted) so the list and the viewer render the same summary from one place.
import type { SampleRow } from '@/types/api'

export type SampleOutcome = 'passed' | 'failed' | 'scored' | 'ungraded' | 'errored'

// The correctness filters a sample list offers. 'ungraded' is a real bucket, not a synonym for wrong.
export type SampleFilter = 'all' | 'correct' | 'incorrect' | 'scored' | 'ungraded'

/** Offer score filtering for continuous tasks and correctness filtering for binary tasks. */
export function sampleFilters(kind: 'binary' | 'continuous' | null, ungradedCount: number | null | undefined): SampleFilter[] {
  const base: SampleFilter[] = kind === 'continuous' ? ['all', 'scored'] : ['all', 'correct', 'incorrect']
  return (ungradedCount ?? 0) > 0 ? [...base, 'ungraded'] : base
}

/** TruthfulQA MC2 keeps its correct-choice vector in the preserved source document. */
export function goldChoiceIndices(row: SampleRow): number[] {
  if (!/^truthfulqa_mc2(?:$|_)/.test(row.task)) return row.target_choice === null ? [] : [row.target_choice]
  try {
    const doc = JSON.parse(row.doc) as { mc2_targets?: { labels?: unknown } }
    const labels = doc.mc2_targets?.labels
    if (!Array.isArray(labels) || labels.length !== row.choices?.length || !labels.every((label) => label === 0 || label === 1)) return []
    return labels.flatMap((label, index) => label === 1 ? [index] : [])
  } catch {
    return []
  }
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
 * `correct === null`) is `scored` when the primary score exists, else `ungraded`;
 * otherwise pass/fail follows `correct`. `errored` is only claimed
 * when the grader actually reported one — it is never inferred from an empty answer.
 */
export function sampleOutcome(row: SampleRow): SampleOutcome {
  if (graderError(row)) return 'errored'
  if (row.correct === null || row.correct === undefined) return row.grading?.score != null ? 'scored' : 'ungraded'
  return row.correct ? 'passed' : 'failed'
}

const OUTCOME_CHIP_CLASS: Record<SampleOutcome, string> = {
  passed: 'bg-status-success-bg text-status-success border-status-success-border',
  failed: 'bg-status-danger-bg text-status-danger border-status-danger-border',
  errored: 'bg-status-warning-bg text-status-warning border-status-warning-border',
  scored: 'bg-accent-subtle text-accent border-accent-border',
  ungraded: 'bg-surface-sunken text-text-muted border-surface-border',
}

/** Tailwind classes for an outcome chip (bg/text/border), shared by the list and the viewer. */
export function outcomeChipClass(outcome: SampleOutcome): string {
  return OUTCOME_CHIP_CLASS[outcome]
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
    if (sampleOutcome(row) === 'scored') {
      const gold = goldChoiceIndices(row).map((index) => row.choices?.[index]?.label ?? '?')
      const score = row.grading?.score
      return `picked ${pick}${gold.length ? `, correct ${gold.join(', ')}` : ''}; score ${score}`
    }
    const gold = row.target_choice !== null ? (row.choices?.[row.target_choice]?.label ?? '?') : '—'
    if (sampleOutcome(row) === 'ungraded') return `picked ${pick} (ungraded)`
    return row.correct ? `picked ${pick} ✓` : `picked ${pick}, gold ${gold}`
  }

  if (row.kind === 'agentic') {
    const reward = row.grading?.score
    return reward !== null && reward !== undefined ? `reward ${reward}` : 'agentic trial'
  }

  if (sampleOutcome(row) === 'scored') return `score ${row.grading?.score}`

  // generation
  const extracted = row.extracted ?? row.output ?? ''
  const target = row.target_text ?? ''
  if (sampleOutcome(row) === 'ungraded') return `“${truncate(extracted)}” (ungraded)`
  return row.correct ? `“${truncate(extracted)}” ✓` : `“${truncate(extracted)}” vs “${truncate(target)}”`
}
