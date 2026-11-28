import assert from 'node:assert/strict'
import test from 'node:test'

import { gsm8kGradedReference, sampleHint, sampleFilters, sampleOutcome } from '../src/utils/samples.ts'

for (const [task, target, expected] of [
  ['gsm8k_0shot', 'Work through the problem.\n#### 2', '2'],
  ['gsm8k-0shot', 'The price is $1,200.\n#### $1,200', '1200'],
]) {
  test(`GSM8K hint compares with the graded reference for ${task}`, () => {
    const row = {
      task,
      kind: 'generation',
      target_text: target,
      extracted: '30',
      correct: false,
      grading: null,
    }

    assert.equal(gsm8kGradedReference(row), expected)
    assert.equal(sampleHint(row), sampleHint({ ...row, task: 'other-task', target_text: expected }))
    assert.notEqual(sampleHint(row), sampleHint({ ...row, task: 'other-task' }))
  })
}


test('continuous scores are shown as scored and have no binary filters', () => {
  const row = { correct: null, grading: { score: 1, passed: null } }
  assert.equal(sampleOutcome(row), 'scored')
  assert.deepEqual(sampleFilters({ all: 2, correct: null, incorrect: null, ungraded: 0 }), ['all'])
  assert.deepEqual(sampleFilters({ all: 3, correct: null, incorrect: null, ungraded: 1 }), ['all', 'ungraded'])
})

test('a generation row with no reference shows its answer without an empty comparison', () => {
  const row = {
    task: 'HaotongYang/NUPA_text',
    kind: 'generation',
    extracted: '8.812521090260602709060e24',
    target_text: null,
    correct: true,
    grading: null,
  }
  assert.equal(sampleHint(row), '“8.812521090260602709060e24” ✓')
  assert.equal(sampleHint({ ...row, correct: false }), '“8.812521090260602709060e24”')
})

test('a generation row with no answer shows the verifier note from grading detail', () => {
  const row = {
    task: 'HaotongYang/NUPA_text',
    kind: 'generation',
    extracted: null,
    output: null,
    target_text: null,
    correct: false,
    grading: { method: 'native-verifier', metric: 'reward', filter: null, score: 0, passed: null, detail: '{"status": "verified", "reason": null, "score": 0.0, "passed": null}' },
  }
  assert.equal(sampleHint(row), 'no answer (verified, score 0)')
  // a verifier reason takes precedence over the status
  assert.equal(
    sampleHint({ ...row, grading: { ...row.grading, detail: '{"status": "error", "reason": "Harbor trial has no verifier score"}' } }),
    'no answer (Harbor trial has no verifier score)',
  )
  // an ungraded row (no score at all) with no answer reads ungraded with the verifier note
  assert.equal(
    sampleHint({ ...row, correct: null, grading: { ...row.grading, score: null, detail: '{"status": "error", "reason": null, "score": null}' } }),
    'ungraded (error)',
  )
})
