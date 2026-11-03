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
