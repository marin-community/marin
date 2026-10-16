import assert from 'node:assert/strict'
import test from 'node:test'

import { gsm8kGradedReference, sampleHint } from '../src/utils/samples.ts'

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
