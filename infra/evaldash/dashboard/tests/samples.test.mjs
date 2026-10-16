import assert from 'node:assert/strict'
import test from 'node:test'

import { sampleHint } from '../src/utils/samples.ts'

function failedGeneration(task, target, extracted) {
  return {
    task,
    kind: 'generation',
    target_text: target,
    extracted,
    correct: false,
    grading: null,
  }
}

test('GSM8K hint compares the extracted answer with the graded reference', () => {
  const row = failedGeneration('gsm8k_0shot', 'Work through the problem.\n#### 2', '30')

  assert.equal(sampleHint(row), '“30” vs “2”')
})

test('GSM8K hint uses the scorer’s normalized numeric reference', () => {
  const row = failedGeneration('gsm8k-0shot', 'The price is $1,200.\n#### $1,200', '$139')

  assert.equal(sampleHint(row), '“$139” vs “1200”')
})

test('other generation hints retain their stored target', () => {
  const row = failedGeneration('other-task', 'The reference solution', '30')

  assert.equal(sampleHint(row), '“30” vs “The reference solution”')
})
