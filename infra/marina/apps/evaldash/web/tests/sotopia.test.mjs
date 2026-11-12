import assert from 'node:assert/strict'
import test from 'node:test'

import { isSotopiaSummary } from '../src/utils/sotopia.ts'

test('SOTOPIA summary is recognized without ATIF steps', () => {
  const summary = {
    transcript: [
      [{ sender: 'Environment', receiver: 'Alice', message: 'Your goal is to negotiate.' }],
      [{ sender: 'Alice', receiver: 'Environment', message: 'Hello.' }],
    ],
    reasoning: 'The agent partly met its goal.',
    scores: [{ goal: 4 }],
    overall_scores: [0.309],
    evaluated_agent_index: 0,
    agent_ids: ['Alice'],
    models: ['candidate'],
    environment_id: 'environment-1',
    combo_id: 'combo-1',
    context_budget: { max_input_tokens: 32768, max_prompt_tokens: 1200, rejected_requests: 0 },
  }

  assert.equal(isSotopiaSummary(summary), true)
  assert.equal(isSotopiaSummary({ steps: [] }), false)
  assert.equal(isSotopiaSummary({ ...summary, transcript: [[{ sender: 'Alice' }]] }), false)
})
