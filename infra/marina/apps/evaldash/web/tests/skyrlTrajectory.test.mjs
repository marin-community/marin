import test from 'node:test'
import assert from 'node:assert/strict'

import { skyrlTurnsToTrajectory } from '../src/utils/skyrlTrajectory.ts'

// A length-stopped row from SkyRL's eval dump, with shortened token-id lists.
const LENGTH_STOPPED = {
  uid: '84',
  attempt_id: 'a73813cd0729',
  step: -1,
  ordinal: 38,
  is_last_step: true,
  task_id: 'NUPA/doc:a2c5cdf7608cba4092bec12fb5169676ded7c5636f504cddb887ee9c677e10bb',
  prompt_token_ids: [128000, 128006, 9125],
  response_token_ids: [128002, 16, 13],
  input_prompt: '<|begin_of_text|><|start_header_id|>system<|end_header_id|>Reasoning: /think\nCompare 7808931378968544 and 7808931378968545 digit by digit.',
  output_response: '<|start_think|>The user wants me to compare two numbers digit by digit and return the larger digit at each position.\nThe numbers are:\nNumber 1: 7808931378968544',
  messages: [
    { role: 'user', content: 'Directly return the answer as a float without any comma separator, like 10.4 . Compare two numbers digit by digit.' },
    { role: 'user', content: 'Give your answer as plain text.' },
    { role: 'assistant', content: '' },
  ],
  score: [0.0, 0.0],
  stop_reason: 'length',
  exception_type: null,
  env_class: 'taskcompendium',
  data_source: 'HaotongYang/NUPA_text',
  env_metrics: { 'timing/grading': 0.41, 'timing/setup': 0.02 },
  verification: {
    status: 'verified',
    reason: null,
    score: 0.0,
    passed: null,
    score_min: 0.0,
    score_max: 1.0,
    diagnostics: {},
  },
}

const COMPLETED = {
  ...LENGTH_STOPPED,
  uid: '172',
  output_response: '<|start_think|>compare digit by digit<|end_think|>\n\n7808931378968545',
  messages: [
    { role: 'user', content: 'Compare two numbers digit by digit.' },
    { role: 'assistant', content: '7808931378968545' },
  ],
  stop_reason: 'stop',
  score: [1.0],
  verification: { ...LENGTH_STOPPED.verification, score: 1.0, passed: true },
}

test('a length-stopped record shows its stop_reason, verification and score', () => {
  const trajectory = skyrlTurnsToTrajectory(LENGTH_STOPPED)
  const last = trajectory.steps.at(-1)
  assert.equal(last.annotations['stop reason'], 'length')
  assert.equal(last.annotations['verification'], 'verified')
  assert.equal(last.annotations['verifier score'], '0')
})

test('messages map to steps by role with the prompt and raw response kept', () => {
  const trajectory = skyrlTurnsToTrajectory(COMPLETED)
  assert.equal(trajectory.steps.length, 4)
  assert.equal(trajectory.steps[0].source, 'user')
  assert.equal(trajectory.steps[1].source, 'user')
  assert.equal(trajectory.steps[2].source, 'agent')
  assert.equal(trajectory.steps[2].message, '7808931378968545')
  assert.ok(trajectory.steps[3].message.includes('<|start_think|>'))
  assert.ok(trajectory.steps[3].message.includes('<|end_think|>'))
  assert.equal(trajectory.steps[3].annotations['verifier score'], '1')
  assert.equal(trajectory.agent.name, 'taskcompendium')
  assert.equal(trajectory.session_id, 'a73813cd0729')
  assert.deepEqual(trajectory.final_metrics, { 'timing/grading': 0.41, 'timing/setup': 0.02 })
})

test('a list of cumulative per-step records does not duplicate earlier turns', () => {
  const first = {
    ...COMPLETED,
    ordinal: 0,
    is_last_step: false,
    messages: [
      { role: 'user', content: 'Compare two numbers digit by digit.' },
      { role: 'assistant', content: 'Reading the task.<|end_think|>' },
    ],
  }
  const second = {
    ...first,
    ordinal: 1,
    is_last_step: true,
    messages: [...first.messages, { role: 'assistant', content: '7808931378968545' }],
  }
  const trajectory = skyrlTurnsToTrajectory([first, second])
  const sources = trajectory.steps.map((step) => step.message)
  assert.equal(sources.filter((text) => text === 'Reading the task.<|end_think|>').length, 1)
  assert.equal(sources.filter((text) => text === '7808931378968545').length, 1)
  const rawSteps = trajectory.steps.filter((step) => step.annotations !== undefined && step.annotations['stop reason'])
  assert.equal(rawSteps.length, 2)
})
