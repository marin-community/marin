import test from 'node:test'
import assert from 'node:assert/strict'

import { skyrlTurnsToTrajectory } from '../src/utils/skyrlTrajectory.ts'
import { EVAL_DUMP_ROW, EVAL_DUMP_ROW_NULL_MESSAGES } from './fixtures/skyrlEvalDump.mjs'
import { SCHEMA_V7_STOP, SCHEMA_V7_LENGTH } from './fixtures/skyrlRetention.mjs'

test('an eval-dump row renders tool-carrying assistant turns, the tool reply, and the raw response', () => {
  const trajectory = skyrlTurnsToTrajectory(EVAL_DUMP_ROW)
  const sources = trajectory.steps.map((step) => step.source)
  assert.deepEqual(sources, ['user', 'system', 'user', 'agent', 'tool', 'agent', 'agent'])
  const callStep = trajectory.steps[3]
  assert.equal(callStep.tool_calls.length, 1)
  assert.equal(callStep.tool_calls[0].tool_call_id, 'chatcmpl-tool-b14c4605f90e32a0')
  assert.equal(callStep.tool_calls[0].function_name, 'bash')
  assert.ok(String(callStep.tool_calls[0].arguments.command).includes('result.json'))
  assert.equal(trajectory.steps[4].message, '(no output)')
  const raw = trajectory.steps.at(-1)
  assert.ok(raw.message.includes('<tool_call>'))
  assert.equal(raw.annotations['verification'], 'verified')
  assert.equal(raw.annotations['verifier score'], '1')
  assert.equal(trajectory.schema_version, 'skyrl-eval-dump')
  assert.equal(trajectory.session_id, 'a73813cd0729')
})

test('a null-messages eval-dump row still renders its prompt and outcome', () => {
  const trajectory = skyrlTurnsToTrajectory(EVAL_DUMP_ROW_NULL_MESSAGES)
  assert.equal(trajectory.steps.length, 2)
  assert.equal(trajectory.steps[0].annotations['content'], 'raw input prompt')
  assert.equal(trajectory.steps[1].annotations['verification'], 'verified')
})

test('a schema_v7 turn record renders its conversation, raw response, and outcome', () => {
  const trajectory = skyrlTurnsToTrajectory(SCHEMA_V7_STOP)
  const sources = trajectory.steps.map((step) => step.source)
  assert.deepEqual(sources, ['user', 'user', 'agent', 'agent'])
  assert.equal(trajectory.steps[2].message, '15')
  const raw = trajectory.steps.at(-1)
  assert.ok(raw.message.includes('<|start_think|>'))
  assert.equal(raw.annotations['stop reason'], 'stop')
  assert.equal(raw.annotations['verifier score'], '1')
  assert.equal(trajectory.schema_version, 'skyrl-turn-v7')
  assert.equal(trajectory.agent.name, 'taskcompendium')
  assert.equal(trajectory.session_id, 'micro210-reval-exp1-rl16@2026.11.24.28')
})

test('a length-stopped schema_v7 record shows its stop_reason', () => {
  const trajectory = skyrlTurnsToTrajectory(SCHEMA_V7_LENGTH)
  const raw = trajectory.steps.at(-1)
  assert.equal(raw.annotations['stop reason'], 'length')
  assert.equal(raw.annotations['verifier score'], '0')
  assert.ok(raw.message.includes('<|start_think|>'))
})

test('a list spanning uids requires an unambiguous viewed sample', () => {
  const stopUid = SCHEMA_V7_STOP.trajectory.instance_id
  const lengthUid = SCHEMA_V7_LENGTH.trajectory.instance_id
  assert.notEqual(stopUid, lengthUid)
  assert.equal(skyrlTurnsToTrajectory([SCHEMA_V7_STOP, SCHEMA_V7_LENGTH]), null)
  assert.equal(skyrlTurnsToTrajectory([SCHEMA_V7_STOP, SCHEMA_V7_LENGTH], 'missing'), null)
  const viewed = skyrlTurnsToTrajectory([SCHEMA_V7_STOP, SCHEMA_V7_LENGTH], lengthUid)
  assert.equal(viewed.steps.at(-1).annotations['stop reason'], 'length')
  assert.equal(viewed.agent.name, 'taskcompendium')
})

test('distinct repetitions of one instance never merge into the viewed trajectory', () => {
  const repeated = {
    ...SCHEMA_V7_STOP,
    trajectory: { ...SCHEMA_V7_STOP.trajectory, repetition_id: 1 },
    response: { ...SCHEMA_V7_STOP.response, text: 'a different trial' },
  }
  const uid = SCHEMA_V7_STOP.trajectory.instance_id
  assert.equal(skyrlTurnsToTrajectory([SCHEMA_V7_STOP, repeated], uid), null)
  assert.equal(skyrlTurnsToTrajectory([SCHEMA_V7_STOP, repeated]), null)
})

test('flat dump lists reject same-uid rows without a producer-defined trial key', () => {
  const another = { ...EVAL_DUMP_ROW, ordinal: EVAL_DUMP_ROW.ordinal + 1, output_response: 'another repetition' }
  assert.equal(skyrlTurnsToTrajectory([EVAL_DUMP_ROW, another], EVAL_DUMP_ROW.uid), null)
  const viewed = skyrlTurnsToTrajectory([EVAL_DUMP_ROW, EVAL_DUMP_ROW_NULL_MESSAGES], EVAL_DUMP_ROW_NULL_MESSAGES.uid)
  assert.equal(viewed.steps.at(-1).annotations['verification'], 'verified')
  assert.equal(viewed.steps.length, 2)
})

test('a no-evidence retention record keeps its saved prompt and shows the raw answer once', () => {
  const response = '<think>work through the question</think>15'
  const record = {
    ...SCHEMA_V7_STOP,
    response: { ...SCHEMA_V7_STOP.response, text: response, messages: [{ role: 'assistant', content: response }] },
  }
  const trajectory = skyrlTurnsToTrajectory(record)
  assert.deepEqual(trajectory.steps.map((step) => step.message), [
    ...record.prompt.messages.map((message) => message.content),
    response,
  ])
  assert.equal(trajectory.steps.at(-1).annotations['stop reason'], 'stop')
  assert.equal(trajectory.steps.at(-1).annotations['verification'], 'verified')
})

test('saved prompt messages already present with response metadata are not duplicated', () => {
  const record = {
    ...SCHEMA_V7_STOP,
    response: {
      ...SCHEMA_V7_STOP.response,
      messages: SCHEMA_V7_STOP.response.messages.map((message) => ({
        role: message.role, content: message.content, annotations: null,
      })),
    },
  }
  const trajectory = skyrlTurnsToTrajectory(record)
  assert.deepEqual(
    trajectory.steps.filter((step) => step.source === 'user').map((step) => step.message),
    record.prompt.messages.map((message) => message.content),
  )
  assert.equal(trajectory.steps.at(-1).message, record.response.text)
})

test('cumulative records from one complete retention identity do not duplicate earlier messages', () => {
  const prompt = { role: 'user', content: 'Compare two numbers digit by digit.' }
  const first = {
    ...SCHEMA_V7_STOP,
    prompt: { messages: [prompt] },
    response: {
      ...SCHEMA_V7_STOP.response,
      text: 'Reading the task.',
      messages: [prompt, { role: 'assistant', content: 'Reading the task.' }],
    },
  }
  const second = {
    ...first,
    response: {
      ...first.response,
      text: '7808931378968545',
      messages: [...first.response.messages, { role: 'assistant', content: '7808931378968545' }],
      step_boundaries: [{ row_index: 1 }],
    },
  }
  const trajectory = skyrlTurnsToTrajectory([first, second])
  assert.deepEqual(trajectory.steps.map((step) => step.message), [
    prompt.content, 'Reading the task.', '7808931378968545',
  ])
  assert.equal(trajectory.steps.filter((step) => step.annotations?.['stop reason']).length, 2)
})
