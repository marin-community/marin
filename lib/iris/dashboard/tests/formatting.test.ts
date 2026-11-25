import assert from 'node:assert/strict'
import test from 'node:test'

import { formatAttemptDuration, formatEgressRule } from '../src/utils/formatting.ts'

const startedAt = { epochMs: '1789024283002' }
const finishedAt = { epochMs: '1789027789369' }
const later = 1789060189369

test('terminal attempts with no finish time show an unknown duration', (t) => {
  t.mock.method(Date, 'now', () => later)
  for (const state of ['SUCCEEDED', 'FAILED', 'KILLED', 'WORKER_FAILED', 'UNSCHEDULABLE', 'PREEMPTED', 'COSCHED_FAILED']) {
    assert.equal(formatAttemptDuration({ attemptId: 1, state: `TASK_STATE_${state}`, startedAt }), '-')
  }
})

test('a completed attempt keeps its duration after a retry starts', (t) => {
  t.mock.method(Date, 'now', () => later)
  assert.equal(formatAttemptDuration({ attemptId: 1, state: 'TASK_STATE_COSCHED_FAILED', startedAt, finishedAt }), '58m 26s')
})

test('active attempts use the current time after execution starts', (t) => {
  t.mock.method(Date, 'now', () => Number(startedAt.epochMs) + 3_600_000)
  for (const state of ['ASSIGNED', 'BUILDING', 'RUNNING']) {
    assert.equal(formatAttemptDuration({ attemptId: 2, state: `TASK_STATE_${state}`, startedAt }), '1h 0m')
    assert.equal(formatAttemptDuration({ attemptId: 2, state: `TASK_STATE_${state}` }), '-')
  }
})

test('egress rules render in the syntax --egress-allow accepts', () => {
  assert.equal(formatEgressRule({ domain: '*.example.com' }), '*.example.com')
  assert.equal(formatEgressRule({ domain: 'api.openai.com', ports: [443, 8443] }), 'api.openai.com:443,8443')
  assert.equal(formatEgressRule({ cidr: '2001:db8::/32', ports: [443] }), '[2001:db8::/32]:443')
})
