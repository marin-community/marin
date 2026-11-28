import assert from 'node:assert/strict'
import test from 'node:test'

import { formatDelta, formatInterval, formatScore } from '../src/utils/formatting.ts'

test('scores and intervals use metric kind even when measurements have the same values', () => {
  const aggregate = { value: 0.85, low: 0.75, high: 0.95 }
  const continuous = { ...aggregate, metric_kind: 'continuous' }
  const binary = { ...aggregate, metric_kind: 'binary' }

  assert.equal(formatScore(continuous), '0.850')
  assert.equal(formatInterval(continuous), '0.750–0.950')
  assert.equal(formatDelta(-0.025, continuous), '-0.025')

  assert.equal(formatScore(binary), '85.0%')
  assert.equal(formatInterval(binary), '75.0–95.0%')
  assert.equal(formatDelta(-0.025, binary), '-2.5')

  assert.equal(formatScore(aggregate), '0.850')
  assert.equal(formatInterval(aggregate), '0.750–0.950')
})
