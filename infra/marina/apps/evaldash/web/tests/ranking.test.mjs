import assert from 'node:assert/strict'
import test from 'node:test'

import { meanReciprocalRanks } from '../src/utils/ranking.ts'

test('models rank by completed selected benchmarks with shared ranks for equal results', () => {
  const rows = [
    { model: 'a', cells: { math: { value: 0.9, low: 0.8 }, reading: { value: 0.6, low: 0.5 } } },
    { model: 'b', cells: { math: { value: 0.8, low: 0.7 }, reading: { value: 0.8, low: 0.7 } } },
    { model: 'c', cells: { math: { value: 0.8, low: 0.7 }, ignored: { value: 1, low: 1 } } },
    { model: 'd', cells: { reading: { value: 0.95, low: 0.9 } } },
    { model: 'e', cells: { ignored: { value: 1, low: 1 } } },
  ]

  const ranks = meanReciprocalRanks(rows, ['math', 'reading'])

  assert.equal(ranks.get('a'), 2 / 3)
  assert.equal(ranks.get('b'), 1 / 2)
  assert.equal(ranks.get('c'), 1 / 2)
  assert.equal(ranks.get('d'), 1)
  assert.equal(ranks.has('e'), false)
})

test('lower interval bound breaks equal score ties', () => {
  const rows = [
    { model: 'a', cells: { math: { value: 0.8, low: 0.7 } } },
    { model: 'b', cells: { math: { value: 0.8, low: 0.6 } } },
  ]

  const ranks = meanReciprocalRanks(rows, ['math'])

  assert.equal(ranks.get('a'), 1)
  assert.equal(ranks.get('b'), 1 / 2)
})
