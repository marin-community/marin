import test from 'node:test'
import assert from 'node:assert/strict'

import { formatValue } from '../src/utils/formatting.ts'

test('the metric Value display rounds to 2 significant digits in plain decimal notation', () => {
  assert.equal(formatValue(0.58193), '0.58')
  assert.equal(formatValue(0.0012345), '0.0012')
  assert.equal(formatValue(12345), '12000')
  assert.equal(formatValue(0), '0')
  assert.equal(formatValue(null), '—')
})

test('rounding across a magnitude boundary never prints a third significant digit', () => {
  assert.equal(formatValue(0.996), '1.0')
  assert.equal(formatValue(0.0996), '0.1')
  assert.equal(formatValue(9.96), '10')
  assert.equal(formatValue(99.6), '100')
  assert.equal(formatValue(-0.999), '-1.0')
})

test('tiny and large finite values render as plain decimals without throwing', () => {
  const cases = [
    [1e-100, `0.${'0'.repeat(99)}1`],
    [1.2345e-101, `0.${'0'.repeat(100)}12`],
    [1e21, `1${'0'.repeat(21)}`],
    [Number.MIN_VALUE, `0.${'0'.repeat(323)}5`],
    [Number.MAX_VALUE, `18${'0'.repeat(307)}`],
  ]
  for (const [value, expected] of cases) {
    assert.equal(formatValue(value), expected)
    assert.equal(formatValue(-value), `-${expected}`)
  }
})

test('the Value display keeps sign, integers, and ordinary magnitudes readable', () => {
  assert.equal(formatValue(3.14159), '3.1')
  assert.equal(formatValue(42), '42')
  assert.equal(formatValue(-0.58193), '-0.58')
  assert.equal(formatValue(1234567), '1200000')
  assert.equal(formatValue(undefined), '—')
  assert.equal(formatValue(Number.NaN), '—')
  assert.equal(formatValue(Infinity), 'Infinity')
  assert.equal(formatValue(-Infinity), '-Infinity')
})
