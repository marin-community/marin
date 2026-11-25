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

test('the Value display keeps sign, integers, and ordinary magnitudes readable', () => {
  assert.equal(formatValue(3.14159), '3.1')
  assert.equal(formatValue(42), '42')
  assert.equal(formatValue(-0.58193), '-0.58')
  assert.equal(formatValue(1234567), '1200000')
  assert.equal(formatValue(undefined), '—')
  assert.equal(formatValue(Number.NaN), '—')
})
