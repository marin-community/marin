import assert from 'node:assert/strict'
import test from 'node:test'

import { decimate, logTicks, niceTicks, smooth } from './chart.ts'

test('decimation keeps the extremes it passes over', () => {
  const points = Array.from({ length: 1000 }, (_, i) => ({ xv: i, yv: i === 617 ? 999 : 1 }))
  const thinned = decimate(points, 50)
  assert.ok(thinned.length <= 50)
  // The spike is the reason someone opened the chart; stride sampling would
  // drop it.
  assert.ok(thinned.some((p) => p.yv === 999))
  // Still in x order, so the line does not double back.
  assert.deepEqual([...thinned].sort((a, b) => a.xv - b.xv), thinned)
})

test('decimation leaves a series that already fits alone', () => {
  const points = [{ xv: 0, yv: 1 }, { xv: 1, yv: 2 }]
  assert.equal(decimate(points, 50), points)
})

test('smoothing starts at the first sample rather than ramping from zero', () => {
  const points = Array.from({ length: 20 }, (_, i) => ({ xv: i, yv: 100 }))
  const smoothed = smooth(points, 0.9)
  // A constant series smooths to itself once debiased; an undebiased EMA would
  // start near 10 and climb.
  for (const p of smoothed) assert.ok(Math.abs(p.yv - 100) < 1e-9)
})

test('smoothing at weight zero is the identity', () => {
  const points = [{ xv: 0, yv: 3 }, { xv: 1, yv: 9 }]
  assert.equal(smooth(points, 0), points)
})

test('smoothing damps a spike toward its neighbours', () => {
  const points = [1, 1, 1, 9, 1, 1, 1].map((yv, xv) => ({ xv, yv }))
  const spike = smooth(points, 0.8)[3]
  assert.ok(spike.yv > 1 && spike.yv < 9)
})

test('linear ticks land on round steps', () => {
  // 1-2-5 steps only, so a request for four gridlines over 0..100 resolves to
  // a step of 50 rather than an unroundable 25.
  assert.deepEqual(niceTicks(0, 100, 4), [0, 50, 100])
  assert.deepEqual(niceTicks(0, 40, 4), [0, 10, 20, 30, 40])
})

test('log ticks follow powers of ten across a wide range', () => {
  assert.deepEqual(logTicks(1, 10_000), [1, 10, 100, 1_000, 10_000])
})

test('log ticks inside one decade borrow the linear steps', () => {
  const ticks = logTicks(4_346, 5_556)
  // No power of ten lands in range, so the raw bounds would be the only
  // labels — and they are unreadable.
  assert.ok(ticks.length >= 2)
  assert.ok(ticks.every((t) => Number.isInteger(t / 100)))
})

test('log ticks refuse a range that is not positive', () => {
  assert.deepEqual(logTicks(0, 100), niceTicks(0, 100, 4))
})
