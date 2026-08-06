import assert from 'node:assert/strict'
import test from 'node:test'

import type { SegmentInfo } from '../types/introspection.ts'
import type { ProtoSchema } from '../types/stats.ts'
import { configuredIndexMethods, segmentIndexSummary } from './segmentIndexes.ts'

const schema: ProtoSchema = {
  columns: [
    { name: 'service', index: { valueCounts: true } },
    { name: 'name', index: { trigram: true, exactValues: ['phase', 'step'] } },
  ],
  projections: [{ name: 'training-status', predicateColumn: 'name' }],
}

function segment(level: number, sections: string[], bundleBytes = 10, externalBytes = 0): SegmentInfo {
  return {
    path: `seg_L${level}.parquet`,
    level,
    minSeq: 1,
    maxSeq: 1,
    rowCount: 1,
    byteSize: 100,
    createdAtMs: 1,
    location: 'LOCAL',
    physical: {
      segmentIdentity: `segment-${level}`,
      layoutCurrent: true,
      rowGroups: 1,
      footerBytes: 8,
      uncompressedBytes: 100,
      indexBundle: {
        bytes: bundleBytes,
        externalBytes,
        checksum: 'crc32c',
        sections: sections.map((id) => ({
          id,
          kind: id,
          exactness: 'exact',
          methodVersion: 1,
          checksum: 'crc32c',
          payloadBytes: 1,
          externalBytes: 0,
          columns:
            id === 'value-counts'
              ? ['service']
              : id === 'exact-postings' || id === 'trigram:name'
                ? ['name']
                : ['seq', 'name'],
          available: true,
        })),
      },
    },
  }
}

test('configured methods mirror the planner-facing index family', () => {
  assert.deepEqual(configuredIndexMethods(schema), [
    'exact-postings',
    'projection:training-status',
    'trigram:name',
    'value-counts',
  ])
})

test('L0 segments are excluded from every derived-index coverage denominator', () => {
  const summary = segmentIndexSummary(
    [
      segment(0, []),
      segment(1, ['exact-postings', 'projection:training-status', 'trigram:name', 'value-counts'], 20, 30),
      segment(2, ['exact-postings', 'trigram:name', 'value-counts']),
    ],
    schema,
  )
  assert.ok(summary)
  assert.deepEqual(
    {
      stable: [summary.stableIndexed, summary.stableEligible],
      l0Unindexed: summary.l0Unindexed,
      projection: summary.methods.find((method) => method.id === 'projection:training-status'),
      counts: summary.methods.find((method) => method.id === 'value-counts'),
      countColumns: summary.countColumns,
      bytes: summary.bytes,
    },
    {
      stable: [1, 2],
      l0Unindexed: 1,
      projection: { id: 'projection:training-status', indexed: 1, eligible: 2 },
      counts: { id: 'value-counts', indexed: 2, eligible: 2 },
      countColumns: [{ id: 'service', indexed: 2, eligible: 2 }],
      bytes: 70,
    },
  )
})

test('a missing external projection is not reported as covered', () => {
  const broken = segment(1, [
    'exact-postings',
    'projection:training-status',
    'trigram:name',
    'value-counts',
  ])
  const projection = broken.physical?.indexBundle?.sections.find(
    (section) => section.id === 'projection:training-status',
  )
  assert.ok(projection)
  projection.available = false

  const summary = segmentIndexSummary([broken], schema)
  assert.ok(summary)
  assert.equal(summary.stableIndexed, 0)
  assert.deepEqual(
    summary.methods.find((method) => method.id === 'projection:training-status'),
    { id: 'projection:training-status', indexed: 0, eligible: 1 },
  )
})
