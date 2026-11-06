import assert from 'node:assert/strict'
import test from 'node:test'

import { metricView } from '../src/utils/runMetrics.ts'

const NUPA_RAW = {
  exact_match: 0,
  digit_match: 0.0030796088040486408,
  format_valid_rate: 0.0046,
  'bucket:S/exact_match': 0,
  'task:add_Integer_Integer/bucket:S/exact_match': 0,
}

function nupaRecord() {
  return {
    eval: {
      name: 'nupa',
      mechanism: 'evalchemy',
      tasks: [
        {
          name: 'NUPA5K-Loose',
          num_fewshot: 0,
          benchmark: {
            schema_version: 1,
            task: 'NUPA5K-Loose',
            primary_metric: 'accuracy',
            metric_kind: 'binary',
            metrics: [
              { name: 'accuracy', source_name: 'exact_match', kind: 'binary', higher_is_better: true },
              { name: 'digit_match', source_name: 'digit_match', kind: 'continuous', higher_is_better: true },
              {
                name: 'format_valid_rate',
                source_name: 'format_valid_rate',
                kind: 'continuous',
                higher_is_better: true,
              },
            ],
            n_benchmark: 5000,
            n_attempted: 5000,
          },
        },
      ],
    },
    metrics: { nupa5k: NUPA_RAW },
    canonical_metrics: {
      nupa5k: { accuracy: 0, digit_match: 0.0030796088040486408, format_valid_rate: 0.0046 },
    },
    metric_rows: [
      { task: 'NUPA5K-Loose', metric: 'accuracy', value: 0, declared: true },
      { task: 'NUPA5K-Loose', metric: 'digit_match', value: 0.0030796088040486408, declared: true },
      { task: 'NUPA5K-Loose', metric: 'format_valid_rate', value: 0.0046, declared: true },
    ],
  }
}

test('declared protocol metrics lead the table while the raw bucket rows stay folded', () => {
  const view = metricView(nupaRecord())

  assert.deepEqual(view.rows, [
    { task: 'NUPA5K-Loose', metric: 'accuracy', value: 0 },
    { task: 'NUPA5K-Loose', metric: 'digit_match', value: 0.0030796088040486408 },
    { task: 'NUPA5K-Loose', metric: 'format_valid_rate', value: 0.0046 },
  ])
  assert.equal(view.hasDeclared, true)
  assert.ok(view.raw.some((row) => row.metric === 'task:add_Integer_Integer/bucket:S/exact_match'))
})

test('rows the API reports as undeclared are the whole top table and need no fold', () => {
  const record = nupaRecord()
  record.metric_rows = [
    { task: 'legacy-task', metric: 'acc', value: 0.5, declared: false },
    { task: 'legacy-task', metric: 'acc_norm', value: 0.4, declared: false },
  ]

  const view = metricView(record)

  assert.deepEqual(view.rows, [
    { task: 'legacy-task', metric: 'acc', value: 0.5 },
    { task: 'legacy-task', metric: 'acc_norm', value: 0.4 },
  ])
  assert.equal(view.hasDeclared, false)
})
