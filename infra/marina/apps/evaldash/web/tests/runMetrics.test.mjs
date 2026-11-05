import assert from 'node:assert/strict'
import test from 'node:test'

import { metricView } from '../src/utils/runMetrics.ts'

const NUPA_RAW = {
  exact_match: 0,
  digit_match: 0.0030796088040486408,
  dlength: 61.5,
  format_valid_rate: 0.0046,
  no_answer_rate: 0.9954,
}

function nupaRecord() {
  const raw = { ...NUPA_RAW }
  for (const bucket of ['S', 'M', 'L', 'XL']) {
    for (const metric of ['exact_match', 'digit_match', 'dlength', 'format_valid_rate', 'no_answer_rate']) {
      raw[`bucket:${bucket}/${metric}`] = 0
      raw[`task:add_Integer_Integer/bucket:${bucket}/${metric}`] = 0
    }
  }
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
    metrics: { nupa5k: raw },
    canonical_metrics: {
      nupa5k: { accuracy: 0, digit_match: 0.0030796088040486408, format_valid_rate: 0.0046 },
    },
  }
}

test('the declared protocol metrics replace the raw bucket and per-task flood', () => {
  const view = metricView(nupaRecord())

  assert.deepEqual(view.declared, [
    { task: 'NUPA5K-Loose', metric: 'accuracy', value: 0 },
    { task: 'NUPA5K-Loose', metric: 'digit_match', value: 0.0030796088040486408 },
    { task: 'NUPA5K-Loose', metric: 'format_valid_rate', value: 0.0046 },
  ])
  // The raw map keeps its whole shape for the folded section: five base metrics, plus a bucket and a
  // per-task row for each of four buckets.
  assert.equal(view.raw.length, 5 + 2 * 4 * 5)
  assert.ok(view.raw.some((row) => row.metric === 'task:add_Integer_Integer/bucket:S/exact_match'))
})

test('a declared value falls back to the raw source metric when no canonical copy exists', () => {
  const record = nupaRecord()
  record.canonical_metrics = {}

  const view = metricView(record)

  assert.deepEqual(view.declared, [
    { task: 'NUPA5K-Loose', metric: 'accuracy', value: 0 },
    { task: 'NUPA5K-Loose', metric: 'digit_match', value: 0.0030796088040486408 },
    { task: 'NUPA5K-Loose', metric: 'format_valid_rate', value: 0.0046 },
  ])
})

test('records without a declared benchmark keep the raw metric list', () => {
  const record = nupaRecord()
  record.eval.tasks = [{ name: 'legacy-task', num_fewshot: 5, benchmark: null }]

  const view = metricView(record)

  assert.deepEqual(view.declared, [])
  assert.equal(view.raw.length, Object.keys(record.metrics.nupa5k).length)
})

test('multi-task records resolve each task by its benchmark name', () => {
  const record = nupaRecord()
  const benchmark = (task, primary) => ({
    schema_version: 1,
    task,
    primary_metric: primary,
    metric_kind: 'binary',
    metrics: [{ name: primary, source_name: 'exact_match', kind: 'binary', higher_is_better: true }],
    n_benchmark: 10,
    n_attempted: 10,
  })
  record.eval.tasks = [
    { name: 'alpha', num_fewshot: 0, benchmark: benchmark('alpha', 'accuracy') },
    { name: 'beta', num_fewshot: 0, benchmark: benchmark('beta', 'accuracy') },
  ]
  record.metrics = { alpha: { exact_match: 0.5 }, beta: { exact_match: 0.25 } }
  record.canonical_metrics = {}

  const view = metricView(record)

  assert.deepEqual(view.declared, [
    { task: 'alpha', metric: 'accuracy', value: 0.5 },
    { task: 'beta', metric: 'accuracy', value: 0.25 },
  ])
})
