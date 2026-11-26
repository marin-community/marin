import { test } from 'node:test';
import assert from 'node:assert/strict';
import { flowGraph, cards } from '../src/contracts.js';
import { createMeasurements, cumulativeRate, waitShare } from '../src/measurements.js';
import {
  canceledGroupJobCount, cardsWithSamples, entryOccupancyAtLongestWait, recordedNodeSettings, timingDistribution,
} from '../src/display.js';
import { driverPhases, driverSetup } from '../src/phases.js';

const window = { from: 0, to: 50_000 };
const counter = (engine, timestamp_ms, value, seq = timestamp_ms + 1) => ({
  cluster: 'fixture-cluster', service: 'marinskyrl', execution_uid: 'iris:fixture',
  process_index: '0', resource: { pid: 10 }, attributes: { engine },
  timestamp_ms, value, seq,
});

test('per-engine rates preserve a zero, make gaps visible, and recover after a reset', () => {
  const rows = [
    ...[[0, 100], [10, 160], [20, 160], [30, 3], [40, 23], [50, 73]]
      .map(([t, value]) => counter('a', t * 1000, value)),
    ...[[0, 200], [10, 340], [20, 340], [30, 450], [50, 650]]
      .map(([t, value]) => counter('b', t * 1000, value)),
  ];
  assert.deepEqual(cumulativeRate(rows, window).map((point) => point.value), [20, 0, null, null, null]);
  assert.deepEqual(cumulativeRate(rows.filter((row) => row.attributes.engine === 'a'), window)
    .map((point) => point.value), [6, 0, null, 2, 5]);

  const shiftedScrapes = [counter('a', -2000, 40), counter('a', 8000, 140),
    counter('a', 18000, 340), counter('a', 18000, 300, 17)];
  assert.equal(cumulativeRate(shiftedScrapes, window)[0].value, 12);
});

test('native deltas sum, an empty bin is missing, and a reported zero remains zero', () => {
  const rows = [0, 2, 3].map((value, index) => ({
    name: 'rollout_groups', value, timestamp_ms: [10_000, 30_000, 30_100][index],
    seq: index, attributes_json: '{"disposition":"consumed"}',
  }));
  const data = createMeasurements(rows, window).series.admissions[0].data;
  assert.deepEqual(data.map((point) => point.value), [null, 0, null, 0.5, null]);
});

test('rates use configured GPUs for the fleet view and preserve real zeros and gaps', () => {
  const rows = [[0, 70], [10_000, 150], [20_000, 150], [40_000, 310]].map(([timestamp_ms, value]) => ({
    name: 'generation_tokens_total', timestamp_ms, value, seq: timestamp_ms + 1,
    attributes_json: '{"engine":"fixture","metric_source":"vllm","source_temporality":"cumulative_snapshot"}',
  }));
  const { series } = createMeasurements(rows, window, [], 8);
  assert.equal(series.generationPerGpu.length, 1);
  assert.deepEqual(series.generationPerGpu[0].data.map((point) => point.value), [1, 0, null, null, null]);
  assert.deepEqual(series.generation[0].data.map((point) => point.value), [8, 0, null, null, null]);
  assert.equal(createMeasurements(rows, window).series.generationPerGpu, undefined);
});

test('inference timing weights engines by completed requests and leaves gaps on resets', () => {
  const rows = (phase, engine, samples) => samples.flatMap(([timestamp, seconds, requests]) =>
    [['sum', seconds], ['count', requests]].map(([suffix, value]) => ({
      ...counter(engine, timestamp, value),
      name: 'request_' + phase + '_time_seconds_' + suffix,
      attributes_json: { engine, metric_source: 'vllm', source_kind: 'histogram', source_temporality: 'cumulative_snapshot' },
    })));
  const samples = [
    ...rows('prefill', 'a', [[0, 0, 0], [10_000, 3, 2], [20_000, 3, 2], [30_000, 4, 1], [40_000, 6, 3]]),
    ...rows('prefill', 'b', [[0, 0, 0], [10_000, 15, 6], [20_000, 15, 6], [30_000, 25, 10], [40_000, 29, 12]]),
    ...rows('decode', 'a', [[0, 0, 0], [10_000, 14, 2]]),
  ];
  const { series } = createMeasurements(samples, window);
  const round = ({ value }) => value === null ? null : Number(value.toFixed(6));
  assert.deepEqual(series.prefillTime[0].data.map(round), [2.25, null, null, 1.5, null]);
  assert.equal(round(series.prefillTime[1].data[0]), 1.5);
  assert.equal(round(series.decodeTime[0].data[0]), 7);

  const incomplete = samples.filter((row) => row.attributes_json.engine !== 'b' || !row.name.endsWith('_sum'));
  const { series: partial } = createMeasurements(incomplete, window);
  assert.equal(partial.prefillTime[0].data.length, 0);
  assert.equal(round(partial.prefillTime[1].data[0]), 1.5);
});

test('six recorded seconds in a ten-second bin is 60 percent', () => {
  const waits = [{ start: 4000, end: 9500 }, { start: 7000, end: 10000 }];
  const phases = [...waits, { start: 10000, end: 20000 }];
  assert.deepEqual(waitShare(waits, phases, window).map((point) => point.value),
    [60, 0, null, null, null]);
});

for (const [leaseLabels, taskLabels] of [
  [['slot', 'slot'], ['prompt', 'prompt']],
  [['lease', 'lease'], ['task', 'task']],
  [['slot', 'lease'], ['prompt', 'task']],
]) {
  test(`dispatch places each ${leaseLabels.join('/')} and ${taskLabels.join('/')} wait on one timeline`, () => {
    const records = [
      { timestamp_ms: 10_000, value: 10, wait: leaseLabels[0] },
      { timestamp_ms: 23_000, value: 13, wait: taskLabels[0] },
      { timestamp_ms: 27_000, value: 4, wait: leaseLabels[1] },
      { timestamp_ms: 30_000, value: 3, wait: taskLabels[1] },
    ].flatMap(({ wait, ...row }, index) => ['sum', 'max'].map((stat, offset) => ({
      ...row, seq: 2 * index + offset, name: 'rollout_wait_seconds',
      attributes_json: JSON.stringify({ wait, stat }),
    })));
    const { series } = createMeasurements(records, { from: 0, to: 40_000 });
    assert.deepEqual(series.leaseWait[0].data.map(({ value, waitStart }) => [value, waitStart]), [[10, 0], [4, 23_000]]);
    assert.deepEqual(series.taskWait[0].data.map(({ value, waitStart }) => [value, waitStart]), [[13, 10_000], [3, 27_000]]);
    assert.deepEqual(series.leaseWaitTime[0].data.map((point) => point.value), [10, 0, 4, null]);
    assert.deepEqual(series.taskWaitTime[0].data.map((point) => point.value), [0, 10, 6, null]);
  });
}

test('a task selection wait ending without a recorded launch adds no group starts', () => {
  const rows = [
    { name: 'rollout_call', timestamp_ms: 18_000, seq: 1, body_json: '{"started_unix_ms":5000}' },
    { name: 'rollout_waits', timestamp_ms: 42_000, seq: 2, value: 1, attributes_json: '{"wait":"prompt"}' },
    { name: 'rollout_wait_seconds', timestamp_ms: 42_000, seq: 3, value: 22,
      attributes_json: '{"wait":"prompt","stat":"sum"}' },
  ];
  const { series } = createMeasurements(rows, window);
  assert.deepEqual(series.entryStarts[0].data.map((point) => point.value), [0.1, null, null, null, null]);
  assert.equal(series.taskWait[0].data[0].value, 22);
});

test('Dispatch summarizes occupancy at the longest task selection wait with run coverage', () => {
  const job = (start, end) => ({
    name: 'rollout_call', timestamp_ms: end, seq: end,
    body_json: { started_unix_ms: start, finished_unix_ms: end },
  });
  const wait = (timestamp_ms, value) => ({
    name: 'rollout_wait_seconds', timestamp_ms, seq: timestamp_ms, value,
    attributes_json: { wait: 'prompt', stat: 'sum' },
  });
  const rows = [job(1000, 23_000), job(2000, 35_000), wait(5500, 5), wait(40_000, 20),
    { name: 'rollout_groups', timestamp_ms: 25_000, seq: 25_000, value: 1, attributes_json: { disposition: 'consumed' } }];
  const context = { startMs: 0, endMs: 45_000, trainingRows: 2 };
  const { series } = createMeasurements(rows, window, [], null, context);
  const waits = series.taskWait[0].data;
  assert.equal(entryOccupancyAtLongestWait(waits, window).outstanding, 2);
  assert.equal(entryOccupancyAtLongestWait(waits, { from: 0, to: 15_000 }).outstanding, 0);
  const partial = { from: 10_000, to: 50_000 };
  const partialWaits = createMeasurements(rows, partial, [], null, context).series.taskWait[0].data;
  assert.equal(entryOccupancyAtLongestWait(partialWaits, partial), null);
});

test('batch volumes use consumed groups from the complete wait, while releases keep their recorded time', () => {
  const disposition = (name, timestamp_ms, value, disposition = 'consumed') => ({
    name, timestamp_ms, value, seq: timestamp_ms, attributes_json: { step: '1', disposition },
  });
  const rows = [
    disposition('rollout_groups', 2_000, 1),
    disposition('rollout_group_tokens', 2_001, 300),
    disposition('rollout_groups', 13_300, 1, 'stale'),
    disposition('rollout_group_tokens', 13_301, 125, 'stale'),
    disposition('rollout_groups', 22_300, 1),
    disposition('rollout_group_tokens', 22_301, 450),
    disposition('rollout_groups', 24_300, 1),
    disposition('rollout_group_tokens', 24_301, 550),
    { name: 'async_phase_window', timestamp_ms: 26_000, seq: 26_000,
      attributes_json: { phase: 'rollout_wait', outcome: 'success', step: '1' },
      body_json: { started_unix_ms: 4_000, finished_unix_ms: 26_000 } },
  ];
  const { series } = createMeasurements(rows, window);
  assert.deepEqual(series.batchGroups[0].data.map(({ t, value }) => [t, value]), [[26_000, 2]]);
  assert.deepEqual(series.batchTokens[0].data.map(({ t, value }) => [t, value]), [[26_000, 1_000]]);
  assert.deepEqual(series.entryReleases[0].data.map((p) => p.count ?? null), [1, null, 2, null, null]);
  const partial = createMeasurements(rows, { from: 10_000, to: 50_000 }).series;
  assert.deepEqual(partial.batchGroups[0].data, []);
  assert.deepEqual(partial.batchTokens[0].data, []);
});

test('recorded worker intervals give concurrency, and only successful calls supply handoff tokens', () => {
  const job = (start, end, outcome, tokens) => ({
    name: 'rollout_call', timestamp_ms: end, seq: end,
    attributes_json: { outcome }, body_json: {
      started_unix_ms: start, finished_unix_ms: end, response_tokens: tokens,
    },
  });
  const rows = [job(2000, 18_000, 'success', 180), job(4000, 8000, 'success', 80),
    job(31_000, 37_000, 'failure', 370)];
  const { series } = createMeasurements(rows, window);
  assert.deepEqual(series.activeGroupJobs[0].data.map((point) => point.value), [2, 1, 0, 1, null]);
  assert.deepEqual(series.groupJobTime[0].data.map((point) => point.value), [4, 16, 6]);
  assert.deepEqual(series.handoffTokens[0].data.map((point) => point.count ?? null), [80, 180, null, null, null]);
});

test('Workers counts canceled group jobs by their recorded finish times', () => {
  const job = (end, outcome) => ({
    name: 'rollout_call', timestamp_ms: end, seq: end, attributes_json: { outcome },
    body_json: { started_unix_ms: end - 1000, finished_unix_ms: end },
  });
  const rows = [job(12_000, 'success'), job(18_001, 'cancelled'),
    job(18_200, 'cancelled'), job(35_500, null)];
  const jobs = createMeasurements(rows, window).series.groupJobTime[0].data;
  assert.equal(canceledGroupJobCount(jobs, window), 2);
  assert.equal(canceledGroupJobCount(jobs, { from: 18_001, to: 18_200 }), 1);
  assert.equal(canceledGroupJobCount(jobs, { from: 10_000, to: 15_000 }), 0);
  assert.equal(canceledGroupJobCount(jobs, { from: 30_000, to: 40_000 }), null);
});

test('timing distributions count observations in the selected window, including zero and bucket boundaries', () => {
  const data = [0, .01, .105, 6, 601, 99].map((value, i) => ({ t: i * 10_000, value }));
  const { count, p50, p95, buckets } = timingDistribution(data, window);
  assert.deepEqual([count, p50, p95], [5, .105, 601]);
  assert.deepEqual(buckets.map((bucket) => bucket.count), [2, 0, 1, 0, 1, 0, 0, 0, 0, 1]);
  assert.equal(timingDistribution(data, { from: 1000, to: 9000 }), null);
});

test('controls have content for recorded zeros and breakdowns, while missing values stay passive', () => {
  const model = {
    window, run: {}, configuration: { strategy: '' }, batchTarget: null, maxStaleness: 0,
    series: {
      quarantined: [{ data: [{ t: 0, value: 0 }] }],
      calls: [{ data: [{ t: 0, value: null }] }],
      kvCache: [{ group: 'Engine 0', data: [{ t: 0, value: 15 }] }],
    },
  };
  assert.deepEqual(recordedNodeSettings('buffer', model).map(({ value }) => value), [0]);
  assert.deepEqual(recordedNodeSettings('trainer', model), []);
  assert.deepEqual([...cardsWithSamples(model)], ['source', 'inference']);
});

test('training-only startup logs show the training dataset count', () => {
  const row = (line) => ({ key: '/user/job/0:0', line });
  const rows = [
    row('skyrl_train.dataset.dataset:_read_files_and_tokenize:88 - Filtered dataset size: 164'),
    row('    batch_policy: rolling'),
    row('    eval_interval: -1'),
  ];
  const startup = { includesStartup: true, asyncTraining: true };
  assert.deepEqual(driverSetup(rows, startup), { batchPolicy: 'rolling', trainingRows: 164 });
  assert.equal(driverSetup(rows, { ...startup, includesStartup: false }).trainingRows, null);
  assert.equal(driverSetup(rows, { ...startup, asyncTraining: false }).trainingRows, null);
  assert.equal(driverSetup(rows.slice(0, 2), startup).trainingRows, null);
});

test('driver timers locate evals and the final checkpoint, and require one execution', () => {
  const row = (epoch_ms, seq, tail) => ({ epoch_ms, seq, key: '/user/job/0:0',
    line: 'skyrl_train.trainer:_finalize_training:682 - ' + tail });
  const rows = [
    row(110_000, 1, "Finished: 'eval', time cost: 9.00s"),
    row(141_000, 2, "Finished: 'save_checkpoints', time cost: 31.00s"),
    row(145_000, 3, 'Saved final checkpoint.'),
  ];
  assert.deepEqual(driverPhases(rows).map(({ kind, start, end }) => [kind, start, end]), [
    ['evaluation', 101_000, 110_000], ['checkpoint', 110_000, 145_000],
  ]);
  assert.deepEqual(driverPhases([...rows, { ...rows[0], key: '/user/job/0:1' }]), []);
});

test('expansion rewires the same typed edge and chart in both directions', () => {
  const collapsed = flowGraph(false, true);
  const expanded = flowGraph(true, true);
  for (const edge of collapsed.edges) {
    const next = expanded.edges.find((candidate) => candidate.id === edge.id);
    assert.equal(next.card, edge.card);
    assert.equal(next.payload, edge.payload);
    assert.equal(next.kind, edge.kind);
    assert.equal(next.from[1], edge.from[1]);
    assert.equal(next.to[1], edge.to[1]);
    assert.equal(next.from[0], edge.from[0] === 'rollouts' ? 'workers' : edge.from[0]);
    assert.equal(next.to[0], edge.to[0] === 'rollouts'
      ? (['entries', 'leases'].includes(edge.id) ? 'dispatch' : 'workers') : edge.to[0]);
    assert.ok(cards[next.card]);
  }
  assert.deepEqual(flowGraph(false, true), collapsed);
  assert.equal(flowGraph(false, false).edges.some(({ id }) => id === 'gradeSubmit'), false);
});
