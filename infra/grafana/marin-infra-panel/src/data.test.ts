import { toDataFrame } from '@grafana/data';
import { frameWithField, nightlyCells } from './data';

function frame(rows: Array<Record<string, unknown>>) {
  return toDataFrame({ fields: Object.keys(rows[0]).reverse().map((name) => ({ name, values: rows.map((row) => row[name]) })) });
}

const CELL = {
  lane_id: 'tpu-ferry', date: '2026-07-21', lane: 'TPU ferry', label: 'TPU ferry',
  group: 'marin', subgroup: 'training', state: 'run', duration_state: 'normal',
  duration_seconds: 3600, healthy: true, due: true, url: 'https://example/run', lane_order: 0,
};

test('nightlyCells reads fields by name and preserves the link contract', () => {
  const [cell] = nightlyCells(frame([CELL]));
  expect(cell).toMatchObject({ laneId: 'tpu-ferry', durationSeconds: 3600, healthy: true, url: 'https://example/run' });
});

test('nightlyCells rejects duplicate lane and date cells', () => {
  expect(() => nightlyCells(frame([CELL, CELL]))).toThrow('Duplicate nightly cell');
});

test('frameWithField rejects ambiguous multiple frames', () => {
  const value = frame([CELL]);
  expect(() => frameWithField([value, value], 'lane_id')).toThrow('received 2');
});
