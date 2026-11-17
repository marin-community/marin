import { toDataFrame } from '@grafana/data';
import {
  frameByRefId,
  clusterNodes,
  frameWithField,
  nightlyCells,
  nodeMetrics,
  smUtilizationPoints,
  smUtilizationRasterData,
  taskUsage,
  workerRegions,
  workloadAllocations,
} from './data';

function frame(rows: Array<Record<string, unknown>>, refId?: string) {
  return toDataFrame({
    refId,
    fields: Object.keys(rows[0]).reverse().map((name) => ({ name, values: rows.map((row) => row[name]) })),
  });
}

const CELL = {
  lane_id: 'tpu-ferry', date: '2026-07-21', lane: 'TPU ferry', label: 'TPU ferry',
  group: 'marin', subgroup: 'training', state: 'run', status: 'completed', duration_state: 'normal',
  duration_seconds: 3600, healthy: true, due: true, url: 'https://example/run', lane_order: 0,
};

test('nightlyCells reads fields by name and preserves the link contract', () => {
  const [cell] = nightlyCells(frame([CELL]));
  expect(cell).toMatchObject({
    laneId: 'tpu-ferry', runStatus: 'completed', durationSeconds: 3600, healthy: true, url: 'https://example/run',
  });
});

test('nightlyCells rejects duplicate lane and date cells', () => {
  expect(() => nightlyCells(frame([CELL, CELL]))).toThrow('Duplicate nightly cell');
});

test('frameWithField rejects ambiguous multiple frames', () => {
  const value = frame([CELL]);
  expect(() => frameWithField([value, value], 'lane_id')).toThrow('received 2');
});

test('frameWithField returns undefined when no frame carries the field', () => {
  expect(frameWithField([frame([CELL])], 'short_oid')).toBeUndefined();
  expect(frameWithField([], 'lane_id')).toBeUndefined();
});

test('status data contract preserves worker resources', () => {
  const [worker] = workerRegions(frame([{
    region: 'us-east5', healthy: 12, cpu_millicores: 96000, memory_bytes: 1024, tpu_chips: 64,
  }]));
  expect(worker).toEqual({
    region: 'us-east5', healthy: 12, cpuMillicores: 96000, memoryBytes: 1024, tpuChips: 64,
  });

});

test('frameByRefId isolates one status source', () => {
  const nightly = frame([CELL], 'N');
  const workers = frame([{ healthy: 12 }], 'W');
  expect(frameByRefId([nightly, workers], 'W')).toBe(workers);
  expect(frameByRefId([nightly], 'W')).toBeUndefined();
});

test('cluster capacity contracts read placement, inventory, and observed usage by field name', () => {
  expect(workloadAllocations(frame([{
    cluster: 'cw-us-east-02a', namespace: 'iris', pod: 'train-0', node: 'gpu-1', job: '/alice/train',
    task: '/alice/train/0', phase: 'Running', ready: true, priority_class: 'production', age_seconds: 120,
    cpu_request_millicores: 8000, memory_request_bytes: 64, gpu_request_count: 4, gpu_variant: 'H100',
  }]))[0]).toMatchObject({ job: '/alice/train', gpuRequestCount: 4, cpuRequestMillicores: 8000 });

  expect(clusterNodes(frame([{
    cluster: 'cw-us-east-02a', node: 'gpu-1', instance_type: 'h100-8', node_pool: 'train', gpu_model: 'H100',
    gpu_capacity: 8, gpu_allocatable: 8, cpu_allocatable: '96', memory_allocatable: '1Ti', ready: true,
    unschedulable: false,
  }]))[0]).toMatchObject({ node: 'gpu-1', gpuAllocatable: 8, cpuAllocatable: '96' });

  expect(taskUsage(frame([{
    cluster: 'cw-us-east-02a', task: '/alice/train/0', pod: 'train-0', cpu_millicores: 3200,
    memory_bytes: 32, sampled_at: 1000,
  }]))[0]).toMatchObject({ pod: 'train-0', cpuMillicores: 3200, memoryBytes: 32 });

  expect(nodeMetrics(frame([{
    cluster: 'cw-us-east-02a', node: 'gpu-1', name: 'gpu_utilization_percent', value: 82, sampled_at: 1000,
  }]))).toEqual([{
    cluster: 'cw-us-east-02a', node: 'gpu-1', name: 'gpu_utilization_percent', value: 82, sampledAt: 1000,
  }]);
});

test('SM raster orders devices and preserves missing time buckets', () => {
  const points = smUtilizationPoints(frame([
    { cluster: 'cw-b', node: 'node-10', gpu: '1', time: 60_000, sm_utilization: 90 },
    { cluster: 'cw-a', node: 'node-2', gpu: '0', time: 60_000, sm_utilization: 30 },
    { cluster: 'cw-a', node: 'node-2', gpu: '0', time: 0, sm_utilization: 20 },
  ]));

  const raster = smUtilizationRasterData(points);

  expect(raster.devices).toEqual([
    { cluster: 'cw-a', node: 'node-2', gpu: '0' },
    { cluster: 'cw-b', node: 'node-10', gpu: '1' },
  ]);
  expect(raster.timestamps).toEqual([0, 60_000]);
  expect(Array.from(raster.values)).toEqual([20, 30, Number.NaN, 90]);
});
