import { toDataFrame } from '@grafana/data';
import {
  frameByRefId,
  clusterNodes,
  frameWithField,
  nightlyCells,
  nodeMetrics,
  provisioningRegions,
  provisioningStatus,
  seriesPoints,
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

test('frameWithField returns undefined when no frame carries the field', () => {
  expect(frameWithField([frame([CELL])], 'short_oid')).toBeUndefined();
  expect(frameWithField([], 'lane_id')).toBeUndefined();
});

test('status data contracts preserve worker resources and provisioning outcomes', () => {
  const [worker] = workerRegions(frame([{
    region: 'us-east5', healthy: 12, cpu_millicores: 96000, memory_bytes: 1024, tpu_chips: 64,
  }]));
  expect(worker).toEqual({
    region: 'us-east5', healthy: 12, cpuMillicores: 96000, memoryBytes: 1024, tpuChips: 64,
  });

  const [provisioning] = provisioningStatus(frame([{
    scope: 'fleet', collected_at: 1000, resource_type: '', scale_group: '', zone: '',
    ready: 8, stockout: 1, error: 1, preempted: 2, outcomes: 10, success_ratio: 0.8,
    pools_placing: 4, pools_no_ready_outcome: 1, latency_p50_seconds: 45,
    latency_p95_seconds: 90, window_hours: 3,
  }]));
  expect(provisioning).toMatchObject({
    scope: 'fleet', ready: 8, successRatio: 0.8, poolsNoReadyOutcome: 1, latencyP95Seconds: 90,
  });
});

test('status history reads the configured series and value fields', () => {
  expect(seriesPoints(frame([{ time: 1000, region: 'us-east5', workers: 12 }]), 'region', 'workers')).toEqual([
    { time: 1000, series: 'us-east5', value: 12 },
  ]);
});

test('provisioning regions combine pool outcomes across zones', () => {
  const rows = [
    { scope: 'fleet', zone: '', ready: 9, outcomes: 11 },
    { scope: 'pool', zone: '', ready: 1, outcomes: 1 },
    { scope: 'pool', zone: 'us-west4-a', ready: 0, outcomes: 0 },
    { scope: 'pool', zone: 'us-east5-a', ready: 6, outcomes: 7 },
    { scope: 'pool', zone: 'us-east5-b', ready: 2, outcomes: 3 },
    { scope: 'pool', zone: 'US-EAST-02A', ready: 1, outcomes: 2 },
    { scope: 'pool', zone: 'US-EAST-08A', ready: 3, outcomes: 4 },
  ];

  expect(provisioningRegions(rows)).toEqual([
    { region: 'us-east5', ready: 8, outcomes: 10, successRatio: 0.8 },
    { region: 'US-EAST-02A', ready: 1, outcomes: 2, successRatio: 0.5 },
    { region: 'US-EAST-08A', ready: 3, outcomes: 4, successRatio: 0.75 },
  ]);
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
