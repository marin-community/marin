import React from 'react';
import { toDataFrame } from '@grafana/data';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { CommitStrip } from './CommitStrip';
import { ClusterCapacity } from './ClusterCapacity';
import { NightlyMatrix } from './NightlyMatrix';
import { SmUtilizationRaster } from './SmUtilizationRaster';
import { StatusPage } from './StatusPage';
import { WandbChart } from './WandbChart';

// An empty query result — what the bridge returns while a source is briefly
// unavailable (missing token, GitHub rate limit, cold cache). The views used to
// throw here, crashing the whole panel into Grafana's error boundary.
test('every view renders a placeholder instead of throwing on empty data', () => {
  const { rerender } = render(<CommitStrip frames={[]} width={480} height={80} />);
  expect(screen.getByText('No commit data')).toBeInTheDocument();

  rerender(<NightlyMatrix frames={[]} width={480} height={200} />);
  expect(screen.getByText('No nightly data')).toBeInTheDocument();

  rerender(<WandbChart frames={[]} width={480} height={200} />);
  expect(screen.getByText('No W&B data')).toBeInTheDocument();

  rerender(<ClusterCapacity frames={[]} width={1200} height={800} />);
  expect(screen.getByText('No Kubernetes node inventory reported.')).toBeInTheDocument();

  rerender(<SmUtilizationRaster frames={[]} width={700} height={240} />);
  expect(screen.getByLabelText('No data')).toBeInTheDocument();
});

function frame(refId: string, rows: Array<Record<string, unknown>>) {
  return toDataFrame({
    refId,
    fields: Object.keys(rows[0]).map((name) => ({ name, values: rows.map((row) => row[name]) })),
  });
}

test('nightly matrix presents active runs as running and summarizes states by category', () => {
  const base = {
    date: '2026-08-31', label: 'Nightly', group: 'marin', subgroup: 'training', state: 'run',
    duration_state: 'normal', conclusion: null, url: 'https://example/run', workflow_url: 'https://example/workflow',
    healthy: false, due: true, source_error: null,
  };
  const rows = [
    { ...base, lane_id: 'queued', lane: 'Queued', label: 'Queued nightly', status: 'queued', duration_seconds: null, lane_order: 0 },
    { ...base, lane_id: 'active', lane: 'Active', label: 'Active nightly', status: 'in_progress', duration_seconds: 600, lane_order: 1 },
    { ...base, lane_id: 'failed', lane: 'Failed', label: 'Failed nightly', status: 'completed', conclusion: 'failure', duration_seconds: 600, lane_order: 2 },
  ];

  render(<NightlyMatrix frames={[frame('N', rows)]} width={1200} height={300} />);

  expect(screen.getAllByRole('link', { name: /Running/ })).toHaveLength(2);
  expect(screen.getByRole('link', { name: /Active nightly.*Running.*10m/ })).toBeInTheDocument();
  expect(screen.getByText(/Today: 2 running · 1 failed/)).toBeInTheDocument();
  const legend = screen.getByRole('note', { name: 'Nightly state legend' });
  for (const state of ['Passed', 'Running', 'Slow', 'Failed', 'No run', 'Data unavailable', 'Not due']) {
    expect(legend).toHaveTextContent(state);
  }
});

test('cluster capacity rolls tasks into jobs and packs requested GPUs onto their nodes', () => {
  const frames = [
    frame('W', [{
      cluster: 'cw-us-east-02a', namespace: 'iris', pod: 'train-0', node: 'gpu-1', job: '/alice/train',
      task: '/alice/train/0', phase: 'Running', ready: true, priority_class: 'iris-production', age_seconds: 120,
      cpu_request_millicores: 8000, memory_request_bytes: 68719476736, gpu_request_count: 2, gpu_variant: 'H100',
    }, {
      cluster: 'cw-us-east-02a', namespace: 'iris', pod: 'eval-0', node: '', job: '/bob/eval',
      task: '/bob/eval/0', phase: 'Pending', ready: false, priority_class: '', age_seconds: 60,
      cpu_request_millicores: 2000, memory_request_bytes: 8589934592, gpu_request_count: 1, gpu_variant: 'H100',
    }]),
    frame('N', [{
      cluster: 'cw-us-east-02a', node: 'gpu-1', instance_type: 'h100-4', node_pool: 'train', gpu_model: 'H100',
      gpu_capacity: 4, gpu_allocatable: 4, cpu_allocatable: '32', memory_allocatable: '256Gi', ready: true,
      unschedulable: false,
    }]),
    frame('J', [{ job: '/alice/train' }]),
    frame('T', [{
      cluster: 'cw-us-east-02a', task: '/alice/train/0', pod: 'train-0', cpu_millicores: 3500,
      memory_bytes: 34359738368, sampled_at: Date.now(),
    }]),
    frame('H', [
      { cluster: 'cw-us-east-02a', node: 'gpu-1', name: 'node_cpu_utilization_percent', value: 44, sampled_at: Date.now() },
      { cluster: 'cw-us-east-02a', node: 'gpu-1', name: 'gpu_utilization_percent', value: 78, sampled_at: Date.now() },
    ]),
  ];

  render(<ClusterCapacity frames={frames} width={1400} height={1000} />);

  expect(screen.getByRole('main', { name: 'Cluster capacity' })).toHaveTextContent('cw-us-east-02a');
  const totals = screen.getByRole('region', { name: 'Cluster totals' });
  expect(within(totals).getByText('2/4')).toBeInTheDocument();
  expect(within(totals).getByText('tasks waiting').parentElement).toHaveTextContent('1');
  const jobs = screen.getByRole('region', { name: 'Active jobs' });
  const trainJob = within(jobs).getByRole('row', { name: /alice\/train/ });
  expect(within(trainJob).getByText('/alice/train')).toBeInTheDocument();
  expect(within(trainJob).getByText('production')).toBeInTheDocument();
  expect(within(trainJob).getByRole('link', { name: 'Open' })).toHaveAttribute(
    'href',
    'https://iris.oa.dev/#/job/%2Falice%2Ftrain?cluster=cw-us-east-02a'
  );
  const directJob = within(jobs).getByRole('row', { name: /bob\/eval/ });
  expect(within(directJob).queryByRole('link', { name: 'Open' })).not.toBeInTheDocument();
  expect(within(jobs).getByText('3.50 cores')).toBeInTheDocument();
  const slots = screen.getByRole('list', { name: 'GPU slots on gpu-1' });
  expect(within(slots).getAllByRole('listitem', { name: '/alice/train GPU' })).toHaveLength(2);
  expect(within(slots).getAllByRole('listitem', { name: 'Unallocated GPU' })).toHaveLength(2);
  expect(screen.getByRole('region', { name: 'Unbound tasks' })).toHaveTextContent('/bob/eval/0');
});

test('status page keeps worker status visible when another source has no data', () => {
  const frames = [
    frame('W', [{
      region: 'us-east5', healthy: 12, cpu_millicores: 96000, memory_bytes: 1099511627776, tpu_chips: 64,
    }]),
  ];

  render(<StatusPage frames={frames} width={1400} height={1500} />);

  expect(screen.getByRole('main', { name: 'Marin infrastructure status' })).toBeInTheDocument();
  const workers = screen.getByRole('region', { name: 'Worker status' });
  expect(within(workers).getByText('healthy workers')).toBeInTheDocument();
  expect(within(workers).getByText('us-east5')).toBeInTheDocument();
  expect(screen.getAllByText('No W&B data')).toHaveLength(3);
});

test('SM raster hover matches the painted time bucket', () => {
  const context = {
    setTransform: jest.fn(), clearRect: jest.fn(), fillRect: jest.fn(), beginPath: jest.fn(),
    moveTo: jest.fn(), lineTo: jest.fn(), stroke: jest.fn(), fillText: jest.fn(),
  } as unknown as CanvasRenderingContext2D;
  const contextSpy = jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context);
  const frames = [frame('A', [
    { time: 0, cluster: 'cw-a', node: 'node-1', gpu: '0', sm_utilization: 25 },
    { time: 60_000, cluster: 'cw-a', node: 'node-1', gpu: '0', sm_utilization: 75 },
    { time: 0, cluster: 'cw-a', node: 'node-1', gpu: '1', sm_utilization: 50 },
  ])];

  render(<SmUtilizationRaster frames={frames} width={700} height={240} />);
  const canvas = screen.getByRole('img');
  jest.spyOn(canvas, 'getBoundingClientRect').mockReturnValue({
    x: 0, y: 0, left: 0, top: 0, right: 700, bottom: 240, width: 700, height: 240, toJSON: () => ({}),
  });

  // The first bucket spans x=100..399. This point is past its midpoint, where
  // nearest-sample hit testing would incorrectly select the second bucket.
  fireEvent.mouseMove(canvas, { clientX: 330, clientY: 10 });

  expect(screen.getByRole('tooltip')).toHaveTextContent('25.0%');
  expect(screen.getByRole('tooltip')).toHaveTextContent('node-1');
  contextSpy.mockRestore();
});
