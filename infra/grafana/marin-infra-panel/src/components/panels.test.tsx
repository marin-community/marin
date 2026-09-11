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

test('nightly matrix presents every nonterminal run as running and summarizes states by category', () => {
  const base = {
    date: '2026-08-31', label: 'Nightly', group: 'marin', subgroup: 'training', state: 'run',
    duration_state: 'normal', conclusion: null, url: 'https://example/run', workflow_url: 'https://example/workflow',
    healthy: false, due: true, source_error: null,
  };
  const rows = [
    { ...base, lane_id: 'queued', lane: 'Queued', label: 'Queued nightly', status: 'queued', duration_seconds: null, lane_order: 0 },
    { ...base, lane_id: 'active', lane: 'Active', label: 'Active nightly', status: 'in_progress', duration_seconds: 600, lane_order: 1 },
    { ...base, lane_id: 'requested', lane: 'Requested', label: 'Requested nightly', status: 'requested', duration_seconds: null, lane_order: 2 },
    { ...base, lane_id: 'waiting', lane: 'Waiting', label: 'Waiting nightly', status: 'waiting', duration_seconds: null, lane_order: 3 },
    { ...base, lane_id: 'pending', lane: 'Pending', label: 'Pending nightly', status: 'pending', duration_seconds: null, lane_order: 4 },
    { ...base, lane_id: 'failed', lane: 'Failed', label: 'Failed nightly', status: 'completed', conclusion: 'failure', duration_seconds: 600, lane_order: 5 },
  ];

  render(<NightlyMatrix frames={[frame('N', rows)]} width={1200} height={300} />);

  expect(screen.getAllByRole('link', { name: /Running/ })).toHaveLength(5);
  expect(screen.getByRole('link', { name: /Active nightly.*Running.*10m/ })).toBeInTheDocument();
  expect(screen.getByText(/Today: 5 running · 1 failed/)).toBeInTheDocument();
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

function wandbFrame(chart: string, samples: Array<{ tokens: number; value: number; run?: string }>) {
  return frame('A', samples.map((sample) => ({
    chart, report_title: 'Hero run', report_url: 'https://example/report', run: 'hero', ...sample,
  })));
}

test('MFU starts at zero and expands above 30 percent without clipping samples', () => {
  const { rerender } = render(<WandbChart frames={[wandbFrame('MFU (%)', [
    { tokens: 1e9, value: 21 }, { tokens: 2e9, value: 24 },
  ])]} width={480} height={260} />);
  for (const tick of ['0%', '10%', '20%', '30%']) {
    expect(screen.getByText(tick)).toBeInTheDocument();
  }
  rerender(<WandbChart frames={[wandbFrame('MFU (%)', [
    { tokens: 1e9, value: 0 }, { tokens: 2e9, value: 35 }, { tokens: 3e9, value: 45 },
  ])]} width={480} height={260} />);
  const axis = Array.from(screen.getByRole('img').querySelectorAll('g text'), (label) => parseFloat(label.textContent!));
  expect(Math.max(...axis)).toBeGreaterThanOrEqual(45);
  const plotted = screen.getByRole('img').querySelector('polyline')!.getAttribute('points')!.split(' ');
  const heights = plotted.map((point) => Number(point.split(',')[1]));
  expect(heights[0]).toBe(240);
  expect(heights[1]).toBeGreaterThan(heights[2]);
  expect(heights[2]).toBeGreaterThan(56);
});

test('train loss uses the preferred range and expands for rare extremes', () => {
  const samples = Array.from({ length: 100 }, (_, index) => ({ tokens: index * 1e9, value: 1.3 }));
  const { rerender } = render(<WandbChart frames={[wandbFrame('Train cross-entropy loss', samples)]} width={480} height={260} />);
  for (const tick of ['1.2', '1.3', '1.4', '1.5', '1.6']) {
    expect(screen.getByText(tick)).toBeInTheDocument();
  }
  samples[0].value = 1.1;
  samples[99].value = 1.8;
  rerender(<WandbChart frames={[wandbFrame('Train cross-entropy loss', samples)]} width={480} height={260} />);
  const svg = screen.getByRole('img');
  const axis = Array.from(svg.querySelectorAll('g text'), (label) => Number(label.textContent));
  expect(Math.min(...axis)).toBeLessThanOrEqual(1.1);
  expect(Math.max(...axis)).toBeGreaterThanOrEqual(1.8);
  const plotted = svg.querySelector('polyline')!.getAttribute('points')!.split(' ');
  const heights = plotted.map((point) => Number(point.split(',')[1]));
  expect(heights[0]).toBeGreaterThan(heights[1]);
  expect(heights[99]).toBeLessThan(heights[1]);
});

test('eval excludes the first percent of tokens and restores all runs with the same colors', () => {
  const samples = [
    { tokens: 0, value: 11.8, run: 'initial' },
    { tokens: 99, value: 10, run: 'initial' },
    { tokens: 10000, value: 2.22, run: 'resumed' },
    { tokens: 100, value: 2.24, run: 'resumed' },
    { tokens: 5000, value: 2.23, run: 'resumed' },
  ];
  render(<WandbChart frames={[wandbFrame('Paloma macro loss (dropless)', samples)]} width={480} height={260} />);
  const control = screen.getByRole('combobox', { name: 'Evaluation range' });
  expect(control).toHaveDisplayValue('After initialization');
  const svg = screen.getByRole('img');
  expect(Array.from(svg.querySelectorAll('g text'), (label) => label.textContent)).toEqual(['2.218', '2.230', '2.242']);
  const resumed = svg.querySelectorAll('polyline')[1];
  const color = resumed.getAttribute('stroke');
  const plotted = resumed.getAttribute('points')!.split(' ').map((point) => point.split(',').map(Number));
  expect(plotted).toHaveLength(3);
  expect(plotted[0][0]).toBe(60);
  expect(plotted[2][0]).toBe(470);
  expect(plotted[0][1]).toBeGreaterThan(56);
  expect(plotted[2][1]).toBeLessThan(240);
  expect(svg.querySelectorAll('polyline')[0]).toHaveAttribute('points', '');

  fireEvent.change(control, { target: { value: 'full-run' } });
  expect(control).toHaveDisplayValue('Full run');
  expect(svg.querySelectorAll('polyline')[0].getAttribute('points')!.split(' ')).toHaveLength(2);
  expect(svg.querySelectorAll('polyline')[1]).toHaveAttribute('stroke', color);
  expect(Math.max(...Array.from(svg.querySelectorAll('g text'), (label) => Number(label.textContent)))).toBeGreaterThan(11.8);

  fireEvent.change(control, { target: { value: 'after-initialization' } });
  expect(screen.getByText('2.242')).toBeInTheDocument();
});

test('a single eval at zero tokens stays visible with a finite axis', () => {
  render(<WandbChart frames={[wandbFrame('Paloma macro loss (dropless)', [
    { tokens: 0, value: 2.23 },
  ])]} width={280} height={260} />);
  const point = screen.getByRole('img').querySelector('circle')!;
  expect(Number(point.getAttribute('cy'))).toBeGreaterThan(76);
  expect(Number(point.getAttribute('cy'))).toBeLessThan(240);
  expect(screen.getByRole('combobox')).toHaveDisplayValue('After initialization');
});
