import React from 'react';
import { toDataFrame } from '@grafana/data';
import { render, screen, within } from '@testing-library/react';
import { CommitStrip } from './CommitStrip';
import { NightlyMatrix } from './NightlyMatrix';
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
});

function frame(refId: string, rows: Array<Record<string, unknown>>) {
  return toDataFrame({
    refId,
    fields: Object.keys(rows[0]).map((name) => ({ name, values: rows.map((row) => row[name]) })),
  });
}

test('status page keeps worker and provisioning status visible when another source has no data', () => {
  const frames = [
    frame('W', [{
      region: 'us-east5', healthy: 12, cpu_millicores: 96000, memory_bytes: 1099511627776, tpu_chips: 64,
    }]),
    frame('P', [{
      scope: 'fleet', collected_at: Date.now(), zone: '',
      ready: 8, stockout: 1, error: 1, preempted: 2, outcomes: 10, success_ratio: 0.8,
      pools_placing: 4, pools_no_ready_outcome: 1, latency_p50_seconds: 45,
      latency_p95_seconds: 90, window_hours: 3,
    }]),
  ];

  render(<StatusPage frames={frames} width={1400} height={1500} />);

  expect(screen.getByRole('main', { name: 'Marin infrastructure status' })).toBeInTheDocument();
  const workers = screen.getByRole('region', { name: 'Worker status' });
  expect(within(workers).getByText('healthy workers')).toBeInTheDocument();
  expect(within(workers).getByText('us-east5')).toBeInTheDocument();
  const provisioning = screen.getByRole('region', { name: 'Provisioning status' });
  expect(within(provisioning).getByText('80%')).toBeInTheDocument();
  expect(within(provisioning).getByText('No region data')).toBeInTheDocument();
  expect(within(provisioning).getByText('pools without ready outcome')).toBeInTheDocument();
  expect(screen.getAllByText('No W&B data')).toHaveLength(3);
});

test('status page keeps extreme provisioning values in the region table and out of the graph', () => {
  const now = Date.now();
  const frames = [
    frame('P', [{
      scope: 'fleet', collected_at: now, zone: '',
      ready: 8, stockout: 1, error: 1, preempted: 2, outcomes: 10, success_ratio: 0.8,
      pools_placing: 4, pools_no_ready_outcome: 1, latency_p50_seconds: 45,
      latency_p95_seconds: 90, window_hours: 3,
    }, {
      scope: 'pool', collected_at: now, zone: 'us-east5-a',
      ready: 6, stockout: 1, error: 0, preempted: 0, outcomes: 7, success_ratio: 6 / 7,
      pools_placing: 0, pools_no_ready_outcome: 0, latency_p50_seconds: 45,
      latency_p95_seconds: 90, window_hours: 3,
    }, {
      scope: 'pool', collected_at: now, zone: 'us-east5-b',
      ready: 2, stockout: 0, error: 1, preempted: 0, outcomes: 3, success_ratio: 2 / 3,
      pools_placing: 0, pools_no_ready_outcome: 0, latency_p50_seconds: 45,
      latency_p95_seconds: 90, window_hours: 3,
    }, {
      scope: 'pool', collected_at: now, zone: 'us-central1-a',
      ready: 10, stockout: 0, error: 0, preempted: 0, outcomes: 10, success_ratio: 1,
      pools_placing: 0, pools_no_ready_outcome: 0, latency_p50_seconds: 45,
      latency_p95_seconds: 90, window_hours: 3,
    }, {
      scope: 'pool', collected_at: now, zone: 'europe-west4-a',
      ready: 1, stockout: 1, error: 0, preempted: 0, outcomes: 2, success_ratio: 0.5,
      pools_placing: 0, pools_no_ready_outcome: 0, latency_p50_seconds: 45,
      latency_p95_seconds: 90, window_hours: 3,
    }]),
    frame('R', [
      { time: now - 60_000, region: 'fleet', success_ratio: 0.7 },
      { time: now, region: 'fleet', success_ratio: 0.8 },
      { time: now - 60_000, region: 'us-east5', success_ratio: 0.7 },
      { time: now, region: 'us-east5', success_ratio: 0.8 },
      { time: now - 60_000, region: 'us-central1', success_ratio: 0.9 },
      { time: now, region: 'us-central1', success_ratio: 1 },
      { time: now - 60_000, region: 'europe-west4', success_ratio: 0.4 },
      { time: now, region: 'europe-west4', success_ratio: 0.5 },
    ]),
  ];

  render(<StatusPage frames={frames} width={1400} height={1500} />);

  const provisioning = screen.getByRole('region', { name: 'Provisioning status' });
  const regionLabel = within(provisioning).getByText('us-east5', { selector: 'code' });
  const regionRow = regionLabel.parentElement;
  expect(regionRow).not.toBeNull();
  expect(regionRow).toHaveTextContent('80%');
  expect(regionRow).toHaveTextContent('8/10');
  const highestRegionRow = within(provisioning).getByText('us-central1', { selector: 'code' }).parentElement;
  expect(highestRegionRow).toHaveTextContent('100%');
  expect(highestRegionRow).toHaveTextContent('10/10');
  const lowestRegionRow = within(provisioning).getByText('europe-west4', { selector: 'code' }).parentElement;
  expect(lowestRegionRow).toHaveTextContent('50%');
  expect(lowestRegionRow).toHaveTextContent('1/2');

  const chart = within(provisioning).getByRole('img', { name: '24 hour status history' });
  const chartContainer = chart.parentElement;
  expect(chartContainer).not.toBeNull();
  expect(within(chartContainer!).queryByText('fleet')).not.toBeInTheDocument();
  expect(within(chartContainer!).getByText('us-east5')).toBeInTheDocument();
  expect(within(chartContainer!).queryByText('us-central1')).not.toBeInTheDocument();
  expect(within(chartContainer!).queryByText('europe-west4')).not.toBeInTheDocument();
  expect(within(provisioning).queryByText('us-east5-a')).not.toBeInTheDocument();
});
