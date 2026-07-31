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

test('status page rolls provisioning pools into a logit region graph', () => {
  const now = Date.now();
  const frames = [
    frame('P', [{
      scope: 'fleet', collected_at: now, zone: '',
      ready: 8, stockout: 1, error: 1, preempted: 2, outcomes: 10, success_ratio: 0.8,
      pools_placing: 4, pools_no_ready_outcome: 1, latency_p50_seconds: 45,
      latency_p95_seconds: 90, window_hours: 3,
    }, {
      scope: 'pool', collected_at: now, zone: 'us-east5-a',
      ready: 1, stockout: 49, error: 0, preempted: 0, outcomes: 50, success_ratio: 0.02,
      pools_placing: 0, pools_no_ready_outcome: 0, latency_p50_seconds: 45,
      latency_p95_seconds: 90, window_hours: 3,
    }, {
      scope: 'pool', collected_at: now, zone: 'us-east5-b',
      ready: 1, stockout: 49, error: 0, preempted: 0, outcomes: 50, success_ratio: 0.02,
      pools_placing: 0, pools_no_ready_outcome: 0, latency_p50_seconds: 45,
      latency_p95_seconds: 90, window_hours: 3,
    }, {
      scope: 'pool', collected_at: now, zone: 'us-central1-a',
      ready: 100, stockout: 0, error: 0, preempted: 0, outcomes: 100, success_ratio: 1,
      pools_placing: 0, pools_no_ready_outcome: 0, latency_p50_seconds: 45,
      latency_p95_seconds: 90, window_hours: 3,
    }, {
      scope: 'pool', collected_at: now, zone: 'europe-west4-a',
      ready: 1, stockout: 99, error: 0, preempted: 0, outcomes: 100, success_ratio: 0.01,
      pools_placing: 0, pools_no_ready_outcome: 0, latency_p50_seconds: 45,
      latency_p95_seconds: 90, window_hours: 3,
    }, {
      scope: 'pool', collected_at: now, zone: 'asia-east1-a',
      ready: 5, stockout: 95, error: 0, preempted: 0, outcomes: 100, success_ratio: 0.05,
      pools_placing: 0, pools_no_ready_outcome: 0, latency_p50_seconds: 45,
      latency_p95_seconds: 90, window_hours: 3,
    }]),
    frame('R', [
      { time: now - 60_000, region: 'fleet', success_ratio: 0.7 },
      { time: now, region: 'fleet', success_ratio: 0.8 },
      { time: now - 60_000, region: 'us-east5', success_ratio: 0.02 },
      { time: now, region: 'us-east5', success_ratio: 0.02 },
      { time: now - 60_000, region: 'us-central1', success_ratio: 0.99 },
      { time: now, region: 'us-central1', success_ratio: 1 },
      { time: now - 60_000, region: 'europe-west4', success_ratio: 0.01 },
      { time: now, region: 'europe-west4', success_ratio: 0.01 },
      { time: now - 60_000, region: 'asia-east1', success_ratio: 0.05 },
      { time: now, region: 'asia-east1', success_ratio: 0.05 },
    ]),
  ];

  render(<StatusPage frames={frames} width={1400} height={1500} />);

  const provisioning = screen.getByRole('region', { name: 'Provisioning status' });
  const regionLabel = within(provisioning).getByText('us-east5', { selector: 'code' });
  const regionRow = regionLabel.parentElement;
  expect(regionRow).not.toBeNull();
  expect(regionRow).toHaveTextContent('2%');
  expect(regionRow).toHaveTextContent('2/100');
  const highestRegionRow = within(provisioning).getByText('us-central1', { selector: 'code' }).parentElement;
  expect(highestRegionRow).toHaveTextContent('100%');
  expect(highestRegionRow).toHaveTextContent('100/100');
  const lowestRegionRow = within(provisioning).getByText('europe-west4', { selector: 'code' }).parentElement;
  expect(lowestRegionRow).toHaveTextContent('1%');
  expect(lowestRegionRow).toHaveTextContent('1/100');

  const chart = within(provisioning).getByRole('img', { name: '24 hour status history' });
  expect(within(chart).getByText('1%')).toBeInTheDocument();
  expect(within(chart).getByText('5%')).toBeInTheDocument();
  expect(within(chart).getByText('95%')).toBeInTheDocument();
  expect(within(chart).getByText('99%')).toBeInTheDocument();
  const tickY = (label: string) => Number(within(chart).getByText(label).getAttribute('y'));
  const lowTailGap = Math.abs(tickY('1%') - tickY('5%'));
  const centerGap = Math.abs(tickY('5%') - tickY('50%'));
  expect(lowTailGap / centerGap).toBeGreaterThan(0.4);

  expect(within(provisioning).queryByText('fleet')).not.toBeInTheDocument();
  const chartSeries = within(provisioning).getByRole('list', { name: 'Status history series' });
  expect(within(chartSeries).getByText('us-east5')).toBeInTheDocument();
  expect(within(chartSeries).getByText('asia-east1')).toBeInTheDocument();
  expect(within(chartSeries).getByText('us-central1')).toBeInTheDocument();
  expect(within(chartSeries).getByText('europe-west4')).toBeInTheDocument();
  expect(within(provisioning).getByText('trailing 3 hour window · logit scale')).toBeInTheDocument();
  expect(within(provisioning).queryByText('us-east5-a')).not.toBeInTheDocument();
});
