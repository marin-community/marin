import React, { useMemo } from 'react';
import { css } from '@emotion/css';
import { DataFrame } from '@grafana/data';
import { useTheme2 } from '@grafana/ui';
import { clusterNodes, frameByRefId, nodeMetrics, taskUsage, workloadAllocations } from '../data';
import { formatBytes } from '../format';
import { ClusterNode, NodeMetric, TaskUsage, WorkloadAllocation } from '../types';
import { SERIES_COLORS } from './palette';

interface Props {
  frames: DataFrame[];
  width: number;
  height: number;
}

interface JobSummary {
  job: string;
  priorityClass: string;
  tasks: number;
  ready: number;
  observed: number;
  cpuRequestMillicores: number;
  memoryRequestBytes: number;
  gpuRequestCount: number;
  cpuMillicores: number;
  memoryBytes: number;
}

const REF = { workloads: 'W', nodes: 'N', tasks: 'T', host: 'H' } as const;
const IRIS_DASHBOARD_ORIGIN = 'https://iris.oa.dev';
const KIB = 1024;
const MILLICORES_PER_CORE = 1000;
const SECONDS_PER_MINUTE = 60;
const SECONDS_PER_HOUR = 60 * SECONDS_PER_MINUTE;

function frameRows<T>(frames: DataFrame[], refId: string, read: (frame: DataFrame) => T[]): T[] {
  const frame = frameByRefId(frames, refId);
  return frame ? read(frame) : [];
}

function formatCores(millicores: number): string {
  const cores = millicores / MILLICORES_PER_CORE;
  return cores >= 100 ? Math.round(cores).toString() : cores.toFixed(cores >= 10 ? 1 : 2);
}

function formatAge(seconds: number): string {
  if (seconds < SECONDS_PER_HOUR) {
    return `${Math.max(1, Math.round(seconds / SECONDS_PER_MINUTE))}m`;
  }
  return `${(seconds / SECONDS_PER_HOUR).toFixed(1)}h`;
}

function parseCpu(quantity: string): number {
  if (!quantity) {
    return 0;
  }
  return quantity.endsWith('m')
    ? Number(quantity.slice(0, -1))
    : Number(quantity) * MILLICORES_PER_CORE;
}

function parseMemory(quantity: string): number {
  const match = quantity.match(/^([0-9.]+)([KMGTEP]i)?$/);
  if (!match) {
    return 0;
  }
  const units = ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi'];
  return Number(match[1]) * KIB ** units.indexOf(match[2] ?? '');
}

function formatPercent(value?: number): string {
  return value === undefined ? '—' : `${Math.round(value)}%`;
}

function shortJob(job: string): string {
  return job.split('/').filter(Boolean).slice(-2).join('/');
}

function jobColor(job: string, jobs: string[]): string {
  return SERIES_COLORS[jobs.indexOf(job) % SERIES_COLORS.length];
}

function metricMap(metrics: NodeMetric[]): Map<string, number> {
  return new Map(metrics.map((metric) => [`${metric.node}\u0000${metric.name}`, metric.value]));
}

function usageMap(usage: TaskUsage[]): Map<string, TaskUsage> {
  return new Map(usage.flatMap((row) => [[row.pod, row], [row.task, row]]));
}

function summarizeJobs(workloads: WorkloadAllocation[], usage: Map<string, TaskUsage>): JobSummary[] {
  const summaries = new Map<string, JobSummary>();
  for (const workload of workloads) {
    const observed = usage.get(workload.pod) ?? usage.get(workload.task);
    const summary = summaries.get(workload.job) ?? {
      job: workload.job,
      priorityClass: workload.priorityClass,
      tasks: 0,
      ready: 0,
      observed: 0,
      cpuRequestMillicores: 0,
      memoryRequestBytes: 0,
      gpuRequestCount: 0,
      cpuMillicores: 0,
      memoryBytes: 0,
    };
    summary.tasks += 1;
    summary.ready += Number(workload.ready);
    summary.observed += Number(observed !== undefined);
    summary.cpuRequestMillicores += workload.cpuRequestMillicores;
    summary.memoryRequestBytes += workload.memoryRequestBytes;
    summary.gpuRequestCount += workload.gpuRequestCount;
    summary.cpuMillicores += observed?.cpuMillicores ?? 0;
    summary.memoryBytes += observed?.memoryBytes ?? 0;
    summaries.set(workload.job, summary);
  }
  return [...summaries.values()].sort((a, b) => b.gpuRequestCount - a.gpuRequestCount || a.job.localeCompare(b.job));
}

function Stat({ value, label }: { value: React.ReactNode; label: string }) {
  const theme = useTheme2();
  return (
    <div className={css`min-width:112px;padding:10px 12px;border:1px solid ${theme.colors.border.weak};border-radius:6px;background:${theme.colors.background.secondary};`}>
      <strong className={css`display:block;font-size:20px;line-height:1.2;`}>{value}</strong>
      <span className={css`font-size:12px;color:${theme.colors.text.secondary};`}>{label}</span>
    </div>
  );
}

function ResourceBar({ label, used, capacity, actual }: { label: string; used: number; capacity: number; actual?: number }) {
  const theme = useTheme2();
  const reserved = capacity > 0 ? Math.min(100, used / capacity * 100) : 0;
  return (
    <div className={css`display:grid;grid-template-columns:44px 1fr auto;gap:7px;align-items:center;font-size:11px;`}>
      <span className={css`color:${theme.colors.text.secondary};`}>{label}</span>
      <div className={css`height:6px;border-radius:4px;background:${theme.colors.background.canvas};overflow:hidden;`}>
        <div className={css`width:${reserved}%;height:100%;background:${theme.colors.primary.main};`} />
      </div>
      <span>{Math.round(reserved)}% req{actual === undefined ? '' : ` · ${formatPercent(actual)} live`}</span>
    </div>
  );
}

function NodeCard({ node, workloads, metrics, jobs }: {
  node: ClusterNode;
  workloads: WorkloadAllocation[];
  metrics: Map<string, number>;
  jobs: string[];
}) {
  const theme = useTheme2();
  const gpuCapacity = node.gpuAllocatable || node.gpuCapacity;
  const assigned = workloads.flatMap((workload) =>
    Array.from({ length: workload.gpuRequestCount }, () => workload)
  );
  const cpuRequested = workloads.reduce((sum, workload) => sum + workload.cpuRequestMillicores, 0);
  const memoryRequested = workloads.reduce((sum, workload) => sum + workload.memoryRequestBytes, 0);
  const cpuCapacity = parseCpu(node.cpuAllocatable);
  const memoryCapacity = parseMemory(node.memoryAllocatable);
  const metric = (name: string) => metrics.get(`${node.node}\u0000${name}`);
  const memoryUsed = metric('node_memory_used_bytes');
  const memoryTotal = metric('node_memory_total_bytes');
  const memoryPercent = memoryUsed !== undefined && memoryTotal ? memoryUsed / memoryTotal * 100 : undefined;
  const gpuMemoryUsed = metric('gpu_memory_used_bytes');
  const gpuMemoryTotal = metric('gpu_memory_total_bytes');
  const gpuMemoryPercent = gpuMemoryUsed !== undefined && gpuMemoryTotal ? gpuMemoryUsed / gpuMemoryTotal * 100 : undefined;
  const unhealthy = !node.ready || node.unschedulable;
  const detailUrl = `/d/marin-nodes?var-cluster=${encodeURIComponent(node.cluster)}&var-node=${encodeURIComponent(node.node)}`;
  return (
    <article aria-label={`Node ${node.node}`} className={css`border:1px solid ${unhealthy ? theme.colors.error.border : theme.colors.border.weak};border-radius:7px;background:${theme.colors.background.secondary};padding:11px;display:flex;flex-direction:column;gap:9px;`}>
      <header className={css`display:flex;justify-content:space-between;gap:10px;`}>
        <div>
          <a href={detailUrl} className={css`font-family:${theme.typography.fontFamilyMonospace};font-weight:600;`}>{node.node}</a>
          <div className={css`font-size:11px;color:${theme.colors.text.secondary};`}>{node.nodePool || node.instanceType || 'unclassified'} · {node.gpuModel || 'CPU'}</div>
        </div>
        <span className={css`font-size:11px;color:${unhealthy ? theme.colors.error.text : theme.colors.text.secondary};`}>{unhealthy ? (node.ready ? 'cordoned' : 'not ready') : 'ready'}</span>
      </header>
      {gpuCapacity > 0 && (
        <div>
          <div className={css`display:flex;justify-content:space-between;font-size:11px;margin-bottom:5px;`}><span>GPU packing</span><strong>{assigned.length}/{gpuCapacity}</strong></div>
          <div role="list" aria-label={`GPU slots on ${node.node}`} className={css`display:grid;grid-template-columns:repeat(${Math.min(gpuCapacity, 8)},minmax(0,1fr));gap:3px;`}>
            {Array.from({ length: Math.max(gpuCapacity, assigned.length) }, (_, index) => {
              const workload = assigned[index];
              return <div role="listitem" aria-label={workload ? `${workload.job} GPU` : 'Unallocated GPU'} title={workload?.task ?? 'Unallocated'} key={index} className={css`height:25px;border-radius:3px;border:1px solid ${workload ? jobColor(workload.job, jobs) : theme.colors.border.weak};background:${workload ? `${jobColor(workload.job, jobs)}33` : theme.colors.background.canvas};overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:4px;box-sizing:border-box;font-size:9px;`}>{workload ? shortJob(workload.job) : ''}</div>;
            })}
          </div>
          <div className={css`font-size:11px;color:${theme.colors.text.secondary};margin-top:4px;`}>GPU {formatPercent(metric('gpu_utilization_percent'))} live · HBM {formatPercent(gpuMemoryPercent)}</div>
        </div>
      )}
      <ResourceBar label="CPU" used={cpuRequested} capacity={cpuCapacity} actual={metric('node_cpu_utilization_percent')} />
      <ResourceBar label="Memory" used={memoryRequested} capacity={memoryCapacity} actual={memoryPercent} />
    </article>
  );
}

export function ClusterCapacity({ frames, width, height }: Props) {
  const theme = useTheme2();
  const workloads = useMemo(() => frameRows(frames, REF.workloads, workloadAllocations), [frames]);
  const nodes = useMemo(() => frameRows(frames, REF.nodes, clusterNodes), [frames]);
  const usage = useMemo(() => usageMap(frameRows(frames, REF.tasks, taskUsage)), [frames]);
  const metrics = useMemo(() => metricMap(frameRows(frames, REF.host, nodeMetrics)), [frames]);
  const summaries = useMemo(() => summarizeJobs(workloads, usage), [workloads, usage]);
  const jobs = summaries.map((summary) => summary.job);
  const cluster = nodes[0]?.cluster ?? workloads[0]?.cluster ?? 'selected cluster';
  const totalGpus = nodes.reduce((sum, node) => sum + (node.gpuAllocatable || node.gpuCapacity), 0);
  const allocatedGpus = workloads.filter((workload) => workload.node).reduce(
    (sum, workload) => sum + workload.gpuRequestCount,
    0
  );
  const bound = new Map<string, WorkloadAllocation[]>();
  for (const workload of workloads.filter((row) => row.node)) {
    const rows = bound.get(workload.node) ?? [];
    rows.push(workload);
    bound.set(workload.node, rows);
  }
  const unbound = workloads.filter((workload) => !workload.node);
  const orderedNodes = [...nodes].sort((a, b) => {
    const aUnavailable = Number(!a.ready || a.unschedulable);
    const bUnavailable = Number(!b.ready || b.unschedulable);
    const aGpus = (bound.get(a.node) ?? []).reduce((sum, workload) => sum + workload.gpuRequestCount, 0);
    const bGpus = (bound.get(b.node) ?? []).reduce((sum, workload) => sum + workload.gpuRequestCount, 0);
    return bUnavailable - aUnavailable || bGpus - aGpus || a.node.localeCompare(b.node);
  });
  return (
    <main aria-label="Cluster capacity" className={css`width:${width}px;height:${height}px;padding:14px 16px 24px;box-sizing:border-box;color:${theme.colors.text.primary};background:${theme.colors.background.canvas};overflow:auto;`}>
      <header className={css`display:flex;align-items:baseline;justify-content:space-between;gap:20px;margin-bottom:14px;`}>
        <div>
          <h1 className={css`font-size:24px;line-height:1.2;margin:0;`}>Cluster capacity · {cluster}</h1>
          <span className={css`font-size:12px;color:${theme.colors.text.secondary};`}>Live Kubernetes placement; observed CPU and memory from the latest node-agent samples</span>
        </div>
        <nav className={css`display:flex;gap:14px;font-size:13px;`}><a href="/d/marin-jobs">Jobs</a><a href="/d/marin-clusters">Fleet health</a><a href="/d/marin-nodes">Node details</a></nav>
      </header>
      <section aria-label="Cluster totals" className={css`display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px;`}>
        <Stat value={summaries.length} label="active jobs" />
        <Stat value={workloads.length} label="live tasks" />
        <Stat value={`${allocatedGpus}/${totalGpus}`} label="GPUs allocated" />
        <Stat value={nodes.length} label="nodes" />
        <Stat value={unbound.length} label="tasks waiting" />
        <Stat value={nodes.filter((node) => !node.ready || node.unschedulable).length} label="nodes unavailable" />
      </section>
      <section aria-label="Active jobs" className={css`margin-bottom:18px;`}>
        <h2 className={css`font-size:15px;margin:0 0 7px;`}>Live jobs and requests</h2>
        {summaries.length === 0 ? <p>No live Iris jobs reported.</p> : (
          <div className={css`border:1px solid ${theme.colors.border.weak};border-radius:7px;overflow:auto;`}>
            <table className={css`border-collapse:collapse;width:100%;font-size:12px;& th,& td{padding:7px 9px;border-bottom:1px solid ${theme.colors.border.weak};text-align:right;white-space:nowrap;}& th:first-child,& td:first-child{text-align:left;}`}>
              <thead><tr><th>Job</th><th>Priority</th><th>Iris</th><th>Ready tasks</th><th>Live samples</th><th>GPU</th><th>CPU requested</th><th>CPU live</th><th>Memory requested</th><th>Memory live</th></tr></thead>
              <tbody>{summaries.map((summary) => <tr key={summary.job}>
                <td><span className={css`display:inline-block;width:8px;height:8px;border-radius:2px;background:${jobColor(summary.job, jobs)};margin-right:7px;`} /><a href={`/d/marin-jobs?var-cluster=${encodeURIComponent(cluster)}&var-job=${encodeURIComponent(summary.job)}`}>{summary.job}</a></td>
                <td>{summary.priorityClass ? summary.priorityClass.replace(/^iris-/, '') : '—'}</td>
                <td><a href={`${IRIS_DASHBOARD_ORIGIN}/#/job/${encodeURIComponent(summary.job)}?cluster=${encodeURIComponent(cluster)}`} target="_blank" rel="noreferrer">Open</a></td>
                <td>{summary.ready}/{summary.tasks}</td><td>{summary.observed}/{summary.tasks}</td><td>{summary.gpuRequestCount}</td><td>{formatCores(summary.cpuRequestMillicores)} cores</td><td>{summary.observed ? `${formatCores(summary.cpuMillicores)} cores` : '—'}</td><td>{formatBytes(summary.memoryRequestBytes)}</td><td>{summary.observed ? formatBytes(summary.memoryBytes) : '—'}</td>
              </tr>)}</tbody>
            </table>
          </div>
        )}
      </section>
      <section aria-label="Node packing">
        <h2 className={css`font-size:15px;margin:0 0 7px;`}>Node packing</h2>
        {nodes.length === 0 ? <p>No Kubernetes node inventory reported.</p> : <div className={css`display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:9px;`}>
          {orderedNodes.map((node) => <NodeCard key={node.node} node={node} workloads={bound.get(node.node) ?? []} metrics={metrics} jobs={jobs} />)}
        </div>}
      </section>
      {unbound.length > 0 && <section aria-label="Unbound tasks" className={css`margin-top:18px;`}><h2 className={css`font-size:15px;margin:0 0 7px;`}>Waiting for placement</h2><ul>{unbound.map((workload) => <li key={workload.pod}><code>{workload.task}</code> · {workload.phase} for {formatAge(workload.ageSeconds)} · {workload.gpuRequestCount} GPU · {formatCores(workload.cpuRequestMillicores)} cores · {formatBytes(workload.memoryRequestBytes)}</li>)}</ul></section>}
    </main>
  );
}
