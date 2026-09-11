import { DataFrame, Field } from '@grafana/data';
import {
  CommitRow,
  ClusterNode,
  NightlyCell,
  NodeMetric,
  SmUtilizationPoint,
  SmUtilizationRasterData,
  TaskUsage,
  WandbPoint,
  WorkloadAllocation,
  WorkerRegion,
} from './types';

type Row = Record<string, unknown>;

function rows(frame: DataFrame): Row[] {
  return Array.from({ length: frame.length }, (_, index) =>
    Object.fromEntries(frame.fields.map((field) => [field.name, field.values[index]]))
  );
}

function requiredString(row: Row, key: string): string {
  const value = row[key];
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`Missing required string field: ${key}`);
  }
  return value;
}

function optionalString(row: Row, key: string): string | undefined {
  const value = row[key];
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

function requiredNumber(row: Row, key: string): number {
  const value = row[key];
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`Missing required number field: ${key}`);
  }
  return value;
}

function optionalNumber(row: Row, key: string): number | undefined {
  const value = row[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function booleanValue(row: Row, key: string): boolean {
  const value = row[key];
  return value === true || value === 1 || value === 'true';
}

export function nightlyCells(frame: DataFrame): NightlyCell[] {
  const seen = new Set<string>();
  return rows(frame).map((row, index) => {
    const date = requiredString(row, 'date');
    const laneId = requiredString(row, 'lane_id');
    const key = `${laneId}\u0000${date}`;
    if (seen.has(key)) {
      throw new Error(`Duplicate nightly cell: ${laneId} on ${date}`);
    }
    seen.add(key);
    return {
      date,
      laneId,
      lane: requiredString(row, 'lane'),
      label: requiredString(row, 'label'),
      group: requiredString(row, 'group'),
      subgroup: requiredString(row, 'subgroup'),
      state: requiredString(row, 'state'),
      runStatus: optionalString(row, 'status'),
      durationState: requiredString(row, 'duration_state'),
      durationSeconds: optionalNumber(row, 'duration_seconds'),
      conclusion: optionalString(row, 'conclusion'),
      url: optionalString(row, 'url'),
      workflowUrl: optionalString(row, 'workflow_url'),
      healthy: booleanValue(row, 'healthy'),
      due: booleanValue(row, 'due'),
      sourceError: optionalString(row, 'source_error'),
      laneOrder: optionalNumber(row, 'lane_order') ?? index,
    };
  });
}

export function commits(frame: DataFrame): CommitRow[] {
  return rows(frame).map((row) => ({
    oid: requiredString(row, 'oid'),
    shortOid: requiredString(row, 'short_oid'),
    headline: requiredString(row, 'headline'),
    author: requiredString(row, 'author'),
    avatarUrl: optionalString(row, 'avatar_url'),
    state: requiredString(row, 'state'),
    committedAt: requiredNumber(row, 'committed_at'),
    url: requiredString(row, 'url'),
    successRate: optionalNumber(row, 'success_rate'),
  }));
}

export function wandbPoints(frame: DataFrame): WandbPoint[] {
  return rows(frame).map((row) => ({
    chart: requiredString(row, 'chart'),
    run: requiredString(row, 'run'),
    tokens: requiredNumber(row, 'tokens'),
    value: requiredNumber(row, 'value'),
    reportTitle: requiredString(row, 'report_title'),
    reportUrl: requiredString(row, 'report_url'),
  }));
}

export function workerRegions(frame: DataFrame): WorkerRegion[] {
  return rows(frame).map((row) => ({
    region: requiredString(row, 'region'),
    healthy: requiredNumber(row, 'healthy'),
    cpuMillicores: requiredNumber(row, 'cpu_millicores'),
    memoryBytes: requiredNumber(row, 'memory_bytes'),
    tpuChips: requiredNumber(row, 'tpu_chips'),
  }));
}

export function frameByRefId(frames: DataFrame[], refId: string): DataFrame | undefined {
  const matching = frames.filter((frame) => frame.refId === refId);
  if (matching.length > 1) {
    throw new Error(`Expected one data frame for ${refId}; received ${matching.length}`);
  }
  return matching[0];
}

export function workloadAllocations(frame: DataFrame): WorkloadAllocation[] {
  return rows(frame).filter((row) => optionalString(row, 'task')).map((row) => ({
    cluster: requiredString(row, 'cluster'),
    namespace: requiredString(row, 'namespace'),
    pod: requiredString(row, 'pod'),
    node: optionalString(row, 'node') ?? '',
    job: requiredString(row, 'job'),
    task: requiredString(row, 'task'),
    phase: requiredString(row, 'phase'),
    ready: booleanValue(row, 'ready'),
    priorityClass: optionalString(row, 'priority_class') ?? '',
    ageSeconds: requiredNumber(row, 'age_seconds'),
    cpuRequestMillicores: requiredNumber(row, 'cpu_request_millicores'),
    memoryRequestBytes: requiredNumber(row, 'memory_request_bytes'),
    gpuRequestCount: requiredNumber(row, 'gpu_request_count'),
    gpuVariant: optionalString(row, 'gpu_variant') ?? '',
  }));
}

export function jobIds(frame: DataFrame): string[] {
  return rows(frame).map((row) => requiredString(row, 'job'));
}

export function clusterNodes(frame: DataFrame): ClusterNode[] {
  return rows(frame).filter((row) => optionalString(row, 'node')).map((row) => ({
    cluster: requiredString(row, 'cluster'),
    node: requiredString(row, 'node'),
    instanceType: optionalString(row, 'instance_type') ?? '',
    nodePool: optionalString(row, 'node_pool') ?? '',
    gpuModel: optionalString(row, 'gpu_model') ?? '',
    gpuCapacity: requiredNumber(row, 'gpu_capacity'),
    gpuAllocatable: requiredNumber(row, 'gpu_allocatable'),
    cpuAllocatable: optionalString(row, 'cpu_allocatable') ?? '',
    memoryAllocatable: optionalString(row, 'memory_allocatable') ?? '',
    ready: booleanValue(row, 'ready'),
    unschedulable: booleanValue(row, 'unschedulable'),
  }));
}

export function taskUsage(frame: DataFrame): TaskUsage[] {
  return rows(frame).filter((row) => optionalString(row, 'task')).map((row) => ({
    cluster: requiredString(row, 'cluster'),
    task: requiredString(row, 'task'),
    pod: requiredString(row, 'pod'),
    cpuMillicores: requiredNumber(row, 'cpu_millicores'),
    memoryBytes: requiredNumber(row, 'memory_bytes'),
    sampledAt: requiredNumber(row, 'sampled_at'),
  }));
}

export function nodeMetrics(frame: DataFrame): NodeMetric[] {
  return rows(frame).filter((row) => optionalString(row, 'node')).map((row) => ({
    cluster: requiredString(row, 'cluster'),
    node: requiredString(row, 'node'),
    name: requiredString(row, 'name'),
    value: requiredNumber(row, 'value'),
    sampledAt: requiredNumber(row, 'sampled_at'),
  }));
}

export function smUtilizationPoints(frame: DataFrame): SmUtilizationPoint[] {
  return rows(frame).map((row) => ({
    cluster: requiredString(row, 'cluster'),
    node: requiredString(row, 'node'),
    gpu: requiredString(row, 'gpu'),
    sampledAt: requiredNumber(row, 'time'),
    percent: requiredNumber(row, 'sm_utilization'),
  }));
}

export function smUtilizationRasterData(points: SmUtilizationPoint[]): SmUtilizationRasterData {
  const devicesByKey = new Map<string, SmUtilizationPoint>();
  const timestampSet = new Set<number>();
  for (const point of points) {
    devicesByKey.set(`${point.cluster}\u0000${point.node}\u0000${point.gpu}`, point);
    timestampSet.add(point.sampledAt);
  }

  const devices = [...devicesByKey.values()]
    .map(({ cluster, node, gpu }) => ({ cluster, node, gpu }))
    .sort((left, right) =>
      left.cluster.localeCompare(right.cluster, undefined, { numeric: true })
      || left.node.localeCompare(right.node, undefined, { numeric: true })
      || left.gpu.localeCompare(right.gpu, undefined, { numeric: true })
    );
  const timestamps = [...timestampSet].sort((left, right) => left - right);
  const deviceIndexes = new Map(
    devices.map((device, index) => [`${device.cluster}\u0000${device.node}\u0000${device.gpu}`, index])
  );
  const timestampIndexes = new Map(timestamps.map((timestamp, index) => [timestamp, index]));
  const values = new Float32Array(devices.length * timestamps.length);
  values.fill(Number.NaN);
  for (const point of points) {
    const deviceIndex = deviceIndexes.get(`${point.cluster}\u0000${point.node}\u0000${point.gpu}`);
    const timestampIndex = timestampIndexes.get(point.sampledAt);
    if (deviceIndex !== undefined && timestampIndex !== undefined) {
      values[deviceIndex * timestamps.length + timestampIndex] = point.percent;
    }
  }
  return { devices, timestamps, values };
}

/**
 * The single query frame carrying `fieldName`, or `undefined` when the query
 * returned nothing yet (empty, still loading, or upstream error). Callers render
 * an empty state for `undefined`; only a genuinely ambiguous result — more than
 * one frame with the field — is a misconfiguration worth failing loudly on.
 */
export function frameWithField(frames: DataFrame[], fieldName: string): DataFrame | undefined {
  const matching = frames.filter((frame) => frame.fields.some((field: Field) => field.name === fieldName));
  if (matching.length > 1) {
    throw new Error(`Expected one data frame containing ${fieldName}; received ${matching.length}`);
  }
  return matching[0];
}
