export type InfraPanelView = 'status' | 'cluster' | 'nightlies' | 'commits' | 'wandb';

export interface InfraPanelOptions {
  view: InfraPanelView;
}

export interface NightlyCell {
  date: string;
  laneId: string;
  lane: string;
  label: string;
  group: string;
  subgroup: string;
  state: string;
  durationState: string;
  durationSeconds?: number;
  conclusion?: string;
  url?: string;
  workflowUrl?: string;
  healthy: boolean;
  due: boolean;
  sourceError?: string;
  laneOrder: number;
}

export interface CommitRow {
  oid: string;
  shortOid: string;
  headline: string;
  author: string;
  avatarUrl?: string;
  state: string;
  committedAt: number;
  url: string;
  successRate?: number;
}

export interface WandbPoint {
  chart: string;
  run: string;
  tokens: number;
  value: number;
  reportTitle: string;
  reportUrl: string;
}

export interface WorkerRegion {
  region: string;
  healthy: number;
  cpuMillicores: number;
  memoryBytes: number;
  tpuChips: number;
}

export interface WorkloadAllocation {
  cluster: string;
  namespace: string;
  pod: string;
  node: string;
  job: string;
  task: string;
  phase: string;
  ready: boolean;
  priorityClass: string;
  ageSeconds: number;
  cpuRequestMillicores: number;
  memoryRequestBytes: number;
  gpuRequestCount: number;
  gpuVariant: string;
}

export interface ClusterNode {
  cluster: string;
  node: string;
  instanceType: string;
  nodePool: string;
  gpuModel: string;
  gpuCapacity: number;
  gpuAllocatable: number;
  cpuAllocatable: string;
  memoryAllocatable: string;
  ready: boolean;
  unschedulable: boolean;
}

export interface TaskUsage {
  cluster: string;
  task: string;
  pod: string;
  cpuMillicores: number;
  memoryBytes: number;
  sampledAt: number;
}

export interface NodeMetric {
  cluster: string;
  node: string;
  name: string;
  value: number;
  sampledAt: number;
}
