export type InfraPanelView = 'status' | 'nightlies' | 'commits' | 'wandb';

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

export interface ProvisioningRow {
  scope: string;
  collectedAt: number;
  zone: string;
  ready: number;
  stockout: number;
  error: number;
  preempted: number;
  outcomes: number;
  successRatio?: number;
  poolsPlacing: number;
  poolsNoReadyOutcome: number;
  latencyP50Seconds?: number;
  latencyP95Seconds?: number;
  windowHours?: number;
}

export interface ProvisioningRegion {
  region: string;
  ready: number;
  outcomes: number;
  successRatio: number;
}

export interface SeriesPoint {
  time: number;
  series: string;
  value: number;
}
