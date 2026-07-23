export type InfraPanelView = 'nightlies' | 'commits' | 'wandb';

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
  runState: string;
  tokens: number;
  value: number;
  reportTitle: string;
  reportUrl: string;
}
