import { DataFrame, Field } from '@grafana/data';
import { CommitRow, NightlyCell, WandbPoint } from './types';

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
    runState: requiredString(row, 'run_state'),
    tokens: requiredNumber(row, 'tokens'),
    value: requiredNumber(row, 'value'),
    reportTitle: requiredString(row, 'report_title'),
    reportUrl: requiredString(row, 'report_url'),
  }));
}

export function frameWithField(frames: DataFrame[], fieldName: string): DataFrame {
  const matching = frames.filter((frame) => frame.fields.some((field: Field) => field.name === fieldName));
  if (matching.length !== 1) {
    throw new Error(`Expected one data frame containing ${fieldName}; received ${matching.length}`);
  }
  return matching[0];
}
