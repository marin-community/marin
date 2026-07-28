import React from 'react';
import { css, cx } from '@emotion/css';
import { DataFrame } from '@grafana/data';
import { useTheme2 } from '@grafana/ui';
import { frameWithField, nightlyCells } from '../data';
import { PanelMessage } from './PanelMessage';
import { NightlyCell } from '../types';

interface Props { frames: DataFrame[]; width: number; height: number }

const GROUP_NAMES: Record<string, string> = { marin: 'Marin', forks: 'Forks' };
const SUBGROUP_NAMES: Record<string, string> = { training: 'Training', data: 'Data', cluster: 'Cluster', evaluation: 'Evaluation', rl: 'RL', inference: 'Inference' };

function formatDuration(seconds?: number): string {
  if (seconds === undefined) {return '—';}
  if (seconds < 60) {return `${seconds}s`;}
  const minutes = Math.round(seconds / 60);
  return minutes < 60 ? `${minutes}m` : `${Math.floor(minutes / 60)}h${String(minutes % 60).padStart(2, '0')}`;
}

function status(cell: NightlyCell): { icon: string; tone: string; label: string } {
  if (cell.state === 'run') {
    if (!cell.healthy) {return { icon: '×', tone: '#f43f5e', label: cell.conclusion ?? 'failed' };}
    if (cell.durationState === 'very-slow') {return { icon: '✓', tone: '#b45309', label: 'very slow success' };}
    if (cell.durationState === 'slow') {return { icon: '✓', tone: '#92400e', label: 'slow success' };}
    return { icon: '✓', tone: '#064e3b', label: 'success' };
  }
  if (cell.state === 'missing') {return { icon: '!', tone: '#9f1239', label: 'missing' };}
  if (cell.state === 'unavailable') {return { icon: '!', tone: '#9a3412', label: 'source unavailable' };}
  if (cell.state === 'not-yet-due') {return { icon: '◷', tone: '#1e3a8a', label: 'not yet due' };}
  return { icon: '–', tone: 'transparent', label: cell.state.replaceAll('-', ' ') };
}

function spans<T>(items: T[], key: (item: T) => string): Array<{ key: string; span: number }> {
  const result: Array<{ key: string; span: number }> = [];
  for (const item of items) {
    const value = key(item);
    const prior = result.at(-1);
    if (prior?.key === value) {prior.span += 1;}
    else {result.push({ key: value, span: 1 });}
  }
  return result;
}

export function NightlyMatrix({ frames, width, height }: Props) {
  const theme = useTheme2();
  const frame = frameWithField(frames, 'lane_id');
  const cells = frame ? nightlyCells(frame) : [];
  if (cells.length === 0) {return <PanelMessage width={width} height={height}>No nightly data</PanelMessage>;}
  const laneById = new Map<string, NightlyCell>();
  for (const cell of cells) {laneById.set(cell.laneId, cell);}
  const lanes = [...laneById.values()].sort((a, b) => a.laneOrder - b.laneOrder);
  const dates = [...new Set(cells.map((cell) => cell.date))].sort().reverse();
  const byKey = new Map(cells.map((cell) => [`${cell.laneId}\u0000${cell.date}`, cell]));
  const today = dates[0];
  const todayCells = cells.filter((cell) => cell.date === today && cell.due);
  const groups = spans(lanes, (lane) => lane.group);
  const subgroups = spans(lanes, (lane) => `${lane.group}/${lane.subgroup}`);
  const border = theme.colors.border.weak;
  return (
    <section className={css`width:${width}px;height:${height}px;overflow:auto;color:${theme.colors.text.primary};padding:2px 4px;`} aria-label="Nightly regression status">
      <div className={css`display:flex;align-items:baseline;justify-content:space-between;margin:0 2px 5px;font-size:11px;color:${theme.colors.text.secondary};`}>
        <span><strong className={css`color:${theme.colors.text.primary};font-size:13px;`}>Nightly regressions</strong>{todayCells.length > 0 && ` · Today: ${todayCells.filter((cell) => cell.healthy).length}/${todayCells.length} healthy`}</span>
        <span>✓ healthy · amber slow · × failed · ! missing</span>
      </div>
      <table className={css`width:100%;min-width:980px;border-collapse:separate;border-spacing:2px;table-layout:fixed;font-size:10px;`}>
        <caption className={css`position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);`}>Seven UTC days of scheduled regression status and duration by lane</caption>
        <colgroup><col className={css`width:72px;`} />{lanes.map((lane) => <col key={lane.laneId} />)}</colgroup>
        <thead>
          <tr><th rowSpan={3} scope="col" className={css`text-align:left;color:${theme.colors.text.secondary};`}>UTC</th>{groups.map((group) => <th key={group.key} scope="colgroup" colSpan={group.span} className={css`text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid ${border};`}>{GROUP_NAMES[group.key] ?? group.key}</th>)}</tr>
          <tr>{subgroups.map((group) => { const subgroup = group.key.split('/')[1]; return <th key={group.key} scope="colgroup" colSpan={group.span} className={css`color:${theme.colors.text.secondary};border-bottom:1px solid ${border};`}>{SUBGROUP_NAMES[subgroup] ?? subgroup}</th>; })}</tr>
          <tr>{lanes.map((lane) => <th key={lane.laneId} scope="col" title={lane.label} className={css`height:24px;line-height:1.05;overflow-wrap:anywhere;`}>{lane.lane}</th>)}</tr>
        </thead>
        <tbody>{dates.map((date) => <tr key={date}>
          <th scope="row" className={css`text-align:left;white-space:nowrap;color:${theme.colors.text.secondary};`}>{new Date(`${date}T00:00:00Z`).toLocaleDateString('en', { weekday:'short', month:'short', day:'numeric', timeZone:'UTC' })}</th>
          {lanes.map((lane) => {
            const cell = byKey.get(`${lane.laneId}\u0000${date}`);
            if (!cell) {return <td key={lane.laneId} className={css`height:29px;background:${theme.colors.background.secondary};border-radius:3px;`} />;}
            const display = status(cell);
            const duration = formatDuration(cell.durationSeconds);
            const label = `${cell.label}, ${date}: ${display.label}${cell.durationSeconds === undefined ? '' : `, ${duration}`}`;
            const content = <><span aria-hidden="true" className={css`font-size:13px;line-height:1;`}>{display.icon}</span><span className={css`font-family:${theme.typography.fontFamilyMonospace};font-size:10px;`}>{duration}</span></>;
            const cellClass = cx(css`display:flex;align-items:center;justify-content:center;gap:4px;height:29px;border-radius:3px;background:${display.tone};color:${display.tone === 'transparent' ? theme.colors.text.secondary : '#f8fafc'};text-decoration:none;&:hover,&:focus-visible{outline:2px solid ${theme.colors.text.primary};outline-offset:1px;}`, cell.durationState === 'too-short' && css`box-shadow:inset 0 0 0 1px #fbbf24;background-image:repeating-linear-gradient(135deg,rgba(251,191,36,.22) 0,rgba(251,191,36,.22) 2px,transparent 2px,transparent 6px);`);
            return <td key={lane.laneId}>{cell.url || cell.workflowUrl ? <a href={cell.url ?? cell.workflowUrl} target="_blank" rel="noreferrer" aria-label={label} title={`${label}${cell.sourceError ? ` · ${cell.sourceError}` : ''}`} className={cellClass}>{content}</a> : <div aria-label={label} title={label} className={cellClass}>{content}</div>}</td>;
          })}
        </tr>)}</tbody>
      </table>
    </section>
  );
}
