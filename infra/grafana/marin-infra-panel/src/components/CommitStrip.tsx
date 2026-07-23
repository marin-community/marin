import React from 'react';
import { css } from '@emotion/css';
import { DataFrame } from '@grafana/data';
import { useTheme2 } from '@grafana/ui';
import { commits, frameWithField } from '../data';
import { STATUS_COLORS } from './palette';

interface Props { frames: DataFrame[]; width: number; height: number }

function relativeTime(epoch: number): string {
  const minutes = Math.max(0, Math.round((Date.now() - epoch) / 60_000));
  if (minutes < 60) {return `${minutes}m ago`;}
  const hours = Math.round(minutes / 60);
  return hours < 48 ? `${hours}h ago` : `${Math.round(hours / 24)}d ago`;
}

export function CommitStrip({ frames, width, height }: Props) {
  const theme = useTheme2();
  const values = commits(frameWithField(frames, 'short_oid')).sort((a, b) => b.committedAt - a.committedAt);
  if (values.length === 0) {throw new Error('No commits returned');}
  const latest = values[0];
  const finalized = values.filter((row) => ['SUCCESS', 'FAILURE', 'ERROR'].includes(row.state)).length;
  const rate = latest.successRate === undefined ? '—' : `${Math.round(latest.successRate * 100)}%`;
  return (
    <section className={css`width:${width}px;height:${height}px;overflow:hidden;color:${theme.colors.text.primary};padding:4px 8px;`} aria-label="Main branch build history">
      <div className={css`display:flex;align-items:center;gap:8px;min-width:0;font-size:12px;`}>
        <a href={latest.url} target="_blank" rel="noreferrer" className={css`display:flex;align-items:center;gap:7px;min-width:0;flex:1;color:inherit;text-decoration:none;&:hover{text-decoration:underline;}`}>
          {latest.avatarUrl && <img src={latest.avatarUrl} alt="" className={css`width:22px;height:22px;border-radius:50%;`} />}
          <span aria-hidden="true" className={css`color:${STATUS_COLORS[latest.state] ?? STATUS_COLORS.NONE};font-size:16px;`}>{latest.state === 'SUCCESS' ? '●' : latest.state === 'FAILURE' || latest.state === 'ERROR' ? '×' : '◷'}</span>
          <code>{latest.shortOid}</code>
          <span className={css`overflow:hidden;text-overflow:ellipsis;white-space:nowrap;`}>{latest.headline}</span>
          <span className={css`color:${theme.colors.text.secondary};white-space:nowrap;`}>· {relativeTime(latest.committedAt)}</span>
        </a>
        <span className={css`color:${theme.colors.text.secondary};white-space:nowrap;`}>{rate} success over {finalized}</span>
      </div>
      <div className={css`display:flex;gap:2px;margin-top:8px;height:${Math.max(12, height - 44)}px;`}>
        {[...values].reverse().map((row) => (
          <a key={row.oid} href={row.url} target="_blank" rel="noreferrer" aria-label={`${row.shortOid}: ${row.state.toLowerCase()}, ${row.headline}, ${relativeTime(row.committedAt)}`} title={`${row.shortOid} · ${row.state.toLowerCase()} · ${row.headline}`} className={css`min-width:2px;flex:1;border-radius:2px;background:${STATUS_COLORS[row.state] ?? STATUS_COLORS.NONE};&:hover,&:focus-visible{outline:2px solid ${theme.colors.text.primary};outline-offset:1px;}`} />
        ))}
      </div>
    </section>
  );
}
