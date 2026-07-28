import React, { useMemo } from 'react';
import { css } from '@emotion/css';
import { DataFrame } from '@grafana/data';
import { useTheme2 } from '@grafana/ui';
import { frameWithField, wandbPoints } from '../data';
import { PanelMessage } from './PanelMessage';
import { SERIES_COLORS } from './palette';

interface Props { frames: DataFrame[]; width: number; height: number }

function compact(value: number): string {
  if (value >= 1e12) {return `${(value / 1e12).toFixed(1)}T`;}
  if (value >= 1e9) {return `${Math.round(value / 1e9)}B`;}
  if (value >= 1e6) {return `${Math.round(value / 1e6)}M`;}
  return String(Math.round(value));
}

export function WandbChart({ frames, width, height }: Props) {
  const theme = useTheme2();
  const frame = frameWithField(frames, 'tokens');
  const points = useMemo(() => (frame ? wandbPoints(frame) : []), [frame]);
  const { paths, xMin, xMax, yMin, yMax } = useMemo(() => {
    if (points.length === 0) {return { paths: [] as Array<[string, typeof points]>, xMin: 0, xMax: 1, yMin: 0, yMax: 1 };}
    const groups = new Map<string, typeof points>();
    for (const point of points) {groups.set(point.run, [...(groups.get(point.run) ?? []), point]);}
    for (const values of groups.values()) {values.sort((a, b) => a.tokens - b.tokens);}
    const xs = points.map((point) => point.tokens);
    const ys = [...points.map((point) => point.value)].sort((a, b) => a - b);
    const low = ys[Math.floor(ys.length * 0.02)] ?? 0;
    const high = ys[Math.min(ys.length - 1, Math.floor(ys.length * 0.98))] ?? 1;
    return { paths: [...groups.entries()], xMin: Math.min(...xs), xMax: Math.max(...xs), yMin: low, yMax: high === low ? low + 1 : high };
  }, [points]);
  if (points.length === 0) {return <PanelMessage width={width} height={height}>No W&B data</PanelMessage>;}
  const pad = { left: 45, right: 10, top: 28, bottom: 28 };
  const chartWidth = Math.max(1, width - pad.left - pad.right);
  const chartHeight = Math.max(1, height - pad.top - pad.bottom);
  const x = (value: number) => pad.left + ((value - xMin) / Math.max(1, xMax - xMin)) * chartWidth;
  const y = (value: number) => pad.top + (1 - (value - yMin) / Math.max(1e-12, yMax - yMin)) * chartHeight;
  return <section className={css`width:${width}px;height:${height}px;color:${theme.colors.text.primary};position:relative;overflow:hidden;`} aria-label={`${points[0].chart} W&B chart`}>
    <div className={css`position:absolute;top:2px;left:6px;right:6px;display:flex;justify-content:space-between;font-size:11px;z-index:1;`}><strong>{points[0].chart}</strong><a href={points[0].reportUrl} target="_blank" rel="noreferrer" className={css`color:${theme.colors.text.link};`}>W&B report ↗</a></div>
    <svg width={width} height={height} role="img" aria-label={`${points[0].chart} versus cumulative training tokens`}>
      {[0, .5, 1].map((fraction) => { const yy = pad.top + fraction * chartHeight; const value = yMax - fraction * (yMax - yMin); return <g key={fraction}><line x1={pad.left} x2={width-pad.right} y1={yy} y2={yy} stroke={theme.colors.border.weak} strokeDasharray="2 4"/><text x={pad.left-5} y={yy+3} textAnchor="end" fill={theme.colors.text.secondary} fontSize="9">{value.toFixed(2)}</text></g>; })}
      {[0, .5, 1].map((fraction) => { const xx = pad.left + fraction * chartWidth; const value = xMin + fraction * (xMax-xMin); return <text key={fraction} x={xx} y={height-7} textAnchor={fraction===0?'start':fraction===1?'end':'middle'} fill={theme.colors.text.secondary} fontSize="9">{compact(value)}</text>; })}
      {paths.map(([run, values], index) => <polyline key={run} fill="none" stroke={SERIES_COLORS[index % SERIES_COLORS.length]} strokeWidth="2" points={values.map((point) => `${x(point.tokens)},${y(Math.min(yMax, Math.max(yMin, point.value)))}`).join(' ')} />)}
    </svg>
    <div className={css`position:absolute;left:${pad.left}px;right:8px;bottom:15px;display:flex;gap:10px;overflow:hidden;font-size:9px;`}>
      {paths.map(([run, values], index) => <span key={run} title={run} className={css`display:flex;gap:3px;min-width:0;`}><span aria-hidden="true" className={css`color:${SERIES_COLORS[index % SERIES_COLORS.length]};`}>━</span><span className={css`overflow:hidden;text-overflow:ellipsis;white-space:nowrap;`}>{run}{values[0].runState === 'running' ? '' : ` (${values[0].runState})`}</span></span>)}
    </div>
  </section>;
}
