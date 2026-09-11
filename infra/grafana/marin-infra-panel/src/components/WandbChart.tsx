import React, { useMemo, useState } from 'react';
import { css } from '@emotion/css';
import { DataFrame } from '@grafana/data';
import { useTheme2 } from '@grafana/ui';
import { frameWithField, wandbPoints } from '../data';
import { PanelMessage } from './PanelMessage';
import { SERIES_COLORS } from './palette';

interface Props { frames: DataFrame[]; width: number; height: number }

const MFU_TITLE = 'MFU (%)';
const EVAL_TITLE = 'Paloma macro loss (dropless)';
const INITIAL_TOKEN_FRACTION = 0.01;
const LOSS_PADDING = 0.1;
const MFU_SOFT_MAX = 30;
const TRAIN_SOFT_MIN = 1.2;
const TRAIN_SOFT_MAX = 1.6;

function compact(value: number): string {
  if (value >= 1e12) {return `${(value / 1e12).toFixed(1)}T`;}
  if (value >= 1e9) {return `${Math.round(value / 1e9)}B`;}
  if (value >= 1e6) {return `${Math.round(value / 1e6)}M`;}
  return String(Math.round(value));
}

export function WandbChart({ frames, width, height }: Props) {
  const theme = useTheme2();
  const [evalView, setEvalView] = useState('after-initialization');
  const frame = frameWithField(frames, 'tokens');
  const points = useMemo(() => (frame ? wandbPoints(frame) : []), [frame]);
  const isMfu = points[0]?.chart === MFU_TITLE;
  const isEval = points[0]?.chart === EVAL_TITLE;
  const { paths, xMin, xMax, yMin, yMax, ticks, decimals } = useMemo(() => {
    const latestTokens = Math.max(0, ...points.map((point) => point.tokens));
    const cutoff = isEval && evalView === 'after-initialization' ? latestTokens * INITIAL_TOKEN_FRACTION : 0;
    // Keep every run in the map so the view control does not change run colors.
    const groups = new Map<string, typeof points>();
    for (const point of points) {
      if (!groups.has(point.run)) {groups.set(point.run, []);}
      if (point.tokens >= cutoff) {groups.get(point.run)!.push(point);}
    }
    for (const values of groups.values()) {values.sort((a, b) => a.tokens - b.tokens);}
    const visible = [...groups.values()].flat();
    const xs = visible.map((point) => point.tokens);
    const ys = visible.map((point) => point.value);
    const low = ys.length ? Math.min(...ys) : 0;
    const high = ys.length ? Math.max(...ys) : 1;
    let yMin: number;
    let yMax: number;
    let ticks: number[];
    let decimals: number;
    if (isEval) {
      const padding = (high - low || Math.max(Math.abs(low), 1)) * LOSS_PADDING;
      yMin = low - padding;
      yMax = high + padding;
      ticks = [yMin, (yMin + yMax) / 2, yMax];
      decimals = Math.max(2, -Math.floor(Math.log10(yMax - yMin)) + 1);
    } else {
      const minimum = isMfu ? 0 : Math.min(TRAIN_SOFT_MIN, low);
      const maximum = Math.max(isMfu ? MFU_SOFT_MAX : TRAIN_SOFT_MAX, high);
      const targetStep = (maximum - minimum) / 4;
      const magnitude = 10 ** Math.floor(Math.log10(targetStep));
      const step = [1, 2, 5, 10].find((factor) => factor * magnitude >= targetStep - 1e-12)! * magnitude;
      yMin = Math.floor(minimum / step + 1e-12) * step;
      yMax = Math.ceil(maximum / step - 1e-12) * step;
      ticks = Array.from({ length: Math.round((yMax - yMin) / step) + 1 }, (_, index) => yMin + index * step);
      decimals = isMfu ? 0 : Math.max(1, -Math.floor(Math.log10(step)));
    }
    return { paths: [...groups.entries()], xMin: xs.length ? Math.min(...xs) : 0, xMax: latestTokens, yMin, yMax, ticks, decimals };
  }, [points, isMfu, isEval, evalView]);
  if (points.length === 0) {return <PanelMessage width={width} height={height}>No W&B data</PanelMessage>;}
  const pad = { left: 60, right: 10, top: width < 400 ? 76 : 56, bottom: 20 };
  const chartWidth = Math.max(1, width - pad.left - pad.right);
  const chartHeight = Math.max(1, height - pad.top - pad.bottom);
  const x = (value: number) => pad.left + ((value - xMin) / Math.max(1, xMax - xMin)) * chartWidth;
  const y = (value: number) => pad.top + (1 - (value - yMin) / (yMax - yMin)) * chartHeight;
  return <section className={css`width:${width}px;height:${height}px;color:${theme.colors.text.primary};position:relative;overflow:hidden;`} aria-label={`${points[0].chart} W&B chart`}>
    <div className={css`position:absolute;top:2px;left:6px;right:6px;font-size:12px;z-index:1;`}>
      <div className={css`display:flex;justify-content:space-between;gap:8px;`}>
        <strong>{points[0].chart}</strong>
        <a href={points[0].reportUrl} target="_blank" rel="noreferrer" className={css`color:${theme.colors.text.link};white-space:nowrap;`}>W&B report ↗</a>
      </div>
      <div className={css`display:flex;align-items:center;justify-content:space-between;gap:8px;margin-top:4px;color:${theme.colors.text.secondary};`}>
        <span>{isMfu ? 'Model FLOP utilization' : 'nats/token'}</span>
        {isEval && <select aria-label="Evaluation range" value={evalView} onChange={(event) => setEvalView(event.target.value)} title="After initialization excludes the first 1% of the latest cumulative token count." className={css`color:${theme.colors.text.primary};background:${theme.colors.background.secondary};border:1px solid ${theme.colors.border.weak};border-radius:3px;font:inherit;`}>
          <option value="after-initialization">After initialization</option>
          <option value="full-run">Full run</option>
        </select>}
      </div>
    </div>
    <svg width={width} height={height} role="img" aria-label={`${points[0].chart} versus cumulative training tokens`}>
      {ticks.map((value) => { const yy = y(value); return <g key={value}><line x1={pad.left} x2={width-pad.right} y1={yy} y2={yy} stroke={theme.colors.border.weak} strokeDasharray="2 4"/><text x={pad.left-5} y={yy+3} textAnchor="end" fill={theme.colors.text.secondary} fontSize="11">{value.toFixed(decimals)}{isMfu ? '%' : ''}</text></g>; })}
      {[0, .5, 1].map((fraction) => { const xx = pad.left + fraction * chartWidth; const value = xMin + fraction * (xMax-xMin); return <text key={fraction} x={xx} y={height-7} textAnchor={fraction===0?'start':fraction===1?'end':'middle'} fill={theme.colors.text.secondary} fontSize="11">{compact(value)}</text>; })}
      {paths.map(([run, values], index) => values.length === 1
        ? <circle key={run} cx={x(values[0].tokens)} cy={y(values[0].value)} r="2" fill={SERIES_COLORS[index % SERIES_COLORS.length]} />
        : <polyline key={run} fill="none" stroke={SERIES_COLORS[index % SERIES_COLORS.length]} strokeWidth="2" points={values.map((point) => `${x(point.tokens)},${y(point.value)}`).join(' ')} />)}
    </svg>
  </section>;
}
