import React, { useMemo } from 'react';
import { css } from '@emotion/css';
import { DataFrame } from '@grafana/data';
import { useTheme2 } from '@grafana/ui';
import { frameByRefId, provisioningRegions, provisioningStatus, seriesPoints, workerRegions } from '../data';
import { SeriesPoint } from '../types';
import { CommitStrip } from './CommitStrip';
import { NightlyMatrix } from './NightlyMatrix';
import { SERIES_COLORS } from './palette';
import { PanelMessage } from './PanelMessage';
import { WandbChart } from './WandbChart';

interface Props {
  frames: DataFrame[];
  width: number;
  height: number;
}

const REF = {
  nightlies: 'N',
  builds: 'G',
  workers: 'W',
  provisioning: 'P',
  workerHistory: 'H',
  provisioningHistory: 'R',
  trainLoss: 'T',
  paloma: 'L',
  mfu: 'M',
} as const;

const STATUS_GRID_CLASS = css`display:grid;grid-template-columns:minmax(220px,.8fr) minmax(360px,1.8fr);gap:22px;@media(max-width:900px){grid-template-columns:1fr;}`;
const STATUS_CARD_CLASS = css`flex:1;`;
const STATUS_SECTION_CLASS = css`display:flex;flex-direction:column;`;
const FLEET_SCOPE = 'fleet';
const PERCENT_SCALE_LIMIT = 0.001;
const PERCENT_TICKS = [1, 0.99, 0.95, 0.5, 0.05, 0.01, 0];

function seriesColor(name: string): string {
  let value = 0;
  for (const character of name) {
    value = (value * 31 + character.charCodeAt(0)) >>> 0;
  }
  return SERIES_COLORS[value % SERIES_COLORS.length];
}

function logitPercent(value: number): number {
  const bounded = Math.min(1 - PERCENT_SCALE_LIMIT, Math.max(PERCENT_SCALE_LIMIT, value));
  return Math.log(bounded / (1 - bounded));
}

function one(frames: DataFrame[], refId: string): DataFrame[] {
  const frame = frameByRefId(frames, refId);
  return frame ? [frame] : [];
}

function formatCores(millicores: number): string {
  const cores = millicores / 1000;
  return cores >= 1000 ? `${(cores / 1000).toFixed(1)}k` : Math.round(cores).toString();
}

function formatBytes(bytes: number): string {
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 100 ? Math.round(value) : value.toFixed(1)} ${units[unit]}`;
}

function formatPercent(value?: number): string {
  return value === undefined ? '—' : `${Math.round(value * 100)}%`;
}

function formatLatency(value?: number): string {
  if (value === undefined) {
    return '—';
  }
  return value < 90 ? `${Math.round(value)}s` : `${(value / 60).toFixed(1)}m`;
}

function relativeTime(epoch: number): string {
  const minutes = Math.max(0, Math.round((Date.now() - epoch) / 60_000));
  if (minutes < 60) {
    return `${minutes}m ago`;
  }
  const hours = Math.round(minutes / 60);
  return hours < 48 ? `${hours}h ago` : `${Math.round(hours / 24)}d ago`;
}

function MiniSeriesChart({ points, unit }: { points: SeriesPoint[]; unit: 'count' | 'percent' }) {
  const theme = useTheme2();
  const series = useMemo(() => {
    const grouped = new Map<string, SeriesPoint[]>();
    for (const point of points) {
      grouped.set(point.series, [...(grouped.get(point.series) ?? []), point]);
    }
    for (const values of grouped.values()) {
      values.sort((a, b) => a.time - b.time);
    }
    return [...grouped.entries()];
  }, [points]);

  if (points.length < 2) {
    return <div className={css`height:160px;display:flex;align-items:center;justify-content:center;color:${theme.colors.text.secondary};font-size:14px;`}>History is not available</div>;
  }

  const width = 800;
  const height = 170;
  const pad = { left: 42, right: 12, top: 10, bottom: 28 };
  const times = points.map((point) => point.time);
  const values = points.map((point) => point.value);
  const xMin = Math.min(...times);
  const xMax = Math.max(...times);
  const rawYMin = unit === 'percent' ? 0 : Math.min(0, ...values);
  const rawYMax = unit === 'percent' ? 1 : Math.max(1, ...values);
  const scaleY = unit === 'percent' ? logitPercent : (value: number) => value;
  const yMin = scaleY(rawYMin);
  const yMax = scaleY(rawYMax);
  const x = (value: number) => pad.left + ((value - xMin) / Math.max(1, xMax - xMin)) * (width - pad.left - pad.right);
  const y = (value: number) => pad.top + (1 - (scaleY(value) - yMin) / Math.max(1e-9, yMax - yMin)) * (height - pad.top - pad.bottom);
  const tickValues = unit === 'percent'
    ? PERCENT_TICKS
    : [rawYMax, (rawYMax + rawYMin) / 2, rawYMin];

  return (
    <div>
      <svg viewBox={`0 0 ${width} ${height}`} width="100%" height="170" role="img" aria-label="24 hour status history">
        {tickValues.map((value) => {
          const rowY = y(value);
          return (
            <g key={value}>
              <line x1={pad.left} x2={width - pad.right} y1={rowY} y2={rowY} stroke={theme.colors.border.weak} strokeDasharray="2 5" />
              <text x={pad.left - 6} y={rowY + 4} textAnchor="end" fill={theme.colors.text.secondary} fontSize="12">
                {unit === 'percent' ? `${Math.round(value * 100)}%` : Math.round(value)}
              </text>
            </g>
          );
        })}
        {series.map(([name, samples]) => (
          <polyline
            key={name}
            aria-label={`${name} history`}
            fill="none"
            stroke={seriesColor(name)}
            strokeWidth="2"
            points={samples.map((point) => `${x(point.time)},${y(point.value)}`).join(' ')}
          />
        ))}
        <text x={pad.left} y={height - 7} fill={theme.colors.text.secondary} fontSize="12">
          {new Date(xMin).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}
        </text>
        <text x={width - pad.right} y={height - 7} textAnchor="end" fill={theme.colors.text.secondary} fontSize="12">
          {new Date(xMax).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}
        </text>
      </svg>
      <div role="list" aria-label="Status history series" className={css`display:flex;flex-wrap:wrap;gap:5px 14px;font-size:12px;color:${theme.colors.text.secondary};`}>
        {series.map(([name]) => (
          <span role="listitem" key={name}><span className={css`color:${seriesColor(name)};`}>━</span> {name}</span>
        ))}
      </div>
      {unit === 'percent' && (
        <div className={css`margin-top:4px;font-size:12px;color:${theme.colors.text.secondary};`}>
          The logit scale expands values near 0% and 100%.
        </div>
      )}
    </div>
  );
}

function SectionTitle({ children, detail }: { children: React.ReactNode; detail?: string }) {
  const theme = useTheme2();
  return (
    <div className={css`display:flex;align-items:baseline;gap:10px;margin:0 0 10px;`}>
      <h2 className={css`font-size:19px;line-height:1.2;margin:0;font-weight:600;color:${theme.colors.text.primary};`}>{children}</h2>
      {detail && <span className={css`font-size:13px;color:${theme.colors.text.secondary};`}>{detail}</span>}
    </div>
  );
}

function Card({ children, className }: { children: React.ReactNode; className?: string }) {
  const theme = useTheme2();
  return <div className={`${css`border:1px solid ${theme.colors.border.weak};border-radius:7px;background:${theme.colors.background.secondary};padding:14px;box-shadow:0 12px 30px rgba(0,0,0,.10);`} ${className ?? ''}`}>{children}</div>;
}

function RegionList({ title, rows }: { title: string; rows: Array<{ region: string; value: React.ReactNode }> }) {
  const theme = useTheme2();
  return (
    <div>
      <div className={css`font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:${theme.colors.text.secondary};margin-bottom:5px;`}>{title}</div>
      {rows.map((row) => (
        <div key={row.region} className={css`display:grid;grid-template-columns:1fr auto;gap:12px;padding:6px 0;border-top:1px solid ${theme.colors.border.weak};font-size:14px;`}>
          <code>{row.region}</code>
          {row.value}
        </div>
      ))}
    </div>
  );
}

function WorkerStatus({ frames }: { frames: DataFrame[] }) {
  const theme = useTheme2();
  const currentFrame = frameByRefId(frames, REF.workers);
  const historyFrame = frameByRefId(frames, REF.workerHistory);
  const regions = currentFrame ? workerRegions(currentFrame).sort((a, b) => b.healthy - a.healthy) : [];
  const history = historyFrame ? seriesPoints(historyFrame, 'region', 'workers') : [];
  const totals = regions.reduce(
    (value, region) => ({
      healthy: value.healthy + region.healthy,
      cpu: value.cpu + region.cpuMillicores,
      memory: value.memory + region.memoryBytes,
      chips: value.chips + region.tpuChips,
    }),
    { healthy: 0, cpu: 0, memory: 0, chips: 0 }
  );

  return (
    <section className={STATUS_SECTION_CLASS} aria-label="Worker status">
      <SectionTitle detail="current capacity and 24 hour history">Workers</SectionTitle>
      <Card className={STATUS_CARD_CLASS}>
        {regions.length === 0 ? (
          <PanelMessage width={400} height={220}>No worker data</PanelMessage>
        ) : (
          <>
            <div className={css`display:flex;align-items:baseline;flex-wrap:wrap;gap:6px 14px;margin-bottom:14px;`}>
              <strong className={css`font-size:36px;color:${theme.colors.success.text};`}>{totals.healthy}</strong>
              <span className={css`color:${theme.colors.text.secondary};font-size:16px;`}>healthy workers</span>
              <span className={css`color:${theme.colors.text.disabled};`}>·</span>
              <span><strong>{formatCores(totals.cpu)}</strong> CPU</span>
              <span><strong>{formatBytes(totals.memory)}</strong> memory</span>
              <span><strong>{totals.chips}</strong> TPU chips</span>
            </div>
            <div className={STATUS_GRID_CLASS}>
              <RegionList
                title="Availability by region"
                rows={regions.map((region) => ({ region: region.region, value: <strong>{region.healthy}</strong> }))}
              />
              <MiniSeriesChart points={history} unit="count" />
            </div>
          </>
        )}
      </Card>
    </section>
  );
}

function Outcome({ label, value, color }: { label: string; value: number; color: string }) {
  return <span><strong className={css`font-family:monospace;color:${color};`}>{value}</strong> <span>{label}</span></span>;
}

function ProvisioningStatus({ frames }: { frames: DataFrame[] }) {
  const theme = useTheme2();
  const currentFrame = frameByRefId(frames, REF.provisioning);
  const historyFrame = frameByRefId(frames, REF.provisioningHistory);
  const rows = currentFrame ? provisioningStatus(currentFrame) : [];
  const fleet = rows.find((row) => row.scope === FLEET_SCOPE);
  const regions = provisioningRegions(rows).sort(
    (a, b) => b.outcomes - a.outcomes || a.region.localeCompare(b.region)
  );
  const history = historyFrame
    ? seriesPoints(historyFrame, 'region', 'success_ratio').filter((point) => point.series !== FLEET_SCOPE)
    : [];

  return (
    <section className={STATUS_SECTION_CLASS} aria-label="Provisioning status">
      <SectionTitle detail={fleet?.windowHours === undefined ? undefined : `trailing ${fleet.windowHours} hour window`}>Provisioning</SectionTitle>
      <Card className={STATUS_CARD_CLASS}>
        {!fleet ? (
          <PanelMessage width={400} height={220}>No provisioning data</PanelMessage>
        ) : (
          <>
            <div className={css`display:flex;align-items:center;flex-wrap:wrap;gap:10px 26px;margin-bottom:14px;`}>
              <div>
                <strong className={css`display:block;font-size:34px;color:${fleet.successRatio !== undefined && fleet.successRatio >= .9 ? theme.colors.success.text : theme.colors.warning.text};`}>{formatPercent(fleet.successRatio)}</strong>
                <span className={css`font-size:13px;color:${theme.colors.text.secondary};`}>create success · {fleet.ready}/{fleet.outcomes} attempts</span>
              </div>
              <div className={css`display:flex;flex-wrap:wrap;gap:8px 16px;color:${theme.colors.text.secondary};font-size:13px;`}>
                <Outcome label="ready" value={fleet.ready} color={theme.colors.success.text} />
                <Outcome label="stockout" value={fleet.stockout} color={theme.colors.warning.text} />
                <Outcome label="error" value={fleet.error} color={theme.colors.error.text} />
                <Outcome label="preempted" value={fleet.preempted} color={theme.colors.text.primary} />
                <Outcome label="pools placing" value={fleet.poolsPlacing} color={theme.colors.success.text} />
                <Outcome label="pools without ready outcome" value={fleet.poolsNoReadyOutcome} color={fleet.poolsNoReadyOutcome > 0 ? theme.colors.error.text : theme.colors.text.primary} />
              </div>
              <div className={css`margin-left:auto;font-size:13px;color:${theme.colors.text.secondary};`}>
                latency p50 {formatLatency(fleet.latencyP50Seconds)} · p95 {formatLatency(fleet.latencyP95Seconds)}
                <br />collected {relativeTime(fleet.collectedAt)}
              </div>
            </div>
            <div className={STATUS_GRID_CLASS}>
              {regions.length === 0 ? (
                <PanelMessage width={220} height={160}>No region data</PanelMessage>
              ) : (
                <RegionList
                  title="Create success by region"
                  rows={regions.map((region) => ({
                    region: region.region,
                    value: (
                      <span>
                        <strong>{formatPercent(region.successRatio)}</strong>
                        <span className={css`margin-left:6px;color:${theme.colors.text.secondary};font-size:12px;`}>
                          {region.ready}/{region.outcomes}
                        </span>
                      </span>
                    ),
                  }))}
                />
              )}
              <MiniSeriesChart points={history} unit="percent" />
            </div>
          </>
        )}
      </Card>
    </section>
  );
}

export function StatusPage({ frames, width, height }: Props) {
  const theme = useTheme2();
  const contentWidth = Math.max(320, width - 32);
  return (
    <main className={css`width:${width}px;min-height:${height}px;padding:18px 16px 28px;box-sizing:border-box;color:${theme.colors.text.primary};background:${theme.colors.background.canvas};overflow:hidden;`} aria-label="Marin infrastructure status">
      <header className={css`display:flex;align-items:baseline;justify-content:space-between;gap:20px;margin-bottom:16px;`}>
        <div>
          <h1 className={css`font-size:27px;line-height:1.2;margin:0;font-weight:700;letter-spacing:-.02em;`}>Marin Infra Status</h1>
          <span className={css`font-size:13px;color:${theme.colors.text.secondary};`}>Grafana refresh controls update all sources</span>
        </div>
        <nav className={css`display:flex;flex-wrap:wrap;justify-content:flex-end;gap:5px 14px;font-size:13px;`}>
          <a href="/d/marin-fleet">Fleet</a><a href="/d/marin-iris">Iris</a><a href="/d/marin-k8s">Kubernetes</a>
          <a href="/d/marin-training">Training</a><a href="https://github.com/marin-community/marin/actions" target="_blank" rel="noreferrer">GitHub Actions ↗</a>
        </nav>
      </header>

      <div className={css`display:flex;flex-direction:column;gap:20px;`}>
        <section>
          <NightlyMatrix frames={one(frames, REF.nightlies)} width={contentWidth} height={260} />
        </section>
        <section>
          <SectionTitle>GitHub status</SectionTitle>
          <Card><CommitStrip frames={one(frames, REF.builds)} width={contentWidth - 30} height={40} /></Card>
        </section>
        <div className={css`display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:24px;align-items:stretch;@media(max-width:1100px){grid-template-columns:1fr;}`}>
          <WorkerStatus frames={frames} />
          <ProvisioningStatus frames={frames} />
        </div>
        <section aria-label="Hero run charts">
          <SectionTitle detail="public W&B report · cumulative training tokens">Hero run</SectionTitle>
          <div className={css`display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;@media(max-width:1000px){grid-template-columns:1fr;}`}>
            {[REF.trainLoss, REF.paloma, REF.mfu].map((refId) => (
              <Card key={refId}><WandbChart frames={one(frames, refId)} width={Math.max(280, (contentWidth - 90) / 3)} height={260} /></Card>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}
