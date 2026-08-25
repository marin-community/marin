import React from 'react';
import { css } from '@emotion/css';
import { DataFrame } from '@grafana/data';
import { useTheme2 } from '@grafana/ui';
import { frameByRefId, workerRegions } from '../data';
import { formatBytes } from '../format';
import { CommitStrip } from './CommitStrip';
import { NightlyMatrix } from './NightlyMatrix';
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
  trainLoss: 'T',
  paloma: 'L',
  mfu: 'M',
} as const;

const STATUS_CARD_CLASS = css`flex:1;`;
const STATUS_SECTION_CLASS = css`display:flex;flex-direction:column;`;

function one(frames: DataFrame[], refId: string): DataFrame[] {
  const frame = frameByRefId(frames, refId);
  return frame ? [frame] : [];
}

function formatCores(millicores: number): string {
  const cores = millicores / 1000;
  return cores >= 1000 ? `${(cores / 1000).toFixed(1)}k` : Math.round(cores).toString();
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
  const regions = currentFrame ? workerRegions(currentFrame).sort((a, b) => b.healthy - a.healthy) : [];
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
      <SectionTitle detail="current capacity">Workers</SectionTitle>
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
            <RegionList
              title="Availability by region"
              rows={regions.map((region) => ({ region: region.region, value: <strong>{region.healthy}</strong> }))}
            />
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
          <a href="/d/marin-home">Home</a><a href="/d/marin-accel">Accelerators</a><a href="/d/marin-jobs">Jobs</a>
          <a href="/d/marin-runs">Runs</a><a href="/d/marin-clusters">Clusters</a><a href="/d/marin-cluster-capacity">Capacity</a>
          <a href="https://github.com/marin-community/marin/actions" target="_blank" rel="noreferrer">GitHub Actions ↗</a>
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
        <WorkerStatus frames={frames} />
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
