import React, { MouseEvent, useEffect, useMemo, useRef, useState } from 'react';
import { css } from '@emotion/css';
import { DataFrame } from '@grafana/data';
import { useTheme2 } from '@grafana/ui';
import { frameWithField, smUtilizationPoints, smUtilizationRasterData } from '../data';
import { SmUtilizationDevice, SmUtilizationRasterData } from '../types';
import { PanelMessage } from './PanelMessage';

interface Props { frames: DataFrame[]; width: number; height: number }

interface Geometry {
  left: number;
  top: number;
  plotWidth: number;
  plotHeight: number;
  rowHeight: number;
}

interface HoveredCell {
  device: SmUtilizationDevice;
  sampledAt: number;
  percent: number;
  x: number;
  y: number;
}

const AXIS_HEIGHT = 22;
const CLUSTER_LABEL_WIDTH = 100;
const SM_COLOR_STOPS: Array<[number, string]> = [
  [0, '#064e3b'],
  [35, '#059669'],
  [55, '#22c55e'],
  [72, '#a3e635'],
  [85, '#facc15'],
  [100, '#fb923c'],
];

function channel(hex: string, offset: number): number {
  return Number.parseInt(hex.slice(offset, offset + 2), 16);
}

function blend(left: string, right: string, fraction: number): string {
  const value = [1, 3, 5]
    .map((offset) => Math.round(channel(left, offset) + (channel(right, offset) - channel(left, offset)) * fraction))
    .map((component) => component.toString(16).padStart(2, '0'))
    .join('');
  return `#${value}`;
}

function smColor(percent: number): string {
  const value = Math.max(0, Math.min(100, percent));
  const upperIndex = SM_COLOR_STOPS.findIndex(([stop]) => stop >= value);
  if (upperIndex <= 0) {return SM_COLOR_STOPS[0][1];}
  const [lowerValue, lowerColor] = SM_COLOR_STOPS[upperIndex - 1];
  const [upperValue, upperColor] = SM_COLOR_STOPS[upperIndex];
  return blend(lowerColor, upperColor, (value - lowerValue) / (upperValue - lowerValue));
}

const SM_COLORS = Array.from({ length: 101 }, (_, percent) => smColor(percent));

function geometry(width: number, height: number, deviceCount: number): Geometry {
  const left = width >= 520 ? CLUSTER_LABEL_WIDTH : 2;
  const top = 2;
  const plotWidth = Math.max(1, width - left - 2);
  const plotHeight = Math.max(1, height - top - AXIS_HEIGHT);
  return { left, top, plotWidth, plotHeight, rowHeight: plotHeight / deviceCount };
}

function sampleInterval(timestamps: number[]): number {
  const differences = timestamps.slice(1).map((timestamp, index) => timestamp - timestamps[index]);
  if (differences.length === 0) {return 60_000;}
  differences.sort((left, right) => left - right);
  return differences[Math.floor(differences.length / 2)];
}

function formatUtc(timestamp: number): string {
  return new Intl.DateTimeFormat(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    hourCycle: 'h23',
    timeZone: 'UTC',
  }).format(new Date(timestamp));
}

function nearestTimestampIndex(timestamps: number[], target: number): number {
  let low = 0;
  let high = timestamps.length - 1;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (timestamps[middle] < target) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  if (low === 0) {return low;}
  return target - timestamps[low - 1] <= timestamps[low] - target ? low - 1 : low;
}

function drawRaster(
  canvas: HTMLCanvasElement,
  raster: SmUtilizationRasterData,
  width: number,
  height: number,
  colors: { background: string; border: string; text: string }
): void {
  const pixelRatio = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.max(1, Math.ceil(width * pixelRatio));
  canvas.height = Math.max(1, Math.ceil(height * pixelRatio));
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const context = canvas.getContext('2d');
  if (!context) {return;}
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, width, height);

  const layout = geometry(width, height, raster.devices.length);
  const interval = sampleInterval(raster.timestamps);
  const timeStart = raster.timestamps[0];
  const timeEnd = raster.timestamps[raster.timestamps.length - 1] + interval;
  const timeSpan = Math.max(1, timeEnd - timeStart);
  const bucketWidth = layout.plotWidth * interval / timeSpan;
  const columnGap = Math.min(1.5, bucketWidth * 0.14);
  const rowGap = layout.rowHeight >= 2 ? Math.min(1, layout.rowHeight * 0.15) : 0;

  context.fillStyle = colors.background;
  context.fillRect(layout.left, layout.top, layout.plotWidth, layout.plotHeight);
  for (let deviceIndex = 0; deviceIndex < raster.devices.length; deviceIndex += 1) {
    const y = layout.top + deviceIndex * layout.rowHeight;
    for (let timestampIndex = 0; timestampIndex < raster.timestamps.length; timestampIndex += 1) {
      const percent = raster.values[deviceIndex * raster.timestamps.length + timestampIndex];
      if (!Number.isFinite(percent)) {continue;}
      const x = layout.left + (raster.timestamps[timestampIndex] - timeStart) / timeSpan * layout.plotWidth;
      context.fillStyle = SM_COLORS[Math.round(Math.max(0, Math.min(100, percent)))];
      context.fillRect(
        x + columnGap / 2,
        y + rowGap / 2,
        Math.max(0.5, bucketWidth - columnGap),
        Math.max(0.5 / pixelRatio, layout.rowHeight - rowGap)
      );
    }
  }

  context.strokeStyle = colors.border;
  context.lineWidth = 1;
  let clusterStart = 0;
  for (let index = 1; index <= raster.devices.length; index += 1) {
    const clusterEnded = index === raster.devices.length
      || raster.devices[index].cluster !== raster.devices[clusterStart].cluster;
    if (!clusterEnded) {continue;}
    const startY = layout.top + clusterStart * layout.rowHeight;
    const endY = layout.top + index * layout.rowHeight;
    context.beginPath();
    context.moveTo(layout.left, endY);
    context.lineTo(layout.left + layout.plotWidth, endY);
    context.stroke();
    if (layout.left >= CLUSTER_LABEL_WIDTH && endY - startY >= 11) {
      context.fillStyle = colors.text;
      context.font = '11px Inter, sans-serif';
      context.textAlign = 'right';
      context.textBaseline = 'middle';
      context.fillText(raster.devices[clusterStart].cluster, layout.left - 8, (startY + endY) / 2, layout.left - 12);
    }
    clusterStart = index;
  }

  context.fillStyle = colors.text;
  context.font = '11px Inter, sans-serif';
  context.textBaseline = 'bottom';
  for (const fraction of [0, 0.25, 0.5, 0.75, 1]) {
    const x = layout.left + fraction * layout.plotWidth;
    const timestamp = timeStart + fraction * timeSpan;
    context.textAlign = fraction === 0 ? 'left' : fraction === 1 ? 'right' : 'center';
    context.fillText(formatUtc(timestamp), x, height - 2);
  }
}

export function SmUtilizationRaster({ frames, width, height }: Props) {
  const theme = useTheme2();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hovered, setHovered] = useState<HoveredCell | null>(null);
  const frame = frameWithField(frames, 'sm_utilization');
  const points = useMemo(() => (frame ? smUtilizationPoints(frame) : []), [frame]);
  const raster = useMemo(() => smUtilizationRasterData(points), [points]);
  const clusters = useMemo(() => new Set(raster.devices.map((device) => device.cluster)).size, [raster.devices]);
  const nodes = useMemo(
    () => new Set(raster.devices.map((device) => `${device.cluster}\u0000${device.node}`)).size,
    [raster.devices]
  );

  useEffect(() => {
    if (!canvasRef.current || raster.devices.length === 0 || raster.timestamps.length === 0) {return;}
    drawRaster(canvasRef.current, raster, width, height, {
      background: theme.colors.background.secondary,
      border: theme.colors.border.weak,
      text: theme.colors.text.secondary,
    });
  }, [height, raster, theme.colors.background.secondary, theme.colors.border.weak, theme.colors.text.secondary, width]);

  if (points.length === 0) {
    return <PanelMessage width={width} height={height}>No SM utilization data</PanelMessage>;
  }

  const updateHovered = (event: MouseEvent<HTMLCanvasElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    const x = (event.clientX - bounds.left) * width / bounds.width;
    const y = (event.clientY - bounds.top) * height / bounds.height;
    const layout = geometry(width, height, raster.devices.length);
    if (x < layout.left || x > layout.left + layout.plotWidth || y < layout.top || y >= layout.top + layout.plotHeight) {
      setHovered(null);
      return;
    }
    const deviceIndex = Math.min(raster.devices.length - 1, Math.floor((y - layout.top) / layout.rowHeight));
    const interval = sampleInterval(raster.timestamps);
    const timeStart = raster.timestamps[0];
    const timeEnd = raster.timestamps[raster.timestamps.length - 1] + interval;
    const timestamp = timeStart + (x - layout.left) / layout.plotWidth * (timeEnd - timeStart);
    const timestampIndex = nearestTimestampIndex(raster.timestamps, timestamp);
    const percent = raster.values[deviceIndex * raster.timestamps.length + timestampIndex];
    if (!Number.isFinite(percent) || Math.abs(raster.timestamps[timestampIndex] - timestamp) > interval / 2) {
      setHovered(null);
      return;
    }
    setHovered({
      device: raster.devices[deviceIndex],
      sampledAt: raster.timestamps[timestampIndex],
      percent,
      x: Math.max(4, Math.min(width - 224, x + 12)),
      y: Math.max(4, Math.min(height - 64, y + 10)),
    });
  };

  const summary = `SM utilization for ${raster.devices.length} GPUs across ${nodes} ${nodes === 1 ? 'node' : 'nodes'} and ${clusters} ${clusters === 1 ? 'cluster' : 'clusters'}`;
  return (
    <section
      className={css`width:${width}px;height:${height}px;position:relative;overflow:hidden;color:${theme.colors.text.primary};`}
      aria-label="SM utilization raster"
    >
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={summary}
        className={css`display:block;`}
        onMouseMove={updateHovered}
        onMouseLeave={() => setHovered(null)}
      />
      <div
        aria-label="SM utilization color scale from 0 to 100 percent"
        className={css`
          position:absolute;right:8px;top:6px;width:116px;padding:3px 5px;border-radius:3px;
          background:${theme.colors.background.primary};color:${theme.colors.text.secondary};font-size:10px;
        `}
      >
        <div className={css`height:5px;border-radius:2px;background:linear-gradient(90deg,${SM_COLOR_STOPS.map(([value, color]) => `${color} ${value}%`).join(',')});`} />
        <div className={css`display:flex;justify-content:space-between;margin-top:2px;`}><span>0</span><span>SM active</span><span>100%</span></div>
      </div>
      {hovered && (
        <div
          role="tooltip"
          className={css`
            position:absolute;left:${hovered.x}px;top:${hovered.y}px;min-width:210px;padding:7px 9px;border-radius:4px;
            background:${theme.colors.background.elevated};border:1px solid ${theme.colors.border.medium};
            box-shadow:${theme.shadows.z2};font-size:12px;line-height:1.35;pointer-events:none;
          `}
        >
          <strong>{hovered.percent.toFixed(1)}% SM active</strong>
          <div>{hovered.device.cluster} · {hovered.device.node} · GPU {hovered.device.gpu}</div>
          <div className={css`color:${theme.colors.text.secondary};`}>{formatUtc(hovered.sampledAt)} UTC</div>
        </div>
      )}
    </section>
  );
}
