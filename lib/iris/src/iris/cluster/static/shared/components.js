/**
 * Shared Preact components used across dashboard pages.
 */
import { h } from 'preact';
import htm from 'htm';

const html = htm.bind(h);

export function InfoRow({ label, value, valueClass }) {
  return html`<div class="info-row">
    <span class="info-label">${label}</span>
    <span class=${'info-value ' + (valueClass || '')}>${value}</span>
  </div>`;
}

export function InfoCard({ title, children }) {
  return html`<div class="info-card">
    <h3>${title}</h3>
    ${children}
  </div>`;
}

/**
 * Bar gauge showing a percentage with color thresholds.
 *
 * @param {string} label - Left-side label text
 * @param {number} value - Current value
 * @param {number} max - Maximum value (100 for percentages)
 * @param {string} [format] - Display format: 'percent' (default), 'bytes', 'raw'
 * @param {number} [warnAt] - Percentage threshold for warning color (default 70)
 * @param {number} [dangerAt] - Percentage threshold for danger color (default 90)
 */
export function Gauge({ label, value, max, format = 'percent', warnAt = 70, dangerAt = 90 }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  const level = pct >= dangerAt ? 'danger' : pct >= warnAt ? 'warning' : 'ok';

  let displayValue;
  if (format === 'bytes') {
    displayValue = formatBytesCompact(value);
  } else if (format === 'percent') {
    displayValue = Math.round(pct) + '%';
  } else if (format === 'rate') {
    displayValue = formatRate(value);
  } else {
    displayValue = String(value);
  }

  return html`<div class="gauge">
    <span class="gauge-label">${label}</span>
    <div class="gauge-track">
      <div class=${'gauge-fill gauge-fill--' + level}
           style=${'width:' + pct.toFixed(1) + '%'}></div>
    </div>
    <span class=${'gauge-value gauge-value--' + level}>${displayValue}</span>
  </div>`;
}

/**
 * Compact gauge for use in tables. Shows a tiny bar + text value.
 *
 * @param {number} value - Current value
 * @param {number} max - Maximum/limit value
 * @param {string} [format] - 'percent' (default), 'bytes', or 'raw'
 * @param {string} [label] - Override display text (e.g., "1.5 / 4.0 GB")
 */
export function InlineGauge({ value, max, format = 'percent', label }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  const level = pct >= 90 ? 'danger' : pct >= 70 ? 'warning' : 'ok';

  let displayValue;
  if (label) {
    displayValue = label;
  } else if (format === 'bytes') {
    displayValue = formatBytesCompact(value);
  } else if (format === 'percent') {
    displayValue = Math.round(pct) + '%';
  } else {
    displayValue = String(value);
  }

  return html`<span class="inline-gauge">
    <span class="inline-gauge__track">
      <span class=${'inline-gauge__fill gauge-fill--' + level}
            style=${'width:' + pct.toFixed(1) + '%'}></span>
    </span>
    <span class="inline-gauge__text">${displayValue}</span>
  </span>`;
}

/**
 * SVG sparkline showing recent values as a line chart.
 * Useful for visualizing resource usage trends over time.
 *
 * @param {number[]} values - Array of data points (newest last)
 * @param {number} [max] - Y-axis maximum (defaults to max of values)
 * @param {number} [width] - SVG width in pixels (default 64)
 * @param {number} [height] - SVG height in pixels (default 20)
 * @param {string} [color] - Stroke color (default: accent blue)
 * @param {string} [fillColor] - Optional fill color under the line
 */
export function Sparkline({ values, max, width = 64, height = 20, color = 'var(--color-accent)', fillColor }) {
  if (!values || values.length < 1) return null;

  // Duplicate single values to render a flat baseline instead of hiding
  const data = values.length === 1 ? [values[0], values[0]] : values;
  const effectiveMax = max || Math.max(...data);

  // Pad from edges so the line doesn't clip
  const pad = 1;
  const innerW = width - 2 * pad;
  const innerH = height - 2 * pad;

  const points = data.map((v, i) => {
    const x = pad + (i / (data.length - 1)) * innerW;
    // When max is 0 (all idle), draw a flat line at the bottom
    const y = effectiveMax === 0
      ? pad + innerH
      : pad + innerH - (Math.min(v, effectiveMax) / effectiveMax) * innerH;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  const polyline = points.join(' ');

  // Area fill: same points but close the path along the bottom
  const areaPoints = polyline + ` ${(pad + innerW).toFixed(1)},${(pad + innerH).toFixed(1)} ${pad.toFixed(1)},${(pad + innerH).toFixed(1)}`;

  return html`<svg class="sparkline" width=${width} height=${height}
    viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
    ${fillColor && html`<polygon points=${areaPoints} fill=${fillColor} />`}
    <polyline fill="none" stroke=${color} stroke-width="1.5"
      stroke-linecap="round" stroke-linejoin="round" points=${polyline} />
  </svg>`;
}

/**
 * Prominent number display with label and optional detail text.
 *
 * @param {string|number} value - Main number to display
 * @param {string} label - Label below the number
 * @param {string} [detail] - Extra detail text below the label
 * @param {string} [valueClass] - CSS modifier for value color: 'accent', 'success', 'warning', 'danger', 'purple'
 */
export function MetricCard({ value, label, detail, valueClass }) {
  const cls = valueClass ? 'metric-card__value metric-card__value--' + valueClass : 'metric-card__value';
  return html`<div class="metric-card">
    <div class=${cls}>${value}</div>
    <div class="metric-card__label">${label}</div>
    ${detail && html`<div class="metric-card__detail">${detail}</div>`}
  </div>`;
}

/**
 * Section with title and gauge bars for resource display.
 */
export function ResourceSection({ title, children }) {
  return html`<div class="resource-section">
    ${title && html`<div class="resource-section__title">${title}</div>`}
    <div class="resource-section__gauges">${children}</div>
  </div>`;
}

export function Field({ label, value, valueClass, mono }) {
  if (value === undefined || value === null || value === '-' || value === '') return null;
  return html`<div class="worker-detail-field">
    <dt class="worker-detail-field__label">${label}</dt>
    <dd class=${'worker-detail-field__value' + (valueClass ? ' ' + valueClass : '') + (mono ? ' mono' : '')}>${value}</dd>
  </div>`;
}

export function Section({ title, icon, children, muted }) {
  return html`<div class=${'worker-detail-section' + (muted ? ' worker-detail-section--muted' : '')}>
    <h3 class="worker-detail-section__title">${icon && html`<span class="worker-detail-section__icon">${icon}</span>`}${title}</h3>
    <div class="worker-detail-section__body">${children}</div>
  </div>`;
}

function formatBytesCompact(bytes) {
  if (bytes === 0 || bytes == null) return '0';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const val = bytes / Math.pow(1024, i);
  return val >= 100 ? Math.round(val) + units[i] : val.toFixed(1) + units[i];
}

/**
 * Format a pair of MB values as a compact "current / limit" string, choosing
 * GB or MB depending on the larger value so both numbers share the same unit.
 *
 * Examples:
 *   formatMbPair(1536, 4096)  => "1.5 / 4.0 GB"
 *   formatMbPair(200, 512)    => "200 / 512 MB"
 *   formatMbPair(1536, 0)     => "1.5 GB"
 */
export function formatMbPair(currentMb, limitMb) {
  if (!limitMb || limitMb <= 0) {
    return currentMb >= 1024
      ? (currentMb / 1024).toFixed(1) + ' GB'
      : currentMb + ' MB';
  }
  if (limitMb >= 1024) {
    return (currentMb / 1024).toFixed(1) + ' / ' + (limitMb / 1024).toFixed(1) + ' GB';
  }
  return currentMb + ' / ' + limitMb + ' MB';
}

/**
 * Format CPU usage with optional core count context.
 *
 * Examples:
 *   formatCpuLabel(17, 8)    => "17% · 8c"
 *   formatCpuLabel(17, 0)    => "17%"
 *   formatCpuLabel(17, 0.5)  => "17% · 0.5c"
 */
export function formatCpuLabel(cpuPercent, cores) {
  if (!cores || cores <= 0) return cpuPercent + '%';
  const coreStr = cores >= 1 ? Math.round(cores) + 'c' : cores.toFixed(1) + 'c';
  return cpuPercent + '% · ' + coreStr;
}

/**
 * Format a byte rate (bytes/sec) as a human-readable string.
 *
 * Examples:
 *   formatRate(0)            => "0 B/s"
 *   formatRate(512)          => "512 B/s"
 *   formatRate(1536)         => "1.5 KB/s"
 *   formatRate(1572864)      => "1.5 MB/s"
 *   formatRate(1610612736)   => "1.5 GB/s"
 */
export function formatRate(bytesPerSec) {
  if (bytesPerSec === 0 || bytesPerSec == null) return '0 B/s';
  const units = ['B/s', 'KB/s', 'MB/s', 'GB/s'];
  const i = Math.min(Math.floor(Math.log(bytesPerSec) / Math.log(1024)), units.length - 1);
  const val = bytesPerSec / Math.pow(1024, i);
  return (val >= 100 ? Math.round(val) : val.toFixed(1)) + ' ' + units[i];
}

/**
 * Extract the display string from a proto AttributeValue.
 * The wire format is protobuf JSON: {stringValue, intValue, or floatValue}.
 */
export function attrValue(v) {
  if (!v) return '';
  // protobuf JSON uses camelCase field names
  if (v.stringValue !== undefined) return v.stringValue;
  if (v.intValue !== undefined) return String(v.intValue);
  if (v.floatValue !== undefined) return String(v.floatValue);
  return '';
}

/**
 * Format a single Constraint proto (as deserialized from protobuf JSON) into
 * a human-readable expression string.
 *
 * The `op` field is the protobuf enum string name (e.g. "CONSTRAINT_OP_IN").
 * Absent `op` means CONSTRAINT_OP_EQ (protobuf omits zero/default enum values
 * from JSON output).
 */
export function formatConstraint(c) {
  const op = c.op || 'CONSTRAINT_OP_EQ';
  switch (op) {
    case 'CONSTRAINT_OP_EQ':        return `${c.key} = ${attrValue(c.value)}`;
    case 'CONSTRAINT_OP_NE':        return `${c.key} != ${attrValue(c.value)}`;
    case 'CONSTRAINT_OP_EXISTS':    return `${c.key} exists`;
    case 'CONSTRAINT_OP_NOT_EXISTS': return `${c.key} !exists`;
    case 'CONSTRAINT_OP_GT':        return `${c.key} > ${attrValue(c.value)}`;
    case 'CONSTRAINT_OP_GE':        return `${c.key} >= ${attrValue(c.value)}`;
    case 'CONSTRAINT_OP_LT':        return `${c.key} < ${attrValue(c.value)}`;
    case 'CONSTRAINT_OP_LE':        return `${c.key} <= ${attrValue(c.value)}`;
    case 'CONSTRAINT_OP_IN': {
      const vals = (c.values || []).map(attrValue).join(', ');
      return `${c.key} in (${vals})`;
    }
    default: return `${c.key} ${op} ${attrValue(c.value)}`;
  }
}

/**
 * Render a list of Constraint protos as a compact, readable block.
 * Each constraint is shown as a monospace chip on its own line.
 * Renders nothing if the constraints array is empty.
 */
export function ConstraintList({ constraints }) {
  if (!constraints || constraints.length === 0) return null;
  return html`<div class="constraint-list">
    ${constraints.map((c, i) => html`
      <span key=${i} class="constraint-chip">${formatConstraint(c)}</span>
    `)}
  </div>`;
}
