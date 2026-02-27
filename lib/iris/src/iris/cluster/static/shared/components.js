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
 */
export function InlineGauge({ value, max, format = 'percent' }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  const level = pct >= 90 ? 'danger' : pct >= 70 ? 'warning' : 'ok';

  let displayValue;
  if (format === 'bytes') {
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
