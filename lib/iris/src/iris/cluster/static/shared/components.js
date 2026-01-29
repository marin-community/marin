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
