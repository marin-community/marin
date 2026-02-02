import { h } from 'preact';
import htm from 'htm';
import { formatBytes, formatRelativeTime, formatAttributes } from '/static/shared/utils.js';

const html = htm.bind(h);

export function WorkersTab({ workers }) {
  if (!workers || workers.length === 0) {
    return html`<div class="no-jobs">No workers registered</div>`;
  }

  return html`
    <table>
      <thead><tr>
        <th>ID</th><th>Healthy</th><th>CPU</th><th>Memory</th>
        <th>Running Tasks</th><th>Last Heartbeat</th><th>Attributes</th><th>Status</th>
      </tr></thead>
      <tbody>
        ${workers.map(w => {
          const healthClass = w.healthy ? 'healthy' : 'unhealthy';
          const healthIndicator = w.healthy ? '\u25cf' : '\u25cb';
          const healthText = w.healthy ? 'Yes' : 'No';
          const wid = w.worker_id;
          const cpu = w.resources ? w.resources.cpu : '-';
          const memBytes = w.resources ? (w.resources.memory_bytes || 0) : 0;
          const memory = memBytes ? formatBytes(memBytes) : '-';

          return html`<tr>
            <td>${w.address
              ? html`<a href=${'http://' + w.address + '/'} class="worker-link" target="_blank">${wid}</a>`
              : wid}</td>
            <td class=${healthClass}>${healthIndicator} ${healthText}</td>
            <td>${cpu}</td>
            <td>${memory}</td>
            <td>${w.running_tasks}</td>
            <td>${formatRelativeTime(w.last_heartbeat_ms)}</td>
            <td>${formatAttributes(w.attributes)}</td>
            <td style=${'font-size:12px;color:' + (w.healthy ? '#57606a' : '#cf222e')}>${w.status_message || '-'}</td>
          </tr>`;
        })}
      </tbody>
    </table>`;
}
