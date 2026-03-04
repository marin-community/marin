import { h } from 'preact';
import htm from 'htm';
import { formatBytes, formatRelativeTime } from '/static/shared/utils.js';

const html = htm.bind(h);

function formatDevice(w) {
  if (w.gpu_count > 0) {
    const name = w.gpu_name || 'GPU';
    const mem = w.gpu_memory_mb ? ` (${Math.round(w.gpu_memory_mb / 1024)}GB)` : '';
    return `GPU: ${w.gpu_count}x ${name}${mem}`;
  }
  const device = w.device;
  if (device) {
    if (device.tpu) return `TPU: ${device.tpu.variant || 'unknown'}`;
    if (device.gpu) return `GPU: ${device.gpu.count || 1}x ${device.gpu.variant || 'unknown'}`;
  }
  return 'CPU';
}

export function FleetTab({ workers, page, onPageChange }) {
  const hasWorkers = workers && workers.length > 0;

  if (!hasWorkers) {
    return html`<div class="no-jobs">No workers online</div>`;
  }

  const pageSize = 50;
  const totalPages = Math.ceil(workers.length / pageSize);
  const currentPage = Math.max(0, Math.min(page, totalPages - 1));
  const pageRows = workers.slice(currentPage * pageSize, (currentPage + 1) * pageSize);

  return html`
    <table>
      <thead><tr>
        <th>Worker</th><th>Address</th><th>Accelerator</th>
        <th>Health</th><th>CPU</th><th>Memory</th>
        <th>Tasks</th><th>Last Heartbeat</th><th>Error</th>
      </tr></thead>
      <tbody>
        ${pageRows.map(w => {
          const link = '/worker/' + encodeURIComponent(w.worker_id);
          const healthIndicator = w.healthy ? '\u25cf' : '\u25cb';
          const healthClass = w.healthy ? 'healthy' : 'unhealthy';

          const cpuCount = w.resources ? (w.resources.cpu || 0) : 0;
          const cpuDisplay = cpuCount > 0 ? `${cpuCount} cores` : '-';
          const memBytes = w.resources ? (w.resources.memory_bytes || 0) : 0;
          const memory = memBytes ? formatBytes(memBytes) : '-';
          const heartbeat = formatRelativeTime(w.last_heartbeat_ms);
          const error = (!w.healthy && w.status_message) ? w.status_message : '-';

          return html`<tr>
            <td><a href=${link} class="job-link">${w.worker_id}</a></td>
            <td style="font-family:var(--font-mono);font-size:12px">${w.address || '-'}</td>
            <td>${formatDevice(w)}</td>
            <td class=${healthClass}>${healthIndicator} ${w.healthy ? 'healthy' : 'unhealthy'}</td>
            <td>${cpuDisplay}</td>
            <td>${memory}</td>
            <td>${w.running_tasks > 0
              ? html`<span class="task-count-badge">${w.running_tasks}</span>`
              : html`<span style="color:var(--color-text-muted)">0</span>`}</td>
            <td>${heartbeat}</td>
            <td style="font-size:12px;max-width:200px;overflow:hidden;text-overflow:ellipsis">${error}</td>
          </tr>`;
        })}
      </tbody>
    </table>
    ${totalPages > 1 && html`
      <div style="margin-top:10px;display:flex;justify-content:space-between;align-items:center">
        <button disabled=${currentPage === 0} onClick=${() => onPageChange(currentPage - 1)}>Prev</button>
        <span>Page ${currentPage + 1} of ${totalPages}</span>
        <button disabled=${currentPage >= totalPages - 1} onClick=${() => onPageChange(currentPage + 1)}>Next</button>
      </div>
    `}`;
}
