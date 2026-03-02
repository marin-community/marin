import { h } from 'preact';
import htm from 'htm';
import {
  formatVmState, formatAcceleratorDisplay, formatBytes,
  formatRelativeTime
} from '/static/shared/utils.js';

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

function SliceStateSummary({ groups }) {
  if (!groups || groups.length === 0) return null;
  const stateLabels = ['requesting', 'booting', 'initializing', 'ready', 'failed'];
  const stateColors = { requesting: '#6366f1', booting: '#f59e0b', initializing: '#3b82f6', ready: '#22c55e', failed: '#ef4444' };
  return html`
    <div style="display:flex;gap:16px;margin-bottom:12px;flex-wrap:wrap">
      ${groups.map(group => {
        const counts = group.sliceStateCounts || {};
        const badges = stateLabels.filter(s => counts[s] > 0);
        if (badges.length === 0) return null;
        return html`
          <div style="border:1px solid #333;border-radius:6px;padding:6px 10px;font-size:13px">
            <strong>${group.name}</strong>: ${' '}
            ${badges.map(s => html`
              <span style=${'display:inline-block;margin-left:6px;padding:1px 6px;border-radius:3px;background:' + stateColors[s] + '22;color:' + stateColors[s]}>
                ${counts[s]} ${s}
              </span>
            `)}
          </div>`;
      })}
    </div>`;
}

/**
 * Build worker rows from the workers list, enriching with scale-group
 * metadata from autoscaler groups where possible.
 */
function buildWorkerRows(groups, workers) {
  // Build a lookup from worker address (host only) to group/slice info
  const addrToGroup = new Map();
  for (const group of (groups || [])) {
    const config = group.config || {};
    const accel = formatAcceleratorDisplay(config.acceleratorType, config.acceleratorVariant || '');
    for (const slice of (group.slices || [])) {
      for (const vm of (slice.vms || [])) {
        if (vm.address) {
          addrToGroup.set(vm.address, { groupName: group.name, sliceId: slice.sliceId, accel, vm });
        }
      }
    }
  }

  return (workers || []).map(w => {
    const host = w.address && w.address.includes(':') ? w.address.split(':')[0] : w.address;
    const groupInfo = host ? addrToGroup.get(host) : null;
    return {
      worker: w,
      groupName: groupInfo ? groupInfo.groupName : '-',
      sliceId: groupInfo ? groupInfo.sliceId : '-',
      accel: groupInfo ? groupInfo.accel : formatDevice(w),
      vm: groupInfo ? groupInfo.vm : null,
    };
  });
}

export function FleetTab({ groups, workers, page, onPageChange }) {
  const hasWorkers = workers && workers.length > 0;

  if (!hasWorkers) {
    return html`
      <${SliceStateSummary} groups=${groups} />
      <div class="no-jobs">No workers online</div>`;
  }

  const rows = buildWorkerRows(groups, workers);

  const pageSize = 50;
  const totalPages = Math.ceil(rows.length / pageSize);
  const currentPage = Math.max(0, Math.min(page, totalPages - 1));
  const pageRows = rows.slice(currentPage * pageSize, (currentPage + 1) * pageSize);

  return html`
    <${SliceStateSummary} groups=${groups} />
    <table>
      <thead><tr>
        <th>Worker</th><th>Group</th><th>Slice</th><th>Accelerator</th>
        <th>Health</th><th>CPU</th><th>Memory</th>
        <th>Tasks</th><th>Last Heartbeat</th><th>Error</th>
      </tr></thead>
      <tbody>
        ${pageRows.map(row => {
          const w = row.worker;
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
            <td>${row.groupName}</td>
            <td>${row.sliceId}</td>
            <td>${row.accel}</td>
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
