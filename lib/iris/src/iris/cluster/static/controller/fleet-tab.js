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
 * Merge VMs (from autoscaler groups) with workers (from ListWorkers) into a
 * flat list of fleet nodes. Each VM that has a registered worker gets enriched
 * with worker data. Workers not matched to any VM appear as orphan entries.
 */
function mergeFleetNodes(groups, workers) {
  const workerMap = new Map();
  for (const w of (workers || [])) {
    workerMap.set(w.worker_id, w);
  }

  const nodes = [];
  const matchedWorkerIds = new Set();

  for (const group of (groups || [])) {
    const config = group.config || {};
    const accel = formatAcceleratorDisplay(config.acceleratorType, config.acceleratorVariant || '');
    for (const slice of (group.slices || [])) {
      for (const vm of (slice.vms || [])) {
        const worker = vm.workerId ? workerMap.get(vm.workerId) : null;
        if (worker) matchedWorkerIds.add(vm.workerId);
        nodes.push({ vm, worker, groupName: group.name, sliceId: slice.sliceId, accel });
      }
    }
  }

  for (const w of (workers || [])) {
    if (!matchedWorkerIds.has(w.worker_id)) {
      nodes.push({ vm: null, worker: w, groupName: '-', sliceId: '-', accel: formatDevice(w) });
    }
  }

  return nodes;
}

function effectiveState(node) {
  if (node.vm) {
    const vmState = formatVmState(node.vm.state);
    if (vmState === 'ready' && node.worker && !node.worker.healthy) return 'unhealthy';
    return vmState;
  }
  if (node.worker) {
    return node.worker.healthy ? 'ready' : 'unhealthy';
  }
  return 'unknown';
}

export function FleetTab({ groups, workers, page, onPageChange }) {
  const hasGroups = groups && groups.length > 0;
  const hasWorkers = workers && workers.length > 0;

  if (!hasGroups && !hasWorkers) {
    return html`<div class="no-jobs">No machines in fleet</div>`;
  }

  const nodes = mergeFleetNodes(groups, workers);

  if (nodes.length === 0) {
    return html`
      <${SliceStateSummary} groups=${groups} />
      <div class="no-jobs">No machines in fleet</div>`;
  }

  const pageSize = 50;
  const totalPages = Math.ceil(nodes.length / pageSize);
  const currentPage = Math.max(0, Math.min(page, totalPages - 1));
  const pageNodes = nodes.slice(currentPage * pageSize, (currentPage + 1) * pageSize);

  return html`
    <${SliceStateSummary} groups=${groups} />
    <table>
      <thead><tr>
        <th>Machine</th><th>Group</th><th>Slice</th><th>Accelerator</th>
        <th>State</th><th>Health</th><th>CPU</th><th>Memory</th>
        <th>Tasks</th><th>Last Heartbeat</th><th>Error</th>
      </tr></thead>
      <tbody>
        ${pageNodes.map(node => {
          const state = effectiveState(node);
          const vmId = node.vm ? node.vm.vmId : null;
          const workerId = node.worker ? node.worker.worker_id : (node.vm ? (node.vm.workerId || null) : null);
          const label = vmId || workerId || '-';
          const linkId = vmId || workerId;
          const link = linkId ? ('/worker/' + encodeURIComponent(linkId)) : null;

          const healthIndicator = node.worker
            ? (node.worker.healthy ? '\u25cf' : '\u25cb')
            : '-';
          const healthClass = node.worker
            ? (node.worker.healthy ? 'healthy' : 'unhealthy')
            : '';

          const cpu = node.worker && node.worker.resources
            ? (node.worker.resources.cpu || 0)
            : '-';
          const memBytes = node.worker && node.worker.resources
            ? (node.worker.resources.memory_bytes || 0)
            : 0;
          const memory = memBytes ? formatBytes(memBytes) : '-';
          const tasks = node.worker != null ? node.worker.running_tasks : '-';
          const heartbeat = node.worker ? formatRelativeTime(node.worker.last_heartbeat_ms) : '-';
          const error = (node.vm && node.vm.initError) || (node.worker && !node.worker.healthy ? node.worker.status_message : '') || '-';

          return html`<tr>
            <td>${link
              ? html`<a href=${link} class="job-link">${label}</a>`
              : label}</td>
            <td>${node.groupName}</td>
            <td>${node.sliceId}</td>
            <td>${node.accel}</td>
            <td><span class=${'vm-state-indicator ' + state}></span><span class=${'status-' + state}>${state}</span></td>
            <td class=${healthClass}>${healthIndicator}</td>
            <td>${cpu}</td>
            <td>${memory}</td>
            <td>${tasks}</td>
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
