import { h } from 'preact';
import htm from 'htm';
import { formatVmState, formatAcceleratorDisplay } from '/static/shared/utils.js';

const html = htm.bind(h);

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

export function VmsTab({ groups, page, onPageChange }) {
  if (!groups || groups.length === 0) {
    return html`<div class="no-jobs">No scale groups configured</div>`;
  }

  const allVms = [];
  for (const group of groups) {
    const config = group.config || {};
    const accel = formatAcceleratorDisplay(config.acceleratorType, config.acceleratorVariant || '');
    for (const slice of (group.slices || [])) {
      for (const vm of (slice.vms || [])) {
        allVms.push({ vm, groupName: group.name, sliceId: slice.sliceId, accel });
      }
    }
  }

  if (allVms.length === 0) {
    return html`<${SliceStateSummary} groups=${groups} />
      <div class="no-jobs">No VMs</div>`;
  }

  const pageSize = 50;
  const totalPages = Math.ceil(allVms.length / pageSize);
  const currentPage = Math.max(0, Math.min(page, totalPages - 1));
  const pageVms = allVms.slice(currentPage * pageSize, (currentPage + 1) * pageSize);

  return html`
    <${SliceStateSummary} groups=${groups} />
    <table>
      <thead><tr>
        <th>VM ID</th><th>Scale Group</th><th>Slice</th><th>Accelerator</th>
        <th>State</th><th>Address</th><th>Worker</th><th>Error</th>
      </tr></thead>
      <tbody>
        ${pageVms.map(({ vm, groupName, sliceId, accel }) => {
          const state = formatVmState(vm.state);
          return html`<tr>
            <td><a href=${'/vm/' + encodeURIComponent(vm.vmId)} class="job-link">${vm.vmId}</a></td>
            <td>${groupName}</td>
            <td>${sliceId}</td>
            <td>${accel}</td>
            <td><span class=${'vm-state-indicator ' + state}></span><span class=${'status-' + state}>${state}</span></td>
            <td>${vm.address || '-'}</td>
            <td>${vm.workerId || '-'}</td>
            <td>${vm.initError || '-'}</td>
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
