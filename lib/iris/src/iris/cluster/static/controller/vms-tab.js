import { h } from 'preact';
import htm from 'htm';
import { formatVmState, formatAcceleratorDisplay } from '/static/shared/utils.js';

const html = htm.bind(h);

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
    return html`<div class="no-jobs">No VMs</div>`;
  }

  const pageSize = 50;
  const totalPages = Math.ceil(allVms.length / pageSize);
  const currentPage = Math.max(0, Math.min(page, totalPages - 1));
  const pageVms = allVms.slice(currentPage * pageSize, (currentPage + 1) * pageSize);

  return html`
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
