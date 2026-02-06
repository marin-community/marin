import { h } from 'preact';
import { useState, useRef } from 'preact/hooks';
import htm from 'htm';
import { formatDuration, formatRelativeTime } from '/static/shared/utils.js';

const html = htm.bind(h);

/** Sortable column header component. */
function SortHeader({ field, label, sortField, sortDir, onSort }) {
  const isActive = sortField === field;
  const indicator = isActive ? (sortDir === 'asc' ? ' \u25B2' : ' \u25BC') : '';
  return html`<th style="cursor:pointer;user-select:none" onClick=${() => onSort(field)}>
    ${label}${indicator}
  </th>`;
}

function TaskProgressBar({ job }) {
  const counts = job.task_state_counts || {};
  const total = job.task_count || 0;
  if (total === 0) return html`<span style="color:#57606a;font-size:12px">no tasks</span>`;

  const succeeded = counts.succeeded || 0;
  const running = counts.running || 0;
  const building = counts.building || 0;
  const failed = counts.failed || 0;
  const killed = counts.killed || 0;
  const workerFailed = counts.worker_failed || 0;
  const assigned = counts.assigned || 0;
  const pending = total - succeeded - running - building - failed - killed - workerFailed - assigned;

  const segments = [
    {count: succeeded, color: '#1a7f37', label: 'succeeded'},
    {count: running, color: '#0969da', label: 'running'},
    {count: building, color: '#8250df', label: 'building'},
    {count: assigned, color: '#bc4c00', label: 'assigned'},
    {count: failed, color: '#cf222e', label: 'failed'},
    {count: workerFailed, color: '#8250df', label: 'worker_failed'},
    {count: killed, color: '#57606a', label: 'killed'},
    {count: pending, color: '#eaeef2', label: 'pending'},
  ].filter(s => s.count > 0);

  return html`
    <div style="display:flex;align-items:center;gap:6px">
      <div style="display:flex;height:8px;width:120px;border-radius:4px;overflow:hidden;background:#eaeef2">
        ${segments.map(s => html`
          <div style=${'width:' + (s.count / total * 100).toFixed(1) + '%;background:' + s.color}
               title=${s.label + ': ' + s.count}></div>
        `)}
      </div>
      <span style="font-size:11px;color:#57606a">${succeeded}/${total}</span>
    </div>`;
}

export function JobsTab({
  jobs, totalCount, hasMore, page, pageSize,
  sortField, sortDir, nameFilter,
  onPageChange, onSortChange, onFilterChange
}) {
  const filterInputRef = useRef(null);
  const [localFilter, setLocalFilter] = useState(nameFilter || '');

  function handleSort(field) {
    if (sortField === field) {
      onSortChange(field, sortDir === 'asc' ? 'desc' : 'asc');
    } else {
      // Default direction: descending for date, ascending for others
      onSortChange(field, field === 'date' ? 'desc' : 'asc');
    }
  }

  function handleFilterSubmit(e) {
    e.preventDefault();
    onFilterChange(localFilter);
  }

  function handleFilterClear() {
    setLocalFilter('');
    onFilterChange('');
  }

  const totalPages = Math.ceil(totalCount / pageSize);
  const currentPage = page;
  const hasPrev = currentPage > 0;
  const hasNext = hasMore;

  return html`
    <div style="margin-bottom:15px;display:flex;align-items:center;gap:10px">
      <form onSubmit=${handleFilterSubmit} style="display:flex;gap:8px">
        <input
          ref=${filterInputRef}
          type="text"
          placeholder="Filter by name..."
          value=${localFilter}
          onInput=${e => setLocalFilter(e.target.value)}
          style="padding:6px 10px;border:1px solid #d1d9e0;border-radius:4px;font-size:13px;width:200px"
        />
        <button type="submit" style="padding:6px 12px;font-size:13px">Filter</button>
        ${nameFilter && html`<button type="button" onClick=${handleFilterClear} style="padding:6px 12px;font-size:13px">Clear</button>`}
      </form>
      <span style="color:#57606a;font-size:13px">${totalCount} job${totalCount !== 1 ? 's' : ''}</span>
    </div>
    ${(!jobs || jobs.length === 0) ? html`<div class="no-jobs">No jobs${nameFilter ? ' matching filter' : ''}</div>` : html`
    <table>
      <thead><tr>
        <th>ID</th>
        <${SortHeader} field="name" label="Name" sortField=${sortField} sortDir=${sortDir} onSort=${handleSort} />
        <${SortHeader} field="state" label="State" sortField=${sortField} sortDir=${sortDir} onSort=${handleSort} />
        <th>Tasks</th>
        <${SortHeader} field="date" label="Date" sortField=${sortField} sortDir=${sortDir} onSort=${handleSort} />
        <${SortHeader} field="failures" label="Failures" sortField=${sortField} sortDir=${sortDir} onSort=${handleSort} />
        <${SortHeader} field="preemptions" label="Preemptions" sortField=${sortField} sortDir=${sortDir} onSort=${handleSort} />
        <th>Diagnostic</th>
      </tr></thead>
      <tbody>
        ${jobs.map(job => {
          const shortId = job.job_id.slice(0, 8);
          let duration = '-';
          if (job.started_at_ms) {
            duration = formatDuration(job.started_at_ms, job.finished_at_ms || Date.now());
          } else if (job.submitted_at_ms) {
            duration = 'queued ' + formatRelativeTime(job.submitted_at_ms);
          }
          const diagnostic = job.pending_reason || '';
          const diagnosticDisplay = diagnostic.length > 100 ? diagnostic.substring(0, 97) + '...' : diagnostic;

          return html`<tr>
            <td><a href=${'/job/' + job.job_id} class="job-link">${shortId}...</a></td>
            <td>${job.name || 'unnamed'}</td>
            <td><span class=${'status-' + job.state}>${job.state}</span></td>
            <td><${TaskProgressBar} job=${job} /></td>
            <td>${duration}</td>
            <td>${job.failure_count || 0}</td>
            <td>${job.preemption_count || 0}</td>
            <td style="font-size:12px;color:#57606a;max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
                title=${diagnostic}>${diagnosticDisplay || '-'}</td>
          </tr>`;
        })}
      </tbody>
    </table>
    `}
    ${totalPages > 1 && html`
      <div style="margin-top:10px;display:flex;justify-content:space-between;align-items:center">
        <button disabled=${!hasPrev} onClick=${() => onPageChange(currentPage - 1)}>Prev</button>
        <span>Page ${currentPage + 1} of ${totalPages}</span>
        <button disabled=${!hasNext} onClick=${() => onPageChange(currentPage + 1)}>Next</button>
      </div>
    `}`;
}
