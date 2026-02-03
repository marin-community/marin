import { h } from 'preact';
import htm from 'htm';
import { formatDuration, formatRelativeTime } from '/static/shared/utils.js';

const html = htm.bind(h);

const STATE_ORDER = {running: 0, building: 1, assigned: 2, pending: 3, succeeded: 4, failed: 5, killed: 6, worker_failed: 7};

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

export function JobsTab({ jobs, page, onPageChange }) {
  if (!jobs || jobs.length === 0) {
    return html`<div class="no-jobs">No jobs</div>`;
  }

  const sorted = [...jobs].sort((a, b) => (STATE_ORDER[a.state] ?? 99) - (STATE_ORDER[b.state] ?? 99));

  const pageSize = 50;
  const totalPages = Math.ceil(sorted.length / pageSize);
  const currentPage = Math.max(0, Math.min(page, totalPages - 1));
  const pageJobs = sorted.slice(currentPage * pageSize, (currentPage + 1) * pageSize);

  return html`
    <table>
      <thead><tr>
        <th>ID</th><th>Name</th><th>State</th><th>Tasks</th>
        <th>Duration</th><th>Failures</th><th>Preemptions</th><th>Diagnostic</th>
      </tr></thead>
      <tbody>
        ${pageJobs.map(job => {
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
    ${totalPages > 1 && html`
      <div style="margin-top:10px;display:flex;justify-content:space-between;align-items:center">
        <button disabled=${currentPage === 0} onClick=${() => onPageChange(currentPage - 1)}>Prev</button>
        <span>Page ${currentPage + 1} of ${totalPages}</span>
        <button disabled=${currentPage >= totalPages - 1} onClick=${() => onPageChange(currentPage + 1)}>Next</button>
      </div>
    `}`;
}
