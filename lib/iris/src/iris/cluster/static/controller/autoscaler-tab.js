import { h } from 'preact';
import { useState, useEffect } from 'preact/hooks';
import htm from 'htm';
import { formatRelativeTime, timestampFromProto } from '/static/shared/utils.js';
import { controllerRpc } from '/static/shared/rpc.js';

const html = htm.bind(h);

/**
 * Compute per-slice state counts from an array of SliceInfo objects.
 * Each slice is categorized by the aggregate state of its VMs.
 */
function computeSliceStateCounts(slices) {
  const counts = { requesting: 0, booting: 0, initializing: 0, ready: 0, failed: 0 };
  for (const s of slices) {
    const vms = s.vms || [];
    if (vms.length === 0) continue;
    if (vms.every(vm => vm.state === 'VM_STATE_TERMINATED')) continue;
    const anyFailed = vms.some(vm => vm.state === 'VM_STATE_FAILED' || vm.state === 'VM_STATE_PREEMPTED');
    const allReady = vms.every(vm => vm.state === 'VM_STATE_READY');
    if (anyFailed) {
      counts.failed++;
    } else if (allReady) {
      counts.ready++;
    } else if (vms.some(vm => vm.state === 'VM_STATE_REQUESTING')) {
      counts.requesting++;
    } else if (vms.some(vm => vm.state === 'VM_STATE_INITIALIZING')) {
      counts.initializing++;
    } else if (vms.some(vm => vm.state === 'VM_STATE_BOOTING')) {
      counts.booting++;
    }
  }
  return counts;
}

function groupStatusText(group, counts) {
  if (group.backoffUntilMs && parseInt(group.backoffUntilMs) > Date.now()) return ['backoff', 'Backoff'];
  if (counts.requesting > 0 || counts.booting > 0 || counts.initializing > 0) return ['available', 'Scaling Up'];
  if ((group.currentDemand || 0) > (counts.ready || 0)) return ['backoff', 'Pending'];
  if (counts.ready > 0) return ['available', 'Available'];
  return ['disabled', 'Idle'];
}

function AutoscalerLogs() {
  const [logs, setLogs] = useState('Loading logs...');

  useEffect(() => {
    controllerRpc('GetProcessLogs', { prefix: 'iris.cluster.vm', limit: 200 })
      .then(data => {
        const records = data.records || [];
        if (records.length > 0) {
          setLogs(records.map(l => l.message).join('\n'));
        } else {
          setLogs('No recent autoscaler logs');
        }
      })
      .catch(e => setLogs('Failed to load logs: ' + e.message));
  }, []);

  return html`
    <h3>Autoscaler Logs</h3>
    <pre style="background:white;padding:15px;border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,0.12);border:1px solid #d1d9e0;max-height:400px;overflow-y:auto;font-size:12px;white-space:pre-wrap">${logs}</pre>
    <div style="margin-top:8px">
      <a href="/#logs" style="color:#0969da;text-decoration:none;font-size:13px">View full controller logs \u2192</a>
    </div>`;
}

function formatResources(resources) {
  if (!resources) return '-';
  const cpu = resources.cpu || 0;
  const mem = resources.memoryBytes ? `${(resources.memoryBytes / (1024 ** 3)).toFixed(1)}GB` : '-';
  const disk = resources.diskBytes ? `${(resources.diskBytes / (1024 ** 3)).toFixed(1)}GB` : '-';
  const gpu = resources.gpuCount || 0;
  const tpu = resources.tpuCount || 0;
  return `cpu=${cpu}, mem=${mem}, disk=${disk}, gpu=${gpu}, tpu=${tpu}`;
}

function taskIdToJob(taskId) {
  if (!taskId) return 'unknown';
  const idx = taskId.lastIndexOf('/');
  if (idx <= 0) return taskId;
  return taskId.slice(0, idx);
}

function formatReasonCounts(reasonCounts) {
  const entries = Object.entries(reasonCounts || {});
  if (entries.length === 0) return '-';
  return entries.map(([reason, count]) => `${reason} (${count})`).join(', ');
}

function formatActionTime(timestamp) {
  const value = timestampFromProto(timestamp);
  if (!value) return '-';
  return new Date(value).toLocaleTimeString();
}

export function AutoscalerTab({ autoscaler }) {
  if (!autoscaler || autoscaler.enabled === false) {
    return html`
      <div class="autoscaler-status">
        <div class="status-item">
          <span class="status-indicator disabled"></span>
          <span>Status: <strong>Disabled</strong></span>
        </div>
        <div class="status-item"><span>Last Evaluation: <strong>-</strong></span></div>
      </div>
      <h3>Scale Groups</h3>
      <table class="scale-groups-table">
        <thead><tr><th>Group</th><th>Booting</th><th>Init</th><th>Ready</th><th>Failed</th><th>Demand</th><th>Status</th></tr></thead>
      </table>
      <h3>Recent Actions</h3>
      <div class="actions-log"><div class="action-entry">Autoscaler not configured</div></div>`;
  }

  const groups = autoscaler.groups || [];
  const actions = (autoscaler.recentActions || []).slice().reverse();
  const routing = autoscaler.lastRoutingDecision || null;
  const routedEntries = routing ? (routing.routedEntries || {}) : {};
  const groupToLaunch = routing ? (routing.groupToLaunch || {}) : {};
  const groupReasons = routing ? (routing.groupReasons || {}) : {};
  const unmetEntries = routing ? (routing.unmetEntries || []) : [];
  const routedGroups = Object.keys(routedEntries);
  const launchGroups = Object.keys(groupToLaunch);
  const allGroups = Array.from(new Set([...routedGroups, ...launchGroups]));

  return html`
    <div class="autoscaler-status">
      <div class="status-item">
        <span class="status-indicator active"></span>
        <span>Status: <strong>Active</strong></span>
      </div>
      <div class="status-item">
        <span>Last Evaluation: <strong>${autoscaler.lastEvaluationMs ? formatRelativeTime(parseInt(autoscaler.lastEvaluationMs)) : '-'}</strong></span>
      </div>
    </div>

    <h3>Routing Decision</h3>
    ${routing ? html`
      <table class="scale-groups-table">
        <thead><tr><th>Group</th><th>Assigned</th><th>Launch</th><th>Reason</th></tr></thead>
        <tbody>
          ${allGroups.length === 0 ? html`
            <tr><td colSpan="4">No routing decisions</td></tr>
          ` : allGroups.map(name => {
            const entries = (routedEntries[name] && routedEntries[name].entries) ? routedEntries[name].entries : [];
            const toLaunch = groupToLaunch[name] || 0;
            const reason = groupReasons[name] || '-';
            return html`<tr>
              <td><strong>${name}</strong></td>
              <td>${entries.length}</td>
              <td>${toLaunch}</td>
              <td>${reason}</td>
            </tr>`;
          })}
        </tbody>
      </table>
    ` : html`<div class="action-entry">No routing decision yet</div>`}

    ${routing && unmetEntries.length > 0 && html`
      <h3>Unmet Demand</h3>
      <table class="scale-groups-table">
        <thead><tr><th>Job</th><th>Reasons</th><th>Tasks</th><th>Example Task</th><th>Accelerator</th><th>Resources</th></tr></thead>
        <tbody>
          ${(() => {
            const byJob = new Map();
            for (const u of unmetEntries) {
              const entry = u.entry || {};
              const reason = u.reason || 'unknown';
              const accel = `${entry.acceleratorType || 'UNKNOWN'}${entry.acceleratorVariant ? ':' + entry.acceleratorVariant : ''}`;
              const resourceText = formatResources(entry.resources);
              const taskIds = entry.taskIds || [];
              if (taskIds.length === 0) {
                const job = 'unknown';
                if (!byJob.has(job)) {
                  byJob.set(job, {
                    job,
                    taskCount: 0,
                    exampleTask: null,
                    reasonCounts: {},
                    accelerators: new Set(),
                    resources: new Set(),
                  });
                }
                const row = byJob.get(job);
                row.reasonCounts[reason] = (row.reasonCounts[reason] || 0) + 1;
                row.accelerators.add(accel);
                row.resources.add(resourceText);
                continue;
              }
              for (const taskId of taskIds) {
                const job = taskIdToJob(taskId);
                if (!byJob.has(job)) {
                  byJob.set(job, {
                    job,
                    taskCount: 0,
                    exampleTask: null,
                    reasonCounts: {},
                    accelerators: new Set(),
                    resources: new Set(),
                  });
                }
                const row = byJob.get(job);
                row.taskCount += 1;
                if (!row.exampleTask) row.exampleTask = taskId;
                row.reasonCounts[reason] = (row.reasonCounts[reason] || 0) + 1;
                row.accelerators.add(accel);
                row.resources.add(resourceText);
              }
            }
            const rows = Array.from(byJob.values()).sort((a, b) => a.job.localeCompare(b.job));
            return rows.map(row => {
              const accelText = row.accelerators.size === 1 ? [...row.accelerators][0] : 'mixed';
              const resourceText = row.resources.size === 1 ? [...row.resources][0] : 'mixed';
              return html`<tr>
                <td><strong>${row.job}</strong></td>
                <td>${formatReasonCounts(row.reasonCounts)}</td>
                <td>${row.taskCount || '-'}</td>
                <td>${row.exampleTask || '-'}</td>
                <td>${accelText}</td>
                <td>${resourceText}</td>
              </tr>`;
            });
          })()}
        </tbody>
      </table>
    `}

    <h3>Scale Groups</h3>
    <table class="scale-groups-table">
      <thead><tr><th>Group</th><th>Booting</th><th>Init</th><th>Ready</th><th>Failed</th><th>Demand</th><th>Status</th></tr></thead>
      <tbody>
        ${groups.map(g => {
          const counts = computeSliceStateCounts(g.slices || []);
          const [statusClass, statusText] = groupStatusText(g, counts);
          return html`<tr>
            <td><strong>${g.name}</strong></td>
            <td>${counts.booting || 0}</td>
            <td>${counts.initializing || 0}</td>
            <td>${counts.ready || 0}</td>
            <td>${counts.failed || 0}</td>
            <td>${g.currentDemand || 0}</td>
            <td><span class="group-status"><span class=${'group-status-dot ' + statusClass}></span> ${statusText}</span></td>
          </tr>`;
        })}
      </tbody>
    </table>

    <h3>Recent Actions</h3>
    <div class="actions-log">
      ${actions.length === 0
        ? html`<div class="action-entry">No recent actions</div>`
        : actions.map(a => {
            const time = formatActionTime(a.timestamp);
            const actionType = a.actionType || 'unknown';
            const sliceInfo = a.sliceId ? ' [' + a.sliceId.slice(0, 20) + '...]' : '';
            const status = a.status || 'completed';
            const statusClass = status === 'pending' ? 'status-pending' :
                                status === 'failed' ? 'status-failed' : 'status-succeeded';
            return html`<div class="action-entry">
              <span class="action-time">${time}</span>
              <span class=${'action-type ' + actionType}>${actionType.replace('_', ' ')}</span>
              ${status !== 'completed' && html`<span class=${statusClass} style="margin-left:5px">[${status}]</span>`}
              ${' '}<strong>${a.scaleGroup}</strong>${sliceInfo}
              ${a.reason ? ' - ' + a.reason : ''}
            </div>`;
          })
      }
    </div>

    <${AutoscalerLogs} />`;
}
