import { h } from 'preact';
import { useState, useEffect } from 'preact/hooks';
import htm from 'htm';
import { formatRelativeTime, timestampFromProto } from '/static/shared/utils.js';
import { controllerRpc } from '/static/shared/rpc.js';

const html = htm.bind(h);

function AutoscalerLogs() {
  const [logs, setLogs] = useState('Loading logs...');

  useEffect(() => {
    controllerRpc('GetProcessLogs', { prefix: 'iris.cluster.controller.autoscaler', limit: 200 })
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
  const cpu = (resources.cpuMillicores || 0) / 1000;
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
  return entries.map(([reason, count]) => {
    const display = reason.replace(/^[a-z_]+:\s*/, '');
    return `${display} (${count})`;
  }).join(', ');
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
      ${(() => {
        const lastEvalMs = timestampFromProto(autoscaler.lastEvaluation);
        const groupStatuses = routing ? (routing.groupStatuses || []) : [];
        const totalDemand = groups.reduce((n, g) => n + (g.currentDemand || 0), 0);
        const launchPlanned = groupStatuses.length > 0
          ? groupStatuses.reduce((n, gs) => n + (gs.launch || 0), 0)
          : Object.values(groupToLaunch).reduce((n, v) => n + (v || 0), 0);
        return html`
          <div class="status-item">
            <span class="status-indicator active"></span>
            <span>Demand: <strong>${totalDemand}</strong></span>
          </div>
          <div class="status-item">
            <span>Launch Planned: <strong>${launchPlanned}</strong></span>
          </div>
          <div class="status-item">
            <span>Unmet: <strong>${unmetEntries.length}</strong></span>
          </div>
          <div class="status-item">
            <span>Last Decision: <strong>${lastEvalMs ? formatRelativeTime(lastEvalMs) : '-'}</strong></span>
          </div>
        `;
      })()}
    </div>

    <h3>Waterfall Routing</h3>
    ${routing ? html`
      <table class="scale-groups-table">
        <thead><tr><th>Priority</th><th>Group</th><th>Assigned</th><th>Launch</th><th>Decision</th><th>Reason</th></tr></thead>
        <tbody>
          ${(() => {
            const statuses = (routing.groupStatuses || []).slice().sort((a, b) => {
              const pa = a.priority || 100;
              const pb = b.priority || 100;
              if (pa !== pb) return pa - pb;
              return (a.group || '').localeCompare(b.group || '');
            });
            if (statuses.length === 0) {
              return allGroups
                .slice()
                .sort((a, b) => a.localeCompare(b))
                .map(name => {
                const entries = (routedEntries[name] && routedEntries[name].entries) ? routedEntries[name].entries : [];
                return {
                  priority: 100,
                  group: name,
                  assigned: entries.length,
                  launch: groupToLaunch[name] || 0,
                  decision: entries.length > 0 ? 'selected' : 'idle',
                  reason: groupReasons[name] || '',
                };
              });
            }
            return statuses;
          })().map(gs => {
            const decision = (gs.decision || 'idle').replace('_', ' ');
            return html`<tr>
              <td>${gs.priority || 100}</td>
              <td><strong>${gs.group}</strong></td>
              <td>${gs.assigned || 0}</td>
              <td>${gs.launch || 0}</td>
              <td>${decision}</td>
              <td>${gs.reason || '-'}</td>
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
