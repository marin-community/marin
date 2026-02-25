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

/** Build a map of group name -> ScaleGroupStatus from autoscaler.groups */
function buildGroupIndex(groups) {
  const index = {};
  for (const g of groups) {
    if (g.name) index[g.name] = g;
  }
  return index;
}

/** Sum slice state counts across all groups. Returns {ready: N, booting: N, ...} */
function aggregateSliceCounts(groups) {
  const totals = {};
  for (const g of groups) {
    for (const [state, count] of Object.entries(g.sliceStateCounts || {})) {
      totals[state] = (totals[state] || 0) + (count || 0);
    }
  }
  return totals;
}

/** Render compact slice badges like "2R 1B" with color coding */
function SliceBadges({ counts }) {
  if (!counts) return html`<span style="color:#8c959f">-</span>`;

  const order = [
    ['ready', 'R', 'ready'],
    ['requesting', 'Q', 'requesting'],
    ['booting', 'B', 'booting'],
    ['initializing', 'I', 'initializing'],
    ['failed', 'F', 'failed'],
  ];

  const badges = [];
  for (const [key, letter, cls] of order) {
    const n = counts[key] || 0;
    if (n > 0) {
      badges.push(html`<span class=${'slice-badge ' + cls}>${n}${letter}</span>`);
    }
  }

  if (badges.length === 0) return html`<span style="color:#8c959f">-</span>`;
  return html`<span>${badges}</span>`;
}

/** Format aggregate slice counts for the status bar, e.g. "5 (3 ready, 1 booting, 1 initializing)" */
function formatSliceSummary(totals) {
  const total = Object.values(totals).reduce((a, b) => a + b, 0);
  if (total === 0) return '0';

  const order = ['ready', 'requesting', 'booting', 'initializing', 'failed'];
  const parts = [];
  for (const state of order) {
    const n = totals[state] || 0;
    if (n > 0) parts.push(`${n} ${state}`);
  }
  return `${total} (${parts.join(', ')})`;
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
  const groupIndex = buildGroupIndex(groups);
  const actions = (autoscaler.recentActions || []).slice().reverse();
  const routing = autoscaler.lastRoutingDecision || null;
  const routedEntries = routing ? (routing.routedEntries || {}) : {};
  const groupToLaunch = routing ? (routing.groupToLaunch || {}) : {};
  const groupReasons = routing ? (routing.groupReasons || {}) : {};
  const unmetEntries = routing ? (routing.unmetEntries || []) : [];

  const sliceTotals = aggregateSliceCounts(groups);
  const totalSlices = Object.values(sliceTotals).reduce((a, b) => a + b, 0);
  const onlineGroups = groups.filter(g => {
    const counts = g.sliceStateCounts || {};
    return Object.values(counts).reduce((a, b) => a + b, 0) > 0;
  }).length;

  return html`
    <div class="autoscaler-status" style="flex-wrap:wrap">
      ${(() => {
        const lastEvalMs = timestampFromProto(autoscaler.lastEvaluation);
        const groupStatuses = routing ? (routing.groupStatuses || []) : [];
        const totalDemand = groups.reduce((n, g) => n + (g.currentDemand || 0), 0);
        const launchPlanned = groupStatuses.length > 0
          ? groupStatuses.reduce((n, gs) => n + (gs.launch || 0), 0)
          : Object.values(groupToLaunch).reduce((n, v) => n + (v || 0), 0);
        return html`
          <div class="status-item">
            <span class="status-indicator ${totalSlices > 0 ? 'active' : 'disabled'}"></span>
            <span>Groups: <strong>${onlineGroups} / ${groups.length} online</strong></span>
          </div>
          <div class="status-item">
            <span>Slices: <strong>${formatSliceSummary(sliceTotals)}</strong></span>
          </div>
          <div class="status-item">
            <span>Demand: <strong>${totalDemand}</strong></span>
          </div>
          <div class="status-item">
            <span>Launch Planned: <strong>${launchPlanned}</strong></span>
          </div>
          ${unmetEntries.length > 0 && html`
            <div class="status-item">
              <span>Unmet: <strong class="status-failed">${unmetEntries.length}</strong></span>
            </div>
          `}
          <div class="status-item">
            <span>Last Decision: <strong>${lastEvalMs ? formatRelativeTime(lastEvalMs) : '-'}</strong></span>
          </div>
        `;
      })()}
    </div>

    <h3>Waterfall Routing</h3>
    ${routing ? html`
      <table class="scale-groups-table">
        <thead><tr>
          <th>Priority</th><th>Group</th><th>Slices</th><th>Demand</th>
          <th>Assigned</th><th>Launch</th><th>Decision</th><th>Reason</th>
        </tr></thead>
        <tbody>
          ${(() => {
            const statuses = (routing.groupStatuses || []).slice().sort((a, b) => {
              const pa = a.priority || 100;
              const pb = b.priority || 100;
              if (pa !== pb) return pa - pb;
              return (a.group || '').localeCompare(b.group || '');
            });

            const rows = statuses.length > 0 ? statuses : Object.keys(routedEntries).concat(Object.keys(groupToLaunch))
              .filter((v, i, arr) => arr.indexOf(v) === i)
              .sort()
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

            return rows.map(gs => {
              const groupName = gs.group || '';
              const groupStatus = groupIndex[groupName];
              const sliceCounts = groupStatus ? (groupStatus.sliceStateCounts || {}) : {};
              const totalGroupSlices = Object.values(sliceCounts).reduce((a, b) => a + b, 0);
              const demand = groupStatus ? (groupStatus.currentDemand || 0) : 0;
              const failures = groupStatus ? (groupStatus.consecutiveFailures || 0) : 0;
              const decision = (gs.decision || 'idle').replace('_', ' ');
              const decisionRaw = gs.decision || 'idle';
              const decisionClass = 'decision-' + decisionRaw;
              const isInactive = totalGroupSlices === 0 && decisionRaw === 'idle';

              let reason = gs.reason || '';
              if (groupStatus && (groupStatus.availabilityStatus === 'backoff' || groupStatus.availabilityStatus === 'quota_exceeded')) {
                const blockedMs = timestampFromProto(groupStatus.blockedUntil);
                if (blockedMs && blockedMs > Date.now()) {
                  const secsLeft = Math.ceil((blockedMs - Date.now()) / 1000);
                  reason = reason ? reason + ` (unblocks in ${secsLeft}s)` : `unblocks in ${secsLeft}s`;
                }
              }

              return html`<tr class=${isInactive ? 'row-inactive' : ''}>
                <td>${gs.priority || 100}</td>
                <td>
                  <strong>${groupName}</strong>
                  ${failures > 0 && html`<span class="failure-badge">\u26a0 ${failures} fail${failures > 1 ? 's' : ''}</span>`}
                </td>
                <td><${SliceBadges} counts=${sliceCounts} /></td>
                <td>${demand || ''}</td>
                <td>${gs.assigned || 0}</td>
                <td>${gs.launch || 0}</td>
                <td><span class=${decisionClass}>${decision}</span></td>
                <td>${reason || '-'}</td>
              </tr>`;
            });
          })()}
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
