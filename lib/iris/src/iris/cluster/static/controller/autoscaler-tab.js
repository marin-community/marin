import { h } from 'preact';
import { useState, useEffect } from 'preact/hooks';
import htm from 'htm';
import { formatRelativeTime, formatVmState, timestampFromProto } from '/static/shared/utils.js';
import { controllerRpc } from '/static/shared/rpc.js';
import { LogViewer } from '/static/shared/log-viewer.js';

const html = htm.bind(h);

function AutoscalerLogs() {
  return html`
    <${LogViewer}
      rpc=${controllerRpc}
      source="/process"
      title="Autoscaler Logs"
      defaultRegex="autoscaler"
      defaultMaxLines=${200}
      showControls=${false}
    />
    <div style="margin-top:8px">
      <a href="/#logs" style="color:#0969da;text-decoration:none;font-size:13px">View full controller logs \u2192</a>
    </div>
  `;
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

/** Reservation task IDs use the exact format `{job}:reservation:{digits}`. */
const RESERVATION_RE = /^(.+):reservation:\d+$/;

function taskIdToJob(taskId) {
  if (!taskId) return 'unknown';
  const rsvMatch = taskId.match(RESERVATION_RE);
  if (rsvMatch) return rsvMatch[1];
  const idx = taskId.lastIndexOf('/');
  if (idx <= 0) return taskId;
  return taskId.slice(0, idx);
}

function isReservationEntry(entry) {
  const taskIds = entry.taskIds || [];
  return taskIds.length > 0 && RESERVATION_RE.test(taskIds[0]);
}

/**
 * Aggregate routed DemandEntries by job.
 *
 * Each element in `entries` is one DemandEntry (one unit of routed demand).
 * Coscheduled entries carry a `coscheduleGroupId`; otherwise the job name is
 * derived from the first task ID.  Counts reflect the number of demand entries
 * (not individual task IDs) so they match the demand total shown in the main row.
 */
function aggregateEntriesByJob(entries) {
  const byJob = new Map();
  for (const entry of entries) {
    const isRsv = isReservationEntry(entry);
    const job = entry.coscheduleGroupId || taskIdToJob((entry.taskIds || [])[0]);
    if (!byJob.has(job)) byJob.set(job, { job, taskEntries: 0, reservationEntries: 0 });
    const row = byJob.get(job);
    if (isRsv) row.reservationEntries++; else row.taskEntries++;
  }
  return Array.from(byJob.values()).sort((a, b) => a.job.localeCompare(b.job));
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

/** Count idle slices across all groups */
function countIdleSlices(groups) {
  let idle = 0;
  for (const g of groups) {
    for (const slice of (g.slices || [])) {
      if (slice.idle) idle++;
    }
  }
  return idle;
}

/** Render compact slice badges like "2R 1B" with color coding, plus idle count */
function SliceBadges({ counts, idleCount }) {
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

  return html`<span>
    ${badges}
    ${idleCount > 0 && html`<span class="slice-idle-badge idle" title="${idleCount} idle slice${idleCount > 1 ? 's' : ''} (eligible for reclamation)">${idleCount} idle</span>`}
  </span>`;
}

/** Render an availability-state badge below the group name when state is noteworthy. */
function GroupStatusBadge({ group }) {
  if (!group) return null;

  const status = group.availabilityStatus;
  const blockedMs = timestampFromProto(group.blockedUntil);
  const cooldownMs = timestampFromProto(group.scaleUpCooldownUntil);
  const now = Date.now();

  if (status === 'requesting') {
    return html`<div><span class="group-status-badge requesting">in-flight</span></div>`;
  }
  if (status === 'backoff') {
    const label = blockedMs && blockedMs > now
      ? `backoff ${Math.ceil((blockedMs - now) / 1000)}s`
      : 'backoff';
    return html`<div><span class="group-status-badge backoff">${label}</span></div>`;
  }
  if (status === 'quota_exceeded') {
    const label = blockedMs && blockedMs > now
      ? `quota exceeded ${Math.ceil((blockedMs - now) / 1000)}s`
      : 'quota exceeded';
    return html`<div><span class="group-status-badge quota_exceeded">${label}</span></div>`;
  }
  if (status === 'at_capacity') {
    return html`<div><span class="group-status-badge at_capacity">at capacity</span></div>`;
  }

  // "available" but in cooldown — show cooldown countdown
  if (cooldownMs && cooldownMs > now) {
    const secs = Math.ceil((cooldownMs - now) / 1000);
    return html`<div><span class="group-status-badge cooldown">cooldown ${secs}s</span></div>`;
  }

  return null;
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

/** Format idle duration from last_active timestamp */
function formatIdleSince(lastActiveProto) {
  const ms = timestampFromProto(lastActiveProto);
  if (!ms) return 'never active';
  const elapsed = Date.now() - ms;
  if (elapsed < 60000) return `${Math.floor(elapsed / 1000)}s`;
  if (elapsed < 3600000) return `${Math.floor(elapsed / 60000)}m`;
  return `${Math.floor(elapsed / 3600000)}h ${Math.floor((elapsed % 3600000) / 60000)}m`;
}

/** Render per-slice detail rows for a group (expanded from the Slices column) */
function SliceDetailRows({ slices, idleThresholdMs }) {
  if (!slices || slices.length === 0) return null;

  const thresholdLabel = idleThresholdMs
    ? (idleThresholdMs >= 60000 ? `${Math.floor(idleThresholdMs / 60000)}m` : `${Math.floor(idleThresholdMs / 1000)}s`)
    : '?';

  return slices.map(slice => {
    const vmCount = (slice.vms || []).length;
    const state = (slice.vms || []).length > 0 ? formatVmState(slice.vms[0].state) : 'unknown';
    const isReady = state === 'ready';
    const isIdle = slice.idle;
    const createdMs = timestampFromProto(slice.createdAt);
    const age = createdMs ? formatRelativeTime(createdMs) : '-';

    let statusBadge;
    if (isIdle) {
      const idleDur = formatIdleSince(slice.lastActive);
      statusBadge = html`<span class="slice-idle-badge idle" title="Idle for ${idleDur}, threshold ${thresholdLabel}">idle ${idleDur}</span>`;
    } else if (isReady) {
      statusBadge = html`<span class="slice-idle-badge active">active</span>`;
    } else {
      statusBadge = html`<span class="slice-idle-badge booting">${state}</span>`;
    }

    return html`<div class="slice-row">
      <span class="slice-id" title=${slice.sliceId}>${slice.sliceId}</span>
      <span class="slice-vm-count">${vmCount} vm${vmCount !== 1 ? 's' : ''}</span>
      ${statusBadge}
      <span style="color:#8c959f;font-size:11px">${age}</span>
    </div>`;
  });
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

  const [expandedGroups, setExpandedGroups] = useState(new Set());
  const [expandedSlices, setExpandedSlices] = useState(new Set());
  function toggleGroup(name) {
    setExpandedGroups(prev => {
      const next = new Set(prev);
      next.has(name) ? next.delete(name) : next.add(name);
      return next;
    });
  }
  function toggleSlices(name) {
    setExpandedSlices(prev => {
      const next = new Set(prev);
      next.has(name) ? next.delete(name) : next.add(name);
      return next;
    });
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
  const totalIdle = countIdleSlices(groups);
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
          ${totalIdle > 0 && html`
            <div class="status-item">
              <span>Idle: <strong style="color:#9a6700">${totalIdle}</strong></span>
            </div>
          `}
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

            return rows.flatMap(gs => {
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

              const entries = (routedEntries[groupName] && routedEntries[groupName].entries) || [];
              const jobRows = demand > 0 ? aggregateEntriesByJob(entries) : [];
              const hasSources = jobRows.length > 0;
              const demandExpanded = expandedGroups.has(groupName);
              const slicesExpanded = expandedSlices.has(groupName);
              const groupSlices = groupStatus ? (groupStatus.slices || []) : [];
              const hasSlices = groupSlices.length > 0;

              // Count idle slices in this group
              const groupIdleCount = groupSlices.filter(s => s.idle).length;
              const idleThresholdMs = groupStatus ? parseInt(groupStatus.idleThresholdMs || '0', 10) : 0;

              const mainRow = html`<tr class=${isInactive ? 'row-inactive' : ''}>
                <td>${gs.priority || 100}</td>
                <td>
                  <strong>${groupName}</strong>
                  ${failures > 0 && html`<span class="failure-badge">\u26a0 ${failures} fail${failures > 1 ? 's' : ''}</span>`}
                  <${GroupStatusBadge} group=${groupStatus} />
                </td>
                <td>${hasSlices
                  ? html`<span class="slice-toggle" onClick=${() => toggleSlices(groupName)}>
                      <span style="font-size:10px;color:#57606a">${slicesExpanded ? '\u25BC' : '\u25B6'}</span>${' '}
                      <${SliceBadges} counts=${sliceCounts} idleCount=${groupIdleCount} />
                    </span>`
                  : html`<${SliceBadges} counts=${sliceCounts} idleCount=${0} />`}</td>
                <td>${hasSources
                  ? html`<span class="demand-toggle" onClick=${() => toggleGroup(groupName)}
                      >${demandExpanded ? '\u25BC' : '\u25B6'} ${demand}</span>`
                  : (demand || '')}</td>
                <td>${gs.assigned || 0}</td>
                <td>${gs.launch || 0}</td>
                <td><span class=${decisionClass}>${decision}</span></td>
                <td>${reason || '-'}</td>
              </tr>`;

              const extraRows = [];

              // Slice detail row (expanded from Slices column)
              if (slicesExpanded && hasSlices) {
                extraRows.push(html`<tr class="slice-detail-row">
                  <td colspan="8">
                    <div class="slice-detail-content">
                      <${SliceDetailRows} slices=${groupSlices} idleThresholdMs=${idleThresholdMs} />
                    </div>
                  </td>
                </tr>`);
              }

              // Demand detail row (expanded from Demand column)
              if (demandExpanded && hasSources) {
                const MAX_JOBS = 5;
                const shownJobs = jobRows.slice(0, MAX_JOBS);
                const overflow = jobRows.length - MAX_JOBS;

                extraRows.push(html`<tr class="demand-detail-row">
                  <td colspan="8">
                    <div class="demand-detail-content">
                      ${shownJobs.map(j => {
                        const parts = [];
                        if (j.taskEntries > 0) parts.push(`${j.taskEntries} task${j.taskEntries > 1 ? 's' : ''}`);
                        if (j.reservationEntries > 0) parts.push(`${j.reservationEntries} rsv`);
                        return html`<div class="demand-detail-job">
                          <span class="demand-detail-job-name">${j.job}</span>
                          <span class="demand-detail-job-counts">${parts.join(', ')}</span>
                        </div>`;
                      })}
                      ${overflow > 0 && html`<div class="demand-detail-overflow">+${overflow} more</div>`}
                    </div>
                  </td>
                </tr>`);
              }

              return [mainRow, ...extraRows];
            });
          })()}
        </tbody>
      </table>
    ` : html`<div class="action-entry">No routing decision yet</div>`}

    ${routing && unmetEntries.length > 0 && html`
      <h3>Unmet Demand</h3>
      <table class="scale-groups-table">
        <thead><tr><th>Job</th><th>Reasons</th><th>Entries</th><th>Example Task</th><th>Accelerator</th><th>Resources</th></tr></thead>
        <tbody>
          ${(() => {
            // Each element in unmetEntries is one UnmetDemand wrapping one DemandEntry.
            // Aggregate by job, counting entries (not individual task IDs).
            const byJob = new Map();
            for (const u of unmetEntries) {
              const entry = u.entry || {};
              const reason = u.reason || 'unknown';
              const accel = `${entry.acceleratorType || 'UNKNOWN'}${entry.acceleratorVariant ? ':' + entry.acceleratorVariant : ''}`;
              const resourceText = formatResources(entry.resources);
              const taskIds = entry.taskIds || [];
              const job = entry.coscheduleGroupId || taskIdToJob(taskIds[0]) || 'unknown';
              if (!byJob.has(job)) {
                byJob.set(job, {
                  job,
                  entryCount: 0,
                  exampleTask: null,
                  reasonCounts: {},
                  accelerators: new Set(),
                  resources: new Set(),
                });
              }
              const row = byJob.get(job);
              row.entryCount += 1;
              if (!row.exampleTask && taskIds.length > 0) row.exampleTask = taskIds[0];
              row.reasonCounts[reason] = (row.reasonCounts[reason] || 0) + 1;
              row.accelerators.add(accel);
              row.resources.add(resourceText);
            }
            const rows = Array.from(byJob.values()).sort((a, b) => a.job.localeCompare(b.job));
            return rows.map(row => {
              const accelText = row.accelerators.size === 1 ? [...row.accelerators][0] : 'mixed';
              const resourceText = row.resources.size === 1 ? [...row.resources][0] : 'mixed';
              return html`<tr>
                <td><strong>${row.job}</strong></td>
                <td>${formatReasonCounts(row.reasonCounts)}</td>
                <td>${row.entryCount || '-'}</td>
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
