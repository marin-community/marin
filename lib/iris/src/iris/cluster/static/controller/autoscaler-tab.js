import { h } from 'preact';
import { useState, useEffect } from 'preact/hooks';
import htm from 'htm';
import { formatRelativeTime } from '/static/shared/utils.js';
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
    controllerRpc('GetControllerLogs', { prefix: 'iris.cluster.vm', limit: 200 })
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
      <a href="/logs" style="color:#0969da;text-decoration:none;font-size:13px">View full controller logs \u2192</a>
    </div>`;
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
            const time = new Date(parseInt(a.timestampMs)).toLocaleTimeString();
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
