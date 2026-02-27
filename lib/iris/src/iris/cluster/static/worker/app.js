import { h, render } from 'preact';
import { useState, useEffect, useCallback, useRef } from 'preact/hooks';
import htm from 'htm';
import { stateToName, formatTimestamp, timestampFromProto } from '/static/shared/utils.js';
import { workerRpc } from '/static/shared/rpc.js';
import { Gauge, MetricCard, ResourceSection } from '/static/shared/components.js';

const html = htm.bind(h);

const AUTO_REFRESH_INTERVAL_MS = 5000;

function taskIndexFromId(taskId) {
  const last = taskId.split('/').pop();
  const parsed = Number.parseInt(last, 10);
  return Number.isNaN(parsed) ? null : parsed;
}

function jobIdFromTaskId(taskId) {
  // task_id format: "/job/path/0" -> job_id is "/job/path"
  const parts = taskId.split('/');
  parts.pop();
  return parts.join('/');
}

function shortId(id) {
  const last = id.split('/').pop();
  return last;
}

function StatsBar({ tasks }) {
  const running = tasks.filter(t => t.state === 'TASK_STATE_RUNNING').length;
  const assigned = tasks.filter(t => t.state === 'TASK_STATE_ASSIGNED').length;
  const pending = tasks.filter(t => t.state === 'TASK_STATE_PENDING').length;
  const building = tasks.filter(t => t.state === 'TASK_STATE_BUILDING').length;
  const completed = tasks.filter(t => t.state === 'TASK_STATE_SUCCEEDED').length;
  const failed = tasks.filter(t => t.state === 'TASK_STATE_FAILED').length;
  const killed = tasks.filter(t => t.state === 'TASK_STATE_KILLED').length;

  return html`<div class="metric-row">
    <${MetricCard} value=${running} label="Running" valueClass=${running > 0 ? 'accent' : ''} />
    <${MetricCard} value=${assigned} label="Assigned" valueClass=${assigned > 0 ? 'warning' : ''} />
    <${MetricCard} value=${building} label="Building" valueClass=${building > 0 ? 'purple' : ''} />
    <${MetricCard} value=${pending} label="Pending" valueClass=${pending > 0 ? 'warning' : ''} />
    <${MetricCard} value=${completed} label="Completed" />
    <${MetricCard} value=${failed} label="Failed" valueClass=${failed > 0 ? 'danger' : ''} />
    <${MetricCard} value=${killed} label="Killed" valueClass=${killed > 0 ? 'danger' : ''} />
  </div>`;
}

function AggregateResources({ tasks }) {
  const runningTasks = tasks.filter(t => t.state === 'TASK_STATE_RUNNING');
  const tasksWithResources = runningTasks.filter(t => t.resourceUsage);
  if (tasksWithResources.length === 0) return null;

  const totalMemMb = tasksWithResources.reduce((sum, t) => sum + (t.resourceUsage?.memoryMb || 0), 0);
  const totalPeakMemMb = tasksWithResources.reduce((sum, t) => sum + (t.resourceUsage?.memoryPeakMb || 0), 0);
  const avgCpu = tasksWithResources.length > 0
    ? tasksWithResources.reduce((sum, t) => sum + (t.resourceUsage?.cpuPercent || 0), 0) / tasksWithResources.length
    : 0;
  const totalDiskMb = tasksWithResources.reduce((sum, t) => sum + (t.resourceUsage?.diskMb || 0), 0);
  const totalProcesses = tasksWithResources.reduce((sum, t) => sum + (t.resourceUsage?.processCount || 0), 0);

  const title = 'Aggregate Resource Usage (' + tasksWithResources.length + ' running task' + (tasksWithResources.length !== 1 ? 's' : '') + ')';

  return html`<${ResourceSection} title=${title}>
    <${Gauge} label="Memory" value=${totalMemMb} max=${totalPeakMemMb || totalMemMb}
              format="raw" />
    <${Gauge} label="CPU (avg)" value=${Math.round(avgCpu)} max=${100}
              format="percent" />
    <${Gauge} label="Disk" value=${totalDiskMb} max=${totalDiskMb || 1}
              format="raw" />
    <div class="gauge">
      <span class="gauge-label">Processes</span>
      <span class="gauge-value">${totalProcesses}</span>
    </div>
  </${ResourceSection}>`;
}

function TaskRow({ task }) {
  const started = formatTimestamp(timestampFromProto(task.startedAt));
  const finished = formatTimestamp(timestampFromProto(task.finishedAt));
  const exitCode = task.exitCode !== null && task.exitCode !== undefined ? task.exitCode : '-';
  const attemptId = task.currentAttemptId || 0;
  const taskShort = shortId(task.taskId);
  const taskDisplay = attemptId > 0
    ? `${taskShort} (attempt ${attemptId})`
    : taskShort;
  const statusClass = 'status-' + stateToName(task.state);
  const res = task.resourceUsage || {};
  const taskIndex = taskIndexFromId(task.taskId);
  const jobId = jobIdFromTaskId(task.taskId);

  return html`<tr>
    <td><a href=${'/task/' + encodeURIComponent(task.taskId)} class="task-link" target="_blank">${taskDisplay}</a></td>
    <td>${jobId}</td>
    <td>${taskIndex ?? '-'}</td>
    <td class=${statusClass}>${task.state}</td>
    <td>${exitCode}</td>
    <td>${res.memoryMb || 0}/${res.memoryPeakMb || 0} MB</td>
    <td>${(res.cpuPercent || 0) + '%'}</td>
    <td>${started}</td>
    <td>${finished}</td>
    <td>${task.error || '-'}</td>
  </tr>`;
}

function App() {
  const [tasks, setTasks] = useState([]);
  const [error, setError] = useState(null);
  const intervalRef = useRef(null);

  const refresh = useCallback(async () => {
    try {
      const resp = await workerRpc('ListTasks', {});
      setTasks(resp.tasks || []);
      setError(null);
    } catch (e) {
      console.error('Failed to refresh:', e);
      setError('Failed to load tasks: ' + e.message);
    }
  }, []);

  useEffect(() => {
    refresh();
    intervalRef.current = setInterval(refresh, AUTO_REFRESH_INTERVAL_MS);
    return () => clearInterval(intervalRef.current);
  }, [refresh]);

  return html`
    <div style="display:flex;justify-content:space-between;align-items:center">
      <h1 style="flex:1">Iris Worker Dashboard</h1>
      <div style="display:flex;align-items:center;gap:8px">
        <span class="auto-refresh-badge">auto-refresh: 5s</span>
        <button onClick=${refresh}>\u21bb Refresh</button>
      </div>
    </div>
    ${error && html`<div class="error-message">${error}</div>`}

    <${StatsBar} tasks=${tasks} />
    <${AggregateResources} tasks=${tasks} />

    <h2>Tasks</h2>
    <table>
      <thead>
        <tr>
          <th>Task ID</th><th>Job ID</th><th>Index</th><th>Status</th><th>Exit</th>
          <th>Memory</th><th>CPU</th><th>Started</th><th>Finished</th><th>Error</th>
        </tr>
      </thead>
      <tbody>
        ${tasks.length === 0
          ? html`<tr><td colspan="10" style="text-align:center;padding:20px;color:#666">No tasks</td></tr>`
          : tasks.map(t => html`<${TaskRow} key=${t.taskId} task=${t} />`)}
      </tbody>
    </table>

    <div style="margin-top:20px">
      <a href="/logs" class="back-link">View Process Logs \u2192</a>
    </div>
  `;
}

render(html`<${App} />`, document.getElementById('root'));
