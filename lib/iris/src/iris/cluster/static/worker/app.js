import { h, render } from 'preact';
import { useState, useEffect, useCallback } from 'preact/hooks';
import htm from 'htm';
import { stateToName, formatTimestamp, timestampFromProto } from '/static/shared/utils.js';
import { workerRpc } from '/static/shared/rpc.js';

const html = htm.bind(h);

function taskIndexFromId(taskId) {
  const last = taskId.split('/').pop();
  const parsed = Number.parseInt(last, 10);
  return Number.isNaN(parsed) ? null : parsed;
}

function jobIdFromTaskId(taskId) {
  // task_id format: "/job/path/0" -> job_id is "/job/path"
  const parts = taskId.split('/');
  parts.pop(); // remove task index
  return parts.join('/');
}

function StatsBar({ tasks }) {
  const running = tasks.filter(t => t.state === 'TASK_STATE_RUNNING').length;
  const pending = tasks.filter(t => t.state === 'TASK_STATE_PENDING').length;
  const assigned = tasks.filter(t => t.state === 'TASK_STATE_ASSIGNED').length;
  const building = tasks.filter(t => t.state === 'TASK_STATE_BUILDING').length;
  const completed = tasks.filter(t =>
    t.state === 'TASK_STATE_SUCCEEDED' || t.state === 'TASK_STATE_FAILED' || t.state === 'TASK_STATE_KILLED'
  ).length;

  return html`<div>
    <b>Running:</b> ${running} | <b>Assigned:</b> ${assigned} | <b>Pending:</b> ${pending} |
    <b>Building:</b> ${building} | <b>Completed:</b> ${completed}
  </div>`;
}

function TaskRow({ task }) {
  const started = formatTimestamp(timestampFromProto(task.startedAt));
  const finished = formatTimestamp(timestampFromProto(task.finishedAt));
  const exitCode = task.exitCode !== null && task.exitCode !== undefined ? task.exitCode : '-';
  const attemptId = task.currentAttemptId || 0;
  const taskDisplay = attemptId > 0
    ? `${task.taskId.slice(0, 12)}... (attempt ${attemptId})`
    : `${task.taskId.slice(0, 12)}...`;
  const statusClass = 'status-' + stateToName(task.state);
  const res = task.resourceUsage || {};
  const taskIndex = taskIndexFromId(task.taskId);
  const jobId = jobIdFromTaskId(task.taskId);

  return html`<tr>
    <td><a href=${'/task/' + encodeURIComponent(task.taskId)} class="task-link" target="_blank">${taskDisplay}</a></td>
    <td>${jobId.slice(0, 12)}...</td>
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

  // Initial load
  useEffect(() => { refresh(); }, [refresh]);

  return html`
    <h1>Iris Worker Dashboard</h1>
    ${error && html`<div class="error-message">${error}</div>`}
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
      <${StatsBar} tasks=${tasks} />
      <button onClick=${refresh} style="font-size:14px">\u21bb Refresh</button>
    </div>
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
  `;
}

render(html`<${App} />`, document.getElementById('root'));
