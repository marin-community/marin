import { h, render } from 'preact';
import { useState, useEffect } from 'preact/hooks';
import htm from 'htm';
import { stateToName, formatTimestamp } from '/static/shared/utils.js';
import { workerRpc } from '/static/shared/rpc.js';

const html = htm.bind(h);

function StatsBar({ tasks }) {
  const running = tasks.filter(t => t.state === 'TASK_STATE_RUNNING').length;
  const pending = tasks.filter(t => t.state === 'TASK_STATE_PENDING').length;
  const building = tasks.filter(t => t.state === 'TASK_STATE_BUILDING').length;
  const completed = tasks.filter(t =>
    t.state === 'TASK_STATE_SUCCEEDED' || t.state === 'TASK_STATE_FAILED' || t.state === 'TASK_STATE_KILLED'
  ).length;

  return html`<div>
    <b>Running:</b> ${running} | <b>Pending:</b> ${pending} |
    <b>Building:</b> ${building} | <b>Completed:</b> ${completed}
  </div>`;
}

function TaskRow({ task }) {
  const started = formatTimestamp(task.startedAtMs);
  const finished = formatTimestamp(task.finishedAtMs);
  const exitCode = task.exitCode !== null && task.exitCode !== undefined ? task.exitCode : '-';
  const attemptId = task.currentAttemptId || 0;
  const taskDisplay = attemptId > 0
    ? `${task.taskId.slice(0, 12)}... (attempt ${attemptId})`
    : `${task.taskId.slice(0, 12)}...`;
  const statusClass = 'status-' + stateToName(task.state);
  const res = task.resourceUsage || {};

  return html`<tr>
    <td><a href=${'/task/' + encodeURIComponent(task.taskId)} class="task-link" target="_blank">${taskDisplay}</a></td>
    <td>${task.jobId.slice(0, 8)}...</td>
    <td>${task.taskIndex}/${'?'}</td>
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

  async function refresh() {
    try {
      const resp = await workerRpc('ListTasks', {});
      setTasks(resp.tasks || []);
    } catch (e) {
      console.error('Failed to refresh:', e);
    }
  }

  useEffect(() => { refresh(); }, []);

  return html`
    <h1>Iris Worker Dashboard</h1>
    <${StatsBar} tasks=${tasks} />
    <h2>Tasks</h2>
    <table>
      <thead>
        <tr>
          <th>Task ID</th><th>Job ID</th><th>Index</th><th>Status</th><th>Exit</th>
          <th>Memory</th><th>CPU</th><th>Started</th><th>Finished</th><th>Error</th>
        </tr>
      </thead>
      <tbody>
        ${tasks.map(t => html`<${TaskRow} key=${t.taskId} task=${t} />`)}
      </tbody>
    </table>
  `;
}

render(html`<${App} />`, document.getElementById('root'));
