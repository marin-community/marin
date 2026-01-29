import { h, render } from 'preact';
import { useState, useEffect } from 'preact/hooks';
import htm from 'htm';
import { stateToName, formatTimestamp } from '/static/shared/utils.js';

const html = htm.bind(h);

function StatsBar({ stats }) {
  return html`<div>
    <b>Running:</b> ${stats.running} | <b>Pending:</b> ${stats.pending} |
    <b>Building:</b> ${stats.building} | <b>Completed:</b> ${stats.completed}
  </div>`;
}

function TaskRow({ task }) {
  const started = formatTimestamp(task.started_at);
  const finished = formatTimestamp(task.finished_at);
  const exitCode = task.exit_code !== null && task.exit_code !== undefined ? task.exit_code : '-';
  const taskDisplay = task.attempt_id > 0
    ? `${task.task_id.slice(0, 12)}... (attempt ${task.attempt_id})`
    : `${task.task_id.slice(0, 12)}...`;
  const statusClass = 'status-' + stateToName(task.status);

  return html`<tr>
    <td><a href=${'/task/' + encodeURIComponent(task.task_id)} class="task-link" target="_blank">${taskDisplay}</a></td>
    <td>${task.job_id.slice(0, 8)}...</td>
    <td>${task.task_index}/${task.num_tasks || '?'}</td>
    <td class=${statusClass}>${task.status}</td>
    <td>${exitCode}</td>
    <td>${task.memory_mb || 0}/${task.memory_peak_mb || 0} MB</td>
    <td>${(task.cpu_percent || 0) + '%'}</td>
    <td>${started}</td>
    <td>${finished}</td>
    <td>${task.error || '-'}</td>
  </tr>`;
}

function App() {
  const [stats, setStats] = useState({ running: 0, pending: 0, building: 0, completed: 0 });
  const [tasks, setTasks] = useState([]);

  async function refresh() {
    const [statsData, tasksData] = await Promise.all([
      fetch('/api/stats').then(r => r.json()),
      fetch('/api/tasks').then(r => r.json()),
    ]);
    setStats(statsData);
    setTasks(tasksData);
  }

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, []);

  return html`
    <h1>Iris Worker Dashboard</h1>
    <${StatsBar} stats=${stats} />
    <h2>Tasks</h2>
    <table>
      <thead>
        <tr>
          <th>Task ID</th><th>Job ID</th><th>Index</th><th>Status</th><th>Exit</th>
          <th>Memory</th><th>CPU</th><th>Started</th><th>Finished</th><th>Error</th>
        </tr>
      </thead>
      <tbody>
        ${tasks.map(t => html`<${TaskRow} key=${t.task_id} task=${t} />`)}
      </tbody>
    </table>
  `;
}

render(html`<${App} />`, document.getElementById('root'));
