import { h, render } from 'preact';
import { useState, useEffect } from 'preact/hooks';
import htm from 'htm';
import { stateToName, formatTimestamp } from '/static/shared/utils.js';

const html = htm.bind(h);

const taskId = decodeURIComponent(window.location.pathname.split('/task/')[1]);

function StatusSection({ task }) {
  if (!task) return html`<div class="section"><h2>Loading...</h2></div>`;
  if (task.error === 'Not found') return html`<div class="section"><h2>Status: Not Found</h2></div>`;

  const statusClass = 'status-' + stateToName(task.status);
  return html`<div class="section">
    <h2>Status: <span class=${statusClass}>${task.status}</span></h2>
    <p><b>Job ID:</b> ${task.job_id}</p>
    <p><b>Task Index:</b> ${task.task_index}</p>
    <p><b>Attempt:</b> ${task.attempt_id}</p>
    <p><b>Started:</b> ${formatTimestamp(task.started_at)}</p>
    <p><b>Finished:</b> ${formatTimestamp(task.finished_at)}</p>
    <p><b>Exit Code:</b> ${task.exit_code !== null ? task.exit_code : '-'}</p>
    <p><b>Error:</b> ${task.error || '-'}</p>
    <p><b>Ports:</b> ${JSON.stringify(task.ports)}</p>
  </div>`;
}

function ResourcesSection({ task }) {
  if (!task || !task.resources) return null;
  const r = task.resources;
  return html`<div class="section">
    <h2>Resources</h2>
    <div class="metrics">
      <div class="metric"><div class="metric-value">${r.memory_mb}</div><div class="metric-label">Memory (MB)</div></div>
      <div class="metric"><div class="metric-value">${r.memory_peak_mb}</div><div class="metric-label">Peak Memory (MB)</div></div>
      <div class="metric"><div class="metric-value">${r.cpu_percent}%</div><div class="metric-label">CPU</div></div>
      <div class="metric"><div class="metric-value">${r.process_count}</div><div class="metric-label">Processes</div></div>
      <div class="metric"><div class="metric-value">${r.disk_mb}</div><div class="metric-label">Disk (MB)</div></div>
    </div>
  </div>`;
}

function BuildSection({ task }) {
  if (!task || !task.build) return null;
  const b = task.build;
  const duration = b.duration_ms > 0 ? (b.duration_ms / 1000).toFixed(2) + 's' : '-';
  return html`<div class="section">
    <h2>Build</h2>
    <p><b>Image:</b> <code>${b.image_tag || '-'}</code></p>
    <p><b>Build Time:</b> ${duration}</p>
    <p><b>From Cache:</b> ${b.from_cache ? 'Yes' : 'No'}</p>
  </div>`;
}

function LogsSection({ logs }) {
  const [activeTab, setActiveTab] = useState('all');

  const formatLog = (entries) =>
    entries.map(l => `[${new Date(l.timestamp).toLocaleTimeString()}] ${l.data}`).join('\n') || 'No logs';

  const filtered = activeTab === 'all' ? logs : logs.filter(l => l.source === activeTab);
  const tabs = ['all', 'stdout', 'stderr', 'build'];

  return html`<div class="section">
    <h2>Logs</h2>
    <div class="tabs">
      ${tabs.map(tab => html`
        <div class=${'tab' + (activeTab === tab ? ' active' : '')}
             onClick=${() => setActiveTab(tab)}>
          ${tab.toUpperCase()}
        </div>
      `)}
    </div>
    <div class="tab-content active">${formatLog(filtered)}</div>
  </div>`;
}

function TaskDetail() {
  const [task, setTask] = useState(null);
  const [logs, setLogs] = useState([]);

  async function refresh() {
    const taskData = await fetch(`/api/tasks/${encodeURIComponent(taskId)}`).then(r => r.json());
    setTask(taskData);
    if (taskData.error !== 'Not found') {
      const logsData = await fetch(`/api/tasks/${encodeURIComponent(taskId)}/logs`).then(r => r.json());
      setLogs(logsData);
    }
  }

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, []);

  return html`
    <h1>Task: <code>${taskId}</code></h1>
    <a href="/">← Back to Dashboard</a>
    <${StatusSection} task=${task} />
    <${ResourcesSection} task=${task} />
    <${BuildSection} task=${task} />
    <${LogsSection} logs=${logs} />
  `;
}

render(html`<${TaskDetail} />`, document.getElementById('root'));
