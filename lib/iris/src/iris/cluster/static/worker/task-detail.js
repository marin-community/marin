import { h, render } from 'preact';
import { useState, useEffect } from 'preact/hooks';
import htm from 'htm';
import { stateToName, formatTimestamp } from '/static/shared/utils.js';
import { workerRpc } from '/static/shared/rpc.js';

const html = htm.bind(h);

const taskId = decodeURIComponent(window.location.pathname.split('/task/')[1]);

function taskIndexFromId(taskIdValue) {
  const last = taskIdValue.split('/').pop();
  const parsed = Number.parseInt(last, 10);
  return Number.isNaN(parsed) ? null : parsed;
}

function StatusSection({ task }) {
  if (!task) return html`<div class="section"><h2>Loading...</h2></div>`;
  if (task.error === 'Not found') return html`<div class="section"><h2>Status: Not Found</h2></div>`;

  const statusClass = 'status-' + stateToName(task.state);
  const taskIndex = taskIndexFromId(task.taskId || taskId);
  return html`<div class="section">
    <h2>Status: <span class=${statusClass}>${task.state}</span></h2>
    <p><b>Job ID:</b> ${task.jobId}</p>
    <p><b>Task Index:</b> ${taskIndex ?? '-'}</p>
    <p><b>Attempt:</b> ${task.currentAttemptId}</p>
    <p><b>Started:</b> ${formatTimestamp(task.startedAtMs)}</p>
    <p><b>Finished:</b> ${formatTimestamp(task.finishedAtMs)}</p>
    <p><b>Exit Code:</b> ${task.exitCode !== null ? task.exitCode : '-'}</p>
    <p><b>Error:</b> ${task.error || '-'}</p>
    <p><b>Ports:</b> ${JSON.stringify(task.ports)}</p>
  </div>`;
}

function ResourcesSection({ task }) {
  if (!task || !task.resourceUsage) return null;
  const r = task.resourceUsage;
  return html`<div class="section">
    <h2>Resources</h2>
    <div class="metrics">
      <div class="metric"><div class="metric-value">${r.memoryMb}</div><div class="metric-label">Memory (MB)</div></div>
      <div class="metric"><div class="metric-value">${r.memoryPeakMb}</div><div class="metric-label">Peak Memory (MB)</div></div>
      <div class="metric"><div class="metric-value">${r.cpuPercent}%</div><div class="metric-label">CPU</div></div>
      <div class="metric"><div class="metric-value">${r.processCount}</div><div class="metric-label">Processes</div></div>
      <div class="metric"><div class="metric-value">${r.diskMb}</div><div class="metric-label">Disk (MB)</div></div>
    </div>
  </div>`;
}

function BuildSection({ task }) {
  if (!task || !task.buildMetrics) return null;
  const b = task.buildMetrics;
  const durationMs = (b.buildStartedMs && b.buildFinishedMs)
    ? b.buildFinishedMs - b.buildStartedMs : 0;
  const duration = durationMs > 0 ? (durationMs / 1000).toFixed(2) + 's' : '-';
  return html`<div class="section">
    <h2>Build</h2>
    <p><b>Image:</b> <code>${b.imageTag || '-'}</code></p>
    <p><b>Build Time:</b> ${duration}</p>
    <p><b>From Cache:</b> ${b.fromCache ? 'Yes' : 'No'}</p>
  </div>`;
}

function LogsSection({ logs }) {
  const [activeTab, setActiveTab] = useState('all');

  const formatLog = (entries) =>
    entries.map(l => `[${new Date(l.timestampMs).toLocaleTimeString()}] ${l.data}`).join('\n') || 'No logs';

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
    try {
      const taskData = await workerRpc('GetTaskStatus', { taskId });
      setTask(taskData);
      const logsResp = await workerRpc('FetchTaskLogs', { taskId });
      setLogs(logsResp.logs || []);
    } catch (e) {
      setTask({ error: 'Not found' });
    }
  }

  useEffect(() => { refresh(); }, []);

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
