import { h, render } from 'preact';
import { useState, useEffect, useCallback, useRef } from 'preact/hooks';
import htm from 'htm';
import { stateToName, formatTimestamp, formatRelativeTime, timestampFromProto } from '/static/shared/utils.js';
import { workerRpc } from '/static/shared/rpc.js';
import { profileAndDownload } from '/static/shared/profiling.js';
import { Gauge, MetricCard, ResourceSection, Field, Section } from '/static/shared/components.js';

const html = htm.bind(h);

const taskId = decodeURIComponent(window.location.pathname.split('/task/')[1]);

function taskIndexFromId(taskIdValue) {
  const last = taskIdValue.split('/').pop();
  const parsed = Number.parseInt(last, 10);
  return Number.isNaN(parsed) ? null : parsed;
}

function jobIdFromTaskId(taskIdValue) {
  const parts = taskIdValue.split('/');
  parts.pop();
  return parts.join('/');
}

function StatusSection({ task }) {
  if (!task) return html`<div class="worker-detail-loading">
    <div class="worker-detail-loading__spinner"></div>
    <span>Loading task details\u2026</span>
  </div>`;
  if (task.error === 'Not found') return html`<div class="error-message">Task not found: <code>${taskId}</code></div>`;

  const state = stateToName(task.state);
  const statusClass = 'status-' + state;
  const taskIndex = taskIndexFromId(task.taskId || taskId);
  const jobId = jobIdFromTaskId(task.taskId || taskId);
  const startMs = timestampFromProto(task.startedAt);
  const finishMs = timestampFromProto(task.finishedAt);

  return html`
    <div class="worker-detail-grid">
      <${Section} title="Status">
        <dl class="worker-detail-fields">
          <${Field} label="State" value=${html`<span class=${statusClass}>${state}</span>`} />
          <${Field} label="Job ID" value=${jobId} mono />
          <${Field} label="Task Index" value=${taskIndex ?? '-'} />
          <${Field} label="Attempt" value=${task.currentAttemptId} />
          <${Field} label="Exit Code" value=${task.exitCode !== null && task.exitCode !== undefined ? task.exitCode : null} />
        </dl>
      <//>
      <${Section} title="Timing">
        <dl class="worker-detail-fields">
          <${Field} label="Started" value=${formatTimestamp(startMs)} />
          ${startMs > 0 && html`<${Field} label="Elapsed" value=${formatRelativeTime(startMs)} />`}
          <${Field} label="Finished" value=${formatTimestamp(finishMs)} />
          <${Field} label="Ports" value=${task.ports && Object.keys(task.ports).length > 0 ? JSON.stringify(task.ports) : null} mono />
        </dl>
      <//>
    </div>
    ${task.error && html`<div class="error-message">${task.error}</div>`}
  `;
}

function ResourcesSection({ task }) {
  if (!task || !task.resourceUsage) return null;
  const r = task.resourceUsage;
  const memMb = r.memoryMb || 0;
  const peakMb = r.memoryPeakMb || 0;
  const cpu = r.cpuPercent || 0;
  const procs = r.processCount || 0;
  const diskMb = r.diskMb || 0;

  const cpuClass = cpu >= 90 ? 'danger' : cpu >= 70 ? 'warning' : 'accent';
  const memClass = peakMb > 0 && (memMb / peakMb) >= 0.9 ? 'warning' : 'accent';

  return html`
    <h2>Resources</h2>
    <div class="metric-row">
      <${MetricCard} value=${memMb + ' MB'} label="Memory" valueClass=${memClass} />
      <${MetricCard} value=${peakMb + ' MB'} label="Peak Memory" />
      <${MetricCard} value=${cpu + '%'} label="CPU" valueClass=${cpuClass} />
      <${MetricCard} value=${procs} label="Processes" />
      ${diskMb > 0 && html`<${MetricCard} value=${diskMb >= 1024 ? (diskMb / 1024).toFixed(1) + ' GB' : diskMb + ' MB'} label="Disk" />`}
    </div>
    <${ResourceSection}>
      ${peakMb > 0 && html`<${Gauge} label="Memory" value=${memMb} max=${peakMb} format="raw" warnAt=${80} dangerAt=${95} />`}
      <${Gauge} label="CPU" value=${cpu} max=${100} format="percent" />
    <//>
  `;
}

function BuildSection({ task }) {
  if (!task || !task.buildMetrics) return null;
  const b = task.buildMetrics;
  const startMs = timestampFromProto(b.buildStarted);
  const endMs = timestampFromProto(b.buildFinished);
  const durationMs = (startMs && endMs) ? endMs - startMs : 0;
  const duration = durationMs > 0 ? (durationMs / 1000).toFixed(2) + 's' : '-';
  return html`
    <${Section} title="Build">
      <dl class="worker-detail-fields">
        <${Field} label="Image" value=${b.imageTag || null} mono />
        <${Field} label="Build Time" value=${duration} />
        <${Field} label="From Cache" value=${b.fromCache ? 'Yes' : 'No'} />
      </dl>
    <//>
  `;
}

function LogsSection({ logs }) {
  const [activeTab, setActiveTab] = useState('all');

  const formatLog = (entries) =>
    entries.map(l => `[${new Date(timestampFromProto(l.timestamp) || 0).toLocaleTimeString()}] ${l.data}`).join('\n') || 'No logs';

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

function ProfileSection({ task, onCpuProfile, onMemoryProfile, profiling }) {
  if (!task || task.state !== 'TASK_STATE_RUNNING') return null;
  return html`<div class="section">
    <h2>Profiling</h2>
    <div style="display:flex;gap:8px;margin-bottom:8px">
      <button
        onClick=${onCpuProfile}
        disabled=${profiling}
        style="padding:6px 16px;font-size:13px;background:#8250df;color:white;border:none;border-radius:4px;cursor:pointer"
      >
        ${profiling ? 'Profiling (10s)...' : 'CPU Profile (py-spy)'}
      </button>
      <button
        onClick=${onMemoryProfile}
        disabled=${profiling}
        style="padding:6px 16px;font-size:13px;background:#1f883d;color:white;border:none;border-radius:4px;cursor:pointer"
      >
        ${profiling ? 'Profiling (10s)...' : 'Memory Profile (memray)'}
      </button>
    </div>
    <p style="color:#57606a;font-size:12px">CPU: 10-second sample, open in <a href="https://www.speedscope.app" target="_blank">speedscope.app</a>. Memory: HTML flamegraph with timeline.</p>
  </div>`;
}

function TaskDetail() {
  const [task, setTask] = useState(null);
  const [logs, setLogs] = useState([]);
  const [profiling, setProfiling] = useState(false);
  const intervalRef = useRef(null);

  const refresh = useCallback(async () => {
    try {
      const taskData = await workerRpc('GetTaskStatus', { taskId });
      setTask(taskData);
      const logsResp = await workerRpc('FetchTaskLogs', { taskId });
      setLogs(logsResp.logs || []);
    } catch (e) {
      setTask({ error: 'Not found' });
    }
  }, []);

  async function handleCpuProfile() {
    setProfiling(true);
    try {
      await profileAndDownload(workerRpc, taskId, { profilerType: 'cpu', format: 'SPEEDSCOPE' });
    } catch (e) {
      alert('CPU profile failed: ' + e.message);
    } finally {
      setProfiling(false);
    }
  }

  async function handleMemoryProfile() {
    setProfiling(true);
    try {
      await profileAndDownload(workerRpc, taskId, { profilerType: 'memory', format: 'FLAMEGRAPH' });
    } catch (e) {
      alert('Memory profile failed: ' + e.message);
    } finally {
      setProfiling(false);
    }
  }

  useEffect(() => { refresh(); }, [refresh]);

  // Auto-refresh every 5 seconds while the task is running
  useEffect(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    const isRunning = task && (task.state === 'TASK_STATE_RUNNING' || task.state === 'TASK_STATE_BUILDING' || task.state === 'TASK_STATE_ASSIGNED');
    if (isRunning) {
      intervalRef.current = setInterval(refresh, 5000);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [task && task.state, refresh]);

  const isRunning = task && (task.state === 'TASK_STATE_RUNNING' || task.state === 'TASK_STATE_BUILDING' || task.state === 'TASK_STATE_ASSIGNED');

  return html`
    <h1 style="display:flex;align-items:center;gap:12px">
      Task: <code>${taskId}</code>
      ${isRunning && html`<span class="auto-refresh-badge">auto-refresh</span>`}
    </h1>
    <a href="/" class="back-link">\u2190 Back to Dashboard</a>
    <${StatusSection} task=${task} />
    <${ProfileSection} task=${task} onCpuProfile=${handleCpuProfile} onMemoryProfile=${handleMemoryProfile} profiling=${profiling} />
    <${ResourcesSection} task=${task} />
    <${BuildSection} task=${task} />
    <${LogsSection} logs=${logs} />
  `;
}

render(html`<${TaskDetail} />`, document.getElementById('root'));
