import { h, render } from 'preact';
import { useState, useEffect, useCallback, useRef } from 'preact/hooks';
import htm from 'htm';
import { stateToName, formatTimestamp, formatRelativeTime, timestampFromProto } from '/static/shared/utils.js';
import { workerRpc } from '/static/shared/rpc.js';
import { profileAndDownload } from '/static/shared/profiling.js';
import { Gauge, MetricCard, ResourceSection, Sparkline, Field, Section, formatMbPair } from '/static/shared/components.js';
import { LogViewer } from '/static/shared/log-viewer.js';

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

function ResourcesSection({ task, memHistory, cpuHistory }) {
  if (!task || !task.resourceUsage) return null;
  const r = task.resourceUsage;
  const memMb = r.memoryMb || 0;
  const peakMb = r.memoryPeakMb || 0;
  const cpu = r.cpuPercent || 0;
  const procs = r.processCount || 0;
  const diskMb = r.diskMb || 0;

  const cpuClass = cpu >= 90 ? 'danger' : cpu >= 70 ? 'warning' : 'accent';
  const memClass = peakMb > 0 && (memMb / peakMb) >= 0.9 ? 'warning' : 'accent';

  // Format memory values with appropriate units
  const memDisplay = memMb >= 1024 ? (memMb / 1024).toFixed(1) + ' GB' : memMb + ' MB';
  const peakDisplay = peakMb >= 1024 ? (peakMb / 1024).toFixed(1) + ' GB' : peakMb + ' MB';

  return html`
    <h2>Resources</h2>
    <div class="metric-row">
      <${MetricCard} value=${memDisplay} label="Memory" detail=${formatMbPair(memMb, peakMb)} valueClass=${memClass} />
      <${MetricCard} value=${peakDisplay} label="Peak Memory" />
      <${MetricCard} value=${cpu + '%'} label="CPU" valueClass=${cpuClass} />
      <${MetricCard} value=${procs} label="Processes" />
      ${diskMb > 0 && html`<${MetricCard} value=${diskMb >= 1024 ? (diskMb / 1024).toFixed(1) + ' GB' : diskMb + ' MB'} label="Disk" />`}
    </div>
    <${ResourceSection}>
      ${peakMb > 0 && html`
        <div style="display:flex;align-items:center;gap:12px">
          <div style="flex:1"><${Gauge} label="Memory" value=${memMb} max=${peakMb} format="raw" warnAt=${80} dangerAt=${95} /></div>
          ${memHistory && memHistory.length >= 2 && html`
            <${Sparkline} values=${memHistory} max=${peakMb} width=${100} height=${28}
              color="var(--color-success)" fillColor="rgba(26,127,55,0.1)" />
          `}
        </div>
      `}
      <div style="display:flex;align-items:center;gap:12px">
        <div style="flex:1"><${Gauge} label="CPU" value=${cpu} max=${100} format="percent" /></div>
        ${cpuHistory && cpuHistory.length >= 2 && html`
          <${Sparkline} values=${cpuHistory} max=${100} width=${100} height=${28}
            color="var(--color-accent)" fillColor="rgba(9,105,218,0.1)" />
        `}
      </div>
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

function LogsSection() {
  return html`<div class="section">
    <h2>Worker Process Logs</h2>
    <${LogViewer}
      rpc=${workerRpc}
      source="/process"
      title=""
      defaultMaxLines=${200}
    />
    <p style="color:#57606a;margin-top:8px;font-size:13px">Task stdout/stderr logs are available on the controller dashboard.</p>
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

// Keep up to 60 samples (5 minutes at 5s intervals) of resource usage
// for sparkline visualization. Stored outside the component to survive
// re-renders without needing useRef.
const MAX_HISTORY = 60;

function TaskDetail() {
  const [task, setTask] = useState(null);
  const [profiling, setProfiling] = useState(false);
  const [memHistory, setMemHistory] = useState([]);
  const [cpuHistory, setCpuHistory] = useState([]);
  const intervalRef = useRef(null);

  const refresh = useCallback(async () => {
    try {
      const taskData = await workerRpc('GetTaskStatus', { taskId });
      setTask(taskData);

      // Accumulate resource usage snapshots for sparklines
      if (taskData.resourceUsage) {
        const memMb = taskData.resourceUsage.memoryMb || 0;
        const cpuPct = taskData.resourceUsage.cpuPercent || 0;
        setMemHistory(prev => [...prev.slice(-(MAX_HISTORY - 1)), memMb]);
        setCpuHistory(prev => [...prev.slice(-(MAX_HISTORY - 1)), cpuPct]);
      }

      // Logs are forwarded via heartbeats to the controller's in-memory log store;
      // use the controller dashboard to view them.
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
    <${ResourcesSection} task=${task} memHistory=${memHistory} cpuHistory=${cpuHistory} />
    <${BuildSection} task=${task} />
    <${LogsSection} />
  `;
}

render(html`<${TaskDetail} />`, document.getElementById('root'));
