/**
 * Job detail page — fetches job info, tasks, and task logs via controller RPC.
 * Extracts job_id from the URL path: /job/{job_id}
 */
import { h, render } from 'preact';
import { useState, useEffect, useRef, useCallback } from 'preact/hooks';
import htm from 'htm';
import { controllerRpc } from '../shared/rpc.js';
import { escapeHtml, formatBytes, formatDuration, formatTimestamp, stateToName } from '../shared/utils.js';

const html = htm.bind(h);

const jobId = decodeURIComponent(window.location.pathname.split('/job/')[1]);

function getStateClass(state) {
  const map = {
    pending: 'status-pending',
    building: 'status-building',
    running: 'status-running',
    succeeded: 'status-succeeded',
    failed: 'status-failed',
    killed: 'status-killed',
    worker_failed: 'status-worker_failed',
    unschedulable: 'status-unschedulable',
  };
  return map[state] || '';
}

function InfoRow({ label, value, valueClass }) {
  return html`<div class="info-row">
    <span class="info-label">${label}</span>
    <span class=${'info-value ' + (valueClass || '')}>${value}</span>
  </div>`;
}

function InfoCard({ title, children }) {
  return html`<div class="info-card">
    <h3>${title}</h3>
    ${children}
  </div>`;
}

const TERMINAL_STATES = new Set(['succeeded', 'failed', 'killed', 'worker_failed', 'unschedulable']);

function JobDetailApp() {
  const [job, setJob] = useState(null);
  const [tasks, setTasks] = useState([]);
  const [error, setError] = useState(null);
  const [selectedTaskIndex, setSelectedTaskIndex] = useState(null);
  const [taskLogs, setTaskLogs] = useState('Loading logs...');
  const [taskLogStatus, setTaskLogStatus] = useState('');
  const autoSelectedRef = useRef(false);

  const fetchTaskLogs = useCallback(async (taskIndex, tasksList) => {
    if (taskIndex === null || taskIndex === undefined || isNaN(taskIndex)) {
      setTaskLogs('Select a task to view logs');
      setTaskLogStatus('');
      return;
    }

    const selected = (tasksList || tasks).find(t => t.task_index === taskIndex);
    if (selected && (selected.state === 'pending' || selected.state === 'building')) {
      const label = selected.state === 'pending' ? 'Waiting to be scheduled' : 'Building container image';
      setTaskLogStatus(html`<span style="color:#9a6700;font-weight:500">${label}</span>`);
      setTaskLogs(selected.pending_reason || 'No logs yet \u2014 task has not started running.');
      return;
    }

    setTaskLogs('Loading logs...');
    setTaskLogStatus('');
    try {
      const resp = await controllerRpc('GetTaskLogs', { jobId, taskIndex, limit: 1000 });
      const logs = resp.logs || [];
      if (selected && selected.error) {
        setTaskLogStatus(html`<span style="color:#cf222e;font-weight:600">Error: ${selected.error}</span>`);
      } else {
        setTaskLogStatus(resp.workerAddress ? 'from ' + resp.workerAddress : '');
      }
      setTaskLogs(logs.length === 0 ? 'No logs available' : logs.map(l => l.data || '').join('\n'));
    } catch (e) {
      setTaskLogs('Failed to load logs: ' + e.message);
      setTaskLogStatus('');
    }
  }, [tasks]);

  const refresh = useCallback(async () => {
    try {
      const [jobsResp, tasksResp] = await Promise.all([
        controllerRpc('ListJobs'),
        controllerRpc('ListTasks', { jobId }),
      ]);

      const jobs = (jobsResp.jobs || []).map(j => ({
        job_id: j.jobId,
        name: j.name || '',
        state: stateToName(j.state),
        failure_count: j.failureCount || 0,
        error: j.error,
        started_at_ms: parseInt(j.startedAtMs || 0),
        finished_at_ms: parseInt(j.finishedAtMs || 0),
        resources: {
          cpu: j.resources ? j.resources.cpu : 0,
          memory_bytes: j.resources ? parseInt(j.resources.memoryBytes || 0) : 0,
        },
      }));

      const tasksList = (tasksResp.tasks || []).map(t => ({
        task_id: t.taskId,
        task_index: t.taskIndex,
        state: stateToName(t.state),
        worker_id: t.workerId || '',
        started_at_ms: parseInt(t.startedAtMs || 0),
        finished_at_ms: parseInt(t.finishedAtMs || 0),
        exit_code: t.exitCode,
        error: t.error || '',
        num_attempts: (t.attempts || []).length || 1,
        pending_reason: t.pendingReason || '',
      }));

      const found = jobs.find(j => j.job_id === jobId);
      if (!found) {
        setError('Job not found');
        return;
      }
      setJob(found);
      setTasks(tasksList);
      setError(null);

      // Auto-select most interesting task on first load
      if (!autoSelectedRef.current && tasksList.length > 0) {
        autoSelectedRef.current = true;
        const failedTask = tasksList.find(t => t.state === 'failed' || t.state === 'worker_failed');
        const runningTask = tasksList.find(t => t.state === 'running' || t.state === 'building');
        const autoTask = failedTask || runningTask || tasksList[0];
        if (autoTask !== undefined) {
          setSelectedTaskIndex(autoTask.task_index);
          fetchTaskLogs(autoTask.task_index, tasksList);
        }
      }
    } catch (e) {
      setError('Failed to load job details: ' + e.message);
    }
  }, [fetchTaskLogs]);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 5000);
    return () => clearInterval(interval);
  }, [refresh]);

  const logsPreRef = useRef(null);
  useEffect(() => {
    if (logsPreRef.current) logsPreRef.current.scrollTop = logsPreRef.current.scrollHeight;
  }, [taskLogs]);

  if (error) {
    return html`
      <a href="/" class="back-link">\u2190 Back to Dashboard</a>
      <h1>Job: ${jobId}</h1>
      <div class="error-message">${error}</div>
    `;
  }

  if (!job) {
    return html`
      <a href="/" class="back-link">\u2190 Back to Dashboard</a>
      <h1>Job: ${jobId}</h1>
      <p>Loading...</p>
    `;
  }

  const isTerminal = TERMINAL_STATES.has(job.state);
  const title = (job.name && job.name !== jobId) ? job.name : 'Job: ' + jobId;
  const subtitle = (job.name && job.name !== jobId) ? 'ID: ' + jobId : '';

  // Count task states
  const counts = { total: tasks.length, completed: 0, running: 0, pending: 0, failed: 0 };
  for (const t of tasks) {
    if (t.state === 'succeeded' || t.state === 'killed') counts.completed++;
    else if (t.state === 'running' || t.state === 'building') counts.running++;
    else if (t.state === 'pending') counts.pending++;
    else if (t.state === 'failed' || t.state === 'worker_failed') counts.failed++;
  }

  return html`
    <a href="/" class="back-link">\u2190 Back to Dashboard</a>
    <h1>${title}</h1>
    ${subtitle && html`<div style="color:#57606a;font-size:14px;margin-bottom:20px">${subtitle}</div>`}
    ${job.error && html`<div class="error-message"><strong>Error:</strong> ${job.error}</div>`}

    <div class="info-grid">
      <${InfoCard} title="Job Status">
        <${InfoRow} label="State" value=${job.state} valueClass=${getStateClass(job.state)} />
        <${InfoRow} label="Exit Code" value=${job.failure_count > 0 ? 'Failed' : (job.state === 'succeeded' ? '0' : '-')} />
        <${InfoRow} label="Started" value=${formatTimestamp(job.started_at_ms)} />
        <${InfoRow} label="Finished" value=${isTerminal ? formatTimestamp(job.finished_at_ms) : '-'} />
        <${InfoRow} label="Duration" value=${isTerminal
          ? formatDuration(job.started_at_ms, job.finished_at_ms)
          : (job.started_at_ms ? formatDuration(job.started_at_ms, Date.now()) : '-')} />
      <//>
      <${InfoCard} title="Task Summary">
        <${InfoRow} label="Total Tasks" value=${counts.total} />
        <${InfoRow} label="Completed" value=${counts.completed} />
        <${InfoRow} label="Running" value=${counts.running} />
        <${InfoRow} label="Pending" value=${counts.pending} />
        <${InfoRow} label="Failed" value=${counts.failed} />
      <//>
      <${InfoCard} title="Resource Request">
        <${InfoRow} label="CPU" value=${job.resources.cpu || '-'} />
        <${InfoRow} label="Memory" value=${job.resources.memory_bytes ? formatBytes(job.resources.memory_bytes) : '-'} />
        <${InfoRow} label="Replicas" value=${tasks.length || '-'} />
      <//>
    </div>

    <h2>Task Logs</h2>
    <div style="margin-bottom:10px;display:flex;align-items:center;gap:12px">
      <select value=${selectedTaskIndex ?? ''} onChange=${e => {
        const idx = parseInt(e.target.value);
        setSelectedTaskIndex(isNaN(idx) ? null : idx);
        fetchTaskLogs(isNaN(idx) ? null : idx, tasks);
      }}>
        <option value="">Select a task...</option>
        ${tasks.map(t => html`<option value=${t.task_index}>Task ${t.task_index} (${t.state})</option>`)}
      </select>
      <span style="color:#57606a;font-size:13px">${taskLogStatus}</span>
    </div>
    <pre ref=${logsPreRef} style="background:white;padding:15px;border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,0.12);border:1px solid #d1d9e0;max-height:600px;overflow-y:auto;font-size:12px;white-space:pre-wrap">${taskLogs}</pre>

    <h2>Tasks</h2>
    <table>
      <thead><tr>
        <th>Task ID</th><th>Index</th><th>State</th><th>Worker</th>
        <th>Attempts</th><th>Started</th><th>Duration</th><th>Exit Code</th><th>Error</th>
      </tr></thead>
      <tbody>
        ${tasks.map(t => html`<tr>
          <td>${t.task_id}</td>
          <td>${t.task_index}</td>
          <td class=${getStateClass(t.state)}>${t.state}${t.pending_reason ? html`<br/><span class="pending-reason">${t.pending_reason}</span>` : ''}</td>
          <td>${t.worker_id || '-'}</td>
          <td>${t.num_attempts}</td>
          <td>${formatTimestamp(t.started_at_ms)}</td>
          <td>${formatDuration(t.started_at_ms, t.finished_at_ms)}</td>
          <td>${TERMINAL_STATES.has(t.state) && t.exit_code !== null && t.exit_code !== undefined ? t.exit_code : '-'}</td>
          <td>${t.error || '-'}</td>
        </tr>`)}
      </tbody>
    </table>
  `;
}

render(html`<${JobDetailApp} />`, document.getElementById('root'));
