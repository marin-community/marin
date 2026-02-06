/**
 * Job detail page — fetches job info, tasks, and task logs via controller RPC.
 * Extracts job_id from the URL path: /job/{job_id}
 */
import { h, render } from 'preact';
import { useState, useEffect, useRef, useCallback } from 'preact/hooks';
import htm from 'htm';
import { controllerRpc } from '/static/shared/rpc.js';
import { formatBytes, formatDuration, formatTimestamp, stateToName, timestampFromProto } from '/static/shared/utils.js';
import { InfoRow, InfoCard } from '/static/shared/components.js';

const html = htm.bind(h);

const jobId = decodeURIComponent(window.location.pathname.split('/job/')[1]);

function getStateClass(state) {
  const map = {
    pending: 'status-pending',
    assigned: 'status-assigned',
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

const TERMINAL_STATES = new Set(['succeeded', 'failed', 'killed', 'worker_failed', 'unschedulable']);

function taskIndexFromId(taskId) {
  const last = taskId.split('/').pop();
  const parsed = Number.parseInt(last, 10);
  return Number.isNaN(parsed) ? null : parsed;
}

function JobDetailApp() {
  const [job, setJob] = useState(null);
  const [tasks, setTasks] = useState([]);
  const [error, setError] = useState(null);
  const [selectedTaskId, setSelectedTaskId] = useState(null);
  const [taskLogs, setTaskLogs] = useState('Loading logs...');
  const [taskLogStatus, setTaskLogStatus] = useState('');
  const tasksRef = useRef([]);

  const fetchTaskLogs = useCallback(async (taskId, tasksList) => {
    if (!taskId) {
      setTaskLogs('Select a task to view logs');
      setTaskLogStatus('');
      return;
    }

    const selected = tasksList.find(t => t.task_id === taskId);
    // For pending/assigned tasks, we can't fetch logs (no worker yet)
    if (selected && (selected.state === 'pending' || selected.state === 'assigned')) {
      let label = 'No logs yet';
      if (selected.state === 'pending') label = 'Waiting to be scheduled';
      else if (selected.state === 'assigned') label = 'Assigned to worker, starting soon';
      setTaskLogStatus(html`<span style="color:#9a6700;font-weight:500">${label}</span>`);
      setTaskLogs(selected.pending_reason || 'No logs yet \u2014 task has not started running.');
      return;
    }

    setTaskLogs('Loading logs...');
    // For building tasks, show status while we fetch logs
    if (selected && selected.state === 'building') {
      setTaskLogStatus(html`<span style="color:#8250df;font-weight:500">Building container image</span>`);
    } else {
      setTaskLogStatus('');
    }
    try {
      // Use batch API with id field (supports both task and job IDs)
      const resp = await controllerRpc('GetTaskLogs', { id: taskId });
      const taskBatches = resp.taskLogs || [];
      // Find the batch for this specific task
      const batch = taskBatches.find(b => b.taskId === taskId) || taskBatches[0];
      if (batch && batch.error) {
        setTaskLogStatus(html`<span style="color:#cf222e;font-weight:600">Error: ${batch.error}</span>`);
        setTaskLogs(batch.error);
        return;
      }
      const logs = batch ? (batch.logs || []) : [];
      if (selected && selected.error) {
        setTaskLogStatus(html`<span style="color:#cf222e;font-weight:600">Error: ${selected.error}</span>`);
      } else if (selected && selected.state === 'building') {
        // Keep the building status but add worker address
        const addr = resp.workerAddress ? ' (from ' + resp.workerAddress + ')' : '';
        setTaskLogStatus(html`<span style="color:#8250df;font-weight:500">Building container image${addr}</span>`);
      } else {
        setTaskLogStatus('');
      }
      setTaskLogs(logs.length === 0 ? 'No logs available yet' : logs.map(l => l.data || '').join('\n'));
    } catch (e) {
      setTaskLogs('Failed to load logs: ' + e.message);
      setTaskLogStatus('');
    }
  }, []);

  useEffect(() => {
    async function load() {
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
          started_at_ms: timestampFromProto(j.startedAt),
          finished_at_ms: timestampFromProto(j.finishedAt),
          resources: {
            cpu: j.resources ? j.resources.cpu : 0,
            memory_bytes: j.resources ? parseInt(j.resources.memoryBytes || 0) : 0,
          },
        }));

        const tasksList = (tasksResp.tasks || []).map(t => ({
          task_id: t.taskId,
          task_index: taskIndexFromId(t.taskId),
          state: stateToName(t.state),
          worker_id: t.workerId || '',
          started_at_ms: timestampFromProto(t.startedAt),
          finished_at_ms: timestampFromProto(t.finishedAt),
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
        tasksRef.current = tasksList;
        setError(null);

        // Auto-select most interesting task
        if (tasksList.length > 0) {
          const failedTask = tasksList.find(t => t.state === 'failed' || t.state === 'worker_failed');
          const runningTask = tasksList.find(t => t.state === 'running' || t.state === 'building' || t.state === 'assigned');
          const autoTask = failedTask || runningTask || tasksList[0];
          if (autoTask !== undefined) {
            setSelectedTaskId(autoTask.task_id);
            fetchTaskLogs(autoTask.task_id, tasksList);
          }
        }
      } catch (e) {
        setError('Failed to load job details: ' + e.message);
      }
    }
    load();
  }, [fetchTaskLogs]);

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
  const counts = { total: tasks.length, completed: 0, building: 0, running: 0, assigned: 0, pending: 0, failed: 0 };
  for (const t of tasks) {
    if (t.state === 'succeeded' || t.state === 'killed') counts.completed++;
    else if (t.state === 'building') counts.building++;
    else if (t.state === 'running') counts.running++;
    else if (t.state === 'assigned') counts.assigned++;
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
        <${InfoRow} label="Building" value=${counts.building} />
        <${InfoRow} label="Assigned" value=${counts.assigned} />
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
      <select value=${selectedTaskId ?? ''} onChange=${e => {
        const value = e.target.value;
        setSelectedTaskId(value || null);
        fetchTaskLogs(value || null, tasksRef.current);
      }}>
        <option value="">Select a task...</option>
        ${tasks.map(t => html`<option value=${t.task_id}>Task ${t.task_index ?? '-'} (${t.state})</option>`)}
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
          <td>${t.task_index ?? '-'}</td>
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
