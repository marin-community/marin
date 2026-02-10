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

/** Format a proto Duration (milliseconds field) to human-readable. */
function formatProtoTimeout(duration) {
  if (!duration || !duration.milliseconds) return null;
  const ms = parseInt(duration.milliseconds, 10);
  if (!ms) return null;
  const secs = Math.floor(ms / 1000);
  if (secs < 60) return `${secs}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m`;
  return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
}

/** Format device config from proto to a readable string. */
function formatDevice(device) {
  if (!device) return null;
  if (device.tpu) {
    const t = device.tpu;
    let s = t.variant || 'tpu';
    if (t.topology) s += ` (${t.topology})`;
    if (t.count) s += ` x${t.count}`;
    return s;
  }
  if (device.gpu) {
    const g = device.gpu;
    let s = g.variant || 'gpu';
    if (g.count) s += ` x${g.count}`;
    return s;
  }
  if (device.cpu) return 'cpu';
  return null;
}

/** Collapsible section with a disclosure triangle. */
function Collapsible({ title, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);
  return html`
    <div style="margin-bottom:20px">
      <div onClick=${() => setOpen(!open)}
           style="cursor:pointer;display:flex;align-items:center;gap:8px;padding:12px 16px;background:white;border:1px solid #d1d9e0;border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,0.12)">
        <span style="font-size:12px;color:#57606a;transition:transform 0.15s;transform:rotate(${open ? '90deg' : '0deg'})">\u25B6</span>
        <h3 style="margin:0;font-size:16px;font-weight:600;color:#1f2328">${title}</h3>
      </div>
      ${open && html`
        <div style="background:white;border:1px solid #d1d9e0;border-top:none;border-radius:0 0 6px 6px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.12)">
          ${children}
        </div>
      `}
    </div>
  `;
}

/** Render a key-value row in the request detail section. */
function DetailRow({ label, value }) {
  if (value === null || value === undefined || value === '') return null;
  return html`<div style="display:flex;padding:6px 0;border-bottom:1px solid #f0f0f0">
    <span style="color:#57606a;min-width:160px;font-size:13px">${label}</span>
    <span style="font-size:13px;font-family:ui-monospace,SFMono-Regular,SF Mono,Menlo,monospace;word-break:break-all">${value}</span>
  </div>`;
}

/** Render the original LaunchJobRequest as a collapsible details section. */
function JobRequestDetail({ request }) {
  if (!request) return null;

  const ep = request.entrypoint;
  const cmd = ep && ep.command && ep.command.argv ? ep.command.argv.join(' ') : null;
  const workdirFiles = ep && ep.workdirFiles ? Object.keys(ep.workdirFiles) : [];
  const env = request.environment || {};
  const pipPkgs = (env.pipPackages || []).join(', ');
  const extras = (env.extras || []).join(', ');
  const envVars = env.envVars ? Object.entries(env.envVars).map(([k, v]) => `${k}=${v}`).join(', ') : '';
  const dockerfile = env.dockerfile || '';
  const pythonVersion = env.pythonVersion || '';
  const bundlePath = request.bundleGcsPath || '';
  const bundleHash = request.bundleHash || '';
  const replicas = request.replicas || 1;
  const timeout = formatProtoTimeout(request.timeout);
  const schedTimeout = formatProtoTimeout(request.schedulingTimeout);
  const ports = (request.ports || []).join(', ');
  const maxTaskFailures = request.maxTaskFailures || 0;
  const maxRetriesFailure = request.maxRetriesFailure || 0;
  const maxRetriesPreemption = request.maxRetriesPreemption;
  const constraints = (request.constraints || []).map(c => {
    const opNames = { 0: '=', 1: '!=', 2: 'exists', 3: '!exists', 4: '>', 5: '>=', 6: '<', 7: '<=' };
    const op = opNames[c.op] || c.op || '=';
    const val = c.value ? (c.value.stringValue || c.value.intValue || c.value.floatValue || '') : '';
    return `${c.key} ${op} ${val}`;
  }).join(', ');
  const cosched = request.coscheduling && request.coscheduling.groupBy ? `group_by: ${request.coscheduling.groupBy}` : '';

  const res = request.resources || {};
  const device = formatDevice(res.device);

  return html`
    <${Collapsible} title="Job Request">
      <${DetailRow} label="Command" value=${cmd} />
      ${workdirFiles.length > 0 && html`<${DetailRow} label="Workdir Files" value=${workdirFiles.join(', ')} />`}
      <${DetailRow} label="Replicas" value=${replicas > 1 ? replicas : null} />
      <${DetailRow} label="Device" value=${device} />
      <${DetailRow} label="CPU" value=${res.cpu || null} />
      <${DetailRow} label="Memory" value=${res.memoryBytes ? formatBytes(parseInt(res.memoryBytes)) : null} />
      <${DetailRow} label="Disk" value=${res.diskBytes ? formatBytes(parseInt(res.diskBytes)) : null} />
      <${DetailRow} label="Regions" value=${(res.regions || []).join(', ') || null} />
      <${DetailRow} label="Timeout" value=${timeout} />
      <${DetailRow} label="Scheduling Timeout" value=${schedTimeout} />
      <${DetailRow} label="Ports" value=${ports || null} />
      <${DetailRow} label="Constraints" value=${constraints || null} />
      <${DetailRow} label="Coscheduling" value=${cosched || null} />
      <${DetailRow} label="Max Task Failures" value=${maxTaskFailures || null} />
      <${DetailRow} label="Max Retries (failure)" value=${maxRetriesFailure || null} />
      <${DetailRow} label="Max Retries (preemption)" value=${maxRetriesPreemption || null} />
      <${DetailRow} label="Bundle" value=${bundlePath || null} />
      <${DetailRow} label="Bundle Hash" value=${bundleHash || null} />
      <${DetailRow} label="Pip Packages" value=${pipPkgs || null} />
      <${DetailRow} label="Extras" value=${extras || null} />
      <${DetailRow} label="Python Version" value=${pythonVersion || null} />
      <${DetailRow} label="Env Vars" value=${envVars || null} />
      ${dockerfile && html`
        <div style="padding:6px 0">
          <span style="color:#57606a;font-size:13px">Dockerfile</span>
          <pre style="margin:8px 0 0;padding:10px;background:#f6f8fa;border-radius:4px;font-size:12px;overflow-x:auto;white-space:pre-wrap">${dockerfile}</pre>
        </div>
      `}
    <//>
  `;
}


function JobDetailApp() {
  const [job, setJob] = useState(null);
  const [jobRequest, setJobRequest] = useState(null);
  const [tasks, setTasks] = useState([]);
  const [error, setError] = useState(null);
  const [selectedTaskId, setSelectedTaskId] = useState(null);
  const [selectedAttemptId, setSelectedAttemptId] = useState(-1);
  const [taskLogs, setTaskLogs] = useState('Loading logs...');
  const [taskLogStatus, setTaskLogStatus] = useState('');
  const [expandedTasks, setExpandedTasks] = useState(new Set());
  const tasksRef = useRef([]);

  const toggleExpanded = useCallback((taskId) => {
    setExpandedTasks(prev => {
      const next = new Set(prev);
      if (next.has(taskId)) {
        next.delete(taskId);
      } else {
        next.add(taskId);
      }
      return next;
    });
  }, []);

  const fetchTaskLogs = useCallback(async (taskId, tasksList, attemptId = -1) => {
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
      const params = { id: taskId };
      if (attemptId >= 0) {
        params.attemptId = attemptId;
      }
      const resp = await controllerRpc('GetTaskLogs', params);
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
        const [jobResp, tasksResp] = await Promise.all([
          controllerRpc('GetJobStatus', { jobId }),
          controllerRpc('ListTasks', { jobId }),
        ]);

        const j = jobResp.job;
        if (!j) {
          setError('Job not found');
          return;
        }

        const found = {
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
        };

        const tasksList = (tasksResp.tasks || []).map(t => ({
          task_id: t.taskId,
          task_index: taskIndexFromId(t.taskId),
          state: stateToName(t.state),
          worker_id: t.workerId || '',
          started_at_ms: timestampFromProto(t.startedAt),
          finished_at_ms: timestampFromProto(t.finishedAt),
          exit_code: t.exitCode,
          error: t.error || '',
          pending_reason: t.pendingReason || '',
          current_attempt_id: t.currentAttemptId || 0,
          attempts: (t.attempts || []).map(a => ({
            attempt_id: a.attemptId,
            worker_id: a.workerId || '',
            state: stateToName(a.state),
            started_at_ms: timestampFromProto(a.startedAt),
            finished_at_ms: timestampFromProto(a.finishedAt),
            exit_code: a.exitCode,
            error: a.error || '',
            is_worker_failure: a.isWorkerFailure,
          })),
        }));

        setJob(found);
        setJobRequest(jobResp.request || null);
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
            setSelectedAttemptId(-1);
            fetchTaskLogs(autoTask.task_id, tasksList, -1);
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

  // Get currently selected task for attempt dropdown
  const selectedTask = tasks.find(t => t.task_id === selectedTaskId);

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

    <${JobRequestDetail} request=${jobRequest} />

    <h2>Task Logs</h2>
    <div style="margin-bottom:10px;display:flex;align-items:center;gap:12px;flex-wrap:wrap">
      <select value=${selectedTaskId ?? ''} onChange=${e => {
        const value = e.target.value;
        setSelectedTaskId(value || null);
        setSelectedAttemptId(-1);
        fetchTaskLogs(value || null, tasksRef.current, -1);
      }}>
        <option value="">Select a task...</option>
        ${tasks.map(t => html`<option value=${t.task_id}>Task ${t.task_index ?? '-'} (${t.state})</option>`)}
      </select>
      ${selectedTask && selectedTask.attempts.length > 1 && html`
        <select value=${selectedAttemptId} onChange=${e => {
          const value = parseInt(e.target.value, 10);
          setSelectedAttemptId(value);
          fetchTaskLogs(selectedTaskId, tasksRef.current, value);
        }}>
          <option value="-1">All attempts</option>
          ${selectedTask.attempts.map(a => html`
            <option value=${a.attempt_id}>
              Attempt #${a.attempt_id} (${a.state})${a.attempt_id === selectedTask.current_attempt_id ? ' *' : ''}
            </option>
          `)}
        </select>
      `}
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
        ${tasks.flatMap(t => {
          const mainRow = html`<tr key=${t.task_id}>
            <td>${t.task_id}</td>
            <td>${t.task_index ?? '-'}</td>
            <td class=${getStateClass(t.state)}>${t.state}${t.pending_reason ? html`<br/><span class="pending-reason">${t.pending_reason}</span>` : ''}</td>
            <td>${t.worker_id || '-'}</td>
            <td style="cursor:${t.attempts.length > 1 ? 'pointer' : 'default'}" onClick=${() => t.attempts.length > 1 && toggleExpanded(t.task_id)}>
              ${t.attempts.length > 1 ? (expandedTasks.has(t.task_id) ? '\u25BC ' : '\u25B6 ') : ''}${t.attempts.length}
            </td>
            <td>${formatTimestamp(t.started_at_ms)}</td>
            <td>${formatDuration(t.started_at_ms, t.finished_at_ms)}</td>
            <td>${TERMINAL_STATES.has(t.state) && t.exit_code !== null && t.exit_code !== undefined ? t.exit_code : '-'}</td>
            <td>${t.error || '-'}</td>
          </tr>`;

          if (!expandedTasks.has(t.task_id) || t.attempts.length <= 1) {
            return [mainRow];
          }

          const attemptRows = t.attempts.map(a => html`
            <tr key=${t.task_id + '-attempt-' + a.attempt_id} style="background:#f6f8fa">
              <td style="padding-left:24px;color:#57606a">\u2514\u2500 #${a.attempt_id}</td>
              <td></td>
              <td class=${getStateClass(a.state)}>${a.state}${a.is_worker_failure ? html`<br/><span style="font-size:11px;color:#9a6700">(worker failure)</span>` : ''}</td>
              <td style="font-size:12px">${a.worker_id || '-'}</td>
              <td></td>
              <td>${formatTimestamp(a.started_at_ms)}</td>
              <td>${formatDuration(a.started_at_ms, a.finished_at_ms)}</td>
              <td>${TERMINAL_STATES.has(a.state) && a.exit_code !== null && a.exit_code !== undefined ? a.exit_code : '-'}</td>
              <td style="font-size:12px">${a.error || '-'}</td>
            </tr>
          `);

          return [mainRow, ...attemptRows];
        })}
      </tbody>
    </table>
  `;
}

render(html`<${JobDetailApp} />`, document.getElementById('root'));
