/**
 * Machine detail page — calls GetMachineStatus with the URL identifier
 * (VM ID or worker ID) and renders VM info, worker info, bootstrap logs,
 * and worker daemon logs.
 *
 * URL: /vm/{identifier}
 */
import { h, render } from 'preact';
import { useState, useEffect, useRef, useCallback } from 'preact/hooks';
import htm from 'htm';
import { controllerRpc } from '/static/shared/rpc.js';
import { formatVmState, formatBytes, formatRelativeTime } from '/static/shared/utils.js';
import { InfoRow, InfoCard } from '/static/shared/components.js';

const html = htm.bind(h);

const identifier = decodeURIComponent(window.location.pathname.split('/vm/')[1]);

function VmDetailApp() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const logsRef = useRef(null);
  const workerLogsRef = useRef(null);

  const refresh = useCallback(async () => {
    try {
      const resp = await controllerRpc('GetMachineStatus', { id: identifier });
      setData(resp);
      setError(null);
    } catch (e) {
      setError('Failed to load details: ' + e.message);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  useEffect(() => {
    if (logsRef.current) logsRef.current.scrollTop = logsRef.current.scrollHeight;
  }, [data?.bootstrapLogs]);

  useEffect(() => {
    if (workerLogsRef.current) workerLogsRef.current.scrollTop = workerLogsRef.current.scrollHeight;
  }, [data?.workerLogs]);

  if (error) {
    return html`
      <a href="/#fleet" class="back-link">\u2190 Back to Dashboard</a>
      <h1>Machine: ${identifier}</h1>
      <div class="error-message">${error}</div>
    `;
  }

  if (!data) {
    return html`
      <a href="/#fleet" class="back-link">\u2190 Back to Dashboard</a>
      <h1>Machine: ${identifier}</h1>
      <p>Loading...</p>
    `;
  }

  const vm = data.vm;
  const worker = data.worker;
  const bootstrapLogs = data.bootstrapLogs || '';
  const workerLogs = data.workerLogs || [];

  const state = vm ? formatVmState(vm.state) : '-';
  const stateClass = state !== '-' ? 'status-' + state : '';
  const displayId = (vm && vm.vmId) || identifier;

  const workerHealthy = worker ? worker.healthy : false;
  const workerCpu = worker && worker.metadata ? (worker.metadata.cpuCount || '-') : '-';
  const workerMem = worker && worker.metadata
    ? formatBytes(parseInt(worker.metadata.memoryBytes || 0))
    : '-';
  const workerHeartbeat = worker && worker.lastHeartbeat
    ? formatRelativeTime(parseInt(worker.lastHeartbeat.epochMs || 0))
    : '-';
  const workerTasks = worker ? (worker.runningJobIds || []).length : 0;

  return html`
    <a href="/#fleet" class="back-link">\u2190 Back to Dashboard</a>
    <h1>Machine: ${displayId}</h1>

    <div class="info-grid">
      ${vm && html`
        <${InfoCard} title="VM Info">
          <${InfoRow} label="VM ID" value=${vm.vmId} />
          <${InfoRow} label="State" value=${state} valueClass=${stateClass} />
          <${InfoRow} label="Address" value=${vm.address || '-'} />
          <${InfoRow} label="Init Phase" value=${vm.initPhase || '-'} />
        <//>
      `}
      ${vm && html`
        <${InfoCard} title="Scale Group">
          <${InfoRow} label="Group" value=${data.scaleGroup || '-'} />
          <${InfoRow} label="Slice" value=${vm.sliceId || '-'} />
        <//>
      `}
      ${worker && html`
        <${InfoCard} title="Worker">
          <${InfoRow} label="Worker ID" value=${worker.workerId} />
          <${InfoRow} label="Healthy" value=${workerHealthy ? 'Yes' : 'No'}
            valueClass=${workerHealthy ? 'healthy' : 'unhealthy'} />
          <${InfoRow} label="CPU Cores" value=${workerCpu} />
          <${InfoRow} label="Memory" value=${workerMem} />
          <${InfoRow} label="Running Tasks" value=${workerTasks} />
          <${InfoRow} label="Last Heartbeat" value=${workerHeartbeat} />
          ${worker.statusMessage && html`
            <${InfoRow} label="Status" value=${worker.statusMessage} />
          `}
        <//>
      `}
      ${!vm && !worker && html`
        <${InfoCard} title="Not Found">
          <${InfoRow} label="ID" value=${identifier} />
          <${InfoRow} label="Status" value="No matching VM or worker found" />
        <//>
      `}
    </div>

    ${vm && vm.initError && html`
      <div class="error-message"><strong>Init Error:</strong> ${vm.initError}</div>
    `}

    ${bootstrapLogs && html`
      <h2>Bootstrap Logs</h2>
      <pre ref=${logsRef} style="background:white;padding:15px;border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,0.12);max-height:600px;overflow-y:auto;font-size:12px;white-space:pre-wrap">${bootstrapLogs}</pre>
    `}

    ${workerLogs.length > 0 && html`
      <h2>Worker Daemon Logs</h2>
      <div ref=${workerLogsRef} id="log-container" style="max-height:400px;overflow-y:auto;background:white;padding:15px;border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,0.12)">
        ${workerLogs.map(log => html`
          <div class="log-line ${(log.level || '').toLowerCase()}" style="font-size:12px;font-family:monospace;white-space:pre-wrap">
            <span style="color:#6b7280">${log.timestamp || ''}</span>${' '}
            <span style="font-weight:bold">${log.level || ''}</span>${' '}
            ${log.message || ''}
          </div>
        `)}
      </div>
    `}
  `;
}

render(html`<${VmDetailApp} />`, document.getElementById('root'));
