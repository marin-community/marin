/**
 * Unified worker detail page — resolves a VM ID or worker ID via
 * GetWorkerStatus and renders all available information in a single view.
 *
 * URL: /worker/{identifier}
 */
import { h, render } from 'preact';
import { useState, useEffect, useRef, useCallback } from 'preact/hooks';
import htm from 'htm';
import { controllerRpc } from '/static/shared/rpc.js';
import { formatVmState, formatBytes, formatRelativeTime, formatAcceleratorDisplay, stateToName, formatDuration } from '/static/shared/utils.js';
import { MetricCard, ResourceSection, Gauge, InlineGauge, Field, Section } from '/static/shared/components.js';

const html = htm.bind(h);

const identifier = decodeURIComponent(window.location.pathname.split('/worker/')[1]);

function StatusBadge({ state }) {
  if (!state || state === '-') return null;
  return html`<span class=${'worker-detail-status-badge status-' + state}>
    <span class=${'vm-state-indicator ' + state}></span>${state}
  </span>`;
}

function WorkerDetailApp() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [logsExpanded, setLogsExpanded] = useState(false);
  const bootstrapRef = useRef(null);
  const workerLogsRef = useRef(null);

  const refresh = useCallback(async () => {
    try {
      const resp = await controllerRpc('GetWorkerStatus', { id: identifier });
      setData(resp);
      setError(null);
    } catch (e) {
      setError(e.message);
    }
  }, []);

  useEffect(() => { refresh(); const iv = setInterval(refresh, 5000); return () => clearInterval(iv); }, [refresh]);

  useEffect(() => {
    if (bootstrapRef.current) bootstrapRef.current.scrollTop = bootstrapRef.current.scrollHeight;
  }, [data?.bootstrapLogs]);

  useEffect(() => {
    if (workerLogsRef.current) workerLogsRef.current.scrollTop = workerLogsRef.current.scrollHeight;
  }, [data?.workerLogs]);

  // Error state
  if (error && !data) {
    return html`
      <div class="worker-detail">
        <a href="/#fleet" class="back-link">\u2190 Fleet</a>
        <div class="worker-detail-header">
          <h1 class="worker-detail-header__title">${identifier}</h1>
        </div>
        <div class="error-message">Could not load worker: ${error}</div>
      </div>`;
  }

  // Loading state
  if (!data) {
    return html`
      <div class="worker-detail">
        <a href="/#fleet" class="back-link">\u2190 Fleet</a>
        <div class="worker-detail-header">
          <h1 class="worker-detail-header__title">${identifier}</h1>
        </div>
        <div class="worker-detail-loading">
          <div class="worker-detail-loading__spinner"></div>
          <span>Loading worker details\u2026</span>
        </div>
      </div>`;
  }

  const vm = data.vm;
  const worker = data.worker;
  const bootstrapLogs = data.bootstrapLogs || '';
  const workerLogs = data.workerLogs || [];
  const liveRes = data.currentResources || null;

  // Compute the effective state: worker health overrides VM state when relevant
  let effectiveState = 'unknown';
  if (vm) {
    effectiveState = formatVmState(vm.state);
    if (effectiveState === 'ready' && worker && !worker.healthy) effectiveState = 'unhealthy';
  } else if (worker) {
    effectiveState = worker.healthy ? 'ready' : 'unhealthy';
  }

  const displayId = (vm && vm.vmId) || (worker && worker.workerId) || identifier;

  // Worker resource info (static capacity from metadata)
  const workerCpu = worker && worker.metadata ? (worker.metadata.cpuCount || null) : null;
  const workerMem = worker && worker.metadata ? parseInt(worker.metadata.memoryBytes || 0) : 0;
  const workerDisk = worker && worker.metadata ? parseInt(worker.metadata.diskBytes || 0) : 0;
  const workerHeartbeat = worker && worker.lastHeartbeat
    ? formatRelativeTime(parseInt(worker.lastHeartbeat.epochMs || 0)) : null;
  const recentTasks = data.recentTasks || [];
  const runningTaskCount = recentTasks.filter(t => stateToName(t.state) === 'running').length;

  // Live resource utilization from heartbeat snapshots
  const liveCpuPct = liveRes ? (liveRes.cpuPercent || 0) : null;
  const liveMemUsed = liveRes ? parseInt(liveRes.memoryUsedBytes || 0) : 0;
  const liveMemTotal = liveRes ? parseInt(liveRes.memoryTotalBytes || 0) : 0;
  const liveDiskUsed = liveRes ? parseInt(liveRes.diskUsedBytes || 0) : 0;
  const liveDiskTotal = liveRes ? parseInt(liveRes.diskTotalBytes || 0) : 0;

  // Accelerator info from worker metadata
  let accelDisplay = null;
  if (worker && worker.metadata) {
    const md = worker.metadata;
    if (md.device) {
      if (md.device.tpu) accelDisplay = 'TPU: ' + (md.device.tpu.variant || 'unknown');
      else if (md.device.gpu) accelDisplay = 'GPU: ' + (md.device.gpu.count || 1) + 'x ' + (md.device.gpu.variant || 'unknown');
    } else if (md.gpuCount > 0) {
      const name = md.gpuName || 'GPU';
      const mem = md.gpuMemoryMb ? ` (${Math.round(md.gpuMemoryMb / 1024)}GB)` : '';
      accelDisplay = `GPU: ${md.gpuCount}x ${name}${mem}`;
    }
  }

  return html`
    <div class="worker-detail">
      <a href="/#fleet" class="back-link">\u2190 Fleet</a>

      <div class="worker-detail-header">
        <div class="worker-detail-header__top">
          <h1 class="worker-detail-header__title">${displayId}</h1>
          <${StatusBadge} state=${effectiveState} />
          <button class="worker-detail-header__refresh" onClick=${refresh} title="Refresh">\u21bb</button>
        </div>
        ${(data.scaleGroup || (vm && vm.sliceId)) && html`
          <div class="worker-detail-header__meta">
            ${data.scaleGroup && html`<span class="worker-detail-header__tag">${data.scaleGroup}</span>`}
            ${vm && vm.sliceId && html`<span class="worker-detail-header__tag">${vm.sliceId}</span>`}
            ${vm && vm.zone && html`<span class="worker-detail-header__tag">${vm.zone}</span>`}
          </div>
        `}
      </div>

      ${vm && vm.initError && html`
        <div class="error-message">${vm.initError}</div>
      `}
      ${worker && !worker.healthy && worker.statusMessage && html`
        <div class="error-message">${worker.statusMessage}</div>
      `}

      ${worker && html`
        <div class="metric-row">
          <${MetricCard} value=${runningTaskCount} label="Running Tasks"
            valueClass=${runningTaskCount > 0 ? 'accent' : undefined} />
          ${liveCpuPct !== null
            ? html`<${MetricCard} value=${liveCpuPct + '%'} label="CPU Usage"
                valueClass=${liveCpuPct >= 90 ? 'danger' : liveCpuPct >= 70 ? 'warning' : undefined} />`
            : workerCpu && html`<${MetricCard} value=${workerCpu} label="CPU Cores" />`}
          ${liveMemTotal > 0
            ? html`<${MetricCard} value=${formatBytes(liveMemUsed) + ' / ' + formatBytes(liveMemTotal)} label="Memory" />`
            : workerMem > 0 && html`<${MetricCard} value=${formatBytes(workerMem)} label="Memory" />`}
          ${accelDisplay ? html`<${MetricCard} value=${accelDisplay} label="Accelerator" />`
            : html`<${MetricCard} value="None" label="Accelerator" />`}
        </div>
      `}

      <div class="worker-detail-grid">
        ${'' /* --- Identity & Infrastructure --- */}
        <${Section} title="Identity">
          <dl class="worker-detail-fields">
            ${vm && html`<${Field} label="VM ID" value=${vm.vmId} mono />`}
            ${worker && html`<${Field} label="Worker ID" value=${worker.workerId} mono />`}
            ${vm && html`<${Field} label="Address" value=${vm.address} mono />`}
            ${!vm && worker && worker.address && html`<${Field} label="Address" value=${worker.address} mono />`}
            ${vm && html`<${Field} label="VM State" value=${formatVmState(vm.state)} valueClass=${'status-' + formatVmState(vm.state)} />`}
            ${vm && vm.initPhase && html`<${Field} label="Init Phase" value=${vm.initPhase} />`}
          </dl>
        <//>

        ${'' /* --- Health & Resources --- */}
        ${worker ? html`
          <${Section} title="Health & Resources">
            <dl class="worker-detail-fields">
              <${Field} label="Healthy" value=${worker.healthy ? 'Yes' : 'No'}
                valueClass=${worker.healthy ? 'healthy' : 'unhealthy'} />
              ${workerHeartbeat && html`<${Field} label="Last Heartbeat" value=${workerHeartbeat} />`}
              ${workerCpu && html`<${Field} label="CPU Cores" value=${workerCpu} />`}
              ${accelDisplay && html`<${Field} label="Accelerator" value=${accelDisplay} />`}
            </dl>
            ${liveRes && html`
              <${ResourceSection} title="Live Utilization">
                <${Gauge} label="CPU" value=${liveCpuPct || 0} max=${100} format="percent" />
                ${liveMemTotal > 0 && html`
                  <${Gauge} label="Memory" value=${liveMemUsed} max=${liveMemTotal} format="bytes" />
                `}
                ${liveDiskTotal > 0 && html`
                  <${Gauge} label="Disk" value=${liveDiskUsed} max=${liveDiskTotal} format="bytes" />
                `}
              <//>
            `}
            ${!liveRes && html`
              <dl class="worker-detail-fields">
                ${workerMem > 0 && html`<${Field} label="Memory" value=${formatBytes(workerMem)} />`}
                ${workerDisk > 0 && html`<${Field} label="Disk" value=${formatBytes(workerDisk)} />`}
              </dl>
            `}
          <//>
        ` : vm ? html`
          <${Section} title="Health & Resources" muted>
            <div class="worker-detail-section__empty">Worker has not registered yet.\u2002Bootstrap logs below may show progress.</div>
          <//>
        ` : null}

      </div>

      ${'' /* --- Task History --- */}
      ${worker && html`
        <div class="worker-detail-logs-section">
          <h2 class="worker-detail-logs-section__title">Task History (${recentTasks.length})${runningTaskCount > 0 ? html`<span class="worker-detail-logs-section__running"> \u2014 ${runningTaskCount} running</span>` : null}</h2>
          ${recentTasks.length > 0 ? html`
            <table class="worker-detail-task-table">
              <thead><tr>
                <th>Task</th><th>Job</th><th>State</th><th>Mem</th><th>CPU</th><th>Started</th><th>Duration</th><th>Error</th>
              </tr></thead>
              <tbody>
                ${recentTasks.map(t => {
                  const taskState = stateToName(t.state);
                  const startMs = t.startedAt && t.startedAt.epochMs ? parseInt(t.startedAt.epochMs) : 0;
                  const endMs = t.finishedAt && t.finishedAt.epochMs ? parseInt(t.finishedAt.epochMs) : 0;
                  const duration = startMs ? formatDuration(startMs, endMs || undefined) : '-';
                  const started = startMs ? formatRelativeTime(startMs) : '-';
                  const jobId = t.taskId.replace(/\/[^/]*$/, '');
                  const ru = t.resourceUsage || null;
                  const memMb = ru ? parseInt(ru.memoryMb || 0) : 0;
                  const cpuPct = ru ? (ru.cpuPercent || 0) : 0;
                  return html`<tr>
                    <td><a href=${'/job/' + encodeURIComponent(jobId)} class="job-link">${t.taskId}</a></td>
                    <td>${jobId || '-'}</td>
                    <td><span class=${'status-' + taskState}>${taskState}</span></td>
                    <td>${ru && memMb ? html`<${InlineGauge} value=${memMb} max=${parseInt(ru.memoryPeakMb || memMb)} label=${memMb + ' MB'} />` : '-'}</td>
                    <td>${ru && cpuPct ? html`<${InlineGauge} value=${cpuPct} max=${100} label=${cpuPct + '%'} />` : '-'}</td>
                    <td>${started}</td>
                    <td>${duration}</td>
                    <td class="worker-detail-task-error">${t.error || '-'}</td>
                  </tr>`;
                })}
              </tbody>
            </table>
          ` : html`
            <div class="worker-detail-logs-container" style="padding:20px;color:#57606a;font-size:13px;font-style:italic">
              No tasks have run on this worker yet.
            </div>
          `}
        </div>
      `}

      ${'' /* --- Bootstrap Logs (VM initializing / pre-worker) --- */}
      ${bootstrapLogs && html`
        <div class="worker-detail-logs-section">
          <button class="worker-detail-logs-toggle" onClick=${() => setLogsExpanded(!logsExpanded)}>
            <span class=${'worker-detail-logs-toggle__arrow' + (logsExpanded ? ' expanded' : '')}>\u25b6</span>
            Bootstrap Logs
            <span class="worker-detail-logs-toggle__hint">${bootstrapLogs.split('\n').length} lines</span>
          </button>
          ${logsExpanded && html`
            <pre ref=${bootstrapRef} class="worker-detail-logs-pre">${bootstrapLogs}</pre>
          `}
        </div>
      `}

      ${'' /* --- Worker Daemon Logs --- */}
      ${worker && html`
        <div class="worker-detail-logs-section">
          <h2 class="worker-detail-logs-section__title">Worker Daemon Logs</h2>
          ${workerLogs.length > 0 ? html`
            <div ref=${workerLogsRef} class="worker-detail-logs-container">
              ${workerLogs.map(log => html`
                <div class=${'log-line ' + (log.level || '').toLowerCase()}>
                  <span class="log-ts">${log.timestamp || ''}</span>${' '}
                  <span class="log-level">${log.level || ''}</span>${' '}
                  ${log.message || ''}
                </div>
              `)}
            </div>
          ` : html`
            <div class="worker-detail-logs-container" style="padding:20px;color:#57606a;font-size:13px;font-style:italic">
              ${worker.healthy ? 'No log records returned.' : 'Logs unavailable (worker unhealthy).'}
            </div>
          `}
        </div>
      `}

      ${!vm && !worker && html`
        <div class="worker-detail-section">
          <div class="worker-detail-section__empty">
            No matching VM or worker found for <code>${identifier}</code>.
            The worker may have been terminated or the ID may be incorrect.
          </div>
        </div>
      `}
    </div>`;
}

render(html`<${WorkerDetailApp} />`, document.getElementById('root'));
