/**
 * Worker detail page — keyed entirely by worker ID.
 *
 * Workers and VMs are independent: this page shows worker state only.
 * VM status lives on the Autoscaler tab.
 *
 * URL: /worker/{worker_id}
 */
import { h, render } from 'preact';
import { useState, useEffect, useRef, useCallback } from 'preact/hooks';
import htm from 'htm';
import { controllerRpc } from '/static/shared/rpc.js';
import { formatBytes, formatRelativeTime, stateToName, formatDuration } from '/static/shared/utils.js';
import { MetricCard, InlineGauge, Field, Section, Sparkline, formatMbPair, formatRate } from '/static/shared/components.js';

const html = htm.bind(h);

const identifier = decodeURIComponent(window.location.pathname.split('/worker/')[1]);

function StatusBadge({ healthy }) {
  const state = healthy ? 'ready' : 'unhealthy';
  return html`<span class=${'worker-detail-status-badge status-' + state}>
    <span class=${'vm-state-indicator ' + state}></span>${state}
  </span>`;
}

function WorkerDetailApp() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
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
    if (workerLogsRef.current) workerLogsRef.current.scrollTop = workerLogsRef.current.scrollHeight;
  }, [data?.workerLogEntries]);

  // Error state
  if (error && !data) {
    return html`
      <div class="worker-detail">
        <a href="/#workers" class="back-link">\u2190 Workers</a>
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
        <a href="/#workers" class="back-link">\u2190 Workers</a>
        <div class="worker-detail-header">
          <h1 class="worker-detail-header__title">${identifier}</h1>
        </div>
        <div class="worker-detail-loading">
          <div class="worker-detail-loading__spinner"></div>
          <span>Loading worker details\u2026</span>
        </div>
      </div>`;
  }

  const worker = data.worker;
  const workerLogs = data.workerLogEntries || [];
  const liveRes = data.currentResources || null;
  const resourceHistory = data.resourceHistory || [];

  if (!worker) {
    return html`
      <div class="worker-detail">
        <a href="/#workers" class="back-link">\u2190 Workers</a>
        <div class="worker-detail-header">
          <h1 class="worker-detail-header__title">${identifier}</h1>
        </div>
        <div class="worker-detail-section">
          <div class="worker-detail-section__empty">
            No worker found for <code>${identifier}</code>.
          </div>
        </div>
      </div>`;
  }

  const recentTasks = data.recentTasks || [];
  const runningTaskCount = recentTasks.filter(t => stateToName(t.state) === 'running').length;

  // Worker resource info (static capacity from metadata)
  const workerCpu = worker.metadata ? (worker.metadata.cpuCount || null) : null;
  const workerMem = worker.metadata ? parseInt(worker.metadata.memoryBytes || 0) : 0;
  const workerDisk = worker.metadata ? parseInt(worker.metadata.diskBytes || 0) : 0;
  const workerHeartbeat = worker.lastHeartbeat
    ? formatRelativeTime(parseInt(worker.lastHeartbeat.epochMs || 0)) : null;

  // Live resource utilization from heartbeat snapshots
  const liveCpuPct = liveRes ? (liveRes.cpuPercent || 0) : null;
  const liveMemUsed = liveRes ? parseInt(liveRes.memoryUsedBytes || 0) : 0;
  const liveMemTotal = liveRes ? parseInt(liveRes.memoryTotalBytes || 0) : 0;
  const liveDiskUsed = liveRes ? parseInt(liveRes.diskUsedBytes || 0) : 0;
  const liveDiskTotal = liveRes ? parseInt(liveRes.diskTotalBytes || 0) : 0;
  const liveNetRecv = liveRes ? parseInt(liveRes.netRecvBps || 0) : 0;
  const liveNetSent = liveRes ? parseInt(liveRes.netSentBps || 0) : 0;

  // Accelerator info from worker metadata
  let accelDisplay = null;
  if (worker.metadata) {
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
      <a href="/#workers" class="back-link">\u2190 Workers</a>

      <div class="worker-detail-header">
        <div class="worker-detail-header__top">
          <h1 class="worker-detail-header__title">${worker.workerId}</h1>
          <${StatusBadge} healthy=${worker.healthy} />
          <button class="worker-detail-header__refresh" onClick=${refresh} title="Refresh">\u21bb</button>
        </div>
        ${worker.address && html`
          <div class="worker-detail-header__meta">
            <span class="worker-detail-header__tag">${worker.address}</span>
          </div>
        `}
      </div>

      ${!worker.healthy && worker.statusMessage && html`
        <div class="error-message">${worker.statusMessage}</div>
      `}

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

      <div class="worker-detail-grid">
        <${Section} title="Identity">
          <dl class="worker-detail-fields">
            <${Field} label="Worker ID" value=${worker.workerId} mono />
            ${worker.address && html`<${Field} label="Address" value=${worker.address} mono />`}
          </dl>
        <//>

        <${Section} title="Health & Resources">
          <dl class="worker-detail-fields">
            <${Field} label="Healthy" value=${worker.healthy ? 'Yes' : 'No'}
              valueClass=${worker.healthy ? 'healthy' : 'unhealthy'} />
            ${workerHeartbeat && html`<${Field} label="Last Heartbeat" value=${workerHeartbeat} />`}
            ${workerCpu && html`<${Field} label="CPU Cores" value=${workerCpu} />`}
            ${accelDisplay && html`<${Field} label="Accelerator" value=${accelDisplay} />`}
            ${!liveRes && workerMem > 0 && html`<${Field} label="Memory" value=${formatBytes(workerMem)} />`}
            ${!liveRes && workerDisk > 0 && html`<${Field} label="Disk" value=${formatBytes(workerDisk)} />`}
          </dl>
        <//>
      </div>

      ${liveRes && html`
        <div class="utilization-panel">
          <h2 class="utilization-panel__title">Live Utilization</h2>
          <div class="utilization-panel__grid">
            <div class="utilization-metric">
              <div class="utilization-metric__header">
                <span class="utilization-metric__label">CPU</span>
                <span class="utilization-metric__value utilization-metric__value--accent">${(liveCpuPct || 0) + '%'}</span>
              </div>
              <div class="utilization-metric__chart">
                <${Sparkline} values=${resourceHistory.map(s => s.cpuPercent || 0)}
                  max=${100} width=${240} height=${40}
                  color="var(--color-accent)"
                  fillColor="rgba(9,105,218,0.10)" />
              </div>
            </div>

            ${liveMemTotal > 0 && html`
              <div class="utilization-metric">
                <div class="utilization-metric__header">
                  <span class="utilization-metric__label">Memory</span>
                  <span class="utilization-metric__value utilization-metric__value--success">${formatBytes(liveMemUsed) + ' / ' + formatBytes(liveMemTotal)}</span>
                </div>
                <div class="utilization-metric__chart">
                  <${Sparkline} values=${resourceHistory.map(s => parseInt(s.memoryUsedBytes || 0))}
                    max=${liveMemTotal} width=${240} height=${40}
                    color="var(--color-success)"
                    fillColor="rgba(26,127,55,0.10)" />
                </div>
              </div>
            `}

            ${liveDiskTotal > 0 && html`
              <div class="utilization-metric">
                <div class="utilization-metric__header">
                  <span class="utilization-metric__label">Disk</span>
                  <span class="utilization-metric__value utilization-metric__value--warning">${formatBytes(liveDiskUsed) + ' / ' + formatBytes(liveDiskTotal)}</span>
                </div>
                <div class="utilization-metric__chart">
                  <${Sparkline} values=${resourceHistory.map(s => parseInt(s.diskUsedBytes || 0))}
                    max=${liveDiskTotal} width=${240} height=${40}
                    color="var(--color-warning)"
                    fillColor="rgba(154,103,0,0.10)" />
                </div>
              </div>
            `}

            <div class="utilization-metric">
              <div class="utilization-metric__header">
                <span class="utilization-metric__label">Network</span>
                <span class="utilization-metric__value utilization-metric__value--purple">
                  ${'↓ ' + formatRate(liveNetRecv) + '  ↑ ' + formatRate(liveNetSent)}
                </span>
              </div>
              <div class="utilization-metric__chart">
                <${Sparkline} values=${resourceHistory.map(s => parseInt(s.netRecvBps || 0) + parseInt(s.netSentBps || 0))}
                  width=${240} height=${40}
                  color="var(--color-purple, #8250df)"
                  fillColor="rgba(130,80,223,0.10)" />
              </div>
            </div>
          </div>
        </div>
      `}

      ${'' /* --- Task History --- */}
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
                  <td>${ru && memMb ? (() => {
                    const peakMb = parseInt(ru.memoryPeakMb || memMb);
                    return html`<${InlineGauge} value=${memMb} max=${peakMb}
                      label=${formatMbPair(memMb, peakMb)} />`;
                  })() : '-'}</td>
                  <td>${ru && cpuPct ? html`<${InlineGauge} value=${cpuPct} max=${100}
                    label=${cpuPct + '%'} />` : '-'}</td>
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

      ${'' /* --- Worker Daemon Logs --- */}
      <div class="worker-detail-logs-section">
        <h2 class="worker-detail-logs-section__title">Worker Daemon Logs</h2>
        ${workerLogs.length > 0 ? html`
          <div ref=${workerLogsRef} class="worker-detail-logs-container">
            ${workerLogs.map(log => {
              const ms = log.timestamp ? (log.timestamp.epochMs || 0) : 0;
              const time = ms ? new Date(Number(ms)).toLocaleTimeString() : '';
              return html`
                <div class="log-line">
                  <span class="log-ts">${time}</span>${' '}
                  <span class="log-data">${log.data || ''}</span>
                </div>`;
            })}
          </div>
        ` : html`
          <div class="worker-detail-logs-container" style="padding:20px;color:#57606a;font-size:13px;font-style:italic">
            ${worker.healthy ? 'No log records returned.' : 'Logs unavailable (worker unhealthy).'}
          </div>
        `}
      </div>
    </div>`;
}

render(html`<${WorkerDetailApp} />`, document.getElementById('root'));
