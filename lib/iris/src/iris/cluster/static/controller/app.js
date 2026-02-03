import { h, render } from 'preact';
import { useState, useEffect, useCallback } from 'preact/hooks';
import htm from 'htm';
import { controllerRpc } from '/static/shared/rpc.js';
import { stateToName } from '/static/shared/utils.js';
import { JobsTab } from '/static/controller/jobs-tab.js';
import { WorkersTab } from '/static/controller/workers-tab.js';
import { EndpointsTab } from '/static/controller/endpoints-tab.js';
import { VmsTab } from '/static/controller/vms-tab.js';
import { AutoscalerTab } from '/static/controller/autoscaler-tab.js';
import { LogsTab } from '/static/controller/logs-tab.js';

const html = htm.bind(h);

const VALID_TABS = ['jobs', 'workers', 'endpoints', 'vms', 'autoscaler', 'logs'];

function getInitialTab() {
  const hash = window.location.hash.slice(1);
  return VALID_TABS.includes(hash) ? hash : 'jobs';
}

/**
 * Transform raw RPC responses (camelCase) into the flat objects each tab expects.
 * This mirrors the refresh() function from the original inline JS.
 */
function transformData(workersResp, jobsResp, endpointsResp, autoscalerResp) {
  const workers = (workersResp.workers || []).map(w => ({
    worker_id: w.workerId,
    address: w.address,
    healthy: w.healthy,
    last_heartbeat_ms: parseInt(w.lastHeartbeatMs || 0),
    running_tasks: (w.runningJobIds || []).length,
    resources: {
      cpu: w.metadata ? w.metadata.cpuCount : 0,
      memory_bytes: w.metadata ? parseInt(w.metadata.memoryBytes || 0) : 0
    },
    attributes: w.metadata && w.metadata.attributes ? w.metadata.attributes : {},
    status_message: w.statusMessage || ''
  }));

  const jobs = (jobsResp.jobs || []).map(j => ({
    job_id: j.jobId,
    name: j.name,
    state: stateToName(j.state),
    started_at_ms: parseInt(j.startedAtMs || 0),
    finished_at_ms: parseInt(j.finishedAtMs || 0),
    submitted_at_ms: parseInt(j.submittedAtMs || 0),
    failure_count: j.failureCount || 0,
    preemption_count: j.preemptionCount || 0,
    task_count: j.taskCount || 0,
    completed_count: j.completedCount || 0,
    task_state_counts: j.taskStateCounts || {},
    pending_reason: j.pendingReason || ''
  }));

  const endpoints = (endpointsResp.endpoints || []).map(e => ({
    name: e.name,
    address: e.address,
    job_id: e.jobId,
    metadata: e.metadata || {}
  }));

  const autoscaler = autoscalerResp.status || { enabled: false, groups: [], recentActions: [] };

  return { workers, jobs, endpoints, autoscaler };
}

function App() {
  const [activeTab, setActiveTab] = useState(getInitialTab);
  const [data, setData] = useState(null);
  const [jobsPage, setJobsPage] = useState(0);
  const [vmsPage, setVmsPage] = useState(0);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    try {
      const [workersResp, jobsResp, endpointsResp, autoscalerResp] = await Promise.all([
        controllerRpc('ListWorkers'),
        controllerRpc('ListJobs'),
        controllerRpc('ListEndpoints', { prefix: '' }),
        controllerRpc('GetAutoscalerStatus')
      ]);
      setData(transformData(workersResp, jobsResp, endpointsResp, autoscalerResp));
      setError(null);
    } catch (e) {
      setError('Failed to load dashboard data: ' + e.message);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  useEffect(() => {
    function handleHash() {
      const hash = window.location.hash.slice(1);
      if (VALID_TABS.includes(hash)) setActiveTab(hash);
    }
    window.addEventListener('hashchange', handleHash);
    return () => window.removeEventListener('hashchange', handleHash);
  }, []);

  function switchTab(tab) {
    window.location.hash = tab;
    setActiveTab(tab);
  }

  const tabContent = data && {
    jobs: html`<${JobsTab} jobs=${data.jobs} page=${jobsPage} onPageChange=${setJobsPage} />`,
    workers: html`<${WorkersTab} workers=${data.workers} />`,
    endpoints: html`<${EndpointsTab} endpoints=${data.endpoints} />`,
    vms: html`<${VmsTab} groups=${(data.autoscaler.groups || [])} page=${vmsPage} onPageChange=${setVmsPage} />`,
    autoscaler: html`<${AutoscalerTab} autoscaler=${data.autoscaler} />`,
    logs: html`<${LogsTab} />`,
  };

  return html`
    <h1>Iris Controller Dashboard</h1>
    ${error && html`<div class="error-message">${error}</div>`}
    <div class="tab-nav">
      ${VALID_TABS.map(tab => html`
        <button class=${'tab-btn' + (activeTab === tab ? ' active' : '')}
                onClick=${() => switchTab(tab)}>${tab.charAt(0).toUpperCase() + tab.slice(1)}</button>
      `)}
      <button class="tab-btn" onClick=${refresh} style="margin-left:auto;font-size:14px">\u21bb Refresh</button>
    </div>
    <div class="tab-content active">
      ${data ? tabContent[activeTab] : html`<div>Loading...</div>`}
    </div>`;
}

render(html`<${App} />`, document.getElementById('root'));
