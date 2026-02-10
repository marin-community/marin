import { h, render } from 'preact';
import { useState, useEffect, useCallback } from 'preact/hooks';
import htm from 'htm';
import { controllerRpc } from '/static/shared/rpc.js';
import { stateToName, timestampFromProto } from '/static/shared/utils.js';
import { JobsTab } from '/static/controller/jobs-tab.js';
import { WorkersTab } from '/static/controller/workers-tab.js';
import { EndpointsTab } from '/static/controller/endpoints-tab.js';
import { VmsTab } from '/static/controller/vms-tab.js';
import { AutoscalerTab } from '/static/controller/autoscaler-tab.js';
import { LogsTab } from '/static/controller/logs-tab.js';
import { TransactionsTab } from '/static/controller/transactions-tab.js';

const html = htm.bind(h);

const VALID_TABS = ['jobs', 'workers', 'endpoints', 'vms', 'autoscaler', 'logs', 'transactions'];
const JOBS_PAGE_SIZE = 50;

// Map frontend sort field names to proto enum values
const SORT_FIELD_MAP = {
  date: 'JOB_SORT_FIELD_DATE',
  name: 'JOB_SORT_FIELD_NAME',
  state: 'JOB_SORT_FIELD_STATE',
  failures: 'JOB_SORT_FIELD_FAILURES',
  preemptions: 'JOB_SORT_FIELD_PREEMPTIONS',
};

function getInitialTab() {
  const hash = window.location.hash.slice(1);
  return VALID_TABS.includes(hash) ? hash : 'jobs';
}

/**
 * Transform workers and other data (not jobs - jobs are handled separately now).
 */
function transformOtherData(workersResp, endpointsResp, autoscalerResp) {
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

  const endpoints = (endpointsResp.endpoints || []).map(e => ({
    name: e.name,
    address: e.address,
    job_id: e.jobId,
    metadata: e.metadata || {}
  }));

  const autoscaler = autoscalerResp.status || { enabled: false, groups: [], recentActions: [] };

  return { workers, endpoints, autoscaler };
}

/** Transform jobs response from server. */
function transformJobsResp(jobsResp) {
  const jobs = (jobsResp.jobs || []).map(j => ({
    job_id: j.jobId,
    name: j.name,
    state: stateToName(j.state),
    started_at_ms: timestampFromProto(j.startedAt),
    finished_at_ms: timestampFromProto(j.finishedAt),
    submitted_at_ms: timestampFromProto(j.submittedAt),
    failure_count: j.failureCount || 0,
    preemption_count: j.preemptionCount || 0,
    task_count: j.taskCount || 0,
    completed_count: j.completedCount || 0,
    task_state_counts: j.taskStateCounts || {},
    pending_reason: j.pendingReason || ''
  }));
  return {
    jobs,
    totalCount: jobsResp.totalCount || jobs.length,
    hasMore: jobsResp.hasMore || false,
  };
}

function App() {
  const [activeTab, setActiveTab] = useState(getInitialTab);
  const [otherData, setOtherData] = useState(null);
  const [jobsData, setJobsData] = useState(null);
  const [vmsPage, setVmsPage] = useState(0);
  const [error, setError] = useState(null);

  // Jobs pagination and sorting state
  const [jobsPage, setJobsPage] = useState(0);
  const [jobsSortField, setJobsSortField] = useState('date');
  const [jobsSortDir, setJobsSortDir] = useState('desc');
  const [jobsNameFilter, setJobsNameFilter] = useState('');

  // Fetch jobs with server-side pagination/sorting
  const fetchJobs = useCallback(async (page, sortField, sortDir, nameFilter) => {
    try {
      const jobsResp = await controllerRpc('ListJobs', {
        offset: page * JOBS_PAGE_SIZE,
        limit: JOBS_PAGE_SIZE,
        sortField: SORT_FIELD_MAP[sortField] || 'JOB_SORT_FIELD_DATE',
        sortDirection: sortDir === 'asc' ? 'SORT_DIRECTION_ASC' : 'SORT_DIRECTION_DESC',
        nameFilter: nameFilter || '',
      });
      setJobsData(transformJobsResp(jobsResp));
    } catch (e) {
      console.error('Failed to load jobs:', e);
    }
  }, []);

  // Fetch other data (workers, endpoints, autoscaler)
  const fetchOtherData = useCallback(async () => {
    try {
      const [workersResp, endpointsResp, autoscalerResp] = await Promise.all([
        controllerRpc('ListWorkers'),
        controllerRpc('ListEndpoints', { prefix: '' }),
        controllerRpc('GetAutoscalerStatus')
      ]);
      setOtherData(transformOtherData(workersResp, endpointsResp, autoscalerResp));
      setError(null);
    } catch (e) {
      setError('Failed to load dashboard data: ' + e.message);
    }
  }, []);

  // Initial load
  useEffect(() => {
    fetchOtherData();
    fetchJobs(jobsPage, jobsSortField, jobsSortDir, jobsNameFilter);
  }, [fetchOtherData, fetchJobs, jobsPage, jobsSortField, jobsSortDir, jobsNameFilter]);

  // Full refresh
  const refresh = useCallback(() => {
    fetchOtherData();
    fetchJobs(jobsPage, jobsSortField, jobsSortDir, jobsNameFilter);
  }, [fetchOtherData, fetchJobs, jobsPage, jobsSortField, jobsSortDir, jobsNameFilter]);

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

  // Handlers for jobs tab
  const handleJobsPageChange = (newPage) => {
    setJobsPage(newPage);
  };

  const handleJobsSortChange = (field, dir) => {
    setJobsSortField(field);
    setJobsSortDir(dir);
    setJobsPage(0); // Reset to first page on sort change
  };

  const handleJobsFilterChange = (nameFilter) => {
    setJobsNameFilter(nameFilter);
    setJobsPage(0); // Reset to first page on filter change
  };

  const tabContent = otherData && {
    jobs: html`<${JobsTab}
      jobs=${jobsData?.jobs || []}
      totalCount=${jobsData?.totalCount || 0}
      hasMore=${jobsData?.hasMore || false}
      page=${jobsPage}
      pageSize=${JOBS_PAGE_SIZE}
      sortField=${jobsSortField}
      sortDir=${jobsSortDir}
      nameFilter=${jobsNameFilter}
      onPageChange=${handleJobsPageChange}
      onSortChange=${handleJobsSortChange}
      onFilterChange=${handleJobsFilterChange}
    />`,
    workers: html`<${WorkersTab} workers=${otherData.workers} />`,
    endpoints: html`<${EndpointsTab} endpoints=${otherData.endpoints} />`,
    vms: html`<${VmsTab} groups=${(otherData.autoscaler.groups || [])} page=${vmsPage} onPageChange=${setVmsPage} />`,
    autoscaler: html`<${AutoscalerTab} autoscaler=${otherData.autoscaler} />`,
    logs: html`<${LogsTab} />`,
    transactions: html`<${TransactionsTab} />`,
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
      ${otherData ? tabContent[activeTab] : html`<div>Loading...</div>`}
    </div>`;
}

render(html`<${App} />`, document.getElementById('root'));
