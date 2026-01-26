# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HTTP dashboard with Connect RPC and web UI.

The dashboard serves:
- Web UI at / (main dashboard)
- Web UI at /job/{job_id} (job detail page)
- Connect RPC at /iris.cluster.ControllerService/* (called directly by JS)
- Health check at /health

All data fetching happens via Connect RPC calls from the browser JavaScript.
"""

import logging

import uvicorn
from starlette.applications import Starlette
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Mount, Route

from iris.cluster.controller.service import ControllerServiceImpl
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import ControllerServiceWSGIApplication

logger = logging.getLogger(__name__)

DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>Iris Controller</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      max-width: 1400px;
      margin: 40px auto;
      padding: 0 20px;
      color: #333;
      background: #f5f5f5;
    }
    h1 {
      color: #2c3e50;
      border-bottom: 3px solid #3498db;
      padding-bottom: 10px;
    }
    h2 {
      color: #34495e;
      margin-top: 30px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      background: white;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      border-radius: 8px;
      overflow: hidden;
    }
    th {
      background-color: #3498db;
      color: white;
      padding: 12px;
      text-align: left;
      font-weight: 600;
    }
    td {
      padding: 10px 12px;
      border-bottom: 1px solid #ecf0f1;
    }
    tr:hover {
      background-color: #f8f9fa;
    }
    .status-pending { color: #f39c12; }
    .status-running { color: #3498db; }
    .status-succeeded { color: #27ae60; }
    .status-failed { color: #e74c3c; }
    .status-killed { color: #95a5a6; }
    .status-worker_failed { color: #9b59b6; }
    .healthy { color: #27ae60; }
    .unhealthy { color: #e74c3c; }
    .worker-link, .job-link { color: #2196F3; text-decoration: none; }
    .worker-link:hover, .job-link:hover { text-decoration: underline; }
    .status-building { color: #9b59b6; }
    .actions-log {
      background: white;
      padding: 15px;
      border-radius: 8px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      max-height: 300px;
      overflow-y: auto;
      font-family: monospace;
      font-size: 13px;
    }
    .action-entry {
      padding: 5px 0;
      border-bottom: 1px solid #ecf0f1;
    }
    .action-time {
      color: #7f8c8d;
      margin-right: 10px;
    }
    .future-feature {
      color: #95a5a6;
      font-style: italic;
      padding: 20px;
      background: white;
      border-radius: 8px;
      text-align: center;
    }
    /* Autoscaler status */
    .autoscaler-status {
      background: white;
      padding: 15px 20px;
      border-radius: 8px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      margin-bottom: 20px;
      display: flex;
      gap: 30px;
      align-items: center;
    }
    .autoscaler-status .status-item {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .autoscaler-status .status-indicator {
      width: 10px;
      height: 10px;
      border-radius: 50%;
    }
    .autoscaler-status .status-indicator.active { background: #27ae60; }
    .autoscaler-status .status-indicator.disabled { background: #95a5a6; }
    .autoscaler-status .status-indicator.backoff { background: #f39c12; }
    .scale-groups-table {
      width: 100%;
      border-collapse: collapse;
      background: white;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      border-radius: 8px;
      overflow: hidden;
      margin-bottom: 20px;
    }
    .group-status {
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }
    .group-status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
    }
    .group-status-dot.available { background: #27ae60; }
    .group-status-dot.backoff { background: #f39c12; }
    .action-type {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
    }
    .action-type.scale_up { background: #d4edda; color: #155724; }
    .action-type.scale_down { background: #cce5ff; color: #004085; }
    .action-type.quota_exceeded { background: #f8d7da; color: #721c24; }
    .action-type.backoff_triggered { background: #fff3cd; color: #856404; }
    .action-type.worker_failed { background: #f8d7da; color: #721c24; }
    /* VM tree display */
    .scale-group {
      background: white;
      border-radius: 8px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      margin-bottom: 20px;
      overflow: hidden;
    }
    .scale-group-header {
      background: #3498db;
      color: white;
      padding: 15px 20px;
    }
    .scale-group-header h3 {
      margin: 0 0 5px 0;
    }
    .scale-group-meta {
      font-size: 14px;
      opacity: 0.9;
    }
    .slice-row {
      padding: 10px 20px;
      border-bottom: 1px solid #ecf0f1;
      cursor: pointer;
    }
    .slice-row:hover {
      background: #f8f9fa;
    }
    .slice-toggle {
      margin-right: 10px;
      font-family: monospace;
    }
    .slice-state {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 12px;
      font-weight: 500;
      margin-left: 10px;
    }
    .slice-state.ready { background: #d4edda; color: #155724; }
    .slice-state.initializing { background: #fff3cd; color: #856404; }
    .slice-state.booting { background: #cce5ff; color: #004085; }
    .slice-state.failed { background: #f8d7da; color: #721c24; }
    .slice-state.stopping { background: #e2e3e5; color: #383d41; }
    .slice-state.preempted { background: #e2d3f0; color: #6c3483; }
    .vm-list {
      display: none;
      padding: 10px 20px 10px 50px;
      background: #f8f9fa;
    }
    .vm-list.expanded {
      display: block;
    }
    .vm-row {
      padding: 5px 0;
      font-family: monospace;
      font-size: 13px;
    }
    .vm-state-indicator {
      display: inline-block;
      width: 8px;
      height: 8px;
      border-radius: 50%;
      margin-right: 8px;
    }
    .vm-state-indicator.ready { background: #27ae60; }
    .vm-state-indicator.initializing { background: #f39c12; }
    .vm-state-indicator.booting { background: #3498db; }
    .vm-state-indicator.failed { background: #e74c3c; }
    .vm-state-indicator.unhealthy { background: #e74c3c; }
    .vm-state-indicator.stopping { background: #95a5a6; }
    .vm-state-indicator.terminated { background: #bdc3c7; }
    .vm-state-indicator.preempted { background: #9b59b6; }
    .no-vms-message {
      padding: 40px;
      text-align: center;
      color: #7f8c8d;
    }
    /* Compact job view */
    .job-list {
      background: white;
      border-radius: 8px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      overflow: hidden;
    }
    .job-row {
      padding: 12px 20px;
      border-bottom: 1px solid #ecf0f1;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 15px;
    }
    .job-row:hover {
      background: #f8f9fa;
    }
    .job-toggle {
      font-family: monospace;
      color: #7f8c8d;
      width: 15px;
    }
    .job-id {
      font-family: monospace;
      color: #3498db;
      width: 80px;
    }
    .job-name {
      flex: 1;
      font-weight: 500;
      word-break: break-word;
    }
    .task-badges {
      display: flex;
      gap: 2px;
      align-items: center;
    }
    .task-badge {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
    }
    .task-badge.succeeded { background: #27ae60; }
    .task-badge.running { background: #3498db; }
    .task-badge.building { background: #9b59b6; }
    .task-badge.pending { background: #ecf0f1; border: 1px solid #bdc3c7; }
    .task-badge.failed { background: #e74c3c; }
    .task-badge.killed { background: #95a5a6; }
    .task-badge.worker_failed { background: #9b59b6; }
    .task-summary {
      color: #7f8c8d;
      font-size: 13px;
      width: 100px;
    }
    .job-state {
      font-weight: 500;
      width: 100px;
    }
    .job-duration {
      color: #7f8c8d;
      font-size: 13px;
      width: 80px;
      text-align: right;
    }
    .task-list {
      display: none;
      background: #f8f9fa;
      padding: 10px 20px 10px 55px;
      border-bottom: 1px solid #ecf0f1;
    }
    .task-list.expanded {
      display: block;
    }
    .task-row {
      padding: 6px 0;
      font-size: 13px;
      display: flex;
      align-items: center;
      gap: 15px;
    }
    .task-row .task-id {
      font-family: monospace;
      width: 120px;
    }
    .task-row .task-worker {
      width: 100px;
      color: #7f8c8d;
    }
    .task-row .task-state {
      width: 100px;
    }
    .task-row .task-attempts {
      color: #7f8c8d;
      font-size: 12px;
    }
    .task-icon {
      margin-right: 5px;
    }
    .task-icon.succeeded { color: #27ae60; }
    .task-icon.running { color: #3498db; }
    .task-icon.building { color: #9b59b6; }
    .task-icon.pending { color: #7f8c8d; }
    .task-icon.failed { color: #e74c3c; }
    .task-icon.killed { color: #95a5a6; }
    .task-icon.worker_failed { color: #9b59b6; }
    .no-jobs {
      padding: 40px;
      text-align: center;
      color: #7f8c8d;
    }
    /* Tab navigation */
    .tab-nav {
      display: flex;
      background: white;
      border-radius: 8px 8px 0 0;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      margin-bottom: 0;
    }
    .tab-btn {
      padding: 15px 30px;
      border: none;
      background: transparent;
      cursor: pointer;
      font-size: 16px;
      font-weight: 500;
      color: #7f8c8d;
      border-bottom: 3px solid transparent;
      transition: all 0.2s;
    }
    .tab-btn:hover {
      color: #3498db;
      background: #f8f9fa;
    }
    .tab-btn.active {
      color: #3498db;
      border-bottom-color: #3498db;
      background: #f8f9fa;
    }
    .tab-content {
      display: none;
      background: white;
      padding: 20px;
      border-radius: 0 0 8px 8px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .tab-content.active {
      display: block;
    }
  </style>
</head>
<body>
  <h1>Iris Controller Dashboard</h1>

  <div class="tab-nav">
    <button class="tab-btn active" data-tab="jobs">Jobs</button>
    <button class="tab-btn" data-tab="workers">Workers</button>
    <button class="tab-btn" data-tab="endpoints">Endpoints</button>
    <button class="tab-btn" data-tab="vms">VMs</button>
    <button class="tab-btn" data-tab="autoscaler">Autoscaler</button>
  </div>

  <div id="tab-jobs" class="tab-content active">
    <div id="jobs-list" class="job-list">
      <div class="no-jobs">Loading...</div>
    </div>
  </div>

  <div id="tab-workers" class="tab-content">
    <table id="workers-table">
      <tr><th>ID</th><th>Healthy</th><th>CPU</th><th>Memory</th><th>Running Tasks</th><th>Last Heartbeat</th></tr>
    </table>
  </div>

  <div id="tab-endpoints" class="tab-content">
    <table id="endpoints-table">
      <tr><th>Name</th><th>Address</th><th>Job</th><th>Metadata</th></tr>
    </table>
    <div id="no-endpoints" class="no-jobs" style="display:none">No endpoints registered</div>
  </div>

  <div id="tab-vms" class="tab-content">
    <div id="vms-container">
      <div class="no-vms-message">No scale groups configured</div>
    </div>
  </div>

  <div id="tab-autoscaler" class="tab-content">
    <div id="autoscaler-status-bar" class="autoscaler-status">
      <div class="status-item">
        <span class="status-indicator disabled"></span>
        <span>Status: <strong id="autoscaler-enabled">Disabled</strong></span>
      </div>
      <div class="status-item">
        <span>Last Evaluation: <strong id="last-evaluation">-</strong></span>
      </div>
    </div>

    <h3>Scale Groups</h3>
    <table class="scale-groups-table" id="scale-groups-table">
      <tr>
        <th>Group</th>
        <th>Booting</th>
        <th>Init</th>
        <th>Ready</th>
        <th>Failed</th>
        <th>Demand</th>
        <th>Status</th>
      </tr>
    </table>

    <h3>Recent Actions</h3>
    <div class="actions-log" id="autoscaler-actions"></div>
  </div>

  <script>
    function escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text || '';
      return div.innerHTML;
    }

    function formatBytes(bytes) {
      if (bytes === 0) return '0 B';
      const k = 1024;
      const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    function formatRelativeTime(timestampMs) {
      if (!timestampMs) return '-';
      const seconds = Math.floor((Date.now() - parseInt(timestampMs)) / 1000);
      if (seconds < 60) return `${seconds}s ago`;
      if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
      if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
      return `${Math.floor(seconds / 86400)}d ago`;
    }

    function formatDuration(startMs, endMs) {
      if (!startMs) return '-';
      const end = endMs || Date.now();
      const seconds = Math.floor((end - parseInt(startMs)) / 1000);
      if (seconds < 60) return `${seconds}s`;
      if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
      return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
    }

    // RPC helper: call Connect RPC endpoint with JSON
    async function rpc(method, body = {}) {
      const response = await fetch(`/iris.cluster.ControllerService/${method}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
      });
      if (!response.ok) {
        throw new Error(`RPC ${method} failed: ${response.status}`);
      }
      return response.json();
    }

    // Convert proto enum to lowercase name (JOB_STATE_PENDING -> pending)
    function stateToName(protoState) {
      if (!protoState) return 'pending';
      return protoState.replace(/^(JOB_STATE_|TASK_STATE_)/, '').toLowerCase();
    }

    // Cache for task data to avoid refetching on expand
    const jobTasksCache = {};

    async function fetchJobTasks(jobId) {
      if (!jobTasksCache[jobId]) {
        // RPC uses camelCase field names (Connect RPC standard)
        const resp = await rpc('ListTasks', {jobId: jobId});
        // Transform tasks for dashboard consumption
        jobTasksCache[jobId] = (resp.tasks || []).map(t => ({
          task_id: t.taskId,
          task_index: t.taskIndex,
          state: stateToName(t.state),
          worker_id: t.workerId,
          worker_address: t.workerAddress,
          started_at_ms: parseInt(t.startedAtMs || 0),
          finished_at_ms: parseInt(t.finishedAtMs || 0),
          exit_code: t.exitCode,
          error: t.error,
          num_attempts: (t.attempts || []).length || 1,
          pending_reason: t.pendingReason,
          can_be_scheduled: t.canBeScheduled
        }));
      }
      return jobTasksCache[jobId];
    }

    function renderJobsTab(jobs) {
      const container = document.getElementById('jobs-list');
      if (!jobs || jobs.length === 0) {
        container.innerHTML = '<div class="no-jobs">No jobs</div>';
        return;
      }

      // Sort: running/building first, then pending, then completed
      const stateOrder = {running: 0, building: 1, pending: 2, succeeded: 3, failed: 4, killed: 5, worker_failed: 6};
      jobs.sort((a, b) => (stateOrder[a.state] || 99) - (stateOrder[b.state] || 99));

      container.innerHTML = jobs.map(job => {
        const jobId = job.job_id;
        const shortId = jobId.slice(0, 8);
        const taskCount = job.task_count || 0;
        const completedCount = job.completed_count || 0;

        // Build task badges based on state counts
        const badges = buildTaskBadges(job);

        // Format duration
        let duration = '-';
        if (job.started_at_ms) {
          const endMs = job.finished_at_ms || Date.now();
          duration = formatDuration(job.started_at_ms, endMs);
        } else if (job.submitted_at_ms) {
          duration = 'queued ' + formatRelativeTime(job.submitted_at_ms);
        }

        return `<div class="job-row" onclick="toggleJobTasks(this, '${jobId}')">
          <span class="job-toggle">\u25b6</span>
          <a href="/job/${jobId}" class="job-id" onclick="event.stopPropagation()">${shortId}...</a>
          <span class="job-name">${escapeHtml(job.name || 'unnamed')}</span>
          <span class="task-badges">${badges}</span>
          <span class="task-summary">${completedCount}/${taskCount} tasks</span>
          <span class="job-state status-${job.state}">${job.state}</span>
          <span class="job-duration">${duration}</span>
        </div>
        <div class="task-list" id="tasks-${jobId}">Loading tasks...</div>`;
      }).join('');
    }

    function buildTaskBadges(job) {
      // Build badges from task state counts if available
      const counts = job.task_state_counts || {};
      const total = job.task_count || 0;

      if (total === 0) {
        return '<span style="color:#7f8c8d;font-size:12px">no tasks</span>';
      }

      // Use task state counts if available, otherwise estimate from job state
      const succeeded = counts.succeeded || 0;
      const running = counts.running || 0;
      const building = counts.building || 0;
      const failed = counts.failed || 0;
      const killed = counts.killed || 0;
      const workerFailed = counts.worker_failed || 0;
      const pending = counts.pending || (total - succeeded - running - building - failed - killed - workerFailed);

      let badges = '';
      for (let i = 0; i < succeeded; i++) badges += '<span class="task-badge succeeded"></span>';
      for (let i = 0; i < running; i++) badges += '<span class="task-badge running"></span>';
      for (let i = 0; i < building; i++) badges += '<span class="task-badge building"></span>';
      for (let i = 0; i < failed; i++) badges += '<span class="task-badge failed"></span>';
      for (let i = 0; i < workerFailed; i++) badges += '<span class="task-badge worker_failed"></span>';
      for (let i = 0; i < killed; i++) badges += '<span class="task-badge killed"></span>';
      for (let i = 0; i < pending; i++) badges += '<span class="task-badge pending"></span>';
      return badges;
    }

    async function toggleJobTasks(row, jobId) {
      const taskList = document.getElementById('tasks-' + jobId);
      const toggle = row.querySelector('.job-toggle');

      if (taskList.classList.contains('expanded')) {
        taskList.classList.remove('expanded');
        toggle.textContent = '\u25b6';
        return;
      }

      toggle.textContent = '\u25bc';
      taskList.classList.add('expanded');

      // Fetch and render tasks
      try {
        const tasks = await fetchJobTasks(jobId);
        taskList.innerHTML = tasks.map(t => {
          const stateIcon = getTaskStateIcon(t.state);
          const taskIdShort = t.task_id.length > 12 ? t.task_id.slice(0, 12) + '...' : t.task_id;
          return `<div class="task-row">
            <span class="task-id">${escapeHtml(taskIdShort)}</span>
            <span class="task-worker">${escapeHtml(t.worker_id || '-')}</span>
            <span class="task-state"><span class="task-icon ${t.state}">${stateIcon}</span>${t.state}</span>
            <span class="task-attempts">(${t.num_attempts} attempt${t.num_attempts !== 1 ? 's' : ''})</span>
          </div>`;
        }).join('') || '<div class="task-row">No tasks</div>';
      } catch (e) {
        taskList.innerHTML = '<div class="task-row" style="color:#e74c3c">Failed to load tasks</div>';
      }
    }

    function getTaskStateIcon(state) {
      const icons = {
        succeeded: '\u2713',
        running: '\u25d0',
        building: '\u25d0',
        pending: '\u25cb',
        failed: '\u2715',
        killed: '\u25cb',
        worker_failed: '\u2715',
        unschedulable: '!'
      };
      return icons[state] || '?';
    }

    function formatVmState(state) {
      if (!state) return 'unknown';
      return state.replace('VM_STATE_', '').toLowerCase();
    }

    function computeSliceStateCounts(slices) {
      // Compute slice state counts from SliceInfo[] (mirrors Python helper)
      const counts = {booting: 0, initializing: 0, ready: 0, failed: 0};
      for (const s of slices) {
        const vms = s.vms || [];
        if (vms.length === 0) continue;
        // Skip terminated VM groups
        if (vms.every(vm => vm.state === "VM_STATE_TERMINATED")) continue;
        const anyFailed = vms.some(vm => vm.state === "VM_STATE_FAILED" || vm.state === "VM_STATE_PREEMPTED");
        const allReady = vms.every(vm => vm.state === "VM_STATE_READY");
        if (anyFailed) {
          counts.failed++;
        } else if (allReady) {
          counts.ready++;
        } else if (vms.some(vm => vm.state === "VM_STATE_INITIALIZING")) {
          counts.initializing++;
        } else if (vms.some(vm => vm.state === "VM_STATE_BOOTING")) {
          counts.booting++;
        }
      }
      return counts;
    }

    function toggleSlice(row) {
      const vmList = row.nextElementSibling;
      const toggle = row.querySelector('.slice-toggle');
      if (vmList.classList.contains('expanded')) {
        vmList.classList.remove('expanded');
        toggle.textContent = '\u25b6';
      } else {
        vmList.classList.add('expanded');
        toggle.textContent = '\u25bc';
      }
    }

    function renderVmsTab(data) {
      const container = document.getElementById('vms-container');
      if (!data.groups || data.groups.length === 0) {
        container.innerHTML = '<div class="no-vms-message">No scale groups configured</div>';
        return;
      }

      container.innerHTML = data.groups.map(group => {
        const config = group.config || {};
        const slices = group.slices || [];
        const counts = computeSliceStateCounts(slices);

        const slicesHtml = slices.map(slice => {
          // Determine slice state from constituent VMs (RPC uses camelCase)
          const vms = slice.vms || [];
          const vmStates = vms.map(vm => formatVmState(vm.state));
          const allReady = vms.length > 0 && vms.every(vm => vm.state === "VM_STATE_READY");
          const anyFailed = vms.some(vm => vm.state === "VM_STATE_FAILED" || vm.state === "VM_STATE_PREEMPTED");
          const sliceState = anyFailed ? 'failed' :
                             allReady ? 'ready' :
                             vmStates.some(s => s === 'booting') ? 'booting' :
                             vmStates.some(s => s === 'stopping') ? 'stopping' :
                             vmStates.some(s => s === 'preempted') ? 'preempted' :
                             'initializing';

          const vmsHtml = vms.map(vm => {
            const vmState = formatVmState(vm.state);
            return `<div class="vm-row">
              <span class="vm-state-indicator ${vmState}"></span>
              ${escapeHtml(vm.vmId)} &nbsp; ${vmState.toUpperCase()} &nbsp;
              ${escapeHtml(vm.address || '-')} &nbsp;
              ${vm.workerId ? escapeHtml(vm.workerId) : ''}
              ${vm.initError ? '<span style="color:#e74c3c">'+escapeHtml(vm.initError)+'</span>' : ''}
            </div>`;
          }).join('');

          return `<div class="slice-row" onclick="toggleSlice(this)">
            <span class="slice-toggle">\u25b6</span>
            <strong>${escapeHtml(slice.sliceId)}</strong>
            <span class="slice-state ${sliceState}">${sliceState.toUpperCase()}</span>
            <span style="color:#7f8c8d">(${vms.length} VMs)</span>
          </div>
          <div class="vm-list">${vmsHtml || '<div>No VMs</div>'}</div>`;
        }).join('');

        const statesSummary = Object.entries(counts)
          .filter(([k, v]) => v > 0)
          .map(([k, v]) => `${k}: ${v}`)
          .join(', ') || 'no slices';

        return `<div class="scale-group">
          <div class="scale-group-header">
            <h3>${escapeHtml(group.name)}</h3>
            <div class="scale-group-meta">
              Accelerator: ${escapeHtml(config.accelerator_type || '-')} |
              Min: ${config.min_slices || 0} Max: ${config.max_slices || 0} |
              Demand: ${group.current_demand || 0} |
              ${statesSummary}
            </div>
          </div>
          ${slicesHtml || '<div class="no-vms-message">No slices in this group</div>'}
        </div>`;
      }).join('');
    }

    function renderAutoscalerTab(data) {
      // Update status bar
      const statusIndicator = document.querySelector('#autoscaler-status-bar .status-indicator');
      const enabledEl = document.getElementById('autoscaler-enabled');
      const lastEvalEl = document.getElementById('last-evaluation');

      if (data.enabled === false) {
        statusIndicator.className = 'status-indicator disabled';
        enabledEl.textContent = 'Disabled';
        lastEvalEl.textContent = '-';
        document.getElementById('scale-groups-table').innerHTML =
          '<tr><th>Group</th><th>Booting</th><th>Init</th><th>Ready</th><th>Failed</th><th>Demand</th><th>Status</th></tr>';
        document.getElementById('autoscaler-actions').innerHTML =
          '<div class="action-entry">Autoscaler not configured</div>';
        return;
      }

      statusIndicator.className = 'status-indicator active';
      enabledEl.textContent = 'Active';
      // RPC uses camelCase: lastEvaluationMs
      lastEvalEl.textContent = data.lastEvaluationMs ?
        formatRelativeTime(parseInt(data.lastEvaluationMs)) : '-';

      // Render scale groups table (RPC uses camelCase)
      const groups = data.groups || [];
      const groupsHtml = groups.map(g => {
        const counts = computeSliceStateCounts(g.slices || []);
        const isBackoff = g.backoffUntilMs && parseInt(g.backoffUntilMs) > Date.now();
        const statusClass = isBackoff ? 'backoff' : 'available';
        const statusText = isBackoff ? 'Backoff' : 'Available';

        return `<tr>
          <td><strong>${escapeHtml(g.name)}</strong></td>
          <td>${counts.booting || 0}</td>
          <td>${counts.initializing || 0}</td>
          <td>${counts.ready || 0}</td>
          <td>${counts.failed || 0}</td>
          <td>${g.currentDemand || 0}</td>
          <td><span class="group-status"><span class="group-status-dot ${statusClass}"></span> ${statusText}</span></td>
        </tr>`;
      }).join('');

      const headerRow = '<tr><th>Group</th><th>Booting</th><th>Init</th>' +
        '<th>Ready</th><th>Failed</th><th>Demand</th><th>Status</th></tr>';
      document.getElementById('scale-groups-table').innerHTML = headerRow + groupsHtml;

      // Render actions log (newest first) - RPC uses camelCase
      const actions = (data.recentActions || []).slice().reverse();
      const actionsHtml = actions.map(a => {
        const time = new Date(parseInt(a.timestampMs)).toLocaleTimeString();
        const actionType = a.actionType || 'unknown';
        const sliceInfo = a.sliceId ? ` [${escapeHtml(a.sliceId.slice(0,20))}...]` : '';
        const status = a.status || 'completed';
        const statusClass = status === 'pending' ? 'status-pending' :
                            status === 'failed' ? 'status-failed' : 'status-succeeded';
        const statusBadge = status !== 'completed'
          ? `<span class="${statusClass}" style="margin-left:5px">[${status}]</span>` : '';
        return `<div class="action-entry">
          <span class="action-time">${time}</span>
          <span class="action-type ${actionType}">${actionType.replace('_', ' ')}</span>${statusBadge}
          <strong>${escapeHtml(a.scaleGroup)}</strong>${sliceInfo}
          ${a.reason ? ' - ' + escapeHtml(a.reason) : ''}
        </div>`;
      }).join('');

      document.getElementById('autoscaler-actions').innerHTML =
        actionsHtml || '<div class="action-entry">No recent actions</div>';
    }

    async function refresh() {
      try {
        // Fetch data via RPCs
        const [workersResp, jobsResp, endpointsResp, autoscalerResp] = await Promise.all([
          rpc('ListWorkers'),
          rpc('ListJobs'),
          rpc('ListEndpoints', {prefix: ''}),
          rpc('GetAutoscalerStatus')
        ]);

        // Transform workers for dashboard consumption (RPC uses camelCase)
        const workers = (workersResp.workers || []).map(w => ({
          worker_id: w.workerId,
          address: w.address,
          healthy: w.healthy,
          last_heartbeat_ms: parseInt(w.lastHeartbeatMs || 0),
          running_tasks: (w.runningJobIds || []).length,
          resources: {
            cpu: w.metadata ? w.metadata.cpuCount : 0,
            memory_bytes: w.metadata ? parseInt(w.metadata.memoryBytes || 0) : 0
          }
        }));

        // Transform jobs for dashboard consumption (RPC uses camelCase)
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
          resources: j.resources || {}
        }));

        // Transform endpoints (camelCase: jobId)
        const endpoints = (endpointsResp.endpoints || []).map(e => ({
          name: e.name,
          address: e.address,
          job_id: e.jobId,
          metadata: e.metadata || {}
        }));

        // Extract autoscaler status (camelCase: recentActions)
        const autoscaler = autoscalerResp.status || {enabled: false, groups: [], recentActions: []};

        // Update workers table
        const workersHtml = workers.map(w => {
          const lastHb = formatRelativeTime(w.last_heartbeat_ms);
          const healthClass = w.healthy ? 'healthy' : 'unhealthy';
          const healthIndicator = w.healthy ? '\u25cf' : '\u25cb';
          const healthText = w.healthy ? 'Yes' : 'No';
          const wid = escapeHtml(w.worker_id);
          const workerLink = w.address
            ? `<a href="http://${escapeHtml(w.address)}/" class="worker-link" target="_blank">${wid}</a>`
            : wid;
          const cpu = w.resources ? w.resources.cpu : '-';
          const memBytes = w.resources ? (w.resources.memory_bytes || 0) : 0;
          const memory = memBytes ? formatBytes(memBytes) : '-';
          return `<tr>
            <td>${workerLink}</td>
            <td class="${healthClass}">${healthIndicator} ${healthText}</td>
            <td>${cpu}</td>
            <td>${memory}</td>
            <td>${w.running_tasks}</td>
            <td>${lastHb}</td>
          </tr>`;
        }).join('');
        const workersHeader = '<tr><th>ID</th><th>Healthy</th><th>CPU</th><th>Memory</th>' +
          '<th>Running Tasks</th><th>Last Heartbeat</th></tr>';
        document.getElementById('workers-table').innerHTML = workersHeader + workersHtml;

        // Update jobs tab with compact view
        renderJobsTab(jobs);

        // Update endpoints table
        if (endpoints.length === 0) {
          document.getElementById('endpoints-table').style.display = 'none';
          document.getElementById('no-endpoints').style.display = 'block';
        } else {
          document.getElementById('endpoints-table').style.display = '';
          document.getElementById('no-endpoints').style.display = 'none';
          const endpointsHtml = endpoints.map(e => {
            const eid = escapeHtml(e.job_id);
            const jobLink = `<a href="/job/${eid}" class="job-link">${eid.slice(0,8)}...</a>`;
            const metaStr = e.metadata ? Object.entries(e.metadata).map(([k,v]) => `${k}=${v}`).join(', ') : '-';
            return `<tr>
              <td>${escapeHtml(e.name)}</td>
              <td>${escapeHtml(e.address)}</td>
              <td>${jobLink}</td>
              <td>${escapeHtml(metaStr)}</td>
            </tr>`;
          }).join('');
          document.getElementById('endpoints-table').innerHTML =
            '<tr><th>Name</th><th>Address</th><th>Job</th><th>Metadata</th></tr>' + endpointsHtml;
        }

        // Update VMs tab (use autoscaler groups)
        renderVmsTab({groups: autoscaler.groups || []});

        // Update Autoscaler tab
        renderAutoscalerTab(autoscaler);
      } catch (e) {
        console.error('Failed to refresh:', e);
      }
    }

    // Tab switching with URL hash routing
    function switchTab(tabName) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

      const btn = document.querySelector(`.tab-btn[data-tab="${tabName}"]`);
      if (btn) {
        btn.classList.add('active');
        document.getElementById('tab-' + tabName).classList.add('active');
      }
    }

    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const tabName = btn.dataset.tab;
        window.location.hash = tabName;
        switchTab(tabName);
      });
    });

    // Handle initial hash and hash changes
    function handleHash() {
      const hash = window.location.hash.slice(1);
      const validTabs = ['jobs', 'workers', 'endpoints', 'vms', 'autoscaler'];
      if (validTabs.includes(hash)) {
        switchTab(hash);
      }
    }

    window.addEventListener('hashchange', handleHash);
    handleHash();

    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>
"""


JOB_DETAIL_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>Job Detail - {{job_id}}</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      max-width: 1400px;
      margin: 40px auto;
      padding: 0 20px;
      color: #333;
      background: #f5f5f5;
    }
    h1 {
      color: #2c3e50;
      border-bottom: 3px solid #3498db;
      padding-bottom: 10px;
    }
    h2 {
      color: #34495e;
      margin-top: 30px;
    }
    .back-link {
      color: #2196F3;
      text-decoration: none;
      margin-bottom: 20px;
      display: inline-block;
    }
    .back-link:hover { text-decoration: underline; }
    .info-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 20px;
      margin-bottom: 20px;
    }
    .info-card {
      background: white;
      padding: 20px;
      border-radius: 8px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-card h3 {
      margin-top: 0;
      color: #34495e;
      border-bottom: 1px solid #ecf0f1;
      padding-bottom: 10px;
    }
    .info-row {
      display: flex;
      justify-content: space-between;
      padding: 8px 0;
      border-bottom: 1px solid #ecf0f1;
    }
    .info-row:last-child { border-bottom: none; }
    .info-label { color: #7f8c8d; }
    .info-value { font-weight: 500; }
    .status-pending { color: #f39c12; }
    .status-building { color: #9b59b6; }
    .status-running { color: #3498db; }
    .status-succeeded { color: #27ae60; }
    .status-failed { color: #e74c3c; }
    .status-killed { color: #95a5a6; }
    .status-worker_failed { color: #9b59b6; }
    .status-unschedulable { color: #e74c3c; }
    .error-message {
      background: #fee;
      border: 1px solid #e74c3c;
      color: #c0392b;
      padding: 15px;
      border-radius: 8px;
      margin-bottom: 20px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      background: white;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      border-radius: 8px;
      overflow: hidden;
      margin-top: 20px;
    }
    th {
      background-color: #3498db;
      color: white;
      padding: 12px;
      text-align: left;
      font-weight: 600;
    }
    td {
      padding: 10px 12px;
      border-bottom: 1px solid #ecf0f1;
    }
    tr:hover {
      background-color: #f8f9fa;
    }
    .pending-reason {
      font-size: 12px;
      color: #7f8c8d;
      font-style: italic;
    }
  </style>
</head>
<body>
  <a href="/" class="back-link">&larr; Back to Dashboard</a>
  <h1>Job: {{job_id}}</h1>

  <div id="error-container"></div>

  <div class="info-grid">
    <div class="info-card">
      <h3>Job Status</h3>
      <div class="info-row">
        <span class="info-label">State</span>
        <span class="info-value" id="job-state">-</span>
      </div>
      <div class="info-row">
        <span class="info-label">Exit Code</span>
        <span class="info-value" id="job-exit-code">-</span>
      </div>
      <div class="info-row">
        <span class="info-label">Started</span>
        <span class="info-value" id="job-started">-</span>
      </div>
      <div class="info-row">
        <span class="info-label">Finished</span>
        <span class="info-value" id="job-finished">-</span>
      </div>
      <div class="info-row">
        <span class="info-label">Duration</span>
        <span class="info-value" id="job-duration">-</span>
      </div>
    </div>

    <div class="info-card">
      <h3>Task Summary</h3>
      <div class="info-row">
        <span class="info-label">Total Tasks</span>
        <span class="info-value" id="total-tasks">-</span>
      </div>
      <div class="info-row">
        <span class="info-label">Completed</span>
        <span class="info-value" id="completed-tasks">-</span>
      </div>
      <div class="info-row">
        <span class="info-label">Running</span>
        <span class="info-value" id="running-tasks">-</span>
      </div>
      <div class="info-row">
        <span class="info-label">Pending</span>
        <span class="info-value" id="pending-tasks">-</span>
      </div>
      <div class="info-row">
        <span class="info-label">Failed</span>
        <span class="info-value" id="failed-tasks">-</span>
      </div>
    </div>

    <div class="info-card">
      <h3>Resource Request</h3>
      <div class="info-row">
        <span class="info-label">CPU</span>
        <span class="info-value" id="resource-cpu">-</span>
      </div>
      <div class="info-row">
        <span class="info-label">Memory</span>
        <span class="info-value" id="resource-memory">-</span>
      </div>
      <div class="info-row">
        <span class="info-label">Replicas</span>
        <span class="info-value" id="resource-replicas">-</span>
      </div>
    </div>
  </div>

  <h2>Tasks</h2>
  <table id="tasks-table">
    <tr>
      <th>Task ID</th>
      <th>Index</th>
      <th>State</th>
      <th>Worker</th>
      <th>Attempts</th>
      <th>Started</th>
      <th>Duration</th>
      <th>Exit Code</th>
      <th>Error</th>
    </tr>
  </table>

  <script>
    const jobId = '{{job_id}}';
    const controllerAddress = window.location.origin;

    function escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text || '';
      return div.innerHTML;
    }

    function formatTimestamp(ms) {
      if (!ms) return '-';
      return new Date(ms).toLocaleString();
    }

    function formatBytes(bytes) {
      if (bytes === 0) return '0 B';
      const k = 1024;
      const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    function formatDuration(startMs, endMs) {
      if (!startMs) return '-';
      const end = endMs || Date.now();
      const seconds = Math.floor((end - startMs) / 1000);
      if (seconds < 60) return `${seconds}s`;
      if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
      return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
    }

    function getStateClass(state) {
      const stateMap = {
        'pending': 'status-pending',
        'building': 'status-building',
        'running': 'status-running',
        'succeeded': 'status-succeeded',
        'failed': 'status-failed',
        'killed': 'status-killed',
        'worker_failed': 'status-worker_failed',
        'unschedulable': 'status-unschedulable'
      };
      return stateMap[state] || '';
    }

    // RPC helper: call Connect RPC endpoint with JSON
    async function rpc(method, body = {}) {
      const response = await fetch(`/iris.cluster.ControllerService/${method}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
      });
      if (!response.ok) {
        throw new Error(`RPC ${method} failed: ${response.status}`);
      }
      return response.json();
    }

    // Convert proto enum to lowercase name (JOB_STATE_PENDING -> pending)
    function stateToName(protoState) {
      if (!protoState) return 'pending';
      return protoState.replace(/^(JOB_STATE_|TASK_STATE_)/, '').toLowerCase();
    }

    async function refresh() {
      try {
        // Fetch job status from controller via RPC (camelCase)
        const [jobsResp, tasksResp] = await Promise.all([
          rpc('ListJobs'),
          rpc('ListTasks', {jobId: jobId})
        ]);

        // Transform job data (RPC uses camelCase)
        const jobs = (jobsResp.jobs || []).map(j => ({
          job_id: j.jobId,
          state: stateToName(j.state),
          failure_count: j.failureCount || 0,
          error: j.error,
          started_at_ms: parseInt(j.startedAtMs || 0),
          finished_at_ms: parseInt(j.finishedAtMs || 0),
          resources: {
            cpu: j.resources ? j.resources.cpu : 0,
            memory_bytes: j.resources ? parseInt(j.resources.memoryBytes || 0) : 0
          }
        }));

        // Transform tasks (RPC uses camelCase)
        const tasksResponse = (tasksResp.tasks || []).map(t => ({
          task_id: t.taskId,
          task_index: t.taskIndex,
          state: stateToName(t.state),
          worker_id: t.workerId || '',
          started_at_ms: parseInt(t.startedAtMs || 0),
          finished_at_ms: parseInt(t.finishedAtMs || 0),
          exit_code: t.exitCode,
          error: t.error || '',
          num_attempts: (t.attempts || []).length || 1,
          pending_reason: t.pendingReason || ''
        }));

        const job = jobs.find(j => j.job_id === jobId);
        if (!job) {
          document.getElementById('error-container').innerHTML =
            '<div class="error-message">Job not found</div>';
          return;
        }

        // Update job status
        const stateEl = document.getElementById('job-state');
        stateEl.textContent = job.state || '-';
        stateEl.className = 'info-value ' + getStateClass(job.state);

        document.getElementById('job-exit-code').textContent =
          job.failure_count > 0 ? 'Failed' : (job.state === 'succeeded' ? '0' : '-');
        document.getElementById('job-started').textContent =
          formatTimestamp(job.started_at_ms);
        document.getElementById('job-finished').textContent =
          formatTimestamp(job.finished_at_ms);
        document.getElementById('job-duration').textContent =
          formatDuration(job.started_at_ms, job.finished_at_ms);

        // Show error if present
        if (job.error) {
          document.getElementById('error-container').innerHTML =
            `<div class="error-message"><strong>Error:</strong> ${escapeHtml(job.error)}</div>`;
        } else {
          document.getElementById('error-container').innerHTML = '';
        }

        // Update resource info
        document.getElementById('resource-cpu').textContent = job.resources.cpu || '-';
        document.getElementById('resource-memory').textContent =
          job.resources.memory_bytes ? formatBytes(job.resources.memory_bytes) : '-';
        document.getElementById('resource-replicas').textContent = tasksResponse.length || '-';

        // Count task states
        const stateCounts = {
          total: tasksResponse.length,
          completed: 0,
          running: 0,
          pending: 0,
          failed: 0
        };

        tasksResponse.forEach(t => {
          if (t.state === 'succeeded' || t.state === 'killed') stateCounts.completed++;
          else if (t.state === 'running' || t.state === 'building') stateCounts.running++;
          else if (t.state === 'pending') stateCounts.pending++;
          else if (t.state === 'failed' || t.state === 'worker_failed') stateCounts.failed++;
        });

        document.getElementById('total-tasks').textContent = stateCounts.total;
        document.getElementById('completed-tasks').textContent = stateCounts.completed;
        document.getElementById('running-tasks').textContent = stateCounts.running;
        document.getElementById('pending-tasks').textContent = stateCounts.pending;
        document.getElementById('failed-tasks').textContent = stateCounts.failed;

        // Update tasks table
        const tasksHtml = tasksResponse.map(t => {
          const errorText = t.error || '';
          const pendingInfo = t.pending_reason
            ? `<br><span class="pending-reason">${escapeHtml(t.pending_reason)}</span>`
            : '';

          return `<tr>
            <td>${escapeHtml(t.task_id)}</td>
            <td>${t.task_index}</td>
            <td class="${getStateClass(t.state)}">${escapeHtml(t.state)}${pendingInfo}</td>
            <td>${escapeHtml(t.worker_id || '-')}</td>
            <td>${t.num_attempts}</td>
            <td>${formatTimestamp(t.started_at_ms)}</td>
            <td>${formatDuration(t.started_at_ms, t.finished_at_ms)}</td>
            <td>${t.exit_code !== null && t.exit_code !== undefined ? t.exit_code : '-'}</td>
            <td>${escapeHtml(errorText) || '-'}</td>
          </tr>`;
        }).join('');

        const tableHeader = `<tr>
          <th>Task ID</th>
          <th>Index</th>
          <th>State</th>
          <th>Worker</th>
          <th>Attempts</th>
          <th>Started</th>
          <th>Duration</th>
          <th>Exit Code</th>
          <th>Error</th>
        </tr>`;

        document.getElementById('tasks-table').innerHTML = tableHeader + tasksHtml;

      } catch (e) {
        console.error('Failed to refresh:', e);
        document.getElementById('error-container').innerHTML =
          `<div class="error-message">Failed to load job details: ${escapeHtml(e.message)}</div>`;
      }
    }

    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>
"""


class ControllerDashboard:
    """HTTP dashboard with Connect RPC and web UI.

    The dashboard serves a single-page web UI that fetches all data directly
    via Connect RPC calls to the ControllerService. This eliminates the need
    for a separate REST API layer and ensures the dashboard shows exactly
    what the RPC returns.
    """

    def __init__(
        self,
        service: ControllerServiceImpl,
        host: str = "0.0.0.0",
        port: int = 8080,
    ):
        self._service = service
        self._host = host
        self._port = port
        self._app = self._create_app()
        self._server: uvicorn.Server | None = None

    @property
    def port(self) -> int:
        return self._port

    def _create_app(self) -> Starlette:
        rpc_wsgi_app = ControllerServiceWSGIApplication(service=self._service)
        rpc_app = WSGIMiddleware(rpc_wsgi_app)

        routes = [
            Route("/", self._dashboard),
            Route("/job/{job_id}", self._job_detail_page),
            Route("/health", self._health),
            Mount(rpc_wsgi_app.path, app=rpc_app),
        ]
        return Starlette(routes=routes)

    def _dashboard(self, _request: Request) -> HTMLResponse:
        return HTMLResponse(DASHBOARD_HTML)

    def _job_detail_page(self, request: Request) -> HTMLResponse:
        import html

        job_id = request.path_params["job_id"]
        return HTMLResponse(JOB_DETAIL_HTML.replace("{{job_id}}", html.escape(job_id)))

    def _health(self, _request: Request) -> JSONResponse:
        """Health check endpoint for controller availability."""
        workers_resp = self._service.list_workers(cluster_pb2.Controller.ListWorkersRequest(), None)
        jobs_resp = self._service.list_jobs(cluster_pb2.Controller.ListJobsRequest(), None)
        worker_count = len(workers_resp.workers)
        job_count = len(jobs_resp.jobs)

        response = {
            "status": "ok",
            "workers": worker_count,
            "jobs": job_count,
        }

        return JSONResponse(response)

    def run(self) -> None:
        uvicorn.run(self._app, host=self._host, port=self._port)

    async def run_async(self) -> None:
        config = uvicorn.Config(self._app, host=self._host, port=self._port)
        self._server = uvicorn.Server(config)
        await self._server.serve()

    async def shutdown(self) -> None:
        if self._server:
            self._server.should_exit = True
