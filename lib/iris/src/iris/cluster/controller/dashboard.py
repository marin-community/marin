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

import html
import logging

import uvicorn
from starlette.applications import Starlette
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Mount, Route

from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.dashboard_common import logs_api_response, logs_page_response
from iris.logging import LogBuffer
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
      color: #1f2328;
      background: #f6f8fa;
      font-size: 14px;
    }
    h1 {
      color: #1f2328;
      border-bottom: 2px solid #d1d9e0;
      padding-bottom: 10px;
      font-size: 24px;
      font-weight: 600;
    }
    h2 {
      color: #1f2328;
      margin-top: 30px;
      font-size: 20px;
      font-weight: 600;
    }
    h3 {
      color: #1f2328;
      font-size: 16px;
      font-weight: 600;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      background: white;
      box-shadow: 0 1px 3px rgba(0,0,0,0.12);
      border-radius: 6px;
      overflow: hidden;
      border: 1px solid #d1d9e0;
    }
    th {
      background-color: #f6f8fa;
      color: #1f2328;
      padding: 10px 12px;
      text-align: left;
      font-weight: 600;
      font-size: 13px;
      border-bottom: 1px solid #d1d9e0;
    }
    td {
      padding: 8px 12px;
      border-bottom: 1px solid #d1d9e0;
      font-size: 13px;
    }
    tr:hover {
      background-color: #f6f8fa;
    }
    .status-pending { color: #9a6700; }
    .status-running { color: #0969da; }
    .status-succeeded { color: #1a7f37; }
    .status-failed { color: #cf222e; }
    .status-killed { color: #57606a; }
    .status-worker_failed { color: #8250df; }
    .status-requesting { color: #bc4c00; }
    .healthy { color: #1a7f37; }
    .unhealthy { color: #cf222e; }
    .worker-link, .job-link { color: #0969da; text-decoration: none; }
    .worker-link:hover, .job-link:hover { text-decoration: underline; }
    .status-building { color: #8250df; }
    .actions-log {
      background: white;
      padding: 15px;
      border-radius: 6px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.12);
      border: 1px solid #d1d9e0;
      max-height: 300px;
      overflow-y: auto;
      font-family: monospace;
      font-size: 13px;
    }
    .action-entry {
      padding: 5px 0;
      border-bottom: 1px solid #d1d9e0;
    }
    .action-time {
      color: #57606a;
      margin-right: 10px;
    }
    .future-feature {
      color: #57606a;
      font-style: italic;
      padding: 20px;
      background: white;
      border-radius: 6px;
      text-align: center;
    }
    .autoscaler-status {
      background: white;
      padding: 15px 20px;
      border-radius: 6px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.12);
      border: 1px solid #d1d9e0;
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
    .autoscaler-status .status-indicator.active { background: #1a7f37; }
    .autoscaler-status .status-indicator.disabled { background: #57606a; }
    .autoscaler-status .status-indicator.backoff { background: #9a6700; }
    .scale-groups-table {
      width: 100%;
      border-collapse: collapse;
      background: white;
      box-shadow: 0 1px 3px rgba(0,0,0,0.12);
      border: 1px solid #d1d9e0;
      border-radius: 6px;
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
    .group-status-dot.available { background: #1a7f37; }
    .group-status-dot.backoff { background: #9a6700; }
    .group-status-dot.disabled { background: #57606a; }
    .action-type {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
    }
    .action-type.scale_up { background: #dafbe1; color: #1a7f37; }
    .action-type.scale_down { background: #ddf4ff; color: #0969da; }
    .action-type.quota_exceeded { background: #ffebe9; color: #cf222e; }
    .action-type.backoff_triggered { background: #fff8c5; color: #9a6700; }
    .action-type.worker_failed { background: #ffebe9; color: #cf222e; }
    .vm-state-indicator {
      display: inline-block;
      width: 8px;
      height: 8px;
      border-radius: 50%;
      margin-right: 8px;
    }
    .vm-state-indicator.ready { background: #1a7f37; }
    .vm-state-indicator.initializing { background: #9a6700; }
    .vm-state-indicator.booting { background: #0969da; }
    .vm-state-indicator.failed { background: #cf222e; }
    .vm-state-indicator.unhealthy { background: #cf222e; }
    .vm-state-indicator.stopping { background: #57606a; }
    .vm-state-indicator.terminated { background: #8c959f; }
    .vm-state-indicator.preempted { background: #8250df; }
    .vm-state-indicator.requesting { background: #bc4c00; }
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
    .task-badge.succeeded { background: #1a7f37; }
    .task-badge.running { background: #0969da; }
    .task-badge.building { background: #8250df; }
    .task-badge.pending { background: #eaeef2; border: 1px solid #8c959f; }
    .task-badge.failed { background: #cf222e; }
    .task-badge.killed { background: #57606a; }
    .task-badge.worker_failed { background: #8250df; }
    .task-icon {
      margin-right: 5px;
    }
    .task-icon.succeeded { color: #1a7f37; }
    .task-icon.running { color: #0969da; }
    .task-icon.building { color: #8250df; }
    .task-icon.pending { color: #57606a; }
    .task-icon.failed { color: #cf222e; }
    .task-icon.killed { color: #57606a; }
    .task-icon.worker_failed { color: #8250df; }
    .no-jobs {
      padding: 40px;
      text-align: center;
      color: #57606a;
    }
    .tab-nav {
      display: flex;
      background: white;
      border-radius: 6px 6px 0 0;
      box-shadow: 0 1px 3px rgba(0,0,0,0.12);
      border: 1px solid #d1d9e0;
      border-bottom: none;
      margin-bottom: 0;
    }
    .tab-btn {
      padding: 12px 24px;
      border: none;
      background: transparent;
      cursor: pointer;
      font-size: 14px;
      font-weight: 500;
      color: #57606a;
      border-bottom: 2px solid transparent;
      transition: all 0.15s;
    }
    .tab-btn:hover {
      color: #1f2328;
      background: #f6f8fa;
    }
    .tab-btn.active {
      color: #0969da;
      border-bottom-color: #0969da;
      font-weight: 600;
    }
    .tab-content {
      display: none;
      background: white;
      padding: 20px;
      border-radius: 0 0 6px 6px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.12);
      border: 1px solid #d1d9e0;
      border-top: none;
    }
    .tab-content.active {
      display: block;
    }
    button:not(.tab-btn) {
      padding: 5px 12px;
      font-size: 12px;
      font-weight: 500;
      color: #1f2328;
      background: #f6f8fa;
      border: 1px solid #d1d9e0;
      border-radius: 6px;
      cursor: pointer;
    }
    button:not(.tab-btn):hover {
      background: #eaeef2;
    }
    button:not(.tab-btn):disabled {
      color: #8c959f;
      cursor: default;
      background: #f6f8fa;
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
    <a href="/logs" class="tab-btn" style="text-decoration:none">Logs</a>
    <button class="tab-btn" onclick="refresh()" style="margin-left:auto;font-size:14px">↻ Refresh</button>
  </div>

  <div id="tab-jobs" class="tab-content active">
    <table id="jobs-table">
      <thead><tr>
        <th>ID</th><th>Name</th><th>State</th><th>Tasks</th>
        <th>Duration</th><th>Failures</th><th>Preemptions</th><th>Diagnostic</th>
      </tr></thead>
      <tbody id="jobs-body"><tr><td colspan="8">Loading...</td></tr></tbody>
    </table>
    <div id="jobs-pagination"
      style="margin-top:10px;display:flex;justify-content:space-between;align-items:center"></div>
  </div>

  <div id="tab-workers" class="tab-content">
    <table id="workers-table">
      <tr><th>ID</th><th>Healthy</th><th>CPU</th><th>Memory</th><th>Running Tasks</th>
        <th>Last Heartbeat</th><th>Attributes</th><th>Status</th></tr>
    </table>
  </div>

  <div id="tab-endpoints" class="tab-content">
    <table id="endpoints-table">
      <tr><th>Name</th><th>Address</th><th>Job</th><th>Metadata</th></tr>
    </table>
    <div id="no-endpoints" class="no-jobs" style="display:none">No endpoints registered</div>
  </div>

  <div id="tab-vms" class="tab-content">
    <table id="vms-table">
      <thead><tr>
        <th>VM ID</th><th>Scale Group</th><th>Slice</th><th>Accelerator</th>
        <th>State</th><th>Address</th><th>Worker</th><th>Error</th>
      </tr></thead>
      <tbody id="vms-body"><tr><td colspan="8">No scale groups configured</td></tr></tbody>
    </table>
    <div id="vms-pagination" style="margin-top:10px;display:flex;justify-content:space-between;align-items:center"></div>
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

    <h3>Autoscaler Logs</h3>
    <pre id="autoscaler-logs" style="background:white;padding:15px;border-radius:6px;
      box-shadow:0 1px 3px rgba(0,0,0,0.12);border:1px solid #d1d9e0;
      max-height:400px;overflow-y:auto;font-size:12px;white-space:pre-wrap">Loading logs...</pre>
    <div style="margin-top:8px"><a href="/logs"
      style="color:#0969da;text-decoration:none;font-size:13px">View full controller logs &rarr;</a></div>
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

    function formatAttributeValue(v) {
      if (!v) return '';
      if (v.stringValue !== undefined) return v.stringValue;
      if (v.intValue !== undefined) return String(v.intValue);
      if (v.floatValue !== undefined) return String(v.floatValue);
      if (typeof v === 'string') return v;
      return JSON.stringify(v);
    }

    function formatAttributes(attrs) {
      if (!attrs) return '-';
      const entries = Object.entries(attrs).map(([k, v]) => k + '=' + formatAttributeValue(v)).join(', ');
      return entries || '-';
    }

    // Convert accelerator type enum to friendly name
    function acceleratorTypeFriendly(accelType) {
      if (typeof accelType === 'string') {
        // Already a string like "ACCELERATOR_TYPE_TPU"
        if (accelType.startsWith('ACCELERATOR_TYPE_')) {
          return accelType.replace('ACCELERATOR_TYPE_', '').toLowerCase();
        }
        return accelType.toLowerCase();
      }
      // Numeric enum value
      const typeMap = {
        0: 'unspecified',
        1: 'cpu',
        2: 'gpu',
        3: 'tpu'
      };
      return typeMap[accelType] || `unknown(${accelType})`;
    }

    // Format accelerator type and variant for display
    function formatAcceleratorDisplay(accelType, variant) {
      const friendly = acceleratorTypeFriendly(accelType);
      if (variant) {
        return `${friendly} (${variant})`;
      }
      return friendly;
    }

    let jobsPage = 0;
    let vmsPage = 0;

    function renderJobsTab(jobs) {
      const tbody = document.getElementById('jobs-body');
      if (!jobs || jobs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8">No jobs</td></tr>';
        document.getElementById('jobs-pagination').innerHTML = '';
        return;
      }

      // Sort: running/building first, then pending, then completed
      const stateOrder = {running: 0, building: 1, pending: 2, succeeded: 3, failed: 4, killed: 5, worker_failed: 6};
      jobs.sort((a, b) => (stateOrder[a.state] || 99) - (stateOrder[b.state] || 99));

      const pageSize = 50;
      const totalPages = Math.ceil(jobs.length / pageSize);
      if (jobsPage >= totalPages) jobsPage = totalPages - 1;
      if (jobsPage < 0) jobsPage = 0;
      const pageJobs = jobs.slice(jobsPage * pageSize, (jobsPage + 1) * pageSize);

      tbody.innerHTML = pageJobs.map(job => {
        const jobId = job.job_id;
        const shortId = jobId.slice(0, 8);
        const badges = buildTaskBadges(job);

        let duration = '-';
        if (job.started_at_ms) {
          const endMs = job.finished_at_ms || Date.now();
          duration = formatDuration(job.started_at_ms, endMs);
        } else if (job.submitted_at_ms) {
          duration = 'queued ' + formatRelativeTime(job.submitted_at_ms);
        }

        // Truncate diagnostic to 100 chars for table display
        const diagnostic = job.pending_reason || '';
        const diagnosticDisplay = diagnostic.length > 100
          ? diagnostic.substring(0, 97) + '...'
          : diagnostic;

        return `<tr>
          <td><a href="/job/${jobId}" class="job-link">${shortId}...</a></td>
          <td>${escapeHtml(job.name || 'unnamed')}</td>
          <td><span class="status-${job.state}">${job.state}</span></td>
          <td>${badges}</td>
          <td>${duration}</td>
          <td>${job.failure_count || 0}</td>
          <td>${job.preemption_count || 0}</td>
          <td style="font-size:12px;color:#57606a;max-width:300px;overflow:hidden;
            text-overflow:ellipsis;white-space:nowrap"
            title="${escapeHtml(diagnostic)}">${escapeHtml(diagnosticDisplay) || '-'}</td>
        </tr>`;
      }).join('');

      const pag = document.getElementById('jobs-pagination');
      if (totalPages <= 1) {
        pag.innerHTML = '';
      } else {
        pag.innerHTML = `<button onclick="jobsPage--;refresh()" ${jobsPage===0?'disabled':''}>Prev</button>` +
          `<span>Page ${jobsPage+1} of ${totalPages}</span>` +
          `<button onclick="jobsPage++;refresh()" ${jobsPage>=totalPages-1?'disabled':''}>Next</button>`;
      }
    }

    function buildTaskBadges(job) {
      const counts = job.task_state_counts || {};
      const total = job.task_count || 0;
      if (total === 0) return '<span style="color:#57606a;font-size:12px">no tasks</span>';

      const succeeded = counts.succeeded || 0;
      const running = counts.running || 0;
      const building = counts.building || 0;
      const failed = counts.failed || 0;
      const killed = counts.killed || 0;
      const workerFailed = counts.worker_failed || 0;
      const pending = total - succeeded - running - building - failed - killed - workerFailed;

      const segments = [
        {count: succeeded, color: '#1a7f37', label: 'succeeded'},
        {count: running, color: '#0969da', label: 'running'},
        {count: building, color: '#8250df', label: 'building'},
        {count: failed, color: '#cf222e', label: 'failed'},
        {count: workerFailed, color: '#8250df', label: 'worker_failed'},
        {count: killed, color: '#57606a', label: 'killed'},
        {count: pending, color: '#eaeef2', label: 'pending'},
      ].filter(s => s.count > 0);

      let bar = '<div style="display:flex;align-items:center;gap:6px">' +
        '<div style="display:flex;height:8px;width:120px;border-radius:4px;overflow:hidden;background:#eaeef2">';
      for (const s of segments) {
        const pct = (s.count / total * 100).toFixed(1);
        bar += `<div style="width:${pct}%;background:${s.color}" title="${s.label}: ${s.count}"></div>`;
      }
      bar += '</div>';
      bar += `<span style="font-size:11px;color:#57606a">${succeeded}/${total}</span></div>`;
      return bar;
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
      const counts = {requesting: 0, booting: 0, initializing: 0, ready: 0, failed: 0};
      for (const s of slices) {
        const vms = s.vms || [];
        if (vms.length === 0) continue;
        if (vms.every(vm => vm.state === "VM_STATE_TERMINATED")) continue;
        const anyFailed = vms.some(vm => vm.state === "VM_STATE_FAILED" || vm.state === "VM_STATE_PREEMPTED");
        const allReady = vms.every(vm => vm.state === "VM_STATE_READY");
        if (anyFailed) {
          counts.failed++;
        } else if (allReady) {
          counts.ready++;
        } else if (vms.some(vm => vm.state === "VM_STATE_REQUESTING")) {
          counts.requesting++;
        } else if (vms.some(vm => vm.state === "VM_STATE_INITIALIZING")) {
          counts.initializing++;
        } else if (vms.some(vm => vm.state === "VM_STATE_BOOTING")) {
          counts.booting++;
        }
      }
      return counts;
    }

    function renderVmsTab(data) {
      const tbody = document.getElementById('vms-body');
      if (!data.groups || data.groups.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8">No scale groups configured</td></tr>';
        document.getElementById('vms-pagination').innerHTML = '';
        return;
      }

      // Flatten all VMs from all groups into rows
      const allVms = [];
      for (const group of data.groups) {
        const config = group.config || {};
        const accel = formatAcceleratorDisplay(config.acceleratorType, config.acceleratorVariant || '');
        for (const slice of (group.slices || [])) {
          for (const vm of (slice.vms || [])) {
            allVms.push({vm, groupName: group.name, sliceId: slice.sliceId, accel});
          }
        }
      }

      if (allVms.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8">No VMs</td></tr>';
        document.getElementById('vms-pagination').innerHTML = '';
        return;
      }

      const pageSize = 50;
      const totalPages = Math.ceil(allVms.length / pageSize);
      if (vmsPage >= totalPages) vmsPage = totalPages - 1;
      if (vmsPage < 0) vmsPage = 0;
      const pageVms = allVms.slice(vmsPage * pageSize, (vmsPage + 1) * pageSize);

      tbody.innerHTML = pageVms.map(({vm, groupName, sliceId, accel}) => {
        const state = formatVmState(vm.state);
        return `<tr>
          <td><a href="/vm/${encodeURIComponent(vm.vmId)}" class="job-link">${escapeHtml(vm.vmId)}</a></td>
          <td>${escapeHtml(groupName)}</td>
          <td>${escapeHtml(sliceId)}</td>
          <td>${escapeHtml(accel)}</td>
          <td><span class="vm-state-indicator ${state}"></span><span class="status-${state}">${state}</span></td>
          <td>${escapeHtml(vm.address || '-')}</td>
          <td>${escapeHtml(vm.workerId || '-')}</td>
          <td>${escapeHtml(vm.initError || '-')}</td>
        </tr>`;
      }).join('');

      const pag = document.getElementById('vms-pagination');
      if (totalPages <= 1) {
        pag.innerHTML = '';
      } else {
        pag.innerHTML = `<button onclick="vmsPage--;refresh()" ${vmsPage===0?'disabled':''}>Prev</button>` +
          `<span>Page ${vmsPage+1} of ${totalPages}</span>` +
          `<button onclick="vmsPage++;refresh()" ${vmsPage>=totalPages-1?'disabled':''}>Next</button>`;
      }
    }

    function groupStatusText(group, counts) {
      if (group.backoffUntilMs && parseInt(group.backoffUntilMs) > Date.now()) return ['backoff', 'Backoff'];
      if (counts.requesting > 0 || counts.booting > 0 || counts.initializing > 0) return ['available', 'Scaling Up'];
      if ((group.currentDemand || 0) > (counts.ready || 0)) return ['backoff', 'Pending'];
      if (counts.ready > 0) return ['available', 'Available'];
      return ['disabled', 'Idle'];
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
        const [statusClass, statusText] = groupStatusText(g, counts);

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

    async function fetchAutoscalerLogs() {
      try {
        const logsResp = await fetch('/api/logs?prefix=iris.cluster.vm&limit=200');
        const logsData = await logsResp.json();
        const logsEl = document.getElementById('autoscaler-logs');
        if (logsData && logsData.length > 0) {
          logsEl.textContent = logsData.map(l => l.message || l.formatted).join('\\n');
          logsEl.scrollTop = logsEl.scrollHeight;
        } else {
          logsEl.textContent = 'No recent autoscaler logs';
        }
      } catch (e) {
        document.getElementById('autoscaler-logs').textContent = 'Failed to load logs: ' + e.message;
      }
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
          },
          attributes: w.metadata && w.metadata.attributes ? w.metadata.attributes : {}
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
          const statusMsg = w.status_message || '-';
          return `<tr>
            <td>${workerLink}</td>
            <td class="${healthClass}">${healthIndicator} ${healthText}</td>
            <td>${cpu}</td>
            <td>${memory}</td>
            <td>${w.running_tasks}</td>
            <td>${lastHb}</td>
            <td>${escapeHtml(formatAttributes(w.attributes))}</td>
            <td style="font-size:12px;color:${w.healthy ? '#57606a' : '#cf222e'}">${escapeHtml(statusMsg)}</td>
          </tr>`;
        }).join('');
        const workersHeader = '<tr><th>ID</th><th>Healthy</th><th>CPU</th><th>Memory</th>' +
          '<th>Running Tasks</th><th>Last Heartbeat</th><th>Attributes</th><th>Status</th></tr>';
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
        fetchAutoscalerLogs();
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

    document.querySelectorAll('.tab-btn[data-tab]').forEach(btn => {
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
      color: #1f2328;
      background: #f6f8fa;
    }
    h1 {
      color: #1f2328;
      border-bottom: 2px solid #d1d9e0;
      padding-bottom: 10px;
    }
    h2 {
      color: #1f2328;
      margin-top: 30px;
    }
    .back-link {
      color: #0969da;
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
      border-radius: 6px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    .info-card h3 {
      margin-top: 0;
      color: #1f2328;
      border-bottom: 1px solid #d1d9e0;
      padding-bottom: 10px;
    }
    .info-row {
      display: flex;
      justify-content: space-between;
      padding: 8px 0;
      border-bottom: 1px solid #d1d9e0;
    }
    .info-row:last-child { border-bottom: none; }
    .info-label { color: #57606a; }
    .info-value { font-weight: 500; }
    .status-pending { color: #9a6700; }
    .status-building { color: #8250df; }
    .status-running { color: #0969da; }
    .status-succeeded { color: #1a7f37; }
    .status-failed { color: #cf222e; }
    .status-killed { color: #57606a; }
    .status-worker_failed { color: #8250df; }
    .status-unschedulable { color: #cf222e; }
    .error-message {
      background: #ffebe9;
      border: 1px solid #cf222e;
      color: #cf222e;
      padding: 15px;
      border-radius: 6px;
      margin-bottom: 20px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      background: white;
      box-shadow: 0 1px 3px rgba(0,0,0,0.12);
      border-radius: 6px;
      overflow: hidden;
      margin-top: 20px;
    }
    th {
      background-color: #f6f8fa;
      color: #1f2328;
      padding: 12px;
      text-align: left;
      font-weight: 600;
    }
    td {
      padding: 10px 12px;
      border-bottom: 1px solid #d1d9e0;
    }
    tr:hover {
      background-color: #f6f8fa;
    }
    .pending-reason {
      font-size: 12px;
      color: #57606a;
      font-style: italic;
    }
  </style>
</head>
<body>
  <a href="/" class="back-link">&larr; Back to Dashboard</a>
  <h1 id="job-title">Job: {{job_id}}</h1>
  <div id="job-subtitle" style="color:#57606a;font-size:14px;margin-bottom:20px"></div>

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

  <h2>Task Logs</h2>
  <div style="margin-bottom:10px;display:flex;align-items:center;gap:12px">
    <select id="task-log-selector" onchange="fetchTaskLogs()">
      <option value="">Select a task...</option>
    </select>
    <span id="task-log-status" style="color:#57606a;font-size:13px"></span>
  </div>
  <pre id="task-logs" style="background:white;padding:15px;border-radius:6px;
    box-shadow:0 1px 3px rgba(0,0,0,0.12);border:1px solid #d1d9e0;
    max-height:600px;overflow-y:auto;font-size:12px;white-space:pre-wrap">Loading logs...</pre>

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
    let cachedTasks = [];

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
          name: j.name || '',
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
        cachedTasks = tasksResponse;

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

        // Fix 5: Show job name in title, ID as subtitle
        if (job.name && job.name !== jobId) {
          document.getElementById('job-title').textContent = job.name;
          document.getElementById('job-subtitle').textContent = 'ID: ' + jobId;
        }

        document.getElementById('job-exit-code').textContent =
          job.failure_count > 0 ? 'Failed' : (job.state === 'succeeded' ? '0' : '-');
        document.getElementById('job-started').textContent =
          formatTimestamp(job.started_at_ms);

        // Fix 2: Show "-" for finished/duration on non-terminal jobs
        const isTerminal = ['succeeded','failed','killed','worker_failed','unschedulable'].includes(job.state);
        document.getElementById('job-finished').textContent =
          isTerminal ? formatTimestamp(job.finished_at_ms) : '-';
        document.getElementById('job-duration').textContent =
          isTerminal ? formatDuration(job.started_at_ms, job.finished_at_ms)
                     : (job.started_at_ms ? formatDuration(job.started_at_ms, Date.now()) : '-');

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
            <td>${['succeeded','failed','killed','worker_failed'].includes(t.state)
              && t.exit_code !== null && t.exit_code !== undefined
              ? t.exit_code : '-'}</td>
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

        // Populate task log selector
        const selector = document.getElementById('task-log-selector');
        const previousValue = selector.value;
        selector.innerHTML = '<option value="">Select a task...</option>' +
          tasksResponse.map(t => `<option value="${t.task_index}">Task ${t.task_index} (${t.state})</option>`).join('');

        // Auto-select: restore previous selection, or pick the most interesting task
        if (previousValue) {
          selector.value = previousValue;
        } else {
          const failedTask = tasksResponse.find(t => t.state === 'failed' || t.state === 'worker_failed');
          const runningTask = tasksResponse.find(t => t.state === 'running' || t.state === 'building');
          const autoTask = failedTask || runningTask || tasksResponse[0];
          if (autoTask !== undefined) {
            selector.value = String(autoTask.task_index);
            fetchTaskLogs();
          }
        }

      } catch (e) {
        console.error('Failed to refresh:', e);
        document.getElementById('error-container').innerHTML =
          `<div class="error-message">Failed to load job details: ${escapeHtml(e.message)}</div>`;
      }
    }

    async function fetchTaskLogs() {
      const selector = document.getElementById('task-log-selector');
      const taskIndex = parseInt(selector.value);
      const logsEl = document.getElementById('task-logs');
      const statusEl = document.getElementById('task-log-status');
      if (isNaN(taskIndex)) {
        logsEl.textContent = 'Select a task to view logs';
        statusEl.textContent = '';
        return;
      }

      // For pending/building tasks, show status message instead of fetching logs
      const selectedTask = cachedTasks.find(t => t.task_index === taskIndex);
      if (selectedTask && (selectedTask.state === 'pending' || selectedTask.state === 'building')) {
        const reason = selectedTask.pending_reason || '';
        const label = selectedTask.state === 'pending' ? 'Waiting to be scheduled' : 'Building container image';
        statusEl.innerHTML = `<span style="color:#9a6700;font-weight:500">${escapeHtml(label)}</span>`;
        logsEl.textContent = reason || 'No logs yet \\u2014 task has not started running.';
        return;
      }

      logsEl.textContent = 'Loading logs...';
      statusEl.textContent = '';
      try {
        const resp = await rpc('GetTaskLogs', {jobId: jobId, taskIndex: taskIndex, limit: 1000});
        const logs = resp.logs || [];
        if (selectedTask && selectedTask.error) {
          statusEl.innerHTML = `<span style="color:#cf222e;font-weight:600">`
            + `Error: ${escapeHtml(selectedTask.error)}</span>`;
        } else {
          statusEl.textContent = resp.workerAddress ? 'from ' + resp.workerAddress : '';
        }
        if (logs.length === 0) {
          logsEl.textContent = 'No logs available';
        } else {
          logsEl.textContent = logs.map(l => l.data || '').join('\\n');
          logsEl.scrollTop = logsEl.scrollHeight;
        }
      } catch (e) {
        logsEl.textContent = 'Failed to load logs: ' + e.message;
        statusEl.textContent = '';
      }
    }

    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>
"""


VM_DETAIL_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>VM Detail - {{vm_id}}</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      max-width: 1400px;
      margin: 40px auto;
      padding: 0 20px;
      color: #1f2328;
      background: #f6f8fa;
    }
    h1 {
      color: #1f2328;
      border-bottom: 2px solid #d1d9e0;
      padding-bottom: 10px;
    }
    h2 {
      color: #1f2328;
      margin-top: 30px;
    }
    .back-link {
      color: #0969da;
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
      border-radius: 6px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    .info-card h3 {
      margin-top: 0;
      color: #1f2328;
      border-bottom: 1px solid #d1d9e0;
      padding-bottom: 10px;
    }
    .info-row {
      display: flex;
      justify-content: space-between;
      padding: 8px 0;
      border-bottom: 1px solid #d1d9e0;
    }
    .info-row:last-child { border-bottom: none; }
    .info-label { color: #57606a; }
    .info-value { font-weight: 500; }
    .error-message {
      background: #ffebe9;
      border: 1px solid #cf222e;
      color: #cf222e;
      padding: 15px;
      border-radius: 6px;
      margin-bottom: 20px;
    }
    .status-ready { color: #1a7f37; }
    .status-booting { color: #0969da; }
    .status-initializing { color: #9a6700; }
    .status-requesting { color: #bc4c00; }
    .status-failed { color: #cf222e; }
    .status-preempted { color: #8250df; }
    .status-terminated { color: #57606a; }
    .status-stopping { color: #57606a; }
    .status-unhealthy { color: #cf222e; }
  </style>
</head>
<body>
  <a href="/#vms" class="back-link">&larr; Back to Dashboard</a>
  <h1>VM: {{vm_id}}</h1>

  <div id="error-container"></div>

  <div class="info-grid">
    <div class="info-card">
      <h3>VM Info</h3>
      <div class="info-row">
        <span class="info-label">ID</span>
        <span class="info-value" id="vm-id">{{vm_id}}</span>
      </div>
      <div class="info-row">
        <span class="info-label">State</span>
        <span class="info-value" id="vm-state">-</span>
      </div>
      <div class="info-row">
        <span class="info-label">Address</span>
        <span class="info-value" id="vm-address">-</span>
      </div>
      <div class="info-row">
        <span class="info-label">Worker</span>
        <span class="info-value" id="vm-worker">-</span>
      </div>
      <div class="info-row">
        <span class="info-label">Init Phase</span>
        <span class="info-value" id="vm-init-phase">-</span>
      </div>
    </div>

    <div class="info-card">
      <h3>Scale Group</h3>
      <div class="info-row">
        <span class="info-label">Group</span>
        <span class="info-value" id="vm-group">-</span>
      </div>
      <div class="info-row">
        <span class="info-label">Slice</span>
        <span class="info-value" id="vm-slice">-</span>
      </div>
    </div>
  </div>

  <div id="error-detail-container"></div>

  <h2>Bootstrap Logs</h2>
  <pre id="vm-logs" style="background:white;padding:15px;border-radius:6px;
    box-shadow:0 1px 3px rgba(0,0,0,0.12);max-height:600px;overflow-y:auto;
    font-size:12px;white-space:pre-wrap">Loading logs...</pre>

  <script>
    const vmId = '{{vm_id}}';

    function escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text || '';
      return div.innerHTML;
    }

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

    function formatVmState(state) {
      if (!state) return 'unknown';
      return state.replace('VM_STATE_', '').toLowerCase();
    }

    async function refresh() {
      try {
        const [autoscalerResp, logsResp] = await Promise.all([
          rpc('GetAutoscalerStatus'),
          rpc('GetVmLogs', {vmId: vmId, tail: 500})
        ]);

        const status = autoscalerResp.status || {};
        let vmInfo = null;
        let groupName = '-';
        let sliceId = '-';

        for (const group of (status.groups || [])) {
          for (const slice of (group.slices || [])) {
            for (const vm of (slice.vms || [])) {
              if (vm.vmId === vmId) {
                vmInfo = vm;
                groupName = group.name;
                sliceId = slice.sliceId;
              }
            }
          }
        }

        if (vmInfo) {
          const state = formatVmState(vmInfo.state);
          const stateEl = document.getElementById('vm-state');
          stateEl.textContent = state;
          stateEl.className = 'info-value status-' + state;
          document.getElementById('vm-address').textContent = vmInfo.address || '-';
          document.getElementById('vm-worker').textContent = vmInfo.workerId || '-';
          document.getElementById('vm-init-phase').textContent = vmInfo.initPhase || '-';
          document.getElementById('vm-group').textContent = groupName;
          document.getElementById('vm-slice').textContent = sliceId;

          if (vmInfo.initError) {
            document.getElementById('error-detail-container').innerHTML =
              '<div class="error-message"><strong>Init Error:</strong> ' + escapeHtml(vmInfo.initError) + '</div>';
          } else {
            document.getElementById('error-detail-container').innerHTML = '';
          }
        } else {
          // VM not found in autoscaler, but logs RPC may still have state
          const logState = formatVmState(logsResp.state);
          if (logState && logState !== 'unknown') {
            const stateEl = document.getElementById('vm-state');
            stateEl.textContent = logState;
            stateEl.className = 'info-value status-' + logState;
          }
        }

        const logsEl = document.getElementById('vm-logs');
        if (logsResp.logs) {
          logsEl.textContent = logsResp.logs;
          logsEl.scrollTop = logsEl.scrollHeight;
        } else {
          logsEl.textContent = 'No bootstrap logs available';
        }

        document.getElementById('error-container').innerHTML = '';
      } catch (e) {
        console.error('Failed to refresh:', e);
        document.getElementById('error-container').innerHTML =
          '<div class="error-message">Failed to load VM details: ' + escapeHtml(e.message) + '</div>';
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
        log_buffer: LogBuffer | None = None,
    ):
        self._service = service
        self._host = host
        self._port = port
        self._log_buffer = log_buffer
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
            Route("/vm/{vm_id}", self._vm_detail_page),
            Route("/logs", self._logs_page),
            Route("/api/logs", self._api_logs),
            Route("/health", self._health),
            Mount(rpc_wsgi_app.path, app=rpc_app),
        ]
        return Starlette(routes=routes)

    def _dashboard(self, _request: Request) -> HTMLResponse:
        return HTMLResponse(DASHBOARD_HTML)

    def _job_detail_page(self, request: Request) -> HTMLResponse:
        job_id = request.path_params["job_id"]
        return HTMLResponse(JOB_DETAIL_HTML.replace("{{job_id}}", html.escape(job_id)))

    def _vm_detail_page(self, request: Request) -> HTMLResponse:
        vm_id = request.path_params["vm_id"]
        return HTMLResponse(VM_DETAIL_HTML.replace("{{vm_id}}", html.escape(vm_id)))

    def _logs_page(self, request: Request) -> HTMLResponse:
        return logs_page_response(request)

    def _api_logs(self, request: Request):
        return logs_api_response(request, self._log_buffer)

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
