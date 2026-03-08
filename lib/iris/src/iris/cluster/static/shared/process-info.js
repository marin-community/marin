/**
 * Shared process info display and profiling controls.
 *
 * Used by both controller and worker status pages.
 */

import { h } from 'preact';
import { useState } from 'preact/hooks';
import htm from 'htm';
import { profileAndDownload } from '/static/shared/profiling.js';

const html = htm.bind(h);

function formatBytes(n) {
  if (!n) return '0 B';
  if (n < 1024) return n + ' B';
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
  if (n < 1024 * 1024 * 1024) return (n / (1024 * 1024)).toFixed(1) + ' MB';
  return (n / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
}

function formatUptime(ms) {
  if (!ms) return '-';
  const s = Math.floor(ms / 1000);
  if (s < 60) return s + 's';
  const m = Math.floor(s / 60);
  if (m < 60) return m + 'm ' + (s % 60) + 's';
  const hrs = Math.floor(m / 60);
  if (hrs < 24) return hrs + 'h ' + (m % 60) + 'm';
  const d = Math.floor(hrs / 24);
  return d + 'd ' + (hrs % 24) + 'h ' + (m % 60) + 'm';
}

/**
 * Process info panel with system metrics and profiling controls.
 *
 * @param {Object} info - ProcessInfo proto (from GetProcessStatus)
 * @param {Function} rpc - RPC function (controllerRpc or workerRpc)
 * @param {string} title - Display title (e.g. "Controller" or "Worker")
 */
export function ProcessInfoPanel({ info, rpc, title = 'Process' }) {
  const [profiling, setProfiling] = useState(null);
  const [profileError, setProfileError] = useState(null);

  const runProfile = async (profilerType) => {
    setProfiling(profilerType);
    setProfileError(null);
    try {
      await profileAndDownload(rpc, '/system/process', {
        profilerType,
        durationSeconds: profilerType === 'threads' ? 1 : 10,
        format: profilerType === 'cpu' ? 'SPEEDSCOPE' : profilerType === 'memory' ? 'FLAMEGRAPH' : undefined,
      });
    } catch (e) {
      setProfileError(e.message);
    } finally {
      setProfiling(null);
    }
  };

  if (!info) {
    return html`<div class="process-info-panel">
      <h2 style="margin:0 0 8px 0;font-size:16px">${title} Status</h2>
      <div style="color:#666">Loading process info...</div>
    </div>`;
  }

  const memPercent = info.memoryTotalBytes
    ? ((parseInt(info.memoryRssBytes || 0) / parseInt(info.memoryTotalBytes)) * 100).toFixed(1)
    : '?';

  return html`
    <div class="process-info-panel">
      <h2 style="margin:0 0 12px 0;font-size:16px">${title} Status</h2>
      <div class="process-info-grid">
        <div class="process-info-card">
          <span class="process-info-label">Host</span>
          <span class="process-info-value">${info.hostname || '-'}</span>
        </div>
        <div class="process-info-card">
          <span class="process-info-label">PID</span>
          <span class="process-info-value">${info.pid || '-'}</span>
        </div>
        <div class="process-info-card">
          <span class="process-info-label">Python</span>
          <span class="process-info-value">${info.pythonVersion || '-'}</span>
        </div>
        <div class="process-info-card">
          <span class="process-info-label">Uptime</span>
          <span class="process-info-value">${formatUptime(parseInt(info.uptimeMs || 0))}</span>
        </div>
        <div class="process-info-card">
          <span class="process-info-label">CPU</span>
          <span class="process-info-value">${(info.cpuPercent || 0).toFixed(1)}% (${info.cpuCount || '?'} cores)</span>
        </div>
        <div class="process-info-card">
          <span class="process-info-label">Memory RSS</span>
          <span class="process-info-value">${formatBytes(parseInt(info.memoryRssBytes || 0))} (${memPercent}%)</span>
        </div>
        <div class="process-info-card">
          <span class="process-info-label">Memory VMS</span>
          <span class="process-info-value">${formatBytes(parseInt(info.memoryVmsBytes || 0))}</span>
        </div>
        <div class="process-info-card">
          <span class="process-info-label">Threads</span>
          <span class="process-info-value">${info.threadCount || '-'}</span>
        </div>
        <div class="process-info-card">
          <span class="process-info-label">Open FDs</span>
          <span class="process-info-value">${info.openFdCount || '-'}</span>
        </div>
      </div>

      <div class="process-profile-controls">
        <span style="font-weight:600;margin-right:8px">Profile:</span>
        <button class="profile-btn" onClick=${() => runProfile('threads')}
                disabled=${!!profiling}>
          ${profiling === 'threads' ? 'Dumping...' : 'Thread Dump'}
        </button>
        <button class="profile-btn" onClick=${() => runProfile('cpu')}
                disabled=${!!profiling}>
          ${profiling === 'cpu' ? 'Profiling...' : 'CPU Profile'}
        </button>
        <button class="profile-btn" onClick=${() => runProfile('memory')}
                disabled=${!!profiling}>
          ${profiling === 'memory' ? 'Profiling...' : 'Memory Profile'}
        </button>
        ${profileError && html`<span class="profile-error">${profileError}</span>`}
      </div>
    </div>
  `;
}
