/**
 * Process Status page for the worker dashboard.
 *
 * Shows process info, profiling controls, and the process log viewer.
 */

import { h, render } from 'preact';
import { useState, useEffect, useCallback } from 'preact/hooks';
import htm from 'htm';
import { workerRpc } from '/static/shared/rpc.js';
import { LogViewer } from '/static/shared/log-viewer.js';
import { ProcessInfoPanel } from '/static/shared/process-info.js';

const html = htm.bind(h);

function StatusPage() {
  const [processInfo, setProcessInfo] = useState(null);
  const [error, setError] = useState(null);

  const fetchStatus = useCallback(async () => {
    try {
      const resp = await workerRpc('GetProcessStatus', { maxLogLines: 0 });
      setProcessInfo(resp.processInfo || null);
      setError(null);
    } catch (e) {
      setError('Failed to load process status: ' + e.message);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const id = setInterval(fetchStatus, 10000);
    return () => clearInterval(id);
  }, [fetchStatus]);

  return html`
    <div>
      <div style="display:flex;justify-content:space-between;align-items:center">
        <h1 style="flex:1">Worker Process Status</h1>
        <a href="/" class="back-link" style="margin-right:12px">Dashboard</a>
      </div>
      ${error && html`<div class="error-message">${error}</div>`}
      <${ProcessInfoPanel} info=${processInfo} rpc=${workerRpc} title="Worker" />
      <${LogViewer}
        rpc=${workerRpc}
        source="/system/process"
        title="Worker Logs"
      />
    </div>
  `;
}

render(html`<${StatusPage} />`, document.getElementById('root'));
