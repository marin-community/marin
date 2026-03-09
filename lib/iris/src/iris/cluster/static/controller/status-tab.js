/**
 * Process Status tab for the controller dashboard.
 *
 * Shows process info (host, PID, memory, CPU, etc.), profiling buttons,
 * and the process log viewer.
 */

import { h } from 'preact';
import { useState, useEffect, useCallback } from 'preact/hooks';
import htm from 'htm';
import { controllerRpc } from '/static/shared/rpc.js';
import { LogViewer } from '/static/shared/log-viewer.js';
import { ProcessInfoPanel } from '/static/shared/process-info.js';

const html = htm.bind(h);

export function StatusTab() {
  const [processInfo, setProcessInfo] = useState(null);
  const [error, setError] = useState(null);

  const fetchStatus = useCallback(async () => {
    try {
      const resp = await controllerRpc('GetProcessStatus', { maxLogLines: 0 });
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
    <div class="status-tab">
      ${error && html`<div class="error-message">${error}</div>`}
      <${ProcessInfoPanel} info=${processInfo} rpc=${controllerRpc} title="Controller" />
      <${LogViewer}
        rpc=${controllerRpc}
        source="/system/process"
        title="Controller Logs"
      />
    </div>
  `;
}
