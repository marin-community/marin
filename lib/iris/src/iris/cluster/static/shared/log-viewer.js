/**
 * Logs viewer Preact component for Iris dashboards.
 * Accepts a fetchLogs(prefix, limit) prop to decouple from specific RPC service.
 */

import { h } from 'preact';
import { useState, useEffect } from 'preact/hooks';
import htm from 'htm';
const html = htm.bind(h);

export function LogsApp({ fetchLogs }) {
  const [prefix, setPrefix] = useState('');
  const [limit, setLimit] = useState('200');
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);

  const doFetch = async () => {
    try {
      const data = await fetchLogs(prefix || '', parseInt(limit) || 200);
      setLogs(data.records || []);
      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch logs:', error);
      setLoading(false);
    }
  };

  useEffect(() => {
    doFetch();
    const interval = setInterval(doFetch, 3000);
    return () => clearInterval(interval);
  }, [prefix, limit]);

  const renderLogContent = () => {
    if (loading) {
      return html`<div class="empty-state">Loading logs...</div>`;
    }
    if (logs.length === 0) {
      return html`<div class="empty-state">No logs found</div>`;
    }
    return logs.map(log => html`
      <div class="log-line ${log.level}" key="${log.timestamp}-${log.message}">
        ${log.message}
      </div>
    `);
  };

  return html`
    <div>
      <a href="/" class="back-link">&larr; Dashboard</a>
      <h1>Process Logs</h1>
      <div class="controls">
        <label>
          Prefix:
          <input
            id="prefix"
            type="text"
            placeholder="e.g. iris.cluster.controller"
            value="${prefix}"
            onInput="${(e) => setPrefix(e.target.value)}"
          />
        </label>
        <label>
          Limit:
          <select
            id="limit"
            value="${limit}"
            onChange="${(e) => setLimit(e.target.value)}"
          >
            <option value="100">100</option>
            <option value="200">200</option>
            <option value="500">500</option>
            <option value="1000">1000</option>
          </select>
        </label>
      </div>
      <div id="log-container">
        ${renderLogContent()}
      </div>
    </div>
  `;
}
